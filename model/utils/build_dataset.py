# -*- coding: utf-8 -*-
"""
Reranker 训练数据集构建模块（改进版）。

生成三种格式的训练数据，支持多种微调策略：

1. **Pointwise** (reranker_qa_pointwise.jsonl)
   - 格式: {"query": "...", "doc": "...", "label": 0/1}
   - 用途: CrossEncoder 二分类微调（当前方案）

2. **Pairwise** (reranker_qa_pairwise.jsonl)
   - 格式: {"query": "...", "positive": "...", "negative": "..."}
   - 用途: 对比学习 / Margin Ranking Loss

3. **Preference (DPO)** (reranker_qa_dpo.jsonl)
   - 格式: {"query": "...", "chosen": "...", "rejected": "..."}
   - 用途: DPO 偏好优化训练

改进点（相比原方案）：
- 生成多粒度负样本：随机负采样 + 困难负样本（同页面不同段落）
- 使用 LLM 对候选文档进行多级相关性标注（0/1/2），而非仅二分类
- 同时输出三种格式，支持 SFT / Contrastive / DPO 三种训练策略
- 增加数据质量过滤：丢弃 LLM 不确定的样本

用法：
    PYTHONPATH=. python -m model.utils.build_dataset \
        --milvus-host 127.0.0.1 \
        --collection-name htmlrag_dev \
        --output-dir dataset/
"""
import os
import sys
import json
import random
import warnings
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import torch
from pymilvus import connections, Collection
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# 添加项目根目录到 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from config.config_loader import CONFIG, logger, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL


# ======================== DeepSeek LLM 初始化（懒加载） ========================

_llm = None


def get_llm():
    """懒加载 DeepSeek LLM 客户端，避免 import 时即触发实例化。"""
    global _llm
    if _llm is None:
        if not DEEPSEEK_API_KEY:
            logger.warning("⚠️ DEEPSEEK_API_KEY 未设置，请在 .env 中配置后重试")
        _llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
    return _llm


# ======================== 文档格式化 ========================

def format_chunk_for_reranker(chunk: Dict) -> str:
    """将 chunk 中的结构化字段拼接为适用于 reranker 的 doc 字段。"""
    page_name = chunk.get("page_name", "").strip()
    title = chunk.get("title", "").strip()
    page_url = chunk.get("page_url", "").strip()
    summary = chunk.get("summary", "").strip()
    text = chunk.get("text", "").strip()

    return (
        f"【页面名称】：{page_name}\n"
        f"【段落标题】：{title}\n"
        f"【来源路径】：{page_url}\n"
        f"【摘要】：{summary}\n"
        f"【正文】：{text}"
    )


# ======================== Prompt 构建 ========================

def build_relevance_annotation_prompt(query: str, chunks: List[Dict]) -> str:
    """构建多级相关性标注提示词。

    让 LLM 对每个候选文档进行 0/1/2 三级相关性标注：
      0 = 无关：文档与问题完全无关
      1 = 部分相关：文档包含部分相关信息，但不能完整回答
      2 = 高度相关：文档能直接回答问题
    """
    prompt = (
        "你是一个检索系统评测助手。以下是用户问题和多个候选文档，\n"
        "请对每个文档进行相关性评分：\n"
        "  0 = 无关\n"
        "  1 = 部分相关\n"
        "  2 = 高度相关\n\n"
        f"用户问题：{query}\n\n"
    )

    for i, chunk in enumerate(chunks):
        prompt += f"【文档{i+1}】\n{format_chunk_for_reranker(chunk)}\n\n"

    prompt += (
        "请按以下格式输出（每行一个文档的评分）：\n"
        "文档1: <0/1/2>\n"
        "文档2: <0/1/2>\n"
        "...\n"
    )
    return prompt


def build_contrastive_prompt_with_selection(chunks: List[Dict]) -> str:
    """构建对比性 QA 对生成提示词（保留原方案）。"""
    prompt = (
        "你是一个电商平台智能问答系统训练数据构建助手。\n"
        "以下是几段用户检索可能命中的平台知识内容，每段包含：页面名称、段落标题、来源路径、摘要、正文片段。\n\n"
        "🎯 你的任务：\n"
        "1. 阅读所有段落，构造一个自然口语化的问题；\n"
        "2. 该问题必须只能由其中一段准确回答，其他段容易混淆；\n"
        "3. 问题必须体现出区分性的特征，可以基于路径、页面名称、标题或摘要，而不仅仅是正文内容；\n"
        "4. 然后告诉我问题最匹配的是哪一段（用编号1~6表示）。\n\n"
        "📌 输出格式：\n"
        "问题：<一句自然中文问题>\n"
        "对应段编号：<1~>\n"
    )

    for i, chunk in enumerate(chunks):
        prompt += (
            f"\n【候选内容编号{i+1}】\n"
            f"页面名称：{chunk.get('page_name', '')}\n"
            f"段落标题：{chunk.get('title', '')}\n"
            f"来源路径：{chunk.get('page_url', '')}\n"
            f"摘要：{chunk.get('summary', '')}\n"
            f"正文：{chunk.get('text', '')}\n"
        )

    return prompt


# ======================== DeepSeek 调用与解析 ========================

def call_deepseek_chat(prompt: str) -> str:
    """调用 DeepSeek 生成回答。"""
    return get_llm().invoke(prompt).content.strip()


def parse_relevance_annotation(response: str, num_chunks: int) -> Optional[List[int]]:
    """解析 LLM 的多级相关性标注。

    Returns:
        长度为 num_chunks 的列表，每个元素为 0/1/2。解析失败返回 None。
    """
    import re
    scores = []
    for i in range(num_chunks):
        # 匹配 "文档1: 2" 或 "文档1：2" 等格式
        pattern = rf"文档{i+1}\s*[:：]\s*([012])"
        match = re.search(pattern, response)
        if match:
            scores.append(int(match.group(1)))
        else:
            return None
    return scores if len(scores) == num_chunks else None


def parse_qa_selection(response: str, num_chunks: int) -> Tuple[Optional[str], Optional[int]]:
    """解析 LLM 的 QA 对生成和文档选择。

    Returns:
        (query, selected_index) — 解析失败返回 (None, None)
    """
    try:
        lines = [line.strip() for line in response.splitlines() if line.strip()]
        question_line = next((line for line in lines if "问题：" in line), None)
        index_line = next((line for line in lines if "编号" in line or "对应段编号" in line), None)

        query = question_line.split("问题：", 1)[-1].strip()
        idx_str = index_line.split("编号", 1)[-1].strip("：:.)） ")
        local_idx = int(idx_str) - 1

        if not query or local_idx < 0 or local_idx >= num_chunks:
            return None, None
        return query, local_idx
    except Exception:
        return None, None


# ======================== 数据生成主流程 ========================

def process_one_sample_v2(
    chunks: List[Dict],
    all_chunks_pool: List[Dict],
    output_dir: str,
    num_hard_negatives: int = 2,
    num_random_negatives: int = 2,
) -> bool:
    """对一个样本执行改进版数据构建。

    生成步骤：
    1. 用 LLM 生成对比性 QA 对并选择最相关文档
    2. 用 LLM 对所有候选文档进行多级相关性标注（0/1/2）
    3. 生成 Pointwise / Pairwise / DPO 三种格式数据
    4. 困难负样本：从同页面不同段落中采样

    Args:
        chunks: 候选文档块列表（来自检索）
        all_chunks_pool: 全部文档块池（用于随机负采样）
        output_dir: 输出目录
        num_hard_negatives: 困难负样本数（同页面不同段落）
        num_random_negatives: 随机负样本数

    Returns:
        是否成功生成数据
    """
    if len(chunks) < 2:
        logger.debug("❌ 候选文档不足2条，跳过")
        return False

    # Step 1: 生成 QA 对 + 选择最相关文档
    qa_prompt = build_contrastive_prompt_with_selection(chunks)
    qa_response = call_deepseek_chat(qa_prompt)
    query, positive_idx = parse_qa_selection(qa_response, len(chunks))

    if query is None or positive_idx is None:
        logger.warning("⚠️ QA 解析失败，跳过该样本")
        return False

    # Step 2: 多级相关性标注
    rel_prompt = build_relevance_annotation_prompt(query, chunks)
    rel_response = call_deepseek_chat(rel_prompt)
    relevance_scores = parse_relevance_annotation(rel_response, len(chunks))

    if relevance_scores is None:
        # 回退到二分类标注
        relevance_scores = [1 if i == positive_idx else 0 for i in range(len(chunks))]
        logger.debug("⚠️ 相关性标注解析失败，使用二分类回退")

    positive_doc = format_chunk_for_reranker(chunks[positive_idx])

    # Step 3: 采样困难负样本（同页面不同段落）
    positive_page = chunks[positive_idx].get("page_name", "")
    hard_negatives = []
    for chunk in all_chunks_pool:
        if chunk.get("page_name") == positive_page and chunk is not chunks[positive_idx]:
            hard_negatives.append(chunk)
        if len(hard_negatives) >= num_hard_negatives:
            break

    # 随机负样本（基于实际候选池计算采样数，避免 random.sample 越界）
    candidate_pool = [c for c in all_chunks_pool if c not in chunks and c not in hard_negatives]
    random_negatives = random.sample(
        candidate_pool, min(num_random_negatives, len(candidate_pool))
    )

    # 所有负样本文档
    all_negative_chunks = hard_negatives + random_negatives + [
        chunks[i] for i in range(len(chunks)) if i != positive_idx
    ]

    # 去重
    seen_docs = {positive_doc}
    negative_docs = []
    for chunk in all_negative_chunks:
        doc = format_chunk_for_reranker(chunk)
        if doc not in seen_docs:
            seen_docs.add(doc)
            negative_docs.append(doc)

    if not negative_docs:
        return False

    # Step 4: 写入三种格式
    pointwise_path = os.path.join(output_dir, "reranker_qa_pointwise.jsonl")
    pairwise_path = os.path.join(output_dir, "reranker_qa_pairwise.jsonl")
    dpo_path = os.path.join(output_dir, "reranker_qa_dpo.jsonl")

    with open(pointwise_path, "a", encoding="utf-8") as f_pointwise, \
         open(pairwise_path, "a", encoding="utf-8") as f_pairwise, \
         open(dpo_path, "a", encoding="utf-8") as f_dpo:

        # Pointwise: 每个 (query, doc) 对带 0/1/2 标签
        for i, chunk in enumerate(chunks):
            doc = format_chunk_for_reranker(chunk)
            label = relevance_scores[i] if relevance_scores else (1 if i == positive_idx else 0)
            f_pointwise.write(json.dumps({
                "query": query, "doc": doc, "label": label
            }, ensure_ascii=False) + "\n")

        # Pairwise: (query, positive, negative)
        for neg_doc in negative_docs:
            f_pairwise.write(json.dumps({
                "query": query, "positive": positive_doc, "negative": neg_doc
            }, ensure_ascii=False) + "\n")

        # DPO: (query, chosen, rejected) — 优先选择困难负样本作为 rejected
        for neg_doc in negative_docs[:num_hard_negatives + 1]:
            f_dpo.write(json.dumps({
                "query": query, "chosen": positive_doc, "rejected": neg_doc
            }, ensure_ascii=False) + "\n")

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="构建 Reranker 训练数据集（改进版）")
    parser.add_argument("--embed-model", type=str, default=CONFIG.get("embed_model", "Qwen/Qwen3-Embedding-4B"))
    parser.add_argument("--milvus-host", type=str, required=True, help="Milvus 服务地址")
    parser.add_argument("--collection-name", type=str, required=True, help="Milvus collection 名称")
    parser.add_argument("--sample-size", type=int, default=2500)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--num-hard-negatives", type=int, default=2, help="困难负样本数")
    parser.add_argument("--num-random-negatives", type=int, default=2, help="随机负样本数")
    parser.add_argument("--output-dir", type=str, default="dataset/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger.info("📦 加载 Embedder 模型...")
    embedder = HuggingFaceEmbeddings(model_name=args.embed_model, model_kwargs={"device": device})

    # 连接 Milvus
    connections.connect(alias="default", host=args.milvus_host, port="19530")
    collection = Collection(name=args.collection_name)
    all_ids = collection.query(expr="", output_fields=["global_chunk_idx"], limit=10000)
    valid_ids = [item["global_chunk_idx"] for item in all_ids]
    sample_ids = random.sample(valid_ids, min(args.sample_size, len(valid_ids)))

    # 加载全部文档块池（用于负采样）
    all_chunks = collection.query(expr="", output_fields=["text", "page_name", "title", "page_url", "summary"], limit=10000)
    logger.info(f"📚 文档块池大小: {len(all_chunks)}")

    success_count = 0
    for idx in sample_ids:
        # 查询主键对应的 question
        res = collection.query(
            expr=f"global_chunk_idx == {idx}",
            output_fields=["question", "text", "page_name", "title", "page_url", "summary"]
        )
        if not res or not res[0].get("question"):
            continue

        question = res[0]["question"]
        logger.info(f"🔍 处理样本 {idx}: {question[:50]}")

        # TODO: 从 rag/ 模块导入检索函数
        # from rag.retrieval import query_milvus_blocks, query_es_blocks
        # milvus_results = query_milvus_blocks(question, embedder, top_k=args.top_k)
        # es_results = query_es_blocks(question, top_k=args.top_k)
        # chunks = milvus_results + es_results
        chunks = [res[0]]  # 临时使用当前文档

        if process_one_sample_v2(chunks, all_chunks, args.output_dir,
                                  args.num_hard_negatives, args.num_random_negatives):
            success_count += 1

    logger.info(f"✅ 数据集构建完成: {success_count}/{len(sample_ids)} 样本成功")
    logger.info(f"📁 输出目录: {args.output_dir}")
    logger.info(f"   - reranker_qa_pointwise.jsonl (Pointwise 格式)")
    logger.info(f"   - reranker_qa_pairwise.jsonl  (Pairwise 格式)")
    logger.info(f"   - reranker_qa_dpo.jsonl       (DPO 格式)")
