# -*- coding: utf-8 -*-
"""
Reranker 训练数据集构建模块（改进版）。

生成三种格式的训练数据，支持多种微调策略：

1. **Pointwise** (reranker_qa_pointwise.jsonl)
   - 格式: {"query": "...", "doc": "...", "label": 0/1/2}（多级相关性标注：
     0=无关，1=部分相关，2=高度相关；见下方 `build_relevance_annotation_prompt`。
     【修复审查报告 M5】此前本行误写为 "0/1"，与 `reranker_ft.py` 的
     docstring（`label: 0/1/2`）及本文件 21 行"多级相关性标注（0/1/2）"、
     `parse_relevance_annotation` 的实际取值范围矛盾，属文档表述错误。）
   - 用途: CrossEncoder 多级相关性微调，训练时按 `label/2.0` 归一化到 [0,1]
     （见 `reranker_ft.py`）

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
        # 【修复 M5】回退到二分类标注时此前用 0/1（positive_idx→1，其余→0），
        # 与正常路径的 0/1/2 三级标注语义不一致：`reranker_ft.py` 按
        # `label/MAX_LABEL(=2.0)` 归一化，若数据集中混入这类 0/1 回退样本，
        # 归一化后目标值仅有 0.0/0.5，模型从未见到「完全相关（1.0）」的
        # 目标，削弱排序区分度。现改为 positive_idx→2（与"高度相关"档位
        # 语义对齐），其余→0，保留三级标注的完整语义空间。
        relevance_scores = [2 if i == positive_idx else 0 for i in range(len(chunks))]
        logger.debug("⚠️ 相关性标注解析失败，使用二分类回退（0/2 两档，保留 0/1/2 标签空间的语义对齐）")

    positive_doc = format_chunk_for_reranker(chunks[positive_idx])

    # Step 3: 采样困难负样本（同页面不同段落）
    positive_page = chunks[positive_idx].get("page_name", "")
    # 【修复 N2】此前用 `chunk is not chunks[positive_idx]`（对象身份比较）和
    # `c not in chunks`（dict == 比较，但 chunks 来自检索结果含 15 键、
    # all_chunks_pool 来自全量拉取仅 6 键，键集不同导致 == 恒不等）排除正文档，
    # 二者均恒失效，正文档会泄漏进负样本池污染训练标签。改为以
    # global_chunk_idx 为统一去重键。
    seen_ids = {c.get("global_chunk_idx") for c in chunks}
    hard_negatives = []
    for chunk in all_chunks_pool:
        if chunk.get("global_chunk_idx") in seen_ids:
            continue
        if chunk.get("page_name") == positive_page:
            hard_negatives.append(chunk)
        if len(hard_negatives) >= num_hard_negatives:
            break

    hard_neg_ids = {c.get("global_chunk_idx") for c in hard_negatives}
    candidate_pool = [
        c for c in all_chunks_pool
        if c.get("global_chunk_idx") not in seen_ids
        and c.get("global_chunk_idx") not in hard_neg_ids
    ]
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
            # relevance_scores 此处必然已被上方逻辑赋值（正常标注或回退标注），
            # 这里的 else 分支仅作防御性兜底；同步为 2/0（而非 1/0）以保持与
            # 上方回退逻辑一致的 0/1/2 标签语义（M5）。
            label = relevance_scores[i] if relevance_scores else (2 if i == positive_idx else 0)
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


def _query_all_blocks(collection: "Collection", output_fields: List[str], page_size: int = 10000) -> List[Dict]:
    """分页拉取 Milvus collection 的全部记录（审查报告 L4 修复）。

    此前的调用方式 `collection.query(expr="", ..., limit=10000)` 存在两个问题：
    1. `expr=""`（空表达式）在部分 Milvus/pymilvus 版本会被判定为非法表达式
       而直接抛异常（且未包裹在 try 中，会直接终止脚本）；
    2. `limit=10000` 硬编码，集合超过 1 万条时会静默截断，采样池/负采样池
       实际远小于真实语料规模而不易察觉。

    改用合法的非空表达式 `global_chunk_idx >= 0`（该字段为非负自增主键，
    等价于"匹配全部记录"但语义合法、不依赖空表达式的未定义行为），并按
    `offset`/`limit` 分页拉取直到覆盖 `collection.num_entities`，避免大集合
    下的静默截断。
    """
    total = collection.num_entities
    if total <= 0:
        return []
    results: List[Dict] = []
    offset = 0
    while offset < total:
        batch = collection.query(
            expr="global_chunk_idx >= 0",
            output_fields=output_fields,
            limit=min(page_size, total - offset),
            offset=offset,
        )
        if not batch:
            break
        results.extend(batch)
        offset += len(batch)
    return results


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
    parser.add_argument(
        "--rag-backend", type=str, default="milvus", choices=["milvus", "local"],
        help="检索使用的向量库后端：milvus（默认，复用 --milvus-host/--collection-name）或 local（rag/ 本地降级库）",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 在导入 rag.* 之前设置环境变量，确保 rag/config.py 读取到与本脚本一致的
    # Milvus host / collection name（RAG_CONFIG 在模块导入时即计算完毕）。
    os.environ.setdefault("RAG_VECTOR_BACKEND", args.rag_backend)
    os.environ.setdefault("RAG_KEYWORD_BACKEND", "es" if args.rag_backend == "milvus" else "local")
    os.environ.setdefault("MILVUS_HOST_DEV", args.milvus_host)
    os.environ.setdefault("MILVUS_COLLECTION_DEV", args.collection_name)

    from rag.retrieval import hybrid_search as rag_hybrid_search

    # 【修复 N11】此前在此加载 ~4B 参数的 HuggingFaceEmbeddings 后全文从未引用
    # （向量检索由 rag_hybrid_search 自带 embedder 完成），白白占用显存/拖慢
    # 启动甚至可能 OOM。已移除。
    # 【修复 N12】此前 _query_all_blocks 的 output_fields 不含 question，随后对
    # 每条抽样再发一次 Milvus query 取 question（N 次额外网络往返）。改为在
    # _query_all_blocks 一次性拉取 question 字段，用字典 O(1) 查找。

    # 连接 Milvus
    connections.connect(alias="default", host=args.milvus_host, port="19530")
    collection = Collection(name=args.collection_name)
    all_chunks = _query_all_blocks(
        collection,
        ["global_chunk_idx", "text", "page_name", "title", "page_url", "summary", "question"],
    )
    # 构建 idx→chunk 字典，供后续 O(1) 查找 question（替代逐条 Milvus query）
    chunk_by_idx = {item["global_chunk_idx"]: item for item in all_chunks}
    valid_ids = [item["global_chunk_idx"] for item in all_chunks if item.get("question")]
    sample_ids = random.sample(valid_ids, min(args.sample_size, len(valid_ids)))
    logger.info(f"📚 文档块池大小: {len(all_chunks)}")

    success_count = 0
    for idx in sample_ids:
        res = chunk_by_idx.get(idx)
        if not res or not res.get("question"):
            continue

        question = res["question"]
        logger.info(f"🔍 处理样本 {idx}: {question[:50]}")

        # 接入 rag/ 真实双模检索（对齐 TODO.md T6：回填依赖）：
        # 向量检索 + 关键词检索 → 融合去重，取代原先 chunks = [res[0]] 的占位实现。
        # 检索到的 rag.schema.DocBlock 字段与本文件的 chunk dict 字段（text/page_name/
        # title/page_url/summary）兼容，可直接传入 format_chunk_for_reranker。
        try:
            milvus_results = rag_hybrid_search.vector_search(question, top_k=args.top_k)
            es_results = rag_hybrid_search.keyword_search(question, top_k=args.top_k)
            fused_blocks = rag_hybrid_search.fuse([milvus_results, es_results])
            chunks = [b.to_dict(with_embedding=False) for b in fused_blocks] or [res]
        except Exception as e:
            logger.warning(f"⚠️ rag/ 检索失败（{e}），回退为当前文档单条占位")
            chunks = [res]

        if process_one_sample_v2(chunks, all_chunks, args.output_dir,
                                  args.num_hard_negatives, args.num_random_negatives):
            success_count += 1

    logger.info(f"✅ 数据集构建完成: {success_count}/{len(sample_ids)} 样本成功")
    logger.info(f"📁 输出目录: {args.output_dir}")
    logger.info(f"   - reranker_qa_pointwise.jsonl (Pointwise 格式)")
    logger.info(f"   - reranker_qa_pairwise.jsonl  (Pairwise 格式)")
    logger.info(f"   - reranker_qa_dpo.jsonl       (DPO 格式)")
