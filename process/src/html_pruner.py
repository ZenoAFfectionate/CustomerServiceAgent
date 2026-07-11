# -*- coding: utf-8 -*-
"""
HtmlRAG 两阶段块树剪枝（Block Tree Pruning）。

对标论文 HtmlRAG（*HTML is Better Than Plain Text for Modeling Retrieved
Knowledge in RAG Systems*）的核心贡献之一：**Two-Stage Block Tree Pruning**。
在「HTML 清洗 → 结构化分块」之后、送入 LLM 之前，根据用户 query 对块树做
两阶段剪枝，既压缩上下文 token，又保留最相关、且保留 HTML 结构的块。

流程：
    清洗后的 HTML
        │
        ▼  Stage 1（粗剪，Embedding-based Pruning）
    在「粗粒度」块树上，用文本嵌入相似度快速裁掉与 query 无关的块
        │
        ▼  Stage 2（精剪，Fine-grained Pruning）
    在 Stage 1 结果的「细粒度」块树上，用更精细的相关性打分（本项目用
    Reranker 交叉编码作为论文「生成式细粒度剪枝」的可部署替代）二次剪枝
        │
        ▼
    剪枝后的 HTML（保留结构，对标论文「HTML 优于纯文本」）→ 送入 LLM

设计要点：
    - **算法与打分后端解耦**：核心贪心剪枝算法（`greedy_prune_indices`）、
      余弦相似度（`cosine_similarity_vec`）、HTML 重建（`rebuild_html`）均为
      纯函数，不依赖任何外部服务，可独立单测。
    - **打分器可注入**：`embed_fn` / `rerank_fn` 以回调形式注入，生产环境使用
      TEI / vLLM，测试环境可用确定性 mock，无需真实起服务。
    - **优雅降级**：打分服务不可用时不崩溃，回退为「不剪枝」（保留全部块）。

核心函数：
    - greedy_prune_indices:  贪心选块（纯算法，token 预算内保留高分块）
    - cosine_similarity_vec: 余弦相似度（纯函数）
    - rebuild_html:          从保留的块重建 HTML（保留结构）
    - prune_by_embedding:    Stage 1 嵌入粗剪
    - prune_by_reranker:     Stage 2 精排精剪
    - two_stage_prune:       完整两阶段剪枝入口
"""

import os
import math
from typing import Callable, List, Optional, Sequence

from bs4 import Tag

from html_utils import build_block_tree
from utils.config import CONFIG, logger

# 打分器类型别名
EmbedFn = Callable[[List[str]], List[List[float]]]
RerankFn = Callable[[str, List[str]], List[float]]


# ======================== 纯算法：余弦相似度 ========================

def cosine_similarity_vec(a: Sequence[float], b: Sequence[float]) -> float:
    """计算两个向量的余弦相似度。

    任一向量为零向量（模长为 0）时返回 0.0，避免除零。
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ======================== 纯算法：贪心剪枝 ========================

def greedy_prune_indices(
    word_counts: Sequence[int],
    scores: Sequence[float],
    max_words: int,
) -> List[int]:
    """贪心选块：在词数预算内，优先保留高分块，返回保留块的原始下标（升序）。

    对标论文的剪枝目标——在给定上下文预算下最大化保留高相关块。等价于按分数
    从高到低依次尝试加入，只要不超预算就保留；分数相同时优先保留文档靠前的块。

    Args:
        word_counts: 每个块的词数（与 scores 一一对应）
        scores:      每个块的相关性分数（越大越相关）
        max_words:   保留块的总词数预算

    Returns:
        保留块的下标列表，按原始文档顺序（升序）排列。

    说明：
        - 即使预算不足以容纳任何块，也**至少保留分数最高的 1 个块**，避免返回空
          上下文导致下游无内容可用。
        - 分数相同时按下标升序优先（保持文档原有顺序的稳定性）。
    """
    n = len(word_counts)
    if n == 0:
        return []
    if len(scores) != n:
        raise ValueError(f"word_counts 与 scores 长度不一致: {n} vs {len(scores)}")

    # 按 (分数降序, 下标升序) 排序：分数高者优先；分数相同则文档靠前者优先
    order = sorted(range(n), key=lambda i: (-scores[i], i))

    kept: List[int] = []
    total = 0
    for i in order:
        w = word_counts[i]
        if not kept:
            # 至少保留分数最高的一个块（即使超预算）
            kept.append(i)
            total += w
        elif total + w <= max_words:
            kept.append(i)
            total += w
    return sorted(kept)


# ======================== 纯算法：HTML 重建 ========================

def rebuild_html(blocks: Sequence[Tag]) -> str:
    """从保留的块标签重建 HTML 字符串（保留每个块的 HTML 结构）。

    按块在列表中的顺序拼接 `str(tag)`，块间以换行分隔。对标论文观点：保留
    HTML 结构（而非退化为纯文本）能为下游 LLM 提供更丰富的结构信息。
    """
    return "\n".join(str(b) for b in blocks)


def rebuild_html_with_domains(
    blocks: Sequence[Tag],
    paths: Sequence[List[str]],
) -> str:
    """从保留的块标签重建 HTML，保留 heading domain 层次结构。

    对标论文 HtmlRAG 的核心观点：保留 HTML 层次结构能为 LLM 提供更丰富的
    上下文信息。将同一 heading domain 下的连续块包装在 <div class="hN_domain">
    中，使 LLM 能理解块之间的从属关系。

    Args:
        blocks: 保留的块标签列表
        paths:  每个块的路径列表（如 ["h1_domain", "h2_domain", "p"]）

    Returns:
        保留层次结构的 HTML 字符串
    """
    if not blocks:
        return ""

    # 提取每个块的顶级 domain（如 "h1_domain" / "h2_domain" / "isolated_domain"）
    top_domains = []
    for path in paths:
        if path:
            # 找第一个以 _domain 结尾的路径段
            domain = next((p for p in path if p.endswith("_domain")), "")
            top_domains.append(domain)
        else:
            top_domains.append("")

    # 按 domain 分组连续的块
    result_parts = []
    current_domain = top_domains[0] if top_domains else ""
    current_group = []

    for i, (block, domain) in enumerate(zip(blocks, top_domains)):
        if domain != current_domain and current_group:
            # 提交当前组
            result_parts.append(_wrap_domain_group(current_group, current_domain))
            current_group = []
            current_domain = domain
        current_group.append(block)

    # 提交最后一组
    if current_group:
        result_parts.append(_wrap_domain_group(current_group, current_domain))

    return "\n".join(result_parts)


def _wrap_domain_group(blocks: List[Tag], domain: str) -> str:
    """将一组块包装在 domain div 中。"""
    inner = "\n".join(str(b) for b in blocks)
    if domain and domain != "isolated_domain":
        return f'<div class="{domain}">\n{inner}\n</div>'
    return inner


# ======================== 内部工具 ========================

def _tag_word_count(tag: Tag, zh_char: bool) -> int:
    """计算块的词数（zh_char=True 时按字符数，适配中文）。"""
    text = tag.get_text()
    return len(text) if zh_char else len(text.split())


def _extract_heading_context(path: List[str]) -> str:
    """从块路径中提取标题上下文文本。

    例如路径 ["h1_domain", "h2_domain", "p"] → "h1 h2"，
    或路径 ["h1_section"] → "h1"，
    用于在打分时为块提供所属章节的上下文信息。
    """
    contexts = []
    for segment in path:
        # 匹配 hN_domain 或 hN_section 格式
        if segment.endswith("_domain") and segment.startswith("h"):
            contexts.append(segment.replace("_domain", ""))
        elif segment.endswith("_section") and segment.startswith("h"):
            contexts.append(segment.replace("_section", ""))
    return " ".join(contexts)


def _prune_html_by_scores(
    html: str,
    query: str,
    score_texts_fn: Callable[[str, List[str]], List[float]],
    max_context_words: int,
    max_node_words: int,
    min_node_words: int,
    zh_char: bool,
    stage_name: str,
) -> str:
    """通用剪枝骨架：分块 → 打分 → 贪心选块 → 重建 HTML。

    Args:
        score_texts_fn: (query, block_texts) -> scores，两个阶段各自注入不同后端。
        stage_name:     日志用阶段名。

    打分失败时优雅降级为「不剪枝」（重建全部块）。

    改进：在打分时为每个块附加标题上下文（如"h1 h2"），使打分器
    能感知块所属的章节层级，提升相关性判断的准确性。
    """
    block_tree, _ = build_block_tree(
        html, max_node_words=max_node_words,
        min_node_words=min_node_words, zh_char=zh_char,
    )
    if not block_tree:
        # HTML 过短或无有效块：无需剪枝，原样返回
        logger.debug(f"[{stage_name}] 块树为空，跳过剪枝")
        return html

    blocks = [item[0] for item in block_tree]
    paths = [item[1] for item in block_tree]
    # 附加标题上下文到块文本，提升打分准确性
    texts = []
    for block, path in zip(blocks, paths):
        block_text = block.get_text()
        heading_ctx = _extract_heading_context(path)
        if heading_ctx:
            texts.append(f"[{heading_ctx}] {block_text}")
        else:
            texts.append(block_text)
    word_counts = [_tag_word_count(t, zh_char) for t in blocks]

    try:
        scores = score_texts_fn(query, texts)
        if len(scores) != len(blocks):
            raise ValueError(
                f"打分数量({len(scores)})与块数({len(blocks)})不一致"
            )
    except Exception as e:
        logger.warning(f"[{stage_name}] 打分失败，降级为不剪枝: {e}")
        return rebuild_html(blocks)

    keep = greedy_prune_indices(word_counts, scores, max_context_words)
    logger.info(
        f"[{stage_name}] 原始 {len(blocks)} 块 → 保留 {len(keep)} 块"
        f"（预算 {max_context_words} 词）"
    )
    kept_blocks = [blocks[i] for i in keep]
    kept_paths = [paths[i] for i in keep]
    return rebuild_html_with_domains(kept_blocks, kept_paths)


# ======================== Stage 1：嵌入粗剪 ========================

def prune_by_embedding(
    html: str,
    query: str,
    max_context_words: int = 4096,
    embed_fn: Optional[EmbedFn] = None,
    max_node_words: int = 512,
    min_node_words: int = 32,
    zh_char: bool = True,
) -> str:
    """Stage 1：基于文本嵌入的粗粒度剪枝。

    在「粗粒度」块树上，用 query 与各块的嵌入余弦相似度打分，贪心保留高分块。
    速度快、召回宽，用于先裁掉明显无关的大块子树。

    Args:
        html:              清洗后的 HTML
        query:             用户查询
        max_context_words: 剪枝后保留的总词数预算
        embed_fn:          (texts) -> vectors 嵌入回调；None 时使用默认 vLLM/TEI 后端
        max_node_words:    分块最大词数（粗粒度，值较大）
        min_node_words:    分块最小词数
        zh_char:           是否按字符计数（中文场景 True）

    Returns:
        剪枝后的 HTML 字符串。
    """
    _embed_fn = embed_fn or default_embed_fn

    def _score(q: str, texts: List[str]) -> List[float]:
        # 单次请求同时嵌入 query 与所有块，减少往返
        vectors = _embed_fn([q] + texts)
        q_vec, block_vecs = vectors[0], vectors[1:]
        return [cosine_similarity_vec(q_vec, v) for v in block_vecs]

    return _prune_html_by_scores(
        html, query, _score, max_context_words,
        max_node_words, min_node_words, zh_char, stage_name="Stage1-Embedding",
    )


# ======================== Stage 2：精排精剪 ========================

def prune_by_reranker(
    html: str,
    query: str,
    max_context_words: int = 2048,
    rerank_fn: Optional[RerankFn] = None,
    max_node_words: int = 256,
    min_node_words: int = 16,
    zh_char: bool = True,
) -> str:
    """Stage 2：基于 Reranker 交叉编码的细粒度剪枝。

    在「细粒度」块树上，用 Reranker 对 (query, block) 逐一精细打分后贪心保留。
    Reranker 的交叉编码相关性判断远比嵌入点积精确，作为论文「生成式细粒度剪枝」
    的可部署替代（本项目已具备 TEI Reranker 服务）。

    Args:
        html:              Stage 1 剪枝后的 HTML
        query:             用户查询
        max_context_words: 剪枝后保留的总词数预算（通常小于 Stage 1）
        rerank_fn:         (query, texts) -> scores 精排回调；None 时使用默认 TEI 后端
        max_node_words:    分块最大词数（细粒度，值较小）
        min_node_words:    分块最小词数
        zh_char:           是否按字符计数

    Returns:
        剪枝后的 HTML 字符串。
    """
    _rerank_fn = rerank_fn or default_rerank_fn
    return _prune_html_by_scores(
        html, query, _rerank_fn, max_context_words,
        max_node_words, min_node_words, zh_char, stage_name="Stage2-Reranker",
    )


# ======================== 完整两阶段入口 ========================

def two_stage_prune(
    html: str,
    query: str,
    stage1_max_context_words: int = 4096,
    stage2_max_context_words: int = 2048,
    embed_fn: Optional[EmbedFn] = None,
    rerank_fn: Optional[RerankFn] = None,
    stage1_max_node_words: int = 512,
    stage2_max_node_words: int = 256,
    min_node_words: int = 32,
    stage2_min_node_words: int = 16,
    zh_char: bool = True,
) -> str:
    """HtmlRAG 完整两阶段剪枝入口：Embedding 粗剪 → Reranker 精剪。

    Args:
        html:                      清洗后的 HTML
        query:                     用户查询
        stage1_max_context_words:  Stage 1 后保留的词数预算（较大）
        stage2_max_context_words:  Stage 2 后保留的词数预算（较小，最终上下文）
        embed_fn / rerank_fn:      打分回调，None 时使用默认后端
        stage1_max_node_words:     Stage 1 粗粒度块最大词数
        stage2_max_node_words:     Stage 2 细粒度块最大词数
        min_node_words:            Stage 1 分块最小词数
        stage2_min_node_words:     Stage 2 分块最小词数
        zh_char:                   是否按字符计数

    Returns:
        两阶段剪枝后的最终 HTML 上下文。
    """
    logger.info(f"🔪 两阶段剪枝开始，query={query[:48]!r}")

    stage1_html = prune_by_embedding(
        html, query,
        max_context_words=stage1_max_context_words,
        embed_fn=embed_fn,
        max_node_words=stage1_max_node_words,
        min_node_words=min_node_words,
        zh_char=zh_char,
    )

    stage2_html = prune_by_reranker(
        stage1_html, query,
        max_context_words=stage2_max_context_words,
        rerank_fn=rerank_fn,
        max_node_words=stage2_max_node_words,
        min_node_words=stage2_min_node_words,
        zh_char=zh_char,
    )

    logger.info("✅ 两阶段剪枝完成")
    return stage2_html


# ======================== 默认打分后端（外部服务） ========================

def default_embed_fn(texts: List[str]) -> List[List[float]]:
    """默认嵌入后端：调用 vLLM/TEI 的 OpenAI 风格 embeddings 接口批量嵌入。

    读取 CONFIG['embed_api_url'] 与 CONFIG['embed_model']。测试环境应注入 mock，
    不会走到此函数。
    """
    import requests

    url = CONFIG.get("embed_api_url", "http://localhost:8010/v1/embeddings")
    payload = {
        "model": CONFIG.get("embed_model", "Qwen/Qwen3-Embedding-4B"),
        "input": texts,
    }
    resp = requests.post(url, json=payload, timeout=CONFIG.get("vllm_timeout", 60))
    resp.raise_for_status()
    data = resp.json()["data"]
    # 按 index 还原顺序，确保与输入一一对应
    data = sorted(data, key=lambda d: d.get("index", 0))
    return [d["embedding"] for d in data]


def default_rerank_fn(query: str, texts: List[str]) -> List[float]:
    """默认精排后端：调用 vLLM /rerank 接口，返回与输入顺序一致的分数列表。

    读取环境变量 RERANK_API_URL（默认 http://localhost:8012），支持 TEI 和 vLLM
    两种返回格式。测试环境应注入 mock，不会走到此函数。
    """
    import requests

    if not texts:
        return []
    url = os.environ.get("RERANK_API_URL", "http://localhost:8012")
    resp = requests.post(
        f"{url}/rerank",
        json={"query": query, "documents": texts},
        timeout=CONFIG.get("vllm_timeout", 60),
    )
    resp.raise_for_status()
    data = resp.json()

    scores = [0.0] * len(texts)
    # vLLM 格式: {"results": [{"index": N, "relevance_score": S}, ...]}
    if "results" in data:
        for item in data["results"]:
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(texts):
                scores[idx] = item.get("relevance_score", 0.0)
    # TEI 格式: [{"index": N, "score": S}, ...]
    elif isinstance(data, list):
        for item in data:
            idx = item.get("index")
            if idx is not None and 0 <= idx < len(texts):
                scores[idx] = item.get("score", 0.0)
    return scores
