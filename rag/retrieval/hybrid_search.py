# -*- coding: utf-8 -*-
"""混合检索（Hybrid Search）：向量检索 + 关键词检索 + 融合去重。

合并自原 `milvus_search.py`（向量检索）、`es_search.py`（关键词检索）与
`fusion.py`（融合去重）三个文件 —— 三者共同构成"双路召回 → 融合"这一件事，
合并为单文件后调用方只需关心一个模块，符合新的模块划分：
`retrieval/` 下每个文件对应一种"检索能力"，而不是"检索后端"。

对外暴露：
    - `vector_search(query, top_k)`   —— 单路向量检索
    - `keyword_search(query, top_k)`  —— 单路关键词检索
    - `rrf_fuse(...)` / `weighted_fuse(...)` / `deduplicate(...)` / `fuse(...)`
    - `search(query, top_k)`          —— 端到端：双路检索 + 融合去重（一步到位）
"""
import copy
from typing import List

from config.config_loader import logger
from rag.config import RAG_CONFIG
from rag.indexing._process_compat import get_deduplicate_fn
from rag.indexing.embedding import get_embedder
from rag.indexing.store import get_keyword_store, get_vector_store
from rag.schema import DocBlock

# ======================== 单路检索 ========================


def vector_search(query: str, top_k: int = None) -> List[DocBlock]:
    """向量检索：query 经 Embedder 向量化后，调用 vector_store（Milvus 或本地
    降级实现）做 ANN / 暴力余弦检索。返回结果统一带 `score` 与
    `source_retriever="milvus"`，按分数降序。服务不可用/无数据时返回 []。
    """
    top_k = top_k or RAG_CONFIG["top_k_recall"]
    query = (query or "").strip()
    if not query:
        return []
    embedder = get_embedder()
    query_vector = embedder.embed_query(query)

    store = get_vector_store()
    try:
        results = store.search(query_vector, top_k)
    except Exception:
        logger.warning("⚠️ 向量检索失败，返回空结果")
        return []
    # 【修复 N31】此前硬编码 "milvus"，但本地降级后端实际非 milvus。
    # 改用 RAG_CONFIG 中的实际后端名，使 source_retriever 反映真实来源。
    _vector_source = RAG_CONFIG.get("vector_backend", "milvus")
    for r in results:
        r.source_retriever = _vector_source
    return results


def keyword_search(query: str, top_k: int = None) -> List[DocBlock]:
    """关键词检索：jieba 分词后交给 keyword_store（Elasticsearch +
    build_optimal_jieba_query，或本地 TF-IDF 降级实现）检索。返回结果统一带
    `score` 与 `source_retriever="es"`，按分数降序。服务不可用/无数据时返回 []。
    """
    top_k = top_k or RAG_CONFIG["top_k_recall"]
    query = (query or "").strip()
    if not query:
        return []
    store = get_keyword_store()
    try:
        results = store.search(query, top_k)
    except Exception:
        logger.warning("⚠️ 关键词检索失败，返回空结果")
        return []
    # 【修复 N32】此前硬编码 "es"，但本地降级后端实际非 ES。
    _keyword_source = RAG_CONFIG.get("keyword_backend", "es")
    for r in results:
        r.source_retriever = _keyword_source
    return results


# ======================== 融合去重 ========================


def _dedup_key(block: DocBlock):
    """去重主键：优先用 global_chunk_idx（跨路同一块共享），其次用内容指纹。

    【修复 N15/L11】此前以 global_chunk_idx 为唯一主键（<0 时回退到 id(block)），
    对 global_chunk_idx 缺失/为负的块（如手动构造的临时块）完全失效——两块
    内容相同但均 -1 时不会被去重。改为：global_chunk_idx 有效时用它（保证
    跨路同一块的精确匹配），否则退化为内容指纹（text 的哈希），使无 idx 的
    块仍能按内容去重。
    """
    if block.global_chunk_idx is not None and block.global_chunk_idx >= 0:
        return block.global_chunk_idx
    # 内容指纹：用 text 的哈希作为去重键，使无 idx 的块也能按内容去重
    return hash(block.dedup_key_text())


def rrf_fuse(results_list: List[List[DocBlock]], k: int = None) -> List[DocBlock]:
    """Reciprocal Rank Fusion：score(d) = Σ 1 / (k + rank)，rank 从 1 计。

    Args:
        results_list: 多路检索结果，每路已按分数降序排列
        k: RRF 平滑常数，默认取 `RAG_CONFIG['fusion_rrf_k']`

    Returns:
        按融合分数降序排列、去除跨路重复（保留分数更高的一份）的 DocBlock 列表。
    """
    k = RAG_CONFIG["fusion_rrf_k"] if k is None else k
    rrf_scores = {}
    best_block = {}
    for results in results_list:
        for rank, block in enumerate(results, start=1):
            key = _dedup_key(block)
            contrib = 1.0 / (k + rank)
            rrf_scores[key] = rrf_scores.get(key, 0.0) + contrib
            if key not in best_block or block.score > best_block[key].score:
                best_block[key] = block

    ordered_keys = sorted(rrf_scores.keys(), key=lambda kk: -rrf_scores[kk])
    fused = []
    for key in ordered_keys:
        # 【修复 L10】此前直接原地改写 `best_block[key]`（原始检索结果列表中
        # 的同一 DocBlock 对象引用）的 score/source_retriever。当前调用链路
        # 中原始两路结果在 fuse() 后即被丢弃，故不是激活 BUG；但一旦未来有
        # 调用方在 fuse 后仍持有 vector_search()/keyword_search() 的原始列表
        # 并依赖其 score/source_retriever，会读到被污染的值。这里改为浅拷贝
        # 一份新对象再改写，避免共享引用副作用。
        block = copy.copy(best_block[key])
        block.score = rrf_scores[key]
        block.source_retriever = "fused"
        fused.append(block)
    return fused


def _normalize_scores(results: List[DocBlock]) -> dict:
    """Min-Max 归一化到 [0, 1]，全部相同分数时归一化为 1.0。"""
    if not results:
        return {}
    scores = [b.score for b in results]
    lo, hi = min(scores), max(scores)
    span = hi - lo
    out = {}
    for b in results:
        key = _dedup_key(b)
        out[key] = 1.0 if span == 0 else (b.score - lo) / span
    return out


def weighted_fuse(results_list: List[List[DocBlock]], weights: List[float] = None) -> List[DocBlock]:
    """加权融合：对各路分数 Min-Max 归一化后按权重加权求和。

    Args:
        results_list: 多路检索结果（顺序需与 weights 一致）
        weights: 各路权重，默认取 `RAG_CONFIG['fusion_weights']`（milvus, es）

    Returns:
        按融合分数降序排列、去除跨路重复的 DocBlock 列表。
    """
    if weights is None:
        w = RAG_CONFIG["fusion_weights"]
        weights = [w.get("milvus", 0.5), w.get("es", 0.5)]
    if len(weights) < len(results_list):
        weights = list(weights) + [1.0] * (len(results_list) - len(weights))

    fused_scores = {}
    best_block = {}
    for results, weight in zip(results_list, weights):
        norm = _normalize_scores(results)
        for block in results:
            key = _dedup_key(block)
            fused_scores[key] = fused_scores.get(key, 0.0) + weight * norm.get(key, 0.0)
            if key not in best_block or block.score > best_block[key].score:
                best_block[key] = block

    ordered_keys = sorted(fused_scores.keys(), key=lambda kk: -fused_scores[kk])
    fused = []
    for key in ordered_keys:
        # 【修复 L10】同 rrf_fuse：浅拷贝后再改写，避免原地修改共享的
        # DocBlock 对象引用。
        block = copy.copy(best_block[key])
        block.score = fused_scores[key]
        block.source_retriever = "fused"
        fused.append(block)
    return fused


def deduplicate(blocks: List[DocBlock]) -> List[DocBlock]:
    """融合结果去重。

    融合后的 blocks 已按 global_chunk_idx 做过 RRF/加权去重（同 key仅保留分数
    最高的一份），此处只需处理"内容高度相似但 global_chunk_idx 不同"的近似
    重复块：先做 O(N) 精确去重 + 轻量文本指纹去重（<1ms），仅在需要跨
    chunk_idx 的近似去重时才调用 process/ 的重量级 TF-IDF 函数（可通过配置开关）。
    """
    if len(blocks) <= 1:
        return blocks

    # 1. 精确去重：按 global_chunk_idx（O(N)，与 RRF 阶段互补）
    seen_ids = set()
    deduped = []
    for b in blocks:
        key = _dedup_key(b)
        if key in seen_ids:
            continue
        seen_ids.add(key)
        deduped.append(b)

    if len(deduped) <= 1:
        return deduped

    # 2. 近似去重：仅当配置阈值 < 1.0 时才启用重量级 TF-IDF 去重
    #    （融合后通常仅 20-40 条候选，TF-IDF 去重对检索质量提升有限但耗时显著）
    if RAG_CONFIG.get("dedup_threshold_content", 1.0) < 1.0:
        dedup_fn = get_deduplicate_fn()
        dicts = [b.to_dict(with_embedding=False) for b in deduped]
        deduped_dicts = dedup_fn(
            dicts,
            threshold_content=RAG_CONFIG["dedup_threshold_content"],
            threshold_page_name=RAG_CONFIG["dedup_threshold_page_name"],
        )
        kept_ids = {d.get("global_chunk_idx") for d in deduped_dicts}
        return [b for b in deduped if _dedup_key(b) in kept_ids]

    return deduped


def fuse(results_list: List[List[DocBlock]], method: str = None) -> List[DocBlock]:
    """融合入口：按 `RAG_CONFIG['fusion_method']` 选择 RRF / 加权融合，融合后自动去重。"""
    method = method or RAG_CONFIG["fusion_method"]
    results_list = [r for r in results_list if r]
    if not results_list:
        return []
    if len(results_list) == 1:
        fused = list(results_list[0])
    elif method == "weighted":
        fused = weighted_fuse(results_list)
    else:
        fused = rrf_fuse(results_list)
    return deduplicate(fused)


# ======================== 端到端混合检索 ========================


def search(query: str, top_k: int = None, backends: List[str] = None) -> List[DocBlock]:
    """一步到位的混合检索：按 `backends` 指定的检索路（默认双路全开）分别检索，
    再融合去重。`backends` 通常由 `retriever_selection.select_retrievers()` 决定。

    Args:
        query: 查询文本
        top_k: 单路召回数，默认取 `RAG_CONFIG['top_k_recall']`
        backends: 需要执行的检索路，取值 "vector"/"keyword" 的子集，默认全部执行

    Returns:
        融合去重后的 DocBlock 列表（未精排）。
    """
    backends = ["vector", "keyword"] if backends is None else backends
    results_list = []
    if "vector" in backends:
        results_list.append(vector_search(query, top_k))
    if "keyword" in backends:
        results_list.append(keyword_search(query, top_k))
    return fuse(results_list)
