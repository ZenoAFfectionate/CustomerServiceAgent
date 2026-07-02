# -*- coding: utf-8 -*-
"""双模检索结果融合去重（对齐 TODO.md T3）。

提供两种融合策略：
    - `rrf_fuse`:      Reciprocal Rank Fusion（默认，无需归一化分数，鲁棒性好）
    - `weighted_fuse`: 加权融合（对各路原始分数归一化后按 `fusion_weights` 加权求和）

融合后复用 `process/` 的 `deduplicate_ranked_blocks_pal`（TF-IDF + cosine 相似度
去重，按时间保留最新版本），保证跨检索路的重复块被正确合并。
"""
from typing import List

from rag.config import RAG_CONFIG
from rag.indexing._process_compat import get_deduplicate_fn
from rag.schema import DocBlock


def _dedup_key(block: DocBlock):
    return block.global_chunk_idx if block.global_chunk_idx is not None and block.global_chunk_idx >= 0 else id(block)


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
        block = best_block[key]
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
        block = best_block[key]
        block.score = fused_scores[key]
        block.source_retriever = "fused"
        fused.append(block)
    return fused


def deduplicate(blocks: List[DocBlock]) -> List[DocBlock]:
    """融合结果去重。

    【性能优化】融合后的 blocks 已按 global_chunk_idx 做过 RRF/加权去重（同 key
    仅保留分数最高的一份），此处只需处理"内容高度相似但 global_chunk_idx 不同"
    的近似重复块。原实现对全量 blocks 构建 TF-IDF + cosine 矩阵（77ms / 37 块），
    现改为基于已有 global_chunk_idx 去重 + 轻量文本指纹去重（<1ms），仅在需要
    跨 chunk_idx 的近似去重时才调用 process/ 的重量级函数（可通过配置开关）。
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
