# -*- coding: utf-8 -*-
"""通用评估指标（Metrics）：检索指标（Recall@K / Precision@K / MRR / NDCG）与
生成指标（词汇重叠 F1）。均为无外部依赖的纯函数，方便单测与复用。
"""
import math
from typing import List, Sequence


def recall_at_k(retrieved_ids: Sequence, relevant_ids: Sequence, k: int) -> float:
    """Recall@K = 命中的相关文档数 / 相关文档总数。"""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    hit = set(retrieved_ids[:k]) & relevant
    return round(len(hit) / len(relevant), 4)


def precision_at_k(retrieved_ids: Sequence, relevant_ids: Sequence, k: int) -> float:
    """Precision@K = 命中的相关文档数 / min(K, 检索返回数)。"""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    relevant = set(relevant_ids)
    hit = sum(1 for i in top_k if i in relevant)
    return round(hit / len(top_k), 4)


def mrr(retrieved_ids: Sequence, relevant_ids: Sequence) -> float:
    """Mean Reciprocal Rank：第一个命中相关文档的排名的倒数（未命中为 0）。"""
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return round(1.0 / rank, 4)
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence, relevant_ids: Sequence, k: int) -> float:
    """NDCG@K（二元相关性版本：命中记 1，否则记 0）。"""
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc_id in enumerate(retrieved_ids[:k], start=1)
        if doc_id in relevant
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return round(dcg / idcg, 4) if idcg > 0 else 0.0


def _char_tokens(text: str) -> List[str]:
    """按字符级 token 化（对中文场景比空格分词更稳健的粗粒度近似）。"""
    return list((text or "").replace(" ", ""))


def lexical_f1(pred_text: str, ref_text: str) -> float:
    """预测文本与参考文本的字符级 token 重叠 F1（衡量表层文本相似度，
    不涉及语义，仅作为快速的自动化评测代理指标）。
    """
    pred_tokens = _char_tokens(pred_text)
    ref_tokens = _char_tokens(ref_text)
    if not pred_tokens or not ref_tokens:
        return 0.0
    pred_set, ref_set = set(pred_tokens), set(ref_tokens)
    overlap = len(pred_set & ref_set)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_set)
    recall = overlap / len(ref_set)
    return round(2 * precision * recall / (precision + recall), 4)


def aggregate_mean(values: List[float]) -> float:
    """对一批指标值求平均（空列表返回 0.0，避免除零异常）。"""
    return round(sum(values) / len(values), 4) if values else 0.0
