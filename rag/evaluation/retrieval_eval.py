# -*- coding: utf-8 -*-
"""检索质量评测（Retrieval Evaluation）：对齐标注的 query→相关文档集合，
计算 Recall@K / Precision@K / MRR / NDCG@K。

评测用例格式（与 `rag/pipeline.retrieve` 的输出天然兼容）：
    [{"query": str, "relevant_ids": [global_chunk_idx, ...]}, ...]
"""
from typing import List, Optional

from rag import pipeline
from rag.evaluation import metrics


def evaluate_retrieval(cases: List[dict], top_k: int = 5) -> dict:
    """批量评测检索质量。

    Args:
        cases: [{"query": str, "relevant_ids": [int, ...]}, ...]
        top_k: 每条 query 的检索条数

    Returns:
        {"num_cases": int, "recall@k": float, "precision@k": float,
         "mrr": float, "ndcg@k": float, "per_case": [...]}
    """
    per_case = []
    for case in cases:
        query = case["query"]
        relevant_ids = case.get("relevant_ids", [])
        results = pipeline.retrieve(query, top_k=top_k)
        retrieved_ids = [r.get("global_chunk_idx") for r in results]

        per_case.append({
            "query": query,
            "recall@k": metrics.recall_at_k(retrieved_ids, relevant_ids, top_k),
            "precision@k": metrics.precision_at_k(retrieved_ids, relevant_ids, top_k),
            "mrr": metrics.mrr(retrieved_ids, relevant_ids),
            "ndcg@k": metrics.ndcg_at_k(retrieved_ids, relevant_ids, top_k),
            "num_retrieved": len(retrieved_ids),
        })

    return {
        "num_cases": len(per_case),
        "recall@k": metrics.aggregate_mean([c["recall@k"] for c in per_case]),
        "precision@k": metrics.aggregate_mean([c["precision@k"] for c in per_case]),
        "mrr": metrics.aggregate_mean([c["mrr"] for c in per_case]),
        "ndcg@k": metrics.aggregate_mean([c["ndcg@k"] for c in per_case]),
        "per_case": per_case,
    }
