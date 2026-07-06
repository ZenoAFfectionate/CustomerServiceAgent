# -*- coding: utf-8 -*-
"""生成质量评测（Generation Evaluation）：关联度（groundedness）、引用覆盖率，
以及（提供参考答案时）与参考答案的词汇相似度。

评测用例格式：
    [{"query": str, "reference_answer": str（可选）}, ...]
"""
from typing import List, Optional
import time

from rag import pipeline
from rag.evaluation import metrics
from rag.generation.hallucination_control import check_citation_validity, groundedness_score
from rag.schema import DocBlock


def evaluate_generation(cases: List[dict], top_k: int = 5) -> dict:
    """批量评测生成质量。

    Returns:
        {"num_cases": int, "citation_coverage": float, "avg_groundedness": float,
         "avg_lexical_f1": float（无参考答案的用例不计入该项均值）,
         "avg_latency_ms": float, "per_case": [...]}

        `per_case` 每项均含 `latency_ms`（单条 `pipeline.answer()` 调用耗时），
        供 `rag_e2e_eval.run_e2e_eval()` 直接复用来计算延迟分位数报告，避免
        额外再对同一批 query 重跑一遍 `benchmark_latency()`（重复消耗真实
        Milvus/ES/vLLM 后端的调用配额与耗时）。
    """
    per_case = []
    for case in cases:
        query = case["query"]
        reference = case.get("reference_answer")

        t0 = time.time()
        result = pipeline.answer(query, top_k=top_k)
        latency_ms = round((time.time() - t0) * 1000, 2)

        contexts = [DocBlock.from_dict(c) for c in result["contexts"]]
        validity = check_citation_validity(result["answer"], len(contexts))
        score = groundedness_score(result["answer"], contexts) if contexts else 0.0
        entry = {
            "query": query,
            "backend_used": result["backend_used"],
            "has_citation": validity["has_citation"],
            "citation_valid": validity["valid"],
            "groundedness": score,
            "latency_ms": latency_ms,
        }
        if reference:
            entry["lexical_f1"] = metrics.lexical_f1(result["answer"], reference)
        per_case.append(entry)

    lexical_scores = [c["lexical_f1"] for c in per_case if "lexical_f1" in c]
    return {
        "num_cases": len(per_case),
        "citation_coverage": metrics.aggregate_mean([1.0 if c["has_citation"] else 0.0 for c in per_case]),
        "avg_groundedness": metrics.aggregate_mean([c["groundedness"] for c in per_case]),
        "avg_lexical_f1": metrics.aggregate_mean(lexical_scores) if lexical_scores else None,
        "avg_latency_ms": metrics.aggregate_mean([c["latency_ms"] for c in per_case]),
        "per_case": per_case,
    }
