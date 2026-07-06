# -*- coding: utf-8 -*-
"""端到端评测编排（RAG E2E Evaluation）：组合 `retrieval_eval.py` /
`generation_eval.py`，面向 jsonl/json 数据集批量运行，产出汇总报告。

用法：
    python -m rag.evaluation.rag_e2e_eval --eval-path path/to/eval_cases.json

评测用例文件格式（json 数组）：
    [{"query": str, "relevant_ids": [int, ...], "reference_answer": str（可选）}, ...]

设计说明：`tests/experiment/run_eval.py` 是面向 Bitext 客服数据集的定制化
全量评测脚本（含知识库导入、按类别/场景统计等业务逻辑），本模块提供的是
与具体数据集无关的通用评测能力，二者可以并存 —— 定制脚本可直接调用
本模块的 `run_e2e_eval()` 复用指标计算逻辑。
"""
import json
import time
from typing import List, Optional


def run_e2e_eval(cases: List[dict], top_k: int = 5) -> dict:
    """对一批评测用例同时运行检索评测与生成评测，返回汇总报告。

    Args:
        cases: [{"query": str, "relevant_ids": [...]（可选）, "reference_answer": str（可选）}, ...]
        top_k: 检索/生成使用的 top_k

    Returns:
        {"retrieval": {...}, "generation": {...}, "latency_ms": {...}}
    """
    from rag.evaluation import metrics
    from rag.evaluation.benchmark import percentile
    from rag.evaluation.generation_eval import evaluate_generation
    from rag.evaluation.retrieval_eval import evaluate_retrieval

    retrieval_cases = [c for c in cases if c.get("relevant_ids")]
    report = {}
    if retrieval_cases:
        report["retrieval"] = evaluate_retrieval(retrieval_cases, top_k=top_k)

    generation_report = evaluate_generation(cases, top_k=top_k)
    report["generation"] = generation_report

    # 【效率修复】此前在这里额外调用一次 `benchmark_latency()` 对同一批 query 重新
    # 跑一遍 pipeline.retrieve()，与 evaluate_retrieval()/evaluate_generation() 内部
    # 已经执行过的调用完全重复——每条评测用例在真实 Milvus/ES/vLLM 后端下会被
    # 白白多执行一次全链路调用（延迟、成本均翻倍）。现直接复用
    # `evaluate_generation()` 已经记录的每条用例耗时（`per_case[i]["latency_ms"]`，
    # 该耗时本身就覆盖了检索+生成的完整链路）计算延迟分位数，不再重复调用。
    latencies = [c["latency_ms"] for c in generation_report["per_case"]]
    report["latency_ms"] = {
        "count": len(latencies),
        "avg_ms": metrics.aggregate_mean(latencies),
        "p50_ms": percentile(latencies, 0.5),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
    }
    return report


def load_cases(eval_path: str, max_cases: Optional[int] = None) -> List[dict]:
    with open(eval_path, "r", encoding="utf-8") as f:
        cases = json.load(f)
    if max_cases:
        cases = cases[:max_cases]
    return cases


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RAG 端到端评测（检索 + 生成）")
    parser.add_argument("--eval-path", required=True, help="评测用例 JSON 文件路径")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    cases = load_cases(args.eval_path, args.max_cases)
    t0 = time.time()
    report = run_e2e_eval(cases, top_k=args.top_k)
    report["elapsed_s"] = round(time.time() - t0, 2)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
