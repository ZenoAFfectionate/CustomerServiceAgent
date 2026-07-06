# -*- coding: utf-8 -*-
"""性能基准测试（Benchmark）：测量 `retrieve()`/`answer()` 的延迟分位数与吞吐量。
"""
import time
from typing import List


def percentile(values: List[float], p: float) -> float:
    """计算给定分位数（p 为 0-1 之间的小数，如 0.95 表示 P95）。

    公开为模块级函数（而非 `_percentile`），供 `rag/evaluation/rag_e2e_eval.py`
    和 `rag/observability/monitoring.py` 复用同一套分位数计算逻辑，避免各
    模块各自重复实现。

    【修复 N36】此前用 ``idx = min(int(len * p), len-1)`` 的最近秩法，
    既不校验 p 越界（p>1 静默返回最大值），小样本下也偏离标准分位估计
    （如 [10,100] 的 p50 返回 100 而非 55）。现改为线性插值法（与 numpy
    默认的 ``linear`` 插值一致），并校验 0 <= p <= 1。
    """
    if not values:
        return 0.0
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"分位数 p 必须在 [0, 1] 范围内，收到 {p}")
    values = sorted(values)
    n = len(values)
    if n == 1:
        return round(values[0], 2)
    # 线性插值：rank = p * (n - 1)，在相邻两个值间线性插值
    rank = p * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    result = values[lo] * (1 - frac) + values[hi] * frac
    return round(result, 2)


def benchmark_latency(queries: List[str], top_k: int = 5, mode: str = "retrieve") -> dict:
    """对一批 query 依次调用 `retrieve()`（或 `answer()`），统计延迟分位数与 QPS。

    Args:
        queries: 待测试的 query 列表（会被依次调用，非并发压测——如需并发压测
            请在调用侧自行使用线程池包裹本函数）
        top_k: 传给 retrieve/answer 的 top_k
        mode: "retrieve" 或 "answer"

    Returns:
        {"count": int, "avg_ms": float, "p50_ms": float, "p95_ms": float,
         "p99_ms": float, "qps": float}
    """
    from rag import pipeline

    fn = pipeline.answer if mode == "answer" else pipeline.retrieve
    latencies = []
    t_start = time.time()
    for q in queries:
        t0 = time.time()
        fn(q, top_k=top_k)
        latencies.append((time.time() - t0) * 1000)
    elapsed = time.time() - t_start

    return {
        "count": len(latencies),
        "avg_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
        "p50_ms": percentile(latencies, 0.5),
        "p95_ms": percentile(latencies, 0.95),
        "p99_ms": percentile(latencies, 0.99),
        "qps": round(len(latencies) / elapsed, 2) if elapsed > 0 else 0.0,
    }
