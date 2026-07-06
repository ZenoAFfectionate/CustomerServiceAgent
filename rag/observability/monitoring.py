# -*- coding: utf-8 -*-
"""进程内滚动指标采集（Monitoring）：请求量、延迟分位数、后端使用分布、错误率。

无需外部时序数据库，用 `collections.deque` 维护一个有界的滚动窗口（默认最近
1000 条），保证内存占用恒定；`snapshot()` 返回聚合后的统计快照，供
`alerting.py`（阈值告警）与 `dashboard.py`（FastAPI 展示）使用。

线程安全性：仅用一把全局 `threading.Lock` 保护写入，读取（`snapshot`）为
只读拷贝，不阻塞写入；与 `rag/indexing/*_store.py` 的单例模式一致，单进程内
安全，多进程部署下每个 worker 各自维护独立快照（符合"进程内观测"的定位）。
"""
import threading
import time
from collections import deque
from typing import Deque, Dict, Optional

# 【修复 L7】此前本文件自行实现了一份与 `rag/evaluation/benchmark.py` 的
# `percentile` 完全一致的分位数函数（`_percentile`），两处重复实现，未来
# 修正分位算法（如改用插值法）需同时改两处、容易漏改一处。改为直接复用
# `benchmark.percentile` 这一处实现。
from rag.evaluation.benchmark import percentile

_MAX_SAMPLES = 1000
_lock = threading.Lock()

_retrieval_samples: Deque[dict] = deque(maxlen=_MAX_SAMPLES)
_answer_samples: Deque[dict] = deque(maxlen=_MAX_SAMPLES)
_error_samples: Deque[dict] = deque(maxlen=_MAX_SAMPLES)
_error_count = 0
_request_count = 0


def record_retrieval(trace: dict, num_results: int) -> None:
    """记录一次检索调用的耗时明细（供 `pipeline._retrieve_blocks` 调用）。"""
    global _request_count
    with _lock:
        _request_count += 1
        _retrieval_samples.append({
            "ts": time.time(),
            "total_ms": trace.get("total_ms", 0.0),
            "spans": trace.get("spans", {}),
            "num_results": num_results,
        })


def record_answer(latency_ms: float, backend_used: str, generation_ok: bool = True) -> None:
    """记录一次端到端问答调用的耗时与所用生成后端（供 `pipeline.answer` 调用）。

    Args:
        latency_ms: 端到端耗时（毫秒）
        backend_used: 实际使用的生成后端（"vllm"/"local"/"no_context"），
            或生成阶段发生**意外**异常时的占位值（见 `pipeline.answer`）。
        generation_ok: 生成阶段是否正常完成（未发生意外异常）。
            【修复 L15】此前"生成阶段发生意外异常"（`backend_used="error"`）
            与"检索成功但确实无生成后端可用"等正常业务分支被同等地计入
            `backend_usage` 直方图，二者语义完全不同却被混在同一维度里，
            使按 `backend_used` 聚合的告警规则可能掩盖真实的生成失败率
            （既无法从 `backend_usage` 里干净地读出"生成异常"占比，也会让
            "error" 这个人为占位字符串污染本应只包含真实后端名的分布）。
            现通过独立的 `generation_ok` 标记区分，`snapshot()` 中的
            `backend_usage` 分布只统计 `generation_ok=True` 的正常样本，
            异常样本单独计入 `generation_error_count`/`generation_error_rate`。
    """
    with _lock:
        _answer_samples.append({
            "ts": time.time(), "latency_ms": latency_ms,
            "backend_used": backend_used, "generation_ok": generation_ok,
        })


def record_error(stage: str, message: str = "") -> None:
    """记录一次异常，供 `pipeline.py` 各阶段（vector/keyword/fusion/rerank/
    generation）的 except 分支调用。除全局错误计数外，同时保留按阶段的明细
    （截断到最近 `_MAX_SAMPLES` 条），供 `dashboard.py` 展示"哪个阶段在出错"，
    而不只是一个无法定位问题的笼统计数。
    """
    global _error_count
    with _lock:
        _error_count += 1
        _error_samples.append({"ts": time.time(), "stage": stage, "message": message})


def snapshot() -> dict:
    """返回当前监控快照（聚合统计，只读）。"""
    with _lock:
        retrieval = list(_retrieval_samples)
        answers = list(_answer_samples)
        errors = list(_error_samples)
        error_count = _error_count
        request_count = _request_count

    retrieval_latencies = [r["total_ms"] for r in retrieval]
    answer_latencies = [a["latency_ms"] for a in answers]
    # 【修复 L15】backend_usage 只统计生成阶段正常完成的样本（generation_ok
    # 默认为 True；旧样本/未显式传入时视为正常，向后兼容），避免"生成阶段
    # 意外异常"污染真实的后端使用分布。
    normal_answers = [a for a in answers if a.get("generation_ok", True)]
    generation_error_count = len(answers) - len(normal_answers)
    backend_hist: Dict[str, int] = {}
    for a in normal_answers:
        backend_hist[a["backend_used"]] = backend_hist.get(a["backend_used"], 0) + 1
    errors_by_stage: Dict[str, int] = {}
    for e in errors:
        errors_by_stage[e["stage"]] = errors_by_stage.get(e["stage"], 0) + 1

    total_answers = len(answers)
    return {
        "request_count": request_count,
        "error_count": error_count,
        "error_rate": round(error_count / request_count, 4) if request_count else 0.0,
        "errors_by_stage": errors_by_stage,
        "retrieval": {
            "count": len(retrieval),
            "avg_ms": round(sum(retrieval_latencies) / len(retrieval_latencies), 2) if retrieval_latencies else 0.0,
            "p50_ms": percentile(retrieval_latencies, 0.5),
            "p95_ms": percentile(retrieval_latencies, 0.95),
            "p99_ms": percentile(retrieval_latencies, 0.99),
        },
        "answer": {
            "count": total_answers,
            "avg_ms": round(sum(answer_latencies) / len(answer_latencies), 2) if answer_latencies else 0.0,
            "p50_ms": percentile(answer_latencies, 0.5),
            "p95_ms": percentile(answer_latencies, 0.95),
            "backend_usage": backend_hist,
            # 【修复 N30】所有 answer 级比率统一以 total_answers 为分母，
            # 使快照内百分比可横向比较。fallback_rate 此前以
            # len(normal_answers) 为分母，与 generation_error_rate 的
            # len(answers) 分母不一致。
            "fallback_rate": round(backend_hist.get("local", 0) / total_answers, 4) if total_answers else 0.0,
            "generation_error_count": generation_error_count,
            "generation_error_rate": round(generation_error_count / total_answers, 4) if total_answers else 0.0,
        },
    }


def reset(scope: Optional[str] = None) -> None:
    """重置监控数据（供测试隔离使用）。scope 为 None 时全部重置。"""
    global _error_count, _request_count
    with _lock:
        if scope in (None, "retrieval"):
            _retrieval_samples.clear()
        if scope in (None, "answer"):
            _answer_samples.clear()
        if scope in (None, "error"):
            _error_samples.clear()
        if scope is None:
            _error_count = 0
            _request_count = 0
