# -*- coding: utf-8 -*-
"""rag/evaluation/benchmark.py 单元测试：延迟分位数与吞吐量基准测试。

组合测试：benchmark_latency 依赖 pipeline.retrieve/answer，属于 evaluation/ 与
retrieval/generation/observability 的组合场景。
"""
import pytest

from rag.evaluation.benchmark import benchmark_latency
from rag.indexing import index_builder as indexer

pytestmark = pytest.mark.usefixtures("clean_rag_data")


def _seed():
    indexer.ingest_blocks([
        {"text": "广告限流是一种常见的风控手段，触发后曝光量下降"},
        {"text": "退款需在签收后七天内申请，原路退回支付账户"},
    ], filename="seed.json")


class TestBenchmarkLatency:
    def test_retrieve_mode_returns_expected_fields(self):
        _seed()
        report = benchmark_latency(["广告限流", "退款政策"], top_k=3, mode="retrieve")
        assert report["count"] == 2
        for field in ["avg_ms", "p50_ms", "p95_ms", "p99_ms", "qps"]:
            assert field in report

    def test_answer_mode_returns_expected_fields(self):
        _seed()
        report = benchmark_latency(["广告限流"], top_k=3, mode="answer")
        assert report["count"] == 1
        assert report["avg_ms"] >= 0

    def test_empty_queries_returns_zero_count(self):
        report = benchmark_latency([], top_k=3)
        assert report["count"] == 0
        assert report["avg_ms"] == 0.0

    def test_qps_is_positive_for_nonempty_queries(self):
        _seed()
        report = benchmark_latency(["广告限流", "退款", "限流"], top_k=2)
        assert report["qps"] > 0
