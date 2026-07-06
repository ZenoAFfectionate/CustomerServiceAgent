# -*- coding: utf-8 -*-
"""rag/observability/monitoring.py 单元测试：进程内滚动指标采集。"""
import pytest

from rag.observability import monitoring


@pytest.fixture(autouse=True)
def _reset_monitoring():
    monitoring.reset()
    yield
    monitoring.reset()


class TestRecordRetrieval:
    def test_record_increases_count(self):
        monitoring.record_retrieval({"total_ms": 10.0, "spans": {}}, num_results=3)
        assert monitoring.snapshot()["retrieval"]["count"] == 1

    def test_latency_percentiles_computed(self):
        for ms in [10.0, 20.0, 30.0, 40.0, 50.0]:
            monitoring.record_retrieval({"total_ms": ms, "spans": {}}, num_results=1)
        snap = monitoring.snapshot()["retrieval"]
        assert snap["avg_ms"] == 30.0
        assert snap["p50_ms"] > 0


class TestRecordAnswer:
    def test_record_increases_count_and_backend_usage(self):
        monitoring.record_answer(latency_ms=50.0, backend_used="local")
        monitoring.record_answer(latency_ms=60.0, backend_used="vllm")
        snap = monitoring.snapshot()["answer"]
        assert snap["count"] == 2
        assert snap["backend_usage"] == {"local": 1, "vllm": 1}

    def test_fallback_rate_computed_from_local_backend_ratio(self):
        monitoring.record_answer(latency_ms=10.0, backend_used="local")
        monitoring.record_answer(latency_ms=10.0, backend_used="local")
        monitoring.record_answer(latency_ms=10.0, backend_used="vllm")
        snap = monitoring.snapshot()["answer"]
        assert snap["fallback_rate"] == pytest.approx(2 / 3, rel=1e-3)


class TestRecordError:
    def test_error_count_increments(self):
        monitoring.record_retrieval({"total_ms": 5.0, "spans": {}}, num_results=0)
        monitoring.record_error("retrieval", "模拟错误")
        snap = monitoring.snapshot()
        assert snap["error_count"] == 1
        assert snap["error_rate"] == 1.0  # 1 error / 1 request


class TestSnapshotEmptyState:
    def test_empty_snapshot_has_zero_counts(self):
        snap = monitoring.snapshot()
        assert snap["request_count"] == 0
        assert snap["retrieval"]["count"] == 0
        assert snap["answer"]["count"] == 0
        assert snap["error_rate"] == 0.0


class TestReset:
    def test_reset_clears_all_by_default(self):
        monitoring.record_retrieval({"total_ms": 1.0, "spans": {}}, num_results=1)
        monitoring.record_answer(latency_ms=1.0, backend_used="local")
        monitoring.reset()
        snap = monitoring.snapshot()
        assert snap["retrieval"]["count"] == 0
        assert snap["answer"]["count"] == 0

    def test_reset_scope_retrieval_only(self):
        monitoring.record_retrieval({"total_ms": 1.0, "spans": {}}, num_results=1)
        monitoring.record_answer(latency_ms=1.0, backend_used="local")
        monitoring.reset(scope="retrieval")
        snap = monitoring.snapshot()
        assert snap["retrieval"]["count"] == 0
        assert snap["answer"]["count"] == 1
