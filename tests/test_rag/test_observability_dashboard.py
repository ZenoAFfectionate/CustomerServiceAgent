# -*- coding: utf-8 -*-
"""rag/observability/dashboard.py 单元测试：运维看板数据聚合。

组合测试：dashboard() 组合了 monitoring（指标快照）+ alerting（告警检查）+
retriever_selection（后端配置）+ index_builder（知识库统计）四个来源。
"""
import pytest

from rag.observability import monitoring
from rag.observability.dashboard import dashboard
from rag.indexing import index_builder as indexer

pytestmark = pytest.mark.usefixtures("clean_rag_data")


class TestDashboard:
    def test_returns_all_top_level_sections(self):
        result = dashboard()
        for key in ["metrics", "alerts", "backends", "corpus_stats"]:
            assert key in result

    def test_corpus_stats_reflects_ingested_documents(self):
        indexer.ingest_blocks([{"text": "内容一"}, {"text": "内容二"}], filename="a.json")
        result = dashboard()
        assert result["corpus_stats"]["num_documents"] == 1
        assert result["corpus_stats"]["num_vector_chunks"] == 2

    def test_backends_reflects_current_config(self):
        from rag.config import RAG_CONFIG
        result = dashboard()
        assert result["backends"]["vector_backend"] == RAG_CONFIG["vector_backend"]

    def test_metrics_reflects_monitoring_snapshot(self):
        monitoring.record_answer(latency_ms=10.0, backend_used="local")
        result = dashboard()
        assert result["metrics"]["answer"]["count"] >= 1

    def test_alerts_is_list_type(self):
        result = dashboard()
        assert isinstance(result["alerts"], list)

    def test_alerts_triggered_when_thresholds_exceeded(self):
        """组合回归测试：曾发现的 bug —— alerting.log_event 关键字参数冲突会导致
        dashboard() 在存在告警时直接抛异常，这里验证高 fallback_rate 场景下
        dashboard() 仍能正常返回而不报错。"""
        for _ in range(5):
            monitoring.record_answer(latency_ms=10.0, backend_used="local")
        result = dashboard()
        assert any(a["metric"] == "fallback_rate" for a in result["alerts"])
