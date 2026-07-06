# -*- coding: utf-8 -*-
"""rag/pipeline.py 集成测试：mock 检索/精排/生成组件，验证编排逻辑与降级策略。

`pipeline.py` 本身就是 retrieval/ + generation/ + observability/ 三个子模块的组合层，
因此本文件同时承担"跨模块组合测试"的职责：检索链路（含 query_rewrite）→ 生成 →
监控数据落盘的完整闭环。
"""
import pytest

from rag import pipeline
from rag.indexing import index_builder as indexer
from rag.observability import monitoring

pytestmark = pytest.mark.usefixtures("clean_rag_data")


def _seed_knowledge_base():
    indexer.ingest_blocks([
        {
            "text": "当账户存在异常投放行为时，系统将自动限流，限流期间广告曝光量将下降 50%-80%。",
            "title": "广告限流触发条件",
            "page_url": "http://help.example.com/ad-limit-trigger",
            "block_path": "html>body>div0>p",
        },
        {
            "text": "限流通常持续 24-72 小时，可通过优化创意质量、降低出价频次申请人工复核解除。",
            "title": "广告限流解除方式",
            "page_url": "http://help.example.com/ad-limit-release",
            "block_path": "html>body>div1>p",
        },
        {
            "text": "商品签收后七天内可申请无理由退款，退款将原路返回至支付账户。",
            "title": "退款政策",
            "page_url": "http://help.example.com/refund",
            "block_path": "html>body>div2>p",
        },
    ], filename="seed.json")


class TestRetrieve:
    def test_retrieve_returns_relevant_blocks(self):
        _seed_knowledge_base()
        results = pipeline.retrieve("广告限流之后要怎么解除", top_k=3)
        assert len(results) >= 1
        assert isinstance(results[0], dict)
        assert "page_url" in results[0] and "block_path" in results[0] and "score" in results[0]

    def test_retrieve_empty_kb_returns_empty(self):
        results = pipeline.retrieve("任意查询", top_k=5)
        assert results == []

    def test_retrieve_respects_top_k(self):
        _seed_knowledge_base()
        results = pipeline.retrieve("规则", top_k=1)
        assert len(results) <= 1

    def test_retrieve_result_fields_complete(self):
        _seed_knowledge_base()
        results = pipeline.retrieve("退款", top_k=3)
        assert len(results) >= 1
        for field in ["text", "html_content", "page_url", "block_path", "score"]:
            assert field in results[0]


class TestRetrieveDegradation:
    def test_milvus_failure_still_returns_es_results(self, monkeypatch):
        """向量检索异常时，应仍基于关键词检索结果返回，而不是整体失败。"""
        _seed_knowledge_base()

        def _raise(*a, **kw):
            raise RuntimeError("模拟向量检索服务异常")

        monkeypatch.setattr(pipeline.hybrid_search, "vector_search", _raise)
        results = pipeline.retrieve("退款政策是什么", top_k=3)
        assert isinstance(results, list)  # 不抛异常，优雅降级

    def test_both_retrievers_fail_returns_empty_not_crash(self, monkeypatch):
        _seed_knowledge_base()
        monkeypatch.setattr(pipeline.hybrid_search, "vector_search", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("x")))
        monkeypatch.setattr(pipeline.hybrid_search, "keyword_search", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("x")))
        results = pipeline.retrieve("任意问题", top_k=3)
        assert results == []


class TestAnswer:
    def test_answer_end_to_end(self):
        _seed_knowledge_base()
        result = pipeline.answer("广告为什么被限流了")
        assert result["query"] == "广告为什么被限流了"
        assert result["answer"]
        assert result["backend_used"] in ("local", "vllm", "no_context")
        assert isinstance(result["contexts"], list)

    def test_answer_on_empty_kb_returns_no_context(self):
        result = pipeline.answer("任意问题")
        assert result["backend_used"] == "no_context"
        assert result["contexts"] == []

    def test_answer_includes_citations_matching_contexts(self):
        _seed_knowledge_base()
        result = pipeline.answer("如何解除广告限流", top_k=2)
        assert len(result["citations"]) == len(result["contexts"])


class TestQueryRewriteIntegration:
    """组合测试：pipeline 检索链路第一步会调用 retrieval/query_rewrite.py，
    验证多轮对话场景下重写结果被正确用于后续检索与返回字段。"""

    def test_rewrite_failure_falls_back_to_original_query(self, monkeypatch):
        """query_rewrite 内部依赖 process/ 的 LLM 重写服务，不可用时应静默回退为原始
        query，不影响检索继续执行（见 retrieval/query_rewrite.py 的降级设计）。"""
        _seed_knowledge_base()
        dialogue = [{"speaker": "user", "text": "广告为什么被限流了"}, {"speaker": "bot", "text": "触发了风控规则"}]
        results = pipeline.retrieve("那要多久才能解除", dialogue=dialogue, top_k=3)
        assert isinstance(results, list)

    def test_rewritten_query_field_set_when_rewrite_changes_query(self, monkeypatch):
        """当 query_rewrite 实际改写了 query 时，answer() 返回的 rewritten_query 应
        反映改写后的文本；未改写（含重写失败回退）时应为 None。"""
        _seed_knowledge_base()
        monkeypatch.setattr(pipeline, "rewrite_query", lambda query, dialogue: "改写后的问题：广告限流解除方式")
        result = pipeline.answer("那怎么办", dialogue=[{"speaker": "user", "text": "占位"}], top_k=2)
        assert result["rewritten_query"] == "改写后的问题：广告限流解除方式"

    def test_no_dialogue_keeps_query_unchanged(self):
        _seed_knowledge_base()
        result = pipeline.answer("广告限流触发条件", top_k=2)
        assert result["rewritten_query"] is None


class TestObservabilityIntegration:
    """组合测试：pipeline 每次 retrieve/answer 均应通过 observability/monitoring.py
    落盘一条监控样本，供 dashboard/alerting 使用。"""

    def test_retrieve_records_monitoring_sample(self):
        _seed_knowledge_base()
        before = monitoring.snapshot()["retrieval"]["count"]
        pipeline.retrieve("广告限流", top_k=2)
        after = monitoring.snapshot()["retrieval"]["count"]
        assert after == before + 1

    def test_answer_records_monitoring_sample_with_backend(self):
        _seed_knowledge_base()
        pipeline.answer("退款政策", top_k=2)
        snap = monitoring.snapshot()
        assert snap["answer"]["count"] >= 1
        assert "local" in snap["answer"]["backend_usage"] or "no_context" in snap["answer"]["backend_usage"]

    def test_generation_exception_does_not_pollute_backend_usage(self, monkeypatch):
        """回归测试：修复审查报告 L15——生成阶段意外异常此前会把 `backend_used
        ="error"` 计入 `backend_usage` 直方图，与真实后端使用分布混淆。现应
        单独计入 `generation_error_count`/`generation_error_rate`，不出现在
        `backend_usage` 里。"""
        import rag.pipeline as pipeline_mod

        _seed_knowledge_base()

        def _boom(*a, **kw):
            raise RuntimeError("模拟生成阶段未预期异常")

        monkeypatch.setattr(pipeline_mod, "generate_answer", _boom)

        before = monitoring.snapshot()["answer"]
        result = pipeline.answer("退款政策", top_k=2)
        after = monitoring.snapshot()["answer"]

        assert result["backend_used"] == "error"
        assert "error" not in after["backend_usage"]
        assert after["generation_error_count"] == before.get("generation_error_count", 0) + 1
        assert after["generation_error_rate"] > 0
