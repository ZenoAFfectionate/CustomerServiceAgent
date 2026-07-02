# -*- coding: utf-8 -*-
"""rag/pipeline.py 集成测试：mock 检索/精排/生成组件，验证编排逻辑与降级策略。"""
import pytest

from rag import pipeline
from rag.indexing import indexer

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

        monkeypatch.setattr(pipeline.milvus_search, "search", _raise)
        results = pipeline.retrieve("退款政策是什么", top_k=3)
        assert isinstance(results, list)  # 不抛异常，优雅降级

    def test_both_retrievers_fail_returns_empty_not_crash(self, monkeypatch):
        _seed_knowledge_base()
        monkeypatch.setattr(pipeline.milvus_search, "search", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("x")))
        monkeypatch.setattr(pipeline.es_search, "search", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("x")))
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
