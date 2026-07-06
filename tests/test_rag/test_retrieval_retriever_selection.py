# -*- coding: utf-8 -*-
"""rag/retrieval/retriever_selection.py 单元测试：检索器选择策略与当前后端配置查询。"""
from rag.retrieval.retriever_selection import ALL_BACKENDS, get_active_backends, select_retrievers


class TestSelectRetrievers:
    def test_normal_query_selects_all_backends(self):
        result = select_retrievers("广告限流怎么解除")
        assert set(result) == set(ALL_BACKENDS)

    def test_empty_query_returns_empty_list(self):
        assert select_retrievers("") == []
        assert select_retrievers("   ") == []
        assert select_retrievers(None) == []

    def test_query_hints_with_zero_length_returns_empty(self):
        """当传入的 query_understanding 分析结果显式标注 length=0 时应短路返回空。"""
        result = select_retrievers("广告限流", query_hints={"length": 0})
        assert result == []

    def test_query_hints_normal_length_does_not_affect_default_strategy(self):
        result = select_retrievers("广告限流", query_hints={"length": 4})
        assert set(result) == set(ALL_BACKENDS)


class TestGetActiveBackends:
    def test_returns_all_backend_keys(self):
        backends = get_active_backends()
        for key in ["vector_backend", "keyword_backend", "embed_backend", "rerank_backend"]:
            assert key in backends

    def test_values_match_rag_config(self):
        from rag.config import RAG_CONFIG
        backends = get_active_backends()
        assert backends["vector_backend"] == RAG_CONFIG["vector_backend"]
        assert backends["keyword_backend"] == RAG_CONFIG["keyword_backend"]
