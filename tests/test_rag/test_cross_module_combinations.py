# -*- coding: utf-8 -*-
"""跨模块组合测试：验证 retrieval/indexing/knowledge_base/generation/evaluation/
observability/integration 之间以"非默认编排方式"组合时依然能协同工作。

`test_pipeline.py`（retrieval+generation+observability）、`test_api.py`
（全链路 HTTP）、`test_knowledge_base_*.py`（knowledge_base+indexing）、
`test_evaluation_*.py`（evaluation+pipeline）、`test_observability_dashboard.py`
（observability+indexing+retriever_selection）、`test_integration_*.py`
（integration+pipeline+api）已分别覆盖了各自的组合场景；本文件专注于覆盖
"手动拼装检索/生成链路"以及"多个模块首尾串联的完整业务闭环"这两类尚未被
覆盖的组合方式，避免测试覆盖面出现遗漏。
"""
import pytest

from rag.indexing import index_builder as indexer

pytestmark = pytest.mark.usefixtures("clean_rag_data")


def _seed_kb():
    return indexer.ingest_blocks([
        {"text": "当账户存在异常投放行为时，系统将自动限流，限流期间广告曝光量将下降50%-80%。",
         "title": "广告限流触发条件", "page_url": "http://a"},
        {"text": "商品签收后七天内可申请无理由退款，退款将原路返回至支付账户。",
         "title": "退款政策", "page_url": "http://b"},
    ], filename="seed.json")


class TestManuallyComposedRetrievalChain:
    """手动串联 query_understanding → query_rewrite → retriever_selection →
    hybrid_search → rerank，验证各模块可脱离 pipeline.py 的固定编排独立组合使用
    （例如需要自定义编排逻辑的场景）。"""

    def test_full_chain_produces_reranked_results(self):
        from rag.retrieval import hybrid_search, query_understanding, query_rewrite, retriever_selection
        from rag.retrieval.rerank import rerank

        _seed_kb()
        raw_query = "广告为什么被限流了"

        hints = query_understanding.analyze_query(raw_query)
        assert hints["language"] == "zh"

        rewritten = query_rewrite.rewrite_query(raw_query, dialogue=None)
        assert rewritten == raw_query  # 无历史对话，应原样返回

        backends = retriever_selection.select_retrievers(rewritten, query_hints=hints)
        assert set(backends) == {"vector", "keyword"}

        fused = hybrid_search.search(rewritten, top_k=5, backends=backends)
        assert len(fused) >= 1

        reranked = rerank(rewritten, fused, top_k=2)
        assert len(reranked) <= 2
        assert reranked[0].source_retriever == "reranked"

    def test_chain_with_empty_query_short_circuits(self):
        from rag.retrieval import hybrid_search, retriever_selection

        _seed_kb()
        backends = retriever_selection.select_retrievers("")
        assert backends == []
        # 空 backends 列表应导致 search 直接返回空，不发起任何检索
        assert hybrid_search.search("广告限流", top_k=5, backends=backends) == []


class TestKnowledgeBaseToRetrievalLifecycle:
    """knowledge_base（增量同步 + 版本追踪）→ retrieval（pipeline 检索）的完整闭环。"""

    def test_directory_sync_then_retrieve_then_history(self, tmp_path):
        import json

        from rag import pipeline
        from rag.knowledge_base import corpus_management, update_sync

        (tmp_path / "kb.json").write_text(
            json.dumps([{"text": "广告限流是一种常见的风控手段，触发后曝光量下降"}], ensure_ascii=False),
            encoding="utf-8",
        )
        sync_result = update_sync.sync_directory(str(tmp_path))
        assert sync_result["ingested"] == 1

        results = pipeline.retrieve("广告限流", top_k=3)
        assert len(results) >= 1

        doc_id = sync_result["files"][0]["doc_id"]
        history = corpus_management.get_document_history(doc_id)
        assert len(history) == 1

    def test_content_update_reflected_in_next_retrieve(self, tmp_path):
        import json

        from rag import pipeline
        from rag.knowledge_base import update_sync

        file_path = tmp_path / "kb.json"
        file_path.write_text(json.dumps([{"text": "初始版本内容关于退款流程说明"}], ensure_ascii=False), encoding="utf-8")
        update_sync.sync_directory(str(tmp_path))

        file_path.write_text(json.dumps([{"text": "更新版本内容关于物流查询说明"}], ensure_ascii=False), encoding="utf-8")
        update_sync.sync_directory(str(tmp_path))

        results = pipeline.retrieve("物流查询", top_k=3)
        assert any("物流" in r["text"] for r in results)


class TestEvaluationOverKnowledgeBase:
    """knowledge_base 导入的知识库 → evaluation 端到端评测的组合场景。"""

    def test_e2e_eval_over_freshly_ingested_kb(self):
        from rag.evaluation.rag_e2e_eval import run_e2e_eval
        from rag.knowledge_base import corpus_management

        meta = corpus_management.ingest_blocks(
            [{"text": "广告限流是一种常见的风控手段，触发后曝光量下降"}], filename="kb.json",
        )
        cases = [{"query": "广告限流规则", "relevant_ids": [meta["chunk_ids"][0]]}]
        report = run_e2e_eval(cases, top_k=3)
        assert report["retrieval"]["recall@k"] > 0
        assert report["generation"]["num_cases"] == 1


class TestAgentToolConsistentWithDirectPipeline:
    """integration.agent_integration 的工具调用结果应与直接调用 pipeline 完全一致
    （验证适配层只是"薄封装"，未引入行为差异）。"""

    def test_tool_answer_matches_direct_pipeline_answer(self):
        from rag import pipeline
        from rag.integration.tool_usage import dispatch_tool_call

        _seed_kb()
        direct = pipeline.answer("怎么退款", top_k=3)
        via_tool = dispatch_tool_call("rag_answer", {"query": "怎么退款", "top_k": 3})

        assert via_tool["ok"] is True
        assert via_tool["answer"] == direct["answer"]
        assert via_tool["citations"] == direct["citations"]

    def test_tool_retrieve_matches_direct_pipeline_retrieve(self):
        from rag import pipeline
        from rag.integration.tool_usage import dispatch_tool_call

        _seed_kb()
        direct = pipeline.retrieve("广告限流", top_k=3)
        via_tool = dispatch_tool_call("rag_retrieve", {"query": "广告限流", "top_k": 3})

        assert via_tool["ok"] is True
        assert via_tool["results"] == direct


class TestFullLifecycleObservability:
    """完整业务闭环：knowledge_base 导入 → pipeline 多次检索/问答 → observability
    监控快照与告警 → integration.deployment 就绪检查，均应反映一致的系统状态。"""

    def test_full_lifecycle_reflected_consistently_across_modules(self):
        from rag import pipeline
        from rag.integration.deployment import check_deployment_readiness
        from rag.knowledge_base import corpus_management
        from rag.observability import monitoring
        from rag.observability.dashboard import dashboard

        corpus_management.ingest_blocks(
            [{"text": "广告限流是一种常见的风控手段"}, {"text": "退款需在签收后七天内申请"}],
            filename="lifecycle.json",
        )

        for q in ["广告限流", "退款政策", "任意问题"]:
            pipeline.answer(q, top_k=3)

        snap = monitoring.snapshot()
        assert snap["answer"]["count"] == 3

        board = dashboard()
        assert board["corpus_stats"]["num_documents"] == 1
        assert board["metrics"]["answer"]["count"] == 3

        readiness = check_deployment_readiness()
        assert readiness["ready"] is True
