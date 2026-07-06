# -*- coding: utf-8 -*-
"""rag/evaluation/retrieval_eval.py 单元测试：检索质量评测。

组合测试：evaluate_retrieval 依赖真实的 indexing/ 写入 + pipeline.retrieve 检索链路，
验证 evaluation/ 与 indexing/ + retrieval/ 的组合行为。
"""
import pytest

from rag.evaluation.retrieval_eval import evaluate_retrieval
from rag.indexing import index_builder as indexer

pytestmark = pytest.mark.usefixtures("clean_rag_data")


def _seed():
    meta1 = indexer.ingest_blocks(
        [{"text": "广告限流是一种常见的风控手段，触发后曝光量下降50%-80%"}], filename="a.json",
    )
    meta2 = indexer.ingest_blocks(
        [{"text": "退款需在签收后七天内申请，原路退回支付账户"}], filename="b.json",
    )
    return meta1["chunk_ids"][0], meta2["chunk_ids"][0]


class TestEvaluateRetrieval:
    def test_returns_expected_aggregate_fields(self):
        _seed()
        cases = [{"query": "广告限流", "relevant_ids": [0]}]
        report = evaluate_retrieval(cases, top_k=3)
        for field in ["num_cases", "recall@k", "precision@k", "mrr", "ndcg@k", "per_case"]:
            assert field in report

    def test_relevant_doc_found_scores_positive_recall(self):
        gid_ad, gid_refund = _seed()
        cases = [{"query": "广告为什么被限流", "relevant_ids": [gid_ad]}]
        report = evaluate_retrieval(cases, top_k=5)
        assert report["recall@k"] > 0

    def test_irrelevant_query_scores_zero_recall(self):
        _seed()
        cases = [{"query": "广告限流", "relevant_ids": [9999]}]  # 不存在的 ID
        report = evaluate_retrieval(cases, top_k=5)
        assert report["recall@k"] == 0.0

    def test_multiple_cases_aggregated(self):
        gid_ad, gid_refund = _seed()
        cases = [
            {"query": "广告限流规则", "relevant_ids": [gid_ad]},
            {"query": "退款政策说明", "relevant_ids": [gid_refund]},
        ]
        report = evaluate_retrieval(cases, top_k=5)
        assert report["num_cases"] == 2
        assert len(report["per_case"]) == 2

    def test_empty_kb_returns_zero_metrics(self):
        cases = [{"query": "任意查询", "relevant_ids": [0]}]
        report = evaluate_retrieval(cases, top_k=5)
        assert report["recall@k"] == 0.0
