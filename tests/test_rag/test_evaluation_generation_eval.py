# -*- coding: utf-8 -*-
"""rag/evaluation/generation_eval.py 单元测试：生成质量评测（关联度/引用覆盖率/
参考答案相似度）。

组合测试：evaluate_generation 依赖 pipeline.answer（检索+生成全链路）与
generation/hallucination_control.py 的评分函数。
"""
import pytest

from rag.evaluation.generation_eval import evaluate_generation
from rag.indexing import index_builder as indexer

pytestmark = pytest.mark.usefixtures("clean_rag_data")


def _seed():
    indexer.ingest_blocks([
        {"text": "退款需在签收后七天内申请，退款将原路返回至支付账户"},
    ], filename="seed.json")


class TestEvaluateGeneration:
    def test_returns_expected_aggregate_fields(self):
        _seed()
        cases = [{"query": "怎么退款"}]
        report = evaluate_generation(cases, top_k=3)
        for field in ["num_cases", "citation_coverage", "avg_groundedness", "avg_lexical_f1", "avg_latency_ms", "per_case"]:
            assert field in report

    def test_per_case_includes_latency_ms(self):
        """`latency_ms` 供 `rag_e2e_eval.run_e2e_eval()` 直接复用计算延迟分位数，
        避免额外重跑一遍 pipeline 调用（见 test_evaluation_rag_e2e_eval.py 的
        效率回归测试）。"""
        _seed()
        cases = [{"query": "怎么退款"}]
        report = evaluate_generation(cases, top_k=3)
        assert "latency_ms" in report["per_case"][0]
        assert report["per_case"][0]["latency_ms"] >= 0
        assert report["avg_latency_ms"] >= 0

    def test_local_backend_has_citation_and_full_groundedness(self):
        """local 抽取式生成天然带引用且完全贴合上下文（摘录自身）。"""
        _seed()
        cases = [{"query": "退款政策是什么"}]
        report = evaluate_generation(cases, top_k=3)
        assert report["citation_coverage"] == 1.0

    def test_empty_kb_case_has_no_citation(self):
        cases = [{"query": "任意问题"}]
        report = evaluate_generation(cases, top_k=3)
        assert report["per_case"][0]["has_citation"] is False

    def test_reference_answer_enables_lexical_f1(self):
        _seed()
        cases = [{"query": "怎么退款", "reference_answer": "退款需要在签收后七天内申请"}]
        report = evaluate_generation(cases, top_k=3)
        assert report["avg_lexical_f1"] is not None
        assert "lexical_f1" in report["per_case"][0]

    def test_no_reference_answer_lexical_f1_is_none(self):
        _seed()
        cases = [{"query": "怎么退款"}]
        report = evaluate_generation(cases, top_k=3)
        assert report["avg_lexical_f1"] is None
        assert "lexical_f1" not in report["per_case"][0]
