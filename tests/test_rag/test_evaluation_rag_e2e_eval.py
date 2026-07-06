# -*- coding: utf-8 -*-
"""rag/evaluation/rag_e2e_eval.py 单元测试：端到端评测编排（组合检索+生成+基准）。"""
import json

import pytest

from rag.evaluation.rag_e2e_eval import load_cases, run_e2e_eval
from rag.indexing import index_builder as indexer

pytestmark = pytest.mark.usefixtures("clean_rag_data")


def _seed():
    return indexer.ingest_blocks(
        [{"text": "广告限流是一种常见的风控手段，触发后曝光量下降"}], filename="seed.json",
    )


class TestRunE2EEval:
    def test_report_contains_generation_and_latency(self):
        _seed()
        report = run_e2e_eval([{"query": "广告限流"}], top_k=3)
        assert "generation" in report
        assert "latency_ms" in report

    def test_report_contains_retrieval_when_relevant_ids_provided(self):
        meta = _seed()
        cases = [{"query": "广告限流规则", "relevant_ids": [meta["chunk_ids"][0]]}]
        report = run_e2e_eval(cases, top_k=3)
        assert "retrieval" in report
        assert report["retrieval"]["num_cases"] == 1

    def test_report_skips_retrieval_when_no_relevant_ids(self):
        _seed()
        report = run_e2e_eval([{"query": "广告限流"}], top_k=3)
        assert "retrieval" not in report

    def test_multiple_cases_all_evaluated(self):
        _seed()
        cases = [{"query": "广告限流"}, {"query": "退款政策"}]
        report = run_e2e_eval(cases, top_k=3)
        assert report["generation"]["num_cases"] == 2
        assert report["latency_ms"]["count"] == 2

    def test_latency_ms_derived_from_generation_per_case_not_extra_calls(self, monkeypatch):
        """效率回归测试：`run_e2e_eval` 不应为了统计延迟而对同一批 query 额外
        重跑一遍 `pipeline.retrieve()`/`answer()`（此前的实现会导致每条用例被
        执行 2-3 次，在真实后端下延迟与成本成倍增加）。这里验证 `pipeline.answer`
        对每条用例只被调用一次。
        """
        _seed()
        from rag import pipeline

        call_count = {"n": 0}
        original_answer = pipeline.answer

        def _counting_answer(*args, **kwargs):
            call_count["n"] += 1
            return original_answer(*args, **kwargs)

        monkeypatch.setattr(pipeline, "answer", _counting_answer)
        cases = [{"query": "广告限流"}, {"query": "退款政策"}]
        report = run_e2e_eval(cases, top_k=3)

        assert call_count["n"] == len(cases)
        assert report["latency_ms"]["count"] == len(cases)


class TestLoadCases:
    def test_load_cases_from_file(self, tmp_path):
        path = tmp_path / "cases.json"
        cases = [{"query": "q1"}, {"query": "q2"}, {"query": "q3"}]
        path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")

        loaded = load_cases(str(path))
        assert loaded == cases

    def test_load_cases_respects_max_cases(self, tmp_path):
        path = tmp_path / "cases.json"
        cases = [{"query": f"q{i}"} for i in range(10)]
        path.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")

        loaded = load_cases(str(path), max_cases=3)
        assert len(loaded) == 3
