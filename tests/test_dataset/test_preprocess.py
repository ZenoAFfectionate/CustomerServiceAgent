# -*- coding: utf-8 -*-
"""dataset/preprocess.py 单元测试：知识块构建去重 + 评测用例构建的边界场景。"""
import pandas as pd
import pytest

from dataset.preprocess import build_eval_cases, build_kb_blocks, run_retrieval_eval


def _df(rows):
    return pd.DataFrame(rows)


class TestBuildKbBlocksDedup:
    def test_dedup_ignores_all_placeholder_differences(self):
        """回归测试：修复审查报告 L18——此前仅剔除 {{order number}}，其他
        占位符（如 {{product id}}）移除花括号后残留文本，导致"仅占位符不同"
        的近重复 instruction 未被去重。"""
        rows = [
            {"category": "ORDER", "intent": "track_order",
             "instruction": "how do i track my {{order number}}", "response": "resp1"},
            {"category": "ORDER", "intent": "track_order",
             "instruction": "how do i track my {{product id}}", "response": "resp2"},
        ]
        blocks = build_kb_blocks(_df(rows))
        # 二者规范化后应完全一致（占位符均被移除），只保留一条
        assert len(blocks) == 1

    def test_distinct_instructions_both_kept(self):
        rows = [
            {"category": "ORDER", "intent": "track_order",
             "instruction": "how do i track my order today", "response": "resp1"},
            {"category": "ORDER", "intent": "cancel_order",
             "instruction": "how can i cancel my subscription", "response": "resp2"},
        ]
        blocks = build_kb_blocks(_df(rows))
        assert len(blocks) == 2

    def test_blocks_carry_category_and_intent_for_eval(self):
        rows = [{"category": "REFUND", "intent": "get_refund",
                 "instruction": "how do i get a refund for my order", "response": "resp"}]
        blocks = build_kb_blocks(_df(rows))
        assert blocks[0]["category"] == "REFUND"
        assert blocks[0]["intent"] == "get_refund"


class TestBuildEvalCasesNaNHandling:
    def test_nan_instruction_row_excluded_from_basic_case(self):
        """回归测试：修复审查报告 L18——某 intent 分组首行 instruction 为
        NaN 时，此前会生成 query="nan" 的评测用例污染评测集。现应过滤掉
        instruction 为空/NaN 的行，取下一条有效行作为代表。"""
        rows = [
            {"category": "ORDER", "intent": "track_order", "instruction": None, "response": "resp0"},
            {"category": "ORDER", "intent": "track_order", "instruction": "how do i track my order",
             "response": "resp1"},
        ]
        cases = build_eval_cases(_df(rows), n_per_category=1)
        basic_cases = [c for c in cases if c["type"] == "basic"]
        assert len(basic_cases) == 1
        assert basic_cases[0]["query"] != "nan"
        assert "nan" not in basic_cases[0]["query"].lower()

    def test_empty_instruction_row_excluded(self):
        rows = [
            {"category": "REFUND", "intent": "get_refund", "instruction": "   ", "response": "resp0"},
            {"category": "REFUND", "intent": "get_refund", "instruction": "how do i get my money back",
             "response": "resp1"},
        ]
        cases = build_eval_cases(_df(rows), n_per_category=1)
        basic_cases = [c for c in cases if c["type"] == "basic"]
        assert len(basic_cases) == 1
        assert "money back" in basic_cases[0]["query"]


class TestRunRetrievalEval:
    """`--eval` 模式底层依赖的 run_retrieval_eval（修复审查报告 L17）。"""

    def test_skips_multiturn_cases(self, monkeypatch):
        from rag import pipeline

        calls = []

        def _fake_retrieve(query, top_k=5):
            calls.append(query)
            return [{"category": "ORDER", "intent": "track_order"}]

        monkeypatch.setattr(pipeline, "retrieve", _fake_retrieve)

        cases = [
            {"id": "basic_1", "type": "basic", "query": "q1",
             "expected_category": "ORDER", "expected_intent": "track_order"},
            {"id": "multiturn_1", "type": "multiturn", "query": "q2",
             "expected_category": "ORDER", "expected_intent": "track_order"},
        ]
        report = run_retrieval_eval(cases, top_k=3)
        assert report["total_cases"] == 1
        assert calls == ["q1"]

    def test_hit_rate_computed_correctly(self, monkeypatch):
        from rag import pipeline

        def _fake_retrieve(query, top_k=5):
            if query == "hit_query":
                return [{"category": "ORDER", "intent": "track_order"}]
            return [{"category": "REFUND", "intent": "get_refund"}]

        monkeypatch.setattr(pipeline, "retrieve", _fake_retrieve)

        cases = [
            {"id": "basic_1", "type": "basic", "query": "hit_query",
             "expected_category": "ORDER", "expected_intent": "track_order"},
            {"id": "basic_2", "type": "basic", "query": "miss_query",
             "expected_category": "ORDER", "expected_intent": "track_order"},
        ]
        report = run_retrieval_eval(cases, top_k=3)
        assert report["total_cases"] == 2
        assert report["hit_count"] == 1
        assert report["hit_rate"] == pytest.approx(0.5)
