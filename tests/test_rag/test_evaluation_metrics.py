# -*- coding: utf-8 -*-
"""rag/evaluation/metrics.py 单元测试：Recall@K / Precision@K / MRR / NDCG@K / 词汇 F1。"""
from rag.evaluation.metrics import (
    aggregate_mean, lexical_f1, mrr, ndcg_at_k, precision_at_k, recall_at_k,
)


class TestRecallAtK:
    def test_full_recall(self):
        assert recall_at_k([1, 2, 3], [1, 2], k=3) == 1.0

    def test_partial_recall(self):
        assert recall_at_k([1, 3, 5], [1, 2], k=3) == 0.5

    def test_zero_recall(self):
        assert recall_at_k([3, 4, 5], [1, 2], k=3) == 0.0

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k([1, 2], [], k=3) == 0.0

    def test_respects_k_truncation(self):
        # relevant=1 只在第 4 位，k=3 时不应命中
        assert recall_at_k([5, 6, 7, 1], [1], k=3) == 0.0
        assert recall_at_k([5, 6, 7, 1], [1], k=4) == 1.0


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k([1, 2], [1, 2, 3], k=2) == 1.0

    def test_half_relevant(self):
        assert precision_at_k([1, 5], [1, 2], k=2) == 0.5

    def test_empty_retrieved_returns_zero(self):
        assert precision_at_k([], [1, 2], k=5) == 0.0


class TestMRR:
    def test_first_result_relevant(self):
        assert mrr([1, 2, 3], [1]) == 1.0

    def test_second_result_relevant(self):
        assert mrr([2, 1, 3], [1]) == 0.5

    def test_no_hit_returns_zero(self):
        assert mrr([2, 3, 4], [1]) == 0.0

    def test_multiple_relevant_uses_first_hit(self):
        assert mrr([2, 1, 3], [1, 3]) == 0.5


class TestNDCGAtK:
    def test_perfect_ranking_scores_one(self):
        assert ndcg_at_k([1, 2], [1, 2], k=2) == 1.0

    def test_reversed_ranking_scores_less_than_one(self):
        score = ndcg_at_k([3, 1, 2], [1, 2], k=3)
        assert 0.0 < score < 1.0

    def test_no_relevant_returns_zero(self):
        assert ndcg_at_k([1, 2, 3], [], k=3) == 0.0

    def test_no_hit_returns_zero(self):
        assert ndcg_at_k([3, 4, 5], [1, 2], k=3) == 0.0


class TestLexicalF1:
    def test_identical_text_scores_one(self):
        assert lexical_f1("退款需要七天", "退款需要七天") == 1.0

    def test_completely_different_text_scores_low(self):
        score = lexical_f1("退款需要七天", "今天天气很好呀")
        assert score < 0.5

    def test_empty_text_returns_zero(self):
        assert lexical_f1("", "参考答案") == 0.0
        assert lexical_f1("预测答案", "") == 0.0

    def test_partial_overlap_scores_between_zero_and_one(self):
        score = lexical_f1("广告限流规则说明", "广告限流的解除方法")
        assert 0.0 < score < 1.0


class TestAggregateMean:
    def test_normal_list(self):
        assert aggregate_mean([1.0, 2.0, 3.0]) == 2.0

    def test_empty_list_returns_zero(self):
        assert aggregate_mean([]) == 0.0
