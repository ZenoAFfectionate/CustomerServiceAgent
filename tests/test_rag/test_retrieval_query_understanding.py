# -*- coding: utf-8 -*-
"""rag/retrieval/query_understanding.py 单元测试：语言检测、疑问句判断、关键词
提取、复杂度估计（纯规则 + jieba，无外部依赖）。"""
from rag.retrieval.query_understanding import (
    analyze_query, detect_language, estimate_complexity, extract_keywords, is_question,
)


class TestDetectLanguage:
    def test_pure_chinese(self):
        assert detect_language("广告限流规则") == "zh"

    def test_pure_english(self):
        assert detect_language("what is the refund policy") == "en"

    def test_mixed_language(self):
        assert detect_language("如何使用 API 接口") == "mixed"

    def test_no_language_signal(self):
        assert detect_language("123456!!!") == "unknown"

    def test_empty_string(self):
        assert detect_language("") == "unknown"


class TestIsQuestion:
    def test_question_mark_suffix(self):
        assert is_question("这是什么？") is True
        assert is_question("what is this?") is True

    def test_question_word_without_mark(self):
        assert is_question("怎么申请退款") is True
        assert is_question("为什么被限流了") is True

    def test_statement_sentence_not_question(self):
        assert is_question("退款已经完成") is False

    def test_empty_string_not_question(self):
        assert is_question("") is False
        assert is_question(None) is False


class TestExtractKeywords:
    def test_returns_nonempty_list_for_normal_query(self):
        keywords = extract_keywords("广告投放异常触发限流规则")
        assert isinstance(keywords, list)
        assert len(keywords) >= 1

    def test_empty_query_returns_empty_list(self):
        assert extract_keywords("") == []
        assert extract_keywords("   ") == []

    def test_respects_top_n(self):
        keywords = extract_keywords("广告投放异常触发限流规则申请解除人工复核", top_n=3)
        assert len(keywords) <= 3

    def test_no_duplicate_keywords(self):
        keywords = extract_keywords("退款退款退款政策政策")
        assert len(keywords) == len(set(keywords))


class TestEstimateComplexity:
    def test_short_simple_query(self):
        assert estimate_complexity("怎么退款") == "simple"

    def test_query_with_conjunction_is_complex(self):
        assert estimate_complexity("广告被限流了，并且订单也无法正常发货") == "complex"

    def test_long_query_is_complex(self):
        assert estimate_complexity("我" * 41) == "complex"

    def test_empty_query_is_simple(self):
        assert estimate_complexity("") == "simple"


class TestAnalyzeQuery:
    def test_returns_all_expected_fields(self):
        result = analyze_query("广告为什么被限流了？")
        for field in ["query", "length", "language", "is_question", "keywords", "complexity"]:
            assert field in result

    def test_length_matches_query(self):
        result = analyze_query("测试查询文本")
        assert result["length"] == len("测试查询文本")

    def test_empty_query_handled_gracefully(self):
        result = analyze_query("")
        assert result["length"] == 0
        assert result["is_question"] is False
        assert result["keywords"] == []

    def test_none_query_handled_gracefully(self):
        result = analyze_query(None)
        assert result["query"] == ""
        assert result["length"] == 0
