# -*- coding: utf-8 -*-
"""rag/generation/hallucination_control.py 单元测试：引用有效性校验、关联度评分、
事后护栏。"""
from rag.generation.hallucination_control import (
    apply_guardrail, check_citation_validity, extract_cited_indices, groundedness_score,
)
from rag.schema import DocBlock


def _ctx(idx, text, title=""):
    return DocBlock(global_chunk_idx=idx, text=text, title=title)


class TestExtractCitedIndices:
    def test_extracts_all_citation_markers(self):
        assert extract_cited_indices("回答内容 [1] 补充说明 [2]") == [1, 2]

    def test_deduplicates_repeated_markers(self):
        assert extract_cited_indices("[1] 内容 [1] 内容 [3]") == [1, 3]

    def test_no_markers_returns_empty_list(self):
        assert extract_cited_indices("没有任何引用标记的回答") == []

    def test_empty_text_returns_empty_list(self):
        assert extract_cited_indices("") == []
        assert extract_cited_indices(None) == []


class TestCheckCitationValidity:
    def test_valid_citation_within_range(self):
        result = check_citation_validity("回答内容 [1] [2]", num_contexts=2)
        assert result["valid"] is True
        assert result["has_citation"] is True
        assert result["invalid_indices"] == []

    def test_invalid_citation_out_of_range(self):
        result = check_citation_validity("回答内容 [5]", num_contexts=2)
        assert result["valid"] is False
        assert result["invalid_indices"] == [5]

    def test_no_citation_detected(self):
        result = check_citation_validity("没有引用的回答", num_contexts=2)
        assert result["has_citation"] is False
        assert result["valid"] is True  # 没有引用不算"无效"，只是"缺失"

    def test_citation_index_zero_is_invalid(self):
        result = check_citation_validity("回答 [0]", num_contexts=2)
        assert result["invalid_indices"] == [0]


class TestGroundednessScore:
    def test_answer_overlapping_with_context_scores_higher(self):
        contexts = [_ctx(1, "广告限流是常见的风控手段，触发后曝光量下降")]
        grounded_answer = "广告限流触发后曝光量下降"
        unrelated_answer = "今天天气晴朗适合出门散步"
        assert groundedness_score(grounded_answer, contexts) > groundedness_score(unrelated_answer, contexts)

    def test_empty_answer_returns_zero(self):
        contexts = [_ctx(1, "任意内容")]
        assert groundedness_score("", contexts) == 0.0

    def test_empty_contexts_returns_zero(self):
        assert groundedness_score("任意回答内容", []) == 0.0

    def test_score_within_valid_range(self):
        contexts = [_ctx(1, "广告限流规则说明")]
        score = groundedness_score("广告限流规则说明的回答内容", contexts)
        assert 0.0 <= score <= 1.0


class TestApplyGuardrail:
    def test_local_backend_not_modified(self):
        """local 抽取式生成天然 grounded，护栏不应介入（仅对 vllm 后端生效）。"""
        contexts = [_ctx(1, "内容")]
        answer = "抽取式回答内容 [1]"
        assert apply_guardrail(answer, contexts, backend_used="local") == answer

    def test_no_context_not_modified(self):
        answer = "抽取式回答"
        assert apply_guardrail(answer, [], backend_used="vllm") == answer

    def test_vllm_low_groundedness_logs_but_does_not_modify_by_default(self):
        """RAG_CONFIG['hallucination_append_caveat'] 默认 False，护栏应只记录日志，
        不修改回答文本（避免默认改变既有生成结果）。"""
        contexts = [_ctx(1, "广告限流规则说明与解除方式")]
        unrelated_answer = "今天天气晴朗适合出门散步 [1]"
        result = apply_guardrail(unrelated_answer, contexts, backend_used="vllm")
        assert result == unrelated_answer

    def test_vllm_low_groundedness_appends_caveat_when_enabled(self, monkeypatch):
        """组合测试：开启 RAG_CONFIG['hallucination_append_caveat'] 后，低关联度回答
        应被追加提示语（验证配置项真正生效）。"""
        from rag.config import RAG_CONFIG
        monkeypatch.setitem(RAG_CONFIG, "hallucination_append_caveat", True)

        contexts = [_ctx(1, "广告限流规则说明与解除方式")]
        unrelated_answer = "今天天气晴朗适合出门散步 [1]"
        result = apply_guardrail(unrelated_answer, contexts, backend_used="vllm")
        assert result != unrelated_answer
        assert "提示" in result

    def test_high_groundedness_answer_unmodified(self, monkeypatch):
        from rag.config import RAG_CONFIG
        monkeypatch.setitem(RAG_CONFIG, "hallucination_append_caveat", True)

        contexts = [_ctx(1, "广告限流规则说明与解除方式，通常持续24小时")]
        grounded_answer = "广告限流规则说明与解除方式 [1]"
        result = apply_guardrail(grounded_answer, contexts, backend_used="vllm")
        assert result == grounded_answer
