# -*- coding: utf-8 -*-
"""rag/generation/generator.py 单元测试：抽取式兜底生成 + vLLM 降级逻辑。"""
from rag.generation.generator import _build_context_text, build_citations, generate_answer
from rag.schema import DocBlock


def _ctx(idx, title, text, page_url="http://x", score=0.8):
    return DocBlock(global_chunk_idx=idx, title=title, text=text, page_url=page_url, score=score)


class TestLocalGeneration:
    def test_no_context_returns_fallback_message(self):
        result = generate_answer("任意问题", [], backend="local")
        assert result["backend_used"] == "no_context"
        assert "未找到" in result["answer"]
        assert result["citations"] == []

    def test_local_backend_extracts_context_snippets(self):
        contexts = [_ctx(1, "退款规则", "退款需在签收后七天内申请退款")]
        result = generate_answer("怎么退款", contexts, backend="local")
        assert result["backend_used"] == "local"
        assert "退款规则" in result["answer"]
        assert len(result["citations"]) == 1
        assert result["citations"][0]["index"] == 1

    def test_multiple_contexts_all_cited(self):
        contexts = [_ctx(1, "标题1", "内容1"), _ctx(2, "标题2", "内容2")]
        result = generate_answer("问题", contexts, backend="local")
        assert len(result["citations"]) == 2
        assert result["citations"][0]["index"] == 1
        assert result["citations"][1]["index"] == 2

    def test_long_snippet_is_truncated(self):
        long_text = "字" * 500
        contexts = [_ctx(1, "长文档", long_text)]
        result = generate_answer("问题", contexts, backend="local")
        assert "..." in result["answer"]


class TestVLLMFallback:
    def test_vllm_unavailable_falls_back_to_local(self, monkeypatch):
        import rag.generation.generator as gen_mod

        monkeypatch.setattr(gen_mod, "_generate_with_vllm", lambda *a, **kw: None)
        contexts = [_ctx(1, "标题", "内容")]
        result = generate_answer("问题", contexts, backend="vllm")
        assert result["backend_used"] == "local"

    def test_vllm_available_uses_llm_answer(self, monkeypatch):
        import rag.generation.generator as gen_mod

        monkeypatch.setattr(gen_mod, "_generate_with_vllm", lambda *a, **kw: "这是 LLM 生成的回答 [1]")
        contexts = [_ctx(1, "标题", "内容")]
        result = generate_answer("问题", contexts, backend="vllm")
        assert result["backend_used"] == "vllm"
        assert result["answer"] == "这是 LLM 生成的回答 [1]"


class TestCitations:
    def test_citation_fields(self):
        contexts = [_ctx(1, "标题A", "内容A", page_url="http://example.com/a", score=0.756789)]
        result = generate_answer("问题", contexts, backend="local")
        citation = result["citations"][0]
        assert citation["page_url"] == "http://example.com/a"
        assert citation["title"] == "标题A"
        assert citation["score"] == 0.7568

    def test_build_citations_is_public_and_matches_generate_answer(self):
        """`build_citations` 现为公开函数（原 `_citations`），供 API 流式接口复用。"""
        contexts = [_ctx(1, "标题A", "内容A")]
        assert build_citations(contexts) == generate_answer("问题", contexts, backend="local")["citations"]


class TestBuildContextTextTruncation:
    """回归测试：修复 `_build_context_text` 遇超长上下文整体丢弃的 Bug（见 generator.py 注释）。"""

    def test_oversized_top1_context_not_dropped_entirely(self):
        """Top-1 上下文本身超过 max_chars 时，原 Bug 会返回空字符串；修复后应保留截断内容。"""
        long_text = "关键信息" * 1000  # 远超过 max_chars
        contexts = [_ctx(1, "唯一上下文", long_text)]
        result = _build_context_text(contexts, max_chars=100)
        assert result != ""
        assert "关键信息" in result
        assert len(result) <= 100 + 10  # 允许截断标记 "...\n" 带来的少量长度浮动

    def test_multiple_contexts_within_budget_all_included(self):
        contexts = [_ctx(1, "A", "短内容A"), _ctx(2, "B", "短内容B")]
        result = _build_context_text(contexts, max_chars=3000)
        assert "短内容A" in result
        assert "短内容B" in result

    def test_later_context_truncated_when_budget_exhausted(self):
        contexts = [_ctx(1, "A", "A" * 80), _ctx(2, "B", "B" * 80)]
        result = _build_context_text(contexts, max_chars=100)
        # 预算只够第一条完整或部分保留，第二条应被截断或完全排除，但不应报错
        assert "A" * 10 in result

    def test_empty_contexts_returns_empty_string(self):
        assert _build_context_text([], max_chars=100) == ""
