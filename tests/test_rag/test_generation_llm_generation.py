# -*- coding: utf-8 -*-
"""rag/generation/llm_generation.py 单元测试：抽取式兜底生成 + vLLM 降级逻辑。"""
from rag.generation.llm_generation import generate_answer
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

    def test_default_backend_reads_from_rag_config(self):
        """未显式指定 backend 时应读取 RAG_CONFIG['generation_backend']（默认 local）。"""
        contexts = [_ctx(1, "标题", "内容")]
        result = generate_answer("问题", contexts)
        assert result["backend_used"] in ("local", "vllm")


class TestVLLMFallback:
    def test_vllm_unavailable_falls_back_to_local(self, monkeypatch):
        import rag.generation.llm_generation as gen_mod

        # _generate_with_vllm 返回 (answer, included_contexts) 二元组（M6 修复后签名）
        monkeypatch.setattr(gen_mod, "_generate_with_vllm", lambda *a, **kw: (None, []))
        contexts = [_ctx(1, "标题", "内容")]
        result = generate_answer("问题", contexts, backend="vllm")
        assert result["backend_used"] == "local"

    def test_vllm_available_uses_llm_answer(self, monkeypatch):
        import rag.generation.llm_generation as gen_mod

        contexts = [_ctx(1, "标题", "内容")]
        monkeypatch.setattr(
            gen_mod, "_generate_with_vllm", lambda *a, **kw: ("这是 LLM 生成的回答 [1]", contexts)
        )
        result = generate_answer("问题", contexts, backend="vllm")
        assert result["backend_used"] == "vllm"
        # 幻觉护栏默认只记录日志、不修改回答文本（RAG_CONFIG['hallucination_append_caveat'] 默认 False）
        assert result["answer"] == "这是 LLM 生成的回答 [1]"

    def test_vllm_result_passes_through_apply_guardrail(self, monkeypatch):
        """组合测试：vLLM 成功路径应经过 generation/hallucination_control.py 的
        apply_guardrail 后处理（即使默认不修改文本，也应被调用而不抛异常）。"""
        import rag.generation.llm_generation as gen_mod

        called = {}

        def _fake_guardrail(answer_text, contexts, backend_used):
            called["invoked"] = True
            called["backend_used"] = backend_used
            return answer_text

        contexts = [_ctx(1, "标题", "内容")]
        monkeypatch.setattr(gen_mod, "_generate_with_vllm", lambda *a, **kw: ("回答内容 [1]", contexts))
        monkeypatch.setattr(gen_mod, "apply_guardrail", _fake_guardrail)
        generate_answer("问题", contexts, backend="vllm")
        assert called.get("invoked") is True
        assert called.get("backend_used") == "vllm"

    def test_vllm_citations_scoped_to_included_contexts(self, monkeypatch):
        """M6 回归测试：citations 应仅覆盖 `_generate_with_vllm` 实际返回的
        included_contexts，而非全部传入的 contexts（预算截断场景）。"""
        import rag.generation.llm_generation as gen_mod

        all_contexts = [_ctx(1, "A", "内容A"), _ctx(2, "B", "内容B"), _ctx(3, "C", "内容C")]
        included = all_contexts[:2]  # 模拟第 3 条因预算不足被 build_context_text 丢弃
        monkeypatch.setattr(gen_mod, "_generate_with_vllm", lambda *a, **kw: ("回答 [1][2]", included))
        result = generate_answer("问题", all_contexts, backend="vllm")
        assert len(result["citations"]) == 2
        assert [c["index"] for c in result["citations"]] == [1, 2]


class TestCitationsIntegration:
    def test_build_citations_matches_generate_answer_citations(self):
        """组合测试：generate_answer 内部复用 generation/citation.py 的 build_citations，
        结果应与直接调用完全一致。"""
        from rag.generation.citation import build_citations

        contexts = [_ctx(1, "标题A", "内容A")]
        assert build_citations(contexts) == generate_answer("问题", contexts, backend="local")["citations"]


class TestSelectCitedContexts:
    """select_cited_contexts：为 SSE 提前推送引用 与 generate_answer 内部
    共用同一套裁剪逻辑（M6）。"""

    def test_local_backend_returns_all_contexts(self):
        from rag.generation.llm_generation import select_cited_contexts

        contexts = [_ctx(1, "A", "内容A"), _ctx(2, "B", "内容B")]
        assert select_cited_contexts(contexts, backend="local") == contexts

    def test_empty_contexts_returns_empty(self):
        from rag.generation.llm_generation import select_cited_contexts

        assert select_cited_contexts([], backend="vllm") == []

    def test_vllm_backend_truncates_by_budget(self, monkeypatch):
        from rag.generation.llm_generation import select_cited_contexts
        import rag.generation.llm_generation as gen_mod

        # piece 格式为 "[i] {title}\n{text}\n"：title="A"/text=60 个字符 → 恰好
        # 消耗 6+60+1=67 字符；max_chars=67 使第一条恰好用尽预算，第二条被
        # `remaining<=0` 短路跳过（不进入 truncate 分支），断言更精确。
        contexts = [_ctx(1, "A", "x" * 60), _ctx(2, "B", "y" * 60)]
        monkeypatch.setitem(gen_mod.RAG_CONFIG, "generation_max_context_chars", 67)
        included = select_cited_contexts(contexts, backend="vllm")
        assert included == contexts[:1]


class TestStreamAnswer:
    """stream_answer：真流式生成入口（M3）。"""

    def test_no_context_yields_fallback_and_done(self):
        from rag.generation.llm_generation import stream_answer

        events = list(stream_answer("问题", [], backend="local"))
        assert events[-1][0] == "done"
        assert events[-1][1]["backend_used"] == "no_context"
        assert any(kind == "delta" for kind, _ in events)

    def test_local_backend_streams_chunked_deltas(self):
        from rag.generation.llm_generation import stream_answer

        contexts = [_ctx(1, "标题", "内容")]
        events = list(stream_answer("问题", contexts, backend="local"))
        assert events[-1] == ("done", {"answer": events[-1][1]["answer"], "backend_used": "local"})
        full_text = "".join(data for kind, data in events if kind == "delta")
        assert full_text == events[-1][1]["answer"]

    def test_vllm_stream_failure_falls_back_to_local(self, monkeypatch):
        from rag.generation.llm_generation import stream_answer
        import rag.generation.llm_generation as gen_mod

        def _boom(*a, **kw):
            raise RuntimeError("连接失败")
            yield  # pragma: no cover - 使函数成为 generator

        monkeypatch.setattr(gen_mod, "_stream_with_vllm", _boom)
        contexts = [_ctx(1, "标题", "内容")]
        events = list(stream_answer("问题", contexts, backend="vllm"))
        assert events[-1][0] == "done"
        assert events[-1][1]["backend_used"] == "local"

    def test_vllm_stream_success_yields_deltas_in_order(self, monkeypatch):
        from rag.generation.llm_generation import stream_answer
        import rag.generation.llm_generation as gen_mod

        contexts = [_ctx(1, "标题", "内容")]
        monkeypatch.setattr(gen_mod, "_stream_with_vllm", lambda *a, **kw: iter(["你好", "，世界"]))
        events = list(stream_answer("问题", contexts, backend="vllm"))
        deltas = [data for kind, data in events if kind == "delta"]
        assert "".join(deltas).startswith("你好，世界")
        assert events[-1][0] == "done"
        assert events[-1][1]["backend_used"] == "vllm"
