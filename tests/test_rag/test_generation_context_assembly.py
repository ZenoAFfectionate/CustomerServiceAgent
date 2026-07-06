# -*- coding: utf-8 -*-
"""rag/generation/context_assembly.py 单元测试：检索上下文 → LLM 输入文本的
字符预算控制。"""
from rag.generation.context_assembly import build_context_text
from rag.schema import DocBlock


def _ctx(idx, title, text, page_url="http://x", score=0.8):
    return DocBlock(global_chunk_idx=idx, title=title, text=text, page_url=page_url, score=score)


class TestBuildContextTextTruncation:
    """回归测试：修复 `build_context_text` 遇超长上下文整体丢弃的 Bug。

    `build_context_text` 返回 `(text, included_contexts)` 二元组，其中
    `included_contexts` 用于修复审查报告 M6（引用编号与送入 LLM 的上下文
    不匹配）：调用方应据此裁剪 `citations`，而非对全部 Top-K 编号。
    """

    def test_oversized_top1_context_not_dropped_entirely(self):
        """Top-1 上下文本身超过 max_chars 时，原 Bug 会返回空字符串；修复后应保留截断内容。"""
        long_text = "关键信息" * 1000  # 远超过 max_chars
        contexts = [_ctx(1, "唯一上下文", long_text)]
        result, included = build_context_text(contexts, max_chars=100)
        assert result != ""
        assert "关键信息" in result
        assert len(result) <= 100 + 10  # 允许截断标记 "...\n" 带来的少量长度浮动
        assert included == contexts

    def test_multiple_contexts_within_budget_all_included(self):
        contexts = [_ctx(1, "A", "短内容A"), _ctx(2, "B", "短内容B")]
        result, included = build_context_text(contexts, max_chars=3000)
        assert "短内容A" in result
        assert "短内容B" in result
        assert included == contexts

    def test_later_context_truncated_when_budget_exhausted(self):
        contexts = [_ctx(1, "A", "A" * 80), _ctx(2, "B", "B" * 80)]
        result, included = build_context_text(contexts, max_chars=100)
        # 预算只够第一条完整或部分保留，第二条应被截断或完全排除，但不应报错
        assert "A" * 10 in result

    def test_empty_contexts_returns_empty_string(self):
        text, included = build_context_text([], max_chars=100)
        assert text == ""
        assert included == []

    def test_contexts_ordered_by_relevance_prioritized(self):
        """预算不足以容纳全部上下文时，排序靠前（更相关）的内容应优先完整保留。"""
        contexts = [_ctx(1, "最相关", "A" * 30), _ctx(2, "次相关", "B" * 200)]
        result, included = build_context_text(contexts, max_chars=50)
        assert "A" * 30 in result


class TestBuildContextTextIncludedContexts:
    """`included_contexts` 应准确反映"实际写入 LLM 上下文文本"的块子集，
    供 `llm_generation.select_cited_contexts` 用于裁剪 citations（M6）。"""

    def test_dropped_tail_context_excluded_from_included(self):
        """预算耗尽后被整段跳过（`remaining <= 0` 即 break）的靠后块，
        不应出现在 included_contexts 中——否则 citations 会指向 LLM 未见过的块。
        每条 piece 格式为 "[i] {title}\\n{text}\\n"，此处 title="A"/text=60 个 "A"，
        恰好消耗 6+60+1=67 字符；max_chars=67 使第一条恰好用尽预算，第二条在
        truncate 逻辑之前就被 `remaining<=0` 短路跳过，不会出现"部分截断"的
        歧义情况，断言更精确。"""
        contexts = [_ctx(1, "A", "A" * 60), _ctx(2, "B", "B" * 60), _ctx(3, "C", "C" * 60)]
        text, included = build_context_text(contexts, max_chars=67)
        assert included == contexts[:1]
        assert "B" not in text
        assert "C" * 10 not in text

    def test_included_is_prefix_of_original_order(self):
        """included_contexts 应保持与原 contexts 一致的相对顺序（均为前缀）。"""
        contexts = [_ctx(i, f"T{i}", "x" * 20) for i in range(1, 6)]
        text, included = build_context_text(contexts, max_chars=45)
        assert included == contexts[: len(included)]
        assert len(included) < len(contexts)
