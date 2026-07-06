# -*- coding: utf-8 -*-
"""agent/src/context/builder.py 单元测试：ContextBuilder 的 GSSC 流水线。

重点覆盖审查报告 L5 的回归验证：中文相关性打分（此前 `.split()` 对中文
整句因无空白分隔而近似恒为单 token，相关性分近似恒 0/1，`min_relevance`
过滤器对中文场景基本失效）。
"""
from hello_agents.context.builder import (
    ContextBuilder,
    ContextConfig,
    ContextPacket,
    _tokenize_for_relevance,
)


class TestTokenizeForRelevance:
    def test_chinese_text_split_into_multiple_tokens(self):
        """回归测试：修复 L5——中文整句此前会被当成一个 token。"""
        tokens = _tokenize_for_relevance("广告限流之后应该怎么办")
        assert len(tokens) > 1

    def test_english_text_still_split_by_whitespace(self):
        tokens = _tokenize_for_relevance("Hello World Test")
        assert tokens == {"hello", "world", "test"}

    def test_empty_text_returns_empty_set(self):
        assert _tokenize_for_relevance("") == set()
        assert _tokenize_for_relevance(None) == set()


class TestContextBuilderSelectRelevance:
    """`_select` 中文场景下的相关性打分应能正确反映关键词重叠程度，而非
    因整句未分词而近似恒为 0 或 1。"""

    def test_chinese_query_with_relevant_and_irrelevant_packets(self):
        builder = ContextBuilder(config=ContextConfig(min_relevance=0.0, max_tokens=8000))
        relevant = ContextPacket(content="广告限流后应该如何申请恢复投放", metadata={"type": "tool_result"})
        irrelevant = ContextPacket(content="今天天气晴朗适合出门散步", metadata={"type": "tool_result"})
        packets = [relevant, irrelevant]

        selected = builder._select(packets, user_query="广告限流了怎么办")

        relevant_selected = next(p for p in selected if p.content == relevant.content)
        irrelevant_selected = next(p for p in selected if p.content == irrelevant.content)
        # 修复前两者相关性分近似相同（均接近 0 或 1）；修复后应能区分出
        # 关键词重叠更高的 relevant packet 分数显著更高。
        assert relevant_selected.relevance_score > irrelevant_selected.relevance_score

    def test_min_relevance_filters_out_irrelevant_chinese_packet(self):
        """min_relevance 过滤器在中文场景下应能真正生效，过滤掉不相关内容。"""
        builder = ContextBuilder(config=ContextConfig(min_relevance=0.3, max_tokens=8000))
        relevant = ContextPacket(content="广告限流后应该如何申请恢复投放", metadata={"type": "tool_result"})
        irrelevant = ContextPacket(content="今天天气晴朗适合出门散步逛街", metadata={"type": "tool_result"})

        selected = builder._select([relevant, irrelevant], user_query="广告限流了怎么办")
        selected_contents = {p.content for p in selected}
        assert relevant.content in selected_contents
        assert irrelevant.content not in selected_contents
