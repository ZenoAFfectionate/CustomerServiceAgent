# -*- coding: utf-8 -*-
"""rag/generation/prompt_template.py 单元测试：系统提示词与用户提示词拼装。"""
from rag.generation.prompt_template import SYSTEM_PROMPT, build_dialogue_history_text, build_user_prompt


class TestSystemPrompt:
    def test_system_prompt_mentions_grounding_constraint(self):
        assert "检索上下文" in SYSTEM_PROMPT
        assert "编造" in SYSTEM_PROMPT


class TestBuildDialogueHistoryText:
    def test_empty_dialogue_returns_empty_string(self):
        assert build_dialogue_history_text(None) == ""
        assert build_dialogue_history_text([]) == ""

    def test_renders_user_and_bot_turns(self):
        dialogue = [
            {"speaker": "user", "text": "广告为什么被限流了"},
            {"speaker": "bot", "text": "触发了风控规则"},
        ]
        result = build_dialogue_history_text(dialogue)
        assert "用户：广告为什么被限流了" in result
        assert "助手：触发了风控规则" in result

    def test_multiple_turns_preserve_order(self):
        dialogue = [
            {"speaker": "user", "text": "第一轮"},
            {"speaker": "bot", "text": "回复一"},
            {"speaker": "user", "text": "第二轮"},
        ]
        result = build_dialogue_history_text(dialogue)
        assert result.index("第一轮") < result.index("回复一") < result.index("第二轮")


class TestBuildUserPrompt:
    def test_contains_context_history_and_query(self):
        prompt = build_user_prompt("当前问题", "上下文内容", dialogue=[{"speaker": "user", "text": "历史问题"}])
        assert "上下文内容" in prompt
        assert "历史问题" in prompt
        assert "当前问题" in prompt

    def test_no_dialogue_still_valid(self):
        prompt = build_user_prompt("问题", "上下文", dialogue=None)
        assert "问题" in prompt
        assert "上下文" in prompt
