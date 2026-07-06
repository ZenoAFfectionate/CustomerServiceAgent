# -*- coding: utf-8 -*-
"""Prompt 模板（Prompt Template）：生成阶段使用的系统提示词与用户提示词拼装。

从原 `generator.py` 中抽出，便于后续针对不同场景（如多语言、不同风格的客服
话术）维护多套模板而不影响 `llm_generation.py` 的调用逻辑。
"""
from typing import Optional

SYSTEM_PROMPT = (
    "你是一个严谨的企业知识库客服助手。请仅根据下面提供的“检索上下文”回答用户问题，"
    "不要编造不存在的信息；如果上下文不足以回答，请明确说明“知识库中未找到相关信息”。"
    "回答末尾请用 [数字] 标注引用的上下文编号。"
)


def build_dialogue_history_text(dialogue: Optional[list]) -> str:
    """将多轮历史对话渲染为提示词中的文本片段。"""
    if not dialogue:
        return ""
    lines = []
    for turn in dialogue:
        role = "用户" if turn.get("speaker") == "user" else "助手"
        lines.append(f"{role}：{turn.get('text', '')}")
    return "\n".join(lines)


def build_user_prompt(query: str, context_text: str, dialogue: Optional[list] = None) -> str:
    """拼装送入 LLM 的用户消息内容：检索上下文 + 历史对话 + 当前问题。"""
    history = build_dialogue_history_text(dialogue)
    return f"【检索上下文】\n{context_text}\n\n【历史对话】\n{history}\n【当前问题】\n{query}"
