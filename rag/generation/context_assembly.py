# -*- coding: utf-8 -*-
"""上下文组装（Context Assembly）：将精排后的检索上下文拼接为送入 LLM 的
文本，并控制在字符预算（`RAG_CONFIG['generation_max_context_chars']`）以内。

从原 `generator.py` 的私有函数 `_build_context_text` 抽出并公开导出。
"""
from typing import List, Tuple

from rag.schema import DocBlock


def build_context_text(contexts: List[DocBlock], max_chars: int) -> Tuple[str, List[DocBlock]]:
    """将检索上下文拼接为送入 LLM 的文本，并限制在 max_chars 以内。

    预算不足时对当前片段**截断**以填满剩余空间，而非整体丢弃 —— 若 Top-1
    上下文本身长度就超过 max_chars，直接整体跳过会导致最终拼接结果为空
    字符串，但 `citations` 仍会包含该条引用，产生"回答未见任何上下文却
    标注引用"的逻辑错误。截断保证排序在前（更相关）的上下文始终至少被
    部分保留。

    Args:
        contexts: 精排后的 DocBlock 列表（已按相关性降序排列）
        max_chars: 拼接结果的最大字符数

    Returns:
        (拼接后的上下文文本, 实际被写入文本的上下文子集)：
        后者用于 `llm_generation.py` 按"实际进入 LLM 上下文的块"而非全部
        Top-K 构建 `citations`——此前 `build_citations(contexts)` 对全部
        Top-K 编号，但预算不足时靠后块会被本函数整段丢弃（`remaining<=0`
        即 `break`），导致引用编号可能指向 LLM 实际未见过的块（审查报告
        M6：引用编号与送入 LLM 的上下文不匹配）。contexts 为空时返回
        `("", [])`。
    """
    parts = []
    included: List[DocBlock] = []
    remaining = max_chars
    for i, c in enumerate(contexts, start=1):
        if remaining <= 0:
            break
        piece = f"[{i}] {c.title or c.page_name}\n{c.text}\n"
        if len(piece) > remaining:
            piece = piece[:remaining] + "...\n"
        parts.append(piece)
        included.append(c)
        remaining -= len(piece)
    return "\n".join(parts), included
