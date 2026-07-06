# -*- coding: utf-8 -*-
"""引用构建（Citation）：根据检索上下文构建结构化引用列表。

从原 `generator.py` 的公开函数 `build_citations` 抽出（保持函数名/签名不变，
`rag/api/routers/chat.py` 的 SSE 流式接口需要在生成阶段开始前就先推送
`citations` 事件，复用同一份引用构建逻辑可避免多处实现不一致）。
"""
from typing import List

from rag.schema import DocBlock


def build_citations(contexts: List[DocBlock]) -> List[dict]:
    """根据上下文构建引用列表（公开函数，供 API 层在流式响应中提前推送引用）。"""
    return [
        {"index": i + 1, "page_url": c.page_url, "block_path": c.block_path,
         "title": c.title, "score": round(c.score, 4)}
        for i, c in enumerate(contexts)
    ]
