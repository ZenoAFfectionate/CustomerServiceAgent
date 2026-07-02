# -*- coding: utf-8 -*-
"""API 路由层共享工具函数。

【优化点】原 `chat.py` 与 `retrieve.py` 各自实现了一份几乎相同的
`_parse_dialogue`（将 Pydantic `DialogueTurn` 列表转为 dict 列表），抽取到此
处统一维护，避免后续修改遗漏其中一处。
"""
from typing import List, Optional

from rag.api.models import DialogueTurn


def parse_dialogue(dialogue: Optional[List[DialogueTurn]]) -> Optional[List[dict]]:
    """将请求体中的 `DialogueTurn` 列表转换为 `rag.pipeline` 期望的 dict 列表。"""
    if not dialogue:
        return None
    return [t.model_dump() for t in dialogue]
