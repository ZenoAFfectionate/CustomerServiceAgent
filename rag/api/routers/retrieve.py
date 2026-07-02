# -*- coding: utf-8 -*-
"""检索接口：向量 + 关键词双模检索 → 融合去重 → Reranker 精排。"""
import time

from fastapi import APIRouter

from rag.api.errors import ValidationError
from rag.api.models import RetrieveRequest, RetrieveResponse
from rag.api.utils import parse_dialogue
from rag import pipeline

router = APIRouter(tags=["retrieve"])


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    summary="检索问答上下文",
    description=(
        "对用户查询执行「向量检索 + 关键词检索 → 融合去重 → Reranker 精排」全流程，"
        "返回精排后的 Top-K 文档块（含来源 page_url、block_path 与相关性分数）。"
    ),
)
def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
    query = (payload.query or "").strip()
    if not query:
        raise ValidationError("query 不能为空")

    t0 = time.time()
    results = pipeline.retrieve(query, dialogue=parse_dialogue(payload.dialogue), top_k=payload.top_k)
    latency_ms = (time.time() - t0) * 1000
    return RetrieveResponse(query=query, results=results, latency_ms=round(latency_ms, 2))
