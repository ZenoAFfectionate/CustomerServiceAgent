# -*- coding: utf-8 -*-
"""健康检查与统计接口。"""
from fastapi import APIRouter

from rag.api.models import HealthResponse
from rag.indexing.indexer import get_stats

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="返回服务状态与各组件（向量库/关键词库/嵌入/精排/生成）当前使用的后端与统计信息。",
)
def health() -> HealthResponse:
    return HealthResponse(status="ok", stats=get_stats())
