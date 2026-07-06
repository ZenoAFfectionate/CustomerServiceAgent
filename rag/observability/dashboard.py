# -*- coding: utf-8 -*-
"""运维看板（Dashboard）：以 JSON 形式暴露 `monitoring.snapshot()` 与
`alerting.check_alerts()` 的结果，供简易运维监控使用。

作为 FastAPI `APIRouter` 提供，由 `rag/api/main.py` 挂载在 `/api` 前缀下
（`GET /api/dashboard`），不引入 Grafana 等外部看板系统，与 `rag/` 一贯的
"零外部依赖、开箱即用"设计原则一致。
"""
from fastapi import APIRouter

from rag.indexing.index_builder import get_stats
from rag.observability import monitoring
from rag.observability.alerting import check_alerts
from rag.retrieval.retriever_selection import get_active_backends

router = APIRouter(tags=["observability"])


@router.get(
    "/dashboard",
    summary="运维监控看板",
    description="返回请求量/延迟分位数/后端使用分布/错误率等监控快照，以及触发的阈值告警列表。",
)
def dashboard() -> dict:
    snap = monitoring.snapshot()
    return {
        "metrics": snap,
        "alerts": check_alerts(snap),
        "backends": get_active_backends(),
        "corpus_stats": get_stats(),
    }
