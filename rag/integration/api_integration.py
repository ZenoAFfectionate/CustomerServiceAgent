# -*- coding: utf-8 -*-
"""API 集成（API Integration）：将 `rag/api` 的 FastAPI 应用作为独立服务
获取，或挂载进宿主服务（如与 `agent/` 共用一个进程/端口时）。
"""
from typing import Optional


def get_app():
    """返回 `rag/` 独立部署的 FastAPI 应用实例（`rag/api/main.py` 的 `app`）。"""
    from rag.api.main import app
    return app


def mount_rag_api(host_app, prefix: str = "/rag/api") -> None:
    """将 RAG 的全部路由挂载进一个已存在的宿主 FastAPI 应用（而不是让 rag/
    独立跑一个进程/端口）。适用于希望把 RAG 能力嵌入到更大的后端服务
    （如与 `agent/` 共用同一进程）的场景。

    Args:
        host_app: 宿主 FastAPI 应用实例
        prefix: 挂载路径前缀，默认 `/rag/api`（避免与宿主自身的 `/api` 冲突）
    """
    from rag.api.routers import chat, documents, health, retrieve
    from rag.observability import dashboard

    for router in (health.router, documents.router, retrieve.router, chat.router, dashboard.router):
        host_app.include_router(router, prefix=prefix)
