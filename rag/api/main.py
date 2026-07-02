# -*- coding: utf-8 -*-
"""RAG 后端服务入口。

启动：
    uvicorn rag.api.main:app --host 0.0.0.0 --port 8090 --reload

接口文档：
    - Swagger UI:  http://localhost:8090/docs
    - ReDoc:       http://localhost:8090/redoc
    - OpenAPI JSON http://localhost:8090/openapi.json
    - 前端页面:     http://localhost:8090/ui/

Agent 友好的纯文本接口文档见 `rag/docs/API_REFERENCE.md`。
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from config.config_loader import logger
from rag.api.errors import RagAPIError, rag_error_handler, unhandled_error_handler
from rag.api.routers import chat, documents, health, retrieve

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_WEB_DIR = os.path.join(os.path.dirname(_CURRENT_DIR), "web")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """【优化点】服务启动时预热单例（Embedder / 知识库登记表 / 向量库 / 关键词库），
    避免首个真实请求承担懒加载初始化开销（尤其真实 Milvus/ES 后端的首次连接耗时）。
    预热失败不影响服务启动（本地降级后端下几乎不会失败；真实后端连接失败会在
    请求时按既有降级逻辑处理，这里仅做尽力预热）。

    使用 FastAPI 推荐的 `lifespan` 上下文管理器写法，替代已废弃的 `@app.on_event`。
    """
    try:
        from rag.indexing.embedder import get_embedder
        from rag.indexing.registry import get_registry
        from rag.indexing.vector_store import get_vector_store
        from rag.indexing.keyword_store import get_keyword_store

        get_embedder()
        get_registry()
        get_vector_store()
        get_keyword_store()
    except Exception as e:  # pragma: no cover - 预热失败不应阻塞启动
        logger.warning(f"⚠️ 服务预热失败（不影响启动，将在首次请求时懒加载）: {e}")
    yield


app = FastAPI(
    title="CustomerServiceAgent RAG API",
    description=(
        "检索增强生成（RAG）系统后端接口。覆盖知识库文档管理、双模检索（向量 + 关键词）+ "
        "融合去重 + Reranker 精排、以及检索增强问答（RAG QA）。\n\n"
        "详见纯文本版接口文档：`rag/docs/API_REFERENCE.md`（专为 Agent 自动化调用优化）。"
    ),
    version="0.1.0",
    contact={"name": "CustomerServiceAgent"},
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # 【优化点/安全修复】原配置 `allow_credentials=True` 与 `allow_origins=["*"]`
    # 同时启用属于 CORS 反模式：规范上通配符 origin 不应与凭证请求（Cookie/
    # Authorization）共用，部分浏览器会直接拒绝该组合；本 API 当前无鉴权/
    # Cookie 会话依赖，禁用 allow_credentials 更安全且不影响任何现有调用方式。
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(RagAPIError, rag_error_handler)
app.add_exception_handler(Exception, unhandled_error_handler)

app.include_router(health.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(retrieve.router, prefix="/api")
app.include_router(chat.router, prefix="/api")

if os.path.isdir(_WEB_DIR):
    app.mount("/ui", StaticFiles(directory=_WEB_DIR, html=True), name="ui")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/" if os.path.isdir(_WEB_DIR) else "/docs")
