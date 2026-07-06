# -*- coding: utf-8 -*-
"""FastAPI 请求/响应 Pydantic 模型（同时驱动自动生成的 OpenAPI Schema）。"""
from typing import List, Optional

from pydantic import BaseModel, Field


class DialogueTurn(BaseModel):
    speaker: str = Field(..., description="发言角色：user 或 bot", examples=["user"])
    text: str = Field(..., description="发言内容", examples=["广告为什么被限流了？"])


class RetrieveRequest(BaseModel):
    query: str = Field(..., description="用户查询", examples=["广告限流后怎么办"], min_length=1)
    dialogue: Optional[List[DialogueTurn]] = Field(default=None, description="多轮历史对话，用于 query 重写（可选）")
    top_k: Optional[int] = Field(default=None, description="返回条数，默认使用服务端配置", ge=1, le=50)


class ContextItem(BaseModel):
    # 【修复 L16】此前全部字段无默认值（必填），与 `DocBlock.to_dict()` 的
    # 输出脆弱耦合——真实后端（Milvus/ES）若返回缺字段的记录（如字段被
    # 截断、mapping 变更后旧数据缺新字段），`RetrieveResponse(...)` 校验会
    # 直接抛 422，而不是优雅降级。现对齐 `rag/schema.py` 中 `DocBlock` 各
    # 字段的默认值，缺字段时兜底为该默认值而非直接校验失败。
    global_chunk_idx: int = -1
    doc_id: str = ""
    chunk_idx: int = 0
    page_name: str = ""
    title: str = ""
    page_url: str = ""
    text: str = ""
    html_content: str = ""
    block_path: str = ""
    summary: str = ""
    question: str = ""
    time: str = ""
    score: float = 0.0
    source_retriever: str = ""
    # 【修复 N35】DocBlock.to_dict() 始终输出 source 字段（文档来源标识），
    # 但 ContextItem 此前无此字段，经 API 返回时 source 被静默丢弃。
    source: str = ""


class RetrieveResponse(BaseModel):
    query: str
    results: List[ContextItem]
    latency_ms: float


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题", examples=["千川广告投放规则是什么"], min_length=1)
    dialogue: Optional[List[DialogueTurn]] = Field(default=None, description="多轮历史对话（可选）")
    top_k: Optional[int] = Field(default=None, description="检索上下文条数", ge=1, le=20)


class CitationItem(BaseModel):
    index: int
    page_url: str
    block_path: str = ""
    title: str = ""
    score: float = 0.0


class ChatResponse(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    answer: str
    citations: List[CitationItem]
    backend_used: str
    contexts: List[ContextItem]
    latency_ms: float


class DocumentUploadResponse(BaseModel):
    doc_id: str
    filename: str
    source: str
    num_chunks: int
    chunk_ids: List[int]
    created_at: str


class IngestBlocksRequest(BaseModel):
    blocks: List[dict] = Field(..., description="文档块列表（process/ 输出的 JSON 数组）")
    filename: str = Field(default="manual", description="来源标识（如文件名/页面名）")


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    source: str
    num_chunks: int
    chunk_ids: List[int]
    created_at: str


class DocumentListResponse(BaseModel):
    total: int
    documents: List[DocumentInfo]


class DeleteResponse(BaseModel):
    doc_id: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    stats: dict


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    detail: Optional[str] = None
