# -*- coding: utf-8 -*-
"""知识库管理接口：文档上传 / 批量导入 / 列表 / 详情 / 删除。"""
from fastapi import APIRouter, File, UploadFile

from rag.api.errors import NotFoundError, ValidationError, PayloadTooLargeError
from rag.api.models import (
    DeleteResponse, DocumentInfo, DocumentListResponse, DocumentUploadResponse, IngestBlocksRequest,
)
from rag.config import RAG_CONFIG
from rag.indexing import indexer
from rag.indexing.registry import get_registry

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="上传文档并建立索引",
    description=(
        "上传单个文档文件，服务端自动解析（.txt/.md/.html/.htm/.json/.pdf）、分块、"
        "向量化并写入向量库与关键词库。支持的最大文件大小见服务端配置。"
    ),
)
async def upload_document(file: UploadFile = File(..., description="待上传的文档文件")) -> DocumentUploadResponse:
    raw = await file.read()
    if len(raw) > RAG_CONFIG["upload_max_size_mb"] * 1024 * 1024:
        raise PayloadTooLargeError(f"文件大小超过限制（{RAG_CONFIG['upload_max_size_mb']}MB）")
    try:
        meta = indexer.ingest_file(file.filename, raw)
    except ValueError as e:
        raise ValidationError(str(e))
    return DocumentUploadResponse(**meta)


@router.post(
    "/ingest_blocks",
    response_model=DocumentUploadResponse,
    summary="批量导入已分块的文档（如 process/ 输出的 JSON 块）",
    description="直接接收结构化文档块数组进行索引，跳过文档解析与分块步骤，适用于对接 process/ 数据处理流水线。",
)
def ingest_blocks(payload: IngestBlocksRequest) -> DocumentUploadResponse:
    try:
        meta = indexer.ingest_blocks(payload.blocks, filename=payload.filename)
    except ValueError as e:
        raise ValidationError(str(e))
    return DocumentUploadResponse(**meta)


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="知识库文档列表",
    description="返回当前知识库中已索引的全部文档及其分块数量。",
)
def list_documents() -> DocumentListResponse:
    docs = indexer.list_documents()
    return DocumentListResponse(total=len(docs), documents=[DocumentInfo(**d) for d in docs])


@router.get(
    "/{doc_id}",
    response_model=DocumentInfo,
    summary="文档详情",
    responses={404: {"description": "文档不存在"}},
)
def get_document(doc_id: str) -> DocumentInfo:
    meta = get_registry().get(doc_id)
    if meta is None:
        raise NotFoundError(f"文档不存在: {doc_id}")
    return DocumentInfo(**meta)


@router.delete(
    "/{doc_id}",
    response_model=DeleteResponse,
    summary="删除文档",
    description="从向量库、关键词库与知识库登记表中同时删除该文档的全部分块。",
    responses={404: {"description": "文档不存在"}},
)
def delete_document(doc_id: str) -> DeleteResponse:
    deleted = indexer.delete_document(doc_id)
    if not deleted:
        raise NotFoundError(f"文档不存在: {doc_id}")
    return DeleteResponse(doc_id=doc_id, deleted=True)
