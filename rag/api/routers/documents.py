# -*- coding: utf-8 -*-
"""知识库管理接口：文档上传 / 批量导入 / 列表 / 详情 / 删除。"""
from fastapi import APIRouter, File, UploadFile

from rag.api.errors import NotFoundError, ValidationError, PayloadTooLargeError
from rag.api.models import (
    DeleteResponse, DocumentInfo, DocumentListResponse, DocumentUploadResponse, IngestBlocksRequest,
)
from rag.config import RAG_CONFIG
from rag.knowledge_base import corpus_management

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
    max_size = RAG_CONFIG["upload_max_size_mb"] * 1024 * 1024
    raw = await _read_upload_within_limit(file, max_size)
    try:
        meta = corpus_management.ingest_upload(file.filename, raw)
    except ValueError as e:
        raise ValidationError(str(e))
    return DocumentUploadResponse(**meta)


async def _read_upload_within_limit(file: UploadFile, max_size: int, chunk_size: int = 1024 * 1024) -> bytes:
    """分块读取上传文件，一旦累计字节数超过 `max_size` 立即中止并抛出
    `PayloadTooLargeError`。

    【修复 L8】此前 `raw = await file.read()` 会先把整个文件完整读入内存，
    再校验 `len(raw) > max_size`——上传超大文件时，会在被拒绝之前就已经
    完整占用了对应内存（存在 OOM 风险）。改为按 `chunk_size`（默认 1MB）
    分块读取并累计计数，超限时立即中止，不再继续读取/占用后续内存。
    """
    chunks = []
    total = 0
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > max_size:
            raise PayloadTooLargeError(f"文件大小超过限制（{RAG_CONFIG['upload_max_size_mb']}MB）")
        chunks.append(chunk)
    return b"".join(chunks)


@router.post(
    "/ingest_blocks",
    response_model=DocumentUploadResponse,
    summary="批量导入已分块的文档（如 process/ 输出的 JSON 块）",
    description="直接接收结构化文档块数组进行索引，跳过文档解析与分块步骤，适用于对接 process/ 数据处理流水线。",
)
def ingest_blocks(payload: IngestBlocksRequest) -> DocumentUploadResponse:
    try:
        meta = corpus_management.ingest_blocks(payload.blocks, filename=payload.filename)
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
    docs = corpus_management.list_documents()
    return DocumentListResponse(total=len(docs), documents=[DocumentInfo(**d) for d in docs])


@router.get(
    "/{doc_id}",
    response_model=DocumentInfo,
    summary="文档详情",
    responses={404: {"description": "文档不存在"}},
)
def get_document(doc_id: str) -> DocumentInfo:
    meta = corpus_management.get_document(doc_id)
    if meta is None:
        raise NotFoundError(f"文档不存在: {doc_id}")
    return DocumentInfo(**meta)


@router.get(
    "/{doc_id}/history",
    summary="文档版本历史",
    description="返回该文档每次（重新）导入的内容哈希、块数与时间，见 `rag/knowledge_base/versioning.py`。",
)
def get_document_history(doc_id: str) -> list:
    return corpus_management.get_document_history(doc_id)


@router.delete(
    "/{doc_id}",
    response_model=DeleteResponse,
    summary="删除文档",
    description="从向量库、关键词库与知识库登记表中同时删除该文档的全部分块。",
    responses={404: {"description": "文档不存在"}},
)
def delete_document(doc_id: str) -> DeleteResponse:
    deleted = corpus_management.delete_document(doc_id)
    if not deleted:
        raise NotFoundError(f"文档不存在: {doc_id}")
    return DeleteResponse(doc_id=doc_id, deleted=True)
