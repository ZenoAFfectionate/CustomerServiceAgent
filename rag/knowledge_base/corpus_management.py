# -*- coding: utf-8 -*-
"""知识库高层编排（Corpus Management）：对外统一入口，组合
`indexing/index_builder.py`（写索引）+ `quality_control.py`（质检）+
`versioning.py`（版本追踪），供 API 层（`rag/api/routers/documents.py`）与
`knowledge_base/update_sync.py` 调用。

`indexing.index_builder` 仍保留独立的低层单测（`tests/test_rag/test_rag_indexer.py`），
本模块只做"编排"，不重复实现底层解析/分块/写入逻辑。
"""
from typing import List, Optional

from config.config_loader import logger
from rag.indexing import index_builder
from rag.indexing.metadata import get_registry
from rag.knowledge_base.quality_control import check_blocks_quality
from rag.knowledge_base.versioning import compute_content_hash, get_version_store
from rag.observability.logging import log_event


def ingest_upload(filename: str, raw: bytes, doc_id: Optional[str] = None) -> dict:
    """上传文件入库（质检 + 索引写入 + 版本记录）。"""
    meta = index_builder.ingest_file(filename, raw, doc_id=doc_id, source="upload")
    # 【修复 L14】统一用入库前的纯内容块计算哈希（与 ingest_blocks 路径
    # 一致），不再对原始字节算哈希——两条路径的哈希空间此前不一致，
    # 同一 doc_id 跨路径导入时版本哈希无法互相比较。
    content_blocks = meta.pop("_content_blocks", None) or []
    _record_version_best_effort(meta["doc_id"], content_hash=compute_content_hash(content_blocks), meta=meta)
    # 【日志完善】导入成功此前完全没有任何日志（只有质检警告才会记录），导致
    # "什么文档、什么时候、被导入了多少块"这类审计/回溯信息只能去翻
    # versions.json 而无法从日志直接检索。补充结构化事件日志（`event=rag.ingest`），
    # 与 `observability/logging.py` docstring 承诺的事件名保持一致。
    log_event("rag.ingest", doc_id=meta["doc_id"], filename=filename, source="upload", num_chunks=meta["num_chunks"])
    return meta


def ingest_blocks(blocks: List[dict], filename: str = "manual", doc_id: Optional[str] = None, source: str = "ingest_blocks") -> dict:
    """批量导入已成型的知识块（质检 + 索引写入 + 版本记录）。"""
    report = check_blocks_quality(blocks)
    if report["warnings"]:
        logger.warning(f"⚠️ 知识块质检发现问题（filename={filename}）: {report['warnings']}")

    meta = index_builder.ingest_blocks(blocks, filename=filename, doc_id=doc_id, source=source)
    # 【修复 L14】统一用入库前的纯内容块计算哈希（与 ingest_upload 路径
    # 一致），不再各自维护过滤逻辑。
    content_blocks = meta.pop("_content_blocks", None) or []
    _record_version_best_effort(meta["doc_id"], content_hash=compute_content_hash(content_blocks), meta=meta)
    log_event("rag.ingest", doc_id=meta["doc_id"], filename=filename, source=source, num_chunks=meta["num_chunks"])
    return meta


def _record_version_best_effort(doc_id: str, content_hash: str, meta: dict) -> None:
    try:
        get_version_store().record_version(doc_id, content_hash, meta["num_chunks"], filename=meta.get("filename", ""))
    except Exception as e:  # pragma: no cover - 版本记录失败不应影响主导入流程
        logger.warning(f"⚠️ 版本记录失败（不影响导入结果）: {e}")


def delete_document(doc_id: str) -> bool:
    """删除文档及其全部索引块。删除属于不可逆的破坏性操作，此前完全没有任何
    日志记录——一旦发生误删，事后无法从日志追溯"谁在什么时候删了什么文档"。
    补充结构化事件日志（无论删除是否命中都记录，`deleted=False` 表示 doc_id
    本身不存在，便于区分"误删成功"与"删除了不存在的 ID 未生效"两种场景）。
    """
    try:
        deleted = index_builder.delete_document(doc_id)
    except Exception as e:
        # 【修复 H3】index_builder.delete_document 现在会在存储环节部分失败时
        # 抛出 RuntimeError（而非静默返回 False）。这里补充失败事件日志后
        # 原样上抛，交由 API 层的全局异常处理器转换为 500——避免把"部分删除
        # 失败、数据可能残留不一致"误报为更轻量的"文档不存在"（404）。
        log_event("rag.delete", level="error", doc_id=doc_id, deleted=False, error=str(e))
        raise
    # 删除命中记为 info（正常的预期操作）；未命中（doc_id 不存在）记为 warning，
    # 提示调用方可能存在重复删除/ID 过期等异常调用场景，值得留意但不阻断流程。
    log_event("rag.delete", level="info" if deleted else "warning", doc_id=doc_id, deleted=deleted)
    return deleted


def list_documents() -> List[dict]:
    return index_builder.list_documents()


def get_document(doc_id: str) -> Optional[dict]:
    return get_registry().get(doc_id)


def get_document_history(doc_id: str) -> List[dict]:
    """返回该文档的历史版本列表（见 `versioning.py`）。"""
    return get_version_store().get_history(doc_id)


def get_corpus_stats() -> dict:
    return index_builder.get_stats()
