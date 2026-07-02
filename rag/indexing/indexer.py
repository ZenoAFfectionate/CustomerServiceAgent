# -*- coding: utf-8 -*-
"""索引编排入口：文档解析 → 分块 → 向量化 → 写入向量库/关键词库 → 登记。

供 API 层（`rag/api/routers/documents.py`）与 `scripts/build_index.py` 共同调用，
是 objective 1「文档解析、文本分块、embedding、向量存储」四个环节的编排层。
"""
import os
from typing import List, Optional

from rag.config import RAG_CONFIG
from rag.indexing.chunker import chunk_text
from rag.indexing.embedder import get_embedder
from rag.indexing.loader import parse_document, ParseError
from rag.indexing.registry import get_registry
from rag.indexing.vector_store import get_vector_store
from rag.indexing.keyword_store import get_keyword_store
from rag.schema import DocBlock


class IndexError_(Exception):
    """索引写入失败。"""


def _build_blocks_from_text(text: str, doc_id: str, filename: str, global_ids: List[int]) -> List[DocBlock]:
    blocks = []
    for i, (chunk, gid) in enumerate(zip(chunk_text(text, RAG_CONFIG["chunk_size"], RAG_CONFIG["chunk_overlap"]), global_ids)):
        blocks.append(DocBlock(
            text=chunk, global_chunk_idx=gid, doc_id=doc_id, source=filename,
            chunk_idx=i, page_name=filename, title=filename, page_url=filename,
        ))
    return blocks


def _build_blocks_from_process_json(raw_blocks: List[dict], doc_id: str, filename: str, global_ids: List[int]) -> List[DocBlock]:
    blocks = []
    for i, (gid, raw) in enumerate(zip(global_ids, raw_blocks)):
        known = {k: v for k, v in raw.items() if k in DocBlock.__dataclass_fields__}
        known["global_chunk_idx"] = gid
        known["doc_id"] = doc_id
        known.setdefault("source", filename)
        # 【Bug 修复】原实现未回填 chunk_idx：当 process/ 输出的知识块本身未携带
        # 该字段时，DocBlock 默认值 0 会被所有块共享，导致同一文档内多条块的
        # chunk_idx 全部重复为 0，干扰引用溯源与去重。缺省或为空时按数组位置回填。
        if known.get("chunk_idx") is None:
            known["chunk_idx"] = i
        blocks.append(DocBlock(**known))
    return blocks


def _filter_non_empty_blocks(raw_blocks: List[dict]) -> List[dict]:
    """【Bug 修复】过滤 text 字段为空的知识块。

    原实现未做任何过滤：空文本块会被 Embedder 编码为零向量并写入向量库/
    关键词库，不但检索无意义，还会污染 `num_vector_chunks` 等统计数据。
    """
    return [b for b in raw_blocks if isinstance(b, dict) and (b.get("text") or "").strip()]


def _check_block_count_limit(n: int) -> None:
    limit = RAG_CONFIG["max_blocks_per_ingest"]
    if n > limit:
        raise ValueError(f"单次导入的知识块数量（{n}）超过限制（{limit}），请分批导入")


def ingest_file(filename: str, raw: bytes, doc_id: Optional[str] = None, source: str = "upload") -> dict:
    """解析并索引一个上传文件，返回文档登记元信息。"""
    # 【Bug 修复】原实现未校验 filename 为空/None 的情况，`os.path.splitext(None)`
    # 会直接抛出 TypeError（表现为 500），改为提前显式校验并返回语义化错误。
    if not filename or not filename.strip():
        raise ValueError("文件名不能为空")

    ext = os.path.splitext(filename)[1].lower()
    max_size = RAG_CONFIG["upload_max_size_mb"] * 1024 * 1024
    if len(raw) > max_size:
        raise ValueError(f"文件大小超过限制（{RAG_CONFIG['upload_max_size_mb']}MB）")
    if ext not in RAG_CONFIG["allowed_upload_ext"]:
        raise ValueError(f"不支持的文件类型: {ext}，支持: {RAG_CONFIG['allowed_upload_ext']}")

    # 【Bug 修复】原实现未捕获 `ParseError`（如 JSON 格式错误、PDF 解析失败等
    # 恶意/损坏文件场景），会导致该异常穿透到 API 层被通用 500 处理器捕获，
    # 语义上应等价于参数校验失败（422），故在此统一转换为 ValueError。
    try:
        parsed = parse_document(filename, raw)
    except ParseError as e:
        raise ValueError(str(e)) from e

    registry = get_registry()
    doc_id = doc_id or registry.new_doc_id()

    if isinstance(parsed, list):
        raw_blocks = _filter_non_empty_blocks(parsed)
        if not raw_blocks:
            raise ValueError("文档解析后未产生任何有效文本块（内容为空或全部被过滤）")
        _check_block_count_limit(len(raw_blocks))
        global_ids = registry.next_global_ids(len(raw_blocks))
        blocks = _build_blocks_from_process_json(raw_blocks, doc_id, filename, global_ids)
    else:
        chunks = chunk_text(parsed, RAG_CONFIG["chunk_size"], RAG_CONFIG["chunk_overlap"])
        if not chunks:
            raise ValueError("文档解析后未产生任何有效文本块（内容为空或全部被过滤）")
        _check_block_count_limit(len(chunks))
        global_ids = registry.next_global_ids(len(chunks))
        blocks = _build_blocks_from_text(parsed, doc_id, filename, global_ids)

    return _write_blocks(blocks, doc_id, filename, source=source)


def ingest_blocks(raw_blocks: List[dict], filename: str = "manual", doc_id: Optional[str] = None, source: str = "ingest_blocks") -> dict:
    """直接索引一批已成型的文档块（如 process/ 产出的 JSON）。"""
    if not raw_blocks:
        raise ValueError("blocks 不能为空")

    filtered = _filter_non_empty_blocks(raw_blocks)
    if not filtered:
        raise ValueError("blocks 中所有文档块的 text 字段均为空，无法建立索引")
    _check_block_count_limit(len(filtered))

    registry = get_registry()
    doc_id = doc_id or registry.new_doc_id()
    global_ids = registry.next_global_ids(len(filtered))
    blocks = _build_blocks_from_process_json(filtered, doc_id, filename, global_ids)
    return _write_blocks(blocks, doc_id, filename, source=source)


def _write_blocks(blocks: List[DocBlock], doc_id: str, filename: str, source: str = "upload") -> dict:
    embedder = get_embedder()
    vectors = embedder.embed_texts([b.text for b in blocks])
    for b, v in zip(blocks, vectors):
        b.embedding = v

    # 【优化点】vector_store/keyword_store 现为进程级单例（见 vector_store.py 注释），
    # create_collection()/create_index() 内部已做"仅首次真正建表/建索引"的短路优化，
    # 此处调用不再产生每次写入都重复的远程存在性检查开销。
    vector_store = get_vector_store()
    vector_store.create_collection(dim=embedder.get_dim())
    vector_store.upsert(blocks)

    keyword_store = get_keyword_store()
    keyword_store.create_index()
    keyword_store.upsert(blocks)

    registry = get_registry()
    meta = registry.register(
        doc_id=doc_id, filename=filename, num_chunks=len(blocks),
        chunk_ids=[b.global_chunk_idx for b in blocks], source=source,
    )
    return meta


def delete_document(doc_id: str) -> bool:
    """从向量库、关键词库、登记表中同时删除一个文档的全部块。"""
    registry = get_registry()
    if registry.get(doc_id) is None:
        return False
    get_vector_store().delete_by_doc_id(doc_id)
    get_keyword_store().delete_by_doc_id(doc_id)
    registry.delete(doc_id)
    return True


def list_documents() -> List[dict]:
    return get_registry().list_documents()


def get_stats() -> dict:
    return {
        "num_documents": get_registry().count_docs(),
        "num_vector_chunks": get_vector_store().count(),
        "num_keyword_chunks": get_keyword_store().count(),
        "vector_backend": RAG_CONFIG["vector_backend"],
        "keyword_backend": RAG_CONFIG["keyword_backend"],
        "embed_backend": RAG_CONFIG["embed_backend"],
        "rerank_backend": RAG_CONFIG["rerank_backend"],
        "generation_backend": RAG_CONFIG["generation_backend"],
    }
