# -*- coding: utf-8 -*-
"""索引构建编排入口（Index Builder，原 indexer.py，重命名以对齐新的模块命名规范）：
文档解析 → 分块 → 向量化 → 写入向量库/关键词库 → 登记。

供 `rag/knowledge_base/corpus_management.py`、API 层
（`rag/api/routers/documents.py`）与 `scripts/build_index.py` 共同调用，
是"文档解析、文本分块、embedding、向量存储"四个环节的编排层。
"""
import os
from typing import List, Optional

from config.config_loader import logger
from rag.config import RAG_CONFIG
from rag.indexing.chunking import chunk_text
from rag.indexing.embedding import get_embedder
from rag.indexing.document_loader import parse_document, ParseError
from rag.indexing.metadata import get_registry
from rag.indexing.store import get_keyword_store, get_vector_store
from rag.schema import DocBlock


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
        # 缺省或为空时按数组位置回填 chunk_idx，避免同一文档内多条块的
        # chunk_idx 全部重复为 0，干扰引用溯源与去重。
        if known.get("chunk_idx") is None:
            known["chunk_idx"] = i
        blocks.append(DocBlock(**known))
    return blocks


def _filter_non_empty_blocks(raw_blocks: List[dict]) -> List[dict]:
    """过滤 text 字段为空的知识块：空文本块会被 Embedder 编码为零向量并写入
    向量库/关键词库，不但检索无意义，还会污染 `num_vector_chunks` 等统计数据。
    """
    return [b for b in raw_blocks if isinstance(b, dict) and (b.get("text") or "").strip()]


def filter_non_empty_blocks(raw_blocks: List[dict]) -> List[dict]:
    """`_filter_non_empty_blocks` 的公开导出别名。

    供 `rag/knowledge_base/corpus_management.py` 复用同一份过滤逻辑计算
    版本哈希——此前 `corpus_management.ingest_blocks` 对调用方传入的
    **原始** `blocks` 计算哈希，而本模块入库前会先过滤空文本块，二者不一致
    导致版本哈希无法代表"实际入库内容"（审查报告 L13）。通过导出该函数，
    调用方可以对"过滤后、与实际入库一致"的内容计算哈希，而不是各自维护
    一份可能漂移的过滤逻辑。
    """
    return _filter_non_empty_blocks(raw_blocks)


def _check_block_count_limit(n: int) -> None:
    limit = RAG_CONFIG["max_blocks_per_ingest"]
    if n > limit:
        raise ValueError(f"单次导入的知识块数量（{n}）超过限制（{limit}），请分批导入")


def ingest_file(filename: str, raw: bytes, doc_id: Optional[str] = None, source: str = "upload") -> dict:
    """解析并索引一个上传文件，返回文档登记元信息。"""
    if not filename or not filename.strip():
        raise ValueError("文件名不能为空")

    ext = os.path.splitext(filename)[1].lower()
    # 【修复 L8】此前此处对 raw 再做一次 len(raw) > max_size 校验，但 API 层
    # （documents.py 的 _read_upload_within_limit）已对同一份 raw 做了分块读取
    # + 超限中止的校验，两处阈值须保持同步否则出现"API 放行但 index_builder
    # 拦截"的不一致。移除此处冗余校验，统一由 API 层负责大小限制。
    if ext not in RAG_CONFIG["allowed_upload_ext"]:
        raise ValueError(f"不支持的文件类型: {ext}，支持: {RAG_CONFIG['allowed_upload_ext']}")

    # ParseError（如 JSON 格式错误、PDF 解析失败等恶意/损坏文件场景）统一转换为
    # ValueError：语义上应等价于参数校验失败（422），而非未处理异常导致的 500。
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
        content_blocks = raw_blocks  # 过滤后的原始块 dict
    else:
        chunks = chunk_text(parsed, RAG_CONFIG["chunk_size"], RAG_CONFIG["chunk_overlap"])
        if not chunks:
            raise ValueError("文档解析后未产生任何有效文本块（内容为空或全部被过滤）")
        _check_block_count_limit(len(chunks))
        global_ids = registry.next_global_ids(len(chunks))
        blocks = _build_blocks_from_text(parsed, doc_id, filename, global_ids)
        content_blocks = [{"text": c} for c in chunks]

    meta = _write_blocks(blocks, doc_id, filename, source=source)
    # 【修复 L14】返回过滤后的原始块，供调用方计算与 ingest_blocks 路径
    # 一致的版本哈希（统一哈希空间）。
    meta["_content_blocks"] = content_blocks
    return meta


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
    meta = _write_blocks(blocks, doc_id, filename, source=source)
    # 【修复 L14】返回过滤后的原始块，供调用方计算版本哈希。
    meta["_content_blocks"] = filtered
    return meta


def _write_blocks(blocks: List[DocBlock], doc_id: str, filename: str, source: str = "upload") -> dict:
    embedder = get_embedder()
    vectors = embedder.embed_texts([b.text for b in blocks])
    for b, v in zip(blocks, vectors):
        b.embedding = v

    # vector_store/keyword_store 为进程级单例，create_collection()/create_index()
    # 内部已做"仅首次真正建表/建索引"的短路优化，此处调用不会产生重复的远程检查开销。
    vector_store = get_vector_store()
    vector_store.create_collection(dim=embedder.get_dim())
    vector_store.upsert(blocks)

    # 【修复 H3】写入分三步（向量库 → 关键词库 → 登记表），此前无 try/except，
    # 任一环节失败时已成功的环节不回滚，会形成"向量有/关键词无"或"向量+关键词
    # 有/registry 无"的孤儿数据，且 list_documents()/删除接口都看不到它们。
    # 现改为：后续环节失败时显式回滚已成功写入的环节，保证"要么全部生效、
    # 要么不留下任何残留数据"。
    keyword_store = get_keyword_store()
    try:
        keyword_store.create_index()
        keyword_store.upsert(blocks)
    except Exception as e:
        _rollback_stores(doc_id, vector=True, keyword=False)
        raise RuntimeError(f"写入关键词库失败，已回滚向量库中该文档的数据（doc_id={doc_id}）: {e}") from e

    registry = get_registry()
    try:
        meta = registry.register(
            doc_id=doc_id, filename=filename, num_chunks=len(blocks),
            chunk_ids=[b.global_chunk_idx for b in blocks], source=source,
        )
    except Exception as e:
        _rollback_stores(doc_id, vector=True, keyword=True)
        raise RuntimeError(f"登记文档元数据失败，已回滚向量库/关键词库中该文档的数据（doc_id={doc_id}）: {e}") from e
    return meta


def _rollback_stores(doc_id: str, vector: bool, keyword: bool) -> None:
    """回滚补偿：删除向量库/关键词库中已写入的某文档数据（供 `_write_blocks`
    在后续环节失败时调用，避免留下"部分写入成功"的孤儿数据）。回滚本身的
    异常只记录日志、不再抛出——避免"回滚失败"掩盖原始失败原因，也不应让
    调用方误以为回滚本身是主流程的一部分。
    """
    if vector:
        try:
            get_vector_store().delete_by_doc_id(doc_id)
        except Exception as rollback_err:
            logger.error(f"❌ 回滚向量库失败（doc_id={doc_id}，可能残留孤儿数据，需人工核查）: {rollback_err}")
    if keyword:
        try:
            get_keyword_store().delete_by_doc_id(doc_id)
        except Exception as rollback_err:
            logger.error(f"❌ 回滚关键词库失败（doc_id={doc_id}，可能残留孤儿数据，需人工核查）: {rollback_err}")


def delete_document(doc_id: str) -> bool:
    """从向量库、关键词库、登记表中同时删除一个文档的全部块。

    【修复 H3】此前逻辑为：先查 registry，若 registry 无该 doc_id 直接返回
    False（不触碰向量库/关键词库）。但 registry 与存储后端可能因此前的写入/
    删除部分失败而不同步——`False` 并不能保证向量库/关键词库中真的没有该
    doc_id 的残留数据，会误导 API 层的 404 判定（见 `documents.py`）。

    现改为：无论 registry 中是否存在该 doc_id，都无条件尝试删除向量库/
    关键词库中的残留数据（用于清理孤儿数据）；任一存储环节抛出真实异常时
    整体抛出 `RuntimeError`（不吞异常，交由上层感知为"删除未完全成功"，
    对应 API 层会被全局异常处理器转换为 500 而非误报 404）；只有当
    registry、向量库、关键词库均确认"本就没有这个文档"时才返回 False，
    真正实现"任一处存在残留即视为删除成功且返回 True"的语义。
    """
    registry = get_registry()
    existed_in_registry = registry.get(doc_id) is not None

    errors: List[str] = []
    vector_deleted = 0
    keyword_deleted = 0
    try:
        vector_deleted = get_vector_store().delete_by_doc_id(doc_id)
    except Exception as e:
        errors.append(f"向量库删除失败: {e}")

    try:
        keyword_deleted = get_keyword_store().delete_by_doc_id(doc_id)
    except Exception as e:
        errors.append(f"关键词库删除失败: {e}")

    registry_deleted = False
    if existed_in_registry:
        try:
            registry_deleted = registry.delete(doc_id)
        except Exception as e:
            errors.append(f"登记表删除失败: {e}")

    if errors:
        raise RuntimeError(f"删除文档 {doc_id} 时部分存储环节失败，可能残留不一致数据: {'; '.join(errors)}")

    return existed_in_registry or registry_deleted or bool(vector_deleted) or bool(keyword_deleted)


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
