# -*- coding: utf-8 -*-
"""向量检索（对齐 TODO.md T2）。

`search(query, top_k) -> List[DocBlock]`：query 经 Embedder 向量化后，
调用 `vector_store`（Milvus 或本地降级实现）做 ANN / 暴力余弦检索。
返回结果统一带 `score` 与 `source_retriever="milvus"`，按分数降序。
"""
from typing import List

from config.config_loader import logger
from rag.config import RAG_CONFIG
from rag.indexing.embedder import get_embedder
from rag.indexing.vector_store import get_vector_store
from rag.schema import DocBlock


def search(query: str, top_k: int = None) -> List[DocBlock]:
    """向量检索。

    Args:
        query: 查询文本
        top_k: 返回条数，默认取 `RAG_CONFIG['top_k_recall']`

    Returns:
        按相似度降序排列的 DocBlock 列表（长度 <= top_k）。服务不可用/无数据时返回 []。
    """
    top_k = top_k or RAG_CONFIG["top_k_recall"]
    query = (query or "").strip()
    if not query:
        return []
    embedder = get_embedder()
    query_vector = embedder.embed_query(query)

    store = get_vector_store()
    try:
        results = store.search(query_vector, top_k)
    except Exception:
        logger.warning("⚠️ 向量检索失败，返回空结果")
        return []
    for r in results:
        r.source_retriever = "milvus"
    return results
