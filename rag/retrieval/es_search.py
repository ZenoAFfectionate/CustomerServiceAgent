# -*- coding: utf-8 -*-
"""关键词检索（对齐 TODO.md T2）。

`search(query, top_k) -> List[DocBlock]`：jieba 分词后交给 `keyword_store`
（Elasticsearch + `build_optimal_jieba_query`，或本地 TF-IDF 降级实现）检索。
返回结果统一带 `score` 与 `source_retriever="es"`，按分数降序。
"""
from typing import List

from config.config_loader import logger
from rag.config import RAG_CONFIG
from rag.indexing.keyword_store import get_keyword_store
from rag.schema import DocBlock


def search(query: str, top_k: int = None) -> List[DocBlock]:
    """关键词检索。

    Args:
        query: 查询文本
        top_k: 返回条数，默认取 `RAG_CONFIG['top_k_recall']`

    Returns:
        按相关性降序排列的 DocBlock 列表（长度 <= top_k）。服务不可用/无数据时返回 []。
    """
    top_k = top_k or RAG_CONFIG["top_k_recall"]
    query = (query or "").strip()
    if not query:
        return []
    store = get_keyword_store()
    try:
        results = store.search(query, top_k)
    except Exception:
        logger.warning("⚠️ 关键词检索失败，返回空结果")
        return []
    for r in results:
        r.source_retriever = "es"
    return results
