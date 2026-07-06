# -*- coding: utf-8 -*-
"""Reranker 精排（原 reranker.py，重命名以对齐新的模块命名规范）。

`rerank(query, blocks, top_k) -> List[DocBlock]`：

    - backend="tei":   对接 `model/inference/tei_client.py` 的 TEI Reranker 服务
                        （交叉编码精排）。服务不可用时降级为**跳过精排**，
                        直接返回原融合结果（截断到 top_k）并告警，保证不崩溃。
    - backend="local": 用 Embedder 的余弦相似度作为精排代理分数（无需外部服务）。

输入文档拼接策略：`title + summary + text`，并截断到 Reranker 可接受长度。
"""
from typing import List, Optional

from config.config_loader import logger
from rag.config import RAG_CONFIG
from rag.schema import DocBlock


def _build_doc_text(block: DocBlock) -> str:
    parts = [block.title, block.summary, block.text]
    text = "\n".join(p for p in parts if p)
    return text[:RAG_CONFIG["rerank_doc_max_chars"]]


def rerank(query: str, blocks: List[DocBlock], top_k: int = None, backend: Optional[str] = None) -> List[DocBlock]:
    """对候选文档块做精排。

    Args:
        query: 查询文本
        blocks: 待精排的候选块（通常是融合去重后的结果）
        top_k: 精排后返回条数，默认取 `RAG_CONFIG['top_k_final']`
        backend: "tei" / "local"，默认取 `RAG_CONFIG['rerank_backend']`

    Returns:
        按精排分数降序排列的 Top-K DocBlock 列表。
    """
    top_k = top_k or RAG_CONFIG["top_k_final"]
    backend = backend or RAG_CONFIG["rerank_backend"]
    query = (query or "").strip()
    if not blocks or not query:
        return blocks[:top_k]

    if backend == "tei":
        reranked = _rerank_with_tei(query, blocks)
        if reranked is None:
            logger.warning("⚠️ TEI Reranker 不可用，跳过精排，直接返回融合结果")
            return blocks[:top_k]
        return reranked[:top_k]

    return _rerank_with_local(query, blocks)[:top_k]


def _rerank_with_tei(query: str, blocks: List[DocBlock]) -> Optional[List[DocBlock]]:
    try:
        from model.inference.tei_client import get_tei_client
        client = get_tei_client()
        if not client.health_check("rerank"):
            logger.warning("⚠️ TEI Reranker health_check 未通过")
            return None
        texts = [_build_doc_text(b) for b in blocks]
        scores = client.rerank_scores(query, texts)
    except Exception as e:
        logger.warning(f"⚠️ TEI Reranker 调用失败: {e}")
        return None

    for b, s in zip(blocks, scores):
        b.score = float(s)
        b.source_retriever = "reranked"
    return sorted(blocks, key=lambda b: -b.score)


def _rerank_with_local(query: str, blocks: List[DocBlock]) -> List[DocBlock]:
    import numpy as np

    from rag.indexing.embedding import get_embedder

    embedder = get_embedder()
    query_vec = np.array(embedder.embed_query(query), dtype=np.float32)
    texts = [_build_doc_text(b) for b in blocks]
    doc_vecs = np.array(embedder.embed_texts(texts), dtype=np.float32)

    q_norm = np.linalg.norm(query_vec) or 1e-12
    doc_norms = np.linalg.norm(doc_vecs, axis=1)
    doc_norms[doc_norms == 0] = 1e-12
    sims = (doc_vecs @ query_vec) / (doc_norms * q_norm)

    for b, s in zip(blocks, sims):
        b.score = float(s)
        b.source_retriever = "reranked"
    return sorted(blocks, key=lambda b: -b.score)
