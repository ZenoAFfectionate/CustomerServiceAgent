# -*- coding: utf-8 -*-
"""Reranker 精排（对齐 TODO.md T4）。

`rerank(query, blocks, top_k) -> List[DocBlock]`：

    - backend="tei":   对接 `model/inference/tei_client.py` 的 TEI Reranker 服务
                        （交叉编码精排）。服务不可用时按 DoD 要求**降级为跳过精排**，
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
    return text[:RAG_CONFIG["rerank_doc_max_chars"]]  # 【优化点】截断长度收敛至 RAG_CONFIG，可通过环境变量调整


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
            # DoD：无 Reranker 服务时降级为跳过精排，直接返回融合结果并告警
            logger.warning("⚠️ TEI Reranker 不可用，跳过精排，直接返回融合结果")
            return blocks[:top_k]
        return reranked[:top_k]

    return _rerank_with_local(query, blocks)[:top_k]


def _rerank_with_tei(query: str, blocks: List[DocBlock]) -> Optional[List[DocBlock]]:
    try:
        from model.inference.tei_client import get_tei_client
        client = get_tei_client()
        if not client.health_check("rerank"):
            return None
        texts = [_build_doc_text(b) for b in blocks]
        scores = client.rerank_scores(query, texts)
    except Exception:
        return None

    for b, s in zip(blocks, scores):
        b.score = float(s)
        b.source_retriever = "reranked"
    return sorted(blocks, key=lambda b: -b.score)


def _rerank_with_local(query: str, blocks: List[DocBlock]) -> List[DocBlock]:
    # 【优化点】移除了从未使用的 `LocalVectorStore` 导入（原代码靠 noqa 掩盖死代码）
    import numpy as np

    from rag.indexing.embedder import get_embedder

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
