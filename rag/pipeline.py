# -*- coding: utf-8 -*-
"""全流程编排入口（对齐 TODO.md T5）。

    query 重写（可选）→ 双模检索（向量 + 关键词） → 融合去重 → Reranker 精排 → Top-K

以及在此基础上叠加生成融合，提供端到端问答 `answer()`。全链路带计时日志，
任一检索路失败时仍能基于另一路结果返回（失败降级）。
"""
import time
from typing import List, Optional, Tuple

from config.config_loader import logger
from rag.config import RAG_CONFIG
from rag.retrieval import milvus_search, es_search
from rag.retrieval.fusion import fuse
from rag.retrieval.reranker import rerank
from rag.generation.generator import generate_answer
from rag.schema import DocBlock


def _rewrite_query(query: str, dialogue: Optional[list]) -> str:
    """多轮对话指代补全（可选，复用 `process/utils/llm_api.rewrite_query_vllm`）。

    重写失败或未配置多轮历史时，原样返回 query，不影响主链路。
    """
    if not dialogue:
        return query
    try:
        from rag.indexing._process_compat import _PROCESS_DIR, _PROCESS_SRC_DIR  # noqa: F401  确保 sys.path 已注入
        from utils.llm_api import rewrite_query_vllm
        return rewrite_query_vllm(dialogue, query)
    except Exception:
        return query


def _retrieve_blocks(
    query: str,
    dialogue: Optional[list] = None,
    top_k: Optional[int] = None,
) -> Tuple[str, List[DocBlock]]:
    """检索链路的共享实现：query 重写 → 双模检索 → 融合去重 → 精排。

    【优化点】原 `retrieve()` 与 `answer()` 各自完整复制了一份几乎相同的检索
    编排代码（query 重写 → 双模检索 → 融合 → 精排 → 计时日志），任何环节调整
    都需要同步修改两处、容易遗漏。现提取为共享内部函数，两者均基于它实现，
    也方便 API 层（`rag/api/routers/chat.py` 的流式接口）在生成阶段开始前
    单独复用检索结果、提前推送引用信息。

    Returns:
        (rewritten_query, reranked_blocks)
    """
    t0 = time.time()
    rewritten_query = _rewrite_query(query, dialogue)
    t1 = time.time()

    recall_k = RAG_CONFIG["top_k_recall"]
    milvus_results, es_results = [], []
    try:
        milvus_results = milvus_search.search(rewritten_query, recall_k)
    except Exception as e:
        logger.warning(f"⚠️ 向量检索异常: {e}")
    t2 = time.time()

    try:
        es_results = es_search.search(rewritten_query, recall_k)
    except Exception as e:
        logger.warning(f"⚠️ 关键词检索异常: {e}")
    t3 = time.time()

    fused = fuse([milvus_results, es_results])
    t4 = time.time()

    final_top_k = top_k or RAG_CONFIG["top_k_final"]
    reranked = rerank(rewritten_query, fused, top_k=final_top_k)
    t5 = time.time()

    logger.info(
        f"🔍 retrieve(query={query[:40]!r}) 耗时: "
        f"重写={1000*(t1-t0):.0f}ms 向量={1000*(t2-t1):.0f}ms "
        f"关键词={1000*(t3-t2):.0f}ms 融合={1000*(t4-t3):.0f}ms "
        f"精排={1000*(t5-t4):.0f}ms 总计={1000*(t5-t0):.0f}ms "
        f"命中: milvus={len(milvus_results)} es={len(es_results)} "
        f"融合后={len(fused)} 精排后={len(reranked)}"
    )
    return rewritten_query, reranked


def retrieve(
    query: str,
    dialogue: Optional[list] = None,
    top_k: Optional[int] = None,
) -> List[dict]:
    """检索入口：返回精排后的 Top-K 文档块（每条含 text/html_content/page_url/block_path/score）。

    Args:
        query: 用户查询
        dialogue: 多轮历史，用于 query 重写（可选）
        top_k: 覆盖默认 `RAG_CONFIG['top_k_final']`

    Returns:
        List[dict]，长度 <= top_k；任一环节失败均优雅降级，不抛异常中断链路。
    """
    _, reranked = _retrieve_blocks(query, dialogue, top_k)
    return [b.to_dict(with_embedding=False) for b in reranked]


def retrieve_context(
    query: str,
    dialogue: Optional[list] = None,
    top_k: Optional[int] = None,
) -> Tuple[str, List[DocBlock]]:
    """检索入口（返回 DocBlock 而非 dict，供需要自行调用生成模块的场景使用，
    如 `rag/api/routers/chat.py` 的 SSE 流式问答——先拿到上下文构建 citations
    并提前推送，再单独调用 `generate_answer()`，从而缩短用户感知的首字节延迟）。

    Returns:
        (rewritten_query, reranked_blocks)
    """
    return _retrieve_blocks(query, dialogue, top_k)


def answer(
    query: str,
    dialogue: Optional[list] = None,
    top_k: Optional[int] = None,
) -> dict:
    """端到端问答：检索 + 生成融合。

    Returns:
        {"query": str, "answer": str, "citations": [...], "backend_used": str,
         "contexts": [...]}
    """
    t0 = time.time()
    rewritten_query, reranked = _retrieve_blocks(query, dialogue, top_k)
    gen_result = generate_answer(rewritten_query, reranked, dialogue=dialogue)

    logger.info(f"💬 answer(query={query[:40]!r}) 端到端耗时: {1000*(time.time()-t0):.0f}ms")

    return {
        "query": query,
        "rewritten_query": rewritten_query if rewritten_query != query else None,
        "answer": gen_result["answer"],
        "citations": gen_result["citations"],
        "backend_used": gen_result["backend_used"],
        "contexts": [b.to_dict(with_embedding=False) for b in reranked],
    }
