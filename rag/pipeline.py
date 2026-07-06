# -*- coding: utf-8 -*-
"""全流程编排入口（对齐 TODO.md T5）。

    query 理解 → query 重写（可选）→ 检索器选择 → 混合检索（向量 + 关键词）→
    融合去重 → Reranker 精排 → Top-K

以及在此基础上叠加生成融合，提供端到端问答 `answer()`。全链路带计时日志，
任一检索路失败时仍能基于另一路结果返回（失败降级），并通过
`rag/observability/` 记录延迟与后端使用情况，供监控/告警使用。
"""
import time
from typing import List, Optional, Tuple

from config.config_loader import logger
from rag.config import RAG_CONFIG
from rag.retrieval import hybrid_search, query_understanding, retriever_selection
from rag.retrieval.query_rewrite import rewrite_query
from rag.retrieval.rerank import rerank
from rag.generation.llm_generation import generate_answer
from rag.observability import monitoring
from rag.observability.logging import log_event
from rag.observability.tracing import Trace
from rag.schema import DocBlock


def _retrieve_blocks(
    query: str,
    dialogue: Optional[list] = None,
    top_k: Optional[int] = None,
) -> Tuple[str, List[DocBlock]]:
    """检索链路的共享实现：query 理解 → query 重写 → 检索器选择 → 混合检索 → 精排。

    提取为共享内部函数，`retrieve()` 与 `answer()` 均基于它实现，也方便 API 层
    （`rag/api/routers/chat.py` 的流式接口）在生成阶段开始前单独复用检索结果、
    提前推送引用信息。全链路任一环节失败均捕获异常并降级（详见各 span 内的
    try/except），不会导致整条链路中断；异常均通过 `monitoring.record_error`
    计入错误率，供 `observability/alerting.py` 告警使用。

    Returns:
        (rewritten_query, reranked_blocks)
    """
    trace = Trace(f"retrieve(query={query[:40]!r})")

    hints = query_understanding.analyze_query(query)

    with trace.span("rewrite"):
        rewritten_query = rewrite_query(query, dialogue)

    backends = retriever_selection.select_retrievers(rewritten_query, query_hints=hints)
    recall_k = RAG_CONFIG["top_k_recall"]

    milvus_results = []
    with trace.span("vector"):
        if "vector" in backends:
            try:
                milvus_results = hybrid_search.vector_search(rewritten_query, recall_k)
            except Exception as e:
                logger.warning(f"⚠️ 向量检索异常: {e}")
                monitoring.record_error("vector", str(e))

    es_results = []
    with trace.span("keyword"):
        if "keyword" in backends:
            try:
                es_results = hybrid_search.keyword_search(rewritten_query, recall_k)
            except Exception as e:
                logger.warning(f"⚠️ 关键词检索异常: {e}")
                monitoring.record_error("keyword", str(e))

    with trace.span("fusion"):
        try:
            fused = hybrid_search.fuse([milvus_results, es_results])
        except Exception as e:
            # 融合/去重阶段异常时，降级为"双路结果简单拼接"，但仍需调用
            # `hybrid_search.deduplicate` 去重——此前直接拼接不去重，若同一块
            # 同时被向量与关键词召回会重复出现在最终结果/引用中（审查报告 M9）。
            logger.warning(f"⚠️ 融合去重异常，降级为原始结果拼接（仍执行去重）: {e}")
            monitoring.record_error("fusion", str(e))
            try:
                fused = hybrid_search.deduplicate(milvus_results + es_results)
            except Exception as dedup_err:
                logger.warning(f"⚠️ 降级路径去重仍失败，返回未去重拼接结果: {dedup_err}")
                fused = milvus_results + es_results

    final_top_k = top_k or RAG_CONFIG["top_k_final"]
    with trace.span("rerank"):
        try:
            reranked = rerank(rewritten_query, fused, top_k=final_top_k)
        except Exception as e:
            # 精排异常时降级为"融合结果按原分数截断"，保证仍有可用结果返回。
            logger.warning(f"⚠️ 精排异常，降级为融合结果直接截断: {e}")
            monitoring.record_error("rerank", str(e))
            reranked = fused[:final_top_k]

    logger.info(
        f"🔍 {trace.summary()} "
        f"命中: milvus={len(milvus_results)} es={len(es_results)} "
        f"融合后={len(fused)} 精排后={len(reranked)}"
    )
    # 【日志完善】此前仅有上面这条人类可读的 emoji 日志，`observability/logging.py`
    # docstring 中承诺的 `rag.retrieve` 结构化事件实际从未被调用——补充调用以便
    # 后续用日志分析工具（grep `event=rag.retrieve` / ELK 等）按字段过滤/统计，
    # 而不必解析非结构化的自然语言日志文本。
    log_event(
        "rag.retrieve", query=query,
        rewritten_query=rewritten_query if rewritten_query != query else None,
        backends=backends, num_vector=len(milvus_results), num_keyword=len(es_results),
        num_fused=len(fused), num_final=len(reranked),
        spans=trace.spans, total_ms=trace.total_ms,
    )
    monitoring.record_retrieval(trace.as_dict(), num_results=len(reranked))
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

    generation_ok = True
    try:
        gen_result = generate_answer(rewritten_query, reranked, dialogue=dialogue)
    except Exception as e:
        # generate_answer 内部已对 vLLM 调用失败做了降级处理，此处捕获的是更
        # 意外的异常（如引用构建/上下文组装出现未预期的 bug），降级为明确的
        # 错误提示而不是让 answer() 直接抛异常中断（检索结果仍已产出，不应
        # 因生成阶段的意外故障而丢弃）。
        logger.warning(f"⚠️ 生成阶段异常，降级为错误提示: {e}")
        monitoring.record_error("generation", str(e))
        generation_ok = False
        gen_result = {
            "answer": "抱歉，生成回答时发生了内部错误，请稍后重试或联系管理员。",
            "citations": [],
            "backend_used": "error",
        }

    latency_ms = 1000 * (time.time() - t0)
    logger.info(f"💬 answer(query={query[:40]!r}) 端到端耗时: {latency_ms:.0f}ms")
    # 【日志完善】补充结构化 `rag.answer` 事件（同 `rag.retrieve`），记录生成后端、
    # 引用数量与端到端耗时，供后续回溯"某次问答具体用了哪个生成后端/命中了
    # 多少引用/耗时是否异常"。
    log_event(
        "rag.answer", query=query, backend_used=gen_result["backend_used"],
        num_citations=len(gen_result["citations"]), num_contexts=len(reranked),
        latency_ms=round(latency_ms, 2),
    )
    # 【修复 L15】generation_ok=False 时（生成阶段意外异常）单独计入
    # monitoring 的 generation_error 统计，不再与 backend_usage 混在一起，
    # 避免"error"这个占位字符串污染真实的后端使用率分布。
    monitoring.record_answer(latency_ms=latency_ms, backend_used=gen_result["backend_used"], generation_ok=generation_ok)

    return {
        "query": query,
        "rewritten_query": rewritten_query if rewritten_query != query else None,
        "answer": gen_result["answer"],
        "citations": gen_result["citations"],
        "backend_used": gen_result["backend_used"],
        "contexts": [b.to_dict(with_embedding=False) for b in reranked],
    }
