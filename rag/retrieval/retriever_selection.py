# -*- coding: utf-8 -*-
"""检索器选择（Retriever Selection）：决定一次检索应该跑哪些召回路径
（向量 / 关键词 / 双路），以及当前生效的检索后端配置。

默认策略是"双路全开 + 融合"（对齐 TODO.md 的双模检索设计），但暴露显式的
选择逻辑，方便后续按 `query_understanding` 的分析结果做更精细的路由
（例如：极短的纯关键词查询可跳过向量检索以降低延迟；长语义问句可降低
关键词路权重）。当前实现保守 —— 仅在明确判断"某一路检索无意义"时才跳过，
避免因误判而漏检索，召回率优先于延迟。
"""
from typing import List, Optional

from rag.config import RAG_CONFIG

VECTOR = "vector"
KEYWORD = "keyword"
ALL_BACKENDS = (VECTOR, KEYWORD)


def select_retrievers(query: str, query_hints: Optional[dict] = None) -> List[str]:
    """按 query（及可选的 `query_understanding.analyze_query` 结果）选择需要
    执行的检索路径。

    Args:
        query: 查询文本
        query_hints: `query_understanding.analyze_query(query)` 的返回值（可选，
            未提供时仅按 query 本身做保守判断）

    Returns:
        需要执行的检索路径子集，取值来自 `ALL_BACKENDS`；当前默认双路全开，
        仅在 query 为空时返回空列表（上游应直接短路，不发起任何检索）。
    """
    query = (query or "").strip()
    if not query:
        return []

    # 极短查询（<=1 个字符）对向量语义检索意义有限，但仍保留双路以保证召回率——
    # 这里预留 hints 参数是为了未来引入更可靠的路由信号时无需改动调用方签名。
    if query_hints is not None and query_hints.get("length", len(query)) == 0:
        return []

    return list(ALL_BACKENDS)


def get_active_backends() -> dict:
    """返回当前生效的检索后端配置（供 `observability`/`integration.deployment`
    等模块做健康检查/展示使用）。"""
    return {
        "vector_backend": RAG_CONFIG["vector_backend"],
        "keyword_backend": RAG_CONFIG["keyword_backend"],
        "embed_backend": RAG_CONFIG["embed_backend"],
        "rerank_backend": RAG_CONFIG["rerank_backend"],
    }
