# -*- coding: utf-8 -*-
"""查询重写（Query Rewrite）：多轮对话指代补全 / 查询改写。

原逻辑内嵌于 `pipeline.py` 的私有函数 `_rewrite_query`，现独立为 `retrieval/`
子模块，与 `query_understanding.py` 并列为"检索前置处理"的两个独立能力
（一个负责"理解"，一个负责"改写"），符合单一职责原则。

复用 `process/utils/llm_api.rewrite_query_vllm` 做多轮指代消解（如"它""这个"
→ 补全为完整实体）；未配置多轮历史或重写失败时原样返回 query，不影响主链路。
"""
from typing import Optional

from config.config_loader import logger


def rewrite_query(query: str, dialogue: Optional[list]) -> str:
    """基于多轮历史对话补全 query 中的指代/省略成分。

    Args:
        query: 当前轮用户输入
        dialogue: 历史对话 [{"speaker": "user"/"bot", "text": ...}, ...]，为空则直接返回原 query

    Returns:
        重写后的 query；未配置历史或重写失败时返回原始 query（优雅降级，不抛异常）。
    """
    if not dialogue:
        return query
    try:
        # 确保 process/ 与 process/src/ 已注入 sys.path（与 indexing/_process_compat.py 同一约定）
        from rag.indexing._process_compat import _PROCESS_DIR, _PROCESS_SRC_DIR  # noqa: F401
        from utils.llm_api import rewrite_query_vllm
        return rewrite_query_vllm(dialogue, query)
    except Exception as e:
        # 记录失败原因（如 LLM 服务不可用/API Key 未配置/网络超时），保留可追踪的
        # 排查线索，同时仍原样返回 query 保证主检索链路不受影响。
        logger.warning(f"⚠️ 多轮 query 重写失败，回退为原始 query: {e}")
        return query
