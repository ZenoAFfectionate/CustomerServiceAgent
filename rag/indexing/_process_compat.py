# -*- coding: utf-8 -*-
"""process/ 复用桥接层。

`process/src/text_process.py` 使用顶层 import（`from utils.llm_api import ...`），
要求 `process/` 与 `process/src/` 同时位于 `sys.path`（与 `tests/conftest.py`
的约定一致）。本模块统一处理该 sys.path 注入，并对导入失败（如 process/ 依赖
缺失）做优雅降级，返回等价的本地兜底实现，避免 rag/ 因 process/ 不可用而崩溃。
"""
import os
import sys

from config.config_loader import logger

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CURRENT_DIR))  # rag/indexing → rag → 项目根
_PROCESS_DIR = os.path.join(_PROJECT_ROOT, "process")
_PROCESS_SRC_DIR = os.path.join(_PROCESS_DIR, "src")

for _p in (_PROCESS_DIR, _PROCESS_SRC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 降级使用兜底实现只在首次发生时告警一次，避免高频调用路径（如每次检索融合都
# 会调用 get_deduplicate_fn）产生刷屏式重复日志。
_warned_fallbacks = set()


def _warn_fallback_once(name: str, reason: Exception) -> None:
    if name in _warned_fallbacks:
        return
    _warned_fallbacks.add(name)
    logger.warning(f"⚠️ process/ 的 {name} 不可用（{reason}），已降级为本地兜底实现")


def get_deduplicate_fn():
    """返回 `deduplicate_ranked_blocks_pal`；不可用时返回按 chunk_idx 去重的兜底实现。"""
    try:
        from text_process import deduplicate_ranked_blocks_pal
        return deduplicate_ranked_blocks_pal
    except Exception as e:
        _warn_fallback_once("deduplicate_ranked_blocks_pal", e)

        def _fallback_dedup(docs: list, threshold_content: float = 0.9, threshold_page_name: float = 0.6) -> list:
            seen = set()
            out = []
            for d in docs:
                key = d.get("global_chunk_idx", d.get("chunk_idx"))
                if key in seen:
                    continue
                seen.add(key)
                out.append(d)
            return out
        return _fallback_dedup


def get_build_optimal_jieba_query_fn():
    """返回 `build_optimal_jieba_query`；不可用时返回 None（调用方需自行降级为 multi_match）。"""
    try:
        from text_process import build_optimal_jieba_query
        return build_optimal_jieba_query
    except Exception as e:
        _warn_fallback_once("build_optimal_jieba_query", e)
        return None
