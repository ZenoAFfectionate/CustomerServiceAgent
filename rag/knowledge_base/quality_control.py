# -*- coding: utf-8 -*-
"""导入前质量检查（Quality Control）：在知识块写入索引之前做轻量质检，
提前暴露数据质量问题（而不是让脏数据静默进入知识库，污染检索结果）。

检查项：
    - 空文本率：`text` 字段为空/极短的块占比
    - 重复率：完全相同 `text` 内容的块占比（跨块级去重，不等价于
      `retrieval/hybrid_search.py` 的检索期近似去重，这里是导入期精确去重检测）
    - HTML 残留：`text` 字段中疑似未清洗干净的 HTML 标签

默认仅产出报告（`passed=True/False` + `warnings`），由调用方决定是否阻断
导入 —— 当前 `corpus_management.py` 默认只记录告警不阻断，避免因质检误判
影响正常导入流程；如需强制拦截，可在调用侧检查 `report["passed"]`。
"""
import re
from typing import List

_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(?:\s[^>]*)?>")

DEFAULT_MAX_EMPTY_RATIO = 0.2
DEFAULT_MAX_DUPLICATE_RATIO = 0.3
DEFAULT_MIN_AVG_LENGTH = 5


def check_blocks_quality(blocks: List[dict]) -> dict:
    """对一批待导入的知识块做质量检查，返回质检报告。

    Args:
        blocks: 待导入的知识块（dict 列表，至少含 `text` 字段）

    Returns:
        {
            "passed": bool,
            "total": int,
            "empty_ratio": float,
            "duplicate_ratio": float,
            "avg_length": float,
            "html_leftover_count": int,
            "warnings": [str, ...],
        }
    """
    total = len(blocks)
    if total == 0:
        return {
            "passed": False, "total": 0, "empty_ratio": 1.0, "duplicate_ratio": 0.0,
            "avg_length": 0.0, "html_leftover_count": 0, "warnings": ["空批次：blocks 为空"],
        }

    texts = [(b.get("text") or "").strip() for b in blocks]
    empty_count = sum(1 for t in texts if not t)
    non_empty_texts = [t for t in texts if t]
    avg_length = sum(len(t) for t in non_empty_texts) / len(non_empty_texts) if non_empty_texts else 0.0
    duplicate_count = len(non_empty_texts) - len(set(non_empty_texts))
    html_leftover_count = sum(1 for t in non_empty_texts if _HTML_TAG_RE.search(t))

    empty_ratio = round(empty_count / total, 4)
    duplicate_ratio = round(duplicate_count / len(non_empty_texts), 4) if non_empty_texts else 0.0

    warnings = []
    if empty_ratio > DEFAULT_MAX_EMPTY_RATIO:
        warnings.append(f"空文本占比过高: {empty_ratio:.1%} > {DEFAULT_MAX_EMPTY_RATIO:.0%}")
    if duplicate_ratio > DEFAULT_MAX_DUPLICATE_RATIO:
        warnings.append(f"重复文本占比过高: {duplicate_ratio:.1%} > {DEFAULT_MAX_DUPLICATE_RATIO:.0%}")
    if non_empty_texts and avg_length < DEFAULT_MIN_AVG_LENGTH:
        warnings.append(f"平均文本长度过短: {avg_length:.1f} 字符 < {DEFAULT_MIN_AVG_LENGTH}")
    if html_leftover_count > 0:
        warnings.append(f"检测到 {html_leftover_count} 条文本疑似残留未清洗的 HTML 标签")

    return {
        "passed": not warnings,
        "total": total,
        "empty_ratio": empty_ratio,
        "duplicate_ratio": duplicate_ratio,
        "avg_length": round(avg_length, 1),
        "html_leftover_count": html_leftover_count,
        "warnings": warnings,
    }
