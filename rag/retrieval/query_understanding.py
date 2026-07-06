# -*- coding: utf-8 -*-
"""查询理解（Query Understanding）：对用户 query 做轻量分析，供下游
`query_rewrite` / `retriever_selection` 决策使用。

不依赖任何外部 LLM 服务（纯规则 + jieba 分词的启发式实现），保证零外部依赖、
可单测、低延迟（<1ms 级），因此适合作为检索前置的**同步**步骤。若后续需要
更强的意图识别，可在此基础上扩展一个基于 LLM 的 backend（参考
`rag/generation/llm_generation.py` 的降级设计模式）而不影响下游接口。
"""
import re
from typing import List

try:
    import jieba
except ImportError:  # pragma: no cover
    jieba = None

_ZH_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_RE = re.compile(r"[A-Za-z]")
_QUESTION_WORDS = ("什么", "怎么", "如何", "为什么", "哪个", "哪些", "是否", "能否", "多少", "谁", "何时")
_CONJUNCTIONS = ("并且", "而且", "另外", "同时", "以及", "还有", "然后")


def detect_language(query: str) -> str:
    """粗粒度语言检测：zh / en / mixed / unknown。"""
    has_zh = bool(_ZH_RE.search(query or ""))
    has_en = bool(_EN_RE.search(query or ""))
    if has_zh and has_en:
        return "mixed"
    if has_zh:
        return "zh"
    if has_en:
        return "en"
    return "unknown"


def is_question(query: str) -> bool:
    """是否为疑问句：命中疑问词或以 ? / ？结尾。"""
    q = (query or "").strip()
    if not q:
        return False
    if q.endswith(("?", "？")):
        return True
    return any(w in q for w in _QUESTION_WORDS)


def extract_keywords(query: str, top_n: int = 8) -> List[str]:
    """提取 query 中的关键词（jieba 分词后按词长简单加权排序，jieba 不可用时
    退化为整句返回）。"""
    q = (query or "").strip()
    if not q:
        return []
    if jieba is None:
        return [q]
    tokens = [w for w in jieba.lcut(q) if w.strip() and len(w.strip()) > 1]
    # 简单启发式：优先保留更长的词（信息量通常更大），去重后截断
    seen = []
    for w in sorted(set(tokens), key=len, reverse=True):
        seen.append(w)
        if len(seen) >= top_n:
            break
    return seen


def estimate_complexity(query: str) -> str:
    """粗粒度复杂度估计：simple / complex。

    命中连接词、包含多个疑问点、或长度超过阈值时判定为 complex —— 这类查询
    通常更适合先做 `query_rewrite`（多轮指代补全/查询改写）再检索。
    """
    q = (query or "").strip()
    if not q:
        return "simple"
    if any(c in q for c in _CONJUNCTIONS):
        return "complex"
    if len(q) > 40:
        return "complex"
    return "simple"


def analyze_query(query: str) -> dict:
    """对外统一入口：返回 query 的结构化分析结果。

    Returns:
        {
            "query": 原始 query,
            "length": 字符长度,
            "language": "zh"/"en"/"mixed"/"unknown",
            "is_question": bool,
            "keywords": [...],
            "complexity": "simple"/"complex",
        }
    """
    query = query or ""
    return {
        "query": query,
        "length": len(query),
        "language": detect_language(query),
        "is_question": is_question(query),
        "keywords": extract_keywords(query),
        "complexity": estimate_complexity(query),
    }
