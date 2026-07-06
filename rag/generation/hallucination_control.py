# -*- coding: utf-8 -*-
"""幻觉控制（Hallucination Control）：生成结果的事后校验与兜底提示。

不依赖额外的 LLM 校验调用（避免翻倍生成延迟/成本），采用两类轻量启发式：
    1. 引用有效性校验：解析回答中的 `[n]` 引用标记，检测是否存在越界引用
       （模型编造了不存在的引用编号）或完全没有引用（应有引用却没有）。
    2. 关联度（groundedness）评分：回答文本与检索上下文的词汇重叠率，
       重叠过低说明回答可能脱离了检索上下文"自由发挥"（潜在幻觉）。

仅对 `generation_backend="vllm"`（真实 LLM 生成）生效 —— `local`（抽取式）
后端的回答本身就是直接摘录自上下文，天然 100% grounded，无需校验。
校验结果仅记录告警日志并在关联度过低时附加提示语，不会阻断响应返回
（保证可用性优先于严格性，与 `rag/` 一贯的"优雅降级"设计一致）。
"""
import re
from typing import List

from rag.config import RAG_CONFIG
from rag.observability.logging import log_event
from rag.schema import DocBlock

_CITATION_RE = re.compile(r"\[(\d+)\]")
DEFAULT_MIN_GROUNDEDNESS = 0.12


def extract_cited_indices(answer_text: str) -> List[int]:
    """解析回答文本中的 `[n]` 引用标记，返回去重排序后的编号列表。"""
    return sorted({int(m) for m in _CITATION_RE.findall(answer_text or "")})


def check_citation_validity(answer_text: str, num_contexts: int) -> dict:
    """校验引用标记是否越界、是否存在引用。

    Returns:
        {"cited_indices": [...], "invalid_indices": [...], "has_citation": bool, "valid": bool}
    """
    cited = extract_cited_indices(answer_text)
    invalid = [i for i in cited if i < 1 or i > num_contexts]
    return {
        "cited_indices": cited,
        "invalid_indices": invalid,
        "has_citation": bool(cited),
        "valid": not invalid,
    }


def _simple_tokens(text: str) -> set:
    """轻量分词（复用 embedding.py 的分词策略，延迟导入避免循环依赖）。"""
    from rag.indexing.embedding import _tokenize
    return set(_tokenize(text or ""))


def groundedness_score(answer_text: str, contexts: List[DocBlock]) -> float:
    """计算回答文本与检索上下文的词汇重叠率（Jaccard 近似：
    |answer_tokens ∩ context_tokens| / |answer_tokens|）。

    取值范围 [0, 1]；越高说明回答内容越"贴合"检索上下文。这是一种廉价的
    代理指标，不能替代真正的事实核查，仅用于捕捉"完全脱离上下文编造"的
    明显异常场景。
    """
    answer_tokens = _simple_tokens(answer_text)
    if not answer_tokens:
        return 0.0
    context_tokens = set()
    for c in contexts:
        context_tokens |= _simple_tokens(c.text) | _simple_tokens(c.summary)
    if not context_tokens:
        return 0.0
    overlap = answer_tokens & context_tokens
    return round(len(overlap) / len(answer_tokens), 4)


def apply_guardrail(answer_text: str, contexts: List[DocBlock], backend_used: str) -> str:
    """对 LLM 生成结果做事后校验，返回（可能附加提示语的）回答文本。

    仅在 `backend_used == "vllm"` 且存在上下文时生效；异常/告警仅记录日志，
    不阻断返回，保证服务可用性。
    """
    if backend_used != "vllm" or not contexts or not answer_text:
        return answer_text

    validity = check_citation_validity(answer_text, len(contexts))
    score = groundedness_score(answer_text, contexts)
    warnings = []
    if not validity["has_citation"]:
        warnings.append("no_citation")
    # 【修复 L9】此前 `validity["valid"]` 字段计算后完全未被引用（死字段），
    # 而是重复判断 `invalid_indices` 是否非空——二者逻辑等价（`valid` 即
    # `not invalid_indices`），直接改用 `valid` 字段使其真正被消费，同时
    # 语义上更直观（"是否有效"而非"是否存在无效索引"）。
    if not validity["valid"]:
        warnings.append("invalid_citation_index")
    if score < DEFAULT_MIN_GROUNDEDNESS:
        warnings.append("low_groundedness")

    if warnings:
        log_event(
            "rag.hallucination_guard", level="warning",
            warnings=warnings, groundedness=score, backend=backend_used,
        )
    # 是否将提示语拼接进最终回答文本默认关闭（避免默认改变既有生成结果，
    # 影响下游展示/断言），可通过 RAG_CONFIG['hallucination_append_caveat'] 开启。
    if "low_groundedness" in warnings and RAG_CONFIG.get("hallucination_append_caveat", False):
        answer_text = answer_text.rstrip() + (
            "\n\n（提示：以上回答与知识库上下文的关联度较低，请结合引用来源自行核实。）"
        )
    return answer_text
