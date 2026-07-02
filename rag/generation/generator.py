# -*- coding: utf-8 -*-
"""生成融合（Generation）：基于检索上下文生成最终答案。

后端可切换：
    - "vllm":  调用 `config/config.json` 的 `vllm_api_url`（OpenAI 兼容 Chat
               接口），做 LLM 生成式回答，支持 grounding + 引用标注。
    - "local": 无 LLM 服务时的抽取式兜底——直接摘录检索到的最相关片段并附
               来源引用，保证服务在无 GPU/LLM 部署时依然可用（out-of-the-box）。

统一返回结构：
    {"answer": str, "citations": [{"page_url", "block_path", "score"}, ...], "backend_used": str}
"""
from typing import List, Optional

from config.config_loader import logger
from rag.config import RAG_CONFIG
from rag.schema import DocBlock

SYSTEM_PROMPT = (
    "你是一个严谨的企业知识库客服助手。请仅根据下面提供的“检索上下文”回答用户问题，"
    "不要编造不存在的信息；如果上下文不足以回答，请明确说明“知识库中未找到相关信息”。"
    "回答末尾请用 [数字] 标注引用的上下文编号。"
)


def _build_context_text(contexts: List[DocBlock], max_chars: int) -> str:
    """将检索上下文拼接为送入 LLM 的文本，并限制在 max_chars 以内。

    【Bug 修复】原实现遇到超长单条上下文时直接整体跳过（`if total + len(piece)
    > max_chars: break`，且 break 发生在 append 之前），若 Top-1 上下文本身
    长度就超过 max_chars，会导致最终拼接结果为空字符串——但 `citations` 仍会
    包含该条引用，产生"回答未见任何上下文却标注引用"的逻辑错误。
    现改为：预算不足时对当前片段**截断**以填满剩余空间，而非整体丢弃，
    保证排序在前（更相关）的上下文始终至少被部分保留。
    """
    parts = []
    remaining = max_chars
    for i, c in enumerate(contexts, start=1):
        if remaining <= 0:
            break
        piece = f"[{i}] {c.title or c.page_name}\n{c.text}\n"
        if len(piece) > remaining:
            piece = piece[:remaining] + "...\n"
        parts.append(piece)
        remaining -= len(piece)
    return "\n".join(parts)


def build_citations(contexts: List[DocBlock]) -> List[dict]:
    """根据上下文构建引用列表（公开函数，供 API 层在流式响应中提前推送引用）。

    【优化点】原为私有函数 `_citations`，改为公开导出：`rag/api/routers/chat.py`
    的 SSE 流式接口需要在生成阶段开始前就先推送 `citations` 事件（见该文件
    `_gen()` 的说明），复用同一份引用构建逻辑可避免两处实现不一致。
    """
    return [
        {"index": i + 1, "page_url": c.page_url, "block_path": c.block_path,
         "title": c.title, "score": round(c.score, 4)}
        for i, c in enumerate(contexts)
    ]


def generate_answer(
    query: str,
    contexts: List[DocBlock],
    dialogue: Optional[list] = None,
    backend: Optional[str] = None,
) -> dict:
    """基于检索上下文生成回答。

    Args:
        query: 用户问题（已重写/补全指代）
        contexts: 精排后的 Top-K DocBlock 上下文
        dialogue: 多轮历史 [{"speaker": "user"/"bot", "text": ...}, ...]，可选
        backend: "vllm" / "local"，默认取 `RAG_CONFIG['generation_backend']`

    Returns:
        {"answer": str, "citations": [...], "backend_used": str}
    """
    backend = backend or RAG_CONFIG["generation_backend"]
    query = (query or "").strip()

    if not contexts:
        return {
            "answer": "抱歉，知识库中未找到与该问题相关的信息，建议您换一种问法或转人工客服。",
            "citations": [],
            "backend_used": "no_context",
        }

    if backend == "vllm":
        answer = _generate_with_vllm(query, contexts, dialogue)
        if answer is not None:
            return {"answer": answer, "citations": build_citations(contexts), "backend_used": "vllm"}
        logger.warning("⚠️ vLLM 生成服务不可用，降级为抽取式回答")

    return {
        "answer": _generate_local(query, contexts),
        "citations": build_citations(contexts),
        "backend_used": "local",
    }


def _generate_with_vllm(query: str, contexts: List[DocBlock], dialogue: Optional[list]) -> Optional[str]:
    try:
        import requests
        from config.config_loader import CONFIG

        context_text = _build_context_text(contexts, RAG_CONFIG["generation_max_context_chars"])
        history = ""
        if dialogue:
            for turn in dialogue:
                role = "用户" if turn.get("speaker") == "user" else "助手"
                history += f"{role}：{turn.get('text', '')}\n"

        user_content = f"【检索上下文】\n{context_text}\n\n【历史对话】\n{history}\n【当前问题】\n{query}"
        payload = {
            "model": CONFIG.get("llm_model", "glm"),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": RAG_CONFIG["generation_max_new_tokens"],
            "temperature": 0.3,
        }
        resp = requests.post(CONFIG["vllm_api_url"], json=payload, timeout=CONFIG.get("vllm_timeout", 60))
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


def _generate_local(query: str, contexts: List[DocBlock]) -> str:
    """抽取式兜底回答：摘录最相关片段并附引用编号，不依赖任何 LLM 服务。"""
    snippet_max_chars = RAG_CONFIG["generation_snippet_max_chars"]  # 【优化点】魔法数字收敛至 RAG_CONFIG
    lines = [f"根据知识库检索结果，为您找到以下 {len(contexts)} 条相关信息："]
    for i, c in enumerate(contexts, start=1):
        snippet = (c.text or c.summary or "").strip().replace("\n", " ")
        if len(snippet) > snippet_max_chars:
            snippet = snippet[:snippet_max_chars] + "..."
        lines.append(f"[{i}] {c.title or c.page_name}：{snippet}")
    lines.append("（以上内容直接摘录自知识库，未经 LLM 润色；如需更完整的智能问答，请部署生成模型服务）")
    return "\n".join(lines)
