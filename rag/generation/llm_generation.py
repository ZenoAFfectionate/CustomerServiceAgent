# -*- coding: utf-8 -*-
"""LLM 生成（LLM Generation）：基于检索上下文调用生成模型产出最终回答，
以及无生成模型服务时的抽取式兜底实现。

后端可切换：
    - "vllm":  调用 `config/config.json` 的 `vllm_api_url`（OpenAI 兼容 Chat
               接口），做 LLM 生成式回答，支持 grounding + 引用标注。
    - "local": 无 LLM 服务时的抽取式兜底——直接摘录检索到的最相关片段并附
               来源引用，保证服务在无 GPU/LLM 部署时依然可用（out-of-the-box）。

统一返回结构：
    {"answer": str, "citations": [{"page_url", "block_path", "score"}, ...], "backend_used": str}
"""
import json
from typing import Generator, List, Optional, Tuple

from config.config_loader import logger
from rag.config import RAG_CONFIG
from rag.generation.citation import build_citations
from rag.generation.context_assembly import build_context_text
from rag.generation.hallucination_control import apply_guardrail
from rag.generation.prompt_template import SYSTEM_PROMPT, build_user_prompt
from rag.schema import DocBlock


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
        answer, cited_contexts = _generate_with_vllm(query, contexts, dialogue)
        if answer is not None:
            answer = apply_guardrail(answer, cited_contexts, backend_used="vllm")
            return {"answer": answer, "citations": build_citations(cited_contexts), "backend_used": "vllm"}
        logger.warning("⚠️ vLLM 生成服务不可用，降级为抽取式回答")

    return {
        "answer": _generate_local(query, contexts),
        "citations": build_citations(contexts),
        "backend_used": "local",
    }


def select_cited_contexts(
    contexts: List[DocBlock],
    backend: Optional[str] = None,
) -> List[DocBlock]:
    """返回引用编号应覆盖的上下文子集，供 API 层（如 SSE 流式接口需要在生成
    开始前提前推送 `citations` 事件）与 `generate_answer` 内部共用同一套裁剪
    逻辑，保证"提前推送的引用"与"实际写入 LLM 上下文的引用"始终一致。

    - `backend="vllm"`：与 `build_context_text` 的预算截断结果对齐（修复
      审查报告 M6：引用编号与送入 LLM 的上下文不匹配）；
    - 其他后端（"local"）：`_generate_local` 直接摘录全部 contexts，天然
      与全部 contexts 对齐，无需裁剪。
    """
    backend = backend or RAG_CONFIG["generation_backend"]
    if backend != "vllm" or not contexts:
        return contexts
    _, included = build_context_text(contexts, RAG_CONFIG["generation_max_context_chars"])
    return included


def _generate_with_vllm(
    query: str, contexts: List[DocBlock], dialogue: Optional[list]
) -> "tuple[Optional[str], List[DocBlock]]":
    context_text, included_contexts = build_context_text(contexts, RAG_CONFIG["generation_max_context_chars"])
    try:
        import requests
        from config.config_loader import CONFIG

        user_content = build_user_prompt(query, context_text, dialogue)
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
        return resp.json()["choices"][0]["message"]["content"].strip(), included_contexts
    except Exception as e:
        # 记录具体失败原因（连接超时/HTTP 错误/响应格式不符预期等），避免上层
        # 仅看到笼统的"vLLM 生成服务不可用"日志而无法定位根因。
        logger.warning(f"⚠️ vLLM 生成请求失败: {e}")
        return None, included_contexts


def _generate_local(query: str, contexts: List[DocBlock]) -> str:
    """抽取式兜底回答：摘录最相关片段并附引用编号，不依赖任何 LLM 服务。"""
    snippet_max_chars = RAG_CONFIG["generation_snippet_max_chars"]
    lines = [f"根据知识库检索结果，为您找到以下 {len(contexts)} 条相关信息："]
    for i, c in enumerate(contexts, start=1):
        snippet = (c.text or c.summary or "").strip().replace("\n", " ")
        if len(snippet) > snippet_max_chars:
            snippet = snippet[:snippet_max_chars] + "..."
        lines.append(f"[{i}] {c.title or c.page_name}：{snippet}")
    lines.append("（以上内容直接摘录自知识库，未经 LLM 润色；如需更完整的智能问答，请部署生成模型服务）")
    return "\n".join(lines)


# ======================== 真流式生成（修复审查报告 M3） ========================
#
# 此前 `/chat/stream` 的"流式"实为：等 `generate_answer` 拿到完整回答后，
# 再按 `stream_answer_chunk_chars` 固定字符数切片伪流式推送——用户感知的
# 首字节延迟与非流式接口一致，只是把同一段文本拆成了多个 SSE data 帧。
# 下面的 `stream_answer` 改为真正对接 vLLM OpenAI 兼容接口的 `stream=True`，
# 逐 token/逐片段从 HTTP 响应流中读取并即时 yield，从生成模型开始产出内容
# 起就能被前端感知到，而不必等待整段回答生成完毕。

def stream_answer(
    query: str,
    contexts: List[DocBlock],
    dialogue: Optional[list] = None,
    backend: Optional[str] = None,
) -> Generator[Tuple[str, object], None, None]:
    """流式生成回答（真流式，非固定字符数切片的伪流式）。

    Args:
        query / contexts / dialogue / backend: 语义同 `generate_answer`。

    Yields:
        ("delta", text_piece)：到达的文本增量（可能为空字符串）；
        ("done", {"answer": 完整回答文本, "backend_used": str})：流结束时的
        最终结果，且保证是最后一次 yield（供调用方据此关闭本轮 SSE 流）。
    """
    backend = backend or RAG_CONFIG["generation_backend"]
    query = (query or "").strip()

    if not contexts:
        answer = "抱歉，知识库中未找到与该问题相关的信息，建议您换一种问法或转人工客服。"
        yield from _chunk_deltas(answer)
        yield "done", {"answer": answer, "backend_used": "no_context"}
        return

    if backend == "vllm":
        full_text = ""
        stream_failed = False
        try:
            for piece in _stream_with_vllm(query, contexts, dialogue):
                if not piece:
                    continue
                full_text += piece
                yield "delta", piece
        except Exception as e:
            # 记录具体失败原因（连接超时/HTTP 错误/响应格式不符预期等），与
            # 非流式路径 `_generate_with_vllm` 保持一致的可观测性。
            logger.warning(f"⚠️ vLLM 流式生成请求失败: {e}")
            stream_failed = True

        if not stream_failed and full_text.strip():
            cited_contexts = select_cited_contexts(contexts, backend="vllm")
            guarded = apply_guardrail(full_text, cited_contexts, backend_used="vllm")
            if guarded != full_text and guarded.startswith(full_text):
                # apply_guardrail 可能在低关联度时追加提示语，把差量作为最后
                # 一段 delta 一并推送，保证前端拼接结果与最终 answer 一致。
                yield "delta", guarded[len(full_text):]
            yield "done", {"answer": guarded, "backend_used": "vllm"}
            return
        logger.warning("⚠️ vLLM 流式生成服务不可用，降级为抽取式回答")

    # 【修复 N6】降级到 local 时，_generate_local 此前对全部 contexts 按
    # enumerate(start=1) 编号，但 citations 事件已基于 vllm 预算子集推送，
    # 二者编号不一致导致引用错位。改为使用 select_cited_contexts 获取与
    # citations 事件一致的子集再生成，保证 [i] 指向正确的块。
    cited = select_cited_contexts(contexts, backend="vllm") if contexts else contexts
    answer = _generate_local(query, cited)
    yield from _chunk_deltas(answer)
    yield "done", {"answer": answer, "backend_used": "local"}


def _chunk_deltas(text: str) -> Generator[Tuple[str, str], None, None]:
    """按 `RAG_CONFIG['stream_answer_chunk_chars']` 将文本切片为多段 delta。

    仅用于本身不支持真流式的场景（local 抽取式兜底、无上下文兜底提示语），
    为前端提供与此前一致的渐进展示体验；vLLM 真流式路径不经过此函数。
    """
    step = RAG_CONFIG["stream_answer_chunk_chars"]
    for i in range(0, len(text), step):
        yield "delta", text[i:i + step]


def _stream_with_vllm(
    query: str, contexts: List[DocBlock], dialogue: Optional[list]
) -> Generator[str, None, None]:
    """向 vLLM OpenAI 兼容接口发起 `stream=True` 请求，解析 SSE 响应并逐段
    yield 文本增量（`choices[0].delta.content`）。"""
    import requests
    from config.config_loader import CONFIG

    context_text, _ = build_context_text(contexts, RAG_CONFIG["generation_max_context_chars"])
    user_content = build_user_prompt(query, context_text, dialogue)
    payload = {
        "model": CONFIG.get("llm_model", "glm"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": RAG_CONFIG["generation_max_new_tokens"],
        "temperature": 0.3,
        "stream": True,
    }
    resp = requests.post(
        CONFIG["vllm_api_url"], json=payload,
        timeout=CONFIG.get("vllm_timeout", 60), stream=True,
    )
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or [{}]
        delta = (choices[0].get("delta") or {}).get("content")
        if delta:
            yield delta
