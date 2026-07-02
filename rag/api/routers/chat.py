# -*- coding: utf-8 -*-
"""问答接口：检索 + 生成融合，支持普通响应与流式（SSE）响应。"""
import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from rag.api.errors import ValidationError
from rag.api.models import ChatRequest, ChatResponse
from rag.api.utils import parse_dialogue
from rag.config import RAG_CONFIG
from rag import pipeline
from rag.generation.generator import build_citations, generate_answer

router = APIRouter(tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="检索增强问答（RAG QA）",
    description=(
        "对用户问题执行「检索 → 融合去重 → 精排 → 生成」全流程，返回基于知识库上下文生成的"
        "回答、引用来源与检索上下文。无生成模型服务时自动降级为抽取式摘录回答。"
    ),
)
def chat(payload: ChatRequest) -> ChatResponse:
    query = (payload.query or "").strip()
    if not query:
        raise ValidationError("query 不能为空")

    t0 = time.time()
    result = pipeline.answer(query, dialogue=parse_dialogue(payload.dialogue), top_k=payload.top_k)
    latency_ms = (time.time() - t0) * 1000
    return ChatResponse(**result, latency_ms=round(latency_ms, 2))


@router.post(
    "/chat/stream",
    summary="检索增强问答（SSE 流式输出）",
    description=(
        "与 /chat 相同的问答流程，但以 Server-Sent Events 方式逐块返回答案文本，"
        "便于前端流式展示。事件类型：`citations`（先返回引用信息）、`answer`（逐段答案文本）、`done`（结束标记）。"
    ),
)
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    query = (payload.query or "").strip()
    if not query:
        raise ValidationError("query 不能为空")

    dialogue = parse_dialogue(payload.dialogue)

    def _gen():
        # 【优化点/性能修复】原实现在生成器内部一次性同步调用 `pipeline.answer()`
        # （检索 + 生成融合合并为一步），导致必须等待生成完全结束才能产出第一个
        # SSE 事件——名为"流式"实则与非流式接口的用户感知延迟无异。
        # 现拆分为"先检索、立即推送引用，再生成"两步：`citations` 事件可在生成
        # 开始前就送达前端，让用户更快看到"知识库命中了哪些来源"，缩短首字节延迟。
        rewritten_query, contexts = pipeline.retrieve_context(query, dialogue=dialogue, top_k=payload.top_k)
        citations = build_citations(contexts)
        yield f"event: citations\ndata: {json.dumps(citations, ensure_ascii=False)}\n\n"

        gen_result = generate_answer(rewritten_query, contexts, dialogue=dialogue)
        answer_text = gen_result["answer"]
        step = RAG_CONFIG["stream_answer_chunk_chars"]
        for i in range(0, len(answer_text), step):
            chunk = answer_text[i:i + step]
            yield f"event: answer\ndata: {json.dumps({'delta': chunk}, ensure_ascii=False)}\n\n"
        yield f"event: done\ndata: {json.dumps({'backend_used': gen_result['backend_used']}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")
