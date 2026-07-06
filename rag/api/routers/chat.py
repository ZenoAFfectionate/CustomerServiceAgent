# -*- coding: utf-8 -*-
"""问答接口：检索 + 生成融合，支持普通响应与流式（SSE）响应。"""
import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from config.config_loader import logger
from rag.api.errors import ValidationError
from rag.api.models import ChatRequest, ChatResponse
from rag.api.utils import parse_dialogue
from rag.config import RAG_CONFIG
from rag import pipeline
from rag.generation.citation import build_citations
from rag.generation.llm_generation import select_cited_contexts, stream_answer

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
        try:
            rewritten_query, contexts = pipeline.retrieve_context(query, dialogue=dialogue, top_k=payload.top_k)
            # 【修复 M6】citations 应与 generate_answer 实际使用的上下文子集对齐
            # （vllm 后端下预算不足时靠后的块会被 build_context_text 整段丢弃），
            # 复用 select_cited_contexts 保证"提前推送的引用"与"生成阶段真正
            # 采用的引用"始终一致，不会出现引用编号指向未进入上下文的块。
            cited_contexts = select_cited_contexts(contexts, backend=RAG_CONFIG["generation_backend"])
            citations = build_citations(cited_contexts)
            yield f"event: citations\ndata: {json.dumps(citations, ensure_ascii=False)}\n\n"

            # 【修复 M3】此前先等 generate_answer 拿到完整回答，再按固定字符数
            # 切片伪流式推送——用户感知延迟与非流式接口无异。现改用 stream_answer
            # 对接 vLLM `stream=True` 的真流式接口，生成模型每产出一个 token/
            # 片段就立即通过 SSE 推送，而不必等待整段回答生成完毕。
            backend_used = "local"
            full_answer = ""
            for kind, data in stream_answer(rewritten_query, contexts, dialogue=dialogue):
                if kind == "delta":
                    if data:
                        full_answer += data
                        yield f"event: answer\ndata: {json.dumps({'delta': data}, ensure_ascii=False)}\n\n"
                else:  # kind == "done"
                    backend_used = data["backend_used"]
                    # 【修复 N18】stream_answer 的 done 事件携带完整 answer，但
                    # 此前只取 backend_used 丢弃了 answer。前端若依赖 done 事件
                    # 落库完整文本（而非拼接 delta）将拿到不含 answer 的事件。
                    full_answer = data.get("answer", full_answer)
            yield f"event: done\ndata: {json.dumps({'answer': full_answer, 'backend_used': backend_used}, ensure_ascii=False)}\n\n"
        except Exception as e:
            # 【修复 H5】此前整段无任何异常捕获：检索/生成任一环节抛出未处理异常
            # 时，Starlette 会直接中断 SSE 流、关闭连接，且不会发送 docstring 中
            # 承诺的 done/error 事件，前端只能一直挂起或遭遇连接异常断开，且无法
            # 区分"正常结束"与"异常中断"。这里补齐 error 事件 + done 收尾，保证
            # 前端总能收到明确的流结束信号；同时记录服务端日志便于排查根因。
            logger.error(f"❌ SSE 流式问答异常（query={query[:60]!r}）: {e}", exc_info=e)
            yield (
                "event: error\n"
                f"data: {json.dumps({'error_code': 'RAG_INTERNAL_ERROR', 'message': str(e)}, ensure_ascii=False)}\n\n"
            )
            yield f"event: done\ndata: {json.dumps({'backend_used': 'error'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_gen(), media_type="text/event-stream")
