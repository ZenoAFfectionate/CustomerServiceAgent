# -*- coding: utf-8 -*-
"""典型用法示例（Workflow Examples）：以可运行函数的形式演示 `rag/` 的常见
使用场景，作为"文档即代码"补充 `rag/README.md`。每个示例函数均可独立调用，
默认使用本地降级后端（零外部依赖），可直接 `python -m rag.integration.workflow_examples`
运行全部示例。
"""


def example_ingest_and_retrieve() -> dict:
    """示例 1：导入一批知识块，随后检索。"""
    from rag.knowledge_base import corpus_management

    blocks = [
        {"text": "商品签收后七天内可申请无理由退款，退款将原路返回至支付账户。",
         "title": "退款政策", "page_url": "http://help.example.com/refund"},
        {"text": "账户存在异常投放行为时，系统将自动限流，限流期间广告曝光量下降 50%-80%。",
         "title": "广告限流触发条件", "page_url": "http://help.example.com/ad-limit"},
    ]
    meta = corpus_management.ingest_blocks(blocks, filename="example.json")

    from rag import pipeline
    results = pipeline.retrieve("怎么退款", top_k=3)
    return {"ingest_meta": meta, "retrieve_results": results}


def example_multi_turn_dialogue() -> dict:
    """示例 2：多轮对话问答（第二轮问题依赖第一轮上下文，触发 query_rewrite）。"""
    from rag import pipeline

    dialogue = [
        {"speaker": "user", "text": "广告为什么被限流了"},
        {"speaker": "bot", "text": "账户存在异常投放行为时会触发限流。"},
    ]
    return pipeline.answer("那要多久才能解除", dialogue=dialogue, top_k=3)


def example_agent_tool_call() -> dict:
    """示例 3：以 Agent 工具调用的方式使用 RAG（不抛异常，返回结构化结果）。"""
    from rag.integration.tool_usage import dispatch_tool_call

    return dispatch_tool_call("rag_answer", {"query": "退款政策是什么", "top_k": 3})


def example_streaming_chat_client(base_url: str = "http://localhost:8090") -> None:
    """示例 4：消费 `/api/chat/stream` 的 SSE 流式响应（需要服务已启动）。

    仅打印说明，不在导入期发起真实网络请求；需要真实运行时手动调用本函数。
    """
    print(
        "SSE 流式问答客户端示例：\n"
        f"  import requests\n"
        f"  resp = requests.post('{base_url}/api/chat/stream', "
        "json={'query': '退款政策是什么'}, stream=True)\n"
        "  for line in resp.iter_lines(decode_unicode=True):\n"
        "      if line:\n"
        "          print(line)  # event: citations / answer / done\n"
    )


if __name__ == "__main__":
    import json

    print("=== 示例 1：导入 + 检索 ===")
    print(json.dumps(example_ingest_and_retrieve(), ensure_ascii=False, indent=2, default=str))

    print("\n=== 示例 2：多轮对话问答 ===")
    print(json.dumps(example_multi_turn_dialogue(), ensure_ascii=False, indent=2, default=str))

    print("\n=== 示例 3：Agent 工具调用 ===")
    print(json.dumps(example_agent_tool_call(), ensure_ascii=False, indent=2, default=str))

    print("\n=== 示例 4：SSE 流式客户端用法 ===")
    example_streaming_chat_client()
