# -*- coding: utf-8 -*-
"""工具调用 Schema（Tool Usage）：OpenAI Function Calling 风格的工具定义，
供 Agent 框架（如 `agent/` 的 hello-agents）注册为可被 LLM 自主决策调用的
"函数"。`dispatch_tool_call()` 提供一个按名称路由到
`agent_integration.py` 实现的最小调度器，方便 Agent 框架直接对接。
"""
from typing import Any, Dict

from rag.integration.agent_integration import rag_answer_tool, rag_retrieve_tool

RAG_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "rag_retrieve",
            "description": "在企业知识库中检索与问题相关的文档片段（不生成回答，仅返回原始检索结果）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用户问题或检索关键词"},
                    "top_k": {"type": "integer", "description": "返回条数，默认 5"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rag_answer",
            "description": "基于企业知识库进行检索增强问答，返回自然语言回答与引用来源。支持传入多轮对话历史以补全指代（如“它”“那个”等），提升多轮问答的检索准确率。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "用户问题"},
                    # 【修复 H1】此前 schema 缺少 dialogue 参数：`rag_answer_tool`
                    # 函数签名支持 `dialogue` 以驱动多轮 query 重写，但 Function
                    # Calling schema 未声明该参数，LLM 无法通过 Function Calling
                    # 传入多轮历史，导致 `rag_answer` 永远只能单轮问答，RAG 侧
                    # 精心实现的 query_rewrite（多轮改写）能力被静默阉割。
                    "dialogue": {
                        "type": "array",
                        "description": (
                            '多轮历史对话（可选），用于指代补全后的 query 重写。'
                            '每项为 {"speaker": "user"/"bot", "text": "..."}。'
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "speaker": {"type": "string", "description": "发言角色：user 或 bot"},
                                "text": {"type": "string", "description": "发言内容"},
                            },
                            "required": ["speaker", "text"],
                        },
                    },
                    "top_k": {"type": "integer", "description": "检索上下文条数，默认 5"},
                },
                "required": ["query"],
            },
        },
    },
]

_DISPATCH_TABLE = {
    "rag_retrieve": rag_retrieve_tool,
    "rag_answer": rag_answer_tool,
}


def dispatch_tool_call(name: str, arguments: Dict[str, Any]) -> dict:
    """按工具名分发调用，未知工具名/缺必填参数返回错误信息（不抛异常）。

    【修复 N14】此前直接 ``fn(**arguments)``，若 arguments 缺 ``query`` 或含
    多余键会抛 TypeError，打破"不抛异常"契约。现先校验必填键，再用
    inspect.signature 容错忽略多余键。
    """
    import inspect

    fn = _DISPATCH_TABLE.get(name)
    if fn is None:
        return {"ok": False, "error": f"未知工具: {name}，可用工具: {list(_DISPATCH_TABLE)}"}

    arguments = arguments or {}
    sig = inspect.signature(fn)
    # 校验必填参数
    missing = [
        p.name for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty and p.name not in arguments
    ]
    if missing:
        return {"ok": False, "error": f"工具 '{name}' 缺少必填参数: {missing}"}
    # 仅传递函数实际声明的参数，忽略多余键
    valid_keys = set(sig.parameters.keys())
    filtered = {k: v for k, v in arguments.items() if k in valid_keys}
    try:
        return fn(**filtered)
    except Exception as e:
        return {"ok": False, "error": f"工具 '{name}' 执行失败: {e}"}
