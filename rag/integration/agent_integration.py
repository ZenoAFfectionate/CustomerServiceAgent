# -*- coding: utf-8 -*-
"""Agent 集成（Agent Integration）：将 `rag.pipeline` 的检索/问答能力封装为
"Agent 工具"，供 `agent/`（hello-agents 框架）的 `ReActAgent` 在多轮对话中
自主判断"是否需要检索"并调用。

对齐 `README.md`「RAG 与 Agent 的职责划分」章节规划的集成方式：
把 `rag.pipeline.retrieve()`/`answer()` 封装为一个工具，注册进 Agent 的
`ToolRegistry`，交由 `ReActAgent` 在多轮对话中自主判断"是否需要检索"。

本模块提供两套等价的封装，供不同场景选用：

1. `rag_retrieve_tool`/`rag_answer_tool`（函数版，返回原生 dict、内部捕获全部
   异常不抛出）：适合直接以普通 Python 函数形式调用，或用于
   `rag/integration/tool_usage.py` 的 `dispatch_tool_call()` 简单调度。

2. `RAGRetrieveTool`/`RAGAnswerTool`（`hello-agents` `Tool` 协议适配器类，见
   `get_rag_tools()`）：**推荐**通过 `ToolRegistry.register_tool()` 注册的
   方式——修复审查报告 H2 记录的真实契约问题：
   - `ToolRegistry.register_function(func)` 路径下，`func(input_text)` 执行完
     后无条件包成 `ToolResponse.success(...)`（`registry.py:194`），即使
     `func` 返回的 dict 里 `ok=False` 也会被视为"成功"；
   - `CircuitBreaker.record_result` 仅以 `response.status == ToolStatus.ERROR`
     判定失败，成功状态会重置失败计数——熔断器因此永远不会打开，Agent 在
     ReAct 循环里会反复调用故障的 RAG 工具直到 `max_steps` 耗尽；
   - 结构化的 `results`/`citations` 被 `str()` 成 Python 字面量塞进
     `ToolResponse.text`，模型难以解析。

   `RAGRetrieveTool`/`RAGAnswerTool` 的 `run()` 直接返回 `ToolResponse`：
   失败时用 `ToolResponse.error(...)`（`status=ERROR`），使熔断器与框架的
   错误识别逻辑正确生效；成功时把 `results`/`citations` 放进 `data`
   （结构化字段），而不是塞进 `text` 里变成不可解析的字面量字符串。
"""
from typing import Any, Dict, List, Optional


def rag_retrieve_tool(query: str, top_k: int = 5) -> dict:
    """Agent 工具（函数版）：检索知识库，返回精排后的 Top-K 上下文（不生成回答）。

    Returns:
        成功: {"ok": True, "results": [...]}
        失败: {"ok": False, "error": str}
    """
    try:
        from rag import pipeline
        results = pipeline.retrieve(query, top_k=top_k)
        return {"ok": True, "results": results}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def rag_answer_tool(query: str, dialogue: Optional[list] = None, top_k: int = 5) -> dict:
    """Agent 工具（函数版）：检索增强问答，返回回答与引用来源。

    Args:
        query: 用户问题
        dialogue: 多轮历史 [{"speaker": "user"/"bot", "text": ...}, ...]（可选）
        top_k: 检索上下文条数

    Returns:
        成功: {"ok": True, "answer": str, "citations": [...]}
        失败: {"ok": False, "error": str}
    """
    try:
        from rag import pipeline
        result = pipeline.answer(query, dialogue=dialogue, top_k=top_k)
        return {"ok": True, "answer": result["answer"], "citations": result["citations"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def _lazy_import_tool_base():
    """延迟导入 `hello_agents.tools` 基础设施，避免 `rag/` 在未挂载 `agent/`
    （或未安装 hello-agents 包）时因模块级 import 而硬性依赖 Agent 框架——
    `rag/` 设计上应可独立于 `agent/` 运行（详见 README「RAG 与 Agent 的职责
    划分」）。仅在真正调用 `get_rag_tools()`/实例化 `RAGRetrieveTool` 等时
    才尝试导入，找不到时给出清晰的错误提示而非令人费解的 ImportError 堆栈。
    """
    try:
        from hello_agents.tools.base import Tool, ToolParameter
        from hello_agents.tools.response import ToolResponse
        from hello_agents.tools.errors import ToolErrorCode
    except ImportError as e:
        raise RuntimeError(
            "无法导入 hello-agents 工具基础设施（hello_agents.tools）。"
            "请确认 agent/ 或 agent/src 已加入 sys.path（或已 pip 安装 hello-agents 包），"
            "再使用 RAGRetrieveTool/RAGAnswerTool/get_rag_tools()。"
        ) from e
    return Tool, ToolParameter, ToolResponse, ToolErrorCode


def _make_rag_retrieve_tool_class():
    Tool, ToolParameter, ToolResponse, ToolErrorCode = _lazy_import_tool_base()

    class RAGRetrieveTool(Tool):
        """检索知识库工具（`hello_agents.tools.Tool` 协议适配器，修复 H2）。"""

        def __init__(self):
            super().__init__(
                name="rag_retrieve",
                description="在企业知识库中检索与问题相关的文档片段（不生成回答，仅返回原始检索结果）。",
            )

        def get_parameters(self) -> List[ToolParameter]:
            return [
                ToolParameter(name="query", type="string", description="用户问题或检索关键词", required=True),
                ToolParameter(name="top_k", type="integer", description="返回条数", required=False, default=5),
            ]

        def run(self, parameters: Dict[str, Any]) -> "ToolResponse":
            query = (parameters or {}).get("query", "")
            # 【修复 N13】此前 `params.get("top_k") or 5` 会把合法的 0 改写为 5，
            # 改用显式 None 检查。
            top_k = (parameters or {}).get("top_k")
            if top_k is None:
                top_k = 5
            if not str(query).strip():
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message="query 不能为空",
                    context={"parameters": parameters},
                )
            try:
                from rag import pipeline
                results = pipeline.retrieve(query, top_k=top_k)
            except Exception as e:
                # 【修复 H2】失败时返回 status=ERROR 的 ToolResponse，而非原
                # 函数版那样吞掉异常返回 {"ok": False, ...} 的普通 dict——
                # 后者经 ToolRegistry.register_function 注册后会被无条件包成
                # SUCCESS，导致熔断器永远无法感知到失败。
                return ToolResponse.error(
                    code=ToolErrorCode.EXECUTION_ERROR,
                    message=f"知识库检索失败: {e}",
                    context={"query": query, "top_k": top_k},
                )
            summary_lines = [f"检索到 {len(results)} 条相关结果："]
            for i, r in enumerate(results, start=1):
                title = r.get("title") or r.get("page_name") or ""
                snippet = (r.get("text") or "")[:120]
                summary_lines.append(f"[{i}] {title}: {snippet}")
            return ToolResponse.success(
                text="\n".join(summary_lines),
                # 结构化 data（而非 str() 成 Python 字面量塞进 text），模型/
                # 下游代码可直接解析 results 字段。
                data={"results": results},
            )

    return RAGRetrieveTool


def _make_rag_answer_tool_class():
    Tool, ToolParameter, ToolResponse, ToolErrorCode = _lazy_import_tool_base()

    class RAGAnswerTool(Tool):
        """检索增强问答工具（`hello_agents.tools.Tool` 协议适配器，修复 H1+H2）。"""

        def __init__(self):
            super().__init__(
                name="rag_answer",
                description="基于企业知识库进行检索增强问答，返回自然语言回答与引用来源。支持传入多轮对话历史以补全指代。",
            )

        def get_parameters(self) -> List[ToolParameter]:
            return [
                ToolParameter(name="query", type="string", description="用户问题", required=True),
                ToolParameter(
                    name="dialogue", type="array",
                    description=(
                        '多轮对话历史（可选），用于指代补全后的 query 重写，格式如 '
                        '[{"speaker": "user", "text": "..."}, {"speaker": "bot", "text": "..."}]'
                    ),
                    required=False, default=None,
                    # 【修复 N5】为 array 类型提供 items schema，使 ReActAgent 的
                    # _build_tool_schemas 能生成完整的 JSON Schema（含元素结构），
                    # 否则严格校验的 LLM 服务可能拒绝该工具定义。
                    items={
                        "type": "object",
                        "properties": {
                            "speaker": {"type": "string", "description": "发言角色：user 或 bot"},
                            "text": {"type": "string", "description": "发言内容"},
                        },
                        "required": ["speaker", "text"],
                    },
                ),
                ToolParameter(name="top_k", type="integer", description="检索上下文条数", required=False, default=5),
            ]

        def run(self, parameters: Dict[str, Any]) -> "ToolResponse":
            params = parameters or {}
            query = params.get("query", "")
            dialogue = params.get("dialogue")
            # 【修复 N13】同 RAGRetrieveTool：避免 `or 5` 吞掉合法的 0。
            top_k = params.get("top_k")
            if top_k is None:
                top_k = 5
            if not str(query).strip():
                return ToolResponse.error(
                    code=ToolErrorCode.INVALID_PARAM,
                    message="query 不能为空",
                    context={"parameters": parameters},
                )
            try:
                from rag import pipeline
                result = pipeline.answer(query, dialogue=dialogue, top_k=top_k)
            except Exception as e:
                # 【修复 H2】同 RAGRetrieveTool：失败时以 ToolResponse.error
                # 上抛，使熔断器能正确识别失败并计数。
                return ToolResponse.error(
                    code=ToolErrorCode.EXECUTION_ERROR,
                    message=f"知识库问答失败: {e}",
                    context={"query": query, "top_k": top_k},
                )
            return ToolResponse.success(
                text=result["answer"],
                data={"answer": result["answer"], "citations": result["citations"], "backend_used": result["backend_used"]},
            )

    return RAGAnswerTool


def get_rag_tools() -> list:
    """返回可直接 `ToolRegistry.register_tool()` 注册的 `Tool` 实例列表
    （`RAGRetrieveTool`、`RAGAnswerTool`），供 Agent 侧一行代码完成接线：

        from agent.src.tools.registry import ToolRegistry
        from rag.integration.agent_integration import get_rag_tools

        registry = ToolRegistry()
        for tool in get_rag_tools():
            registry.register_tool(tool)

    使用 `register_tool()`（而非 `register_function()`）是修复审查报告 H2
    的关键——只有走 `Tool` 协议，工具执行失败时才会被框架正确识别为
    `ToolStatus.ERROR` 并驱动熔断器计数。
    """
    RAGRetrieveTool = _make_rag_retrieve_tool_class()
    RAGAnswerTool = _make_rag_answer_tool_class()
    return [RAGRetrieveTool(), RAGAnswerTool()]
