# -*- coding: utf-8 -*-
"""rag/integration/agent_integration.py 单元测试：Agent 工具函数适配器。

组合测试：rag_retrieve_tool/rag_answer_tool 依赖 pipeline.retrieve/answer（检索+
生成全链路），验证 integration/ 层"不抛异常、返回结构化结果"的契约。
"""
import pytest

from rag.indexing import index_builder as indexer
from rag.integration.agent_integration import rag_answer_tool, rag_retrieve_tool

pytestmark = pytest.mark.usefixtures("clean_rag_data")


def _seed():
    indexer.ingest_blocks([{"text": "退款需在签收后七天内申请，原路退回支付账户"}], filename="seed.json")


class TestRagRetrieveTool:
    def test_success_returns_ok_and_results(self):
        _seed()
        result = rag_retrieve_tool("怎么退款", top_k=3)
        assert result["ok"] is True
        assert isinstance(result["results"], list)

    def test_empty_kb_returns_ok_with_empty_results(self):
        result = rag_retrieve_tool("任意问题", top_k=3)
        assert result["ok"] is True
        assert result["results"] == []

    def test_internal_exception_returns_ok_false_not_raise(self, monkeypatch):
        """工具函数必须捕获内部异常，绝不能让异常穿透到 Agent 调用方。"""
        from rag import pipeline

        def _raise(*a, **kw):
            raise RuntimeError("模拟内部异常")

        monkeypatch.setattr(pipeline, "retrieve", _raise)
        result = rag_retrieve_tool("任意问题")
        assert result["ok"] is False
        assert "error" in result


class TestRagAnswerTool:
    def test_success_returns_ok_answer_and_citations(self):
        _seed()
        result = rag_answer_tool("怎么退款")
        assert result["ok"] is True
        assert result["answer"]
        assert isinstance(result["citations"], list)

    def test_supports_dialogue_parameter(self):
        _seed()
        dialogue = [{"speaker": "user", "text": "上一轮问题"}]
        result = rag_answer_tool("这一轮问题", dialogue=dialogue)
        assert result["ok"] is True

    def test_internal_exception_returns_ok_false_not_raise(self, monkeypatch):
        from rag import pipeline

        def _raise(*a, **kw):
            raise RuntimeError("模拟内部异常")

        monkeypatch.setattr(pipeline, "answer", _raise)
        result = rag_answer_tool("任意问题")
        assert result["ok"] is False
        assert "error" in result


class TestRAGToolClasses:
    """`RAGRetrieveTool`/`RAGAnswerTool`（`get_rag_tools()`）：回归测试修复
    审查报告 H2——`Tool` 协议适配器应正确返回 `ToolResponse`（成功
    `status=SUCCESS`、失败 `status=ERROR`），而非原函数版返回的普通 dict。
    """

    def test_get_rag_tools_returns_two_tool_instances(self):
        from hello_agents.tools.base import Tool
        from rag.integration.agent_integration import get_rag_tools

        tools = get_rag_tools()
        names = {t.name for t in tools}
        assert names == {"rag_retrieve", "rag_answer"}
        assert all(isinstance(t, Tool) for t in tools)

    def test_retrieve_tool_success_returns_success_status_and_structured_data(self):
        from hello_agents.tools.response import ToolStatus
        from rag.integration.agent_integration import get_rag_tools

        _seed()
        tool = next(t for t in get_rag_tools() if t.name == "rag_retrieve")
        resp = tool.run({"query": "怎么退款", "top_k": 3})
        assert resp.status == ToolStatus.SUCCESS
        assert isinstance(resp.data["results"], list)

    def test_retrieve_tool_failure_returns_error_status(self, monkeypatch):
        """核心回归点：pipeline.retrieve 失败时必须是 `status=ERROR`，
        而不是被误判为成功——否则熔断器永远不会打开（H2 的核心问题）。"""
        from hello_agents.tools.response import ToolStatus
        from rag import pipeline
        from rag.integration.agent_integration import get_rag_tools

        def _raise(*a, **kw):
            raise RuntimeError("模拟检索后端故障")

        monkeypatch.setattr(pipeline, "retrieve", _raise)
        tool = next(t for t in get_rag_tools() if t.name == "rag_retrieve")
        resp = tool.run({"query": "任意问题"})
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info is not None

    def test_answer_tool_failure_returns_error_status(self, monkeypatch):
        from hello_agents.tools.response import ToolStatus
        from rag import pipeline
        from rag.integration.agent_integration import get_rag_tools

        def _raise(*a, **kw):
            raise RuntimeError("模拟问答后端故障")

        monkeypatch.setattr(pipeline, "answer", _raise)
        tool = next(t for t in get_rag_tools() if t.name == "rag_answer")
        resp = tool.run({"query": "任意问题"})
        assert resp.status == ToolStatus.ERROR

    def test_answer_tool_supports_dialogue_and_returns_structured_data(self):
        from hello_agents.tools.response import ToolStatus
        from rag.integration.agent_integration import get_rag_tools

        _seed()
        tool = next(t for t in get_rag_tools() if t.name == "rag_answer")
        dialogue = [{"speaker": "user", "text": "上一轮问题"}]
        resp = tool.run({"query": "这一轮问题", "dialogue": dialogue})
        assert resp.status == ToolStatus.SUCCESS
        assert "answer" in resp.data
        assert "citations" in resp.data

    def test_empty_query_returns_error_not_exception(self):
        from hello_agents.tools.response import ToolStatus
        from rag.integration.agent_integration import get_rag_tools

        tool = next(t for t in get_rag_tools() if t.name == "rag_answer")
        resp = tool.run({"query": ""})
        assert resp.status == ToolStatus.ERROR

    def test_circuit_breaker_opens_after_repeated_failures_via_register_tool(self, monkeypatch):
        """端到端回归测试：走 `ToolRegistry.register_tool()`（而非
        `register_function()`）注册后，工具连续失败应能正确驱动熔断器打开
        ——这正是 H2 修复前"永远不会打开"的核心场景。"""
        from hello_agents.tools.registry import ToolRegistry
        from hello_agents.tools.circuit_breaker import CircuitBreaker
        from rag import pipeline
        from rag.integration.agent_integration import get_rag_tools

        def _raise(*a, **kw):
            raise RuntimeError("模拟连续故障")

        monkeypatch.setattr(pipeline, "retrieve", _raise)

        registry = ToolRegistry(circuit_breaker=CircuitBreaker(failure_threshold=3, recovery_timeout=300))
        for tool in get_rag_tools():
            registry.register_tool(tool)

        for _ in range(3):
            registry.execute_tool("rag_retrieve", '{"query": "任意问题"}')

        assert registry.circuit_breaker.is_open("rag_retrieve") is True

        # 熔断打开后，第 4 次调用应被熔断器直接拦截（不再真正执行底层逻辑）
        resp = registry.execute_tool("rag_retrieve", '{"query": "任意问题"}')
        from hello_agents.tools.response import ToolStatus
        assert resp.status == ToolStatus.ERROR
        assert resp.error_info["code"] == "CIRCUIT_OPEN"
