# -*- coding: utf-8 -*-
"""rag/integration/tool_usage.py 单元测试：Function Calling Schema 与调度器。

组合测试：dispatch_tool_call 组合了 tool_usage（Schema/路由）+ agent_integration
（工具实现）+ pipeline（检索/生成），验证按名称分发的调用链路正确。
"""
import pytest

from rag.indexing import index_builder as indexer
from rag.integration.tool_usage import RAG_TOOL_SCHEMAS, dispatch_tool_call

pytestmark = pytest.mark.usefixtures("clean_rag_data")


class TestToolSchemas:
    def test_schemas_contain_rag_retrieve_and_rag_answer(self):
        names = [s["function"]["name"] for s in RAG_TOOL_SCHEMAS]
        assert "rag_retrieve" in names
        assert "rag_answer" in names

    def test_each_schema_has_required_openai_fields(self):
        for schema in RAG_TOOL_SCHEMAS:
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "description" in schema["function"]
            assert "parameters" in schema["function"]
            assert "query" in schema["function"]["parameters"]["properties"]

    def test_rag_answer_schema_declares_dialogue_parameter(self):
        """回归测试：修复审查报告 H1——`rag_answer_tool(query, dialogue=None,
        top_k=5)` 支持多轮历史，但此前 `RAG_TOOL_SCHEMAS` 中 `rag_answer` 的
        `parameters.properties` 仅有 `query`/`top_k`，缺少 `dialogue`，导致
        LLM 无法经 Function Calling 传入多轮历史。"""
        rag_answer_schema = next(s for s in RAG_TOOL_SCHEMAS if s["function"]["name"] == "rag_answer")
        properties = rag_answer_schema["function"]["parameters"]["properties"]
        assert "dialogue" in properties
        assert properties["dialogue"]["type"] == "array"
        # dialogue 非必填（保持向后兼容：单轮问答无需提供）
        assert "dialogue" not in rag_answer_schema["function"]["parameters"]["required"]

    def test_rag_retrieve_schema_unaffected_by_dialogue_fix(self):
        """rag_retrieve 本身不支持多轮改写，不应被误加 dialogue 参数。"""
        rag_retrieve_schema = next(s for s in RAG_TOOL_SCHEMAS if s["function"]["name"] == "rag_retrieve")
        assert "dialogue" not in rag_retrieve_schema["function"]["parameters"]["properties"]


class TestDispatchToolCall:
    def test_dispatch_rag_retrieve(self):
        indexer.ingest_blocks([{"text": "退款政策说明内容"}], filename="a.json")
        result = dispatch_tool_call("rag_retrieve", {"query": "退款", "top_k": 3})
        assert result["ok"] is True
        assert "results" in result

    def test_dispatch_rag_answer(self):
        indexer.ingest_blocks([{"text": "退款政策说明内容"}], filename="a.json")
        result = dispatch_tool_call("rag_answer", {"query": "怎么退款"})
        assert result["ok"] is True
        assert "answer" in result

    def test_dispatch_unknown_tool_returns_error_not_raise(self):
        result = dispatch_tool_call("not_a_real_tool", {"query": "x"})
        assert result["ok"] is False
        assert "未知工具" in result["error"]

    def test_dispatch_passes_arguments_correctly(self):
        """top_k 参数应正确透传到底层 retrieve 调用。"""
        indexer.ingest_blocks(
            [{"text": f"内容第{i}条"} for i in range(5)], filename="a.json",
        )
        result = dispatch_tool_call("rag_retrieve", {"query": "内容", "top_k": 2})
        assert result["ok"] is True
        assert len(result["results"]) <= 2
