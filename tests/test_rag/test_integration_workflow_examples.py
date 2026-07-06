# -*- coding: utf-8 -*-
"""rag/integration/workflow_examples.py 单元测试：典型用法示例可正常运行。

这些示例函数本身就是"多模块组合"的演示（导入+检索、多轮对话、Agent 工具调用），
测试目标是确保示例代码始终可运行（不因模块重构而悄然失效），保护 README 中引用的
用法说明不会腐化（"文档即代码"）。
"""
import pytest

from rag.integration.workflow_examples import (
    example_agent_tool_call, example_ingest_and_retrieve, example_multi_turn_dialogue,
    example_streaming_chat_client,
)

pytestmark = pytest.mark.usefixtures("clean_rag_data")


class TestExampleIngestAndRetrieve:
    def test_returns_ingest_meta_and_results(self):
        result = example_ingest_and_retrieve()
        assert result["ingest_meta"]["num_chunks"] == 2
        assert isinstance(result["retrieve_results"], list)


class TestExampleMultiTurnDialogue:
    def test_returns_answer_with_expected_fields(self):
        result = example_multi_turn_dialogue()
        for field in ["query", "answer", "backend_used", "citations", "contexts"]:
            assert field in result


class TestExampleAgentToolCall:
    def test_returns_ok_result_via_tool_dispatch(self):
        result = example_agent_tool_call()
        assert result["ok"] is True
        assert "answer" in result


class TestExampleStreamingChatClient:
    def test_prints_without_raising(self, capsys):
        example_streaming_chat_client()
        captured = capsys.readouterr()
        assert "chat/stream" in captured.out
