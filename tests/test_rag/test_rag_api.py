# -*- coding: utf-8 -*-
"""rag/api 集成测试：FastAPI 接口可用性（文档管理/检索/问答/健康检查/错误处理）。"""
import json

import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.usefixtures("clean_rag_data")


@pytest.fixture
def client():
    from rag.api.main import app
    return TestClient(app)


class TestHealthAPI:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "vector_backend" in data["stats"]


class TestDocumentAPI:
    def test_upload_txt_document(self, client):
        content = ("广告限流是风控手段之一。" * 10).encode("utf-8")
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("faq.txt", content, "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["filename"] == "faq.txt"
        assert data["num_chunks"] >= 1
        assert data["doc_id"]

    def test_upload_unsupported_extension_returns_422(self, client):
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("virus.exe", b"binary", "application/octet-stream")},
        )
        assert resp.status_code == 422
        assert resp.json()["error_code"] == "RAG_VALIDATION_ERROR"

    def test_upload_empty_file_returns_422(self, client):
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("empty.txt", b"   ", "text/plain")},
        )
        assert resp.status_code == 422

    def test_ingest_blocks_endpoint(self, client):
        payload = {
            "filename": "blocks.json",
            "blocks": [{"text": "批量导入的内容", "title": "标题", "page_url": "http://x"}],
        }
        resp = client.post("/api/documents/ingest_blocks", json=payload)
        assert resp.status_code == 200
        assert resp.json()["num_chunks"] == 1

    def test_ingest_empty_blocks_returns_422(self, client):
        resp = client.post("/api/documents/ingest_blocks", json={"blocks": [], "filename": "x.json"})
        assert resp.status_code == 422

    def test_ingest_blocks_all_empty_text_returns_422(self, client):
        resp = client.post(
            "/api/documents/ingest_blocks",
            json={"blocks": [{"text": ""}, {"text": "   "}], "filename": "x.json"},
        )
        assert resp.status_code == 422
        assert resp.json()["error_code"] == "RAG_VALIDATION_ERROR"

    def test_upload_malformed_json_returns_422_not_500(self, client):
        """回归测试：修复 ParseError 未捕获导致 500 的 Bug，端到端验证经 API 层返回 422。"""
        resp = client.post(
            "/api/documents/upload",
            files={"file": ("broken.json", b"{not valid json!!!", "application/json")},
        )
        assert resp.status_code == 422
        assert resp.json()["error_code"] == "RAG_VALIDATION_ERROR"

    def test_list_documents(self, client):
        client.post("/api/documents/upload", files={"file": ("a.txt", ("内容" * 10).encode(), "text/plain")})
        resp = client.get("/api/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["documents"][0]["filename"] == "a.txt"

    def test_get_document_detail(self, client):
        upload_resp = client.post("/api/documents/upload", files={"file": ("a.txt", ("内容" * 10).encode(), "text/plain")})
        doc_id = upload_resp.json()["doc_id"]
        resp = client.get(f"/api/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["doc_id"] == doc_id

    def test_get_nonexistent_document_returns_404(self, client):
        resp = client.get("/api/documents/does_not_exist")
        assert resp.status_code == 404
        assert resp.json()["error_code"] == "RAG_NOT_FOUND"

    def test_delete_document(self, client):
        upload_resp = client.post("/api/documents/upload", files={"file": ("a.txt", ("内容" * 10).encode(), "text/plain")})
        doc_id = upload_resp.json()["doc_id"]
        resp = client.delete(f"/api/documents/{doc_id}")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True
        # 二次删除应 404
        resp2 = client.delete(f"/api/documents/{doc_id}")
        assert resp2.status_code == 404


class TestRetrieveAPI:
    def _seed(self, client):
        client.post("/api/documents/ingest_blocks", json={
            "filename": "seed.json",
            "blocks": [
                {"text": "账户异常投放行为会触发限流机制。", "title": "限流规则", "page_url": "http://a"},
                {"text": "退款需在签收后七天内申请。", "title": "退款政策", "page_url": "http://b"},
            ],
        })

    def test_retrieve_returns_results(self, client):
        self._seed(client)
        resp = client.post("/api/retrieve", json={"query": "限流规则是什么", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "限流规则是什么"
        assert isinstance(data["results"], list)
        assert data["latency_ms"] >= 0

    def test_retrieve_empty_query_returns_422(self, client):
        resp = client.post("/api/retrieve", json={"query": ""})
        assert resp.status_code == 422

    def test_retrieve_top_k_out_of_range_returns_422(self, client):
        resp = client.post("/api/retrieve", json={"query": "问题", "top_k": 1000})
        assert resp.status_code == 422

    def test_retrieve_missing_query_field_returns_422(self, client):
        resp = client.post("/api/retrieve", json={})
        assert resp.status_code == 422


class TestChatAPI:
    def _seed(self, client):
        client.post("/api/documents/ingest_blocks", json={
            "filename": "seed.json",
            "blocks": [
                {"text": "千川广告投放出现异常会被限流处理，通常持续24小时。", "title": "限流规则", "page_url": "http://a"},
            ],
        })

    def test_chat_returns_answer_with_citations(self, client):
        self._seed(client)
        resp = client.post("/api/chat", json={"query": "千川广告限流怎么处理"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"]
        assert isinstance(data["citations"], list)
        assert data["backend_used"] in ("local", "vllm", "no_context")

    def test_chat_on_empty_kb_returns_no_context_answer(self, client):
        resp = client.post("/api/chat", json={"query": "任意问题"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["backend_used"] == "no_context"

    def test_chat_empty_query_returns_422(self, client):
        resp = client.post("/api/chat", json={"query": ""})
        assert resp.status_code == 422

    def test_chat_stream_returns_sse_events(self, client):
        self._seed(client)
        with client.stream("POST", "/api/chat/stream", json={"query": "限流规则"}) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert "event: citations" in body
        assert "event: answer" in body
        assert "event: done" in body

    def test_chat_stream_citations_event_precedes_answer_event(self, client):
        """回归测试：修复 chat_stream 需等待完整生成才产出首个事件的性能问题。

        现应先检索并推送 `citations` 事件，再进行生成；验证事件在响应体中的
        先后顺序符合"先引用、后回答、最后 done"的预期。
        """
        self._seed(client)
        with client.stream("POST", "/api/chat/stream", json={"query": "限流规则"}) as resp:
            body = "".join(resp.iter_text())
        assert body.index("event: citations") < body.index("event: answer") < body.index("event: done")

    def test_chat_stream_empty_kb_still_emits_citations_and_done(self, client):
        with client.stream("POST", "/api/chat/stream", json={"query": "任意问题"}) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert "event: citations" in body
        assert "event: done" in body


class TestErrorSchema:
    def test_error_response_has_required_fields(self, client):
        resp = client.get("/api/documents/nope")
        data = resp.json()
        assert set(["error_code", "message", "detail"]).issubset(data.keys())

    def test_openapi_schema_available(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        schema = resp.json()
        assert "/api/chat" in schema["paths"]
        assert "/api/retrieve" in schema["paths"]


class TestCORSConfig:
    def test_cors_does_not_allow_credentials_with_wildcard_origin(self, client):
        """回归测试：修复 `allow_origins=["*"]` 与 `allow_credentials=True` 同时启用
        的 CORS 反模式（见 rag/api/main.py 注释）。"""
        resp = client.options(
            "/api/health",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-credentials") != "true"
        assert resp.headers.get("access-control-allow-origin") in ("*", "http://example.com")
