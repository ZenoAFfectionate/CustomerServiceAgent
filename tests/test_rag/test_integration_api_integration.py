# -*- coding: utf-8 -*-
"""rag/integration/api_integration.py 单元测试：FastAPI 应用获取与挂载。

组合测试：mount_rag_api 组合了 rag/api/routers 全部路由 + observability/dashboard，
验证挂载到任意宿主 FastAPI 应用后功能与独立部署完全一致。
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.integration.api_integration import get_app, mount_rag_api

pytestmark = pytest.mark.usefixtures("clean_rag_data")


class TestGetApp:
    def test_returns_fastapi_app_with_rag_routes(self):
        app = get_app()
        paths = [r.path for r in app.routes]
        assert "/api/health" in paths
        assert "/api/chat" in paths


class TestMountRagApi:
    def test_mounted_routes_accessible_under_custom_prefix(self):
        host_app = FastAPI()
        mount_rag_api(host_app, prefix="/rag/api")
        client = TestClient(host_app)

        resp = client.get("/rag/api/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_mounted_documents_and_retrieve_endpoints_work(self):
        host_app = FastAPI()
        mount_rag_api(host_app, prefix="/rag/api")
        client = TestClient(host_app)

        upload_resp = client.post(
            "/rag/api/documents/ingest_blocks",
            json={"filename": "a.json", "blocks": [{"text": "挂载测试内容"}]},
        )
        assert upload_resp.status_code == 200

        retrieve_resp = client.post("/rag/api/retrieve", json={"query": "挂载测试", "top_k": 3})
        assert retrieve_resp.status_code == 200

    def test_host_app_can_have_its_own_routes_alongside(self):
        """挂载不应影响宿主应用自身的路由（验证真正的"嵌入"而非"替换"）。"""
        host_app = FastAPI()

        @host_app.get("/host/ping")
        def _ping():
            return {"pong": True}

        mount_rag_api(host_app, prefix="/rag/api")
        client = TestClient(host_app)

        assert client.get("/host/ping").json() == {"pong": True}
        assert client.get("/rag/api/health").status_code == 200

    def test_default_prefix_is_rag_api(self):
        host_app = FastAPI()
        mount_rag_api(host_app)
        client = TestClient(host_app)
        assert client.get("/rag/api/health").status_code == 200
