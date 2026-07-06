# -*- coding: utf-8 -*-
"""rag/integration/deployment.py 单元测试：部署就绪检查。

组合测试：check_deployment_readiness 聚合了 indexing/store.py（vector_store/
keyword_store）、embedding/rerank/generation 各组件的 health_check，验证本地
降级后端下全部就绪。
"""
from rag.integration.deployment import check_deployment_readiness


class TestCheckDeploymentReadiness:
    def test_local_backend_is_fully_ready(self):
        """默认本地降级后端下，全部组件应无需外部服务即可就绪。"""
        report = check_deployment_readiness()
        assert report["ready"] is True
        assert all(c["healthy"] for c in report["checks"])

    def test_report_includes_all_components(self):
        report = check_deployment_readiness()
        components = {c["component"] for c in report["checks"]}
        assert components == {"vector_store", "keyword_store", "embedder", "reranker", "generation"}

    def test_each_check_has_expected_fields(self):
        report = check_deployment_readiness()
        for check in report["checks"]:
            assert set(["component", "backend", "healthy", "detail"]).issubset(check.keys())

    def test_broken_vector_store_marks_not_ready(self, monkeypatch):
        """任一组件故障应反映到整体 ready=False，而不是被掩盖。"""
        import rag.integration.deployment as deployment_mod

        def _broken_get_vector_store(*a, **kw):
            raise RuntimeError("模拟向量库连接失败")

        monkeypatch.setattr(
            "rag.indexing.store.get_vector_store", _broken_get_vector_store,
        )
        report = check_deployment_readiness()
        assert report["ready"] is False
        vector_check = next(c for c in report["checks"] if c["component"] == "vector_store")
        assert vector_check["healthy"] is False
        assert vector_check["detail"]
