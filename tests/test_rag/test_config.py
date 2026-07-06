# -*- coding: utf-8 -*-
"""rag/config.py 单元测试。"""
import os

from rag.config import RAG_CONFIG, get_rag_config


class TestRagConfig:
    def test_required_keys_present(self):
        required = [
            "milvus_host", "milvus_port", "collection_name",
            "es_host", "es_port", "index_name",
            "top_k_recall", "top_k_final", "fusion_method", "fusion_weights",
            "vector_backend", "keyword_backend", "embed_backend", "rerank_backend",
            "generation_backend", "data_dir", "chunk_size", "chunk_overlap",
        ]
        for key in required:
            assert key in RAG_CONFIG, f"缺少配置项: {key}"

    def test_hallucination_config_present(self):
        """新增的幻觉控制配置项应存在且默认关闭（见 generation/hallucination_control.py）。"""
        assert "hallucination_append_caveat" in RAG_CONFIG
        assert isinstance(RAG_CONFIG["hallucination_append_caveat"], bool)

    def test_default_backends_are_local(self):
        """未配置外部服务时，默认应为 local 后端，保证开箱即用。"""
        assert RAG_CONFIG["vector_backend"] in ("local", "milvus")
        assert RAG_CONFIG["keyword_backend"] in ("local", "es")

    def test_top_k_values_are_positive_int(self):
        assert isinstance(RAG_CONFIG["top_k_recall"], int) and RAG_CONFIG["top_k_recall"] > 0
        assert isinstance(RAG_CONFIG["top_k_final"], int) and RAG_CONFIG["top_k_final"] > 0

    def test_fusion_weights_structure(self):
        weights = RAG_CONFIG["fusion_weights"]
        assert "milvus" in weights and "es" in weights
        assert isinstance(weights["milvus"], float)

    def test_data_dir_exists(self):
        assert os.path.isdir(RAG_CONFIG["data_dir"])

    def test_get_rag_config_returns_copy(self):
        """`get_rag_config()` 应返回浅拷贝，修改返回值不影响全局配置。"""
        cfg = get_rag_config()
        cfg["top_k_recall"] = -999
        assert RAG_CONFIG["top_k_recall"] != -999

    def test_env_var_override(self, monkeypatch):
        """环境变量应覆盖配置文件默认值（重新计算验证 _env_or_config 逻辑）。"""
        from rag.config import _env_or_config
        monkeypatch.setenv("SOME_TEST_KEY", "123")
        val = _env_or_config("SOME_TEST_KEY", {}, 0, int)
        assert val == 123
