# -*- coding: utf-8 -*-
"""rag/indexing/embedding.py 单元测试：本地哈希嵌入的确定性与 TEI 降级逻辑。"""
import math

from rag.indexing.embedding import Embedder, local_hash_embedding, LOCAL_EMBED_DIM


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class TestLocalHashEmbedding:
    def test_deterministic(self):
        v1 = local_hash_embedding("广告限流规则")
        v2 = local_hash_embedding("广告限流规则")
        assert v1 == v2

    def test_dimension(self):
        v = local_hash_embedding("测试文本", dim=64)
        assert len(v) == 64

    def test_default_dimension(self):
        v = local_hash_embedding("测试文本")
        assert len(v) == LOCAL_EMBED_DIM

    def test_empty_text_returns_zero_vector(self):
        v = local_hash_embedding("")
        assert v == [0.0] * LOCAL_EMBED_DIM

    def test_l2_normalized(self):
        v = local_hash_embedding("一段不短的测试文本用于校验归一化")
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-6

    def test_similar_texts_more_similar_than_unrelated(self):
        """包含相同关键词的文本应比完全无关文本更相似（弱语义信号，非精确语义）。"""
        v_a = local_hash_embedding("广告限流后如何解除限流")
        v_b = local_hash_embedding("广告限流的解除方法")
        v_c = local_hash_embedding("今天天气晴朗适合出门散步")

        sim_ab = _cosine(v_a, v_b)
        sim_ac = _cosine(v_a, v_c)
        assert sim_ab > sim_ac

    def test_identical_text_similarity_is_one(self):
        v = local_hash_embedding("完全相同的文本")
        assert abs(_cosine(v, v) - 1.0) < 1e-6


class TestEmbedder:
    def test_local_backend_embed_texts_batch(self):
        embedder = Embedder(backend="local")
        vectors = embedder.embed_texts(["文本一", "文本二", "文本三"])
        assert len(vectors) == 3
        assert all(len(v) == embedder.get_dim() for v in vectors)

    def test_embed_texts_empty_list(self):
        embedder = Embedder(backend="local")
        assert embedder.embed_texts([]) == []

    def test_embed_query_returns_single_vector(self):
        embedder = Embedder(backend="local")
        v = embedder.embed_query("单条查询")
        assert isinstance(v, list)
        assert len(v) == embedder.get_dim()

    def test_tei_backend_falls_back_to_local_when_unavailable(self, monkeypatch):
        """TEI 服务不可用时应自动降级为本地哈希嵌入，而不是抛异常。"""
        embedder = Embedder(backend="tei")

        class _FakeClient:
            def health_check(self, service="embed"):
                return False

        monkeypatch.setattr(embedder, "_get_tei_client", lambda: _FakeClient())
        vectors = embedder.embed_texts(["降级测试文本"])
        assert len(vectors) == 1
        assert len(vectors[0]) == embedder.get_dim()
        # 应与本地哈希嵌入结果一致（证明确实走了降级路径）
        assert vectors[0] == local_hash_embedding("降级测试文本", embedder.get_dim())
