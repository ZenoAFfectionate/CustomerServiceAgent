# -*- coding: utf-8 -*-
"""性能优化回归测试：TF-IDF 矩阵缓存、numpy 矩阵缓存、轻量去重。

覆盖 indexing/ 存储后端与 retrieval/hybrid_search.py 融合去重之间的组合行为：
    1. LocalKeywordStore: TF-IDF 矩阵在 upsert 后失效、search 后缓存复用
    2. LocalVectorStore: numpy 矩阵在 upsert 后失效、search 后缓存复用
    3. hybrid_search.deduplicate: global_chunk_idx 精确去重 + 阈值开关
"""
import json
import time

import numpy as np
import pytest

from rag.indexing.local_index import LocalKeywordStore, LocalVectorStore
from rag.retrieval.hybrid_search import deduplicate, fuse, rrf_fuse
from rag.schema import DocBlock


# ======================== 1. LocalKeywordStore TF-IDF 缓存 ========================


class TestTfidfCache:
    """验证 TF-IDF 矩阵缓存机制的正确性。"""

    def test_search_after_upsert_uses_cached_tfidf(self, tmp_path):
        """首次 search 构建 TF-IDF 后，第二次 search 应直接复用缓存。"""
        store = LocalKeywordStore(path=str(tmp_path / "ks.json"))
        store.upsert([
            DocBlock(text="广告限流规则内容", global_chunk_idx=0, doc_id="d1", title="广告限流"),
            DocBlock(text="退款政策说明", global_chunk_idx=1, doc_id="d1", title="退款"),
        ])
        # 第一次 search：构建 TF-IDF
        r1 = store.search("广告限流", top_k=2)
        assert len(r1) >= 1
        # 缓存应已构建
        assert store._tfidf_dirty is False
        assert store._doc_matrix is not None
        # 第二次 search：复用缓存，结果应一致
        r2 = store.search("广告限流", top_k=2)
        assert len(r2) == len(r1)
        assert r1[0].global_chunk_idx == r2[0].global_chunk_idx

    def test_upsert_marks_tfidf_dirty(self, tmp_path):
        """upsert 后 TF-IDF 缓存应失效，下次 search 重建。"""
        store = LocalKeywordStore(path=str(tmp_path / "ks.json"))
        store.upsert([DocBlock(text="初始内容", global_chunk_idx=0, doc_id="d1", title="初始")])
        store.search("初始", top_k=1)
        assert store._tfidf_dirty is False  # 缓存有效

        store.upsert([DocBlock(text="新增内容", global_chunk_idx=1, doc_id="d2", title="新增")])
        assert store._tfidf_dirty is True  # 写入后缓存失效

        # 再次 search 应重建并包含新数据
        results = store.search("新增", top_k=5)
        assert any(r.global_chunk_idx == 1 for r in results)
        assert store._tfidf_dirty is False  # 重建后缓存有效

    def test_delete_marks_tfidf_dirty(self, tmp_path):
        """delete_by_doc_id 后 TF-IDF 缓存应失效。"""
        store = LocalKeywordStore(path=str(tmp_path / "ks.json"))
        store.upsert([
            DocBlock(text="待删除内容", global_chunk_idx=0, doc_id="d1", title="删除"),
            DocBlock(text="保留内容", global_chunk_idx=1, doc_id="d2", title="保留"),
        ])
        store.search("内容", top_k=5)
        assert store._tfidf_dirty is False

        store.delete_by_doc_id("d1")
        assert store._tfidf_dirty is True

        # 重建后不应再命中已删除的块
        results = store.search("待删除", top_k=5)
        assert all(r.doc_id != "d1" for r in results)

    def test_cached_search_is_consistent_with_cold_search(self, tmp_path):
        """缓存查询的结果应与冷启动查询完全一致（无数据漂移）。"""
        store = LocalKeywordStore(path=str(tmp_path / "ks.json"))
        store.upsert([
            DocBlock(text=f"文档内容第{i}条关于退款", global_chunk_idx=i, doc_id="d1", title=f"退款{i}")
            for i in range(20)
        ])
        query = "退款怎么申请"
        cold = store.search(query, top_k=10)  # 冷启动
        warm = store.search(query, top_k=10)  # 缓存
        assert len(cold) == len(warm)
        for c, w in zip(cold, warm):
            assert c.global_chunk_idx == w.global_chunk_idx
            assert abs(c.score - w.score) < 1e-10

    def test_different_query_uses_same_cached_matrix(self, tmp_path):
        """不同 query 应复用同一 TF-IDF 矩阵，仅 transform query 向量。"""
        store = LocalKeywordStore(path=str(tmp_path / "ks.json"))
        store.upsert([
            DocBlock(text="广告投放策略", global_chunk_idx=0, doc_id="d1", title="广告"),
            DocBlock(text="物流配送时效", global_chunk_idx=1, doc_id="d2", title="物流"),
        ])
        r1 = store.search("广告", top_k=2)
        matrix_ref_1 = id(store._doc_matrix)
        r2 = store.search("物流", top_k=2)
        matrix_ref_2 = id(store._doc_matrix)
        # 矩阵对象应未被替换（同一个 Python 对象）
        assert matrix_ref_1 == matrix_ref_2

    def test_empty_data_does_not_crash_ensure_tfidf(self, tmp_path):
        """空知识库时 _ensure_tfidf 不应崩溃。"""
        store = LocalKeywordStore(path=str(tmp_path / "ks.json"))
        assert store.search("任意", top_k=5) == []
        assert store._doc_matrix is None
        assert store._vectorizer is None


# ======================== 2. LocalVectorStore numpy 矩阵缓存 ========================


class TestVectorMatrixCache:
    """验证 numpy 矩阵缓存机制的正确性。"""

    def test_search_after_upsert_uses_cached_matrix(self, tmp_path):
        """首次 search 构建矩阵后，第二次 search 直接复用。"""
        store = LocalVectorStore(path=str(tmp_path / "vs.json"))
        store.upsert([
            DocBlock(text="文本A", global_chunk_idx=0, doc_id="d1", embedding=[1.0, 0.0, 0.0]),
            DocBlock(text="文本B", global_chunk_idx=1, doc_id="d1", embedding=[0.0, 1.0, 0.0]),
        ])
        r1 = store.search([1.0, 0.0, 0.0], top_k=2)
        assert store._matrix_dirty is False
        assert store._matrix is not None
        r2 = store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(r1) == len(r2)
        assert r1[0].global_chunk_idx == r2[0].global_chunk_idx

    def test_upsert_marks_matrix_dirty(self, tmp_path):
        """upsert 后 numpy 矩阵缓存应失效。"""
        store = LocalVectorStore(path=str(tmp_path / "vs.json"))
        store.upsert([DocBlock(text="初始", global_chunk_idx=0, doc_id="d1", embedding=[1.0, 0.0])])
        store.search([1.0, 0.0], top_k=1)
        assert store._matrix_dirty is False

        store.upsert([DocBlock(text="新增", global_chunk_idx=1, doc_id="d2", embedding=[0.0, 1.0])])
        assert store._matrix_dirty is True

        results = store.search([0.0, 1.0], top_k=5)
        assert any(r.global_chunk_idx == 1 for r in results)
        assert store._matrix_dirty is False

    def test_delete_marks_matrix_dirty(self, tmp_path):
        """delete_by_doc_id 后 numpy 矩阵缓存应失效。"""
        store = LocalVectorStore(path=str(tmp_path / "vs.json"))
        store.upsert([
            DocBlock(text="删除", global_chunk_idx=0, doc_id="dA", embedding=[1.0, 0.0]),
            DocBlock(text="保留", global_chunk_idx=1, doc_id="dB", embedding=[0.0, 1.0]),
        ])
        store.search([1.0, 0.0], top_k=5)
        assert store._matrix_dirty is False

        store.delete_by_doc_id("dA")
        assert store._matrix_dirty is True

        results = store.search([1.0, 0.0], top_k=5)
        assert all(r.doc_id != "dA" for r in results)

    def test_cached_matrix_shape_matches_data(self, tmp_path):
        """缓存矩阵的 shape 应与数据量、维度一致。"""
        store = LocalVectorStore(path=str(tmp_path / "vs.json"))
        store.upsert([
            DocBlock(text=f"块{i}", global_chunk_idx=i, doc_id="d1", embedding=[float(i), float(i+1), 0.0])
            for i in range(10)
        ])
        store.search([1.0, 2.0, 0.0], top_k=5)
        assert store._matrix.shape == (10, 3)

    def test_search_dimension_mismatch_returns_empty(self, tmp_path):
        """query 向量维度与存储不一致时应安全返回空。"""
        store = LocalVectorStore(path=str(tmp_path / "vs.json"))
        store.upsert([DocBlock(text="3维", global_chunk_idx=0, doc_id="d1", embedding=[1.0, 0.0, 0.0])])
        assert store.search([1.0, 0.0], top_k=5) == []  # 2维 vs 3维

    def test_matrix_rebuilt_correctly_after_batch_upsert(self, tmp_path):
        """连续多批 upsert 后矩阵应正确反映全部数据。"""
        store = LocalVectorStore(path=str(tmp_path / "vs.json"))
        for batch_start in range(0, 30, 10):
            store.upsert([
                DocBlock(text=f"块{batch_start+i}", global_chunk_idx=batch_start+i, doc_id="d1",
                        embedding=[float(batch_start+i), 1.0, 0.0])
                for i in range(10)
            ])
        results = store.search([29.0, 1.0, 0.0], top_k=5)
        assert len(results) >= 1
        assert results[0].global_chunk_idx == 29  # 最接近 [29, 1, 0]
        assert store._matrix.shape[0] == 30


# ======================== 3. hybrid_search.deduplicate 轻量去重 ========================


def _block(gid, score, text=None, doc_id="doc1"):
    return DocBlock(
        global_chunk_idx=gid, text=text or f"内容{gid}", score=score,
        doc_id=doc_id, page_name=f"page_{gid}",
    )


class TestLightweightDedup:
    """验证轻量去重的正确性。"""

    def test_exact_dedup_by_global_chunk_idx(self):
        """同 global_chunk_idx 的块应被精确去重，仅保留分数最高的一份。"""
        blocks = [
            _block(1, 0.8, "相同ID的内容A"),
            _block(1, 0.5, "相同ID的内容B"),  # 同 ID，低分，应被去重
            _block(2, 0.7, "不同ID的内容"),
        ]
        result = deduplicate(blocks)
        assert len(result) == 2
        ids = {b.global_chunk_idx for b in result}
        assert ids == {1, 2}
        # 保留分数最高的那份
        block_1 = [b for b in result if b.global_chunk_idx == 1][0]
        assert block_1.score == 0.8

    def test_single_block_passthrough(self):
        """单条 block 直接返回。"""
        result = deduplicate([_block(0, 0.5)])
        assert len(result) == 1

    def test_empty_list_passthrough(self):
        assert deduplicate([]) == []

    def test_all_unique_ids_preserved(self):
        """全部不同 ID 的块应全部保留。"""
        blocks = [_block(i, 0.5, f"独特内容{i}") for i in range(20)]
        result = deduplicate(blocks)
        assert len(result) == 20

    def test_fuse_uses_lightweight_dedup_by_default(self):
        """fuse() 默认应走轻量去重路径（不调用 process/ TF-IDF 去重）。"""
        blocks_a = [_block(0, 0.9, "向量检索结果"), _block(1, 0.7, "向量检索结果2")]
        blocks_b = [_block(0, 0.8, "关键词检索结果"), _block(2, 0.6, "关键词检索结果2")]
        fused = fuse([blocks_a, blocks_b])
        # 同 ID=0 的块应被合并为一份
        ids = [b.global_chunk_idx for b in fused]
        assert ids.count(0) == 1
        assert set(ids) == {0, 1, 2}
