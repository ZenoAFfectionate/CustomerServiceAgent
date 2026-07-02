# -*- coding: utf-8 -*-
"""rag/indexing/local_vector_index.py 单元测试。"""
import os

from rag.indexing.local_vector_index import LocalVectorStore
from rag.schema import DocBlock


def _make_block(gid, doc_id, text, embedding):
    return DocBlock(text=text, global_chunk_idx=gid, doc_id=doc_id, embedding=embedding)


class TestLocalVectorStore:
    def test_upsert_and_count(self, tmp_path):
        store = LocalVectorStore(path=str(tmp_path / "vs.json"))
        blocks = [_make_block(0, "doc1", "文本A", [1.0, 0.0, 0.0]),
                  _make_block(1, "doc1", "文本B", [0.0, 1.0, 0.0])]
        n = store.upsert(blocks)
        assert n == 2
        assert store.count() == 2

    def test_upsert_is_idempotent_on_same_id(self, tmp_path):
        store = LocalVectorStore(path=str(tmp_path / "vs.json"))
        store.upsert([_make_block(0, "doc1", "版本1", [1.0, 0.0])])
        store.upsert([_make_block(0, "doc1", "版本2", [0.0, 1.0])])
        assert store.count() == 1

    def test_search_returns_most_similar_first(self, tmp_path):
        store = LocalVectorStore(path=str(tmp_path / "vs.json"))
        store.upsert([
            _make_block(0, "doc1", "完全相同方向", [1.0, 0.0, 0.0]),
            _make_block(1, "doc1", "正交方向", [0.0, 1.0, 0.0]),
            _make_block(2, "doc1", "反方向", [-1.0, 0.0, 0.0]),
        ])
        results = store.search([1.0, 0.0, 0.0], top_k=3)
        assert len(results) == 3
        assert results[0].global_chunk_idx == 0
        assert results[0].score > results[1].score > results[2].score

    def test_search_respects_top_k(self, tmp_path):
        store = LocalVectorStore(path=str(tmp_path / "vs.json"))
        store.upsert([_make_block(i, "doc1", f"文本{i}", [float(i), 1.0]) for i in range(5)])
        results = store.search([1.0, 1.0], top_k=2)
        assert len(results) == 2

    def test_search_on_empty_store_returns_empty(self, tmp_path):
        store = LocalVectorStore(path=str(tmp_path / "vs.json"))
        assert store.search([1.0, 0.0], top_k=5) == []

    def test_delete_by_doc_id(self, tmp_path):
        store = LocalVectorStore(path=str(tmp_path / "vs.json"))
        store.upsert([
            _make_block(0, "docA", "A的块", [1.0, 0.0]),
            _make_block(1, "docB", "B的块", [0.0, 1.0]),
        ])
        removed = store.delete_by_doc_id("docA")
        assert removed == 1
        assert store.count() == 1
        remaining = store.search([0.0, 1.0], top_k=5)
        assert all(b.doc_id != "docA" for b in remaining)

    def test_delete_nonexistent_doc_id_returns_zero(self, tmp_path):
        store = LocalVectorStore(path=str(tmp_path / "vs.json"))
        store.upsert([_make_block(0, "docA", "A", [1.0])])
        assert store.delete_by_doc_id("not_exist") == 0

    def test_persistence_across_instances(self, tmp_path):
        path = str(tmp_path / "vs.json")
        store1 = LocalVectorStore(path=path)
        store1.upsert([_make_block(0, "docA", "持久化测试", [1.0, 0.0])])

        store2 = LocalVectorStore(path=path)
        assert store2.count() == 1

    def test_health_check_always_true(self, tmp_path):
        store = LocalVectorStore(path=str(tmp_path / "vs.json"))
        assert store.health_check() is True

    def test_create_collection_creates_dir(self, tmp_path):
        nested_path = str(tmp_path / "nested" / "vs.json")
        store = LocalVectorStore(path=nested_path)
        store.create_collection(dim=8)
        assert os.path.isdir(os.path.dirname(nested_path))
