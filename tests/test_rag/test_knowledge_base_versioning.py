# -*- coding: utf-8 -*-
"""rag/knowledge_base/versioning.py 单元测试：文档内容版本历史追踪。"""
from rag.knowledge_base.versioning import VersionStore, compute_content_hash


class TestComputeContentHash:
    def test_same_content_produces_same_hash(self):
        blocks = [{"text": "内容一"}, {"text": "内容二"}]
        assert compute_content_hash(blocks) == compute_content_hash(blocks)

    def test_different_content_produces_different_hash(self):
        h1 = compute_content_hash([{"text": "内容一"}])
        h2 = compute_content_hash([{"text": "内容二"}])
        assert h1 != h2

    def test_key_order_does_not_affect_hash(self):
        h1 = compute_content_hash([{"text": "内容", "title": "标题"}])
        h2 = compute_content_hash([{"title": "标题", "text": "内容"}])
        assert h1 == h2

    def test_plain_text_hash(self):
        assert compute_content_hash("纯文本内容") == compute_content_hash("纯文本内容")
        assert compute_content_hash("纯文本内容A") != compute_content_hash("纯文本内容B")


class TestVersionStore:
    def test_record_and_get_history(self, tmp_path):
        store = VersionStore(path=str(tmp_path / "versions.json"))
        store.record_version("doc1", content_hash="hash1", num_chunks=3, filename="a.json")
        history = store.get_history("doc1")
        assert len(history) == 1
        assert history[0]["hash"] == "hash1"
        assert history[0]["num_chunks"] == 3

    def test_multiple_versions_appended_in_order(self, tmp_path):
        store = VersionStore(path=str(tmp_path / "versions.json"))
        store.record_version("doc1", content_hash="v1", num_chunks=1)
        store.record_version("doc1", content_hash="v2", num_chunks=2)
        history = store.get_history("doc1")
        assert [h["hash"] for h in history] == ["v1", "v2"]

    def test_get_latest_hash(self, tmp_path):
        store = VersionStore(path=str(tmp_path / "versions.json"))
        assert store.get_latest_hash("doc1") is None
        store.record_version("doc1", content_hash="v1", num_chunks=1)
        store.record_version("doc1", content_hash="v2", num_chunks=1)
        assert store.get_latest_hash("doc1") == "v2"

    def test_get_history_for_unknown_doc_returns_empty(self, tmp_path):
        store = VersionStore(path=str(tmp_path / "versions.json"))
        assert store.get_history("not_exist") == []

    def test_clear_removes_history(self, tmp_path):
        store = VersionStore(path=str(tmp_path / "versions.json"))
        store.record_version("doc1", content_hash="v1", num_chunks=1)
        store.clear("doc1")
        assert store.get_history("doc1") == []

    def test_persistence_across_instances(self, tmp_path):
        path = str(tmp_path / "versions.json")
        store1 = VersionStore(path=path)
        store1.record_version("doc1", content_hash="v1", num_chunks=1)

        store2 = VersionStore(path=path)
        assert store2.get_latest_hash("doc1") == "v1"

    def test_independent_docs_do_not_interfere(self, tmp_path):
        store = VersionStore(path=str(tmp_path / "versions.json"))
        store.record_version("docA", content_hash="a1", num_chunks=1)
        store.record_version("docB", content_hash="b1", num_chunks=1)
        assert store.get_latest_hash("docA") == "a1"
        assert store.get_latest_hash("docB") == "b1"
