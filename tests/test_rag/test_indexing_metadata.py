# -*- coding: utf-8 -*-
"""rag/indexing/metadata.py 单元测试：文档级元数据登记表（DocRegistry）。"""
from rag.indexing.metadata import DocRegistry, get_registry


class TestDocRegistryBasics:
    def test_new_doc_id_is_nonempty_and_unique(self, tmp_path):
        registry = DocRegistry(path=str(tmp_path / "reg.json"))
        id1 = registry.new_doc_id()
        id2 = registry.new_doc_id()
        assert id1 and id2 and id1 != id2

    def test_next_global_ids_returns_sequential_and_unique(self, tmp_path):
        registry = DocRegistry(path=str(tmp_path / "reg.json"))
        ids1 = registry.next_global_ids(3)
        ids2 = registry.next_global_ids(2)
        assert len(ids1) == 3 and len(ids2) == 2
        assert set(ids1).isdisjoint(set(ids2))
        assert max(ids1) < min(ids2)

    def test_register_and_get(self, tmp_path):
        registry = DocRegistry(path=str(tmp_path / "reg.json"))
        meta = registry.register(doc_id="d1", filename="a.txt", num_chunks=3, chunk_ids=[0, 1, 2])
        assert meta["doc_id"] == "d1"
        assert meta["filename"] == "a.txt"
        assert meta["num_chunks"] == 3
        assert registry.get("d1") == meta

    def test_get_nonexistent_returns_none(self, tmp_path):
        registry = DocRegistry(path=str(tmp_path / "reg.json"))
        assert registry.get("not_exist") is None

    def test_list_documents_returns_all_registered(self, tmp_path):
        registry = DocRegistry(path=str(tmp_path / "reg.json"))
        registry.register(doc_id="d1", filename="a.txt", num_chunks=1, chunk_ids=[0])
        registry.register(doc_id="d2", filename="b.txt", num_chunks=2, chunk_ids=[1, 2])
        docs = registry.list_documents()
        assert len(docs) == 2

    def test_delete_removes_document(self, tmp_path):
        registry = DocRegistry(path=str(tmp_path / "reg.json"))
        registry.register(doc_id="d1", filename="a.txt", num_chunks=1, chunk_ids=[0])
        assert registry.delete("d1") is True
        assert registry.get("d1") is None
        assert registry.count_docs() == 0

    def test_delete_nonexistent_returns_false(self, tmp_path):
        registry = DocRegistry(path=str(tmp_path / "reg.json"))
        assert registry.delete("not_exist") is False

    def test_count_docs(self, tmp_path):
        registry = DocRegistry(path=str(tmp_path / "reg.json"))
        assert registry.count_docs() == 0
        registry.register(doc_id="d1", filename="a.txt", num_chunks=1, chunk_ids=[0])
        assert registry.count_docs() == 1


class TestDocRegistryPersistence:
    def test_persists_across_instances(self, tmp_path):
        path = str(tmp_path / "reg.json")
        registry1 = DocRegistry(path=path)
        registry1.register(doc_id="d1", filename="a.txt", num_chunks=1, chunk_ids=[0])

        registry2 = DocRegistry(path=path)
        assert registry2.get("d1") is not None

    def test_global_id_counter_persists_across_instances(self, tmp_path):
        """counter.txt 与 registry.json 分离持久化，跨实例仍应保证全局自增不重复。"""
        import os
        data_dir = str(tmp_path / "data")
        os.makedirs(data_dir, exist_ok=True)

        from rag.config import RAG_CONFIG
        original_dir = RAG_CONFIG["data_dir"]
        RAG_CONFIG["data_dir"] = data_dir
        try:
            registry1 = DocRegistry()
            ids1 = registry1.next_global_ids(3)
            registry2 = DocRegistry()
            ids2 = registry2.next_global_ids(3)
            assert set(ids1).isdisjoint(set(ids2))
        finally:
            RAG_CONFIG["data_dir"] = original_dir


class TestGetRegistrySingleton:
    def test_returns_same_instance(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2
