# -*- coding: utf-8 -*-
"""rag/indexing/index_builder.py 单元测试：文档上传/批量导入/删除/列表/统计全流程（本地降级后端）。"""
import pytest

from rag.indexing import index_builder as indexer

pytestmark = pytest.mark.usefixtures("clean_rag_data")


class TestIngestFile:
    def test_ingest_txt_file(self):
        raw = "广告限流是指账户存在异常投放行为时，系统自动降低广告曝光量的一种风控手段。" * 3
        meta = indexer.ingest_file("ad_faq.txt", raw.encode("utf-8"))
        assert meta["filename"] == "ad_faq.txt"
        assert meta["num_chunks"] >= 1
        assert len(meta["chunk_ids"]) == meta["num_chunks"]
        assert meta["doc_id"]

    def test_ingest_updates_stats(self):
        raw = "测试文档内容。" * 5
        indexer.ingest_file("t1.txt", raw.encode("utf-8"))
        stats = indexer.get_stats()
        assert stats["num_documents"] == 1
        assert stats["num_vector_chunks"] >= 1
        assert stats["num_keyword_chunks"] >= 1

    def test_ingest_empty_file_raises(self):
        with pytest.raises(ValueError):
            indexer.ingest_file("empty.txt", b"   ")

    def test_ingest_unsupported_extension_raises(self):
        with pytest.raises(ValueError):
            indexer.ingest_file("file.exe", b"binary content")

    def test_ingest_json_blocks_file(self):
        import json
        blocks = [
            {"text": "测试内容一", "title": "标题一", "page_url": "http://a"},
            {"text": "测试内容二", "title": "标题二", "page_url": "http://b"},
        ]
        raw = json.dumps(blocks, ensure_ascii=False).encode("utf-8")
        meta = indexer.ingest_file("blocks.json", raw)
        assert meta["num_chunks"] == 2

    def test_ingest_malformed_json_raises_value_error_not_crash(self):
        """回归测试：修复 ParseError 未被捕获导致 500 的 Bug（见 index_builder.py 注释）。

        原实现会让 `document_loader.ParseError` 穿透到调用方，本测试确认现在统一转换为
        ValueError（API 层据此映射为 422，而不是未处理异常导致的 500）。
        """
        with pytest.raises(ValueError):
            indexer.ingest_file("broken.json", b"{not valid json!!!")

    def test_ingest_json_single_object_wrapped_as_list(self):
        import json
        raw = json.dumps({"text": "单个对象也应被包装为长度 1 的数组正常处理"}).encode("utf-8")
        meta = indexer.ingest_file("single.json", raw)
        assert meta["num_chunks"] == 1

    def test_ingest_file_empty_filename_raises(self):
        with pytest.raises(ValueError):
            indexer.ingest_file("", b"content")
        with pytest.raises(ValueError):
            indexer.ingest_file(None, b"content")

    def test_ingest_json_blocks_with_empty_text_are_filtered(self):
        """回归测试：修复空 text 块未被过滤直接写入索引的 Bug。"""
        import json
        blocks = [
            {"text": "有效内容", "title": "标题一"},
            {"text": "   ", "title": "空白内容"},
            {"text": "", "title": "空字符串"},
        ]
        raw = json.dumps(blocks, ensure_ascii=False).encode("utf-8")
        meta = indexer.ingest_file("mixed.json", raw)
        assert meta["num_chunks"] == 1

    def test_ingest_json_blocks_all_empty_text_raises(self):
        import json
        blocks = [{"text": "", "title": "A"}, {"text": "   ", "title": "B"}]
        raw = json.dumps(blocks, ensure_ascii=False).encode("utf-8")
        with pytest.raises(ValueError):
            indexer.ingest_file("all_empty.json", raw)

    def test_ingest_json_blocks_missing_chunk_idx_backfilled_by_position(self):
        """回归测试：修复未提供 chunk_idx 时全部默认为 0 的 Bug（现按位置回填）。"""
        import json
        blocks = [{"text": "第一块"}, {"text": "第二块"}, {"text": "第三块"}]
        raw = json.dumps(blocks, ensure_ascii=False).encode("utf-8")
        indexer.ingest_file("no_chunk_idx.json", raw)
        docs = indexer.list_documents()
        assert docs[0]["num_chunks"] == 3


class TestIngestBlocks:
    def test_ingest_blocks_directly(self):
        blocks = [{"text": "直接导入的知识块", "title": "标题", "page_url": "http://x", "block_path": "html>body>p"}]
        meta = indexer.ingest_blocks(blocks, filename="manual_test.json")
        assert meta["num_chunks"] == 1
        assert meta["filename"] == "manual_test.json"

    def test_ingest_empty_blocks_raises(self):
        with pytest.raises(ValueError):
            indexer.ingest_blocks([])

    def test_ingest_blocks_all_empty_text_raises(self):
        with pytest.raises(ValueError):
            indexer.ingest_blocks([{"text": ""}, {"text": "   "}])

    def test_ingest_blocks_exceeding_max_count_raises(self, monkeypatch):
        from rag.config import RAG_CONFIG
        monkeypatch.setitem(RAG_CONFIG, "max_blocks_per_ingest", 2)
        blocks = [{"text": f"块{i}"} for i in range(3)]
        with pytest.raises(ValueError):
            indexer.ingest_blocks(blocks)


class TestDocumentManagement:
    def test_list_documents_after_ingest(self):
        indexer.ingest_file("a.txt", ("内容A" * 10).encode("utf-8"))
        indexer.ingest_file("b.txt", ("内容B" * 10).encode("utf-8"))
        docs = indexer.list_documents()
        assert len(docs) == 2
        filenames = {d["filename"] for d in docs}
        assert filenames == {"a.txt", "b.txt"}

    def test_delete_document_removes_from_all_stores(self):
        meta = indexer.ingest_file("to_delete.txt", ("待删除的内容" * 10).encode("utf-8"))
        doc_id = meta["doc_id"]

        deleted = indexer.delete_document(doc_id)
        assert deleted is True
        assert indexer.list_documents() == []
        stats = indexer.get_stats()
        assert stats["num_vector_chunks"] == 0
        assert stats["num_keyword_chunks"] == 0

    def test_delete_nonexistent_document_returns_false(self):
        assert indexer.delete_document("not_exist_id") is False


class TestWriteBlocksAtomicity:
    """回归测试：修复审查报告 H3——索引写入非原子、缺回滚补偿。

    `_write_blocks` 分三步（向量库 → 关键词库 → 登记表），任一后续环节失败时
    应回滚已成功写入的前序环节，不留下"向量有/关键词无"之类的孤儿数据。
    """

    def test_keyword_store_failure_rolls_back_vector_store(self, monkeypatch):
        keyword_store = indexer.get_keyword_store()

        def _boom_upsert(blocks):
            raise RuntimeError("关键词库写入失败（模拟 ES 网络抖动）")

        monkeypatch.setattr(keyword_store, "upsert", _boom_upsert)

        with pytest.raises(RuntimeError):
            indexer.ingest_file("boom.txt", ("待写入内容" * 10).encode("utf-8"))

        # 向量库应已被回滚，不留下孤儿数据；registry 也不应有该文档记录
        assert indexer.list_documents() == []
        assert indexer.get_stats()["num_vector_chunks"] == 0

    def test_registry_failure_rolls_back_vector_and_keyword_store(self, monkeypatch):
        registry = indexer.get_registry()

        def _boom_register(**kwargs):
            raise RuntimeError("登记表写入失败（模拟磁盘故障）")

        monkeypatch.setattr(registry, "register", _boom_register)

        with pytest.raises(RuntimeError):
            indexer.ingest_file("boom2.txt", ("待写入内容" * 10).encode("utf-8"))

        # 向量库/关键词库应均已被回滚，不留下孤儿数据
        stats = indexer.get_stats()
        assert stats["num_vector_chunks"] == 0
        assert stats["num_keyword_chunks"] == 0


class TestDeleteDocumentPartialFailure:
    """回归测试：H3——delete_document 在存储环节部分失败时应抛出异常，
    而不是静默返回 False/True 误导调用方（掩盖数据不一致）。"""

    def test_partial_store_failure_raises_instead_of_silently_returning(self, monkeypatch):
        meta = indexer.ingest_file("to_delete2.txt", ("待删除的内容" * 10).encode("utf-8"))
        doc_id = meta["doc_id"]

        vector_store = indexer.get_vector_store()

        def _boom_delete(_doc_id):
            raise RuntimeError("向量库删除失败（模拟网络异常）")

        monkeypatch.setattr(vector_store, "delete_by_doc_id", _boom_delete)
        with pytest.raises(RuntimeError):
            indexer.delete_document(doc_id)
