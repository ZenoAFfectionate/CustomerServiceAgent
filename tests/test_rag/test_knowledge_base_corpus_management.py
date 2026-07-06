# -*- coding: utf-8 -*-
"""rag/knowledge_base/corpus_management.py 单元测试：知识库高层编排。

组合测试：corpus_management 组合了 `indexing/index_builder.py`（写索引）+
`quality_control.py`（质检）+ `versioning.py`（版本记录），验证三者协同工作。
"""
import pytest

from rag.knowledge_base import corpus_management

pytestmark = pytest.mark.usefixtures("clean_rag_data")


class TestIngestUpload:
    def test_ingest_upload_writes_index_and_records_version(self):
        raw = ("广告限流是一种常见的风控手段。" * 5).encode("utf-8")
        meta = corpus_management.ingest_upload("faq.txt", raw)
        assert meta["filename"] == "faq.txt"
        assert meta["num_chunks"] >= 1

        history = corpus_management.get_document_history(meta["doc_id"])
        assert len(history) == 1
        assert history[0]["num_chunks"] == meta["num_chunks"]


class TestIngestBlocks:
    def test_ingest_blocks_writes_index_and_records_version(self):
        blocks = [{"text": "退款需在签收后七天内申请"}, {"text": "广告限流是常见的风控手段"}]
        meta = corpus_management.ingest_blocks(blocks, filename="kb.json")
        assert meta["num_chunks"] == 2

        history = corpus_management.get_document_history(meta["doc_id"])
        assert len(history) == 1

    def test_ingest_blocks_quality_warning_does_not_block_ingest(self, caplog):
        """质检未通过（如空文本占比过高）时应仅告警，不阻断导入（当前默认策略）。"""
        blocks = [{"text": "有效内容"}] + [{"text": "   "} for _ in range(9)]
        meta = corpus_management.ingest_blocks(blocks, filename="low_quality.json")
        assert meta["num_chunks"] == 1  # 空文本块已被 index_builder 过滤

    def test_reingest_same_doc_id_creates_second_version(self):
        doc_id = "fixed_doc_id"
        corpus_management.ingest_blocks([{"text": "版本一内容"}], filename="v.json", doc_id=doc_id)
        corpus_management.delete_document(doc_id)
        corpus_management.ingest_blocks([{"text": "版本二内容已更新"}], filename="v.json", doc_id=doc_id)
        history = corpus_management.get_document_history(doc_id)
        assert len(history) == 2

    def test_version_hash_reflects_filtered_content_not_raw_blocks(self):
        """回归测试：修复审查报告 L13/L14——版本哈希应基于实际入库的内容块
        （过滤空文本），使同一内容的哈希在多次导入间稳定。"""
        from rag.knowledge_base.versioning import compute_content_hash
        from rag.indexing.index_builder import filter_non_empty_blocks

        blocks_with_empty = [{"text": "有效内容"}, {"text": "   "}, {"text": ""}]
        blocks_clean = [{"text": "有效内容"}]

        meta1 = corpus_management.ingest_blocks(blocks_with_empty, filename="a.json", doc_id="doc_a")
        history1 = corpus_management.get_document_history("doc_a")
        expected_hash = compute_content_hash(filter_non_empty_blocks(blocks_with_empty))

        assert meta1["num_chunks"] == 1  # 空文本块已被过滤，实际只入库 1 条
        assert history1[-1]["hash"] == expected_hash
        assert expected_hash == compute_content_hash(filter_non_empty_blocks(blocks_clean))



class TestDocumentManagement:
    def test_list_documents_and_get_document(self):
        meta = corpus_management.ingest_blocks([{"text": "内容"}], filename="a.json")
        docs = corpus_management.list_documents()
        assert len(docs) == 1
        assert corpus_management.get_document(meta["doc_id"])["filename"] == "a.json"

    def test_get_document_history_empty_for_unknown_doc(self):
        assert corpus_management.get_document_history("not_exist") == []

    def test_delete_document_removes_from_index(self):
        meta = corpus_management.ingest_blocks([{"text": "待删除内容"}], filename="a.json")
        assert corpus_management.delete_document(meta["doc_id"]) is True
        assert corpus_management.list_documents() == []

    def test_get_corpus_stats_reflects_ingested_data(self):
        corpus_management.ingest_blocks([{"text": "内容一"}, {"text": "内容二"}], filename="a.json")
        stats = corpus_management.get_corpus_stats()
        assert stats["num_documents"] == 1
        assert stats["num_vector_chunks"] == 2
