# -*- coding: utf-8 -*-
"""rag/schema.py 单元测试：DocBlock 数据结构与 Milvus/ES schema 映射。"""
import pytest

from rag.schema import DocBlock, get_es_mapping, get_es_mapping_fallback


class TestDocBlock:
    def test_default_values(self):
        block = DocBlock(text="你好")
        assert block.text == "你好"
        assert block.global_chunk_idx == -1
        assert block.score == 0.0
        assert block.embedding is None

    def test_to_dict_excludes_embedding_by_default(self):
        block = DocBlock(text="hi", embedding=[0.1, 0.2])
        d = block.to_dict()
        assert "embedding" not in d
        assert d["text"] == "hi"

    def test_to_dict_with_embedding(self):
        block = DocBlock(text="hi", embedding=[0.1, 0.2])
        d = block.to_dict(with_embedding=True)
        assert d["embedding"] == [0.1, 0.2]

    def test_from_dict_ignores_unknown_fields(self):
        d = {"text": "abc", "global_chunk_idx": 5, "unknown_field": "xxx"}
        block = DocBlock.from_dict(d)
        assert block.text == "abc"
        assert block.global_chunk_idx == 5

    def test_from_dict_round_trip(self):
        original = DocBlock(text="round trip", title="T", page_url="http://x", score=0.5)
        restored = DocBlock.from_dict(original.to_dict())
        assert restored.text == original.text
        assert restored.title == original.title
        assert restored.score == original.score

    def test_dedup_key_text_fallback(self):
        assert DocBlock(text="正文").dedup_key_text() == "正文"
        assert DocBlock(text="", summary="摘要").dedup_key_text() == "摘要"
        assert DocBlock(text="", summary="", title="标题").dedup_key_text() == "标题"
        assert DocBlock(text="", summary="", title="").dedup_key_text() == ""


class TestMilvusSchema:
    def test_get_milvus_schema_fields(self):
        pymilvus = pytest.importorskip("pymilvus")
        from rag.schema import get_milvus_schema_fields

        fields = get_milvus_schema_fields(embedding_dim=128)
        names = [f.name for f in fields]
        assert "global_chunk_idx" in names
        assert "embedding" in names
        pk_fields = [f for f in fields if f.is_primary]
        assert len(pk_fields) == 1
        assert pk_fields[0].name == "global_chunk_idx"


class TestESMapping:
    def test_es_mapping_has_all_fields(self):
        mapping = get_es_mapping()
        props = mapping["mappings"]["properties"]
        for f in ["global_chunk_idx", "doc_id", "text", "title", "page_url", "block_path", "time"]:
            assert f in props

    def test_es_mapping_uses_ik_analyzer(self):
        mapping = get_es_mapping()
        assert mapping["mappings"]["properties"]["text"]["analyzer"] == "ik_max_word"

    def test_es_mapping_fallback_no_ik(self):
        mapping = get_es_mapping_fallback()
        assert "analyzer" not in mapping["mappings"]["properties"]["text"]
