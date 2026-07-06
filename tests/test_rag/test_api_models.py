# -*- coding: utf-8 -*-
"""rag/api/models.py 单元测试：Pydantic 响应模型的字段默认值兜底。

回归测试审查报告 L16：`ContextItem`/`CitationItem` 此前多数字段无默认值
（必填），与真实检索后端返回的记录字段完整性脆弱耦合——一旦缺字段就会
直接 422，而非优雅降级。
"""
from rag.api.models import CitationItem, ContextItem


class TestContextItemDefaults:
    def test_minimal_fields_do_not_raise(self):
        """仅提供 text 时其余字段应全部使用默认值，不抛校验异常。"""
        item = ContextItem(text="部分字段缺失的记录")
        assert item.global_chunk_idx == -1
        assert item.doc_id == ""
        assert item.chunk_idx == 0
        assert item.page_name == ""
        assert item.title == ""
        assert item.page_url == ""
        assert item.score == 0.0
        assert item.source_retriever == ""

    def test_full_fields_still_work(self):
        item = ContextItem(
            global_chunk_idx=1, doc_id="d1", chunk_idx=0, page_name="p",
            title="t", page_url="http://x", text="内容", score=0.9,
            source_retriever="milvus",
        )
        assert item.global_chunk_idx == 1
        assert item.score == 0.9


class TestCitationItemDefaults:
    def test_minimal_fields_do_not_raise(self):
        item = CitationItem(index=1, page_url="http://x")
        assert item.score == 0.0
        assert item.title == ""
        assert item.block_path == ""
