# -*- coding: utf-8 -*-
"""rag/generation/citation.py 单元测试：引用列表构建。"""
from rag.generation.citation import build_citations
from rag.schema import DocBlock


def _ctx(idx, title, text, page_url="http://x", block_path="", score=0.8):
    return DocBlock(global_chunk_idx=idx, title=title, text=text, page_url=page_url,
                     block_path=block_path, score=score)


class TestBuildCitations:
    def test_empty_contexts_returns_empty_list(self):
        assert build_citations([]) == []

    def test_citation_index_starts_from_one(self):
        contexts = [_ctx(1, "A", "内容A"), _ctx(2, "B", "内容B")]
        citations = build_citations(contexts)
        assert citations[0]["index"] == 1
        assert citations[1]["index"] == 2

    def test_citation_fields_complete(self):
        contexts = [_ctx(1, "标题A", "内容A", page_url="http://example.com/a",
                          block_path="html>body>p", score=0.756789)]
        citation = build_citations(contexts)[0]
        assert citation["page_url"] == "http://example.com/a"
        assert citation["block_path"] == "html>body>p"
        assert citation["title"] == "标题A"

    def test_score_rounded_to_four_decimals(self):
        contexts = [_ctx(1, "A", "内容", score=0.123456789)]
        citation = build_citations(contexts)[0]
        assert citation["score"] == 0.1235

    def test_citation_order_matches_context_order(self):
        contexts = [_ctx(i, f"标题{i}", f"内容{i}") for i in range(5)]
        citations = build_citations(contexts)
        assert [c["title"] for c in citations] == [f"标题{i}" for i in range(5)]
