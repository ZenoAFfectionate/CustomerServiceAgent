# -*- coding: utf-8 -*-
"""
text_process 扩展测试 — 覆盖重构后的共享函数和边界场景。

测试重点：
    - _row_to_text 表格行转换
    - _count_words_in_text 词数统计
    - _generate_summary_and_question 统一生成逻辑
    - _make_chunk_dict 标准块构造
    - _split_table_into_chunks 表格切分
    - deduplicate_ranked_blocks_pal 大集群去重（迭代式 BFS）
    - generate_block_documents 无摘要模式

运行方式：
    PYTHONPATH=src pytest tests/test_text_process_extended.py -v
"""

import os
import json
import tempfile

import pytest
from bs4 import BeautifulSoup

from text_process import (
    clean_text,
    clean_invisible,
    extract_title_from_block,
    build_optimal_jieba_query,
    parse_time,
    str_sim,
    deduplicate_ranked_blocks_pal,
    save_doc_meta_to_block_dir,
    generate_block_documents,
    _row_to_text,
    _count_words_in_text,
    _generate_summary_and_question,
    _make_chunk_dict,
    _split_table_into_chunks,
)
from html_utils import build_block_tree, clean_html


# ======================== _row_to_text ========================

class TestRowToText:
    """测试表格行转文本。"""

    def test_simple_row(self):
        html = "<tr><td>A</td><td>B</td><td>C</td></tr>"
        row = BeautifulSoup(html, "html.parser").find("tr")
        text = _row_to_text(row)
        assert "A" in text
        assert "B" in text
        assert "C" in text
        assert text.endswith("\n")

    def test_row_with_th(self):
        html = "<tr><th>Header</th></tr>"
        row = BeautifulSoup(html, "html.parser").find("tr")
        text = _row_to_text(row)
        assert "Header" in text

    def test_empty_row(self):
        html = "<tr></tr>"
        row = BeautifulSoup(html, "html.parser").find("tr")
        text = _row_to_text(row)
        assert text == "\n"


# ======================== _count_words_in_text ========================

class TestCountWordsInText:
    """测试词数统计。"""

    def test_chinese_text(self):
        assert _count_words_in_text("你好世界") == 4

    def test_english_text(self):
        assert _count_words_in_text("hello123") == 8

    def test_mixed_text(self):
        assert _count_words_in_text("你好hello123") == 10

    def test_empty_text(self):
        assert _count_words_in_text("") == 0

    def test_text_with_punctuation(self):
        # 标点不被统计：你好=2, hello=5, 标点不计 → 7
        assert _count_words_in_text("你好！hello？") == 7


# ======================== _make_chunk_dict ========================

class TestMakeChunkDict:
    """测试标准块构造。"""

    def test_full_fields(self):
        chunk = _make_chunk_dict(
            chunk_idx=0, page_name="page", title="title",
            page_url="url", text="text", time_value="2025-01-01",
            summary="summary", question="question",
        )
        assert chunk["chunk_idx"] == 0
        assert chunk["page_name"] == "page"
        assert chunk["title"] == "title"
        assert chunk["text"] == "text"
        assert chunk["summary"] == "summary"
        assert chunk["question"] == "question"

    def test_default_empty_summary_question(self):
        chunk = _make_chunk_dict(
            chunk_idx=1, page_name="p", title="t",
            page_url="u", text="x", time_value="",
        )
        assert chunk["summary"] == ""
        assert chunk["question"] == ""

    def test_all_keys_present(self):
        chunk = _make_chunk_dict(0, "p", "t", "u", "x", "time")
        expected_keys = {"chunk_idx", "page_name", "title", "page_url", "summary", "question", "text", "html_content", "block_path", "time"}
        assert set(chunk.keys()) == expected_keys

    def test_block_path_and_html_content(self):
        """测试 block_path 和 html_content 字段（论文核心概念）。"""
        chunk = _make_chunk_dict(
            0, "page", "title", "url", "text", "time",
            block_path="html>body>div0>p",
            html_content="<p>text</p>",
        )
        assert chunk["block_path"] == "html>body>div0>p"
        assert chunk["html_content"] == "<p>text</p>"


# ======================== _split_table_into_chunks ========================

class TestSplitTableIntoChunks:
    """测试表格行切分。"""

    def _make_table(self, n_rows):
        rows = "".join(f"<tr><td>行{i}</td><td>数据{i}</td></tr>" for i in range(1, n_rows + 1))
        html = f"<table>{rows}</table>"
        return BeautifulSoup(html, "html.parser").find("table")

    def test_small_table_single_chunk(self):
        """小表格生成单个块。"""
        table = self._make_table(3)
        chunks = _split_table_into_chunks(table, "title", max_node_words=1000)
        assert len(chunks) == 1
        assert chunks[0][1] == 1  # row_range_start
        assert chunks[0][2] == 3  # row_range_end

    def test_large_table_splits(self):
        """大表格按 max_node_words 切分。"""
        table = self._make_table(50)
        chunks = _split_table_into_chunks(table, "title", max_node_words=20)
        assert len(chunks) > 1

    def test_empty_table(self):
        """空表格返回空列表。"""
        table = BeautifulSoup("<table></table>", "html.parser").find("table")
        chunks = _split_table_into_chunks(table, "title", max_node_words=100)
        assert chunks == []

    def test_chunk_text_contains_header(self):
        """每个块的文本应包含表头。"""
        html = "<table><tr><th>列A</th></tr><tr><td>1</td></tr><tr><td>2</td></tr></table>"
        table = BeautifulSoup(html, "html.parser").find("table")
        chunks = _split_table_into_chunks(table, "title", max_node_words=1000)
        assert len(chunks) == 1
        assert "列A" in chunks[0][0]

    def test_row_range_correctness(self):
        """行范围标记正确。"""
        html = (
            "<table>"
            "<tr><th>H</th></tr>"
            + "".join(f"<tr><td>R{i}</td></tr>" for i in range(1, 11))
            + "</table>"
        )
        table = BeautifulSoup(html, "html.parser").find("table")
        chunks = _split_table_into_chunks(table, "title", max_node_words=15)
        # 验证行范围连续且覆盖所有行
        all_ranges = [(c[1], c[2]) for c in chunks]
        assert all_ranges[0][0] == 1
        assert all_ranges[-1][1] == 11  # header + 10 data rows


# ======================== _generate_summary_and_question ========================

class TestGenerateSummaryAndQuestion:
    """测试统一摘要生成逻辑（不依赖 LLM 服务时）。"""

    def test_no_model_returns_truncated(self):
        """use_vllm=False（backward-compat忽略）时返回截断文本兜底。"""
        summary, question = _generate_summary_and_question(
            text="test", page_url="test.html",
            use_vllm=False, summary_model=None, summary_tokenizer=None,
        )
        # 无 vLLM 时 fallback: text[:max_new_tokens] → 截断文本
        assert isinstance(summary, str)
        assert isinstance(question, str)

    def test_vllm_fallback_on_error(self):
        """vLLM 连接失败时返回截断文本（不崩溃）。"""
        # vLLM 服务未启动时会 fallback
        summary, question = _generate_summary_and_question(
            text="这是一段足够长的测试文本" * 20, page_url="test.html",
            use_vllm=True,
        )
        # 应返回截断文本或实际摘要（取决于 vLLM 是否可达）
        assert isinstance(summary, str)
        assert isinstance(question, str)


# ======================== deduplicate_ranked_blocks_pal 大集群 ========================

class TestDeduplicateLargeCluster:
    """测试大集群去重（验证迭代式 BFS 不栈溢出）。"""

    def test_large_cluster_no_crash(self):
        """100 个相同文档不应栈溢出。"""
        # 使用跨月份的有效日期
        from datetime import datetime, timedelta
        base = datetime(2025, 1, 1)
        docs = [
            {"text": "相同内容" * 20, "page_name": "same_page", "time": (base + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")}
            for i in range(100)
        ]
        result = deduplicate_ranked_blocks_pal(docs)
        assert len(result) == 1
        # 保留时间最新的（第 100 天 = 2025-04-10）
        assert result[0]["time"] == "2025-04-10 00:00:00"

    def test_three_clusters(self):
        """3 个独立重复簇各保留 1 个。"""
        docs = [
            # 簇 A
            {"text": "内容A" * 20, "page_name": "pageA", "time": "2025-01-01 00:00:00"},
            {"text": "内容A" * 20, "page_name": "pageA", "time": "2025-01-02 00:00:00"},
            # 簇 B
            {"text": "内容B" * 20, "page_name": "pageB", "time": "2025-01-03 00:00:00"},
            {"text": "内容B" * 20, "page_name": "pageB", "time": "2025-01-04 00:00:00"},
            # 簇 C（独立）
            {"text": "完全不同C" * 20, "page_name": "pageC", "time": "2025-01-05 00:00:00"},
        ]
        result = deduplicate_ranked_blocks_pal(docs)
        assert len(result) == 3

    def test_no_time_field(self):
        """缺失 time 字段时不崩溃。"""
        docs = [
            {"text": "内容" * 20, "page_name": "p"},
            {"text": "内容" * 20, "page_name": "p"},
        ]
        result = deduplicate_ranked_blocks_pal(docs)
        assert len(result) == 1

    def test_threshold_edge_case(self):
        """threshold_content=1.0 时只有完全相同的才去重。"""
        docs = [
            {"text": "完全相同的内容" * 20, "page_name": "p", "time": "2025-01-01 00:00:00"},
            {"text": "完全相同的内容" * 20, "page_name": "p", "time": "2025-01-02 00:00:00"},
            {"text": "几乎相同的内容" * 20, "page_name": "p", "time": "2025-01-03 00:00:00"},
        ]
        result = deduplicate_ranked_blocks_pal(docs, threshold_content=1.0)
        # 只有完全相同的两个被去重
        assert len(result) == 2


# ======================== generate_block_documents 无摘要模式 ========================

class TestGenerateBlockDocumentsNoSummary:
    """测试不生成摘要时的文档块生成。"""

    def test_text_blocks_generated(self):
        """纯文本块正确生成。"""
        html = "<html><body><div>" + "这是测试内容。" * 20 + "</div></body></html>"
        cleaned = clean_html(html)
        blocks, _ = build_block_tree(cleaned, max_node_words=200, min_node_words=10, zh_char=True)

        doc_meta = generate_block_documents(
            block_tree=blocks,
            max_node_words=200,
            page_url="test.html",
            use_vllm=False,
            summary_model=None,
            summary_tokenizer=None,
        )

        assert len(doc_meta) >= 1
        for doc in doc_meta:
            assert doc["page_name"] == "test"
            assert doc["page_url"] == "test.html"
            assert "text" in doc
            assert len(doc["text"]) > 0
            # 无 LLM 时 fallback 返回截断文本，非空
            assert isinstance(doc["summary"], str)
            assert isinstance(doc["question"], str)

    def test_table_blocks_generated(self):
        """表格块正确生成（含行范围标记）。"""
        rows = "".join(f"<tr><td>行{i}</td><td>数据{i}</td></tr>" for i in range(1, 21))
        html = f"<html><body><table>{rows}</table></body></html>"
        cleaned = clean_html(html)
        blocks, _ = build_block_tree(cleaned, max_node_words=200, min_node_words=10, zh_char=True)

        doc_meta = generate_block_documents(
            block_tree=blocks,
            max_node_words=100,
            page_url="table_page.html",
            use_vllm=False,
        )

        assert len(doc_meta) >= 1
        # 至少一个块的标题包含 "表格行"
        assert any("表格行" in doc["title"] for doc in doc_meta)

    def test_empty_block_tree(self):
        """空 block_tree 返回空列表。"""
        doc_meta = generate_block_documents(
            block_tree=[],
            max_node_words=200,
            page_url="empty.html",
            use_vllm=False,
        )
        assert doc_meta == []

    def test_chunk_idx_sequential(self):
        """chunk_idx 从 0 连续递增。"""
        html = (
            "<html><body>"
            + "<div>" + "内容A" * 30 + "</div>"
            + "<div>" + "内容B" * 30 + "</div>"
            + "<div>" + "内容C" * 30 + "</div>"
            + "</body></html>"
        )
        cleaned = clean_html(html)
        blocks, _ = build_block_tree(cleaned, max_node_words=200, min_node_words=10, zh_char=True)

        doc_meta = generate_block_documents(
            block_tree=blocks,
            max_node_words=200,
            page_url="multi.html",
            use_vllm=False,
        )

        for i, doc in enumerate(doc_meta):
            assert doc["chunk_idx"] == i
