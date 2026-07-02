# -*- coding: utf-8 -*-
"""
text_process 模块单元测试。

覆盖核心函数：clean_text、clean_invisible、extract_title_from_block、
build_optimal_jieba_query、parse_time、str_sim、deduplicate_ranked_blocks_pal、
save_doc_meta_to_block_dir。

运行方式：
    PYTHONPATH=src pytest tests/test_text_process.py -v
"""

import os
import json
import tempfile
from datetime import datetime

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
)


# ======================== clean_text ========================

class TestCleanText:
    """测试文本清洗函数"""

    def test_keeps_chinese(self):
        assert clean_text("你好世界") == "你好世界"

    def test_keeps_english_and_numbers(self):
        assert clean_text("hello123") == "hello123"

    def test_removes_special_chars(self):
        assert clean_text("hello!@#world") == "helloworld"

    def test_removes_spaces(self):
        assert clean_text("hello world") == "helloworld"

    def test_empty_input(self):
        assert clean_text("") == ""

    def test_mixed_content(self):
        assert clean_text("你好！hello@123") == "你好hello123"


# ======================== clean_invisible ========================

class TestCleanInvisible:
    """测试不可见字符清理"""

    def test_removes_zero_width_chars(self):
        text = "hello\u200bworld"
        result = clean_invisible(text)
        assert result == "helloworld"

    def test_removes_control_chars(self):
        text = "test\u2060data"
        result = clean_invisible(text)
        assert result == "testdata"

    def test_preserves_normal_text(self):
        text = "正常文本"
        assert clean_invisible(text) == text


# ======================== extract_title_from_block ========================

class TestExtractTitleFromBlock:
    """测试标题提取"""

    def test_extracts_h1_title(self):
        html = '<div><h1>主标题</h1><p>内容</p></div>'
        tag = BeautifulSoup(html, 'html.parser').find('div')
        title = extract_title_from_block(tag)
        assert '主标题' in title

    def test_extracts_h2_title(self):
        html = '<div><h2>副标题</h2><p>内容</p></div>'
        tag = BeautifulSoup(html, 'html.parser').find('div')
        title = extract_title_from_block(tag)
        assert '副标题' in title

    def test_fallback_to_first_text(self):
        html = '<div><p>第一段文本</p><p>第二段</p></div>'
        tag = BeautifulSoup(html, 'html.parser').find('div')
        title = extract_title_from_block(tag)
        assert '第一段文本' in title

    def test_truncates_long_title(self):
        long_title = 'A' * 100
        html = f'<div><h1>{long_title}</h1></div>'
        tag = BeautifulSoup(html, 'html.parser').find('div')
        title = extract_title_from_block(tag)
        assert len(title) <= 48


# ======================== build_optimal_jieba_query ========================

class TestBuildOptimalJiebaQuery:
    """测试 ES 查询构建"""

    def test_returns_dict_with_query(self):
        keywords = ["千川", "运营"]
        fields = {"text": {"boost": 1}, "title": {"boost": 2}}
        result = build_optimal_jieba_query(keywords, fields)
        assert "query" in result
        assert "bool" in result["query"]
        assert "should" in result["query"]["bool"]

    def test_includes_highlight(self):
        keywords = ["千川"]
        fields = {"text": {"boost": 1}}
        result = build_optimal_jieba_query(keywords, fields)
        assert "highlight" in result

    def test_synonym_map(self):
        keywords = ["千川"]
        fields = {"text": {"boost": 1}}
        synonym_map = {"千川": ["千川", "巨量千川"]}
        result = build_optimal_jieba_query(keywords, fields, synonym_map=synonym_map)
        assert len(result["query"]["bool"]["should"]) > 0

    def test_empty_keywords(self):
        keywords = []
        fields = {"text": {"boost": 1}}
        result = build_optimal_jieba_query(keywords, fields)
        assert result["query"]["bool"]["should"] == []


# ======================== parse_time ========================

class TestParseTime:
    """测试时间字符串解析"""

    def test_valid_datetime(self):
        result = parse_time("2025-06-30 12:00:00")
        assert result == datetime(2025, 6, 30, 12, 0, 0)

    def test_invalid_format_returns_min(self):
        result = parse_time("invalid")
        assert result == datetime.min

    def test_empty_string_returns_min(self):
        result = parse_time("")
        assert result == datetime.min


# ======================== str_sim ========================

class TestStrSim:
    """测试字符串相似度"""

    def test_identical_strings(self):
        assert str_sim("hello", "hello") == 1.0

    def test_completely_different(self):
        assert str_sim("abc", "xyz") == 0.0

    def test_partial_match(self):
        score = str_sim("hello", "hallo")
        assert 0 < score < 1

    def test_empty_strings(self):
        assert str_sim("", "") == 1.0


# ======================== deduplicate_ranked_blocks_pal ========================

class TestDeduplicateRankedBlocks:
    """测试文档块去重"""

    def test_single_doc_unchanged(self):
        docs = [{"text": "内容A", "page_name": "page1", "time": "2025-01-01 00:00:00"}]
        result = deduplicate_ranked_blocks_pal(docs)
        assert len(result) == 1

    def test_empty_list(self):
        result = deduplicate_ranked_blocks_pal([])
        assert result == []

    def test_identical_docs_kept_one(self):
        docs = [
            {"text": "相同内容", "page_name": "page1", "time": "2025-01-01 00:00:00"},
            {"text": "相同内容", "page_name": "page1", "time": "2025-01-02 00:00:00"},
        ]
        result = deduplicate_ranked_blocks_pal(docs)
        assert len(result) == 1
        # 应保留时间更新的
        assert result[0]["time"] == "2025-01-02 00:00:00"

    def test_different_docs_all_kept(self):
        docs = [
            {"text": "内容A", "page_name": "page1", "time": "2025-01-01 00:00:00"},
            {"text": "完全不同的内容B", "page_name": "page2", "time": "2025-01-02 00:00:00"},
        ]
        result = deduplicate_ranked_blocks_pal(docs)
        assert len(result) == 2


# ======================== save_doc_meta_to_block_dir ========================

class TestSaveDocMeta:
    """测试 JSON 文件保存"""

    def test_saves_json_file(self):
        doc_meta = [{"chunk_idx": 0, "title": "test", "text": "hello"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            html_dir = os.path.join(tmpdir, "html")
            os.makedirs(html_dir, exist_ok=True)
            html_path = os.path.join(html_dir, "test.html")
            with open(html_path, 'w') as f:
                f.write("<html></html>")

            block_dir = os.path.join(tmpdir, "blocks")
            save_doc_meta_to_block_dir(doc_meta, html_path, html_dir, block_dir)

            expected_json = os.path.join(block_dir, "test.json")
            assert os.path.exists(expected_json)

            with open(expected_json, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            assert saved == doc_meta
