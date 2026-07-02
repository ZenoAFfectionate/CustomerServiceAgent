# -*- coding: utf-8 -*-
"""
html_utils 模块单元测试。

覆盖核心函数：clean_html、clean_xml、clean_html_text、expand_table_spans、
build_block_tree、parse_time_tag、process_html_file。

运行方式：
    PYTHONPATH=src pytest tests/test_html_utils.py -v
"""

import os
import tempfile
import pytest
from bs4 import BeautifulSoup

from html_utils import (
    clean_html,
    clean_xml,
    clean_html_text,
    expand_table_spans,
    build_block_tree,
    parse_time_tag,
    process_html_file,
)


# ======================== clean_xml ========================

class TestCleanXml:
    """测试 XML/Doctype 声明移除"""

    def test_removes_xml_declaration(self):
        html = '<?xml version="1.0"?><html><body>hello</body></html>'
        result = clean_xml(html)
        assert '<?xml' not in result
        assert '<html>' in result

    def test_removes_doctype(self):
        html = '<!DOCTYPE html><html><body>hello</body></html>'
        result = clean_xml(html)
        assert '<!DOCTYPE' not in result.lower()
        assert '<html>' in result

    def test_preserves_content_without_declaration(self):
        html = '<html><body>hello</body></html>'
        result = clean_xml(html)
        assert 'hello' in result


# ======================== clean_html_text ========================

class TestCleanHtmlText:
    """测试 markdown 块和换行符清理"""

    def test_removes_markdown_html_block(self):
        text = '```html\n<div>test</div>\n```'
        result = clean_html_text(text)
        assert '```' not in result
        assert '<div>test</div>' in result

    def test_removes_newlines_between_tags(self):
        text = '<div>\n<span>hello</span>\n</div>'
        result = clean_html_text(text)
        assert '>\n' not in result or result.count('\n') < text.count('\n')


# ======================== expand_table_spans ========================

class TestExpandTableSpans:
    """测试合并单元格展开"""

    def test_expands_colspan(self):
        html = '<table><tr><td colspan="2">A</td></tr><tr><td>B</td><td>C</td></tr></table>'
        result = expand_table_spans(html)
        soup = BeautifulSoup(result, 'html.parser')
        rows = soup.find_all('tr')
        # 第一行应该有 2 个 td（展开后）
        assert len(rows[0].find_all('td')) == 2

    def test_expands_rowspan(self):
        html = '<table><tr><td rowspan="2">A</td><td>B</td></tr><tr><td>C</td></tr></table>'
        result = expand_table_spans(html)
        soup = BeautifulSoup(result, 'html.parser')
        rows = soup.find_all('tr')
        # 第二行应该有 2 个 td（展开后）
        assert len(rows[1].find_all('td')) == 2

    def test_handles_no_spans(self):
        html = '<table><tr><td>A</td><td>B</td></tr></table>'
        result = expand_table_spans(html)
        assert '<table>' in result
        assert 'A' in result and 'B' in result

    def test_ignores_zero_colspan(self):
        """colspan=0 应被跳过"""
        html = '<table><tr><td colspan="0">A</td></tr></table>'
        result = expand_table_spans(html)
        # 不应崩溃
        assert '<table>' in result


# ======================== parse_time_tag ========================

class TestParseTimeTag:
    """测试 <time> 标签提取"""

    def test_extracts_time_value(self):
        html = '<time datetime="2025-01-01">2025-01-01 12:00:00</time><div>content</div>'
        time_value, remaining = parse_time_tag(html)
        assert time_value == '2025-01-01 12:00:00'
        assert '<div>content</div>' in remaining

    def test_returns_empty_when_no_time_tag(self):
        html = '<div>no time tag</div>'
        time_value, remaining = parse_time_tag(html)
        assert time_value == ''
        assert remaining == html


# ======================== build_block_tree ========================

class TestBuildBlockTree:
    """测试 HTML 结构化分块"""

    def test_short_text_returns_empty(self):
        """文本长度小于 min_node_words 时返回空列表"""
        html = '<p>短</p>'
        blocks, raw = build_block_tree(html, max_node_words=512, min_node_words=32, zh_char=True)
        assert blocks == []

    def test_long_text_returns_blocks(self):
        """长文本应返回至少一个块"""
        html = '<html><body><div>' + '这是一段测试文本。' * 50 + '</div></body></html>'
        blocks, raw = build_block_tree(html, max_node_words=512, min_node_words=10, zh_char=True)
        assert len(blocks) >= 1

    def test_table_block_preserved(self):
        """包含大表格的 HTML 应保留表格块"""
        rows = ''.join(f'<tr><td>cell_{i}</td></tr>' for i in range(30))
        html = f'<table>{rows}</table>'
        html = '<div>' + html + '</div>'
        blocks, raw = build_block_tree(html, max_node_words=100, min_node_words=10, zh_char=True)
        # 应至少返回一个块
        assert len(blocks) >= 1


# ======================== process_html_file ========================

class TestProcessHtmlFile:
    """测试单文件处理入口"""

    def test_processes_simple_html_file(self):
        """处理简单 HTML 文件，应输出清洗后的内容"""
        content = '<time>2025-01-01</time><html><body><h1>Title</h1><p>hello world</p></body></html>'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as src:
            src.write(content)
            src_path = src.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as tgt:
            tgt_path = tgt.name

        try:
            result = process_html_file(src_path, tgt_path)
            assert 'hello world' in result
            assert os.path.exists(tgt_path)

            with open(tgt_path, 'r', encoding='utf-8') as f:
                saved = f.read()
            assert 'hello world' in saved
        finally:
            os.unlink(src_path)
            if os.path.exists(tgt_path):
                os.unlink(tgt_path)
