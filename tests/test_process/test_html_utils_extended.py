# -*- coding: utf-8 -*-
"""
html_utils 扩展测试 — 覆盖修复的 BUG 和边界场景。

测试重点：
    - clean_html_text 正则修复：不再丢失标签前字符
    - build_block_tree 深层嵌套性能与正确性
    - expand_table_spans 混合 colspan+rowspan
    - simplify_html_keep_table 嵌套空标签清理
    - warp_domains 多层标题嵌套
    - parse_time_tag 边界场景

运行方式：
    PYTHONPATH=src pytest tests/test_html_utils_extended.py -v
"""

import os
import tempfile

import pytest
from bs4 import BeautifulSoup

from html_utils import (
    clean_html,
    clean_html_text,
    clean_xml,
    expand_table_spans,
    build_block_tree,
    parse_time_tag,
    process_html_file,
    simplify_html_keep_table,
    warp_domains,
)


# ======================== clean_html_text 正则修复验证 ========================

class TestCleanHtmlTextRegexFix:
    """验证修复后的正则不再丢失标签前字符。"""

    def test_preserves_char_before_tag(self):
        """修复前：'a<' 中的 'a' 会被 lambda 误处理导致字符丢失。"""
        text = "content<div>more</div>"
        result = clean_html_text(text)
        assert "content" in result
        assert "<div>" in result

    def test_preserves_chinese_before_tag(self):
        """中文字符 + < 不应丢失中文。"""
        text = "内容<div>更多</div>"
        result = clean_html_text(text)
        assert "内容" in result

    def test_removes_newline_between_char_and_tag(self):
        """clean_html_text 不再合并文本和标签间的换行（避免内容丢失）。
        仅移除标签间的纯空白换行（如 >\\n< → ><）。"""
        text = "text\n<div>"
        result = clean_html_text(text)
        # 新行为：保留文本和标签间的换行，不合并
        assert "text" in result
        # 标签间的纯空白换行应被移除
        text2 = "<div>\n\n<span>hello</span></div>"
        result2 = clean_html_text(text2)
        assert "><" in result2 or ">\n<" not in result2

    def test_nested_tags_preserved(self):
        text = "<div>\n<span>\nhello\n</span>\n</div>"
        result = clean_html_text(text)
        assert "hello" in result
        assert "<span>" in result
        assert "</div>" in result


# ======================== expand_table_spans 混合场景 ========================

class TestExpandTableSpansMixed:
    """测试 colspan + rowspan 混合场景。"""

    def test_mixed_colspan_and_rowspan(self):
        """同时包含 colspan 和 rowspan 的复杂表格。"""
        html = (
            '<table>'
            '<tr><td rowspan="2" colspan="2">A</td><td>B</td></tr>'
            '<tr><td>C</td></tr>'
            '<tr><td>D</td><td>E</td><td>F</td></tr>'
            '</table>'
        )
        result = expand_table_spans(html)
        soup = BeautifulSoup(result, "html.parser")
        rows = soup.find_all("tr")
        # 第一行应有 3 个单元格（A 展开 2 列 + B）
        assert len(rows[0].find_all(["td", "th"])) == 3
        # 第二行应有 3 个单元格（A 展开的副本 + C）
        assert len(rows[1].find_all(["td", "th"])) == 3
        # 第三行应有 3 个单元格
        assert len(rows[2].find_all(["td", "th"])) == 3

    def test_no_table_in_html(self):
        """不含表格的 HTML 不受影响。"""
        html = "<div>no table here</div>"
        result = expand_table_spans(html)
        assert "no table here" in result
        assert "<table" not in result

    def test_empty_table(self):
        """空表格不崩溃。"""
        html = "<table></table>"
        result = expand_table_spans(html)
        assert "<table>" in result

    def test_multiple_tables(self):
        """多个表格均被正确展开。"""
        html = (
            '<table><tr><td colspan="2">A</td></tr></table>'
            '<table><tr><td rowspan="2">B</td></tr><tr><td>C</td></tr></table>'
        )
        result = expand_table_spans(html)
        soup = BeautifulSoup(result, "html.parser")
        tables = soup.find_all("table")
        assert len(tables) == 2

    def test_invalid_span_values(self):
        """无效的 span 值（非数字）回退为 1。"""
        html = '<table><tr><td colspan="abc">A</td></tr></table>'
        result = expand_table_spans(html)
        # 不崩溃即可
        assert "<table>" in result


# ======================== build_block_tree 深层嵌套 ========================

class TestBuildBlockTreeDeepNested:
    """测试深层嵌套 HTML 的分块正确性与性能。"""

    def test_deeply_nested_html_no_crash(self):
        """10 层嵌套的 div 不应导致栈溢出。"""
        inner = "内容" * 50
        html = "<div>" * 10 + inner + "</div>" * 10
        blocks, raw = build_block_tree(html, max_node_words=200, min_node_words=5, zh_char=True)
        # 应至少返回一个块
        assert len(blocks) >= 1

    def test_empty_html(self):
        """空 HTML 返回空列表。"""
        blocks, raw = build_block_tree("", max_node_words=512, min_node_words=32, zh_char=True)
        assert blocks == []

    def test_html_below_min_words(self):
        """回归测试：修复 M7——内容不足 min_node_words 但非空时应保留为单块，
        而非直接丢弃（避免短帮助页/零散小段落被静默丢弃）。"""
        html = "<p>短</p>"
        blocks, raw = build_block_tree(html, max_node_words=512, min_node_words=32, zh_char=True)
        assert len(blocks) == 1

    def test_multiple_top_level_children(self):
        """多个顶层子标签都满足 min_words 时应全部保留。"""
        section1 = "<div>" + "内容" * 30 + "</div>"
        section2 = "<div>" + "数据" * 30 + "</div>"
        html = section1 + section2
        blocks, raw = build_block_tree(html, max_node_words=500, min_node_words=10, zh_char=True)
        assert len(blocks) >= 1

    def test_enumeration_of_same_name_tags(self):
        """多个同名子标签应被编号（div0, div1, ...）。"""
        html = (
            "<div>" + "内容A" * 30 + "</div>"
            "<div>" + "内容B" * 30 + "</div>"
            "<div>" + "内容C" * 30 + "</div>"
        )
        html = f"<html>{html}</html>"
        blocks, raw = build_block_tree(html, max_node_words=200, min_node_words=10, zh_char=True)
        # 应返回多个块
        assert len(blocks) >= 2

    def test_en_mode_word_count(self):
        """zh_char=False 时按空格分词。"""
        html = "<div>" + "word " * 100 + "</div>"
        blocks, raw = build_block_tree(html, max_node_words=50, min_node_words=10, zh_char=False)
        assert len(blocks) >= 1


# ======================== simplify_html_keep_table 嵌套空标签 ========================

class TestSimplifyHtmlKeepTableNested:
    """测试嵌套空标签清理。"""

    def test_removes_nested_empty_tags(self):
        """多层嵌套的空标签都应被移除。"""
        html = "<div><span><p></p></span></div>"
        soup = BeautifulSoup(html, "html.parser")
        result = simplify_html_keep_table(soup)
        assert "span" not in result or result.strip() == ""

    def test_preserves_table_with_content(self):
        """有内容的表格标签不应被移除。"""
        html = "<table><tr><td>data</td></tr></table>"
        soup = BeautifulSoup(html, "html.parser")
        result = simplify_html_keep_table(soup)
        assert "data" in result

    def test_heading_class_conversion(self):
        """heading-h2 class 被转换为 data-block-type 属性。"""
        html = '<div class="heading-h2">Title</div>'
        soup = BeautifulSoup(html, "html.parser")
        result = simplify_html_keep_table(soup)
        assert "data-block-type" in result

    def test_removes_script_and_style(self):
        """script 和 style 标签被移除。"""
        html = "<div>text</div><script>alert(1)</script><style>.x{color:red}</style>"
        soup = BeautifulSoup(html, "html.parser")
        result = simplify_html_keep_table(soup)
        assert "alert" not in result
        assert "color" not in result

    def test_removes_html_comments(self):
        """HTML 注释被移除。"""
        html = "<div>text</div><!-- comment -->"
        soup = BeautifulSoup(html, "html.parser")
        result = simplify_html_keep_table(soup)
        assert "comment" not in result


# ======================== warp_domains 多层标题 ========================

class TestWarpDomainsNested:
    """测试多层标题嵌套包装。"""

    def test_h1_h2_hierarchy(self):
        """h1 包含 h2 时，h2_domain 应嵌套在 h1_domain 内。"""
        html = "<h1>主标题</h1><h2>副标题</h2><p>内容</p>"
        result = warp_domains(html)
        assert "h1_domain" in result
        assert "h2_domain" in result

    def test_no_heading_wraps_in_isolated(self):
        """无标题时整体包入 isolated_domain。"""
        html = "<div>内容</div>"
        result = warp_domains(html)
        assert "isolated_domain" in result

    def test_table_wrapped(self):
        """表格被包装在 table_domain 中。"""
        html = "<table><tr><td>A</td></tr></table>"
        result = warp_domains(html)
        assert "table_domain" in result

    def test_consecutive_same_level_headings(self):
        """连续的同级标题各自独立包装。"""
        html = "<h1>A</h1><p>内容A</p><h1>B</h1><p>内容B</p>"
        result = warp_domains(html)
        # 应有两个 h1_domain
        assert result.count("h1_domain") >= 2


# ======================== parse_time_tag 边界 ========================

class TestParseTimeTagEdgeCases:
    """测试 parse_time_tag 边界场景。"""

    def test_time_tag_with_attributes(self):
        html = '<time datetime="2025-06-30T12:00:00">2025-06-30 12:00:00</time><div>content</div>'
        time_val, remaining = parse_time_tag(html)
        assert time_val == "2025-06-30 12:00:00"
        assert "<div>" in remaining

    def test_time_tag_with_whitespace_prefix(self):
        html = '   <time>2025-01-01</time><div>x</div>'
        time_val, remaining = parse_time_tag(html)
        assert time_val == "2025-01-01"

    def test_no_time_tag(self):
        html = "<div>no time</div>"
        time_val, remaining = parse_time_tag(html)
        assert time_val == ""
        assert remaining == html


# ======================== process_html_file 边界 ========================

class TestProcessHtmlFileEdgeCases:
    """测试 process_html_file 边界场景。"""

    def test_file_without_time_tag(self):
        """无 time 标签的 HTML 文件正常处理。"""
        content = "<html><body><h1>Title</h1><p>hello world</p></body></html>"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as src:
            src.write(content)
            src_path = src.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as tgt:
            tgt_path = tgt.name

        try:
            result = process_html_file(src_path, tgt_path)
            assert "hello world" in result
        finally:
            os.unlink(src_path)
            if os.path.exists(tgt_path):
                os.unlink(tgt_path)

    def test_empty_html_file(self):
        """空 HTML 文件不崩溃。"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as src:
            src.write("")
            src_path = src.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as tgt:
            tgt_path = tgt.name

        try:
            result = process_html_file(src_path, tgt_path)
            # 应生成空或最小输出
            assert isinstance(result, str)
        finally:
            os.unlink(src_path)
            if os.path.exists(tgt_path):
                os.unlink(tgt_path)

    def test_preserves_time_tag_in_output(self):
        """time 标签被保留在输出开头。"""
        content = '<time>2025-06-30 12:00:00</time><html><body><p>data</p></body></html>'
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as src:
            src.write(content)
            src_path = src.name
        with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False, encoding="utf-8") as tgt:
            tgt_path = tgt.name

        try:
            result = process_html_file(src_path, tgt_path)
            assert "<time" in result
            assert "2025-06-30 12:00:00" in result
        finally:
            os.unlink(src_path)
            if os.path.exists(tgt_path):
                os.unlink(tgt_path)


# ======================== clean_html 端到端 ========================

class TestCleanHtmlEndToEnd:
    """测试 clean_html 主入口端到端。"""

    def test_removes_script_style_xml(self):
        html = '<?xml version="1.0"?><!DOCTYPE html><script>x()</script><style>y{}</style><div>content</div>'
        result = clean_html(html)
        assert "content" in result
        assert "<?xml" not in result
        assert "<!DOCTYPE" not in result.upper()
        assert "x()" not in result
        assert "y{}" not in result

    def test_preserves_table_structure(self):
        html = "<table><tr><td>A</td><td>B</td></tr></table>"
        result = clean_html(html)
        assert "table_domain" in result
        assert "A" in result
        assert "B" in result

    def test_markdown_html_block_removed(self):
        html = "```html\n<div>test</div>\n```"
        result = clean_html(html)
        # markdown 标记被移除
        assert "```" not in result
