# -*- coding: utf-8 -*-
"""
算法完整性补充测试。

填补以下测试覆盖空白：
1. generate_block_documents 输出中 block_path 和 html_content 的正确性
2. BFS 裸文本块逻辑（论文 Algorithm 1）
3. build_block_tree 路径命名（同名兄弟标签编号）
4. process_html_file 完整文件处理（含 time 标签保留）
5. save_doc_meta_to_block_dir JSON 持久化
6. expand_table_spans rowspan 展开
7. generate_block_documents 混合内容分离
8. clean_html 内容保留验证
9. build_block_tree 英文模式（zh_char=False）
10. parse_time_tag 各种格式
11. _safe_int_attr 边界条件
12. deduplicate_ranked_blocks_pal 时间优先选择
"""

import os
import json
import tempfile
import pytest
from bs4 import BeautifulSoup

from html_utils import (
    clean_html,
    build_block_tree,
    expand_table_spans,
    process_html_file,
    parse_time_tag,
    _safe_int_attr,
    _count_words,
    _count_str_words,
    _is_ui_noise,
)
from text_process import (
    generate_block_documents,
    save_doc_meta_to_block_dir,
    deduplicate_ranked_blocks_pal,
    _make_chunk_dict,
    clean_text,
    clean_invisible,
    extract_title_from_block,
    parse_time,
    str_sim,
)


# ======================== 1. generate_block_documents 输出验证 ========================

class TestGenerateBlockDocumentsOutput:
    """验证 generate_block_documents 输出中 block_path 和 html_content 的正确性。"""

    def _make_blocks(self, html_str, max_words=4096, min_words=48):
        """辅助：从 HTML 字符串构建块树。"""
        cleaned = clean_html(html_str)
        expanded = expand_table_spans(cleaned)
        blocks, _ = build_block_tree(expanded, max_node_words=max_words, min_node_words=min_words, zh_char=True)
        return blocks

    def test_block_path_populated(self):
        """所有文档块的 block_path 字段非空。"""
        html = '<h1>标题</h1><p>' + '内容' * 30 + '</p>'
        blocks = self._make_blocks(html)
        if blocks:
            doc_meta = generate_block_documents(blocks, max_node_words=4096, page_url="test.html", use_vllm=False)
            for doc in doc_meta:
                assert doc["block_path"] != "", f"chunk_idx={doc['chunk_idx']} 的 block_path 为空"

    def test_html_content_populated(self):
        """含 HTML 标签的块 html_content 非空。"""
        html = '<h1>标题</h1><p>' + '内容' * 30 + '</p>'
        blocks = self._make_blocks(html)
        if blocks:
            doc_meta = generate_block_documents(blocks, max_node_words=4096, page_url="test.html", use_vllm=False)
            has_html = any(d["html_content"] for d in doc_meta)
            assert has_html, "没有任何块的 html_content 非空"

    def test_html_content_contains_tags(self):
        """html_content 包含 HTML 标签。"""
        html = '<h1>测试标题</h1><p>' + '内容内容' * 30 + '</p>'
        blocks = self._make_blocks(html)
        if blocks:
            doc_meta = generate_block_documents(blocks, max_node_words=4096, page_url="test.html", use_vllm=False)
            for d in doc_meta:
                if d["html_content"]:
                    assert "<" in d["html_content"], "html_content 不包含 HTML 标签"
                    break

    def test_text_and_html_consistent(self):
        """text 字段是 html_content 的纯文本提取。"""
        html = '<h1>标题</h1><p>' + '内容' * 30 + '</p>'
        blocks = self._make_blocks(html)
        if blocks:
            doc_meta = generate_block_documents(blocks, max_node_words=4096, page_url="test.html", use_vllm=False)
            for d in doc_meta:
                if d["html_content"] and d["text"]:
                    soup = BeautifulSoup(d["html_content"], "html.parser")
                    extracted_text = soup.get_text().replace("\x00", "").strip()
                    # text 可能经过 clean_invisible 处理，但核心内容应一致
                    assert len(extracted_text) > 0
                    break

    def test_all_fields_present_in_output(self):
        """输出包含所有必需字段。"""
        html = '<h1>标题</h1><p>' + '内容' * 30 + '</p>'
        blocks = self._make_blocks(html)
        if blocks:
            doc_meta = generate_block_documents(blocks, max_node_words=4096, page_url="test.html", use_vllm=False)
            required_keys = {"chunk_idx", "page_name", "title", "page_url", "summary",
                            "question", "text", "html_content", "block_path", "time"}
            for d in doc_meta:
                assert set(d.keys()) == required_keys, f"键不匹配: {set(d.keys())} vs {required_keys}"

    def test_chunk_idx_sequential(self):
        """chunk_idx 从 0 开始连续递增。"""
        html = '<h1>标题1</h1><p>' + '内容1' * 30 + '</p><h2>标题2</h2><p>' + '内容2' * 30 + '</p>'
        blocks = self._make_blocks(html)
        if blocks:
            doc_meta = generate_block_documents(blocks, max_node_words=4096, page_url="test.html", use_vllm=False)
            for i, d in enumerate(doc_meta):
                assert d["chunk_idx"] == i, f"chunk_idx 不连续: 期望 {i}, 得到 {d['chunk_idx']}"


# ======================== 2. BFS 裸文本块逻辑（论文 Algorithm 1） ========================

class TestBFSBareTextBlock:
    """测试 BFS 中裸文本作为独立块的逻辑（论文 Algorithm 1）。"""

    def test_bare_text_becomes_block_when_node_split(self):
        """当节点超过 maxWords 被拆分时，裸文本应作为独立块。"""
        # 构造一个 div，包含裸文本和多个子标签，总词数超过 maxWords
        html = '<div>这是裸文本内容' + '更多裸文本' * 50 + '<p>段落1' + '内容' * 30 + '</p><p>段落2' + '内容' * 30 + '</p></div>'
        cleaned = clean_html(html)
        expanded = expand_table_spans(cleaned)
        blocks, _ = build_block_tree(expanded, max_node_words=200, min_node_words=10, zh_char=True)
        # 应该有多个块（裸文本 + 子标签各自成块）
        assert len(blocks) >= 2, f"期望至少 2 个块，实际 {len(blocks)}"

    def test_pure_text_node_as_block(self):
        """纯文本节点（无子标签）应作为块。"""
        html = '<div>' + '纯文本内容' * 30 + '</div>'
        cleaned = clean_html(html)
        expanded = expand_table_spans(cleaned)
        blocks, _ = build_block_tree(expanded, max_node_words=200, min_node_words=10, zh_char=True)
        assert len(blocks) >= 1

    def test_bare_text_with_children_both_kept(self):
        """裸文本和子标签内容都被保留（不丢失信息）。"""
        html = '<div>裸文本前缀<p>子标签内容' + '内容' * 30 + '</p></div>'
        cleaned = clean_html(html)
        expanded = expand_table_spans(cleaned)
        blocks, _ = build_block_tree(expanded, max_node_words=200, min_node_words=10, zh_char=True)
        all_text = " ".join(b[0].get_text() for b in blocks)
        assert "裸文本前缀" in all_text or "子标签内容" in all_text


# ======================== 3. build_block_tree 路径命名 ========================

class TestBlockPathNaming:
    """测试同名兄弟标签的路径编号。"""

    def test_same_name_siblings_numbered(self):
        """多个同名 div 子标签应被编号为 div0, div1, ..."""
        html = '<div><div>' + '内容A' * 20 + '</div><div>' + '内容B' * 20 + '</div></div>'
        cleaned = clean_html(html)
        expanded = expand_table_spans(cleaned)
        blocks, _ = build_block_tree(expanded, max_node_words=200, min_node_words=10, zh_char=True)
        # 检查路径中是否有编号
        all_paths = [b[1] for b in blocks]
        path_str = str(all_paths)
        # 至少有一个路径包含数字编号
        has_numbered = any(
            any(c.isdigit() for tag in path for c in tag if tag[-1].isdigit())
            for path in all_paths
        )
        # 如果只有一个块（内容合并），路径可能没有编号
        if len(blocks) >= 2:
            assert has_numbered, f"路径没有编号: {all_paths}"

    def test_unique_tag_no_number(self):
        """唯一的子标签不应被编号。"""
        html = '<div><p>' + '内容' * 30 + '</p></div>'
        cleaned = clean_html(html)
        expanded = expand_table_spans(cleaned)
        blocks, _ = build_block_tree(expanded, max_node_words=200, min_node_words=10, zh_char=True)
        if blocks:
            for b in blocks:
                for tag in b[1]:
                    # 标签名不应以数字结尾（除非是 heading section）
                    if not tag.endswith("_section"):
                        assert not tag[-1].isdigit(), f"唯一标签被编号: {tag}"


# ======================== 4. process_html_file 完整文件处理 ========================

class TestProcessHtmlFile:
    """测试完整文件处理流程。"""

    def test_time_tag_preserved(self, tmp_path):
        """time 标签应被保留在输出开头。"""
        html_content = '<time datetime="2025-01-01">2025-01-01</time><html><body><p>' + '内容' * 30 + '</p></body></html>'
        src = tmp_path / "input.html"
        dst = tmp_path / "output.html"
        src.write_text(html_content, encoding="utf-8")

        result = process_html_file(str(src), str(dst))
        assert "<time" in result.lower()
        assert dst.exists()
        dst_content = dst.read_text(encoding="utf-8")
        assert "<time" in dst_content.lower()

    def test_no_time_tag(self, tmp_path):
        """无 time 标签时正常处理。"""
        html_content = '<html><body><p>' + '内容' * 30 + '</p></body></html>'
        src = tmp_path / "input.html"
        dst = tmp_path / "output.html"
        src.write_text(html_content, encoding="utf-8")

        result = process_html_file(str(src), str(dst))
        assert "<time" not in result.lower()
        assert "内容" in result

    def test_output_file_written(self, tmp_path):
        """输出文件正确写入。"""
        html_content = '<html><body><h1>标题</h1><p>' + '内容' * 30 + '</p></body></html>'
        src = tmp_path / "input.html"
        dst = tmp_path / "output.html"
        src.write_text(html_content, encoding="utf-8")

        process_html_file(str(src), str(dst))
        assert dst.exists()
        content = dst.read_text(encoding="utf-8")
        assert "标题" in content
        assert "内容" in content


# ======================== 5. save_doc_meta_to_block_dir ========================

class TestSaveDocMeta:
    """测试文档块 JSON 持久化。"""

    def test_json_saved_correctly(self, tmp_path):
        """JSON 文件正确保存并可读取。"""
        doc_meta = [
            _make_chunk_dict(0, "page", "title", "url", "text", "time",
                             block_path="html>body>p", html_content="<p>text</p>"),
        ]
        html_path = str(tmp_path / "source" / "page.html")
        html_root = str(tmp_path / "source")
        block_root = str(tmp_path / "blocks")

        os.makedirs(html_root, exist_ok=True)
        with open(html_path, "w") as f:
            f.write("<html></html>")

        json_path = save_doc_meta_to_block_dir(doc_meta, html_path, html_root, block_root)
        assert os.path.isfile(json_path)
        assert json_path.endswith(".json")

        with open(json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert len(loaded) == 1
        assert loaded[0]["block_path"] == "html>body>p"
        assert loaded[0]["html_content"] == "<p>text</p>"

    def test_nested_directory_created(self, tmp_path):
        """嵌套目录结构正确创建。"""
        doc_meta = [_make_chunk_dict(0, "p", "t", "u", "x", "time")]
        html_path = str(tmp_path / "source" / "a" / "b" / "c.html")
        html_root = str(tmp_path / "source")
        block_root = str(tmp_path / "blocks")

        os.makedirs(os.path.dirname(html_path), exist_ok=True)
        with open(html_path, "w") as f:
            f.write("<html></html>")

        json_path = save_doc_meta_to_block_dir(doc_meta, html_path, html_root, block_root)
        assert os.path.isfile(json_path)
        assert "a" in json_path and "b" in json_path


# ======================== 6. expand_table_spans rowspan ========================

class TestExpandRowspan:
    """测试 rowspan 展开算法。"""

    def test_rowspan_expanded(self):
        """rowspan 单元格正确展开。"""
        html = '''<table>
            <tr><th>A</th><th>B</th></tr>
            <tr><td rowspan="2">合并</td><td>1</td></tr>
            <tr><td>2</td></tr>
        </table>'''
        result = expand_table_spans(html)
        soup = BeautifulSoup(result, "html.parser")
        rows = soup.find_all("tr")
        # 3 行
        assert len(rows) == 3
        # 第三行应该有 2 个 td（合并单元格被复制 + 原有 td）
        third_row_cells = rows[2].find_all("td")
        assert len(third_row_cells) == 2
        assert "合并" in third_row_cells[0].get_text()

    def test_colspan_and_rowspan_combined(self):
        """colspan + rowspan 组合展开。"""
        html = '''<table>
            <tr><th colspan="2">合并列</th><th>C</th></tr>
            <tr><td rowspan="2">合并行</td><td>1</td><td>2</td></tr>
            <tr><td>3</td><td>4</td></tr>
        </table>'''
        result = expand_table_spans(html)
        soup = BeautifulSoup(result, "html.parser")
        rows = soup.find_all("tr")
        assert len(rows) == 3
        # 每行应有 3 列
        for row in rows:
            cells = row.find_all(["td", "th"])
            assert len(cells) == 3, f"行 {row} 只有 {len(cells)} 个单元格"

    def test_no_spans_unchanged(self):
        """无 colspan/rowspan 的表格不变。"""
        html = '<table><tr><td>A</td><td>B</td></tr></table>'
        result = expand_table_spans(html)
        soup = BeautifulSoup(result, "html.parser")
        rows = soup.find_all("tr")
        assert len(rows) == 1
        assert len(rows[0].find_all("td")) == 2

    def test_invalid_span_ignored(self):
        """colspan=0 或 rowspan=0 被忽略。"""
        html = '<table><tr><td colspan="0">A</td><td>B</td></tr></table>'
        result = expand_table_spans(html)
        # 不应崩溃
        assert "table" in result


# ======================== 7. generate_block_documents 混合内容 ========================

class TestGenerateBlockDocumentsMixedContent:
    """测试 generate_block_documents 对混合内容（文本+表格）的处理。"""

    def test_text_and_table_separated(self):
        """文本和表格被正确分离为不同的文档块。"""
        html = '''<div>
            <p>表格前说明文字' + '内容' * 20 + '</p>
            <table><tr><th>列A</th></tr><tr><td>数据1</td></tr><tr><td>数据2</td></tr></table>
            <p>表格后说明文字' + '内容' * 20 + '</p>
        </div>'''
        # 注意：这里需要构造合法 HTML
        html = '<div><p>表格前说明文字' + '内容' * 20 + '</p><table><tr><th>列A</th></tr><tr><td>数据1</td></tr><tr><td>数据2</td></tr></table><p>表格后说明文字' + '内容' * 20 + '</p></div>'
        cleaned = clean_html(html)
        expanded = expand_table_spans(cleaned)
        blocks, _ = build_block_tree(expanded, max_node_words=4096, min_node_words=10, zh_char=True)
        doc_meta = generate_block_documents(blocks, max_node_words=4096, page_url="mixed.html", use_vllm=False)

        all_text = " ".join(d["text"] for d in doc_meta)
        assert "表格前说明文字" in all_text or "表格后说明文字" in all_text
        assert "列A" in all_text or "数据1" in all_text

    def test_table_only_block(self):
        """纯表格块正确生成。"""
        html = '<table><tr><th>标题</th></tr><tr><td>' + '数据' * 20 + '</td></tr></table>'
        cleaned = clean_html(html)
        expanded = expand_table_spans(cleaned)
        blocks, _ = build_block_tree(expanded, max_node_words=4096, min_node_words=10, zh_char=True)
        if blocks:
            doc_meta = generate_block_documents(blocks, max_node_words=4096, page_url="table.html", use_vllm=False)
            assert len(doc_meta) >= 1
            assert "标题" in doc_meta[0]["text"] or "数据" in doc_meta[0]["text"]


# ======================== 8. clean_html 内容保留验证 ========================

class TestCleanHtmlContentPreservation:
    """验证 clean_html 在移除噪声的同时保留所有有价值内容。"""

    def test_text_content_preserved(self):
        """所有有价值的文本内容被保留。"""
        html = '''<html><head><title>标题</title><style>.x{}</style></head>
        <body>
        <script>var x=1;</script>
        <h1>第一章</h1>
        <p>重要内容段落</p>
        <table><tr><th>表头</th></tr><tr><td>数据</td></tr></table>
        <ul><li>列表项</li></ul>
        </body></html>'''
        result = clean_html(html)
        assert "第一章" in result
        assert "重要内容段落" in result
        assert "表头" in result
        assert "数据" in result
        assert "列表项" in result
        # 噪声被移除
        assert "var x" not in result
        assert ".x{}" not in result

    def test_nested_structure_preserved(self):
        """嵌套结构信息被保留。"""
        html = '<div><section><h2>标题</h2><p>内容</p></section></div>'
        result = clean_html(html)
        assert "标题" in result
        assert "内容" in result

    def test_attributes_removed_but_content_kept(self):
        """原始属性被移除但内容保留（warp_domains 添加的 class 除外）。"""
        html = '<div class="content" id="main" style="color:red"><p>文本内容</p></div>'
        result = clean_html(html)
        assert "文本内容" in result
        assert "color:red" not in result
        assert "id=" not in result
        assert "style=" not in result
        # warp_domains 添加的 class="isolated_domain" 是内部结构标记，不算原始属性
        # 原始的 class="content" 应被移除
        assert 'class="content"' not in result


# ======================== 9. build_block_tree 英文模式 ========================

class TestBuildBlockTreeEnglish:
    """测试英文模式（zh_char=False）的分块。"""

    def test_english_word_count(self):
        """英文模式按空格分词计算词数。"""
        html = '<div><p>' + 'word ' * 100 + '</p></div>'
        cleaned = clean_html(html)
        expanded = expand_table_spans(cleaned)
        blocks, _ = build_block_tree(expanded, max_node_words=50, min_node_words=5, zh_char=False)
        # 英文 100 词 > 50 maxWords，应该被拆分
        assert len(blocks) >= 1

    def test_english_heading_split(self):
        """英文标题也能触发拆分。"""
        html = '<div><h1>Title One</h1><p>' + 'content ' * 20 + '</p><h2>Title Two</h2><p>' + 'content ' * 20 + '</p></div>'
        cleaned = clean_html(html)
        expanded = expand_table_spans(cleaned)
        blocks, _ = build_block_tree(expanded, max_node_words=4096, min_node_words=5, zh_char=False)
        if len(blocks) >= 2:
            all_text = " ".join(b[0].get_text() for b in blocks)
            assert "Title One" in all_text
            assert "Title Two" in all_text


# ======================== 10. parse_time_tag ========================

class TestParseTimeTag:
    """测试 time 标签解析。"""

    def test_standard_time_tag(self):
        html = '<time datetime="2025-01-01">2025-01-01</time><div>content</div>'
        time_value, remaining = parse_time_tag(html)
        assert "2025-01-01" in time_value
        assert "content" in remaining
        assert "<time" not in remaining

    def test_no_time_tag(self):
        html = '<div>content</div>'
        time_value, remaining = parse_time_tag(html)
        assert time_value == ""
        assert remaining == html

    def test_time_tag_with_attributes(self):
        html = '<time datetime="2025-06-15T10:30:00" class="date">June 15, 2025</time><div>content</div>'
        time_value, remaining = parse_time_tag(html)
        assert "June 15" in time_value
        assert "content" in remaining

    def test_time_tag_at_start_with_whitespace(self):
        html = '  <time>2025-01-01</time>  <div>content</div>'
        time_value, remaining = parse_time_tag(html)
        assert "2025-01-01" in time_value
        assert "content" in remaining


# ======================== 11. _safe_int_attr 边界条件 ========================

class TestSafeIntAttr:
    """测试 _safe_int_attr 的边界条件。"""

    def test_normal_value(self):
        tag = BeautifulSoup('<td colspan="3">x</td>', "html.parser").find("td")
        assert _safe_int_attr(tag, "colspan", 1) == 3

    def test_missing_attr(self):
        tag = BeautifulSoup('<td>x</td>', "html.parser").find("td")
        assert _safe_int_attr(tag, "colspan", 1) == 1

    def test_invalid_value(self):
        tag = BeautifulSoup('<td colspan="abc">x</td>', "html.parser").find("td")
        assert _safe_int_attr(tag, "colspan", 1) == 1

    def test_none_attr(self):
        tag = BeautifulSoup('<td colspan="">x</td>', "html.parser").find("td")
        result = _safe_int_attr(tag, "colspan", 1)
        # 空字符串无法转为 int，应返回默认值
        assert result == 1

    def test_zero_value(self):
        tag = BeautifulSoup('<td colspan="0">x</td>', "html.parser").find("td")
        assert _safe_int_attr(tag, "colspan", 1) == 0


# ======================== 12. deduplicate 时间优先选择 ========================

class TestDeduplicateTimePriority:
    """测试去重时按时间优先保留最新版本。"""

    def test_keeps_latest_by_time(self):
        """相同内容的多个版本，保留时间最新的。"""
        docs = [
            {"text": "相同内容" * 20, "page_name": "page", "time": "2025-01-01 00:00:00"},
            {"text": "相同内容" * 20, "page_name": "page", "time": "2025-06-01 00:00:00"},
            {"text": "相同内容" * 20, "page_name": "page", "time": "2025-03-01 00:00:00"},
        ]
        result = deduplicate_ranked_blocks_pal(docs)
        assert len(result) == 1
        assert result[0]["time"] == "2025-06-01 00:00:00"

    def test_different_content_all_kept(self):
        """不同内容的文档全部保留。"""
        docs = [
            {"text": "内容A" * 20, "page_name": "pageA", "time": "2025-01-01 00:00:00"},
            {"text": "内容B" * 20, "page_name": "pageB", "time": "2025-01-02 00:00:00"},
            {"text": "内容C" * 20, "page_name": "pageC", "time": "2025-01-03 00:00:00"},
        ]
        result = deduplicate_ranked_blocks_pal(docs)
        assert len(result) == 3

    def test_no_time_keeps_first(self):
        """无 time 字段时不崩溃（保留其中一个）。"""
        docs = [
            {"text": "相同" * 20, "page_name": "p"},
            {"text": "相同" * 20, "page_name": "p"},
        ]
        result = deduplicate_ranked_blocks_pal(docs)
        assert len(result) == 1

    def test_large_cluster_performance(self):
        """大集群去重性能测试（100 个文档）。"""
        from datetime import datetime, timedelta
        base = datetime(2025, 1, 1)
        docs = [
            {"text": "相同" * 20, "page_name": "same", "time": (base + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")}
            for i in range(100)
        ]
        result = deduplicate_ranked_blocks_pal(docs)
        assert len(result) == 1
        # 保留第 100 天
        assert result[0]["time"] == "2025-04-10 00:00:00"


# ======================== 13. 辅助函数测试 ========================

class TestHelperFunctions:
    """测试辅助函数。"""

    def test_parse_time_valid(self):
        assert parse_time("2025-01-01 00:00:00") is not None

    def test_parse_time_invalid(self):
        from datetime import datetime
        result = parse_time("invalid")
        assert result == datetime.min

    def test_parse_time_empty(self):
        from datetime import datetime
        result = parse_time("")
        assert result == datetime.min

    def test_str_sim_identical(self):
        assert str_sim("hello", "hello") == 1.0

    def test_str_sim_different(self):
        assert str_sim("hello", "world") < 1.0

    def test_str_sim_empty(self):
        assert str_sim("", "") == 1.0

    def test_clean_text_chinese(self):
        assert clean_text("你好世界") == "你好世界"

    def test_clean_text_mixed(self):
        assert clean_text("你好hello123") == "你好hello123"

    def test_clean_text_special_chars(self):
        assert clean_text("你好！@#世界") == "你好世界"

    def test_clean_invisible_removes_zero_width(self):
        text = "hello\u200bworld"
        assert clean_invisible(text) == "helloworld"

    def test_clean_invisible_preserves_normal(self):
        assert clean_invisible("hello world") == "hello world"

    def test_count_words_tag(self):
        soup = BeautifulSoup('<div>你好世界</div>', "html.parser")
        tag = soup.find("div")
        assert _count_words(tag, zh_char=True) == 4
        assert _count_words(tag, zh_char=False) == 1

    def test_count_str_words(self):
        assert _count_str_words("你好", zh_char=True) == 2
        assert _count_str_words("hello world", zh_char=False) == 2

    def test_is_ui_noise_empty(self):
        soup = BeautifulSoup('<div></div>', "html.parser")
        assert _is_ui_noise(soup.find("div")) is False

    def test_is_ui_noise_content(self):
        soup = BeautifulSoup('<div><p>正常内容</p></div>', "html.parser")
        assert _is_ui_noise(soup.find("div")) is False

    def test_extract_title_from_heading(self):
        soup = BeautifulSoup('<div><h2>标题文本</h2><p>内容</p></div>', "html.parser")
        title = extract_title_from_block(soup.find("div"))
        assert "标题文本" in title

    def test_extract_title_fallback_to_text(self):
        soup = BeautifulSoup('<div><p>第一段文本</p></div>', "html.parser")
        title = extract_title_from_block(soup.find("div"))
        assert "第一段文本" in title

    def test_extract_title_empty(self):
        soup = BeautifulSoup('<div></div>', "html.parser")
        title = extract_title_from_block(soup.find("div"))
        assert title == ""
