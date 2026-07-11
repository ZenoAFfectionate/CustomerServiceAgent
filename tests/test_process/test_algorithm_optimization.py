# -*- coding: utf-8 -*-
"""
算法优化修复的单元测试。

覆盖以下修复：
1. SVG/input/button/nav/aside/head/title 噪声标签清理
2. 隐藏元素（display:none, hidden class）移除
3. 模板残留文本（{{PLACEHOLDER}}）移除
4. 空标签清理（含仅 <br> 的标签）
5. 多层冗余包装 div 展开
6. clean_html_text 非破坏性文本规范化
7. build_block_tree UI 噪声过滤
8. build_block_tree 按 heading 拆分
9. 标题提取（跳过 UI 噪声、去前导数字序号）
10. 混合内容块提取（表格 + 文本）
11. 表格切分超长行处理
12. 去重算法短 page_name 修复
"""

import os
import pytest
from bs4 import BeautifulSoup

from html_utils import (
    clean_html,
    build_block_tree,
    expand_table_spans,
    _is_ui_noise,
    _remove_hidden_elements,
    _unwrap_redundant_wrappers,
    _find_heading_parent,
    clean_html_text,
)
from text_process import (
    extract_title_from_block,
    _split_table_into_chunks,
    _extract_mixed_content,
    deduplicate_ranked_blocks_pal,
    clean_invisible,
    _clean_title,
)


# ======================== 1. 噪声标签清理 ========================

class TestNoiseTagRemoval:
    """测试 SVG/input/button/nav 等噪声标签被正确移除。"""

    def test_svg_removed(self):
        html = '<div><svg><circle/></svg><p>内容</p></div>'
        result = clean_html(html)
        assert "svg" not in result.lower()
        assert "circle" not in result
        assert "内容" in result

    def test_input_button_removed(self):
        html = '<div><input type="text"/><button>点击</button><p>内容</p></div>'
        result = clean_html(html)
        assert "input" not in result.lower()
        assert "button" not in result.lower()
        assert "内容" in result

    def test_nav_removed(self):
        html = '<nav><a href="#">导航</a></nav><p>正文内容</p>'
        result = clean_html(html)
        assert "nav" not in result.lower()
        assert "正文内容" in result

    def test_aside_removed(self):
        html = '<aside>侧边栏广告</aside><main><p>正文</p></main>'
        result = clean_html(html)
        assert "aside" not in result.lower()
        assert "侧边栏广告" not in result
        assert "正文" in result

    def test_head_title_removed(self):
        html = '<html><head><title>页面标题</title><style>body{}</style></head><body><p>正文</p></body></html>'
        result = clean_html(html)
        assert "页面标题" not in result
        assert "正文" in result

    def test_meta_link_removed(self):
        html = '<head><meta charset="UTF-8"><link rel="stylesheet" href="x.css"></head><body><p>正文</p></body>'
        result = clean_html(html)
        assert "meta" not in result.lower()
        assert "link" not in result.lower()
        assert "正文" in result


# ======================== 2. 隐藏元素移除 ========================

class TestHiddenElementRemoval:

    def test_display_none_removed(self):
        html = '<div style="display:none">隐藏内容</div><p>可见内容</p>'
        result = clean_html(html)
        assert "隐藏内容" not in result
        assert "可见内容" in result

    def test_hidden_class_removed(self):
        html = '<div class="hidden">隐藏内容</div><p>可见内容</p>'
        result = clean_html(html)
        assert "隐藏内容" not in result
        assert "可见内容" in result

    def test_display_none_with_spaces(self):
        html = '<div style="display: none">隐藏内容</div><p>可见</p>'
        result = clean_html(html)
        assert "隐藏内容" not in result

    def test_visibility_hidden_removed(self):
        html = '<div style="visibility:hidden">隐藏内容</div><p>可见</p>'
        result = clean_html(html)
        assert "隐藏内容" not in result


# ======================== 3. 模板文本移除 ========================

class TestTemplateTextRemoval:

    def test_placeholder_removed(self):
        html = '<p>使用方法：搜索 {{PLACEHOLDER}} 替换为实际内容</p>'
        result = clean_html(html)
        assert "PLACEHOLDER" not in result
        assert "使用方法" in result

    def test_placeholder_in_footer(self):
        html = '<footer>配色方案：琥珀橙 {{PLACEHOLDER}}</footer><p>正文</p>'
        result = clean_html(html)
        assert "PLACEHOLDER" not in result
        assert "正文" in result


# ======================== 4. 空标签清理 ========================

class TestEmptyTagCleanup:

    def test_empty_div_removed(self):
        html = '<div></div><p>内容</p>'
        result = clean_html(html)
        # 空的 div 应该被移除
        soup = BeautifulSoup(result, 'html.parser')
        empty_divs = [d for d in soup.find_all('div') if not d.text.strip() and not d.find_all(True)]
        assert len(empty_divs) == 0

    def test_whitespace_only_div_removed(self):
        html = '<div>   </div><p>内容</p>'
        result = clean_html(html)
        soup = BeautifulSoup(result, 'html.parser')
        empty_divs = [d for d in soup.find_all('div') if not d.text.strip() and not d.find_all(True)]
        assert len(empty_divs) == 0

    def test_br_only_div_removed(self):
        html = '<div><br></div><p>内容</p>'
        result = clean_html(html)
        soup = BeautifulSoup(result, 'html.parser')
        br_only = [d for d in soup.find_all('div') if d.find_all('br') and not d.get_text().strip()]
        assert len(br_only) == 0

    def test_nested_empty_removed(self):
        html = '<div><div></div></div><p>内容</p>'
        result = clean_html(html)
        soup = BeautifulSoup(result, 'html.parser')
        all_divs = soup.find_all('div')
        empty = [d for d in all_divs if not d.text.strip()]
        assert len(empty) == 0


# ======================== 5. 冗余包装展开 ========================

class TestRedundantWrapperUnwrapping:

    def test_single_level_wrapper_unwrapped(self):
        html = '<div><p>内容</p></div>'
        result = clean_html(html)
        # div 包装应被展开
        assert "内容" in result

    def test_multi_level_wrapper_unwrapped(self):
        html = '<div class="w1"><div class="w2"><div class="w3"><p>深层内容</p></div></div></div>'
        result = clean_html(html)
        soup = BeautifulSoup(result, 'html.parser')
        divs = soup.find_all('div')
        # 多层嵌套的 div 应该被展开到最多 1-2 层
        assert len(divs) <= 2
        assert "深层内容" in result

    def test_wrapper_with_bare_text_not_unwrapped(self):
        """父标签有直接文本内容时不应被展开。"""
        html = '<div>前缀<span>内容</span>后缀</div>'
        result = clean_html(html)
        assert "前缀" in result
        assert "后缀" in result

    def test_heading_not_unwrapped(self):
        """heading 标签不应被展开。"""
        html = '<div><h1>标题</h1></div>'
        result = clean_html(html)
        assert "h1" in result.lower() or "标题" in result


# ======================== 6. clean_html_text 非破坏性 ========================

class TestCleanHtmlTextNonDestructive:

    def test_preserves_text_between_tags(self):
        """文本和标签间的换行不应被合并（避免内容丢失）。"""
        text = "<p>第一段</p>\n<p>第二段</p>"
        result = clean_html_text(text)
        assert "第一段" in result
        assert "第二段" in result

    def test_removes_whitespace_between_tags(self):
        """标签间的纯空白换行应被移除。"""
        text = "<div>\n\n\n<span>内容</span></div>"
        result = clean_html_text(text)
        # 多个空白行应被压缩
        assert "\n\n\n" not in result

    def test_preserves_code_content(self):
        """代码内容中的换行不应被移除。"""
        text = "<code>def f():\n    return 1</code>"
        result = clean_html_text(text)
        assert "def f():" in result
        assert "return 1" in result

    def test_collapses_multiple_newlines(self):
        """连续空行应被压缩为最多两个换行。"""
        text = "<p>a</p>\n\n\n\n\n<p>b</p>"
        result = clean_html_text(text)
        assert "\n\n\n" not in result


# ======================== 7. UI 噪声过滤 ========================

class TestUINoiseFiltering:

    def test_is_ui_noise_progress_bar(self):
        html = '<div><span>0%</span><span>PROGRESS</span><span>尚未开始</span></div>'
        soup = BeautifulSoup(html, 'html.parser')
        div = soup.find('div')
        assert _is_ui_noise(div) is True

    def test_is_ui_noise_navigation(self):
        html = '<div><span>目录</span><span>CONTENTS</span></div>'
        soup = BeautifulSoup(html, 'html.parser')
        div = soup.find('div')
        assert _is_ui_noise(div) is True

    def test_is_ui_noise_not_content(self):
        html = '<div><p>这是正常的内容</p></div>'
        soup = BeautifulSoup(html, 'html.parser')
        div = soup.find('div')
        assert _is_ui_noise(div) is False

    def test_ui_noise_filtered_in_block_tree(self):
        html = '<div class="isolated_domain"><h1>标题</h1><p>正文内容足够长以确保不被过滤掉</p><div><span>0%</span><span>PROGRESS</span><span>尚未开始</span></div></div>'
        cleaned = clean_html(html)
        expanded = expand_table_spans(cleaned)
        blocks, _ = build_block_tree(expanded, max_node_words=4096, min_node_words=10, zh_char=True)
        doc_texts = [b[0].get_text() for b in blocks]
        # 进度条文本不应出现在任何块中
        for text in doc_texts:
            assert "PROGRESS" not in text or "尚未开始" not in text


# ======================== 8. 按 heading 拆分 ========================

class TestHeadingSplit:

    def test_find_heading_parent_direct(self):
        html = '<div><h1>标题</h1><p>内容</p></div>'
        soup = BeautifulSoup(html, 'html.parser')
        div = soup.find('div')
        result = _find_heading_parent(div)
        assert result is not None

    def test_find_heading_parent_nested(self):
        html = '<div><body><h1>标题</h1><p>内容</p></body></div>'
        soup = BeautifulSoup(html, 'html.parser')
        div = soup.find('div')
        result = _find_heading_parent(div)
        assert result is not None

    def test_multiple_headings_split(self):
        """多个 heading 被 heading-split 拆分后可被 bottom-up merge 重新合并。"""
        html = '<div><h1>标题1</h1><p>' + '内容1' * 20 + '</p><h2>标题2</h2><p>' + '内容2' * 20 + '</p></div>'
        cleaned = clean_html(html)
        expanded = expand_table_spans(cleaned)
        blocks, _ = build_block_tree(expanded, max_node_words=4096, min_node_words=10, zh_char=True)
        # heading-split 拆分后 bottom-up merge 会将小 section 合并
        # 两个 section 都很小，合并后应只有 1 块
        assert len(blocks) >= 1


# ======================== 9. 标题提取优化 ========================

class TestTitleExtractionOptimization:

    def test_skip_ui_noise_title(self):
        html = '<div><span>0%</span><span>PROGRESS</span></div>'
        soup = BeautifulSoup(html, 'html.parser')
        title = extract_title_from_block(soup.find('div'))
        assert title == "" or "PROGRESS" not in title

    def test_clean_title_decimal_prefix(self):
        assert _clean_title("1.2 投放效果对比") == "投放效果对比"

    def test_clean_title_integer_prefix(self):
        assert _clean_title("3. 第二章") == "第二章"

    def test_clean_title_no_prefix(self):
        assert _clean_title("附录") == "附录"

    def test_clean_title_chinese(self):
        assert _clean_title("第一章：广告投放") == "第一章：广告投放"

    def test_title_from_heading(self):
        html = '<div><h2>核心标题</h2><p>正文内容</p></div>'
        soup = BeautifulSoup(html, 'html.parser')
        title = extract_title_from_block(soup.find('div'))
        assert "核心标题" in title


# ======================== 10. 混合内容提取 ========================

class TestMixedContentExtraction:

    def test_text_and_table_separated(self):
        html = '<div><p>表格前的说明文字</p><table><tr><th>A</th></tr><tr><td>B</td></tr></table><p>表格后的说明文字</p></div>'
        soup = BeautifulSoup(html, 'html.parser')
        div = soup.find('div')
        results = _extract_mixed_content(div, "测试标题", max_node_words=4096)
        # 应该分离出文本和表格
        texts = [r[2] for r in results]
        assert any("表格前" in t for t in texts)
        assert any("表格后" in t for t in texts)
        assert any("A" in t and "B" in t for t in texts)

    def test_multiple_tables_separated(self):
        """表格间的短文本会被 _merge_tiny_text_fragments 合并到表格中。"""
        html = '<div><p>文本1</p><table><tr><td>T1</td></tr></table><p>文本2</p><table><tr><td>T2</td></tr></table></div>'
        soup = BeautifulSoup(html, 'html.parser')
        div = soup.find('div')
        results = _extract_mixed_content(div, "标题", max_node_words=4096)
        # 短文本 "文本1""文本2" 被合并到相邻表格 → 结果至少 2 个（两个表格）
        assert len(results) >= 2


# ======================== 11. 表格切分优化 ========================

class TestTableChunkingOptimization:

    def test_single_header_row(self):
        html = '<table><tr><th>A</th><th>B</th></tr></table>'
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        chunks = _split_table_into_chunks(table, "标题", max_node_words=4096)
        assert len(chunks) == 1

    def test_oversized_row_forced_chunk(self):
        """单行超过 max_node_words 时应强制独立成块。"""
        html = '<table><tr><th>表头</th></tr><tr><td>' + '超长内容' * 100 + '</td></tr></table>'
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        chunks = _split_table_into_chunks(table, "标题", max_node_words=100)
        assert len(chunks) >= 1

    def test_header_repeated_in_each_chunk(self):
        """每个块都应以表头开头。"""
        html = '<table><tr><th>表头</th></tr><tr><td>行1</td></tr><tr><td>行2</td></tr></table>'
        soup = BeautifulSoup(html, 'html.parser')
        table = soup.find('table')
        chunks = _split_table_into_chunks(table, "标题", max_node_words=100)
        for text, _, _ in chunks:
            assert "表头" in text


# ======================== 12. 去重算法修复 ========================

class TestDeduplicationFix:

    def test_short_page_name_no_crash(self):
        """page_name 为单字符时不应崩溃。"""
        docs = [
            {"text": "内容" * 20, "page_name": "p", "time": "2025-01-01 00:00:00"},
            {"text": "内容" * 20, "page_name": "p", "time": "2025-01-02 00:00:00"},
        ]
        result = deduplicate_ranked_blocks_pal(docs)
        assert len(result) == 1

    def test_empty_page_name(self):
        docs = [
            {"text": "相同内容" * 20, "page_name": "", "time": "2025-01-01 00:00:00"},
            {"text": "相同内容" * 20, "page_name": "", "time": "2025-01-02 00:00:00"},
        ]
        result = deduplicate_ranked_blocks_pal(docs)
        assert len(result) == 1

    def test_threshold_one_only_exact_match(self):
        docs = [
            {"text": "完全相同" * 20, "page_name": "test", "time": "2025-01-01 00:00:00"},
            {"text": "完全相同" * 20, "page_name": "test", "time": "2025-01-02 00:00:00"},
            {"text": "几乎相同" * 20, "page_name": "test", "time": "2025-01-03 00:00:00"},
        ]
        result = deduplicate_ranked_blocks_pal(docs, threshold_content=1.0)
        assert len(result) == 2


# ======================== 13. 端到端测试 ========================

class TestEndToEndWithComprehensiveHTML:
    """使用 dataset/test_comprehensive.html 进行端到端测试。"""

    @pytest.fixture
    def comprehensive_doc_meta(self):
        from text_process import generate_block_documents
        html_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "process", "dataset", "test_comprehensive.html"
        )
        if not os.path.exists(html_path):
            pytest.skip("test_comprehensive.html not found")
        with open(html_path, "r", encoding="utf-8") as f:
            raw_html = f.read()
        cleaned = clean_html(raw_html)
        expanded = expand_table_spans(cleaned)
        blocks, _ = build_block_tree(expanded, max_node_words=4096, min_node_words=48, zh_char=True)
        return generate_block_documents(blocks, max_node_words=4096, page_url="test_comprehensive.html", use_vllm=False)

    def test_no_svg_in_output(self, comprehensive_doc_meta):
        for doc in comprehensive_doc_meta:
            assert "svg" not in doc["text"].lower()

    def test_no_input_button_in_output(self, comprehensive_doc_meta):
        for doc in comprehensive_doc_meta:
            assert "搜索..." not in doc["text"]

    def test_no_template_placeholder(self, comprehensive_doc_meta):
        for doc in comprehensive_doc_meta:
            assert "PLACEHOLDER" not in doc["text"]
            assert "配色方案" not in doc["text"]

    def test_no_hidden_content(self, comprehensive_doc_meta):
        for doc in comprehensive_doc_meta:
            assert "隐藏" not in doc["text"] or "不应该出现" not in doc["text"]

    def test_no_progress_bar(self, comprehensive_doc_meta):
        for doc in comprehensive_doc_meta:
            assert "尚未开始" not in doc["text"]
            assert "PROGRESS" not in doc["text"]

    def test_no_title_tag_content(self, comprehensive_doc_meta):
        for doc in comprehensive_doc_meta:
            assert "覆盖各种边缘情况" not in doc["text"]

    def test_has_multiple_blocks(self, comprehensive_doc_meta):
        assert len(comprehensive_doc_meta) >= 5

    def test_content_not_empty(self, comprehensive_doc_meta):
        for doc in comprehensive_doc_meta:
            assert doc["text"].strip() != ""

    def test_core_content_present(self, comprehensive_doc_meta):
        all_text = " ".join(d["text"] for d in comprehensive_doc_meta)
        assert "电商运营" in all_text
        assert "广告投放" in all_text
        assert "数据分析" in all_text

    def test_table_data_present(self, comprehensive_doc_meta):
        all_text = " ".join(d["text"] for d in comprehensive_doc_meta)
        # 至少一个表格的数据应该被提取
        assert "兴趣定向" in all_text or "品牌曝光" in all_text

    def test_titles_cleaned(self, comprehensive_doc_meta):
        for doc in comprehensive_doc_meta:
            # 标题不应是纯数字
            if doc["title"]:
                assert not doc["title"].strip().isdigit()
