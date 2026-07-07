# -*- coding: utf-8 -*-
"""
飞书 HTML 清洗与 HtmlRAG 剪枝改进测试。

覆盖新增功能：
    1. _clean_feishu_noise：飞书噪声预处理
       - 移除 fold-wrapper / placeholder / emoji / SVG / bullet-dot
       - 移除零宽字符（data-enter / data-zero-space）
       - 展平飞书包装层（zone-container / text-editor / heading-block 等）
       - 展平飞书 span（data-leaf / data-string）
       - 不影响普通 HTML span（UI 噪声检测不受影响）
       - callout 内容不被 block-comment 误删
       - heading 文本不被 heading-block-align- 误删
    2. _convert_semantic_blocks：语义块转换
       - text → <p>
       - bullet/ordered → <li>
       - callout → unwrap（保留内容）
       - quote_container → <blockquote>
       - grid/grid_column → unwrap
    3. _convert_headings：标题转换后清除子元素残留 data-block-type
    4. rebuild_html_with_domains：保留 domain 层次结构
    5. _extract_heading_context：标题上下文提取（_domain 和 _section 两种格式）

运行方式：
    PYTHONPATH=process:process/src pytest tests/test_process/test_feishu_cleaning.py -v
"""

import pytest
from bs4 import BeautifulSoup

from html_utils import clean_html, build_block_tree, _clean_feishu_noise
from html_pruner import (
    rebuild_html,
    rebuild_html_with_domains,
    _extract_heading_context,
    _wrap_domain_group,
    greedy_prune_indices,
    prune_by_embedding,
    prune_by_reranker,
)


# ======================== 飞书 HTML 样本 ========================

# 飞书标题块样本
FEISHU_HEADING_HTML = '''<div class="block docx-heading1-block" data-block-type="heading1">
<div class="heading-block"><div class="heading heading-h1 heading-block-align-">
<div class="heading-content"><div class="zone-container text-editor hide-placeholder non-empty">
<div class="ace-line" data-node="true" dir="auto">
<span class="author-123" data-leaf="true" data-string="true">产品简介</span>
<span data-enter="true" data-leaf="true" data-string="true">​</span>
</div></div></div></div></div></div>
<div class="fold-wrapper can-fold fold-block-id-2">
<div class="fold-handler"><div class="svg-wrapper"><svg></svg></div></div>
</div>'''

# 飞书文本块样本
FEISHU_TEXT_HTML = '''<div class="block docx-text-block" data-block-type="text">
<div class="text-block-wrapper"><div class="text-block">
<div class="zone-container text-editor hide-placeholder non-empty">
<div class="ace-line" data-node="true" dir="auto">
<span data-leaf="true" data-string="true">这是一段段落文本</span>
<span data-enter="true">​</span>
</div></div></div></div></div>'''

# 飞书列表块样本
FEISHU_BULLET_HTML = '''<div class="block docx-bullet-block" data-block-type="bullet">
<div class="list-wrapper bullet-list"><div class="list list-style-group-1 list-align-">
<div class="bullet bulletUnedit"><div class="bullet-dot-style">•</div></div>
<div class="list-content"><div class="zone-container text-editor">
<div class="ace-line"><span data-leaf="true" data-string="true">列表项文本</span></div>
</div></div></div></div></div>'''

# 飞书 callout 块样本（内容不应被 block-comment 删除）
FEISHU_CALLOUT_HTML = '''<div class="block docx-callout-block" data-block-type="callout">
<div class="docx-block-zero-space"><span data-zero-space="true">​</span></div>
<div class="block-comment callout-block-comment local-comment-all-third-party">
<div class="docx-callout-block-container docx-block-align-left">
<div class="docx-callout-block-inner-container">
<div class="callout-block">
<div class="callout-emoji-container emoji-for-heading2">
<div class="callout-block-emoji disabled">
<span class="emoji-mart-emoji"><span style="font-size: 20px;">⭐</span></span>
</div></div>
<div class="callout-block-children">
<div class="render-unit-wrapper callout-render-unit">
<div class="block docx-text-block" data-block-type="text">
<div class="text-block-wrapper"><div class="text-block">
<div class="zone-container text-editor">
<div class="ace-line"><span data-leaf="true">callout 内的实际内容</span></div>
</div></div></div></div></div></div></div></div></div></div></div>
<div class="docx-block-zero-space"><span data-zero-space="true">​</span></div>
</div>'''

# 飞书引用块样本
FEISHU_QUOTE_HTML = '''<div class="block docx-quote_container-block" data-block-type="quote_container">
<div class="docx-block-zero-space"><span data-zero-space="true">​</span></div>
<div class="quote-container-block"><div class="quote-container-block-children">
<div class="render-unit-wrapper quote-container-render-unit">
<div class="block docx-text-block" data-block-type="text">
<div class="text-block-wrapper"><div class="text-block">
<div class="zone-container text-editor">
<div class="ace-line"><span data-leaf="true">引用内容文本</span></div>
</div></div></div></div></div></div></div>
<div class="docx-block-zero-space"><span data-zero-space="true">​</span></div>
</div>'''


# ======================== 1. _clean_feishu_noise 测试 ========================

class TestCleanFeishuNoiseHeading:
    """飞书标题块清洗测试"""

    def test_heading_text_preserved(self):
        """标题文本不应被 heading-block-align- 误删"""
        soup = BeautifulSoup(FEISHU_HEADING_HTML, "html.parser")
        _clean_feishu_noise(soup)
        text = soup.get_text()
        assert "产品简介" in text

    def test_fold_wrapper_removed(self):
        """fold-wrapper 折叠按钮应被移除"""
        soup = BeautifulSoup(FEISHU_HEADING_HTML, "html.parser")
        _clean_feishu_noise(soup)
        assert not soup.find(class_="fold-wrapper")

    def test_svg_removed(self):
        """SVG 图标应被移除"""
        soup = BeautifulSoup(FEISHU_HEADING_HTML, "html.parser")
        _clean_feishu_noise(soup)
        assert not soup.find("svg")

    def test_zero_width_removed(self):
        """data-enter 零宽字符应被移除"""
        soup = BeautifulSoup(FEISHU_HEADING_HTML, "html.parser")
        _clean_feishu_noise(soup)
        assert not soup.find(attrs={"data-enter": "true"})

    def test_feishu_span_unwrapped(self):
        """飞书 span（data-leaf）应被展平"""
        soup = BeautifulSoup(FEISHU_HEADING_HTML, "html.parser")
        _clean_feishu_noise(soup)
        spans = soup.find_all("span", attrs={"data-leaf": True})
        assert len(spans) == 0


class TestCleanFeishuNoiseText:
    """飞书文本块清洗测试"""

    def test_text_preserved(self):
        soup = BeautifulSoup(FEISHU_TEXT_HTML, "html.parser")
        _clean_feishu_noise(soup)
        assert "这是一段段落文本" in soup.get_text()

    def test_wrapper_unwrapped(self):
        """text-block-wrapper / text-block / zone-container 等包装层应被展平"""
        soup = BeautifulSoup(FEISHU_TEXT_HTML, "html.parser")
        _clean_feishu_noise(soup)
        assert not soup.find(class_="zone-container")
        assert not soup.find(class_="text-block-wrapper")


class TestCleanFeishuNoiseBullet:
    """飞书列表块清洗测试"""

    def test_bullet_text_preserved(self):
        soup = BeautifulSoup(FEISHU_BULLET_HTML, "html.parser")
        _clean_feishu_noise(soup)
        assert "列表项文本" in soup.get_text()

    def test_bullet_dot_removed(self):
        """bullet-dot-style 项目符号应被移除"""
        soup = BeautifulSoup(FEISHU_BULLET_HTML, "html.parser")
        _clean_feishu_noise(soup)
        assert not soup.find(class_="bullet-dot-style")

    def test_bullet_unedit_removed(self):
        """bulletUnedit 类应被移除"""
        soup = BeautifulSoup(FEISHU_BULLET_HTML, "html.parser")
        _clean_feishu_noise(soup)
        assert not soup.find(class_="bulletUnedit")


class TestCleanFeishuNoiseCallout:
    """飞书 callout 块清洗测试 — 关键回归：内容不被 block-comment 误删"""

    def test_callout_content_preserved(self):
        """callout 内的实际内容不应被 block-comment 误删"""
        soup = BeautifulSoup(FEISHU_CALLOUT_HTML, "html.parser")
        _clean_feishu_noise(soup)
        text = soup.get_text()
        assert "callout 内的实际内容" in text

    def test_emoji_container_removed(self):
        """emoji 容器应被移除"""
        soup = BeautifulSoup(FEISHU_CALLOUT_HTML, "html.parser")
        _clean_feishu_noise(soup)
        assert not soup.find(class_="callout-emoji-container")
        assert not soup.find(class_="emoji-mart-emoji")

    def test_zero_space_removed(self):
        """docx-block-zero-space 零宽空格应被移除"""
        soup = BeautifulSoup(FEISHU_CALLOUT_HTML, "html.parser")
        _clean_feishu_noise(soup)
        assert not soup.find(attrs={"data-zero-space": "true"})

    def test_callout_wrapper_unwrapped(self):
        """block-comment / callout-block-comment 应被展平（非删除）"""
        soup = BeautifulSoup(FEISHU_CALLOUT_HTML, "html.parser")
        _clean_feishu_noise(soup)
        # block-comment 已被 unwrap，不再存在于树中
        assert not soup.find(class_="block-comment")
        # 但内容应保留
        assert "callout 内的实际内容" in soup.get_text()


class TestCleanFeishuNoiseQuote:
    """飞书引用块清洗测试"""

    def test_quote_content_preserved(self):
        soup = BeautifulSoup(FEISHU_QUOTE_HTML, "html.parser")
        _clean_feishu_noise(soup)
        assert "引用内容文本" in soup.get_text()


class TestCleanFeishuNoisePlainHTML:
    """普通 HTML 不受飞书清洗影响"""

    def test_plain_span_not_unwrapped(self):
        """普通 HTML span 不应被展平（UI 噪声检测依赖 span 分隔）"""
        html = '<div><span>0%</span><span>PROGRESS</span></div>'
        soup = BeautifulSoup(html, "html.parser")
        _clean_feishu_noise(soup)
        spans = soup.find_all("span")
        assert len(spans) == 2

    def test_plain_html_content_preserved(self):
        """普通 HTML 内容不受飞书清洗影响"""
        html = '<div><h1>标题</h1><p>正文内容</p></div>'
        soup = BeautifulSoup(html, "html.parser")
        _clean_feishu_noise(soup)
        assert "标题" in soup.get_text()
        assert "正文内容" in soup.get_text()


# ======================== 2. 语义块转换测试 ========================

class TestSemanticBlockConversion:
    """data-block-type → 标准 HTML 标签转换测试"""

    def test_text_to_p(self):
        """data-block-type='text' → <p>"""
        cleaned = clean_html(FEISHU_TEXT_HTML)
        assert "<p>" in cleaned
        assert "这是一段段落文本" in cleaned

    def test_bullet_to_li(self):
        """data-block-type='bullet' → <li>"""
        cleaned = clean_html(FEISHU_BULLET_HTML)
        assert "<li>" in cleaned
        assert "列表项文本" in cleaned

    def test_callout_unwrapped(self):
        """data-block-type='callout' → unwrap（内容保留）"""
        cleaned = clean_html(FEISHU_CALLOUT_HTML)
        assert "callout 内的实际内容" in cleaned

    def test_quote_to_blockquote(self):
        """data-block-type='quote_container' → <blockquote>"""
        cleaned = clean_html(FEISHU_QUOTE_HTML)
        assert "<blockquote>" in cleaned
        assert "引用内容文本" in cleaned

    def test_heading_to_h1(self):
        """data-block-type='heading1' → <h1>"""
        cleaned = clean_html(FEISHU_HEADING_HTML)
        assert "<h1>" in cleaned
        assert "产品简介" in cleaned


class TestHeadingConversionNoNestedDuplicate:
    """标题转换后不应有嵌套重复"""

    def test_no_nested_heading(self):
        """内外两层 heading 不应都被转换为 <h1>"""
        html = '''<div class="block docx-heading1-block" data-block-type="heading1">
<div class="heading-block"><div class="heading heading-h1 heading-block-align-">
<div class="heading-content"><div class="zone-container text-editor">
<div class="ace-line"><span data-leaf="true">标题</span></div>
</div></div></div></div></div>'''
        cleaned = clean_html(html)
        # 只应有一个 <h1>，不应有嵌套 <h1><h1>
        h1_count = cleaned.count("<h1>")
        assert h1_count == 1, f"Expected 1 <h1>, got {h1_count}: {cleaned}"


# ======================== 3. 完整清洗端到端测试 ========================

class TestFeishuEndToEnd:
    """飞书 HTML 完整清洗端到端测试"""

    def test_full_feishu_doc_cleaned(self):
        """完整的飞书文档（含标题 + 文本 + 列表 + 表格）应正确清洗"""
        html = f'''{FEISHU_HEADING_HTML}
{FEISHU_TEXT_HTML}
{FEISHU_BULLET_HTML}'''
        cleaned = clean_html(html)
        # 标题
        assert "<h1>" in cleaned and "产品简介" in cleaned
        # 段落
        assert "<p>" in cleaned and "这是一段段落文本" in cleaned
        # 列表
        assert "<li>" in cleaned and "列表项文本" in cleaned
        # 噪声已移除
        assert "fold-wrapper" not in cleaned
        assert "svg" not in cleaned.lower()
        assert "bullet-dot-style" not in cleaned
        assert "zone-container" not in cleaned
        assert "data-leaf" not in cleaned

    def test_cleaned_html_can_be_block_tree(self):
        """清洗后的 HTML 能正常分块"""
        html = f'''{FEISHU_HEADING_HTML}
{FEISHU_TEXT_HTML}'''
        cleaned = clean_html(html)
        blocks, _ = build_block_tree(cleaned, max_node_words=512, min_node_words=5, zh_char=True)
        assert len(blocks) >= 1
        all_text = "".join(b[0].get_text() for b in blocks)
        assert "产品简介" in all_text
        assert "这是一段段落文本" in all_text


# ======================== 4. rebuild_html_with_domains 测试 ========================

class TestRebuildHtmlWithDomains:
    """保留 domain 层次结构的 HTML 重建测试"""

    def test_empty(self):
        assert rebuild_html_with_domains([], []) == ""

    def test_single_domain(self):
        """同一 domain 下的块应被包装在一起"""
        soup = BeautifulSoup("<p>A</p><p>B</p>", "html.parser")
        blocks = soup.find_all("p")
        paths = [["h1_domain", "p"], ["h1_domain", "p"]]
        result = rebuild_html_with_domains(blocks, paths)
        assert 'class="h1_domain"' in result
        assert "A" in result and "B" in result

    def test_different_domains_separated(self):
        """不同 domain 的块应分别包装"""
        soup = BeautifulSoup("<p>A</p><p>B</p>", "html.parser")
        blocks = soup.find_all("p")
        paths = [["h1_domain", "p"], ["h2_domain", "p"]]
        result = rebuild_html_with_domains(blocks, paths)
        assert 'class="h1_domain"' in result
        assert 'class="h2_domain"' in result

    def test_isolated_domain_not_wrapped(self):
        """isolated_domain 不应生成包装 div"""
        soup = BeautifulSoup("<p>A</p>", "html.parser")
        blocks = [soup.find("p")]
        paths = [["isolated_domain", "p"]]
        result = rebuild_html_with_domains(blocks, paths)
        assert "isolated_domain" not in result
        assert "A" in result

    def test_no_path_not_wrapped(self):
        """空路径的块不应被包装"""
        soup = BeautifulSoup("<p>A</p>", "html.parser")
        blocks = [soup.find("p")]
        paths = [[]]
        result = rebuild_html_with_domains(blocks, paths)
        assert "class=" not in result
        assert "A" in result


# ======================== 5. _extract_heading_context 测试 ========================

class TestExtractHeadingContext:
    """标题上下文提取测试"""

    def test_domain_path(self):
        path = ["h1_domain", "h2_domain", "p"]
        assert _extract_heading_context(path) == "h1 h2"

    def test_section_path(self):
        path = ["h2_section"]
        assert _extract_heading_context(path) == "h2"

    def test_mixed_path(self):
        path = ["h1_domain", "h3_section", "div0"]
        assert _extract_heading_context(path) == "h1 h3"

    def test_empty_path(self):
        assert _extract_heading_context([]) == ""

    def test_no_heading_path(self):
        path = ["isolated_domain", "p"]
        assert _extract_heading_context(path) == ""


# ======================== 6. _wrap_domain_group 测试 ========================

class TestWrapDomainGroup:
    def test_h1_domain_wrapped(self):
        soup = BeautifulSoup("<p>text</p>", "html.parser")
        result = _wrap_domain_group([soup.find("p")], "h1_domain")
        assert 'class="h1_domain"' in result
        assert "text" in result

    def test_isolated_domain_not_wrapped(self):
        soup = BeautifulSoup("<p>text</p>", "html.parser")
        result = _wrap_domain_group([soup.find("p")], "isolated_domain")
        assert "class=" not in result
        assert "text" in result

    def test_empty_domain_not_wrapped(self):
        soup = BeautifulSoup("<p>text</p>", "html.parser")
        result = _wrap_domain_group([soup.find("p")], "")
        assert "class=" not in result
        assert "text" in result


# ======================== 7. 剪枝集成测试（验证新功能不破坏剪枝） ========================

class TestPruningWithImprovements:
    """验证剪枝流程在新改进下正常工作"""

    # 使用足够长的内容确保 build_block_tree 创建多个独立块
    SAMPLE_HTML = """<html>
<div class="h1_domain"><h1>广告投放规则</h1><p>广告投放需要遵守平台的推广素材审核规范才能上线投放广告内容必须符合审核要求</p></div>
<div class="h1_domain"><h1>退款政策</h1><p>退款需要在收到商品七天内申请并保证商品完好无损坏才能获得退款</p></div>
<div class="h1_domain"><h1>物流配送</h1><p>物流配送范围覆盖全国大部分地区并支持次日达服务配送时效有保障</p></div>
</html>"""

    def _mock_embed(self, texts):
        """简单 mock：根据关键词返回向量"""
        vectors = []
        for t in texts:
            vec = [0.0, 0.0, 0.0]
            if "退款" in t or "退货" in t:
                vec[1] = 1.0
            if "广告" in t or "投放" in t:
                vec[0] = 1.0
            if "物流" in t or "配送" in t:
                vec[2] = 1.0
            vectors.append(vec)
        return vectors

    def _mock_rerank(self, query, texts):
        return [1.0 if "退款" in t else 0.1 for t in texts]

    def test_embedding_prune_works(self):
        """Stage 1 嵌入剪枝在新改进下正常工作"""
        out = prune_by_embedding(
            self.SAMPLE_HTML, "如何退款",
            max_context_words=40,
            embed_fn=self._mock_embed,
            max_node_words=30, min_node_words=5, zh_char=True,
        )
        # 退款相关内容必须保留
        assert "退款" in out
        # 剪枝后的内容应比原始内容短（发生了剪枝）
        original_text = BeautifulSoup(self.SAMPLE_HTML, "html.parser").get_text()
        out_text = BeautifulSoup(out, "html.parser").get_text()
        assert len(out_text) < len(original_text)

    def test_reranker_prune_works(self):
        """Stage 2 精排剪枝在新改进下正常工作"""
        out = prune_by_reranker(
            self.SAMPLE_HTML, "退款问题",
            max_context_words=40,
            rerank_fn=self._mock_rerank,
            max_node_words=30, min_node_words=5, zh_char=True,
        )
        # 退款相关内容必须保留
        assert "退款" in out
        # 剪枝后的内容应比原始内容短
        original_text = BeautifulSoup(self.SAMPLE_HTML, "html.parser").get_text()
        out_text = BeautifulSoup(out, "html.parser").get_text()
        assert len(out_text) < len(original_text)

    def test_graceful_degradation(self):
        """打分失败时优雅降级，保留所有块"""
        def broken(query, texts):
            raise RuntimeError("service down")

        out = prune_by_reranker(
            self.SAMPLE_HTML, "退款",
            max_context_words=40,
            rerank_fn=broken,
            max_node_words=30, min_node_words=5, zh_char=True,
        )
        # 降级后应保留所有内容
        assert "退款" in out
        assert "广告" in out
        assert "物流" in out


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
