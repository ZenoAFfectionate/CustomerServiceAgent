# -*- coding: utf-8 -*-
"""rag/indexing/document_loader.py 单元测试：多格式文档解析（txt/md/html/json/pdf）。"""
import pytest

from rag.indexing.document_loader import ParseError, parse_document


class TestParseTxtAndMd:
    def test_parse_txt_returns_decoded_text(self):
        result = parse_document("a.txt", "纯文本内容".encode("utf-8"))
        assert result == "纯文本内容"

    def test_parse_md_returns_decoded_text(self):
        result = parse_document("a.md", "# 标题\n正文内容".encode("utf-8"))
        assert "正文内容" in result

    def test_parse_txt_ignores_undecodable_bytes(self):
        raw = "正常内容".encode("utf-8") + b"\xff\xfe"
        result = parse_document("a.txt", raw)
        assert "正常内容" in result


class TestParseHtml:
    def test_parse_html_strips_tags(self):
        html = "<html><body><p>正文内容</p><script>alert(1)</script></body></html>"
        result = parse_document("a.html", html.encode("utf-8"))
        assert "正文内容" in result
        assert "alert" not in result

    def test_parse_html_removes_nav_and_footer(self):
        html = "<html><body><nav>导航栏</nav><p>核心内容</p><footer>页脚</footer></body></html>"
        result = parse_document("a.htm", html.encode("utf-8"))
        assert "核心内容" in result
        assert "导航栏" not in result
        assert "页脚" not in result

    def test_parse_html_removes_empty_lines(self):
        html = "<html><body><p>行一</p>\n\n\n<p>行二</p></body></html>"
        result = parse_document("a.html", html.encode("utf-8"))
        lines = result.splitlines()
        assert all(ln.strip() for ln in lines)


class TestParseJsonBlocks:
    def test_parse_json_array_returns_list(self):
        raw = '[{"text": "块一"}, {"text": "块二"}]'.encode("utf-8")
        result = parse_document("a.json", raw)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_parse_json_single_object_wrapped_as_list(self):
        raw = '{"text": "单个对象"}'.encode("utf-8")
        result = parse_document("a.json", raw)
        assert isinstance(result, list)
        assert len(result) == 1

    def test_parse_malformed_json_raises_parse_error(self):
        with pytest.raises(ParseError):
            parse_document("a.json", b"{not valid json!!!")

    def test_parse_json_non_list_non_dict_raises_parse_error(self):
        with pytest.raises(ParseError):
            parse_document("a.json", b'"just a string"')


class TestUnsupportedFormat:
    def test_unsupported_extension_raises_parse_error(self):
        with pytest.raises(ParseError):
            parse_document("a.exe", b"binary content")

    def test_extension_matching_is_case_insensitive(self):
        result = parse_document("A.TXT", "内容".encode("utf-8"))
        assert result == "内容"
