# -*- coding: utf-8 -*-
"""文档解析：将上传文件解析为纯文本。

支持格式：.txt / .md / .html / .htm / .json（process/ 输出的知识块数组）/ .pdf（可选）。
每种格式的解析逻辑相互独立，新增格式只需新增一个 `_parse_xxx` 函数并注册到
`PARSERS`，符合"组件可独立替换/扩展"的设计原则。
"""
import json
import os
from typing import List

from bs4 import BeautifulSoup


class ParseError(Exception):
    """文档解析失败。"""


def _parse_txt(raw: bytes, **_) -> str:
    return raw.decode("utf-8", errors="ignore")


def _parse_md(raw: bytes, **_) -> str:
    return raw.decode("utf-8", errors="ignore")


def _parse_html(raw: bytes, **_) -> str:
    html = raw.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "svg", "nav", "footer", "head", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def _parse_json_blocks(raw: bytes, **_) -> List[dict]:
    """解析 process/ 输出的知识块 JSON（list[dict]），直接返回块列表供上层跳过分块步骤。"""
    data = json.loads(raw.decode("utf-8", errors="ignore"))
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ParseError("JSON 内容必须是文档块数组（list[dict]）")
    return data


def _parse_pdf(raw: bytes, **_) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ParseError("解析 PDF 需要安装 pypdf：pip install pypdf") from e
    import io
    reader = PdfReader(io.BytesIO(raw))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


PARSERS = {
    ".txt": _parse_txt,
    ".md": _parse_md,
    ".html": _parse_html,
    ".htm": _parse_html,
    ".json": _parse_json_blocks,
    ".pdf": _parse_pdf,
}


def parse_document(filename: str, raw: bytes):
    """解析文档内容。

    Args:
        filename: 原始文件名（用于判断扩展名）
        raw:      文件二进制内容

    Returns:
        - 对 .json：返回 List[dict]（已是文档块，跳过分块步骤）
        - 其他格式：返回 str（纯文本，交给 chunker 分块）
    """
    ext = os.path.splitext(filename)[1].lower()
    parser = PARSERS.get(ext)
    if parser is None:
        raise ParseError(f"不支持的文件类型: {ext}，支持: {sorted(PARSERS)}")
    try:
        return parser(raw)
    except ParseError:
        raise
    except Exception as e:
        raise ParseError(f"解析 {filename} 失败: {e}") from e
