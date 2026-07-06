# -*- coding: utf-8 -*-
"""文本分块（Chunking，原 chunker.py，重命名以对齐新的模块命名规范）。

独立于 `process/` 的 HTML Block Tree 分块（那是针对结构化 HTML 的专用算法）。
本模块面向"通用纯文本"输入（如用户直接上传的 .txt/.md/.pdf/清洗后的 HTML 文本），
提供一个简单、可控、可单测的**递归字符分块器**，支持重叠窗口以保留上下文连续性。

设计为纯函数，不依赖任何外部服务，可独立替换为更复杂的分块策略
（如按句子/语义分块）而不影响上游 loader 与下游 embedder。
"""
import re
from typing import List

# 优先在这些分隔符处切分，尽量保持语义完整（先按段落，再句子，再逗号，最后硬切字符）
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", ". ", "! ", "? ", ", ", " "]


def _split_keep_separator(text: str, sep: str) -> List[str]:
    """按 sep 拆分，并把 sep 拼回每个片段末尾（保留原文，方便重新拼接）。"""
    parts = text.split(sep)
    n = len(parts)
    return [p + sep if i < n - 1 else p for i, p in enumerate(parts)]


def _recursive_split(text: str, chunk_size: int, sep_idx: int = 0) -> List[str]:
    """按 `_SEPARATORS` 优先级递归拆分为不超过 chunk_size 的原子片段。

    分隔符全部用尽仍超长时，按字符硬切。
    """
    if len(text) <= chunk_size:
        return [text] if text else []
    if sep_idx >= len(_SEPARATORS):
        # 无可用分隔符：按字符硬切
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep = _SEPARATORS[sep_idx]
    if sep not in text:
        return _recursive_split(text, chunk_size, sep_idx + 1)

    result: List[str] = []
    for part in _split_keep_separator(text, sep):
        if not part:
            continue
        if len(part) > chunk_size:
            result.extend(_recursive_split(part, chunk_size, sep_idx + 1))
        else:
            result.append(part)
    return result


def _greedy_pack(atoms: List[str], chunk_size: int) -> List[str]:
    """将原子片段贪心拼装到不超过 chunk_size 的块中。"""
    chunks: List[str] = []
    buf = ""
    for atom in atoms:
        if len(buf) + len(atom) <= chunk_size:
            buf += atom
        else:
            if buf:
                chunks.append(buf)
            buf = atom
    if buf:
        chunks.append(buf)
    return chunks


def _apply_overlap(chunks: List[str], overlap: int, chunk_size: int = 0) -> List[str]:
    """在相邻块之间补上重叠内容（取前一块末尾 overlap 个字符前置到下一块）。

    【修复 N34】此前直接 ``prev_tail + chunks[i]`` 使每块膨胀 overlap 字符，
    长度可能明显超过 chunk_size，重叠内容被重复嵌入/计入。现确保拼接后
    不超过 chunk_size（若传入），超限时截断当前块尾部以容纳重叠前缀。
    """
    if len(chunks) <= 1 or overlap <= 0:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        current = chunks[i]
        # 若拼接后超过 chunk_size，截断当前块尾部以容纳重叠前缀
        if chunk_size > 0 and len(prev_tail) + len(current) > chunk_size:
            current = current[:chunk_size - len(prev_tail)]
        out.append(prev_tail + current)
    return out


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """将长文本切分为若干重叠的块。

    Args:
        text: 原始文本
        chunk_size: 单块最大字符数
        chunk_overlap: 相邻块之间的重叠字符数（用于保留跨块语义连续性）

    Returns:
        文本块列表（已去除首尾空白与空块）；空文本返回 []。

    Raises:
        ValueError: chunk_size <= 0 或 chunk_overlap >= chunk_size
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须为正整数")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须满足 0 <= chunk_overlap < chunk_size")

    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    atoms = _recursive_split(text, chunk_size)
    chunks = _greedy_pack(atoms, chunk_size)
    chunks = _apply_overlap(chunks, chunk_overlap, chunk_size)
    return [c.strip() for c in chunks if c.strip()]


def count_words(text: str, zh_char: bool = True) -> int:
    """统计词数（中文按字符计，英文按空格分词计，对齐 process/ 的 zh_char 约定）。"""
    if not text:
        return 0
    if zh_char:
        return len(re.sub(r"\s", "", text))
    return len(text.split())
