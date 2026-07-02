# -*- coding: utf-8 -*-
"""rag/indexing/chunker.py 单元测试：分块正常场景、边界场景、异常场景。"""
import pytest

from rag.indexing.chunker import chunk_text, count_words


class TestChunkTextNormal:
    def test_short_text_returns_single_chunk(self):
        text = "这是一段很短的文本。"
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert chunks == [text]

    def test_long_text_splits_into_multiple_chunks(self):
        text = "第一段内容。" * 50 + "\n\n" + "第二段内容。" * 50
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 100 + 20  # 允许重叠带来的长度上浮

    def test_no_overlap(self):
        text = "句子一。句子二。句子三。句子四。句子五。" * 20
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=0)
        assert len(chunks) > 1
        # 无重叠时，拼接所有块应能大致还原原文内容（去除分隔标点后长度不减少太多）
        joined_len = sum(len(c) for c in chunks)
        assert joined_len <= len(text) + 5

    def test_overlap_creates_shared_content(self):
        text = "ABCDEFGHIJ" * 20
        chunks = chunk_text(text, chunk_size=30, chunk_overlap=10)
        assert len(chunks) > 1
        # 相邻块之间应有重叠部分
        assert chunks[0][-10:] in chunks[1] or chunks[1].startswith(chunks[0][-10:])


class TestChunkTextBoundary:
    def test_empty_text_returns_empty_list(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []
        assert chunk_text(None) == []

    def test_exact_chunk_size_boundary(self):
        text = "字" * 100
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert chunks == [text]

    def test_one_char_over_chunk_size(self):
        text = "字" * 101
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
        assert len(chunks) >= 2

    def test_single_atom_longer_than_chunk_size_hard_split(self):
        """无标点的长字符串（如长英文单词/哈希串）应能被硬切，不会无限递归或报错。"""
        text = "a" * 500
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=0)
        assert len(chunks) >= 10
        assert all(len(c) <= 50 for c in chunks)

    def test_chunk_size_one(self):
        text = "abcdef"
        chunks = chunk_text(text, chunk_size=1, chunk_overlap=0)
        assert "".join(chunks) == text


class TestChunkTextException:
    def test_invalid_chunk_size_zero(self):
        with pytest.raises(ValueError):
            chunk_text("文本", chunk_size=0)

    def test_invalid_chunk_size_negative(self):
        with pytest.raises(ValueError):
            chunk_text("文本", chunk_size=-10)

    def test_overlap_equal_to_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_text("文本", chunk_size=10, chunk_overlap=10)

    def test_overlap_greater_than_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_text("文本", chunk_size=10, chunk_overlap=20)

    def test_negative_overlap(self):
        with pytest.raises(ValueError):
            chunk_text("文本", chunk_size=10, chunk_overlap=-1)


class TestCountWords:
    def test_zh_char_counts_characters(self):
        assert count_words("你好世界", zh_char=True) == 4

    def test_zh_char_ignores_whitespace(self):
        assert count_words("你 好 世 界", zh_char=True) == 4

    def test_non_zh_counts_by_space(self):
        assert count_words("hello world foo", zh_char=False) == 3

    def test_empty_text(self):
        assert count_words("", zh_char=True) == 0
        assert count_words(None, zh_char=True) == 0
