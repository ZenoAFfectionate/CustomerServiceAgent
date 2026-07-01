# -*- coding: utf-8 -*-
"""
jieba_util 模块单元测试。

覆盖核心函数：split_sentences、is_chinese_word、extract_phrases_by_frequency、
filter_keep_longest_only、filter_by_freq_ratio、save_to_jieba_dict。

运行方式：
    PYTHONPATH=src pytest tests/test_jieba_util.py -v
"""

import os
import tempfile
from collections import Counter

import pytest

from utils.jieba_util import (
    split_sentences,
    is_chinese_word,
    extract_phrases_by_frequency,
    filter_keep_longest_only,
    filter_by_freq_ratio,
    save_to_jieba_dict,
)


# ======================== split_sentences ========================

class TestSplitSentences:
    """测试中文断句"""

    def test_split_by_period(self):
        result = split_sentences("第一句。第二句。")
        # re.split 在末尾分隔符后会产生空字符串
        non_empty = [s for s in result if s.strip()]
        assert len(non_empty) == 2
        assert "第一句" in non_empty[0]

    def test_split_by_exclamation(self):
        result = split_sentences("你好！世界！")
        non_empty = [s for s in result if s.strip()]
        assert len(non_empty) == 2

    def test_split_by_comma(self):
        result = split_sentences("A，B")
        assert len(result) == 2

    def test_split_by_newline(self):
        result = split_sentences("行1\n行2\n行3")
        assert len(result) >= 2

    def test_no_delimiter(self):
        result = split_sentences("没有标点的文本")
        assert len(result) == 1


# ======================== is_chinese_word ========================

class TestIsChineseWord:
    """测试中文词判断"""

    def test_pure_chinese(self):
        assert is_chinese_word("你好") is True

    def test_single_char(self):
        assert is_chinese_word("中") is True

    def test_mixed(self):
        assert is_chinese_word("你a") is False

    def test_english(self):
        assert is_chinese_word("hello") is False

    def test_empty(self):
        assert is_chinese_word("") is False


# ======================== extract_phrases_by_frequency ========================

class TestExtractPhrasesByFrequency:
    """测试高频短语提取"""

    def test_returns_list_of_tuples(self):
        texts = ["这是一个测试文本"]
        result = extract_phrases_by_frequency(texts, ngram_range=(2, 3), top_k=5)
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 2

    def test_finds_repeated_phrases(self):
        texts = ["千川运营", "千川运营"]
        result = extract_phrases_by_frequency(texts, ngram_range=(2, 2), top_k=10)
        phrases = [p for p, _ in result]
        assert "千川" in phrases
        # 千川出现 4 次（每个文本中出现 2 次）
        freq_dict = dict(result)
        assert freq_dict["千川"] >= 2

    def test_empty_texts(self):
        result = extract_phrases_by_frequency([], ngram_range=(2, 3), top_k=5)
        assert result == []


# ======================== filter_keep_longest_only ========================

class TestFilterKeepLongestOnly:
    """测试最长不重叠过滤"""

    def test_removes_substring_of_longer(self):
        phrases = [("千川", 5), ("巨量千川", 3)]
        result = filter_keep_longest_only(phrases)
        words = [p for p, _ in result]
        assert "巨量千川" in words
        assert "千川" not in words

    def test_keeps_independent_words(self):
        phrases = [("苹果", 3), ("香蕉", 2)]
        result = filter_keep_longest_only(phrases)
        assert len(result) == 2


# ======================== filter_by_freq_ratio ========================

class TestFilterByFreqRatio:
    """测试频率比过滤"""

    def test_filters_redundant_short_word(self):
        phrases = [("千川", 10), ("巨量千川", 9)]
        result = filter_by_freq_ratio(phrases, threshold=0.8)
        words = [p for p, _ in result]
        # "千川" 是 "巨量千川" 的子串且频率比 >= 0.8，应被过滤
        assert "巨量千川" in words

    def test_keeps_independent_words(self):
        phrases = [("苹果", 10), ("香蕉", 5)]
        result = filter_by_freq_ratio(phrases, threshold=0.8)
        assert len(result) == 2


# ======================== save_to_jieba_dict ========================

class TestSaveToJiebaDict:
    """测试词典保存"""

    def test_saves_correct_format(self):
        phrases = [("千川", 100), ("运营", 50)]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            output_path = f.name

        try:
            save_to_jieba_dict(phrases, output_path=output_path)
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            assert len(lines) == 2
            # 格式: 词语 频率 词性
            parts = lines[0].strip().split()
            assert parts[0] == "千川"
            assert parts[1] == "100"
            assert parts[2] == "n"
        finally:
            os.unlink(output_path)

    def test_uses_default_freq_for_zero(self):
        phrases = [("测试词", 0)]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            output_path = f.name

        try:
            save_to_jieba_dict(phrases, output_path=output_path, default_freq=9999)
            with open(output_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            parts = lines[0].strip().split()
            assert parts[1] == "9999"
        finally:
            os.unlink(output_path)
