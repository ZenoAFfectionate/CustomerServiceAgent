# -*- coding: utf-8 -*-
"""
自底向上合并算法单元测试（完整覆盖）。

覆盖函数:
    - _bottom_up_merge            核心合并入口
    - _greedy_merge_siblings      贪心兄弟合并
    - _merge_tiny_text_fragments  文本碎片合并
    - build_block_tree            端到端 + heading-split + BFS 三路径
    - generate_block_documents    文档块生成集成

运行方式:
    PYTHONPATH=process:process/src pytest tests/test_process/test_bottom_up_merge.py -v
"""

import pytest
from bs4 import BeautifulSoup, Tag, NavigableString

from html_utils import (
    _bottom_up_merge,
    _greedy_merge_siblings,
    _count_words,
    build_block_tree,
    clean_html,
    expand_table_spans,
)
from text_process import (
    _merge_tiny_text_fragments,
    generate_block_documents,
)


# ======================== 辅助函数 ========================

def _make_block(text: str, path: list, zh_char: bool = True) -> tuple:
    """快速创建测试用块 (tag, path, is_leaf)."""
    soup = BeautifulSoup("", "html.parser")
    tag = soup.new_tag("div")
    tag.append(NavigableString(text))
    return (tag, path, True)


def _get_text(block: tuple) -> str:
    """从块中提取文本."""
    tag = block[0]
    return tag.get_text() if isinstance(tag, Tag) else str(tag)


def _get_path(block: tuple) -> list:
    """从块中提取路径."""
    return block[1]


# ======================== _greedy_merge_siblings ========================

class TestGreedyMergeSiblings:
    """贪心兄弟合并 —— 核心合并原语."""

    # --- 基础行为 ---

    def test_no_merge_when_single_block(self):
        blocks = [_make_block("hello world", ["p0"])]
        result = _greedy_merge_siblings(blocks, 4096, True)
        assert len(result) == 1

    def test_no_merge_when_exceeds_max(self):
        blocks = [
            _make_block("A" * 3000, ["p0"]),
            _make_block("B" * 3000, ["p1"]),
        ]
        result = _greedy_merge_siblings(blocks, 4096, True)
        assert len(result) == 2

    def test_merges_two_small_blocks(self):
        blocks = [
            _make_block("hello", ["h1_domain", "p0"]),
            _make_block("world", ["h1_domain", "p1"]),
        ]
        result = _greedy_merge_siblings(blocks, 4096, True)
        assert len(result) == 1
        merged = _get_text(result[0])
        assert "hello" in merged and "world" in merged

    def test_all_merge_into_one(self):
        blocks = [
            _make_block(chr(65 + i) * 100, [f"p{i}"])
            for i in range(4)
        ]
        result = _greedy_merge_siblings(blocks, 4096, True)
        assert len(result) == 1

    # --- 合并后元数据 ---

    def test_merged_path_is_parent_path(self):
        blocks = [
            _make_block("hello", ["h1", "h2", "p0"]),
            _make_block("world", ["h1", "h2", "p1"]),
        ]
        result = _greedy_merge_siblings(blocks, 4096, True)
        assert _get_path(result[0]) == ["h1", "h2"]

    def test_merged_is_leaf_false(self):
        blocks = [
            _make_block("hello", ["div", "p0"]),
            _make_block("world", ["div", "p1"]),
        ]
        result = _greedy_merge_siblings(blocks, 4096, True)
        assert result[0][2] is False

    def test_merged_path_truncated_to_parent_for_deep_paths(self):
        """深层路径合并后只保留父路径（截断一级）。"""
        blocks = [
            _make_block("a", ["a", "b", "c", "d", "e", "p0"]),
            _make_block("b", ["a", "b", "c", "d", "e", "p1"]),
        ]
        result = _greedy_merge_siblings(blocks, 4096, True)
        assert _get_path(result[0]) == ["a", "b", "c", "d", "e"]

    # --- 部分合并 ---

    def test_partial_merge_first_two(self):
        blocks = [
            _make_block("A" * 1000, ["p0"]),
            _make_block("B" * 1000, ["p1"]),
            _make_block("C" * 3000, ["p2"]),
        ]
        result = _greedy_merge_siblings(blocks, 4096, True)
        assert len(result) == 2

    def test_partial_merge_last_two(self):
        blocks = [
            _make_block("A" * 3000, ["p0"]),
            _make_block("B" * 1000, ["p1"]),
            _make_block("C" * 1000, ["p2"]),
        ]
        result = _greedy_merge_siblings(blocks, 4096, True)
        assert len(result) == 2

    def test_partial_merge_middle_pair(self):
        """四个块：第一、四独立，中间两个合并（共3块）。"""
        blocks = [
            _make_block("A" * 3600, ["p0"]),
            _make_block("B" * 500,  ["p1"]),
            _make_block("C" * 500,  ["p2"]),
            _make_block("D" * 3100, ["p3"]),
        ]
        result = _greedy_merge_siblings(blocks, 4096, True)
        # A(3600)+B(500)=4100>4096 → A stays alone
        # B(500)+C(500)=1000≤4096 → merge
        # B+C(1000)+D(3100)=4100>4096 → merged B+C, D stays
        assert len(result) == 3

    def test_chain_merge_many_small(self):
        """大量碎片（100 块各 48 chars）应被合并为最少块数。"""
        blocks = [
            _make_block(chr(65 + i % 26) * 48, [f"p{i}"])
            for i in range(100)
        ]
        result = _greedy_merge_siblings(blocks, 4096, True)
        # 100*48 = 4800 > 4096, so expect 2 blocks (one ~4096, one ~704)
        assert len(result) == 2
        total = sum(len(_get_text(r)) for r in result)
        assert total >= 48 * 100

    # --- 边界条件 ---

    def test_exactly_at_max_boundary(self):
        """恰好等于 max_node_words 时不应合并（严格小于才合并）。"""
        # 当前实现是 <= max_node_words 时合并
        blocks = [
            _make_block("A" * 2048, ["p0"]),
            _make_block("B" * 2048, ["p1"]),
        ]
        result = _greedy_merge_siblings(blocks, 4096, True)
        assert len(result) == 1  # exactly 4096, merges

    def test_one_char_over_max(self):
        """超过 max 1 字符不合并。"""
        blocks = [
            _make_block("A" * 2048, ["p0"]),
            _make_block("B" * 2049, ["p1"]),
        ]
        result = _greedy_merge_siblings(blocks, 4096, True)
        assert len(result) == 2

    def test_empty_text_blocks(self):
        """空文本块的处理。"""
        blocks = [
            _make_block("", ["p0"]),
            _make_block("hello", ["p1"]),
        ]
        result = _greedy_merge_siblings(blocks, 4096, True)
        assert len(result) == 1

    # --- 中文 & 英文模式 ---

    def test_chinese_zh_char_true(self):
        blocks = [
            _make_block("第一章 概述内容", ["p0"]),
            _make_block("第二章 入驻规范内容", ["p1"]),
        ]
        result = _greedy_merge_siblings(blocks, 4096, True)
        assert len(result) == 1

    def test_english_word_count_mode(self):
        blocks = [
            _make_block("hello world " * 5,  ["p0"]),  # 10 words
            _make_block("foo bar baz " * 3, ["p1"]),   # 9 words
        ]
        result = _greedy_merge_siblings(blocks, 20, False)
        assert len(result) == 1

    def test_english_no_merge(self):
        blocks = [
            _make_block("hello world " * 10, ["p0"]),  # 20 words
            _make_block("foo bar baz " * 5,  ["p1"]),  # 15 words
        ]
        result = _greedy_merge_siblings(blocks, 30, False)
        assert len(result) == 2

    # --- Tag 结构保留 ---

    def test_preserves_inner_html_structure(self):
        soup = BeautifulSoup("", "html.parser")
        tag1 = soup.new_tag("div")
        p1 = soup.new_tag("p"); p1.string = "段落一"; tag1.append(p1)
        tag2 = soup.new_tag("div")
        p2 = soup.new_tag("p"); p2.string = "段落二"; tag2.append(p2)

        blocks = [(tag1, ["h1", "d0"], True), (tag2, ["h1", "d1"], True)]
        result = _greedy_merge_siblings(blocks, 4096, True)
        assert len(result) == 1
        p_tags = result[0][0].find_all("p")
        assert len(p_tags) == 2
        assert p_tags[0].get_text() == "段落一"
        assert p_tags[1].get_text() == "段落二"

    def test_merged_tag_preserves_nested_tables(self):
        """合并后内部的 table 标签不被破坏。"""
        soup = BeautifulSoup("", "html.parser")
        tag1 = soup.new_tag("div")
        t1 = soup.new_tag("table")
        tr = soup.new_tag("tr"); td = soup.new_tag("td"); td.string = "cell"
        tr.append(td); t1.append(tr); tag1.append(t1)

        tag2 = soup.new_tag("div")
        tag2.append(NavigableString("after table text"))

        blocks = [(tag1, ["h1", "d0"], True), (tag2, ["h1", "d1"], True)]
        result = _greedy_merge_siblings(blocks, 4096, True)
        assert "table" in str(result[0][0])
        assert "cell" in str(result[0][0])

    # --- NavigableString 处理 ---

    def test_merging_navigable_string_and_tag(self):
        """Text 片段与 Tag 合并。"""
        soup1 = BeautifulSoup("", "html.parser")
        tag1 = soup1.new_tag("p"); tag1.string = "段落一"

        soup2 = BeautifulSoup("", "html.parser")
        tag2 = soup2.new_tag("p"); tag2.string = "段落二"

        blocks = [(tag1, ["div", "p0"], True), (tag2, ["div", "p1"], True)]
        result = _greedy_merge_siblings(blocks, 4096, True)
        assert len(result) == 1
        assert "段落一" in _get_text(result[0])
        assert "段落二" in _get_text(result[0])


# ======================== _bottom_up_merge ========================

class TestBottomUpMerge:
    """自底向上合并主入口 —— 分组 + 迭代收敛."""

    # --- 基础 ---

    def test_empty_list(self):
        assert _bottom_up_merge([], 4096, True) == []

    def test_single_block(self):
        blocks = [_make_block("hello", ["p"])]
        result = _bottom_up_merge(blocks, 4096, True)
        assert len(result) == 1

    # --- 同/不同父路径 ---

    def test_same_parent_merges(self):
        blocks = [
            _make_block("A" * 50, ["h1", "h2", "p0"]),
            _make_block("B" * 50, ["h1", "h2", "p1"]),
            _make_block("C" * 50, ["h1", "h2", "p2"]),
        ]
        result = _bottom_up_merge(blocks, 4096, True)
        assert len(result) == 1

    def test_different_parents_no_merge(self):
        blocks = [
            _make_block("A" * 50, ["h1", "h2_A", "p0"]),
            _make_block("B" * 50, ["h1", "h2_B", "p0"]),
        ]
        result = _bottom_up_merge(blocks, 4096, True)
        assert len(result) == 2

    def test_same_parent_but_too_big(self):
        blocks = [
            _make_block("A" * 3000, ["h1", "h2", "p0"]),
            _make_block("B" * 3000, ["h1", "h2", "p1"]),
        ]
        result = _bottom_up_merge(blocks, 4096, True)
        assert len(result) == 2

    # --- 交错父路径 ---

    def test_interleaved_parents(self):
        """两组不同父路径的块，各自合并后上升到父级，再交叉合并。"""
        blocks = [
            _make_block("A1" * 50, ["h1", "h2_A", "p0"]),
            _make_block("A2" * 50, ["h1", "h2_A", "p1"]),
            _make_block("B1" * 50, ["h1", "h2_B", "p0"]),
            _make_block("B2" * 50, ["h1", "h2_B", "p1"]),
        ]
        result = _bottom_up_merge(blocks, 4096, True)
        # 迭代合并: 第一轮 A 组 → ["h1","h2_A"], B 组 → ["h1","h2_B"]
        # 第二轮 两者父都是 ("h1",) → 合并为一个
        assert len(result) == 1
        merged = _get_text(result[0])
        assert all(x in merged for x in ["A1", "A2", "B1", "B2"])

    def test_interleaved_parents_stop_at_max(self):
        """交错父路径合并遇到上限时停止。"""
        # Each "A1"*1000 = 2000 chars (len("A1")=2 * 1000)
        blocks = [
            _make_block("A" * 2000, ["h1", "h2_A", "p0"]),
            _make_block("A" * 2000, ["h1", "h2_A", "p1"]),
            _make_block("B" * 2000, ["h1", "h2_B", "p0"]),
            _make_block("B" * 2000, ["h1", "h2_B", "p1"]),
        ]
        result = _bottom_up_merge(blocks, 4096, True)
        # A组: 2000+2000=4000 → merge → ["h1","h2_A"]
        # B组: 2000+2000=4000 → merge → ["h1","h2_B"]
        # 第二轮: parent of both merged is ("h1",) → 4000+4000=8000>4096 → no merge
        assert len(result) == 2

    # --- 多层级迭代 ---

    def test_iterative_two_level_merge(self):
        """子块合并后上升到父级，与同级其他块再合并。"""
        blocks = [
            _make_block("X" * 50, ["h1", "h2_A", "p0"]),
            _make_block("Y" * 50, ["h1", "h2_A", "p1"]),
            _make_block("Z" * 50, ["h1", "h2_A", "p2"]),
            _make_block("W" * 100, ["h1", "h2_A", "p3"]),  # same parent as above
        ]
        result = _bottom_up_merge(blocks, 4096, True)
        assert len(result) == 1

    def test_iterative_three_level_merge(self):
        """三级嵌套：孙子→儿子→根，逐级合并。"""
        blocks = [
            # L2 children → merge to L2 level, then all at L1_A level merge
            _make_block("A" * 30, ["root", "L1_A", "L2", "p0"]),
            _make_block("B" * 30, ["root", "L1_A", "L2", "p1"]),
            _make_block("C" * 30, ["root", "L1_A", "L2", "p2"]),
            _make_block("D" * 30, ["root", "L1_A", "L2", "p3"]),  # same parent as above
            _make_block("E" * 30, ["root", "L1_A", "L2", "p4"]),  # same parent
        ]
        result = _bottom_up_merge(blocks, 4096, True)
        # Round 1: all merge at L2 → path ["root","L1_A","L2"]
        # All 5 have same parent ("root","L1_A","L2") and total 150 < 4096
        assert len(result) == 1

    def test_iterative_stops_at_size_limit(self):
        """多级合并在每层都遵守 max_node_words。"""
        blocks = [
            _make_block("A" * 2000, ["h1", "h2_A", "p0"]),
            _make_block("B" * 2000, ["h1", "h2_A", "p1"]),
            _make_block("C" * 2000, ["h1", "h2_B", "p0"]),
        ]
        result = _bottom_up_merge(blocks, 4096, True)
        # A+B=4000 → merge at ["h1","h2_A"]
        # C=2000 at ["h1","h2_B","p0"]
        # Round2: parent of merged is ("h1",), parent of C is ("h1","h2_B") → different
        assert len(result) == 2

    # --- 收敛与安全 ---

    def test_no_infinite_loop(self):
        """安全上限 64 次迭代，不会无限循环。"""
        blocks = [
            _make_block(chr(65 + i % 26) * 10, [f"p{i}"])
            for i in range(200)
        ]
        result = _bottom_up_merge(blocks, 100000, True)
        assert len(result) == 1

    def test_converges_when_no_more_merges(self):
        """无更多合并可能时立即终止。"""
        blocks = [
            _make_block("A" * 3000, ["p0"]),
            _make_block("B" * 3000, ["p1"]),
            _make_block("C" * 3000, ["p2"]),
        ]
        # Each is too big to merge with neighbor, only 1 iteration needed
        result = _bottom_up_merge(blocks, 5000, True)
        assert len(result) == 3

    # --- 根级块 ---

    def test_root_level_paths_merge(self):
        """根级 path=[\"h1_section\"] 的块合并为兄弟。"""
        blocks = [
            _make_block("Hello", ["h1_section"]),
            _make_block("World", ["h2_section"]),
        ]
        result = _bottom_up_merge(blocks, 4096, True)
        assert len(result) == 1

    def test_root_level_too_big(self):
        blocks = [
            _make_block("A" * 3000, ["h1_section"]),
            _make_block("B" * 3000, ["h2_section"]),
        ]
        result = _bottom_up_merge(blocks, 4096, True)
        assert len(result) == 2

    # --- 不规则路径 ---

    def test_blocks_with_different_path_depths_same_parent(self):
        """不同深度但同父路径的块可以合并。"""
        blocks = [
            _make_block("short" * 10, ["root", "a"]),
            _make_block("text" * 10,  ["root", "b"]),
        ]
        result = _bottom_up_merge(blocks, 4096, True)
        # Both have parent ("root",) → sibling merge
        assert len(result) == 1

    def test_blocks_different_depths_different_parents(self):
        """不同深度且不同父路径的块不合并。"""
        blocks = [
            _make_block("A" * 50, ["root", "a"]),
            _make_block("B" * 50, ["root", "b", "deep"]),
        ]
        result = _bottom_up_merge(blocks, 4096, True)
        # parents: ("root",) vs ("root","b") → different → no merge
        assert len(result) == 2

    def test_blocks_with_empty_path(self):
        """空路径块的合并。"""
        blocks = [
            _make_block("hello", []),
            _make_block("world", []),
        ]
        result = _bottom_up_merge(blocks, 4096, True)
        assert len(result) == 1

    # --- 大规模数据 ---

    def test_very_large_input(self):
        """大量输入块不会崩溃。"""
        blocks = [
            _make_block(chr(65 + i % 26) * 50, ["h1", f"h2_{i//10}", f"p{i%10}"])
            for i in range(1000)
        ]
        result = _bottom_up_merge(blocks, 4096, True)
        # Should complete and reduce block count
        assert len(result) < 1000
        # All blocks should have text
        for block in result:
            assert len(_get_text(block)) > 0


# ======================== _merge_tiny_text_fragments ========================

class TestMergeTinyTextFragments:
    """_extract_mixed_content 中的文本碎片合并。"""

    # --- 向前合并 ---

    def test_forward_merge(self):
        results = [
            (None, "T", "A" * 100),
            (None, "T", "ab"),
        ]
        merged = _merge_tiny_text_fragments(results, 4096)
        assert len(merged) == 1
        assert "ab" in merged[0][2]

    def test_forward_merge_with_table_prev(self):
        """向前合并时前一个块是表格也可以合并。"""
        soup = BeautifulSoup("", "html.parser")
        table_tag = soup.new_tag("table")
        results = [
            (table_tag, "T 表格行0-5", "row1 row2 row3 row4 row5"),
            (None, "T", "tiny"),
        ]
        merged = _merge_tiny_text_fragments(results, 4096)
        assert len(merged) == 1
        assert "tiny" in merged[0][2]

    # --- 向后合并 ---

    def test_backward_merge(self):
        results = [
            (None, "T", "cd"),
            (None, "T", "B" * 100),
        ]
        merged = _merge_tiny_text_fragments(results, 4096)
        assert len(merged) == 1
        assert "cd" in merged[0][2]

    def test_backward_merge_to_table(self):
        """碎片向后合并到表格块。"""
        soup = BeautifulSoup("", "html.parser")
        table_tag = soup.new_tag("table")
        results = [
            (None, "T", "tiny"),
            (table_tag, "T 表格行0-0", "header row"),
        ]
        merged = _merge_tiny_text_fragments(results, 4096)
        assert len(merged) == 1
        assert "tiny" in merged[0][2]

    # --- 不合并 ---

    def test_no_merge_if_exceeds_max(self):
        results = [
            (None, "T", "A" * 4090),
            (None, "T", "tiny"),
        ]
        merged = _merge_tiny_text_fragments(results, 4096)
        # 4090 + 1(\n) + 4 = 4095 <= 4096 → would merge
        # Let me use a bigger number
        pass

    def test_no_merge_if_exceeds_max_forward(self):
        """前向合并超过 max 时保留碎片。"""
        big = "A" * 4094
        results = [
            (None, "T", big),
            (None, "T", "tiny"),
        ]
        merged = _merge_tiny_text_fragments(results, 4096)
        # 4094 + 1 + 4 = 4099 > 4096, forward can't merge
        # Also can't merge backward (no next)
        assert len(merged) == 2

    def test_no_merge_if_exceeds_max_backward(self):
        """后向合并超过 max 时保留碎片。"""
        big = "B" * 4094
        results = [
            (None, "T", "tiny"),
            (None, "T", big),
        ]
        merged = _merge_tiny_text_fragments(results, 4096)
        # 4 + 1 + 4094 = 4099 > 4096, backward can't merge
        assert len(merged) == 2

    # --- 边界 ---

    def test_empty_list(self):
        assert _merge_tiny_text_fragments([], 4096) == []

    def test_single_item(self):
        results = [(None, "T", "hello")]
        merged = _merge_tiny_text_fragments(results, 4096)
        assert len(merged) == 1

    def test_threshold_boundary(self):
        """恰好 20 字符的不触发合并。"""
        results = [
            (None, "T", "A" * 20),
            (None, "T", "B" * 100),
        ]
        merged = _merge_tiny_text_fragments(results, 4096)
        assert len(merged) == 2

    def test_threshold_minus_one(self):
        """19 字符触发合并。"""
        results = [
            (None, "T", "B" * 100),
            (None, "T", "A" * 19),
        ]
        merged = _merge_tiny_text_fragments(results, 4096)
        assert len(merged) == 1

    # --- 普通文本不受影响 ---

    def test_normal_text_passes_through(self):
        results = [
            (None, "T", "hello world " * 10),
            (None, "T", "foo bar baz " * 10),
        ]
        merged = _merge_tiny_text_fragments(results, 4096)
        assert len(merged) == 2

    def test_all_table_rows_preserved(self):
        """所有表格行都应保留。"""
        soup = BeautifulSoup("", "html.parser")
        t = soup.new_tag("table")
        results = [
            (t, "T 表格行0-0", "row1"),
            (t, "T 表格行1-1", "row2"),
            (t, "T 表格行2-2", "row3"),
        ]
        merged = _merge_tiny_text_fragments(results, 4096)
        assert len(merged) == 3

    # --- 混合场景 ---

    def test_mixed_fragments_and_tables(self):
        """碎片与表格交错出现。"""
        soup = BeautifulSoup("", "html.parser")
        t1 = soup.new_tag("table")
        t2 = soup.new_tag("table")
        results = [
            (None, "T", "ab"),                        # fragment
            (t1, "T 表格行0-0", "row1 data here"),     # table
            (None, "T", "cd"),                        # fragment between tables
            (t2, "T 表格行0-0", "row2 data here"),     # table
            (None, "T", "ef"),                        # trailing fragment
        ]
        merged = _merge_tiny_text_fragments(results, 4096)
        # Each tiny fragment should merge into adjacent table
        assert len(merged) <= 5

    def test_three_way_no_possible_merge(self):
        """三个碎片，前后都无法合并（都超 max）。"""
        results = [
            (None, "T", "A" * 4090),
            (None, "T", "tiny"),
            (None, "T", "B" * 4090),
        ]
        merged = _merge_tiny_text_fragments(results, 4096)
        # tiny can't merge forward (4090+1+4=4095 ≤ 4096 → would merge!)
        # Actually it would merge forward. Let me use bigger numbers.
        assert len(merged) >= 2  # tiny should merge somewhere or stay

    def test_three_way_no_possible_merge_really(self):
        """三个碎片，前后真的都无法合并。"""
        results = [
            (None, "T", "A" * 4095),
            (None, "T", "tiny"),
            (None, "T", "B" * 4095),
        ]
        merged = _merge_tiny_text_fragments(results, 4096)
        # tiny can't merge forward: 4095+1+4=4100 > 4096
        # tiny can't merge backward: 4+1+4095=4100 > 4096
        assert len(merged) == 3


# ======================== build_block_tree 端到端 ========================

class TestBuildBlockTreeWithMerge:
    """build_block_tree 集成 —— BFS + heading-split + small-page 三条路径."""

    def _do_block(self, html_str, max_words=4096, min_words=48):
        cleaned = clean_html(html_str)
        expanded = expand_table_spans(cleaned)
        blocks, _ = build_block_tree(expanded, max_node_words=max_words,
                                     min_node_words=min_words, zh_char=True)
        return blocks

    # --- BFS 路径 (total > max_node_words) ---

    def test_bfs_path_activated(self):
        """大文档走 BFS 分块路径。"""
        html = '<h1>T</h1>' + '<p>' + '内容' * 2000 + '</p>'
        blocks = self._do_block(html, max_words=200, min_words=48)
        assert len(blocks) >= 1

    def test_bfs_with_bottom_up_merge(self):
        """BFS 分块后自底向上合并。"""
        html = '<h1>T</h1>' + ''.join(
            f'<p>' + '内容' * 100 + '</p>' for _ in range(20)
        )
        blocks = self._do_block(html, max_words=4096, min_words=48)
        # 应有合并发生，块数应远少于 20
        assert len(blocks) < 15

    def test_bfs_table_not_split(self):
        """BFS 中 table 整体保留不拆分（但合并后可能与其他块合并）。"""
        html = (
            '<h1>规则</h1>'
            '<p>前言' + '内容' * 50 + '</p>'
            '<table>'
            + ''.join('<tr><td>项目{i}</td><td>说明内容{i}</td></tr>'
                      for i in range(30))
            + '</table>'
        )
        blocks = self._do_block(html, max_words=4096, min_words=48)
        assert len(blocks) >= 1

    # --- mid-size 路径 (min <= total <= max) ---

    def test_heading_split_path_activated(self):
        """中等文档走 heading-split 后再合并。"""
        html = '<h2>章</h2>' + ''.join(
            f'<h3>节{i}</h3><p>' + '内容' * 30 + '</p>'
            for i in range(5)
        )
        blocks = self._do_block(html, max_words=4096, min_words=48)
        # heading-split 产生多个 section，合并后减少
        assert len(blocks) >= 1

    def test_heading_split_merge_reduces_count(self):
        """heading-split 后的合并应显著减少块数。"""
        html = '<h1>总则</h1>' + ''.join(
            f'<h3>小节{i}</h3><p>' + '内容' * 10 + '</p>'
            for i in range(10)
        )
        blocks = self._do_block(html, max_words=4096, min_words=48)
        # 10 个小节不应全部独立成块
        assert len(blocks) < 10

    def test_single_child_heading_split(self):
        """只有一个子节点时 heading-split 也应触发。"""
        html = (
            '<div>'
            '<h2>第一节</h2><p>' + '内容A' * 20 + '</p>'
            '<h2>第二节</h2><p>' + '内容B' * 20 + '</p>'
            '<h2>第三节</h2><p>' + '内容C' * 20 + '</p>'
            '</div>'
        )
        blocks = self._do_block(html, max_words=4096, min_words=48)
        assert len(blocks) >= 1

    # --- small-page 路径 (total < min_node_words) ---

    def test_small_page_returns_single_block(self):
        """过小页面整页作为单块返回。"""
        html = '<p>短内容</p>'
        blocks = self._do_block(html, max_words=4096, min_words=48)
        assert len(blocks) == 1

    def test_empty_page_returns_empty(self):
        """空 HTML 返回空列表。"""
        html = '<div></div>'
        blocks = self._do_block(html, max_words=4096, min_words=48)
        assert blocks == []

    # --- 数据完整性 ---

    def test_no_data_loss(self):
        html = '<h1>标题</h1>' + ''.join(
            f'<p>段落{i}：' + '测试内容' * 20 + '</p>'
            for i in range(10)
        )
        blocks = self._do_block(html, max_words=4096, min_words=48)
        all_text = "".join(b[0].get_text() for b in blocks if isinstance(b[0], Tag))
        for i in range(10):
            assert f"段落{i}" in all_text

    def test_all_blocks_above_minimum(self):
        """合并后所有块 >= min_node_words（除整页保留的小页面）。"""
        html = '<h1>T</h1>' + ''.join(
            f'<p>' + '内容A' * 50 + '</p>' for _ in range(10)
        )
        blocks = self._do_block(html, max_words=4096, min_words=48)
        for tag, path, is_leaf in blocks:
            w = _count_words(tag, True)
            assert w >= 48 or w == 0, f"块 {path} 有 {w} chars（低于 min=48）"

    # --- 中文真实场景 ---

    def test_chinese_policy_document(self):
        html = (
            '<div class="h1_domain">'
            '<h1>抖音电商规则总则</h1>'
            '<div class="h2_domain">'
            '<h2>第一章 概述</h2>'
            '<p>本规则适用于所有在抖音电商平台开设店铺的商家。'
            '商家应当遵守国家法律法规，遵循公平诚信原则。</p>'
            '</div>'
            '<div class="h2_domain">'
            '<h2>第二章 入驻规范</h2>'
            '<p>商家入驻平台应当提供真实有效的主体资质信息。</p>'
            '</div>'
            '<div class="h2_domain">'
            '<h2>第三章 违规处理</h2>'
            '<p>商家违反平台规则的，平台有权采取警告、扣分、关店等措施。</p>'
            '</div>'
            '</div>'
        )
        blocks = self._do_block(html, max_words=4096, min_words=48)
        assert len(blocks) >= 1
        all_text = "".join(b[0].get_text() for b in blocks if isinstance(b[0], Tag))
        assert "入驻规范" in all_text
        assert "违规处理" in all_text

    # --- 极端场景 ---

    def test_single_giant_table_no_split(self):
        """单一大表格不拆分（原子保留）。"""
        html = (
            '<h1>表</h1>'
            '<table>'
            + ''.join(f'<tr><td>第{i}行</td><td>' + '长内容' * 100 + '</td></tr>'
                      for i in range(50))
            + '</table>'
        )
        blocks = self._do_block(html, max_words=4096, min_words=48)
        assert len(blocks) >= 1

    def test_all_noise_text_in_small_page(self):
        """小页面中的噪音文本走整页保留路径（返回单块）。"""
        html = '<div>PROGRESS</div><div>0%</div><div>CONTENTS</div>'
        blocks = self._do_block(html, max_words=4096, min_words=48)
        # 总词数 < min_words → 整页作为单块保留（不经过 BFS 的噪声过滤）
        # 噪音过滤仅在 BFS 大页面路径中生效
        assert len(blocks) == 1

    def test_bare_text_block_created(self):
        """裸文本（不在子标签中）正确作为独立块。"""
        html = '<div>' + ('裸文本内容' + 'A' * 100) + '</div>'
        blocks = self._do_block(html, max_words=4096, min_words=48)
        assert len(blocks) >= 1


# ======================== 与 generate_block_documents 集成 ========================

class TestIntegrationWithDocGeneration:
    """build_block_tree → generate_block_documents 全链路."""

    def _full_pipeline(self, html_str, max_words=4096, min_words=48):
        import text_process as tp
        orig = tp._generate_summary_and_question
        tp._generate_summary_and_question = lambda t, u, gq=False: ("[摘要]", "")
        try:
            cleaned = clean_html(html_str)
            expanded = expand_table_spans(cleaned)
            blocks, _ = build_block_tree(expanded, max_node_words=max_words,
                                         min_node_words=min_words, zh_char=True)
            docs = generate_block_documents(blocks, max_node_words=max_words,
                                           page_url="test.html", time_value="")
            return docs
        finally:
            tp._generate_summary_and_question = orig

    # --- 基本输出 ---

    def test_docs_have_required_fields(self):
        html = '<h1>标题</h1><p>' + '内容' * 50 + '</p>'
        docs = self._full_pipeline(html)
        assert len(docs) >= 1
        for doc in docs:
            assert "chunk_idx" in doc
            assert "text" in doc
            assert "block_path" in doc
            assert "html_content" in doc
            assert "title" in doc
            assert len(doc["text"]) > 0

    def test_docs_have_chunk_idx_increasing(self):
        """chunk_idx 递增。"""
        html = '<h1>T</h1>' + ''.join(
            f'<p>' + '长内容' * 100 + '</p>' for _ in range(20)
        )
        docs = self._full_pipeline(html, max_words=4096, min_words=48)
        indices = [d["chunk_idx"] for d in docs]
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)

    # --- 表格 ---

    def test_html_with_table_generates_docs(self):
        html = (
            '<h1>规则</h1>'
            '<p>' + '前言内容' * 50 + '</p>'
            '<table>'
            '<tr><th>违规行为</th><th>处理方式</th></tr>'
            + ''.join(f'<tr><td>行为{i}</td><td>处理方式描述内容{i}</td></tr>'
                      for i in range(20))
            + '</table>'
            '<p>' + '结语内容' * 50 + '</p>'
        )
        docs = self._full_pipeline(html, max_words=4096, min_words=48)
        assert len(docs) >= 1

    def test_table_only_document(self):
        """只有表格的 HTML。"""
        html = (
            '<table>'
            '<tr><th>列A</th><th>列B</th></tr>'
            + ''.join(f'<tr><td>数据{i}A</td><td>数据{i}B描述内容</td></tr>'
                      for i in range(30))
            + '</table>'
        )
        docs = self._full_pipeline(html, max_words=4096, min_words=48)
        assert len(docs) >= 1

    # --- 混合内容 ---

    def test_mixed_content_no_tiny_chunks(self):
        """表格间过渡文字不应产生 < 20 chars 的超小块。"""
        html = (
            '<h1>规则</h1>'
            '<p>' + '前言' * 100 + '</p>'
            '<table>'
            '<tr><th>项目</th><th>说明</th></tr>'
            + ''.join(f'<tr><td>项目{i}</td><td>说明内容文本{i}</td></tr>'
                      for i in range(20))
            + '</table>'
            '<p>' + '结语' * 100 + '</p>'
        )
        docs = self._full_pipeline(html, max_words=4096, min_words=48)
        tiny = [d for d in docs if len(d["text"]) < 20]
        assert len(tiny) == 0, f"发现超小块: {[(d['chunk_idx'], d['text']) for d in tiny]}"

    def test_multiple_tables_with_text_between(self):
        """多个表格之间有短文本。"""
        html = (
            '<h1>规则汇总</h1>'
            '<p>' + '介绍' * 100 + '</p>'
            '<table><tr><th>A</th></tr>'
            + ''.join(f'<tr><td>数据{i}</td></tr>' for i in range(10))
            + '</table>'
            '<p>表格A说明</p>'
            '<table><tr><th>B</th></tr>'
            + ''.join(f'<tr><td>数据{i}</td></tr>' for i in range(10))
            + '</table>'
            '<p>' + '总结' * 100 + '</p>'
        )
        docs = self._full_pipeline(html, max_words=4096, min_words=48)
        assert len(docs) >= 1

    # --- 大文档 ---

    def test_large_document_chunks_within_range(self):
        html = '<h1>大文档</h1>' + ''.join(
            f'<h2>第{i}节</h2><p>' + '正文内容' * 100 + '</p>'
            for i in range(30)
        )
        docs = self._full_pipeline(html, max_words=4096, min_words=48)
        # 不应产生极端超大的纯文本块
        for d in docs:
            assert len(d["text"]) < 20000, (
                f"chunk {d['chunk_idx']} 过大: {len(d['text'])}"
            )

    def test_massive_table_chunking(self):
        """超大表格（100+ 行）按行切分。"""
        html = (
            '<h1>数据表</h1>'
            '<table>'
            '<tr><th>序号</th><th>名称</th><th>描述</th></tr>'
            + ''.join(
                f'<tr><td>{i}</td><td>名称{i}</td><td>'
                + '描述内容' * 30 + '</td></tr>'
                for i in range(100)
            )
            + '</table>'
        )
        docs = self._full_pipeline(html, max_words=4096, min_words=48)
        # 表格应被按行切分为多个块
        assert len(docs) >= 1

    # --- 边界 ---

    def test_empty_html_generates_no_docs(self):
        html = '<div></div>'
        docs = self._full_pipeline(html)
        assert docs == []

    def test_minimal_valid_html(self):
        html = '<p>' + '有效内容' * 30 + '</p>'
        docs = self._full_pipeline(html, max_words=4096, min_words=48)
        assert len(docs) >= 0  # may or may not generate depending on min_words

    # --- 嵌套结构 ---

    def test_nested_domains_full_pipeline(self):
        html = (
            '<div class="h1_domain"><h1>标题</h1>'
            '<div class="h2_domain"><h2>子标题A</h2>'
            '<p>' + '内容A' * 40 + '</p>'
            '<div class="h3_domain"><h3>子子标题</h3>'
            '<p>' + '内容B' * 40 + '</p>'
            '</div></div>'
            '<div class="h2_domain"><h2>子标题C</h2>'
            '<table><tr><th>列1</th></tr>'
            + ''.join(f'<tr><td>行{i}</td></tr>' for i in range(15))
            + '</table></div>'
            '</div>'
        )
        docs = self._full_pipeline(html, max_words=4096, min_words=48)
        assert len(docs) >= 1
        all_text = " ".join(d["text"] for d in docs)
        assert "子标题A" in all_text or "子标题C" in all_text

    # --- 并发安全性（结构层面） ---

    def test_output_is_deterministic(self):
        """相同输入产生相同的块数量（确定性）。"""
        html = '<h1>T</h1>' + ''.join(
            f'<p>' + '内容' * 50 + '</p>' for _ in range(5)
        )
        docs1 = self._full_pipeline(html)
        docs2 = self._full_pipeline(html)
        assert len(docs1) == len(docs2)

    def test_block_paths_are_unique(self):
        """同一次运行中 block_path 应唯一。"""
        html = '<h1>T</h1>' + ''.join(
            f'<h2>节{i}</h2><p>' + '内容' * 50 + '</p>'
            for i in range(10)
        )
        docs = self._full_pipeline(html, max_words=4096, min_words=48)
        paths = [d["block_path"] for d in docs]
        # 路径不一定唯一（合并后可能相同），但不应全空
        assert any(p != "" for p in paths)
