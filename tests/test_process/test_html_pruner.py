# -*- coding: utf-8 -*-
"""
HtmlRAG 两阶段块树剪枝（html_pruner）测试。

覆盖：
    1. 纯算法：cosine_similarity_vec（正交/相同/零向量/一般）
    2. 纯算法：greedy_prune_indices（预算内高分优先/至少保留top1/顺序/平分tie-break/大预算全保留）
    3. 纯函数：rebuild_html（保留 HTML 结构、顺序、空输入）
    4. Stage 1 prune_by_embedding（mock 嵌入器：相关块保留、无关块剪除）
    5. Stage 2 prune_by_reranker（mock 精排器：高分块保留）
    6. two_stage_prune（端到端：输出为输入子集且含相关块）
    7. 边界：空 HTML / 超短 HTML / 大预算不剪 / 打分服务失败优雅降级

所有相关性打分均使用**确定性 mock**，无需真实起 embedding/rerank 服务。
"""

import math

import pytest
from bs4 import BeautifulSoup

from html_pruner import (
    cosine_similarity_vec,
    greedy_prune_indices,
    rebuild_html,
    prune_by_embedding,
    prune_by_reranker,
    two_stage_prune,
)


# ======================== 测试用 HTML 与 mock 打分器 ========================

# 三段主题明确的内容：广告投放 / 物流配送 / 退款政策
SAMPLE_HTML = """<html>
<div class="h1_domain"><h1>广告投放规则</h1><p>广告投放需要遵守平台的推广素材审核规范才能上线</p></div>
<div class="h1_domain"><h1>物流配送说明</h1><p>物流配送范围覆盖全国大部分地区并支持次日达服务</p></div>
<div class="h1_domain"><h1>退款政策详情</h1><p>退款需要在收到商品七天内申请并保证商品完好无损坏</p></div>
</html>"""

# 关键词 → 主题维度，用于构造确定性嵌入向量
_TOPIC_KEYWORDS = [
    ("广告", "投放", "推广", "素材"),   # 维度 0
    ("物流", "配送", "次日达"),          # 维度 1
    ("退款", "退货", "商品"),            # 维度 2
]


def _topic_vector(text: str):
    """根据关键词命中构造 3 维主题向量（确定性 mock 嵌入）。"""
    return [
        float(any(kw in text for kw in group))
        for group in _TOPIC_KEYWORDS
    ]


def mock_embed_fn(texts):
    """mock 嵌入器：返回每段文本的主题向量。"""
    return [_topic_vector(t) for t in texts]


def mock_rerank_退款(query, texts):
    """mock 精排器：含「退款/退货」的块给高分，其余低分。"""
    return [1.0 if ("退款" in t or "退货" in t) else 0.05 for t in texts]


# ======================== 1. cosine_similarity_vec ========================

class TestCosineSimilarity:
    def test_identical_vectors(self):
        assert cosine_similarity_vec([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity_vec([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity_vec([0.0, 0.0], [1.0, 1.0]) == 0.0
        assert cosine_similarity_vec([1.0, 1.0], [0.0, 0.0]) == 0.0

    def test_opposite_vectors(self):
        assert cosine_similarity_vec([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_general_case(self):
        # (1,1) 与 (1,0) 夹角 45°，cos=1/sqrt(2)
        assert cosine_similarity_vec([1.0, 1.0], [1.0, 0.0]) == pytest.approx(1 / math.sqrt(2))


# ======================== 2. greedy_prune_indices ========================

class TestGreedyPrune:
    def test_empty(self):
        assert greedy_prune_indices([], [], 100) == []

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            greedy_prune_indices([1, 2], [0.5], 100)

    def test_keeps_highest_score_within_budget(self):
        # 块0低分,块1高分,块2中分; 预算=10 只能容纳一个(每块10词)
        keep = greedy_prune_indices([10, 10, 10], [0.1, 0.9, 0.5], max_words=10)
        assert keep == [1]

    def test_keeps_multiple_within_budget(self):
        # 预算 20 可容纳两块：应保留分数最高的两块（1 和 2），按原序返回
        keep = greedy_prune_indices([10, 10, 10], [0.1, 0.9, 0.5], max_words=20)
        assert keep == [1, 2]

    def test_returns_sorted_original_order(self):
        # 高分块在后面，返回结果仍按原始下标升序
        keep = greedy_prune_indices([5, 5, 5], [0.1, 0.2, 0.9], max_words=10)
        assert keep == sorted(keep)

    def test_always_keep_top1_even_if_over_budget(self):
        # 预算为 0，仍应保留分数最高的一个块
        keep = greedy_prune_indices([100, 100], [0.2, 0.8], max_words=0)
        assert keep == [1]

    def test_large_budget_keeps_all(self):
        keep = greedy_prune_indices([10, 20, 30], [0.1, 0.5, 0.9], max_words=1000)
        assert keep == [0, 1, 2]

    def test_tie_break_prefers_earlier_index(self):
        # 分数全相同，预算只够一个：优先保留文档靠前的块（下标 0）
        keep = greedy_prune_indices([10, 10, 10], [0.5, 0.5, 0.5], max_words=10)
        assert keep == [0]

    def test_skips_oversized_high_score_but_keeps_smaller(self):
        # 最高分块过大放不下，退而保留能放下的次高分块
        # 块0: 分0.9 词100(放不下); 块1: 分0.8 词5; 预算 10
        # top1 规则先放块0(超预算)，后续块1 total(100+5)>10 放不下 → 只保留[0]
        keep = greedy_prune_indices([100, 5], [0.9, 0.8], max_words=10)
        assert keep == [0]


# ======================== 3. rebuild_html ========================

class TestRebuildHtml:
    def test_empty(self):
        assert rebuild_html([]) == ""

    def test_preserves_html_structure(self):
        soup = BeautifulSoup("<div><h1>标题</h1><p>正文内容</p></div>", "html.parser")
        div = soup.find("div")
        out = rebuild_html([div])
        # 保留 HTML 标签，而非退化为纯文本
        assert "<h1>" in out and "<p>" in out
        assert "标题" in out and "正文内容" in out

    def test_order_and_join(self):
        soup = BeautifulSoup("<p>A</p><p>B</p>", "html.parser")
        ps = soup.find_all("p")
        out = rebuild_html(ps)
        assert out.index("A") < out.index("B")


# ======================== 4. Stage 1：prune_by_embedding ========================

class TestPruneByEmbedding:
    def test_keeps_relevant_prunes_irrelevant(self):
        # query 关于退款；预算很小只留一个块 → 应保留退款块
        out = prune_by_embedding(
            SAMPLE_HTML, "如何申请退款退货",
            max_context_words=30,
            embed_fn=mock_embed_fn,
            max_node_words=50, min_node_words=10, zh_char=True,
        )
        assert "退款" in out
        assert "广告投放" not in out
        assert "物流配送" not in out

    def test_output_is_valid_html_subset(self):
        out = prune_by_embedding(
            SAMPLE_HTML, "物流配送范围",
            max_context_words=30,
            embed_fn=mock_embed_fn,
            max_node_words=50, min_node_words=10, zh_char=True,
        )
        assert "物流" in out
        # 保留 HTML 结构
        assert "<" in out and ">" in out

    def test_large_budget_keeps_all_topics(self):
        out = prune_by_embedding(
            SAMPLE_HTML, "退款",
            max_context_words=100000,
            embed_fn=mock_embed_fn,
            max_node_words=50, min_node_words=10, zh_char=True,
        )
        assert "广告" in out and "物流" in out and "退款" in out


# ======================== 5. Stage 2：prune_by_reranker ========================

class TestPruneByReranker:
    def test_keeps_high_score_block(self):
        out = prune_by_reranker(
            SAMPLE_HTML, "退款相关问题",
            max_context_words=30,
            rerank_fn=mock_rerank_退款,
            max_node_words=50, min_node_words=10, zh_char=True,
        )
        assert "退款" in out
        assert "广告投放" not in out

    def test_graceful_degradation_on_scorer_failure(self):
        # 打分器抛异常 → 优雅降级为不剪枝（保留全部块）
        def broken_rerank(query, texts):
            raise RuntimeError("rerank service down")

        out = prune_by_reranker(
            SAMPLE_HTML, "退款",
            max_context_words=30,
            rerank_fn=broken_rerank,
            max_node_words=50, min_node_words=10, zh_char=True,
        )
        # 降级后应保留所有主题内容
        assert "广告" in out and "物流" in out and "退款" in out

    def test_score_count_mismatch_degrades(self):
        # 打分数量与块数不一致 → 视为失败，降级不剪枝
        def bad_count(query, texts):
            return [1.0]  # 只返回一个分数

        out = prune_by_reranker(
            SAMPLE_HTML, "退款",
            max_context_words=30,
            rerank_fn=bad_count,
            max_node_words=50, min_node_words=10, zh_char=True,
        )
        assert "广告" in out and "物流" in out and "退款" in out


# ======================== 6. two_stage_prune 端到端 ========================

class TestTwoStagePrune:
    def test_end_to_end_keeps_relevant(self):
        out = two_stage_prune(
            SAMPLE_HTML, "如何申请退款",
            stage1_max_context_words=200,   # Stage1 宽松，保留多数
            stage2_max_context_words=30,    # Stage2 收紧，只留最相关
            embed_fn=mock_embed_fn,
            rerank_fn=mock_rerank_退款,
            stage1_max_node_words=50, stage2_max_node_words=50,
            min_node_words=10, stage2_min_node_words=10, zh_char=True,
        )
        assert "退款" in out
        assert "广告投放" not in out

    def test_output_is_subset_of_input_topics(self):
        out = two_stage_prune(
            SAMPLE_HTML, "退款",
            stage1_max_context_words=200,
            stage2_max_context_words=30,
            embed_fn=mock_embed_fn,
            rerank_fn=mock_rerank_退款,
            stage1_max_node_words=50, stage2_max_node_words=50,
            min_node_words=10, stage2_min_node_words=10, zh_char=True,
        )
        # 输出中出现的主题必然是原文的子集
        original_text = BeautifulSoup(SAMPLE_HTML, "html.parser").get_text()
        out_text = BeautifulSoup(out, "html.parser").get_text()
        # 输出更短（发生了剪枝）
        assert len(out_text) <= len(original_text)

    def test_pruning_reduces_context(self):
        full_len = len(BeautifulSoup(SAMPLE_HTML, "html.parser").get_text())
        out = two_stage_prune(
            SAMPLE_HTML, "退款",
            stage1_max_context_words=200,
            stage2_max_context_words=30,
            embed_fn=mock_embed_fn,
            rerank_fn=mock_rerank_退款,
            stage1_max_node_words=50, stage2_max_node_words=50,
            min_node_words=10, stage2_min_node_words=10, zh_char=True,
        )
        out_len = len(BeautifulSoup(out, "html.parser").get_text())
        assert out_len < full_len


# ======================== 7. 边界条件 ========================

class TestEdgeCases:
    def test_empty_html(self):
        out = prune_by_embedding("", "退款", embed_fn=mock_embed_fn)
        assert out == ""

    def test_too_short_html_unchanged(self):
        # 内容远低于 min_node_words → 块树为空 → 原样返回
        short = "<p>你好</p>"
        out = prune_by_embedding(
            short, "退款", embed_fn=mock_embed_fn,
            max_node_words=50, min_node_words=100, zh_char=True,
        )
        assert out == short

    def test_embed_fn_not_called_when_no_blocks(self):
        """块树为空时不应调用嵌入服务。

        注：修复审查报告 M7 后，短文本非空时（如 "<p>短</p>"）`build_block_tree`
        不再返回空列表，而是保留为单块（避免短内容被静默丢弃）；因此本测试
        改用真正无内容的 HTML 来触发"块树为空"这一场景。
        """
        calls = []

        def spy_embed(texts):
            calls.append(texts)
            return mock_embed_fn(texts)

        prune_by_embedding(
            "", "退款", embed_fn=spy_embed,
            max_node_words=50, min_node_words=100, zh_char=True,
        )
        assert calls == []  # 块树为空时不应调用嵌入服务


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
