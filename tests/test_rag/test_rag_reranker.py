# -*- coding: utf-8 -*-
"""rag/retrieval/reranker.py 单元测试：本地余弦精排 + TEI 服务降级逻辑。"""
from rag.retrieval.reranker import rerank
from rag.schema import DocBlock


def _block(gid, title, text, score=0.0):
    return DocBlock(global_chunk_idx=gid, title=title, text=text, score=score)


class TestLocalRerank:
    def test_local_rerank_orders_by_relevance(self):
        blocks = [
            _block(1, "退款政策", "退款需在签收后七天内申请"),
            _block(2, "广告限流规则", "千川广告投放异常会被限流处理"),
            _block(3, "物流查询", "订单详情页可查看物流轨迹"),
        ]
        reranked = rerank("广告限流怎么处理", blocks, top_k=3, backend="local")
        assert len(reranked) == 3
        assert reranked[0].global_chunk_idx == 2

    def test_top_k_truncation(self):
        blocks = [_block(i, f"标题{i}", f"内容{i}") for i in range(10)]
        reranked = rerank("查询", blocks, top_k=3, backend="local")
        assert len(reranked) == 3

    def test_empty_blocks_returns_empty(self):
        assert rerank("查询", [], top_k=5, backend="local") == []

    def test_empty_query_returns_original_blocks_truncated(self):
        blocks = [_block(i, f"标题{i}", f"内容{i}") for i in range(5)]
        result = rerank("", blocks, top_k=3, backend="local")
        assert len(result) == 3

    def test_source_retriever_marked_reranked(self):
        blocks = [_block(1, "标题", "内容")]
        result = rerank("查询", blocks, top_k=1, backend="local")
        assert result[0].source_retriever == "reranked"


class TestTeiRerankFallback:
    def test_tei_unavailable_falls_back_to_fused_results(self, monkeypatch):
        """TEI Reranker 不可用时应降级为跳过精排，直接返回原融合结果（DoD 要求：不崩溃）。"""
        import rag.retrieval.reranker as reranker_mod

        monkeypatch.setattr(reranker_mod, "_rerank_with_tei", lambda query, blocks: None)

        blocks = [_block(i, f"标题{i}", f"内容{i}", score=1.0 - i * 0.1) for i in range(5)]
        result = rerank("任意查询", blocks, top_k=3, backend="tei")
        assert len(result) == 3
        # 降级路径应保持原始顺序（未被精排打乱）
        assert [b.global_chunk_idx for b in result] == [0, 1, 2]

    def test_tei_available_reorders_by_score(self, monkeypatch):
        import rag.retrieval.reranker as reranker_mod

        def _fake_tei_rerank(query, blocks):
            # 模拟 TEI 返回与原顺序相反的分数
            for i, b in enumerate(blocks):
                b.score = float(i)
            return sorted(blocks, key=lambda b: -b.score)

        monkeypatch.setattr(reranker_mod, "_rerank_with_tei", _fake_tei_rerank)
        blocks = [_block(0, "A", "内容A"), _block(1, "B", "内容B"), _block(2, "C", "内容C")]
        result = rerank("查询", blocks, top_k=3, backend="tei")
        assert [b.global_chunk_idx for b in result] == [2, 1, 0]
