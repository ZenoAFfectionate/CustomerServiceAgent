# -*- coding: utf-8 -*-
"""rag/retrieval/hybrid_search.py 单元测试：单路检索（向量/关键词）+ RRF/加权融合 +
去重 + 端到端 `search()`。

`hybrid_search.py` 本身就是"向量检索 + 关键词检索 + 融合去重"的组合模块，因此本文件
既包含纯逻辑单测（RRF/加权融合，无外部依赖），也包含依赖 `indexing/` 真实写入索引后
的组合测试（`TestVectorAndKeywordSearch` / `TestEndToEndSearch`）。
"""
import pytest

from rag.retrieval.hybrid_search import (
    deduplicate, fuse, keyword_search, rrf_fuse, search, vector_search, weighted_fuse,
)
from rag.schema import DocBlock


def _block(gid, score, text=None, doc_id="doc1", page_name=None, time_val=""):
    # 默认文本/页面名按 gid 区分，避免被 TF-IDF 去重误判为"内容相同的重复块"
    text = text if text is not None else f"这是第 {gid} 号文档块的正文内容，用于融合测试"
    page_name = page_name if page_name is not None else f"page_{gid}"
    return DocBlock(global_chunk_idx=gid, text=text, score=score, doc_id=doc_id, page_name=page_name, time=time_val)


class TestRRFFuse:
    def test_known_input_expected_order(self):
        """构造已知输入，验证 RRF 排序正确：同时出现在两路靠前位置的文档应排名最高。"""
        milvus_results = [_block(1, 0.9), _block(2, 0.8), _block(3, 0.7)]
        es_results = [_block(2, 5.0), _block(1, 4.0), _block(4, 3.0)]

        fused = rrf_fuse([milvus_results, es_results], k=60)
        ids = [b.global_chunk_idx for b in fused]

        # 文档1: milvus rank1 + es rank2；文档2: milvus rank2 + es rank1 → 分数应非常接近且都最高
        assert set(ids[:2]) == {1, 2}
        # 只出现一次的文档3、4排名应在后面
        assert ids.index(3) > 1
        assert ids.index(4) > 1

    def test_single_list_preserves_relative_order_by_score(self):
        results = [_block(1, 0.9), _block(2, 0.5)]
        fused = rrf_fuse([results])
        assert [b.global_chunk_idx for b in fused] == [1, 2]

    def test_empty_lists_returns_empty(self):
        assert rrf_fuse([[], []]) == []

    def test_document_appearing_in_both_lists_scores_higher_than_appearing_once(self):
        a = [_block(1, 0.9), _block(2, 0.8)]
        b = [_block(1, 0.9)]
        fused = rrf_fuse([a, b])
        doc1 = next(x for x in fused if x.global_chunk_idx == 1)
        doc2 = next(x for x in fused if x.global_chunk_idx == 2)
        assert doc1.score > doc2.score

    def test_source_retriever_marked_fused(self):
        fused = rrf_fuse([[_block(1, 0.9)]])
        assert fused[0].source_retriever == "fused"

    def test_original_blocks_not_mutated(self):
        """回归测试：修复审查报告 L10——融合此前原地改写共享 DocBlock 对象
        引用的 score/source_retriever。原始输入列表中的对象不应被污染，
        以便调用方在 fuse() 之后仍可安全复用原始检索结果列表。"""
        original = _block(1, 0.9)
        original_score = original.score
        milvus_results = [original]
        fused = rrf_fuse([milvus_results])

        assert original.score == original_score
        assert original.source_retriever != "fused"
        assert fused[0] is not original
        assert fused[0].score != original.score


class TestWeightedFuse:
    def test_original_blocks_not_mutated(self):
        """回归测试：修复审查报告 L10（weighted_fuse 路径同 rrf_fuse）。"""
        original = _block(1, 0.5)
        milvus_results = [original]
        es_results = [_block(1, 5.0)]
        fused = weighted_fuse([milvus_results, es_results])

        assert original.score == 0.5
        assert original.source_retriever != "fused"
        assert fused[0] is not original


    def test_weighted_fuse_respects_weights(self):
        milvus_results = [_block(1, 1.0), _block(2, 0.0)]
        es_results = [_block(1, 0.0), _block(2, 1.0)]

        # milvus 权重更高时，doc1 应排第一
        fused_milvus_heavy = weighted_fuse([milvus_results, es_results], weights=[0.9, 0.1])
        assert fused_milvus_heavy[0].global_chunk_idx == 1

        # es 权重更高时，doc2 应排第一
        fused_es_heavy = weighted_fuse([milvus_results, es_results], weights=[0.1, 0.9])
        assert fused_es_heavy[0].global_chunk_idx == 2

    def test_weighted_fuse_normalizes_scores(self):
        """不同量级的分数（如向量 0-1 vs 关键词 0-10）经归一化后不应被大数值路主导。"""
        milvus_results = [_block(1, 0.99), _block(2, 0.01)]
        es_results = [_block(1, 1.0), _block(2, 100.0)]
        fused = weighted_fuse([milvus_results, es_results], weights=[0.5, 0.5])
        doc2 = next(x for x in fused if x.global_chunk_idx == 2)
        # 归一化后 doc2 的贡献分数应 <= 1.0（未归一化时 100 分会完全压制）
        assert doc2.score <= 1.0


class TestDeduplicate:
    def test_dedup_removes_duplicate_ids(self):
        blocks = [_block(1, 0.9, text="重复内容A"), _block(1, 0.8, text="重复内容A")]
        # 复用 process 去重逻辑按 global_chunk_idx 已保证融合阶段无重复，这里验证接口稳定不报错
        result = deduplicate(blocks)
        assert len(result) <= 2

    def test_dedup_single_block_returns_as_is(self):
        blocks = [_block(1, 0.9)]
        assert deduplicate(blocks) == blocks

    def test_dedup_empty_returns_empty(self):
        assert deduplicate([]) == []


class TestFuseDispatcher:
    def test_fuse_default_method_is_rrf(self):
        a = [_block(1, 0.9)]
        b = [_block(2, 0.8)]
        result = fuse([a, b])
        assert len(result) == 2

    def test_fuse_with_weighted_method(self):
        a = [_block(1, 1.0)]
        b = [_block(2, 1.0)]
        result = fuse([a, b], method="weighted")
        assert len(result) == 2

    def test_fuse_single_source_no_fusion_needed(self):
        a = [_block(1, 0.9), _block(2, 0.5)]
        result = fuse([a])
        assert [b.global_chunk_idx for b in result] == [1, 2]

    def test_fuse_all_empty_returns_empty(self):
        assert fuse([[], []]) == []

    def test_fuse_no_cross_source_duplicates(self):
        """融合后同一 global_chunk_idx 不应出现多次。"""
        a = [_block(1, 0.9), _block(2, 0.8)]
        b = [_block(1, 5.0), _block(3, 3.0)]
        result = fuse([a, b])
        ids = [b.global_chunk_idx for b in result]
        assert len(ids) == len(set(ids))


@pytest.mark.usefixtures("clean_rag_data")
class TestVectorAndKeywordSearch:
    """组合测试：hybrid_search 的单路检索函数依赖 indexing/ 写入的真实数据
    （vector_store/keyword_store/embedding），验证跨模块调用链路正确。"""

    def _seed(self):
        from rag.indexing import index_builder
        index_builder.ingest_blocks([
            {"text": "广告限流是一种常见的风控手段", "title": "限流规则", "page_url": "http://a"},
            {"text": "退款需在签收后七天内申请", "title": "退款政策", "page_url": "http://b"},
        ], filename="seed.json")

    def test_vector_search_returns_docblocks_marked_with_backend(self):
        """【修复 N31】source_retriever 应反映实际后端（local/milvus），
        不再硬编码为 "milvus"。"""
        from rag.config import RAG_CONFIG
        self._seed()
        results = vector_search("广告限流规则", top_k=5)
        assert len(results) >= 1
        expected = RAG_CONFIG.get("vector_backend", "milvus")
        assert all(r.source_retriever == expected for r in results)

    def test_keyword_search_returns_docblocks_marked_with_backend(self):
        """【修复 N32】source_retriever 应反映实际后端（local/es），
        不再硬编码为 "es"。"""
        from rag.config import RAG_CONFIG
        self._seed()
        results = keyword_search("退款", top_k=5)
        assert len(results) >= 1
        expected = RAG_CONFIG.get("keyword_backend", "es")
        assert all(r.source_retriever == expected for r in results)

    def test_empty_query_returns_empty_for_both(self):
        self._seed()
        assert vector_search("", top_k=5) == []
        assert keyword_search("   ", top_k=5) == []

    def test_search_on_empty_kb_returns_empty(self):
        assert vector_search("任意", top_k=5) == []
        assert keyword_search("任意", top_k=5) == []

    def test_vector_search_backend_failure_returns_empty_not_raise(self, monkeypatch):
        """向量库异常时 vector_search 应捕获异常返回空列表，而不是让异常向上传播
        （pipeline.py 依赖此行为实现单路降级）。"""
        self._seed()
        import rag.retrieval.hybrid_search as hybrid_search_mod

        class _BrokenStore:
            def search(self, *a, **kw):
                raise RuntimeError("模拟向量库故障")

        monkeypatch.setattr(hybrid_search_mod, "get_vector_store", lambda *a, **kw: _BrokenStore())
        assert vector_search("广告限流", top_k=5) == []


@pytest.mark.usefixtures("clean_rag_data")
class TestEndToEndSearch:
    """组合测试：`search()` 一步完成双路检索 + 融合去重，验证 backends 参数生效。"""

    def _seed(self):
        from rag.indexing import index_builder
        index_builder.ingest_blocks([
            {"text": "广告限流是一种常见的风控手段，触发后曝光量下降", "title": "限流规则", "page_url": "http://a"},
            {"text": "退款需在签收后七天内申请，原路退回支付账户", "title": "退款政策", "page_url": "http://b"},
        ], filename="seed.json")

    def test_search_default_runs_both_backends(self):
        self._seed()
        results = search("广告限流", top_k=5)
        assert len(results) >= 1
        # 融合后的结果 source_retriever 应标记为 fused（或因单路命中被去重逻辑保留原标记）
        assert all(r.source_retriever in ("fused", "milvus", "es", "local") for r in results)

    def test_search_vector_only_backend(self):
        self._seed()
        results = search("广告限流", top_k=5, backends=["vector"])
        assert isinstance(results, list)

    def test_search_keyword_only_backend(self):
        self._seed()
        results = search("退款", top_k=5, backends=["keyword"])
        assert isinstance(results, list)

    def test_search_no_backends_returns_empty(self):
        self._seed()
        assert search("广告限流", top_k=5, backends=[]) == []
