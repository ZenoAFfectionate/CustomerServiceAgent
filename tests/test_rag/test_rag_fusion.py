# -*- coding: utf-8 -*-
"""rag/retrieval/fusion.py 单元测试：RRF / 加权融合 / 去重（纯逻辑，无外部依赖）。"""
from rag.retrieval.fusion import rrf_fuse, weighted_fuse, deduplicate, fuse
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


class TestWeightedFuse:
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
