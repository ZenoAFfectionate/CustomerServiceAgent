# -*- coding: utf-8 -*-
"""rag/indexing/local_keyword_index.py 单元测试（TF-IDF 关键词检索降级实现）。"""
from rag.indexing.local_keyword_index import LocalKeywordStore
from rag.schema import DocBlock


def _make_block(gid, doc_id, title, text):
    return DocBlock(text=text, global_chunk_idx=gid, doc_id=doc_id, title=title)


class TestLocalKeywordStore:
    def test_upsert_and_count(self, tmp_path):
        store = LocalKeywordStore(path=str(tmp_path / "ks.json"))
        store.upsert([
            _make_block(0, "doc1", "广告限流", "账户存在异常投放行为时会被限流"),
            _make_block(1, "doc1", "退款规则", "退款需在7天内申请"),
        ])
        assert store.count() == 2

    def test_search_returns_relevant_result_first(self, tmp_path):
        store = LocalKeywordStore(path=str(tmp_path / "ks.json"))
        store.upsert([
            _make_block(0, "doc1", "广告限流规则", "千川广告投放出现异常会被限流处理"),
            _make_block(1, "doc1", "退款政策", "商品签收后七天内可申请无理由退款"),
            _make_block(2, "doc1", "物流查询", "可以在订单详情页查看物流轨迹信息"),
        ])
        results = store.search("千川广告限流怎么处理", top_k=3)
        assert len(results) >= 1
        assert results[0].global_chunk_idx == 0

    def test_search_empty_query_returns_empty(self, tmp_path):
        store = LocalKeywordStore(path=str(tmp_path / "ks.json"))
        store.upsert([_make_block(0, "doc1", "标题", "内容")])
        assert store.search("", top_k=5) == []
        assert store.search("   ", top_k=5) == []

    def test_search_on_empty_store_returns_empty(self, tmp_path):
        store = LocalKeywordStore(path=str(tmp_path / "ks.json"))
        assert store.search("任意查询", top_k=5) == []

    def test_search_respects_top_k(self, tmp_path):
        store = LocalKeywordStore(path=str(tmp_path / "ks.json"))
        store.upsert([_make_block(i, "doc1", f"标题{i}", f"广告限流相关内容第{i}条") for i in range(5)])
        results = store.search("广告限流", top_k=2)
        assert len(results) <= 2

    def test_delete_by_doc_id(self, tmp_path):
        store = LocalKeywordStore(path=str(tmp_path / "ks.json"))
        store.upsert([
            _make_block(0, "docA", "A标题", "A的内容"),
            _make_block(1, "docB", "B标题", "B的内容"),
        ])
        removed = store.delete_by_doc_id("docA")
        assert removed == 1
        assert store.count() == 1

    def test_persistence_across_instances(self, tmp_path):
        path = str(tmp_path / "ks.json")
        store1 = LocalKeywordStore(path=path)
        store1.upsert([_make_block(0, "docA", "持久化", "持久化测试内容")])

        store2 = LocalKeywordStore(path=path)
        assert store2.count() == 1

    def test_health_check_always_true(self, tmp_path):
        store = LocalKeywordStore(path=str(tmp_path / "ks.json"))
        assert store.health_check() is True

    def test_unrelated_query_scores_low_or_filtered(self, tmp_path):
        store = LocalKeywordStore(path=str(tmp_path / "ks.json"))
        store.upsert([_make_block(0, "doc1", "退款规则", "退款需在签收后七天内申请")])
        results = store.search("完全无关的量子物理话题", top_k=5)
        # 完全不相关的查询：结果为空，或分数极低（<=0.3）
        assert results == [] or all(r.score <= 0.3 for r in results)
