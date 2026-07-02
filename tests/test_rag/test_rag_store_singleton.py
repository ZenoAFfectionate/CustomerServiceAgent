# -*- coding: utf-8 -*-
"""向量库/关键词库单例缓存回归测试（对齐本轮审计修复 B1/P1/A1）。

覆盖：
    1. `get_vector_store()`/`get_keyword_store()` 多次调用返回同一实例（单例）。
    2. `reset_vector_store()`/`reset_keyword_store()` 能正确清空缓存。
    3. 回归验证"lost update"问题已修复：模拟多次"请求"分别调用
       `get_vector_store()` 后写入，验证所有写入均可见（若退化为每次 new
       实例，后写入会因基于旧快照而覆盖前一次写入，本测试即会失败）。
"""
import pytest

from rag.indexing.vector_store import get_vector_store, reset_vector_store
from rag.indexing.keyword_store import get_keyword_store, reset_keyword_store
from rag.schema import DocBlock

pytestmark = pytest.mark.usefixtures("clean_rag_data")


class TestVectorStoreSingleton:
    def test_get_vector_store_returns_same_instance(self):
        s1 = get_vector_store()
        s2 = get_vector_store()
        assert s1 is s2

    def test_reset_vector_store_clears_cache(self):
        s1 = get_vector_store()
        reset_vector_store()
        s2 = get_vector_store()
        assert s1 is not s2

    def test_reset_vector_store_specific_backend(self):
        s1 = get_vector_store(backend="local")
        reset_vector_store(backend="local")
        s2 = get_vector_store(backend="local")
        assert s1 is not s2

    def test_no_lost_update_across_simulated_requests(self):
        """模拟 3 次独立"请求"依次调用 get_vector_store() 后各自 upsert 一条数据，
        验证 3 条数据全部可见（单例共享同一份内存数据，不会互相覆盖）。
        """
        for i in range(3):
            store = get_vector_store()  # 模拟每次请求重新获取
            store.upsert([DocBlock(text=f"内容{i}", global_chunk_idx=i, embedding=[1.0, 0.0])])
        assert get_vector_store().count() == 3


class TestKeywordStoreSingleton:
    def test_get_keyword_store_returns_same_instance(self):
        s1 = get_keyword_store()
        s2 = get_keyword_store()
        assert s1 is s2

    def test_reset_keyword_store_clears_cache(self):
        s1 = get_keyword_store()
        reset_keyword_store()
        s2 = get_keyword_store()
        assert s1 is not s2

    def test_no_lost_update_across_simulated_requests(self):
        for i in range(3):
            store = get_keyword_store()
            store.upsert([DocBlock(text=f"关键词内容{i}", global_chunk_idx=i)])
        assert get_keyword_store().count() == 3
