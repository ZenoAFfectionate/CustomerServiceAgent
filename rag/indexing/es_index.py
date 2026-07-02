# -*- coding: utf-8 -*-
"""Elasticsearch 关键词索引（生产后端）。

对齐 TODO.md T1：
    - create_index()（IK 分词 mapping，无 IK 插件时自动降级 standard）
    - bulk_index(blocks) / delete_by_page(page_url)
    - 增量更新：以 page_url 为粒度先删后插

依赖 `elasticsearch` 包；服务不可用时 `health_check()` 返回 False，
上层应捕获异常并降级到 `LocalKeywordStore`。

【优化点】本类现由 `keyword_store.get_keyword_store()` 以进程级单例方式持有，
`_client`/`_index_ready` 等状态在进程生命周期内可安全复用。
"""
import threading
from typing import List

from rag.config import RAG_CONFIG
from rag.indexing.keyword_store import BaseKeywordStore
from rag.schema import DocBlock, get_es_mapping, get_es_mapping_fallback

_OUTPUT_FIELDS = [
    "global_chunk_idx", "doc_id", "source", "chunk_idx", "page_name", "title",
    "page_url", "text", "html_content", "block_path", "summary", "question", "time",
]


class ESKeywordStore(BaseKeywordStore):
    def __init__(self, host: str = None, port: int = None, index_name: str = None):
        self.host = host or RAG_CONFIG["es_host"]
        self.port = port or RAG_CONFIG["es_port"]
        self.index_name = index_name or RAG_CONFIG["index_name"]
        self._client = None
        self._index_ready = False  # 【优化点】进程内已确认索引存在，跳过重复的 indices.exists 请求
        self._client_lock = threading.Lock()  # 【优化点】保护并发首次创建客户端

    def _get_client(self):
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    from elasticsearch import Elasticsearch
                    auth = None
                    if RAG_CONFIG.get("es_user"):
                        auth = (RAG_CONFIG["es_user"], RAG_CONFIG.get("es_password") or "")
                    self._client = Elasticsearch(f"http://{self.host}:{self.port}", basic_auth=auth)
        return self._client

    def health_check(self) -> bool:
        try:
            return bool(self._get_client().ping())
        except Exception:
            return False

    def create_index(self) -> None:
        # 【优化点】跳过已确认存在的索引，减少每次写入前的冗余 HTTP 请求
        if self._index_ready:
            return
        client = self._get_client()
        if client.indices.exists(index=self.index_name):
            self._index_ready = True
            return
        try:
            client.indices.create(index=self.index_name, body=get_es_mapping())
        except Exception:
            # 集群未安装 IK 分词插件，降级为 standard analyzer
            client.indices.create(index=self.index_name, body=get_es_mapping_fallback())
        self._index_ready = True

    def upsert(self, blocks: List[DocBlock]) -> int:
        return self.bulk_index(blocks)

    def bulk_index(self, blocks: List[DocBlock]) -> int:
        """批量写入（幂等：以 global_chunk_idx 作为 ES 文档 _id）。"""
        if not blocks:
            return 0
        from elasticsearch.helpers import bulk

        client = self._get_client()
        actions = [
            {
                "_index": self.index_name,
                "_id": b.global_chunk_idx,
                "_source": b.to_dict(with_embedding=False),
            }
            for b in blocks
        ]
        bulk(client, actions)
        client.indices.refresh(index=self.index_name)
        return len(blocks)

    def delete_by_doc_id(self, doc_id: str) -> int:
        client = self._get_client()
        resp = client.delete_by_query(
            index=self.index_name,
            body={"query": {"term": {"doc_id": doc_id}}},
        )
        client.indices.refresh(index=self.index_name)
        return resp.get("deleted", 0)

    def delete_by_page(self, page_url: str) -> int:
        """按 page_url 增量删除（对齐 T1 增量更新要求）。"""
        client = self._get_client()
        resp = client.delete_by_query(
            index=self.index_name,
            body={"query": {"term": {"page_url": page_url}}},
        )
        client.indices.refresh(index=self.index_name)
        return resp.get("deleted", 0)

    def search(self, query: str, top_k: int) -> List[DocBlock]:
        """使用 process/ 的 `build_optimal_jieba_query` 构造 ES bool 查询。"""
        client = self._get_client()
        es_query = self._build_query(query)
        resp = client.search(
            index=self.index_name,
            body={"query": es_query, "size": top_k, "_source": _OUTPUT_FIELDS},
        )
        results = []
        for hit in resp["hits"]["hits"]:
            d = dict(hit["_source"])
            d["score"] = float(hit["_score"] or 0.0)
            d["source_retriever"] = "es"
            results.append(DocBlock.from_dict(d))
        return results

    def _build_query(self, query: str) -> dict:
        from rag.indexing._process_compat import get_build_optimal_jieba_query_fn

        build_fn = get_build_optimal_jieba_query_fn()
        if build_fn is None:
            return {"multi_match": {"query": query, "fields": ["title^5", "summary^3", "text"]}}
        try:
            import jieba
            keywords = jieba.lcut(query)
            return build_fn(
                jieba_keywords=keywords,
                fields_config={
                    "title": {"boost": 5, "fuzzy": False},
                    "summary": {"boost": 3, "fuzzy": True},
                    "text": {"boost": 1, "fuzzy": True},
                },
            )
        except Exception:
            # 复用 process/ 分词逻辑失败时，退化为标准 multi_match
            return {"multi_match": {"query": query, "fields": ["title^5", "summary^3", "text"]}}

    def count(self) -> int:
        try:
            return int(self._get_client().count(index=self.index_name)["count"])
        except Exception:
            return 0
