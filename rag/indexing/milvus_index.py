# -*- coding: utf-8 -*-
"""Milvus 向量库索引（生产后端）。

对齐 TODO.md T1：
    - create_collection() / upsert_blocks(blocks) / delete_by_page(page_url)
    - Schema：`global_chunk_idx` 主键 + text/html_content/summary/block_path/time 等字段
    - 增量更新：以 page_url 为粒度先删后插，避免全量重建

同时实现 `BaseVectorStore` 接口（create_collection/upsert/delete_by_doc_id/search/
count/health_check），供 `rag/retrieval/hybrid_search.py` 与 pipeline 统一调用。

依赖 `pymilvus`；未安装或无法连接时，`health_check()` 返回 False，
上层应捕获异常并降级到 `LocalVectorStore`（见 `rag/indexing/local_index.py`）。

【优化点】本类现由 `store.get_vector_store()` 以进程级单例方式持有
（见 `rag/indexing/store.py` 注释），因此 `_connected`/`_collection_ready` 等
状态在进程生命周期内可安全复用，无需每次调用都重新建连/重复检查集合是否存在。
"""
import threading
from typing import List, Optional

from config.config_loader import logger
from rag.config import RAG_CONFIG
from rag.indexing.store import BaseVectorStore
from rag.schema import DocBlock, get_milvus_schema_fields


class MilvusVectorStore(BaseVectorStore):
    def __init__(self, host: str = None, port: int = None, collection_name: str = None, alias: str = "rag_default"):
        self.host = host or RAG_CONFIG["milvus_host"]
        self.port = port or RAG_CONFIG["milvus_port"]
        self.collection_name = collection_name or RAG_CONFIG["collection_name"]
        self.alias = alias
        self._collection = None
        self._connected = False
        self._collection_ready = False  # 【优化点】进程内已确认集合存在，跳过重复的 has_collection RPC
        self._conn_lock = threading.Lock()  # 【优化点】保护并发首次连接，避免重复 connect

    # -------------------- 连接管理 --------------------

    def _connect(self):
        if self._connected:
            return
        with self._conn_lock:
            if self._connected:
                return
            from pymilvus import connections
            connections.connect(
                alias=self.alias, host=self.host, port=str(self.port),
                user=RAG_CONFIG.get("milvus_user") or "",
                password=RAG_CONFIG.get("milvus_password") or "",
            )
            self._connected = True

    def health_check(self) -> bool:
        try:
            self._connect()
            from pymilvus import utility
            utility.list_collections(using=self.alias)
            return True
        except Exception:
            return False

    # -------------------- 建表 --------------------

    def create_collection(self, dim: int = 256) -> None:
        # 【优化点】进程内已确认集合就绪则直接返回，避免每次写入文档都发起一次
        # has_collection RPC（高频写入场景下可显著减少冗余网络往返）。
        if self._collection_ready and self._collection is not None:
            return

        from pymilvus import Collection, CollectionSchema, utility

        self._connect()
        if utility.has_collection(self.collection_name, using=self.alias):
            self._collection = Collection(self.collection_name, using=self.alias)
            self._collection_ready = True
            return

        schema = CollectionSchema(
            fields=get_milvus_schema_fields(embedding_dim=dim),
            description="RAG 知识块集合（对齐 process/ 输出字段）",
        )
        self._collection = Collection(self.collection_name, schema=schema, using=self.alias)
        self._collection.create_index(
            field_name="embedding",
            index_params={"index_type": "HNSW", "metric_type": "COSINE", "params": {"M": 16, "efConstruction": 200}},
        )
        self._collection.load()
        self._collection_ready = True

    def _get_collection(self):
        from pymilvus import Collection, utility
        if self._collection is None:
            self._connect()
            if not utility.has_collection(self.collection_name, using=self.alias):
                raise RuntimeError(f"Milvus 集合 {self.collection_name} 不存在，请先调用 create_collection()")
            self._collection = Collection(self.collection_name, using=self.alias)
            self._collection.load()
            self._collection_ready = True
        return self._collection

    # -------------------- 写入 --------------------

    def upsert(self, blocks: List[DocBlock]) -> int:
        return self.upsert_blocks(blocks)

    def upsert_blocks(self, blocks: List[DocBlock]) -> int:
        """写入/覆盖文档块（先按主键删除再插入，实现 upsert 语义与幂等）。"""
        if not blocks:
            return 0
        collection = self._get_collection()
        ids = [b.global_chunk_idx for b in blocks]
        collection.delete(expr=f"global_chunk_idx in {ids}")

        rows = [b.to_dict(with_embedding=True) for b in blocks]

        def _col(field):
            return [r[field] for r in rows]

        data = [
            _col("global_chunk_idx"), _col("doc_id"), _col("source"), _col("chunk_idx"),
            _col("page_name"), _col("title"), _col("page_url"), _col("text"),
            _col("html_content"), _col("block_path"), _col("summary"), _col("question"),
            _col("time"), _col("embedding"),
        ]
        collection.insert(data)
        collection.flush()
        return len(blocks)

    # -------------------- 删除 --------------------

    def delete_by_doc_id(self, doc_id: str) -> int:
        collection = self._get_collection()
        # 【修复 N7】此前 expr=f'doc_id == "{doc_id}"' 直接 f-string 拼接，
        # 若 doc_id 含 "/\/不可见字符会导致 Expr 语法错误或匹配错误集合。
        # 现对 doc_id 做白名单校验（与 metadata.new_doc_id 的 [0-9a-f] 格式一致），
        # 非法字符直接拒绝。
        import re
        if not re.match(r'^[0-9a-zA-Z_\-]+$', doc_id or ""):
            raise ValueError(f"doc_id 含非法字符，仅允许 [0-9a-zA-Z_-]: {doc_id!r}")
        res = collection.query(expr=f'doc_id == "{doc_id}"', output_fields=["global_chunk_idx"])
        if not res:
            return 0
        collection.delete(expr=f"doc_id == \"{doc_id}\"")
        collection.flush()
        return len(res)

    def delete_by_page(self, page_url: str) -> int:
        """按 page_url 增量删除（对齐 T1 增量更新要求：以 page_url 为粒度先删后插）。"""
        collection = self._get_collection()
        res = collection.query(expr=f'page_url == "{page_url}"', output_fields=["global_chunk_idx"])
        if not res:
            return 0
        collection.delete(expr=f'page_url == "{page_url}"')
        collection.flush()
        return len(res)

    # -------------------- 检索 --------------------

    def search(self, query_vector: List[float], top_k: int) -> List[DocBlock]:
        collection = self._get_collection()
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=top_k,
            output_fields=[f for f in
                            ["global_chunk_idx", "doc_id", "source", "chunk_idx", "page_name", "title",
                             "page_url", "text", "html_content", "block_path", "summary", "question", "time"]],
        )
        blocks = []
        for hit in results[0]:
            d = {f: hit.entity.get(f) for f in
                 ["global_chunk_idx", "doc_id", "source", "chunk_idx", "page_name", "title",
                  "page_url", "text", "html_content", "block_path", "summary", "question", "time"]}
            d["score"] = float(hit.distance)
            d["source_retriever"] = "milvus"
            blocks.append(DocBlock.from_dict(d))
        return blocks

    def count(self) -> int:
        try:
            return self._get_collection().num_entities
        except Exception as e:
            # 静默返回 0 会让 get_stats()/dashboard() 展示"知识库为空"这种误导性
            # 信息（实际可能是连接故障），记录告警以便与"真的没有数据"区分。
            logger.warning(f"⚠️ Milvus count() 失败，返回 0（可能是连接异常而非真的无数据）: {e}")
            return 0


def create_collection(dim: int = 256, **kwargs) -> MilvusVectorStore:
    store = MilvusVectorStore(**kwargs)
    store.create_collection(dim=dim)
    return store
