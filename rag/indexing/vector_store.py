# -*- coding: utf-8 -*-
"""向量存储抽象层。

定义统一的 `BaseVectorStore` 接口，`indexing/milvus_index.py`（生产，对接真实
Milvus）与 `indexing/local_vector_index.py`（本地降级，无需外部服务）均实现该
接口，`retrieval/milvus_search.py` 与上层 pipeline 只依赖该接口，
不关心具体后端 —— 满足"组件可独立替换"的解耦要求。

【优化点】单例缓存（对齐 `embedder.py`/`registry.py` 的单例模式）：
    原实现 `get_vector_store()` 每次调用都 `new` 一个实例，带来两个问题：
    1) 本地降级后端：每次调用都从磁盘全量重新加载 JSON，且每个实例持有独立的
       `threading.Lock()`，并发写入时锁形同虚设，会发生"读-改-写"竞态导致的
       更新丢失（lost update）。
    2) 生产后端（Milvus）：每次调用都是一个"冷"实例，`_connected=False`，
       每次请求都会重新建立连接，严重拖慢响应并浪费连接资源。
    现改为进程级单例（按 backend 分别缓存），配合 `reset_vector_store()` 供
    测试/热切换场景显式重置。
"""
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from rag.schema import DocBlock


class BaseVectorStore(ABC):
    """向量存储统一接口。"""

    @abstractmethod
    def create_collection(self, dim: int) -> None:
        """创建/确保集合（表）存在。"""

    @abstractmethod
    def upsert(self, blocks: List[DocBlock]) -> int:
        """写入（存在则覆盖）文档块及其向量，返回写入条数。"""

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> int:
        """按 doc_id 删除全部关联块，返回删除条数。"""

    @abstractmethod
    def search(self, query_vector: List[float], top_k: int) -> List[DocBlock]:
        """向量近邻检索，返回按相似度降序排列的 DocBlock（含 score）。"""

    @abstractmethod
    def count(self) -> int:
        """当前集合中的文档块总数。"""

    @abstractmethod
    def health_check(self) -> bool:
        """检查后端服务是否可用。"""


_store_instances: Dict[str, "BaseVectorStore"] = {}
_store_lock = threading.Lock()


def _create_store(backend: str) -> "BaseVectorStore":
    if backend == "milvus":
        from rag.indexing.milvus_index import MilvusVectorStore
        return MilvusVectorStore()
    from rag.indexing.local_vector_index import LocalVectorStore
    return LocalVectorStore()


def get_vector_store(backend: Optional[str] = None) -> "BaseVectorStore":
    """工厂函数：按配置返回 Milvus 或本地向量库的**进程级单例**。

    Args:
        backend: 显式指定后端（"local"/"milvus"），默认取 `RAG_CONFIG['vector_backend']`。
                 不同 backend 分别缓存单例，互不影响。
    """
    from rag.config import RAG_CONFIG
    backend = backend or RAG_CONFIG["vector_backend"]
    instance = _store_instances.get(backend)
    if instance is None:
        with _store_lock:
            instance = _store_instances.get(backend)
            if instance is None:
                instance = _create_store(backend)
                _store_instances[backend] = instance
    return instance


def reset_vector_store(backend: Optional[str] = None) -> None:
    """重置向量库单例缓存（供测试隔离 / 运行时热切换配置使用）。

    Args:
        backend: 仅重置指定 backend 的缓存；为 None 时清空全部缓存。
    """
    with _store_lock:
        if backend is None:
            _store_instances.clear()
        else:
            _store_instances.pop(backend, None)
