# -*- coding: utf-8 -*-
"""存储抽象层（Store）：统一定义向量库与关键词库的接口 + 单例工厂。

本文件合并自此前的 `vector_store.py` + `keyword_store.py`——两者是完全对称的
"接口 + 按 backend 缓存单例的工厂函数"模式（仅 ABC 方法签名与工厂里判断的
`backend` 取值不同），拆成两个文件并不会带来额外的清晰度，合并后能在一处
看清"索引/检索层到底依赖了哪些抽象接口"。

    - `BaseVectorStore` + `get_vector_store()`/`reset_vector_store()`：
      `retrieval/hybrid_search.py`、`indexing/index_builder.py` 只依赖该接口，
      不关心具体是 `indexing/milvus_index.py`（生产）还是
      `indexing/local_index.py`（本地降级）。
    - `BaseKeywordStore` + `get_keyword_store()`/`reset_keyword_store()`：
      对称设计，具体实现是 `indexing/es_index.py` 或 `indexing/local_index.py`。

【为什么本文件不进一步与 `milvus_index.py`/`es_index.py`/`local_index.py` 合并】
接口定义与"某一种具体实现"合并会产生方向反转的依赖关系——例如把
`BaseVectorStore` 塞进 `milvus_index.py` 后，`local_index.py`（本地零依赖实现）
就必须 `from rag.indexing.milvus_index import BaseVectorStore`，让"本地降级、
不需要任何外部服务"的实现反而依赖一个文件名叫"milvus_index"的模块，读者
第一眼会困惑。保持接口独立、由各实现方反向依赖接口，才是单向、清晰的依赖方向。

【单例设计】`get_vector_store()`/`get_keyword_store()` 按 backend 分别缓存为
进程级单例（而非每次调用 new 一个实例），原因：
    1) 本地降级后端：每次调用都从磁盘全量重新加载 JSON，且每个实例持有独立的
       `threading.Lock()`，并发写入时锁形同虚设，会发生"读-改-写"竞态导致的
       更新丢失（lost update）。
    2) 生产后端（Milvus/ES）：每次调用都是一个"冷"实例，每次请求都会重新
       建立连接，严重拖慢响应并浪费连接资源。
配合 `reset_vector_store()`/`reset_keyword_store()` 供测试/热切换场景显式重置。
"""
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from rag.schema import DocBlock


# ======================================================================
# 向量存储
# ======================================================================

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


_vector_store_instances: Dict[str, "BaseVectorStore"] = {}
_vector_store_lock = threading.Lock()


def _create_vector_store(backend: str) -> "BaseVectorStore":
    if backend == "milvus":
        from rag.indexing.milvus_index import MilvusVectorStore
        return MilvusVectorStore()
    from rag.indexing.local_index import LocalVectorStore
    return LocalVectorStore()


def get_vector_store(backend: Optional[str] = None) -> "BaseVectorStore":
    """工厂函数：按配置返回 Milvus 或本地向量库的**进程级单例**。

    Args:
        backend: 显式指定后端（"local"/"milvus"），默认取 `RAG_CONFIG['vector_backend']`。
                 不同 backend 分别缓存单例，互不影响。
    """
    from rag.config import RAG_CONFIG
    backend = backend or RAG_CONFIG["vector_backend"]
    instance = _vector_store_instances.get(backend)
    if instance is None:
        with _vector_store_lock:
            instance = _vector_store_instances.get(backend)
            if instance is None:
                instance = _create_vector_store(backend)
                _vector_store_instances[backend] = instance
    return instance


def reset_vector_store(backend: Optional[str] = None) -> None:
    """重置向量库单例缓存（供测试隔离 / 运行时热切换配置使用）。

    Args:
        backend: 仅重置指定 backend 的缓存；为 None 时清空全部缓存。
    """
    with _vector_store_lock:
        if backend is None:
            _vector_store_instances.clear()
        else:
            _vector_store_instances.pop(backend, None)


# ======================================================================
# 关键词存储
# ======================================================================

class BaseKeywordStore(ABC):
    """关键词存储统一接口（对齐向量存储的解耦设计）。"""

    @abstractmethod
    def create_index(self) -> None:
        """创建/确保索引存在。"""

    @abstractmethod
    def upsert(self, blocks: List[DocBlock]) -> int:
        """写入（存在则覆盖）文档块，返回写入条数。"""

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> int:
        """按 doc_id 删除全部关联块，返回删除条数。"""

    @abstractmethod
    def search(self, query: str, top_k: int) -> List[DocBlock]:
        """关键词检索，返回按相关性降序排列的 DocBlock（含 score）。"""

    @abstractmethod
    def count(self) -> int:
        """当前索引中的文档块总数。"""

    @abstractmethod
    def health_check(self) -> bool:
        """检查后端服务是否可用。"""


_keyword_store_instances: Dict[str, "BaseKeywordStore"] = {}
_keyword_store_lock = threading.Lock()


def _create_keyword_store(backend: str) -> "BaseKeywordStore":
    if backend == "es":
        from rag.indexing.es_index import ESKeywordStore
        return ESKeywordStore()
    from rag.indexing.local_index import LocalKeywordStore
    return LocalKeywordStore()


def get_keyword_store(backend: Optional[str] = None) -> "BaseKeywordStore":
    """工厂函数：按配置返回 Elasticsearch 或本地关键词库的**进程级单例**。

    Args:
        backend: 显式指定后端（"local"/"es"），默认取 `RAG_CONFIG['keyword_backend']`。
    """
    from rag.config import RAG_CONFIG
    backend = backend or RAG_CONFIG["keyword_backend"]
    instance = _keyword_store_instances.get(backend)
    if instance is None:
        with _keyword_store_lock:
            instance = _keyword_store_instances.get(backend)
            if instance is None:
                instance = _create_keyword_store(backend)
                _keyword_store_instances[backend] = instance
    return instance


def reset_keyword_store(backend: Optional[str] = None) -> None:
    """重置关键词库单例缓存（供测试隔离 / 运行时热切换配置使用）。"""
    with _keyword_store_lock:
        if backend is None:
            _keyword_store_instances.clear()
        else:
            _keyword_store_instances.pop(backend, None)
