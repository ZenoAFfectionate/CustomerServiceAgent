# -*- coding: utf-8 -*-
"""关键词存储抽象层（对齐向量存储的解耦设计）。

`retrieval/es_search.py` 与 pipeline 只依赖 `BaseKeywordStore` 接口，
不关心具体是真实 Elasticsearch 还是本地 TF-IDF 降级实现。

【优化点】单例缓存：原因与方案同 `vector_store.py`（见其注释），
避免本地后端全量重复加载 JSON、以及 ES 后端重复建立 HTTP 客户端连接。
"""
import threading
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from rag.schema import DocBlock


class BaseKeywordStore(ABC):
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


_store_instances: Dict[str, "BaseKeywordStore"] = {}
_store_lock = threading.Lock()


def _create_store(backend: str) -> "BaseKeywordStore":
    if backend == "es":
        from rag.indexing.es_index import ESKeywordStore
        return ESKeywordStore()
    from rag.indexing.local_keyword_index import LocalKeywordStore
    return LocalKeywordStore()


def get_keyword_store(backend: Optional[str] = None) -> "BaseKeywordStore":
    """工厂函数：按配置返回 Elasticsearch 或本地关键词库的**进程级单例**。

    Args:
        backend: 显式指定后端（"local"/"es"），默认取 `RAG_CONFIG['keyword_backend']`。
    """
    from rag.config import RAG_CONFIG
    backend = backend or RAG_CONFIG["keyword_backend"]
    instance = _store_instances.get(backend)
    if instance is None:
        with _store_lock:
            instance = _store_instances.get(backend)
            if instance is None:
                instance = _create_store(backend)
                _store_instances[backend] = instance
    return instance


def reset_keyword_store(backend: Optional[str] = None) -> None:
    """重置关键词库单例缓存（供测试隔离 / 运行时热切换配置使用）。"""
    with _store_lock:
        if backend is None:
            _store_instances.clear()
        else:
            _store_instances.pop(backend, None)
