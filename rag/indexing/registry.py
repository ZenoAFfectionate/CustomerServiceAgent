# -*- coding: utf-8 -*-
"""知识库文档元数据登记表（Document Registry）。

用于知识库管理接口（列表 / 删除 / 详情）与全局 `global_chunk_idx` 分配。
持久化在 `RAG_CONFIG['data_dir']/doc_registry.json`，独立于向量库/关键词库，
即使切换 `vector_backend`/`keyword_backend` 到真实服务，登记表逻辑不变。
"""
import json
import os
import threading
import time
import uuid
from typing import List, Optional

from rag.config import RAG_CONFIG


class DocRegistry:
    def __init__(self, path: str = None):
        self.path = path or os.path.join(RAG_CONFIG["data_dir"], "doc_registry.json")
        self._counter_path = os.path.join(RAG_CONFIG["data_dir"], "counter.txt")
        self._lock = threading.Lock()
        self._docs: dict = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._docs, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

    def next_global_ids(self, n: int) -> List[int]:
        """分配 n 个全局自增块 ID（跨文档唯一，用作 Milvus 主键）。"""
        with self._lock:
            os.makedirs(os.path.dirname(self._counter_path), exist_ok=True)
            current = 0
            if os.path.exists(self._counter_path):
                with open(self._counter_path, "r", encoding="utf-8") as f:
                    current = int((f.read() or "0").strip() or 0)
            ids = list(range(current, current + n))
            with open(self._counter_path, "w", encoding="utf-8") as f:
                f.write(str(current + n))
            return ids

    def new_doc_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def register(self, doc_id: str, filename: str, num_chunks: int, chunk_ids: List[int], source: str = "upload") -> dict:
        with self._lock:
            meta = {
                "doc_id": doc_id,
                "filename": filename,
                "source": source,
                "num_chunks": num_chunks,
                "chunk_ids": chunk_ids,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._docs[doc_id] = meta
            self._save()
            return meta

    def list_documents(self) -> List[dict]:
        return list(self._docs.values())

    def get(self, doc_id: str) -> Optional[dict]:
        return self._docs.get(doc_id)

    def delete(self, doc_id: str) -> bool:
        with self._lock:
            if doc_id not in self._docs:
                return False
            del self._docs[doc_id]
            self._save()
            return True

    def count_docs(self) -> int:
        return len(self._docs)


_default_registry: Optional[DocRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> DocRegistry:
    """获取全局默认 DocRegistry 单例。

    注意：`next_global_ids`/`register` 等写操作仅通过实例内部的
    `threading.Lock` 保证**单进程内**线程安全；本地降级存储基于单文件 JSON，
    不具备跨进程/多副本部署下的并发安全性 —— 若需要多 worker/多实例部署，
    请切换到真实 Milvus/ES 后端（其写入天然由数据库自身保证并发安全）。
    """
    global _default_registry
    if _default_registry is None:
        with _registry_lock:
            if _default_registry is None:
                _default_registry = DocRegistry()
    return _default_registry
