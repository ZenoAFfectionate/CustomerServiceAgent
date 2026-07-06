# -*- coding: utf-8 -*-
"""知识库文档元数据登记表（Metadata Registry，原 registry.py，重命名以对齐
新的模块命名规范 —— 该文件本质上就是"文档级元数据管理"）。

用于知识库管理接口（列表 / 删除 / 详情）与全局 `global_chunk_idx` 分配。
持久化在 `RAG_CONFIG['data_dir']/doc_registry.json`，独立于向量库/关键词库，
即使切换 `vector_backend`/`keyword_backend` 到真实服务，登记表逻辑不变。

`rag/knowledge_base/versioning.py` 在此基础上叠加了"内容版本历史"能力
（同一 doc_id 多次重新导入时的版本追踪），二者关注点不同：本模块只记录
"当前最新一份"的元信息，versioning.py 记录"历史所有版本"。
"""
import json
import os
import threading
import time
import uuid
from typing import List, Optional

from rag.config import RAG_CONFIG

# 【修复 M2】next_global_ids 此前仅用进程内 threading.Lock 保护内存字典，
# 不保护 counter.txt 的 read-modify-write，多进程/多 worker（如
# `uvicorn --workers N`）下会读到相同 counter 值并互相覆盖写，导致
# Milvus 主键冲突。Linux/macOS 上用 fcntl.flock 对 counter.txt 加独占
# 文件锁，使跨进程的 read-modify-write 也串行化；Windows 等不支持
# fcntl 的平台自动降级为仅进程内安全（与此前行为一致），并记录警告。
try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:  # pragma: no cover - Windows 等平台无 fcntl
    fcntl = None  # type: ignore
    _HAS_FCNTL = False


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

    def _bootstrap_counter(self) -> int:
        """【修复 N33】从 doc_registry.json 的 max(chunk_ids)+1 自举 counter，
        避免 counter.txt 丢失/置 0 后 ID 从 0 重分配导致 Milvus 主键冲突。
        """
        max_id = 0
        for doc_info in self._docs.values():
            chunk_ids = doc_info.get("chunk_ids", []) if isinstance(doc_info, dict) else []
            if chunk_ids:
                max_id = max(max_id, max(chunk_ids))
        return max_id + 1 if max_id > 0 else 0

    def next_global_ids(self, n: int) -> List[int]:
        """分配 n 个全局自增块 ID（跨文档唯一，用作 Milvus 主键）。

        `threading.Lock` 保护同进程内的并发；在支持 `fcntl` 的平台
        （Linux/macOS）上叠加对 `counter.txt` 的独占文件锁（flock），使
        多进程/多 worker 下的 read-modify-write 也串行化、不发生覆盖写
        （修复审查报告 M2）。不支持 `fcntl` 的平台仍降级为仅进程内安全。

        【修复 N33】若 counter.txt 被误删/置 0，此前会从 0 重分配 ID 与
        Milvus 已有主键冲突。现从 doc_registry.json 的 max(chunk_ids)+1
        自举，确保 counter 丢失后不会回退到已用过的 ID 区间。
        """
        with self._lock:
            os.makedirs(os.path.dirname(self._counter_path), exist_ok=True)
            # "a+" 模式：文件不存在时自动创建且不清空已有内容，同时允许随后
            # seek(0) 读取全文——配合显式 truncate 实现"读→改→写"的原子替换。
            with open(self._counter_path, "a+", encoding="utf-8") as f:
                if _HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    current = int((f.read() or "0").strip() or 0)
                    # 自举：若 counter 丢失/为 0，从 doc_registry.json 的
                    # max(chunk_ids)+1 恢复，避免主键回退冲突。
                    if current == 0:
                        current = self._bootstrap_counter()
                    ids = list(range(current, current + n))
                    f.seek(0)
                    f.truncate()
                    f.write(str(current + n))
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    if _HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
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

    注意（如实描述修复范围，对应审查报告 M2）：
    - `next_global_ids` 在 Linux/macOS（支持 `fcntl`）上已叠加跨进程文件锁，
      多 worker/多进程部署下的 ID 分配不再发生覆盖写；Windows 等不支持
      `fcntl` 的平台仍仅有**单进程内**的 `threading.Lock` 保护。
    - `register`/`delete` 等对 `doc_registry.json` 的写入**仍只有进程内
      `threading.Lock` 保护**，未加跨进程文件锁——多进程同时写文档登记表
      仍可能发生"读旧值→各自写"的丢失更新（影响范围小于主键冲突：登记表
      仅用于列表/删除展示，不影响向量库主键唯一性）。
    - 本地降级存储始终基于单文件 JSON，不具备真实数据库级别的事务保证；
      生产多副本部署仍推荐切换到真实 Milvus/ES 后端（写入由数据库自身
      保证并发安全）。
    """
    global _default_registry
    if _default_registry is None:
        with _registry_lock:
            if _default_registry is None:
                _default_registry = DocRegistry()
    return _default_registry
