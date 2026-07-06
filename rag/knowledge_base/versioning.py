# -*- coding: utf-8 -*-
"""文档版本历史（Versioning）：追踪同一 `doc_id` 多次重新导入时的内容变化。

与 `indexing/metadata.py` 的 `DocRegistry` 关注点不同：`DocRegistry` 只保存
"当前最新一份"的元信息（供列表/删除接口使用），本模块保存**历史全部版本**
（每次导入的内容哈希 + 时间 + 块数），供 `update_sync.py` 判断"内容是否变化"
以及未来审计/回溯之用。持久化在 `RAG_CONFIG['data_dir']/versions.json`。
"""
import hashlib
import json
import os
import threading
import time
from typing import List, Optional

from rag.config import RAG_CONFIG


def compute_content_hash(blocks_or_text) -> str:
    """对知识块数组 / 纯文本 / 原始二进制内容计算哈希（用于判断内容是否变化）。

    支持三种输入：
        - bytes：直接对原始字节计算哈希（如上传的 PDF/图片等二进制文件，
          避免先 decode 为字符串再哈希导致的信息损失或不必要的转换开销）
        - str：直接对字符串编码后哈希
        - 其他（如 List[dict] 知识块数组）：先转为确定性 JSON 字符串再哈希
          （`sort_keys=True` 保证字段顺序不同但内容相同时哈希一致）
    """
    if isinstance(blocks_or_text, bytes):
        payload = blocks_or_text
    elif isinstance(blocks_or_text, str):
        payload = blocks_or_text.encode("utf-8")
    else:
        payload = json.dumps(blocks_or_text, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


# 【修复 L14】计算版本哈希时应排除每次导入都会变化的元数据字段
# （global_chunk_idx/doc_id/source/score/source_retriever/embedding），
# 只保留内容字段，使同一内容的哈希在多次导入间稳定可比较。
_META_FIELDS = frozenset({
    "global_chunk_idx", "doc_id", "source", "score", "source_retriever", "embedding",
})


def compute_blocks_content_hash(blocks: list) -> str:
    """对知识块列表计算内容哈希（排除元数据字段，保证跨导入稳定）。

    供 corpus_management.ingest_upload/ingest_blocks 和 update_sync.sync_directory
    共用同一套哈希计算逻辑，避免各处自行实现导致哈希空间不一致。
    """
    stripped = [
        {k: v for k, v in b.items() if k not in _META_FIELDS}
        for b in blocks
    ]
    return compute_content_hash(stripped)


class VersionStore:
    # 【修复 L12】每个 doc_id 最多保留的历史版本数，避免 versions.json 随
    # 导入次数无限增长后每次 record_version 都全量重写大文件（写放大）。
    MAX_HISTORY_PER_DOC = 50

    def __init__(self, path: str = None):
        self.path = path or os.path.join(RAG_CONFIG["data_dir"], "versions.json")
        self._lock = threading.Lock()
        self._history: dict = self._load()  # doc_id -> [{"hash", "num_chunks", "created_at"}, ...]

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

    def record_version(self, doc_id: str, content_hash: str, num_chunks: int, filename: str = "") -> dict:
        """记录一次新版本（追加到该 doc_id 的历史列表末尾）。

        【修复 L12】每个 doc_id 的历史保留最近 MAX_HISTORY_PER_DOC 条，
        超出时丢弃最旧的版本，避免 versions.json 无限增长导致每次导入都
        全量重写大文件（写放大）。
        """
        with self._lock:
            entry = {
                "hash": content_hash, "num_chunks": num_chunks, "filename": filename,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            hist = self._history.setdefault(doc_id, [])
            hist.append(entry)
            # 保留最近 N 条，丢弃最旧的
            if len(hist) > self.MAX_HISTORY_PER_DOC:
                del hist[: len(hist) - self.MAX_HISTORY_PER_DOC]
            self._save()
            return entry

    def get_history(self, doc_id: str) -> List[dict]:
        return list(self._history.get(doc_id, []))

    def get_latest_hash(self, doc_id: str) -> Optional[str]:
        history = self._history.get(doc_id) or []
        return history[-1]["hash"] if history else None

    def clear(self, doc_id: str) -> None:
        with self._lock:
            self._history.pop(doc_id, None)
            self._save()


_default_store: Optional[VersionStore] = None
_store_lock = threading.Lock()


def get_version_store() -> VersionStore:
    global _default_store
    if _default_store is None:
        with _store_lock:
            if _default_store is None:
                _default_store = VersionStore()
    return _default_store


def reset_version_store() -> None:
    """重置全局 VersionStore 单例（供测试隔离使用，与 reset_vector_store/
    reset_keyword_store 保持一致的命名与用法约定）。"""
    global _default_store
    with _store_lock:
        _default_store = None
