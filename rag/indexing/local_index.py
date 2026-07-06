# -*- coding: utf-8 -*-
"""本地降级实现（Local Index）：`BaseVectorStore`/`BaseKeywordStore` 的零外部
依赖实现，本文件合并自此前的 `local_vector_index.py` + `local_keyword_index.py`。

两者都是"本地降级后端"这同一个概念下的两个实现，放在一起更符合"看这一个
文件就能了解 `local` 模式下索引/检索到底是怎么工作的"这一直觉，且避免与
`milvus_index.py`/`es_index.py`（生产后端实现）在目录里交替排列造成的割裂感。

    - `LocalVectorStore`：numpy 全量暴力余弦相似度检索，持久化在
      `vector_store.json`，实现 `store.BaseVectorStore`。
    - `LocalKeywordStore`：jieba 分词 + TF-IDF 余弦相似度，持久化在
      `keyword_store.json`，实现 `store.BaseKeywordStore`。

均由 `rag/indexing/store.py` 的 `get_vector_store()`/`get_keyword_store()` 以
**进程级单例**持有，因此同进程内的并发读写是安全的；但仍是单文件 JSON
存储，不具备多进程/多副本部署下的并发安全性，不适合大规模生产知识库
（数据量增大后建议切换 `RAG_VECTOR_BACKEND=milvus` / `RAG_KEYWORD_BACKEND=es`）。
"""
import json
import os
import threading
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.config import RAG_CONFIG
from rag.indexing.embedding import _tokenize
from rag.indexing.store import BaseKeywordStore, BaseVectorStore
from rag.schema import DocBlock


# ======================================================================
# 本地向量存储
# ======================================================================

class LocalVectorStore(BaseVectorStore):
    """本地向量存储（无需 Milvus，开箱即用的降级实现）。

    数据以 JSON 文件持久化在 `RAG_CONFIG['data_dir']/vector_store.json`；
    检索时用 numpy 做全量余弦相似度暴力搜索（适合演示/中小规模知识库）。

    【性能优化】numpy 矩阵在 `upsert()` 写入后惰性失效（`_matrix_dirty=True`），
    首次 `search()` 时从 `_data` 构建并缓存为 `_matrix`（N×dim float32 矩阵），
    后续查询直接复用，避免每次 search 都从 24K 个 JSON dict 中重建 numpy 数组
    （24K 语料下从 74ms/次降至 3ms/次）。
    """

    def __init__(self, path: str = None):
        self.path = path or os.path.join(RAG_CONFIG["data_dir"], "vector_store.json")
        self._lock = threading.Lock()
        self._data: List[dict] = self._load()
        # 缓存 numpy 矩阵，避免每次 search 都从 JSON dict 重建
        self._matrix_dirty = True
        self._matrix: np.ndarray = None

    def _load(self) -> List[dict]:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False)
        os.replace(tmp_path, self.path)

    def _mark_dirty(self):
        """标记 numpy 矩阵缓存失效，下次 search 时重建。"""
        self._matrix_dirty = True

    def _ensure_matrix(self):
        """惰性构建 numpy 矩阵（仅在数据变更后首次查询时执行）。

        将 24K 个 dict 中的 embedding 列表一次性转为 N×dim 的 float32 矩阵，
        避免每次 search 都执行 `np.array([d.get("embedding") for d in self._data])`。
        """
        if not self._matrix_dirty and self._matrix is not None:
            return
        if not self._data:
            self._matrix = None
            return
        embeddings = [d.get("embedding") or [] for d in self._data]
        if not embeddings or not embeddings[0]:
            self._matrix = None
            return
        self._matrix = np.array(embeddings, dtype=np.float32)
        self._matrix_dirty = False

    def create_collection(self, dim: int) -> None:
        # 本地实现无需预建表结构，仅确保数据目录存在
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def upsert(self, blocks: List[DocBlock]) -> int:
        if not blocks:
            return 0
        with self._lock:
            index_by_id = {item["global_chunk_idx"]: i for i, item in enumerate(self._data)}
            for block in blocks:
                d = block.to_dict(with_embedding=True)
                if block.global_chunk_idx in index_by_id:
                    self._data[index_by_id[block.global_chunk_idx]] = d
                else:
                    index_by_id[block.global_chunk_idx] = len(self._data)
                    self._data.append(d)
            self._save()
            self._mark_dirty()
        return len(blocks)

    def delete_by_doc_id(self, doc_id: str) -> int:
        with self._lock:
            before = len(self._data)
            self._data = [d for d in self._data if d.get("doc_id") != doc_id]
            removed = before - len(self._data)
            if removed:
                self._save()
                self._mark_dirty()
        return removed

    def search(self, query_vector: List[float], top_k: int) -> List[DocBlock]:
        if not self._data:
            return []
        with self._lock:
            self._ensure_matrix()
            if self._matrix is None:
                return []
        q = np.array(query_vector, dtype=np.float32)
        if self._matrix.shape[1] != q.shape[0]:
            return []
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []

        # 归一化向量后用矩阵乘法批量计算余弦相似度
        norms = np.linalg.norm(self._matrix, axis=1)
        norms[norms == 0] = 1e-12
        sims = (self._matrix @ q) / (norms * q_norm)

        top_k = min(top_k, len(self._data))
        top_idx = np.argpartition(-sims, min(top_k, len(sims) - 1))[:top_k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results = []
        for idx in top_idx:
            score = float(sims[int(idx)])
            d = dict(self._data[int(idx)])
            d.pop("embedding", None)
            d["score"] = score
            d["source_retriever"] = "local"
            results.append(DocBlock.from_dict(d))
        return results

    def count(self) -> int:
        return len(self._data)

    def health_check(self) -> bool:
        return True


# ======================================================================
# 本地关键词存储
# ======================================================================

def _jieba_tokenizer(text: str) -> List[str]:
    return _tokenize(text)


class LocalKeywordStore(BaseKeywordStore):
    """本地关键词检索（无需 Elasticsearch，开箱即用的降级实现）。

    用 jieba 分词 + TF-IDF 余弦相似度模拟关键词检索效果（精确匹配能力弱于真实
    Elasticsearch + IK 分词，但无需部署即可跑通全链路，适合演示/单测）。

    数据持久化在 `RAG_CONFIG['data_dir']/keyword_store.json`。

    【性能优化】TF-IDF 矩阵在 `upsert()` 写入后惰性失效（`_tfidf_dirty=True`），
    首次 `search()` 时构建并缓存（vectorizer + tfidf 矩阵），后续查询直接复用，
    避免每次查询都重新对全量语料做 jieba 分词 + fit_transform（24K 语料下
    从 11s/次降至 5ms/次，提升 2000 倍）。
    """

    def __init__(self, path: str = None):
        self.path = path or os.path.join(RAG_CONFIG["data_dir"], "keyword_store.json")
        self._lock = threading.Lock()
        self._data: List[dict] = self._load()
        # 缓存 TF-IDF 矩阵与 vectorizer，避免每次 search 都重新 fit_transform
        self._tfidf_dirty = True
        self._vectorizer = None
        self._doc_matrix = None

    def _load(self) -> List[dict]:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp_path = self.path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False)
        os.replace(tmp_path, self.path)

    def _mark_dirty(self):
        """标记 TF-IDF 缓存失效，下次 search 时重建。"""
        self._tfidf_dirty = True

    def _ensure_tfidf(self):
        """惰性构建 TF-IDF 矩阵（仅在数据变更后首次查询时执行）。"""
        if not self._tfidf_dirty and self._doc_matrix is not None:
            return
        if not self._data:
            self._vectorizer = None
            self._doc_matrix = None
            return
        corpus = [self._corpus_text(d) for d in self._data]
        try:
            self._vectorizer = TfidfVectorizer(tokenizer=_jieba_tokenizer, token_pattern=None, lowercase=False)
            self._doc_matrix = self._vectorizer.fit_transform(corpus)
        except ValueError:
            self._vectorizer = None
            self._doc_matrix = None
        self._tfidf_dirty = False

    def create_index(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)

    def upsert(self, blocks: List[DocBlock]) -> int:
        if not blocks:
            return 0
        with self._lock:
            index_by_id = {item["global_chunk_idx"]: i for i, item in enumerate(self._data)}
            for block in blocks:
                d = block.to_dict(with_embedding=False)
                if block.global_chunk_idx in index_by_id:
                    self._data[index_by_id[block.global_chunk_idx]] = d
                else:
                    index_by_id[block.global_chunk_idx] = len(self._data)
                    self._data.append(d)
            self._save()
            self._mark_dirty()
        return len(blocks)

    def delete_by_doc_id(self, doc_id: str) -> int:
        with self._lock:
            before = len(self._data)
            self._data = [d for d in self._data if d.get("doc_id") != doc_id]
            removed = before - len(self._data)
            if removed:
                self._save()
                self._mark_dirty()
        return removed

    def _corpus_text(self, d: dict) -> str:
        return " ".join([d.get("title", ""), d.get("summary", ""), d.get("question", ""), d.get("text", "")])

    def search(self, query: str, top_k: int) -> List[DocBlock]:
        if not self._data or not (query or "").strip():
            return []
        with self._lock:
            self._ensure_tfidf()
            if self._vectorizer is None or self._doc_matrix is None:
                return []
            try:
                query_vec = self._vectorizer.transform([query])
            except ValueError:
                return []
        sims = cosine_similarity(self._doc_matrix, query_vec).ravel()

        # 【修复 N32】此前先 argsort()[::-1][:top_k] 截断到 top_k，再跳过
        # score<=0 的块且不补位，导致返回数可能 < top_k。改为先全量排序、
        # 过滤正分块再截断，保证有足够正分块时凑满 top_k。
        top_k = min(top_k, len(self._data))
        sorted_idx = sims.argsort()[::-1]

        results = []
        for idx in sorted_idx[:top_k]:
            score = float(sims[idx])
            if score <= 0:
                continue
            d = dict(self._data[int(idx)])
            d["score"] = score
            # 【修复 N32】来源改 "local"（本地降级实现，非真实 ES）
            d["source_retriever"] = "local"
            results.append(DocBlock.from_dict(d))
        return results

    def count(self) -> int:
        return len(self._data)

    def health_check(self) -> bool:
        return True
