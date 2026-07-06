# -*- coding: utf-8 -*-
"""文本向量化（Embedding，原 embedder.py，重命名以对齐新的模块命名规范）。

后端可通过 `RAG_CONFIG["embed_backend"]` 切换：
    - "tei":   对接 `model/inference/tei_client.py`，调用 TEI Embedding 服务（生产）
    - "local": 无需外部服务的本地哈希嵌入（Feature Hashing + TF 加权），
               保证开箱即用与单测无外部依赖；语义精度低于真实 Embedding 模型，
               仅用于演示/降级场景。

统一接口：`Embedder.embed_texts(texts) -> List[List[float]]`，
上层（indexing / retrieval）只依赖该接口，可无缝切换后端。
"""
import hashlib
import math
import re
import threading
from typing import List, Optional

from config.config_loader import logger
from rag.config import RAG_CONFIG

try:
    import jieba
except ImportError:  # pragma: no cover
    jieba = None

LOCAL_EMBED_DIM = 256

_WORD_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """轻量分词：中文按单字，英文/数字按连续片段（jieba 可用时优先用 jieba）。"""
    if not text:
        return []
    if jieba is not None:
        try:
            return [w for w in jieba.lcut(text) if w.strip()]
        except Exception:
            pass
    return _WORD_RE.findall(text)


def local_hash_embedding(text: str, dim: int = LOCAL_EMBED_DIM) -> List[float]:
    """基于特征哈希（Hashing Trick）的确定性本地向量化。

    原理：对每个 token 计算哈希值映射到 [0, dim) 维度，累加词频（TF）作为该维度权重，
    最后做 L2 归一化。无需训练、无需外部服务，相同文本永远得到相同向量，
    且天然支持增量写入（无需像 TF-IDF 一样重新拟合词表）。

    Args:
        text: 输入文本
        dim: 向量维度

    Returns:
        长度为 dim 的浮点向量（L2 归一化后）
    """
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


class Embedder:
    """统一 Embedding 接口，封装 TEI / 本地降级两种后端。"""

    def __init__(self, backend: Optional[str] = None, dim: int = LOCAL_EMBED_DIM):
        self.backend = backend or RAG_CONFIG["embed_backend"]
        self.dim = dim
        self._tei_client = None
        self._client_lock = threading.Lock()  # 保护懒加载 TEI 客户端的初始化，避免并发首次请求重复创建

    def _get_tei_client(self):
        if self._tei_client is None:
            with self._client_lock:
                if self._tei_client is None:
                    from model.inference.tei_client import get_tei_client
                    self._tei_client = get_tei_client()
        return self._tei_client

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文本。TEI 服务不可用时自动降级为本地哈希嵌入并告警。"""
        if not texts:
            return []
        if self.backend == "tei":
            try:
                client = self._get_tei_client()
                if not client.health_check("embed"):
                    raise RuntimeError("TEI embed 服务健康检查未通过")
                return client.embed_batch(texts)
            except Exception as e:
                logger.warning(f"⚠️ TEI Embedding 不可用（{e}），降级为本地哈希嵌入")
        return [local_hash_embedding(t, self.dim) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def get_dim(self) -> int:
        return self.dim


_default_embedder: Optional[Embedder] = None
_embedder_lock = threading.Lock()


def get_embedder() -> Embedder:
    """获取全局默认 Embedder 单例（读取 RAG_CONFIG 决定后端）。"""
    global _default_embedder
    if _default_embedder is None:
        with _embedder_lock:
            if _default_embedder is None:
                _default_embedder = Embedder()
    return _default_embedder
