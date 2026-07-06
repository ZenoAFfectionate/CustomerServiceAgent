# -*- coding: utf-8 -*-
"""
TEI (Text Embeddings Inference) 客户端。

封装与 TEI 服务的交互，提供 Embedding 和 Reranker 两种推理接口。

    TEI 服务部署方式：
    # Embedding 模型 (Qwen3-Embedding-4B)
    docker run --gpus all -p 8080:80 -v $PWD/data:/data \\
        ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \\
        --model-id Qwen/Qwen3-Embedding-4B

    # Reranker 模型 (Qwen3-Reranker-4B)
    docker run --gpus all -p 8081:80 -v $PWD/data:/data \\
        ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \\
        --model-id Qwen/Qwen3-Reranker-4B

用法：
    from model.inference.tei_client import TEIClient

    client = TEIClient(embed_url="http://localhost:8080", rerank_url="http://localhost:8081")

    # 获取嵌入向量
    embedding = client.embed("你好世界")

    # 批量嵌入
    embeddings = client.embed_batch(["文本1", "文本2"])

    # 重排序
    scores = client.rerank("查询问题", ["文档1", "文档2", "文档3"])
"""
import os
import sys
import threading  # [Optimized] 添加线程锁保护单例初始化，与项目中 embedder/registry/vector_store 的模式一致
import time
import requests
from typing import List, Optional

# 添加项目根目录到 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from config.config_loader import CONFIG, logger


class TEIClient:
    """TEI (Text Embeddings Inference) 服务客户端。

    封装 Embedding 和 Reranker 两种推理接口，
    支持 TEI 的 REST API 和动态批处理。
    """

    def __init__(
        self,
        embed_url: Optional[str] = None,
        rerank_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
    ):
        """初始化 TEI 客户端。

        Args:
            embed_url: Embedding 服务地址（如 http://localhost:8080）
            rerank_url: Reranker 服务地址（如 http://localhost:8081）
            timeout: 请求超时时间（秒）
            max_retries: 【修复 L19】网络/超时类瞬时故障的最大重试次数
                （不含首次请求）。此前 `embed`/`embed_batch`/`rerank` 仅调用
                `raise_for_status()`，未捕获网络超时/连接失败等瞬时异常，
                TEI 服务高负载下偶发超时会直接向上冒泡、无重试，而这类
                瞬时故障往往重试后即可恢复。
            retry_backoff_seconds: 重试的指数退避基准时间（秒），第 n 次重试
                等待 `retry_backoff_seconds * 2**(n-1)` 秒。
        """
        self.embed_url = embed_url or os.environ.get(
            "TEI_EMBED_URL", "http://localhost:8080"
        )
        self.rerank_url = rerank_url or os.environ.get(
            "TEI_RERANK_URL", "http://localhost:8081"
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _post_with_retry(self, url: str, payload: dict) -> requests.Response:
        """带指数退避重试的 POST 请求（修复 L19：TEI 调用无重试）。

        【修复 N19】此前仅对 Timeout/ConnectionError 重试；JSONDecodeError
        （body 非法）与可重试的瞬时 5xx（如 503）不在重试范围。现扩展为：
        - 网络类 RequestException（超时、连接失败、SSL 错误等）→ 重试
        - HTTP 5xx（服务端瞬时错误）→ 重试
        - JSONDecodeError（响应体非法）→ 重试
        - HTTP 4xx（客户端错误，请求本身有问题）→ 不重试，直接抛出
        """
        import json as _json

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout)
                # 5xx 可重试（服务端瞬时错误），4xx 不重试
                if 500 <= resp.status_code < 600:
                    raise requests.exceptions.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                # 提前验证 JSON 可解析（避免调用方拿到非法 body 后报 JSONDecodeError）
                _ = _json.loads(resp.text)
                return resp
            except (requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                    _json.JSONDecodeError) as e:
                last_exc = e
                if attempt >= self.max_retries:
                    break
                wait = self.retry_backoff_seconds * (2 ** attempt)
                logger.warning(
                    f"⚠️ TEI 请求瞬时失败（{url}，第 {attempt + 1}/{self.max_retries} 次重试，"
                    f"{wait:.1f}s 后重试): {e}"
                )
                time.sleep(wait)
            except requests.exceptions.HTTPError as e:
                # 5xx 重试，4xx 不重试
                status = e.response.status_code if e.response is not None else 0
                if 500 <= status < 600 and attempt < self.max_retries:
                    last_exc = e
                    wait = self.retry_backoff_seconds * (2 ** attempt)
                    logger.warning(
                        f"⚠️ TEI 服务端 {status} 错误（{url}，第 {attempt + 1}/{self.max_retries} 次重试，"
                        f"{wait:.1f}s 后重试)"
                    )
                    time.sleep(wait)
                    continue
                raise
        raise last_exc

    # ======================== Embedding 接口 ========================

    def embed(self, text: str) -> List[float]:
        """获取单条文本的嵌入向量。

        Args:
            text: 待嵌入的文本

        Returns:
            嵌入向量（浮点数列表）
        """
        resp = self._post_with_retry(f"{self.embed_url}/embed", {"inputs": text})
        return resp.json()[0]

    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """批量获取嵌入向量。

        利用 TEI 的动态批处理能力，自动分批发送。

        Args:
            texts: 待嵌入的文本列表
            batch_size: 每批最大文本数（TEI 默认限制 32）

        Returns:
            嵌入向量列表（与输入文本一一对应）
        """
        if not texts:
            return []
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self._post_with_retry(f"{self.embed_url}/embed", {"inputs": batch})
            all_embeddings.extend(resp.json())
        return all_embeddings

    # ======================== Reranker 接口 ========================

    def rerank(self, query: str, texts: List[str], return_text: bool = False) -> List[dict]:
        """对候选文档进行重排序。

        Args:
            query: 查询文本
            texts: 候选文档列表
            return_text: 是否在结果中包含原文

        Returns:
            排序后的结果列表，每个元素包含 index 和 score，
            按 score 降序排列
        """
        if not texts:
            # 【修复 L19】空候选列表时直接短路返回空结果，不向 TEI 发送空
            # texts 请求——此前依赖服务端如何处理空 texts（未定义行为，
            # 不同 TEI 版本可能报错或返回空数组）。
            return []
        resp = self._post_with_retry(
            f"{self.rerank_url}/rerank",
            {"query": query, "texts": texts, "return_text": return_text},
        )
        results = resp.json()
        # TEI 返回的结果已按 score 降序排列
        return results

    def rerank_scores(self, query: str, texts: List[str]) -> List[float]:
        """获取重排序分数（不重排，仅返回分数列表）。

        Args:
            query: 查询文本
            texts: 候选文档列表

        Returns:
            分数列表（与输入文档一一对应，不排序）
        """
        if not texts:
            return []
        results = self.rerank(query, texts)
        # 按 TEI 返回的 index 还原原始顺序。
        # 【修复 L19】此前直接 `item["index"]`/`item["score"]`，若 TEI 返回
        # 结构异常（缺字段、index 越界）会抛未加校验的 KeyError/IndexError，
        # 错误信息不包含上下文、难以定位。现做防御性校验并抛出更明确的错误。
        scores = [0.0] * len(texts)
        for item in results:
            if "index" not in item or "score" not in item:
                raise ValueError(f"TEI /rerank 响应缺少 index/score 字段: {item}")
            idx = item["index"]
            if not isinstance(idx, int) or not (0 <= idx < len(texts)):
                raise ValueError(f"TEI /rerank 响应中的 index 越界或非法: {idx}（texts 长度={len(texts)}）")
            scores[idx] = item["score"]
        return scores

    # ======================== 健康检查 ========================

    def health_check(self, service: str = "embed") -> bool:
        """检查 TEI 服务是否正常。

        Args:
            service: 检查的服务类型（"embed" 或 "rerank"）

        Returns:
            服务是否正常
        """
        url = self.embed_url if service == "embed" else self.rerank_url
        try:
            resp = self.session.get(f"{url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def get_model_info(self, service: str = "embed") -> dict:
        """获取 TEI 服务加载的模型信息。

        Args:
            service: 服务类型（"embed" 或 "rerank"）

        Returns:
            模型信息字典
        """
        url = self.embed_url if service == "embed" else self.rerank_url
        resp = self.session.get(f"{url}/info", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    # ======================== 资源管理 ========================

    def close(self):
        """关闭 HTTP 会话。"""
        self.session.close()


# ======================== 全局单例 ========================

_client: Optional[TEIClient] = None
_client_lock = threading.Lock()  # [Optimized] 双重检查锁保护并发首次调用，避免重复创建客户端


def get_tei_client() -> TEIClient:
    """获取全局 TEI 客户端单例。"""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = TEIClient()
                logger.info(f"TEI 客户端初始化: embed={_client.embed_url}, rerank={_client.rerank_url}")
    return _client