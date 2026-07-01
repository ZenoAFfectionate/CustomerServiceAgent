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
    ):
        """初始化 TEI 客户端。

        Args:
            embed_url: Embedding 服务地址（如 http://localhost:8080）
            rerank_url: Reranker 服务地址（如 http://localhost:8081）
            timeout: 请求超时时间（秒）
        """
        self.embed_url = embed_url or os.environ.get(
            "TEI_EMBED_URL", "http://localhost:8080"
        )
        self.rerank_url = rerank_url or os.environ.get(
            "TEI_RERANK_URL", "http://localhost:8081"
        )
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ======================== Embedding 接口 ========================

    def embed(self, text: str) -> List[float]:
        """获取单条文本的嵌入向量。

        Args:
            text: 待嵌入的文本

        Returns:
            嵌入向量（浮点数列表）
        """
        resp = self.session.post(
            f"{self.embed_url}/embed",
            json={"inputs": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
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
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self.session.post(
                f"{self.embed_url}/embed",
                json={"inputs": batch},
                timeout=self.timeout,
            )
            resp.raise_for_status()
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
        resp = self.session.post(
            f"{self.rerank_url}/rerank",
            json={
                "query": query,
                "texts": texts,
                "return_text": return_text,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
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
        results = self.rerank(query, texts)
        # 按 TEI 返回的 index 还原原始顺序
        scores = [0.0] * len(texts)
        for item in results:
            scores[item["index"]] = item["score"]
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


def get_tei_client() -> TEIClient:
    """获取全局 TEI 客户端单例。"""
    global _client
    if _client is None:
        _client = TEIClient()
        logger.info(f"TEI 客户端初始化: embed={_client.embed_url}, rerank={_client.rerank_url}")
    return _client