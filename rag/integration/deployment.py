# -*- coding: utf-8 -*-
"""部署就绪检查（Deployment Readiness）：聚合各组件（向量库/关键词库/
Embedding/Reranker/生成）的 `health_check()`，在切换到生产后端
（Milvus/ES/TEI/vLLM）前快速确认服务是否就绪，是
`scripts/run_RAGserver.sh`/`scripts/start_tei.sh` 的补充自检工具。
"""
from typing import Optional

from rag.config import RAG_CONFIG


def check_deployment_readiness() -> dict:
    """逐项检查当前配置的后端是否可用，返回聚合报告。

    Returns:
        {"ready": bool, "checks": [{"component", "backend", "healthy", "detail"}, ...]}
    """
    checks = []

    def _add(component: str, backend: str, healthy: bool, detail: str = ""):
        checks.append({"component": component, "backend": backend, "healthy": healthy, "detail": detail})

    # 向量库
    try:
        from rag.indexing.store import get_vector_store
        store = get_vector_store()
        _add("vector_store", RAG_CONFIG["vector_backend"], store.health_check())
    except Exception as e:
        _add("vector_store", RAG_CONFIG["vector_backend"], False, str(e))

    # 关键词库
    try:
        from rag.indexing.store import get_keyword_store
        store = get_keyword_store()
        _add("keyword_store", RAG_CONFIG["keyword_backend"], store.health_check())
    except Exception as e:
        _add("keyword_store", RAG_CONFIG["keyword_backend"], False, str(e))

    # Embedding（仅 tei 后端需要检查外部服务；local 后端天然可用）
    embed_backend = RAG_CONFIG["embed_backend"]
    if embed_backend == "tei":
        try:
            from model.inference.tei_client import get_tei_client
            healthy = get_tei_client().health_check("embed")
            _add("embedder", "tei", healthy)
        except Exception as e:
            _add("embedder", "tei", False, str(e))
    else:
        _add("embedder", "local", True)

    # Reranker
    rerank_backend = RAG_CONFIG["rerank_backend"]
    if rerank_backend == "tei":
        try:
            from model.inference.tei_client import get_tei_client
            healthy = get_tei_client().health_check("rerank")
            _add("reranker", "tei", healthy)
        except Exception as e:
            _add("reranker", "tei", False, str(e))
    else:
        _add("reranker", "local", True)

    # 生成（vllm 后端需检查 API 是否可达；local 抽取式天然可用）
    gen_backend = RAG_CONFIG["generation_backend"]
    if gen_backend == "vllm":
        try:
            import requests
            from urllib.parse import urlsplit, urlunsplit
            from config.config_loader import CONFIG

            # 【健壮性修复】此前用 `rsplit("/chat/completions", 1)[0] + "/models"` 猜测
            # 模型列表接口路径，当 vllm_api_url 不包含该确切子串（如带查询参数/末尾
            # 斜杠/自定义网关前缀）时会拼出错误 URL。这里改为只探测服务的 scheme+host
            # 是否可达（GET 根路径），不依赖具体 API 路径结构 —— 只要能建立 HTTP
            # 连接并收到任意响应（即使 404），就说明服务进程本身是活的；连接失败
            # （超时/拒绝连接）才判定为不可达。
            parsed = urlsplit(CONFIG.get("vllm_api_url", ""))
            base_url = urlunsplit((parsed.scheme, parsed.netloc, "/", "", ""))
            resp = requests.get(base_url, timeout=5)
            _add("generation", "vllm", resp.status_code < 500)
        except Exception as e:
            _add("generation", "vllm", False, str(e))
    else:
        _add("generation", "local", True)

    return {"ready": all(c["healthy"] for c in checks), "checks": checks}


def print_checklist() -> None:
    """以人类可读的形式打印部署就绪检查结果（CLI 用）。"""
    report = check_deployment_readiness()
    print("=" * 60)
    print(f"部署就绪检查：{'✅ 全部就绪' if report['ready'] else '⚠️ 存在未就绪组件'}")
    print("=" * 60)
    for c in report["checks"]:
        status = "✅" if c["healthy"] else "❌"
        detail = f"（{c['detail']}）" if c["detail"] else ""
        print(f"  {status} {c['component']:16s} backend={c['backend']:8s} {detail}")


if __name__ == "__main__":
    print_checklist()
