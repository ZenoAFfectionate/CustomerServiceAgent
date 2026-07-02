# -*- coding: utf-8 -*-
"""RAG 专属配置。

读取 `config/config.json` 的 `env_config`（milvus_host/es_host/collection_name/
index_name），并新增 RAG 链路专属配置项：召回数、精排数、融合方式与权重、
各组件的后端选择（真实服务 vs 本地降级实现）。

配置优先级：环境变量 > `.env` > `config/config.json` > 代码默认值。

用法：
    from rag.config import RAG_CONFIG
    print(RAG_CONFIG["top_k_recall"])
"""
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_CURRENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config.config_loader import CONFIG, logger  # noqa: E402


def _env_or_config(key: str, config: dict, default=None, cast=str):
    """从环境变量或 config 字典中读取值，环境变量优先。"""
    env_val = os.environ.get(key)
    if env_val is not None and env_val != "":
        try:
            if cast is bool:
                return env_val.strip().lower() in ("1", "true", "yes", "on")
            return cast(env_val)
        except (ValueError, TypeError):
            return env_val
    val = config.get(key, default)
    if val is not None and cast != str:
        try:
            return cast(val)
        except (ValueError, TypeError):
            pass
    return val


# ======================== dev/prod 环境选择 ========================
_ENV_NAME = _env_or_config("RAG_ENV", CONFIG, CONFIG.get("env_default", "dev"))
_ENV_CFG = CONFIG.get("env_config", {}).get(_ENV_NAME, {})

# ======================== 数据目录（本地降级后端持久化位置） ========================
RAG_DATA_DIR = _env_or_config("RAG_DATA_DIR", CONFIG, os.path.join(_CURRENT_DIR, "data"))
os.makedirs(RAG_DATA_DIR, exist_ok=True)

RAG_CONFIG = {
    # -------- 环境 --------
    "env": _ENV_NAME,
    "data_dir": RAG_DATA_DIR,

    # -------- Milvus --------
    "milvus_host": _env_or_config(f"MILVUS_HOST_{_ENV_NAME.upper()}", CONFIG, _ENV_CFG.get("milvus_host", "127.0.0.1")),
    "milvus_port": _env_or_config("MILVUS_PORT", CONFIG, 19530, int),
    "milvus_user": os.environ.get("MILVUS_USER", ""),
    "milvus_password": os.environ.get("MILVUS_PASSWORD", ""),
    "collection_name": _env_or_config(f"MILVUS_COLLECTION_{_ENV_NAME.upper()}", CONFIG, _ENV_CFG.get("collection_name", f"htmlrag_{_ENV_NAME}")),

    # -------- Elasticsearch --------
    "es_host": _env_or_config(f"ES_HOST_{_ENV_NAME.upper()}", CONFIG, _ENV_CFG.get("es_host", "127.0.0.1")),
    "es_port": _env_or_config("ES_PORT", CONFIG, 9200, int),
    "es_user": os.environ.get("ES_USER", ""),
    "es_password": os.environ.get("ES_PASSWORD", ""),
    "index_name": _env_or_config(f"ES_INDEX_{_ENV_NAME.upper()}", CONFIG, _ENV_CFG.get("index_name", f"htmlrag_{_ENV_NAME}")),

    # -------- 检索参数 --------
    "top_k_recall": _env_or_config("RAG_TOP_K_RECALL", CONFIG, 20, int),      # 单路召回数
    "top_k_final": _env_or_config("RAG_TOP_K_FINAL", CONFIG, 5, int),        # 精排后返回数
    "fusion_method": _env_or_config("RAG_FUSION_METHOD", CONFIG, "rrf", str),  # rrf / weighted
    "fusion_rrf_k": _env_or_config("RAG_FUSION_RRF_K", CONFIG, 60, int),
    "fusion_weights": {
        "milvus": _env_or_config("RAG_FUSION_WEIGHT_MILVUS", CONFIG, 0.5, float),
        "es": _env_or_config("RAG_FUSION_WEIGHT_ES", CONFIG, 0.5, float),
    },
    "dedup_threshold_content": _env_or_config("RAG_DEDUP_THRESHOLD_CONTENT", CONFIG, 0.9, float),
    "dedup_threshold_page_name": _env_or_config("RAG_DEDUP_THRESHOLD_PAGE_NAME", CONFIG, 0.6, float),

    # -------- 分块参数（通用文档解析，process/ 专用 Block Tree 分块不受影响） --------
    "chunk_size": _env_or_config("RAG_CHUNK_SIZE", CONFIG, 500, int),
    "chunk_overlap": _env_or_config("RAG_CHUNK_OVERLAP", CONFIG, 50, int),

    # -------- 后端选择：real（TEI/Milvus/ES/vLLM 真实服务）或 local（本地降级实现） --------
    # 默认 local，保证无外部服务时开箱即用；生产环境通过环境变量切至 real。
    "vector_backend": _env_or_config("RAG_VECTOR_BACKEND", CONFIG, "local", str),     # local / milvus
    "keyword_backend": _env_or_config("RAG_KEYWORD_BACKEND", CONFIG, "local", str),   # local / es
    "embed_backend": _env_or_config("RAG_EMBED_BACKEND", CONFIG, "local", str),       # local / tei
    "rerank_backend": _env_or_config("RAG_RERANK_BACKEND", CONFIG, "local", str),     # local / tei
    "generation_backend": _env_or_config("RAG_GENERATION_BACKEND", CONFIG, "local", str),  # local / vllm

    # -------- 生成 --------
    "generation_max_context_chars": _env_or_config("RAG_GEN_MAX_CONTEXT_CHARS", CONFIG, 3000, int),
    "generation_max_new_tokens": _env_or_config("RAG_GEN_MAX_NEW_TOKENS", CONFIG, 512, int),
    # 抽取式兜底回答中，单条上下文摘录片段的最大字符数（超出截断并加 "..."）
    "generation_snippet_max_chars": _env_or_config("RAG_GEN_SNIPPET_MAX_CHARS", CONFIG, 200, int),

    # -------- 精排 --------
    # Reranker 输入文档拼接（title+summary+text）后的截断长度，避免超出模型 max_length
    "rerank_doc_max_chars": _env_or_config("RAG_RERANK_DOC_MAX_CHARS", CONFIG, 2000, int),

    # -------- 上传 --------
    "upload_max_size_mb": _env_or_config("RAG_UPLOAD_MAX_SIZE_MB", CONFIG, 20, int),
    "allowed_upload_ext": [".txt", ".md", ".html", ".htm", ".json", ".pdf"],
    # 单次 ingest_blocks 接口允许的最大知识块数量，防止超大 payload 拖垮服务
    "max_blocks_per_ingest": _env_or_config("RAG_MAX_BLOCKS_PER_INGEST", CONFIG, 5000, int),

    # -------- SSE 流式问答 --------
    # /api/chat/stream 逐段推送答案文本时，每个 data chunk 的字符数
    "stream_answer_chunk_chars": _env_or_config("RAG_STREAM_ANSWER_CHUNK_CHARS", CONFIG, 20, int),
}


def get_rag_config() -> dict:
    """返回 RAG 配置字典的浅拷贝，避免外部代码意外修改全局配置。"""
    return dict(RAG_CONFIG)


if __name__ == "__main__":
    import json
    print(json.dumps(RAG_CONFIG, ensure_ascii=False, indent=2))
