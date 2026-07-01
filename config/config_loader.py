# -*- coding: utf-8 -*-
"""
项目级配置加载器。

供 model/ 和其他非 process/ 模块使用。
从项目根目录的 .env 和 config/config.json 加载配置。
"""
import os
import json
import logging
from logging.handlers import TimedRotatingFileHandler

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ======================== 加载 .env ========================
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
try:
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
except ImportError:
    pass


# ======================== 辅助函数 ========================
def _env_or_config(key: str, config: dict, default=None, cast=str):
    """从环境变量或 config 字典中读取值，环境变量优先。"""
    env_val = os.environ.get(key)
    if env_val is not None and env_val != "":
        try:
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


# ======================== 加载 config.json ========================
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.json")

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(
        f"配置文件不存在: {CONFIG_PATH}\n"
        f"请复制 config/config.example.json 为 config/config.json 并填写实际配置。"
    )

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    _RAW_CONFIG = json.load(f)

# ======================== 合并环境变量覆盖 ========================
CONFIG = dict(_RAW_CONFIG)

CONFIG["embed_model"] = _env_or_config("EMBED_MODEL", _RAW_CONFIG, "Qwen/Qwen3-Embedding-4B")
CONFIG["rerank_model"] = _env_or_config("RERANK_MODEL", _RAW_CONFIG, "Qwen/Qwen3-Reranker-4B")
CONFIG["llm_model"] = _env_or_config("LLM_MODEL", _RAW_CONFIG, "THUDM/glm-4-9b-chat")
CONFIG["vllm_api_url"] = _env_or_config("VLLM_API_URL", _RAW_CONFIG, "http://localhost:8011/v1/chat/completions")
CONFIG["embed_api_url"] = _env_or_config("VLLM_EMBED_API_URL", _RAW_CONFIG, "http://localhost:8010/v1/embeddings")
CONFIG["vllm_timeout"] = _env_or_config("VLLM_TIMEOUT", _RAW_CONFIG, 60, int)
CONFIG["vllm_max_concurrent_requests"] = _env_or_config("VLLM_MAX_CONCURRENT_REQUESTS", _RAW_CONFIG, 32, int)
CONFIG["vllm_batch_size"] = _env_or_config("VLLM_BATCH_SIZE", _RAW_CONFIG, 32, int)
CONFIG["device"] = _env_or_config("DEVICE", _RAW_CONFIG, "cuda:0")
CONFIG["lang"] = _env_or_config("LANG", _RAW_CONFIG, "zh")
CONFIG["max_node_words_embed"] = _env_or_config("MAX_NODE_WORDS_EMBED", _RAW_CONFIG, 4096, int)
CONFIG["min_node_words_embed"] = _env_or_config("MIN_NODE_WORDS_EMBED", _RAW_CONFIG, 48, int)
CONFIG["log_level"] = _env_or_config("LOG_LEVEL", _RAW_CONFIG, "INFO")
CONFIG["env_default"] = _env_or_config("RAG_ENV", _RAW_CONFIG, "dev")

# 环境特定配置覆盖
_env_config = CONFIG.get("env_config", {})
for env_name in ("dev", "prod"):
    if env_name in _env_config:
        _env_cfg = _env_config[env_name]
        prefix = env_name.upper()
        _env_cfg["milvus_host"] = _env_or_config(f"MILVUS_HOST_{prefix}", {"v": _env_cfg.get("milvus_host")}, _env_cfg.get("milvus_host", "127.0.0.1"))
        _env_cfg["es_host"] = _env_or_config(f"ES_HOST_{prefix}", {"v": _env_cfg.get("es_host")}, _env_cfg.get("es_host", "127.0.0.1"))
        _env_cfg["collection_name"] = _env_or_config(f"MILVUS_COLLECTION_{prefix}", {"v": _env_cfg.get("collection_name")}, _env_cfg.get("collection_name", f"htmlrag_{env_name}"))
        _env_cfg["index_name"] = _env_or_config(f"ES_INDEX_{prefix}", {"v": _env_cfg.get("index_name")}, _env_cfg.get("index_name", f"htmlrag_{env_name}"))

CONFIG["_project_root"] = PROJECT_ROOT

# ======================== 派生常量 ========================
CONFIG["tei_embed_url"] = _env_or_config("TEI_EMBED_URL", _RAW_CONFIG, "http://localhost:8080")
CONFIG["tei_rerank_url"] = _env_or_config("TEI_RERANK_URL", _RAW_CONFIG, "http://localhost:8081")

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
BLACKHOLE_URL = os.environ.get("BLACKHOLE_URL", "")

# ======================== 日志配置 ========================
LOG_LEVEL = CONFIG.get("log_level", "INFO").upper()
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "app.log")
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("GlobalLogger")
logger.setLevel(LOG_LEVEL)

if not logger.handlers:
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATEFMT)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = TimedRotatingFileHandler(
        filename=log_file, when="midnight", interval=1,
        backupCount=7, encoding="utf-8", utc=False
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
