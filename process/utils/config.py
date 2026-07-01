# -*- coding: utf-8 -*-
"""
数据处理配置加载模块（仅用于 process/ 数据处理流水线）。

加载顺序：
  1. .env 文件（通过 python-dotenv 加载）
  2. config/config.json（项目级结构化配置）
  3. 环境变量覆盖（优先级最高）

注意：本模块仅负责 HTML 清洗、分块、摘要生成相关配置。
Milvus/ES 等数据库配置由 rag/ 模块自行管理，不在此处。
"""
import json
import os
import aiohttp
import asyncio
import logging
from logging.handlers import TimedRotatingFileHandler

# ======================== 路径计算 ========================
current_file_dir = os.path.dirname(os.path.abspath(__file__))
# process/utils/config.py → process/ → 项目根
PROCESS_ROOT = os.path.dirname(current_file_dir)   # process/
PROJECT_ROOT = os.path.dirname(PROCESS_ROOT)       # CustomerServiceAgent/
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

# ======================== 加载 .env ========================
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

# LLM 与模型配置（用于摘要生成）
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

# 分块参数
CONFIG["max_node_words_embed"] = _env_or_config("MAX_NODE_WORDS_EMBED", _RAW_CONFIG, 4096, int)
CONFIG["min_node_words_embed"] = _env_or_config("MIN_NODE_WORDS_EMBED", _RAW_CONFIG, 48, int)
CONFIG["log_level"] = _env_or_config("LOG_LEVEL", _RAW_CONFIG, "INFO")

# ======================== 派生常量 ========================
CONFIG["_project_root"] = PROJECT_ROOT
CONFIG["_process_root"] = PROCESS_ROOT

DATA_DIR = os.path.join(PROCESS_ROOT, "dataset")
USER_DICT_PATH = os.path.join(DATA_DIR, "user_dict.txt")

# LLM 服务配置
OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://localhost:11434/v1/chat/completions")

# ======================== 日志配置 ========================
LOG_LEVEL = CONFIG.get("log_level", "INFO").upper()
LOG_DIR = os.path.join(PROCESS_ROOT, "logs")
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
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
        utc=False
    )
    file_handler.suffix = "%Y-%m-%d"
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

logger.info("=" * 80)
logger.info("✅ process/ 配置初始化完成")

# ======================== aiohttp 会话管理 ========================
_session = None
sem = asyncio.Semaphore(CONFIG.get("vllm_max_concurrent_requests", 32))


async def get_aiohttp_session():
    """获取或创建全局 aiohttp 会话。"""
    global _session
    if _session is None or _session.closed:
        timeout = aiohttp.ClientTimeout(total=CONFIG.get("vllm_timeout", 60))
        _session = aiohttp.ClientSession(
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        logger.info("新建 aiohttp 会话")
    return _session


async def close_aiohttp_session():
    """关闭全局 aiohttp 会话。"""
    global _session
    if _session and not _session.closed:
        await _session.close()
        logger.info("aiohttp 会话已关闭")
