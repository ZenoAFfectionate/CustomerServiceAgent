# -*- coding: utf-8 -*-
"""结构化事件日志（Structured Logging）。

基于项目已有的 `config.config_loader.logger`（loguru）做薄封装，统一事件
日志的字段格式（`event=... key=value ...`），方便未来切换到真正的结构化
日志采集（如 ELK/ClickHouse）时只需替换 `log_event` 的实现，不影响调用方。
"""
from config.config_loader import logger as _base_logger


def log_event(event: str, level: str = "info", **fields) -> None:
    """记录一条结构化事件日志。

    Args:
        event: 事件名（如 "rag.retrieve" / "rag.answer" / "rag.ingest"）
        level: "info" / "warning" / "error"
        **fields: 附加字段（如 query、latency_ms、backend_used），按
            `key=value` 拼接在事件名之后
    """
    kv = " ".join(f"{k}={v!r}" for k, v in fields.items())
    message = f"[event={event}] {kv}".strip()
    log_fn = getattr(_base_logger, level, _base_logger.info)
    log_fn(message)


def get_logger():
    """返回底层 logger 实例，供需要直接调用 loguru API 的场景使用。"""
    return _base_logger
