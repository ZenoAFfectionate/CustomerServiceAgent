# -*- coding: utf-8 -*-
"""统一错误码与异常处理。

Agent 调用方可根据 `error_code` 做程序化判断，`message` 为人类可读描述。
错误码列表详见 `rag/docs/API_REFERENCE.md`。
"""
from typing import Optional

from fastapi import Request
from fastapi.responses import JSONResponse

from config.config_loader import logger


class RagAPIError(Exception):
    """RAG API 业务异常基类。"""

    status_code = 400
    error_code = "RAG_BAD_REQUEST"

    def __init__(self, message: str, detail: Optional[str] = None):
        self.message = message
        self.detail = detail
        super().__init__(message)


class NotFoundError(RagAPIError):
    status_code = 404
    error_code = "RAG_NOT_FOUND"


class ValidationError(RagAPIError):
    status_code = 422
    error_code = "RAG_VALIDATION_ERROR"


class PayloadTooLargeError(RagAPIError):
    status_code = 413
    error_code = "RAG_PAYLOAD_TOO_LARGE"


class UnsupportedMediaTypeError(RagAPIError):
    status_code = 415
    error_code = "RAG_UNSUPPORTED_MEDIA_TYPE"


class InternalError(RagAPIError):
    status_code = 500
    error_code = "RAG_INTERNAL_ERROR"


async def rag_error_handler(request: Request, exc: RagAPIError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": exc.error_code, "message": exc.message, "detail": exc.detail},
    )


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    # 【修复】原实现未在服务端记录任何日志——若客户端未完整保存响应体，这类
    # 未预期的 500 错误将在服务端完全不可见，无法排查。这里补充 `logger.error`
    # （附带请求路径 + 完整异常堆栈，便于定位是哪个接口触发了异常）。
    # 显式传入 `exc_info=exc`（而非 `exc_info=True`）以确保无论当前是否处于
    # ambient except 上下文中都能正确附加该异常的堆栈信息。
    logger.error(f"❌ 未处理异常 [{request.method} {request.url.path}]: {exc}", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={"error_code": "RAG_INTERNAL_ERROR", "message": "服务器内部错误", "detail": str(exc)},
    )
