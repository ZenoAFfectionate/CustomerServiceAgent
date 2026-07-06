# -*- coding: utf-8 -*-
"""链路耗时追踪（Tracing）：轻量的 Span/Trace 实现，替代 `pipeline.py` 中原本
分散的 `t0 = time.time()` / `t1 = time.time()` 手写计时代码。

用法：
    trace = Trace("retrieve(query=...)")
    with trace.span("vector"):
        ...  # 向量检索
    with trace.span("keyword"):
        ...  # 关键词检索
    logger.info(trace.summary())      # "retrieve(...) 耗时: vector=12ms keyword=8ms 总计=20ms"
    monitoring.record_retrieval(trace.as_dict())
"""
import time
from contextlib import contextmanager
from typing import Dict


class Trace:
    """记录一次调用链路中各阶段（span）耗时的容器。"""

    def __init__(self, name: str):
        self.name = name
        self._t_start = time.time()
        self.spans: Dict[str, float] = {}  # span 名 -> 耗时（毫秒）

    @contextmanager
    def span(self, name: str):
        """记录一个命名阶段的耗时（毫秒），即使阶段内抛出异常也会记录。"""
        t0 = time.time()
        try:
            yield
        finally:
            self.spans[name] = round((time.time() - t0) * 1000, 2)

    @property
    def total_ms(self) -> float:
        return round((time.time() - self._t_start) * 1000, 2)

    def summary(self) -> str:
        parts = " ".join(f"{k}={v:.0f}ms" for k, v in self.spans.items())
        return f"{self.name} 耗时: {parts} 总计={self.total_ms:.0f}ms"

    def as_dict(self) -> dict:
        return {"name": self.name, "spans": dict(self.spans), "total_ms": self.total_ms}
