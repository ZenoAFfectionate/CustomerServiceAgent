# -*- coding: utf-8 -*-
"""rag/observability/logging.py 单元测试：结构化事件日志封装。"""
from rag.observability.logging import get_logger, log_event


class TestLogEvent:
    def test_log_event_does_not_raise(self, caplog):
        log_event("rag.test_event", query="测试查询", latency_ms=12.3)

    def test_log_event_supports_warning_level(self, caplog):
        log_event("rag.test_warning", level="warning", reason="示例告警")

    def test_log_event_supports_error_level(self, caplog):
        log_event("rag.test_error", level="error", reason="示例错误")

    def test_log_event_with_no_extra_fields(self, caplog):
        log_event("rag.test_no_fields")

    def test_unknown_level_falls_back_to_info(self, caplog):
        """未知 level 应回退为 info，而不是抛异常。"""
        log_event("rag.test_unknown_level", level="not_a_real_level")


class TestGetLogger:
    def test_returns_logger_instance(self):
        logger = get_logger()
        assert hasattr(logger, "info")
        assert hasattr(logger, "warning")
