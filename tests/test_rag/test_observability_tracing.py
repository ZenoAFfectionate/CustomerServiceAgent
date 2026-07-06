# -*- coding: utf-8 -*-
"""rag/observability/tracing.py 单元测试：Span/Trace 链路耗时追踪。"""
import time

from rag.observability.tracing import Trace


class TestTraceSpan:
    def test_span_records_elapsed_time(self):
        trace = Trace("test_op")
        with trace.span("stage1"):
            time.sleep(0.01)
        assert "stage1" in trace.spans
        assert trace.spans["stage1"] >= 10  # 毫秒

    def test_multiple_spans_recorded_independently(self):
        trace = Trace("test_op")
        with trace.span("a"):
            pass
        with trace.span("b"):
            pass
        assert set(trace.spans.keys()) == {"a", "b"}

    def test_span_records_time_even_on_exception(self):
        trace = Trace("test_op")
        try:
            with trace.span("failing_stage"):
                raise RuntimeError("模拟异常")
        except RuntimeError:
            pass
        assert "failing_stage" in trace.spans

    def test_total_ms_increases_over_time(self):
        trace = Trace("test_op")
        t1 = trace.total_ms
        time.sleep(0.01)
        t2 = trace.total_ms
        assert t2 >= t1


class TestTraceSummaryAndDict:
    def test_summary_contains_name_and_spans(self):
        trace = Trace("my_operation")
        with trace.span("step1"):
            pass
        summary = trace.summary()
        assert "my_operation" in summary
        assert "step1" in summary
        assert "总计" in summary

    def test_as_dict_contains_expected_keys(self):
        trace = Trace("my_operation")
        with trace.span("step1"):
            pass
        d = trace.as_dict()
        assert d["name"] == "my_operation"
        assert "step1" in d["spans"]
        assert "total_ms" in d
