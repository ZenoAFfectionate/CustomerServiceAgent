# -*- coding: utf-8 -*-
"""rag/observability/alerting.py 单元测试：基于监控快照的阈值告警检查。

组合测试：alerting.check_alerts 消费 observability/monitoring.py 产出的快照结构，
验证二者数据契约一致。
"""
from rag.observability.alerting import DEFAULT_THRESHOLDS, check_alerts


def _snapshot(error_rate=0.0, retrieval_p95=0.0, answer_p95=0.0, fallback_rate=0.0):
    return {
        "error_rate": error_rate,
        "retrieval": {"p95_ms": retrieval_p95},
        "answer": {"p95_ms": answer_p95, "fallback_rate": fallback_rate},
    }


class TestCheckAlerts:
    def test_healthy_snapshot_produces_no_alerts(self):
        snap = _snapshot(error_rate=0.0, retrieval_p95=100, answer_p95=200, fallback_rate=0.1)
        assert check_alerts(snap) == []

    def test_high_error_rate_triggers_critical_alert(self):
        snap = _snapshot(error_rate=0.5)
        alerts = check_alerts(snap)
        assert any(a["metric"] == "error_rate" and a["level"] == "critical" for a in alerts)

    def test_high_retrieval_latency_triggers_alert(self):
        snap = _snapshot(retrieval_p95=5000)
        alerts = check_alerts(snap)
        assert any(a["metric"] == "retrieval_p95_ms" for a in alerts)

    def test_high_answer_latency_triggers_alert(self):
        snap = _snapshot(answer_p95=10000)
        alerts = check_alerts(snap)
        assert any(a["metric"] == "answer_p95_ms" for a in alerts)

    def test_high_fallback_rate_triggers_alert(self):
        snap = _snapshot(fallback_rate=0.95)
        alerts = check_alerts(snap)
        assert any(a["metric"] == "fallback_rate" for a in alerts)

    def test_custom_thresholds_override_defaults(self):
        snap = _snapshot(error_rate=0.02)  # 低于默认阈值 0.05，但高于自定义 0.01
        alerts = check_alerts(snap, thresholds={"error_rate": 0.01})
        assert any(a["metric"] == "error_rate" for a in alerts)

    def test_missing_fields_do_not_crash(self):
        """快照缺失部分字段时应使用默认值兜底，不抛异常。"""
        assert check_alerts({}) == []

    def test_alert_message_contains_metric_and_threshold(self):
        snap = _snapshot(error_rate=0.5)
        alerts = check_alerts(snap)
        alert = next(a for a in alerts if a["metric"] == "error_rate")
        assert "error_rate" in alert["message"]
        assert str(DEFAULT_THRESHOLDS["error_rate"]) in alert["message"]
