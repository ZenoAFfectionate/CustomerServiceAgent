# -*- coding: utf-8 -*-
"""基于监控快照的阈值告警检查（Alerting）。

不依赖外部告警渠道（如企业微信/钉钉机器人），当前实现仅产出结构化的告警
列表并写入日志；`integration/deployment.py`、`observability/dashboard.py`
或未来的定时任务可基于 `check_alerts()` 的返回值自行决定推送渠道。
"""
from typing import Dict, List, Optional

from rag.observability.logging import log_event

DEFAULT_THRESHOLDS = {
    "error_rate": 0.05,          # 错误率超过 5% 告警
    "retrieval_p95_ms": 2000,    # 检索 P95 延迟超过 2s 告警
    "answer_p95_ms": 5000,       # 问答 P95 延迟超过 5s 告警
    "fallback_rate": 0.8,        # 生成降级为本地抽取式回答的比例超过 80% 告警（可能是 vLLM 挂了）
    # 【修复 L15】生成阶段意外异常率（区别于 fallback_rate——后者是"正常
    # 降级为抽取式"，前者是"生成阶段抛出了未预期的异常"）超过 10% 告警。
    "generation_error_rate": 0.1,
}


def check_alerts(snapshot: dict, thresholds: Optional[Dict[str, float]] = None) -> List[dict]:
    """对比监控快照与阈值，返回触发的告警列表。

    Args:
        snapshot: `rag.observability.monitoring.snapshot()` 的返回值
        thresholds: 自定义阈值，未提供的键使用 `DEFAULT_THRESHOLDS` 默认值

    Returns:
        [{"level": "warning"/"critical", "metric": str, "value": float,
          "threshold": float, "message": str}, ...]
    """
    th = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    alerts: List[dict] = []

    def _check(metric: str, value: float, threshold: float, level: str = "warning"):
        if value > threshold:
            alert = {
                "level": level, "metric": metric, "value": value, "threshold": threshold,
                "message": f"{metric}={value} 超过阈值 {threshold}",
            }
            alerts.append(alert)
            log_event("rag.alert", level="warning", metric=metric, value=value, threshold=threshold, alert_level=level)

    _check("error_rate", snapshot.get("error_rate", 0.0), th["error_rate"], level="critical")
    _check("retrieval_p95_ms", snapshot.get("retrieval", {}).get("p95_ms", 0.0), th["retrieval_p95_ms"])
    _check("answer_p95_ms", snapshot.get("answer", {}).get("p95_ms", 0.0), th["answer_p95_ms"])
    _check("fallback_rate", snapshot.get("answer", {}).get("fallback_rate", 0.0), th["fallback_rate"])
    _check(
        "generation_error_rate", snapshot.get("answer", {}).get("generation_error_rate", 0.0),
        th["generation_error_rate"], level="critical",
    )

    return alerts
