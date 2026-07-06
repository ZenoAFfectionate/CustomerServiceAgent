# -*- coding: utf-8 -*-
"""观测与监控子模块（Observability）：结构化日志、链路追踪、指标监控、告警、
运维看板。

    - logging.py    结构化事件日志（基于 config.config_loader.logger 的轻量封装）
    - tracing.py     链路耗时追踪（Span/Trace，替代 pipeline.py 中原有的手写计时代码）
    - monitoring.py  进程内滚动指标采集（请求量/延迟分位数/后端使用分布/错误率）
    - alerting.py    基于监控快照的阈值告警检查
    - dashboard.py   FastAPI 路由：以 JSON 形式暴露监控快照，供简易运维看板使用

设计原则与 `rag/` 其余模块一致：全部为**进程内、无外部依赖**的轻量实现，
不引入 Prometheus/Grafana 等外部组件，保证开箱即用；生产环境可在此基础上
将 `monitoring.snapshot()` 的输出对接到真实的可观测性平台。
"""
