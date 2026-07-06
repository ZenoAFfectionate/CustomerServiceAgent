# -*- coding: utf-8 -*-
"""集成与部署子模块（Integration）：把 `rag/` 的检索增强问答能力接入到
其他系统 —— HTTP 服务、Agent 框架、部署运维检查、可运行的用法示例。

    - api_integration.py     将 rag/api 的 FastAPI 应用嵌入宿主服务 / 独立获取 app
    - agent_integration.py   将 retrieve/answer 封装为 Agent 可调用的工具函数
    - tool_usage.py          Agent 工具调用（Function Calling）JSON Schema 定义
    - workflow_examples.py   典型用法示例（可直接运行）
    - deployment.py          部署就绪检查（聚合各组件 health_check）
"""
