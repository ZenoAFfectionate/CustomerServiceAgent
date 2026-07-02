# -*- coding: utf-8 -*-
"""rag —— RAG（检索增强生成）核心框架。

对齐 TODO.md M3 里程碑：索引构建 → 双模检索 → 融合去重 → Reranker 精排 →
全流程编排 → 生成融合。每个子模块职责单一、可独立替换：

    rag/
    ├── config.py              RAG 专属配置
    ├── schema.py               字段映射 / Milvus & ES schema 定义
    ├── indexing/               文档解析、分块、向量化、索引写入
    ├── retrieval/              向量检索、关键词检索、融合去重、精排
    ├── generation/             基于检索上下文的答案生成
    ├── pipeline.py             全流程编排入口（retrieve / answer）
    └── api/                    FastAPI 后端服务

设计原则：核心链路（Milvus/ES/TEI/vLLM）不可用时，可通过配置切换到
本地降级实现（`local` backend），保证开箱即用与可测试性。
"""

__version__ = "0.1.0"
