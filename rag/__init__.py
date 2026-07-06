# -*- coding: utf-8 -*-
"""rag —— RAG（检索增强生成）核心框架。

对齐 TODO.md M3 里程碑，并在其基础上扩展为按"能力"划分的模块结构，
每个子模块职责单一、可独立替换：

    rag/
    ├── config.py              RAG 专属配置
    ├── schema.py              字段映射 / Milvus & ES schema 定义
    ├── pipeline.py            全流程编排入口（retrieve / answer）
    ├── retrieval/             检索能力：query 理解、查询重写、检索器选择、混合检索、精排
    ├── indexing/              索引构建：文档解析、分块、向量化、索引写入、元数据、后端存储
    ├── knowledge_base/        知识库管理：数据来源、语料编排、增量同步、质量检查、版本追踪
    ├── generation/            生成能力：Prompt 模板、上下文组装、LLM 生成、引用、幻觉控制
    ├── evaluation/            评估能力：指标、检索/生成/端到端评测、基准测试
    ├── observability/         观测与监控：结构化日志、链路追踪、指标监控、告警、运维看板
    ├── integration/           集成与部署：API 集成、Agent 集成、工具调用、用法示例、部署检查
    ├── api/                   FastAPI 后端服务（HTTP 接口层）
    ├── web/                   轻量前端页面
    └── docs/                  Agent 友好的接口文档

设计原则：核心链路（Milvus/ES/TEI/vLLM）不可用时，可通过配置切换到
本地降级实现（`local` backend），保证开箱即用与可测试性。
"""

__version__ = "0.2.0"
