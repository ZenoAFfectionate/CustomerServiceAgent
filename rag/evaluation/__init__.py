# -*- coding: utf-8 -*-
"""评估能力子模块（Evaluation）：检索质量评测、生成质量评测、端到端评测、
通用指标函数、性能基准测试。

    - metrics.py          通用评估指标（Recall@K / Precision@K / MRR / NDCG / 词汇 F1）
    - retrieval_eval.py    检索质量评测（对齐标注的 query→相关文档集合）
    - generation_eval.py   生成质量评测（关联度/引用覆盖率/与参考答案的相似度）
    - rag_e2e_eval.py      端到端评测编排（组合上述两者，面向数据集批量运行）
    - benchmark.py         延迟/吞吐量基准测试

`tests/experiment/run_eval.py` 中原有的"全量评测脚本"是本模块思想的先行
实现（面向 Bitext 数据集的定制脚本）；本模块将其中通用的指标计算与评测
编排逻辑抽象为可复用的库函数，供任意知识库/评测集调用。
"""
