# -*- coding: utf-8 -*-
"""检索能力子模块（Retrieval）：query 理解与改写、检索器选择、混合检索、精排。

    - query_understanding.py  查询理解（语言/意图/关键词/复杂度）
    - query_rewrite.py        多轮对话查询重写（指代补全）
    - retriever_selection.py  检索器选择（决定跑哪些召回路径 + 当前后端配置）
    - hybrid_search.py        向量检索 + 关键词检索 + 融合去重
    - rerank.py                Reranker 精排
"""
