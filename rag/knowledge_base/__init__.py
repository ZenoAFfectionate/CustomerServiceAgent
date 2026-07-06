# -*- coding: utf-8 -*-
"""知识库管理子模块（Knowledge Base）：在 `indexing/`（文档解析→分块→向量化→
写入索引这一"机械"流程）之上叠加"知识库"这一业务概念 —— 数据从哪里来、
如何做批量/增量同步、导入前的质量把关、以及内容变更的版本追溯。

    - data_sources.py       数据来源接入（上传文件 / process/ 知识块 / 目录批量扫描）
    - corpus_management.py  知识库高层编排（对外统一入口，供 API 层 / scripts 调用）
    - update_sync.py        目录级增量同步（按内容哈希判断变更，避免重复索引）
    - quality_control.py    导入前质量检查（空文本率、重复率、HTML 残留等）
    - versioning.py         文档内容版本历史追踪

`indexing/` 关注"怎么建索引"，`knowledge_base/` 关注"知识库里有什么、
从哪来、是否可信、变化了什么" —— 两者职责互补而不重叠。
"""
