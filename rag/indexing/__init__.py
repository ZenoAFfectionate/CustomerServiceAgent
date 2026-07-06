# -*- coding: utf-8 -*-
"""索引构建子模块（Indexing）：文档解析、分块、向量化、元数据登记、
向量库/关键词库读写。

    - document_loader.py   文档解析（.txt/.md/.html/.json/.pdf → 纯文本或知识块）
    - chunking.py            通用文本分块
    - embedding.py           文本向量化（TEI / 本地哈希嵌入）
    - index_builder.py       索引构建编排（解析→分块→向量化→写入→登记）
    - metadata.py            文档级元数据登记表（列表/删除/global_chunk_idx 分配）

以下为后端存储实现细节（不属于对外的"5 个核心文件"，但索引构建/检索均依赖；
接口与具体实现分文件存放，本地降级实现与生产实现也分文件存放，避免任一实现
的技术细节相互污染，见各文件顶部的详细说明）：
    - store.py                                  向量/关键词存储的抽象接口 + 单例工厂
      （合并自原 vector_store.py + keyword_store.py，两者是完全对称的模式）
    - local_index.py                            本地零依赖降级实现
      （合并自原 local_vector_index.py + local_keyword_index.py）
    - milvus_index.py / es_index.py             生产后端实现（Milvus / Elasticsearch）
    - _process_compat.py                        process/ 复用桥接层
"""
