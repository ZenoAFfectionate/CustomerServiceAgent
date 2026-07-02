# -*- coding: utf-8 -*-
"""文档块字段定义与 Milvus / Elasticsearch schema 映射。

对齐 `process/` 输出字段：
    chunk_idx / page_name / title / page_url / text / html_content /
    block_path / summary / question / time

并新增：
    global_chunk_idx  —— 全局唯一主键（int64，跨文件自增，Milvus 主键）
    doc_id            —— 所属文档 ID（上传/知识库管理用）
    embedding         —— 向量字段（维度由 embedder 决定）
"""
from typing import List, Optional
from dataclasses import dataclass, field, asdict

# process/ 输出的原始字段（不含向量与主键）
BLOCK_FIELDS = [
    "chunk_idx", "page_name", "title", "page_url",
    "text", "html_content", "block_path", "summary", "question", "time",
]

# rag/ 索引时新增的字段
INDEX_EXTRA_FIELDS = ["global_chunk_idx", "doc_id", "source"]

ALL_FIELDS = INDEX_EXTRA_FIELDS + BLOCK_FIELDS


@dataclass
class DocBlock:
    """统一的文档块数据结构，贯穿 索引 → 检索 → 融合 → 精排 → 生成 全链路。"""

    text: str
    global_chunk_idx: int = -1
    doc_id: str = ""
    source: str = ""
    chunk_idx: int = 0
    page_name: str = ""
    title: str = ""
    page_url: str = ""
    html_content: str = ""
    block_path: str = ""
    summary: str = ""
    question: str = ""
    time: str = ""
    # 检索/融合/精排过程中动态附加的字段
    score: float = 0.0
    source_retriever: str = ""  # "milvus" / "es" / "fused"
    embedding: Optional[List[float]] = field(default=None, repr=False)

    def to_dict(self, with_embedding: bool = False) -> dict:
        d = asdict(self)
        if not with_embedding:
            d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "DocBlock":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def dedup_key_text(self) -> str:
        """去重复用字段：优先 text，为空时退化为 summary/title。"""
        return self.text or self.summary or self.title


# ======================== Milvus Schema ========================

def get_milvus_schema_fields(embedding_dim: int = 1024):
    """返回 pymilvus FieldSchema 列表（延迟 import，避免未安装 pymilvus 时报错）。"""
    from pymilvus import FieldSchema, DataType

    return [
        FieldSchema(name="global_chunk_idx", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="chunk_idx", dtype=DataType.INT64),
        FieldSchema(name="page_name", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="page_url", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="html_content", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="block_path", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="question", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="time", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
    ]


# ======================== Elasticsearch Mapping ========================

def get_es_mapping() -> dict:
    """返回 ES 索引 mapping（IK 分词器；若集群未装 IK 插件，自动回退 standard）。"""
    text_analyzer = {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"}
    return {
        "mappings": {
            "properties": {
                "global_chunk_idx": {"type": "long"},
                "doc_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "chunk_idx": {"type": "long"},
                "page_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}, **({} )},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "page_url": {"type": "keyword"},
                "text": text_analyzer,
                "html_content": {"type": "text", "index": False},
                "block_path": {"type": "keyword"},
                "summary": text_analyzer,
                "question": text_analyzer,
                "time": {"type": "keyword"},
            }
        }
    }


def get_es_mapping_fallback() -> dict:
    """无 IK 插件时的降级 mapping（standard 分词器）。"""
    return {
        "mappings": {
            "properties": {
                "global_chunk_idx": {"type": "long"},
                "doc_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "chunk_idx": {"type": "long"},
                "page_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "page_url": {"type": "keyword"},
                "text": {"type": "text"},
                "html_content": {"type": "text", "index": False},
                "block_path": {"type": "keyword"},
                "summary": {"type": "text"},
                "question": {"type": "text"},
                "time": {"type": "keyword"},
            }
        }
    }
