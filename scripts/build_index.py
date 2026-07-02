#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 RAG 索引：读取 `process/` 输出的知识块 JSON，写入向量库 + 关键词库。

对齐 TODO.md T1 DoD：
    给定示例 JSON 块，能成功写入并在两库中查到对应条数；
    重复运行同一文件不产生重复数据（幂等）。

幂等实现：每个源文件映射一个稳定的 `doc_id`（`file::<相对路径>`），
重新导入前先删除该 `doc_id` 下的全部旧块（等价于 TODO 描述的"以 page_url
为粒度先删后插"，这里用更通用的 doc_id 粒度，同时兼容本地降级后端与
真实 Milvus/ES 后端）。

用法：
    python scripts/build_index.py [--source-dir process/dataset/html_cleaned_block]

    # 使用真实 Milvus/ES/TEI（默认使用本地降级后端）
    RAG_VECTOR_BACKEND=milvus RAG_KEYWORD_BACKEND=es RAG_EMBED_BACKEND=tei \\
        python scripts/build_index.py
"""
import argparse
import glob
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def _iter_json_files(source_dir: str):
    pattern = os.path.join(source_dir, "**", "*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        yield path


def main():
    parser = argparse.ArgumentParser(description="构建 RAG 索引（写入向量库 + 关键词库）")
    parser.add_argument(
        "--source-dir", default=os.path.join("process", "dataset", "html_cleaned_block"),
        help="process/ 输出的知识块 JSON 所在目录（递归扫描 *.json）",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    if not os.path.isdir(source_dir):
        print(f"❌ 源目录不存在: {source_dir}")
        print("   请先运行 `bash scripts/process_HTMLdata.sh` 生成知识块 JSON。")
        sys.exit(1)

    from rag.indexing import indexer

    json_files = list(_iter_json_files(source_dir))
    if not json_files:
        print(f"⚠️ 目录 {source_dir} 下未找到任何 *.json 文件，无内容可索引。")
        sys.exit(0)

    total_chunks = 0
    for path in json_files:
        with open(path, "r", encoding="utf-8") as f:
            blocks = json.load(f)
        if isinstance(blocks, dict):
            blocks = [blocks]
        if not blocks:
            print(f"⚠️ 跳过空文件: {path}")
            continue

        doc_id_for_file = f"file::{os.path.relpath(path, source_dir)}"
        # 幂等：先删除该文件上一次导入的全部旧块，再重新写入
        try:
            indexer.delete_document(doc_id_for_file)
        except Exception as e:
            print(f"⚠️ 清理旧数据失败（可忽略首次运行）: {e}")

        meta = indexer.ingest_blocks(blocks, filename=os.path.basename(path), doc_id=doc_id_for_file)
        total_chunks += meta["num_chunks"]
        print(f"✅ {path} → {meta['num_chunks']} 块（doc_id={meta['doc_id']}）")

    stats = indexer.get_stats()
    print("=" * 60)
    print(f"✅ 索引构建完成：共处理 {len(json_files)} 个文件，{total_chunks} 个块")
    print(f"   向量库: {stats['vector_backend']}（当前 {stats['num_vector_chunks']} 块）")
    print(f"   关键词库: {stats['keyword_backend']}（当前 {stats['num_keyword_chunks']} 块）")


if __name__ == "__main__":
    main()
