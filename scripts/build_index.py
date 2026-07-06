#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建 RAG 索引：读取 `process/` 输出的知识块 JSON，写入向量库 + 关键词库。

对齐 TODO.md T1 DoD：
    给定示例 JSON 块，能成功写入并在两库中查到对应条数；
    重复运行同一文件不产生重复数据（幂等）。

底层复用 `rag/knowledge_base/update_sync.py` 的增量同步逻辑：按内容哈希
判断文件是否变化，未变化的文件直接跳过（不重复写入/删除），比旧版"每次
都先删后插"进一步减少了不必要的向量化开销。

用法：
    python scripts/build_index.py [--source-dir process/dataset/html_cleaned_block]
    python scripts/build_index.py --force        # 忽略内容哈希，强制全部重新导入
    python scripts/build_index.py --dry-run       # 仅预览将要执行的变更，不实际写入

    # 使用真实 Milvus/ES/TEI（默认使用本地降级后端）
    RAG_VECTOR_BACKEND=milvus RAG_KEYWORD_BACKEND=es RAG_EMBED_BACKEND=tei \\
        python scripts/build_index.py
"""
import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description="构建 RAG 索引（写入向量库 + 关键词库）")
    parser.add_argument(
        "--source-dir", default=os.path.join("process", "dataset", "html_cleaned_block"),
        help="process/ 输出的知识块 JSON 所在目录（递归扫描 *.json）",
    )
    parser.add_argument("--force", action="store_true", help="忽略内容哈希比对，强制重新导入全部文件")
    parser.add_argument("--dry-run", action="store_true", help="仅预览将要执行的变更计划，不实际写入索引")
    args = parser.parse_args()

    source_dir = args.source_dir
    if not os.path.isdir(source_dir):
        print(f"❌ 源目录不存在: {source_dir}")
        print("   请先运行 `bash scripts/process_HTMLdata.sh` 生成知识块 JSON。")
        sys.exit(1)

    from rag.knowledge_base.update_sync import sync_directory
    from rag.indexing.index_builder import get_stats

    result = sync_directory(source_dir, force=args.force, dry_run=args.dry_run)

    for f in result["files"]:
        icon = {"skip_unchanged": "⏭️ ", "ingested": "✅", "would_ingest": "📝", "would_reingest": "📝"}.get(f["action"], "•")
        print(f"{icon} {f['path']} [{f['action']}] → {f['num_chunks']} 块（doc_id={f['doc_id']}）")

    print("=" * 60)
    if args.dry_run:
        print(f"🔍 预览完成：共扫描 {result['scanned']} 个文件（--dry-run 未实际写入）")
        return

    print(f"✅ 索引构建完成：共扫描 {result['scanned']} 个文件，"
          f"新导入/更新 {result['ingested']} 个，跳过未变化 {result['skipped_unchanged']} 个，"
          f"共写入 {result['total_chunks']} 个块")
    if result["scanned"] == 0:
        print(f"⚠️ 目录 {source_dir} 下未找到任何 *.json 文件，无内容可索引。")
        return

    stats = get_stats()
    print(f"   向量库: {stats['vector_backend']}（当前 {stats['num_vector_chunks']} 块）")
    print(f"   关键词库: {stats['keyword_backend']}（当前 {stats['num_keyword_chunks']} 块）")


if __name__ == "__main__":
    main()
