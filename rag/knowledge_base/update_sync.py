# -*- coding: utf-8 -*-
"""增量同步（Update Sync）：目录级批量导入，按内容哈希判断变更，跳过未变化
的文件，避免每次全量重建索引。是 `scripts/build_index.py` 原有幂等逻辑的
可复用库化实现（原脚本逻辑为"每次都先删后插"，本模块进一步优化为"内容
未变化则完全跳过"，减少不必要的向量化/写入开销）。
"""
from typing import Optional
import os

from config.config_loader import logger
from rag.indexing.index_builder import filter_non_empty_blocks
from rag.knowledge_base import corpus_management
from rag.knowledge_base.data_sources import iter_directory_source, stable_doc_id_for_file
from rag.knowledge_base.versioning import compute_content_hash, get_version_store
from rag.observability.logging import log_event


def sync_directory(source_dir: str, force: bool = False, dry_run: bool = False) -> dict:
    """扫描目录并增量导入知识块。

    Args:
        source_dir: 知识块 JSON 所在目录（递归扫描 *.json）
        force: 为 True 时忽略内容哈希比对，强制重新导入全部文件
        dry_run: 为 True 时仅返回将要执行的变更计划，不实际写入索引

    Returns:
        {
            "scanned": int, "skipped_unchanged": int, "ingested": int,
            "total_chunks": int, "files": [{"path", "doc_id", "action", "num_chunks"}, ...],
        }
    """
    version_store = get_version_store()
    result = {"scanned": 0, "skipped_unchanged": 0, "ingested": 0, "total_chunks": 0, "files": []}

    for path, blocks in iter_directory_source(source_dir):
        result["scanned"] += 1
        doc_id = stable_doc_id_for_file(source_dir, path)
        # 【修复 L14】统一用 compute_content_hash(filter_non_empty_blocks(blocks))
        # 计算哈希，与 corpus_management.ingest_blocks 入库时记录的哈希一致。
        content_hash = compute_content_hash(filter_non_empty_blocks(blocks))
        latest_hash = version_store.get_latest_hash(doc_id)

        if not force and latest_hash == content_hash:
            result["skipped_unchanged"] += 1
            result["files"].append({"path": path, "doc_id": doc_id, "action": "skip_unchanged", "num_chunks": 0})
            continue

        if dry_run:
            action = "would_reingest" if latest_hash else "would_ingest"
            result["files"].append({"path": path, "doc_id": doc_id, "action": action, "num_chunks": len(blocks)})
            continue

        # 内容变化的文件先删除旧块再重新导入，保证幂等（与旧 scripts/build_index.py 一致）。
        # 【修复 H3】此前用宽松 try/except 无条件吞掉 delete_document 的异常
        # （仅打一条 warning），会掩盖删除阶段的真实部分失败——若旧数据未被
        # 完全清理却仍继续 ingest_blocks 写入新数据，会产生新旧数据并存的
        # 重复/不一致状态。`corpus_management.delete_document` 现在对"doc_id
        # 本就不存在"（首次运行的正常场景）返回 False、不抛异常；只有存储
        # 层真实失败时才会抛 RuntimeError。因此这里只需处理真实失败：跳过
        # 本文件的重新导入并在结果中显式记录 `delete_failed`，避免在旧数据
        # 清理不确定的情况下继续写入新数据造成更难排查的不一致。
        delete_failed = False
        try:
            corpus_management.delete_document(doc_id)
        except Exception as e:
            delete_failed = True
            logger.warning(f"⚠️ 清理旧数据失败（doc_id={doc_id}），跳过本文件本次同步: {e}")

        if delete_failed:
            result["files"].append({"path": path, "doc_id": doc_id, "action": "delete_failed", "num_chunks": 0})
            continue

        meta = corpus_management.ingest_blocks(blocks, filename=os.path.basename(path), doc_id=doc_id)
        result["ingested"] += 1
        result["total_chunks"] += meta["num_chunks"]
        result["files"].append({"path": path, "doc_id": doc_id, "action": "ingested", "num_chunks": meta["num_chunks"]})

    # 【日志完善】此前批量同步跑完后没有任何汇总日志——只能靠调用方自己打印返回值，
    # 若通过 scripts/build_index.py 之外的场景调用（如未来的定时任务/API）则完全
    # 不可见。dry_run 场景不记录（属于只读预览，不产生真实副作用，避免日志噪音）。
    if not dry_run:
        log_event(
            "rag.sync_directory", source_dir=source_dir, force=force,
            scanned=result["scanned"], ingested=result["ingested"],
            skipped_unchanged=result["skipped_unchanged"], total_chunks=result["total_chunks"],
        )
    return result


def check_pending_changes(source_dir: str) -> dict:
    """只读预览：返回目录同步会产生的变更计划，不写入任何数据（`sync_directory` 的 dry_run 封装）。"""
    return sync_directory(source_dir, dry_run=True)
