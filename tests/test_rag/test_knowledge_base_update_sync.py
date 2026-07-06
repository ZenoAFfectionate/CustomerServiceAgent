# -*- coding: utf-8 -*-
"""rag/knowledge_base/update_sync.py 单元测试：目录级增量同步。

组合测试：验证 `data_sources.py`（扫描目录）+ `versioning.py`（内容哈希比对）+
`corpus_management.py`（写索引）三者组合而成的增量同步语义：新增文件导入、
未变化文件跳过、内容变化文件重新导入、force/dry_run 参数生效。
"""
import json

import pytest

from rag.knowledge_base import corpus_management, update_sync

pytestmark = pytest.mark.usefixtures("clean_rag_data")


def _write_json(path, blocks):
    path.write_text(json.dumps(blocks, ensure_ascii=False), encoding="utf-8")


class TestSyncDirectory:
    def test_first_sync_ingests_all_files(self, tmp_path):
        _write_json(tmp_path / "a.json", [{"text": "内容一"}])
        _write_json(tmp_path / "b.json", [{"text": "内容二"}])

        result = update_sync.sync_directory(str(tmp_path))
        assert result["scanned"] == 2
        assert result["ingested"] == 2
        assert result["skipped_unchanged"] == 0
        assert len(corpus_management.list_documents()) == 2

    def test_second_sync_skips_unchanged_files(self, tmp_path):
        _write_json(tmp_path / "a.json", [{"text": "内容一"}])
        update_sync.sync_directory(str(tmp_path))

        result = update_sync.sync_directory(str(tmp_path))
        assert result["ingested"] == 0
        assert result["skipped_unchanged"] == 1

    def test_changed_content_triggers_reingest(self, tmp_path):
        file_path = tmp_path / "a.json"
        _write_json(file_path, [{"text": "原始内容"}])
        update_sync.sync_directory(str(tmp_path))

        _write_json(file_path, [{"text": "更新后的内容"}])
        result = update_sync.sync_directory(str(tmp_path))
        assert result["ingested"] == 1
        assert result["skipped_unchanged"] == 0

        docs = corpus_management.list_documents()
        assert len(docs) == 1  # 同一 doc_id，先删后插，不产生重复文档记录

    def test_force_reingests_even_unchanged_files(self, tmp_path):
        _write_json(tmp_path / "a.json", [{"text": "内容"}])
        update_sync.sync_directory(str(tmp_path))

        result = update_sync.sync_directory(str(tmp_path), force=True)
        assert result["ingested"] == 1
        assert result["skipped_unchanged"] == 0

    def test_dry_run_does_not_write_index(self, tmp_path):
        _write_json(tmp_path / "a.json", [{"text": "内容"}])
        result = update_sync.sync_directory(str(tmp_path), dry_run=True)
        assert result["files"][0]["action"] == "would_ingest"
        assert corpus_management.list_documents() == []

    def test_dry_run_on_unchanged_reports_skip(self, tmp_path):
        _write_json(tmp_path / "a.json", [{"text": "内容"}])
        update_sync.sync_directory(str(tmp_path))

        result = update_sync.sync_directory(str(tmp_path), dry_run=True)
        assert result["files"][0]["action"] == "skip_unchanged"

    def test_empty_directory_returns_zero_scanned(self, tmp_path):
        result = update_sync.sync_directory(str(tmp_path))
        assert result["scanned"] == 0
        assert result["ingested"] == 0

    def test_delete_failure_skips_reingest_instead_of_silently_continuing(self, tmp_path, monkeypatch):
        """回归测试：修复审查报告 H3——此前用宽松 try/except 无条件吞掉
        delete_document 的异常，若清理旧数据真实失败仍会继续写入新数据，
        造成新旧数据并存的不一致状态。现改为：真实失败时跳过本文件的
        重新导入，并在结果中显式记录 `delete_failed`。"""
        file_path = tmp_path / "a.json"
        _write_json(file_path, [{"text": "原始内容"}])
        update_sync.sync_directory(str(tmp_path))

        _write_json(file_path, [{"text": "更新后的内容"}])

        def _boom_delete(_doc_id):
            raise RuntimeError("删除旧数据失败（模拟存储后端故障）")

        monkeypatch.setattr(corpus_management, "delete_document", _boom_delete)
        result = update_sync.sync_directory(str(tmp_path))

        assert result["ingested"] == 0
        assert result["files"][0]["action"] == "delete_failed"
        # 旧文档记录应原样保留（未被误删、也未被新内容覆盖）
        docs = corpus_management.list_documents()
        assert len(docs) == 1


class TestCheckPendingChanges:
    def test_returns_dry_run_result_without_side_effects(self, tmp_path):
        _write_json(tmp_path / "a.json", [{"text": "内容"}])
        report = update_sync.check_pending_changes(str(tmp_path))
        assert report["files"][0]["action"] == "would_ingest"
        assert corpus_management.list_documents() == []
