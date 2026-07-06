# -*- coding: utf-8 -*-
"""rag/knowledge_base/data_sources.py 单元测试：数据来源接入（目录批量扫描）。"""
import json

from rag.knowledge_base.data_sources import iter_directory_source, stable_doc_id_for_file


class TestIterDirectorySource:
    def test_scans_json_files_recursively(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "a.json").write_text(json.dumps([{"text": "块一"}]), encoding="utf-8")
        (tmp_path / "sub" / "b.json").write_text(json.dumps([{"text": "块二"}]), encoding="utf-8")

        results = list(iter_directory_source(str(tmp_path)))
        assert len(results) == 2

    def test_single_object_wrapped_as_list(self, tmp_path):
        (tmp_path / "a.json").write_text(json.dumps({"text": "单个对象"}), encoding="utf-8")
        results = list(iter_directory_source(str(tmp_path)))
        assert len(results) == 1
        _, blocks = results[0]
        assert blocks == [{"text": "单个对象"}]

    def test_empty_json_array_skipped(self, tmp_path):
        (tmp_path / "empty.json").write_text("[]", encoding="utf-8")
        (tmp_path / "valid.json").write_text(json.dumps([{"text": "有效"}]), encoding="utf-8")
        results = list(iter_directory_source(str(tmp_path)))
        assert len(results) == 1

    def test_malformed_json_skipped_not_raise(self, tmp_path):
        (tmp_path / "broken.json").write_text("{not valid json", encoding="utf-8")
        (tmp_path / "valid.json").write_text(json.dumps([{"text": "有效"}]), encoding="utf-8")
        results = list(iter_directory_source(str(tmp_path)))
        assert len(results) == 1

    def test_empty_directory_returns_empty(self, tmp_path):
        assert list(iter_directory_source(str(tmp_path))) == []

    def test_non_json_files_ignored(self, tmp_path):
        (tmp_path / "readme.txt").write_text("不是json", encoding="utf-8")
        (tmp_path / "a.json").write_text(json.dumps([{"text": "内容"}]), encoding="utf-8")
        results = list(iter_directory_source(str(tmp_path)))
        assert len(results) == 1


class TestStableDocIdForFile:
    def test_same_file_produces_same_doc_id(self, tmp_path):
        file_path = str(tmp_path / "a.json")
        id1 = stable_doc_id_for_file(str(tmp_path), file_path)
        id2 = stable_doc_id_for_file(str(tmp_path), file_path)
        assert id1 == id2

    def test_doc_id_uses_relative_path_prefix(self, tmp_path):
        file_path = str(tmp_path / "sub" / "a.json")
        doc_id = stable_doc_id_for_file(str(tmp_path), file_path)
        assert doc_id.startswith("file::")
        assert "sub" in doc_id

    def test_different_files_produce_different_doc_ids(self, tmp_path):
        id_a = stable_doc_id_for_file(str(tmp_path), str(tmp_path / "a.json"))
        id_b = stable_doc_id_for_file(str(tmp_path), str(tmp_path / "b.json"))
        assert id_a != id_b
