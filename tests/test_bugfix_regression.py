# -*- coding: utf-8 -*-
"""
本次 Bug 修复的回归测试。

覆盖：
- P0-1: process/utils/config.py 路径计算（PROCESS_ROOT/PROJECT_ROOT/DATA_DIR）
- P1-4: USER_DICT_PATH 与 DATA_DIR 一致
- P1-5: 异步分块 generate_block_documents_async 与同步版对混合内容（文本+表格）分块结果一致
- P2-1: deduplicate_ranked_blocks_pal 大集群去重（deque BFS）无崩溃且保留最新
- P2-2: _is_ui_noise_text 预编译正则行为正确
"""
import os
import asyncio

import pytest

from utils import config as cfg
from utils.config import PROCESS_ROOT, PROJECT_ROOT, DATA_DIR, USER_DICT_PATH
from html_utils import clean_html, build_block_tree, expand_table_spans
import text_process_utils as tpu
from text_process_utils import (
    generate_block_documents,
    generate_block_documents_async,
    deduplicate_ranked_blocks_pal,
    _is_ui_noise_text,
)


# ======================== P0-1 / P1-4：路径正确性 ========================

class TestConfigPaths:
    def test_process_root_is_process_dir(self):
        assert os.path.basename(PROCESS_ROOT) == "process"
        assert os.path.isdir(PROCESS_ROOT)

    def test_project_root_is_parent_of_process(self):
        assert PROJECT_ROOT == os.path.dirname(PROCESS_ROOT)
        assert os.path.basename(PROJECT_ROOT) == "CustomerServiceAgent"

    def test_data_dir_under_process(self):
        assert DATA_DIR == os.path.join(PROCESS_ROOT, "dataset")

    def test_user_dict_path_consistent_with_data_dir(self):
        # P1-4：词典路径必须位于 DATA_DIR 下，保证写入与读取一致
        assert USER_DICT_PATH == os.path.join(DATA_DIR, "user_dict.txt")

    def test_config_path_exists(self):
        assert os.path.isfile(cfg.CONFIG_PATH)


# ======================== P2-2：UI 噪声文本判定 ========================

class TestUiNoiseText:
    @pytest.mark.parametrize("text", ["0%", "50%", "PROGRESS", "CONTENTS", "目录", "尚未开始", "↑顶部"])
    def test_noise_matched(self, text):
        assert _is_ui_noise_text(text) is True

    @pytest.mark.parametrize("text", ["广告限流后怎么办", "商品详情", "50% 折扣活动规则"])
    def test_normal_not_matched(self, text):
        assert _is_ui_noise_text(text) is False


# ======================== P2-1：去重大集群 ========================

class TestDeduplicateLargeCluster:
    def test_large_duplicate_cluster_keeps_latest(self):
        from datetime import datetime, timedelta
        base = datetime(2025, 1, 1)
        docs = [
            {"text": "完全相同的内容" * 20, "page_name": "same_page",
             "time": (base + timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")}
            for i in range(120)
        ]
        result = deduplicate_ranked_blocks_pal(docs)
        # 全部重复 → 只保留 1 条，且为时间最新
        assert len(result) == 1
        assert result[0]["time"] == (base + timedelta(days=119)).strftime("%Y-%m-%d %H:%M:%S")

    def test_distinct_docs_all_kept(self):
        docs = [
            {"text": f"完全不同的主题内容第{i}篇讲述了不一样的业务规则和操作方式" * 3,
             "page_name": f"page_{i}", "time": "2025-01-01 00:00:00"}
            for i in range(5)
        ]
        result = deduplicate_ranked_blocks_pal(docs)
        assert len(result) == 5


# ======================== P1-5：异步/同步分块混合内容一致性 ========================

def _build_mixed_block_tree():
    """构造包含"标题 + 正文 + 表格"的混合内容块。"""
    html = (
        "<html><body>"
        "<h1>促销规则</h1>"
        "<p>" + "这是一段关于促销活动的详细正文说明内容。" * 5 + "</p>"
        "<table>"
        "<tr><td>档位</td><td>门槛</td></tr>"
        "<tr><td>一档</td><td>满100减10</td></tr>"
        "<tr><td>二档</td><td>满200减30</td></tr>"
        "</table>"
        "</body></html>"
    )
    cleaned = clean_html(html)
    expanded = expand_table_spans(cleaned)
    blocks, _ = build_block_tree(expanded, max_node_words=200, min_node_words=5, zh_char=True)
    return blocks


class TestAsyncSyncConsistency:
    def test_async_mixed_content_aligns_with_sync(self, monkeypatch):
        blocks = _build_mixed_block_tree()

        # 同步版（use_vllm=False，摘要走截断，不触发网络）
        sync_meta = generate_block_documents(
            blocks, max_node_words=200, page_url="t.html", use_vllm=False
        )

        # 异步版：monkeypatch 摘要生成，避免真实网络调用
        async def _fake_summary(text, page_url, *a, **k):
            return ""
        monkeypatch.setattr(tpu, "generate_summary_vllm_async", _fake_summary)

        async_meta = asyncio.run(generate_block_documents_async(
            blocks, max_node_words=200, page_url="t.html", use_vllm=True
        ))

        # 关键：异步版必须与同步版产出相同数量的块（修复前异步版会丢失表格外正文）
        assert len(async_meta) == len(sync_meta)
        # 且正文与表格内容都被保留
        all_text = " ".join(d["text"] for d in async_meta)
        assert "促销活动" in all_text
        assert "满100减10" in all_text

    def test_async_produces_block_path_and_html(self, monkeypatch):
        blocks = _build_mixed_block_tree()

        async def _fake_summary(text, page_url, *a, **k):
            return ""
        monkeypatch.setattr(tpu, "generate_summary_vllm_async", _fake_summary)

        async_meta = asyncio.run(generate_block_documents_async(
            blocks, max_node_words=200, page_url="t.html", use_vllm=True
        ))
        assert len(async_meta) >= 1
        for d in async_meta:
            assert "block_path" in d and "html_content" in d
