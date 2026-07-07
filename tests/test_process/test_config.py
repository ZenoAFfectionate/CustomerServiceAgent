# -*- coding: utf-8 -*-
"""
config 模块单元测试。

覆盖配置加载、路径解析、日志初始化。

运行方式：
    PYTHONPATH=process/src pytest process/tests/test_config.py -v
"""

import os
import pytest

from utils.config import CONFIG, logger, PROJECT_ROOT, DATA_DIR, USER_DICT_PATH


class TestConfig:
    """测试全局配置加载"""

    def test_config_is_dict(self):
        assert isinstance(CONFIG, dict)

    def test_config_has_required_keys(self):
        assert "embed_model" in CONFIG
        assert "rerank_model" in CONFIG
        assert "llm_model" in CONFIG

    def test_config_has_chunk_params(self):
        assert "max_node_words_embed" in CONFIG
        assert "min_node_words_embed" in CONFIG

    def test_project_root_exists(self):
        assert os.path.isdir(PROJECT_ROOT)

    def test_data_dir_exists(self):
        # DATA_DIR 指向 process/dataset，可能尚未创建（仅在实际运行数据处理时生成）。
        # 此处仅验证路径定义中的父目录（process/）存在。
        assert os.path.isdir(os.path.dirname(DATA_DIR))

    def test_config_file_exists(self):
        config_path = os.path.join(PROJECT_ROOT, "config", "config.json")
        assert os.path.isfile(config_path)

    def test_user_dict_path_defined(self):
        assert USER_DICT_PATH is not None
        assert "user_dict.txt" in USER_DICT_PATH

    def test_logger_has_handlers(self):
        assert len(logger.handlers) > 0
