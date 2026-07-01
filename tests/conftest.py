# -*- coding: utf-8 -*-
"""
pytest 共享配置与 fixtures。

模块分布：
    - process/utils/：通用工具（config / llm_api / jieba_util），import 为 utils.xxx
    - process/src/  ：核心处理逻辑（html_utils / text_process_utils / html_pruner /
                      main），import 为顶层模块（如 from html_utils import ...）

因此需同时将 process 与 process/src 加入 sys.path。本 conftest 已自动处理，
无需额外设置 PYTHONPATH 即可运行：

    pytest tests/ -v
"""

import sys
import os

_PROCESS_DIR = os.path.join(os.path.dirname(__file__), "..", "process")
# 将 process 加入 path：使 import utils.xxx（config/llm_api/jieba_util）可用
sys.path.insert(0, _PROCESS_DIR)
# 将 process/src 加入 path：使顶层 import（html_utils/text_process_utils/html_pruner）可用
sys.path.insert(0, os.path.join(_PROCESS_DIR, "src"))
