# -*- coding: utf-8 -*-
"""
pytest 共享配置与 fixtures（项目级统一测试入口）。

## 目录组织

```
tests/
├── conftest.py            # 本文件：全局 sys.path 注入 + 共享 fixtures
├── test_process/          # process/ 模块测试（HTML 清洗、分块、去重等）
├── test_rag/              # rag/ 模块测试：按 rag/ 的 7 个子模块（retrieval/indexing/
│                          #   knowledge_base/generation/evaluation/observability/
│                          #   integration）分别组织，文件名前缀标注所属子模块，
│                          #   并包含 test_pipeline.py/test_api.py/
│                          #   test_cross_module_combinations.py 等跨模块组合测试
└── test_agent/            # agent/ 模块测试（hello_agents 框架自带测试套件）
```

## 模块分布与 import 方式

    - process/utils/：通用工具（config / llm_api / jieba_util），import 为 utils.xxx
    - process/src/  ：核心处理逻辑（html_utils / text_process / html_pruner），
                      import 为顶层模块（如 from html_utils import ...）
    - rag/          ：RAG 检索增强生成框架，import 为包（如 from rag.config import ...），
                      需要项目根目录本身在 sys.path 上
    - agent/        ：Agent 模块，核心包源码在 agent/src/（原名 hello_agents/，已改名），
                      import 为已安装包风格（如 from hello_agents import ReActAgent）。
                      agent/hello_agents.py 为兼容性垫片，将 import 透明转发到 src/。
                      需要将 agent/ 与 agent/src/ 同时加入 sys.path。

因此需将 项目根目录 / process / process/src / agent 同时加入 sys.path。本 conftest
已自动处理，无需手动设置 PYTHONPATH、无需 `pip install -e agent/`，在项目根目录运行：

    pytest tests/ -v                     # 运行全部测试
    pytest tests/test_process/ -v        # 仅运行 process/ 测试
    pytest tests/test_rag/ -v            # 仅运行 rag/ 测试
    pytest tests/test_agent/ -v          # 仅运行 agent/ 测试

> 说明：`tests/test_agent/` 下的测试用例部分依赖真实 LLM API（如
> `test_all_agents.py` 会校验 `LLM_API_KEY` 环境变量），需在根目录 `.env` 中
> 配置 `LLM_MODEL_ID`/`LLM_API_KEY`/`LLM_BASE_URL` 后才能完整通过，属于
> hello_agents 框架自身的既有测试设计，与本文件的路径注入逻辑无关。
"""

import shutil
import sys
import os

import pytest

_PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
_PROCESS_DIR = os.path.join(_PROJECT_ROOT, "process")
_AGENT_DIR = os.path.join(_PROJECT_ROOT, "agent")

# 将项目根目录加入 path：使 import rag.xxx / config.xxx / model.xxx 等包可用
sys.path.insert(0, _PROJECT_ROOT)
# 将 process 加入 path：使 import utils.xxx（config/llm_api/jieba_util）可用
sys.path.insert(0, _PROCESS_DIR)
# 将 process/src 加入 path：使顶层 import（html_utils/text_process/html_pruner）可用
sys.path.insert(0, os.path.join(_PROCESS_DIR, "src"))
# 将 agent 加入 path：使 import hello_agents 可用
# （agent/hello_agents.py 垫片会将 import 转发到 agent/src/ 包）
sys.path.insert(0, _AGENT_DIR)
# 同时将 agent/src 直接加入 path：使 hello_agents 的子模块（如 hello_agents.core.llm）
# 能被 Python 直接发现，无需依赖垫片的 importlib 转发
sys.path.insert(0, os.path.join(_AGENT_DIR, "src"))

# 使用独立的测试数据目录，避免测试污染真实的 rag/data 索引数据
os.environ.setdefault("RAG_DATA_DIR", os.path.join(os.path.dirname(__file__), "_rag_test_data"))


@pytest.fixture
def clean_rag_data():
    """清空 RAG 本地降级存储的测试数据目录，并重置全局单例，保证 rag/ 测试间互相隔离。

    用法（在测试模块顶部）：
        pytestmark = pytest.mark.usefixtures("clean_rag_data")
    """
    from rag.config import RAG_CONFIG

    data_dir = RAG_CONFIG["data_dir"]
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    import rag.indexing.metadata as metadata_mod
    import rag.indexing.embedding as embedding_mod
    from rag.indexing.store import reset_keyword_store, reset_vector_store
    from rag.observability import monitoring
    from rag.knowledge_base.versioning import reset_version_store
    metadata_mod._default_registry = None
    embedding_mod._default_embedder = None
    reset_vector_store()
    reset_keyword_store()
    reset_version_store()
    monitoring.reset()

    yield

    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    # 测试结束后重新创建空目录：`rag/config.py` 在模块导入时即假定该目录存在
    # （其他测试模块如 test_rag_config.py 也会校验该目录存在），避免因执行顺序
    # 导致目录被删除后未还原而产生跨测试文件的偶发失败。
    os.makedirs(data_dir, exist_ok=True)
