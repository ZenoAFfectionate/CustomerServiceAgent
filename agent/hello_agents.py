"""兼容性垫片：agent/hello_agents/ 目录已改名为 agent/src/，
但所有 import 语句仍使用 `from hello_agents import ...`。

当 agent/ 在 sys.path 上时，Python 找到本文件（hello_agents.py），
本文件将 src/ 加入 sys.path 前端，然后用 importlib 从 src/ 加载真正的
hello_agents 包，并将自身替换为真正的包模块，实现透明转发。
"""
import importlib.util
import os
import sys

_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# 从 src/__init__.py 加载真正的 hello_agents 包
_pkg_path = os.path.join(_src_dir, "__init__.py")
_spec = importlib.util.spec_from_file_location(
    "hello_agents", _pkg_path,
    submodule_search_locations=[_src_dir],
)
_real_module = importlib.util.module_from_spec(_spec)
sys.modules["hello_agents"] = _real_module
_spec.loader.exec_module(_real_module)
