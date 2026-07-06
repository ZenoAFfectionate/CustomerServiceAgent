# -*- coding: utf-8 -*-
"""配置读取公共辅助函数（Env Utils）。

【修复审查报告 L3】`config/config_loader.py`、`rag/config.py`、
`process/utils/config.py` 此前各自独立定义了一份几乎完全相同的
`_env_or_config(key, config, default, cast)` 实现——"环境变量优先，其次
配置字典，最后默认值，并按 cast 类型转换"这份逻辑改动一处（如修复某个
类型转换边界情况）容易漏改另外两处，产生行为不一致。

本模块提供唯一实现，供以上三处统一 `from config.env_utils import env_or_config`
导入复用。故意保持零外部依赖（仅 `os`），避免被拉入任何模块的循环导入。
"""
import os
from typing import Any, Callable, Dict, Optional


def env_or_config(
    key: str,
    config: Dict[str, Any],
    default: Any = None,
    cast: Callable[[Any], Any] = str,
) -> Any:
    """从环境变量或 config 字典中读取值，环境变量优先。

    Args:
        key: 环境变量名（同时用作 config 字典中的 key）
        config: 配置字典（通常是 `config/config.json` 解析结果），环境变量
            未设置时从此处按 key 读取
        default: 环境变量与 config 中均未提供该 key 时的回退默认值
        cast: 类型转换函数；对 `bool` 类型特殊处理（识别常见的
            "1"/"true"/"yes"/"on" 等真值字符串，而非直接 `bool("false")`
            这种恒为 True 的错误转换）

    Returns:
        转换后的配置值；转换失败时返回未转换的原始值（不抛异常，保证配置
        加载阶段的健壮性优先于类型严格性）。
    """
    env_val = os.environ.get(key)
    if env_val is not None and env_val != "":
        try:
            if cast is bool:
                return env_val.strip().lower() in ("1", "true", "yes", "on")
            return cast(env_val)
        except (ValueError, TypeError):
            return env_val
    val = config.get(key, default)
    if val is not None and cast is not str:
        try:
            return cast(val)
        except (ValueError, TypeError):
            pass
    return val
