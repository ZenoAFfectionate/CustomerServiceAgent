# -*- coding: utf-8 -*-
"""数据来源接入（Data Sources）：定义知识库内容的输入方式。

当前支持三类来源，均落到 `indexing/document_loader.py` + `indexing/index_builder.py`
可消费的形式（文本或已成型的知识块数组）：

    - "upload":         用户通过 API 上传的单个文件（bytes）
    - "process_blocks": `process/` 模块产出的知识块 JSON（list[dict]）
    - "directory":      本地目录下批量扫描的知识块 JSON 文件（如 `process/data/数据集名_blocked/`）

新增数据来源（如数据库表、网页爬虫）时，只需新增一个 `iter_xxx_source()`
函数，不影响已有来源的处理逻辑。
"""
import glob
import json
import os
from typing import Iterator, List, Tuple

from config.config_loader import logger


def iter_directory_source(source_dir: str) -> Iterator[Tuple[str, List[dict]]]:
    """递归扫描目录下的全部 `*.json` 文件，逐个解析为知识块数组。

    Args:
        source_dir: 知识块 JSON 所在目录（如 process/ 的输出目录）

    Yields:
        (文件路径, 知识块列表) —— 单个 JSON 对象会被自动包装为长度 1 的数组；
        空文件/无效 JSON 会被跳过（不抛异常中断整体扫描，但会记录告警日志，
        避免批量导入时用户无法感知哪些文件被静默忽略）。
    """
    pattern = os.path.join(source_dir, "**", "*.json")
    for path in sorted(glob.glob(pattern, recursive=True)):
        try:
            with open(path, "r", encoding="utf-8") as f:
                blocks = json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ 跳过无法解析的文件 {path}: {e}")
            continue
        if isinstance(blocks, dict):
            blocks = [blocks]
        if not isinstance(blocks, list) or not blocks:
            logger.warning(f"⚠️ 跳过空知识块文件: {path}")
            continue
        yield path, blocks


def stable_doc_id_for_file(source_dir: str, file_path: str) -> str:
    """为目录扫描来源生成稳定的 doc_id（基于相对路径），保证重复运行时
    幂等 —— 与旧 `scripts/build_index.py` 的约定一致：`file::<相对路径>`。
    """
    return f"file::{os.path.relpath(file_path, source_dir)}"
