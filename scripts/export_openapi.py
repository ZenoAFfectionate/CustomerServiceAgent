#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出 RAG FastAPI 应用的 OpenAPI Schema 为静态 JSON 文件。

用途：
    - 供无法直接访问运行中服务的 Agent/工具离线解析接口定义
    - CI 中可用于对比 API 契约是否发生非预期变更

用法：
    python scripts/export_openapi.py
    # 输出: rag/docs/openapi.json
"""
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from rag.api.main import app  # noqa: E402

OUTPUT_PATH = os.path.join(_PROJECT_ROOT, "rag", "docs", "openapi.json")


def main():
    schema = app.openapi()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)
    print(f"✅ OpenAPI schema 已导出: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
