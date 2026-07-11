#!/bin/bash
# 构建 jieba 自定义词典（从 HTML 源文件中提取高频短语）
# 用法: bash scripts/build_JIEBAdict.sh [html_dir]
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROCESS_DIR="$PROJECT_ROOT/process"
# jieba_util 属于 utils 包（process/utils），需将 process 目录加入 PYTHONPATH
export PYTHONPATH="$PROCESS_DIR:$PYTHONPATH"

HTML_DIR="${1:-process/data}"

cd "$PROJECT_ROOT"
echo "▶ 从 $HTML_DIR 提取高频短语并生成词典..."
python3 -m utils.jieba_util --html-dir "$HTML_DIR" --output process/data/user_dict.txt  # [Optimized] python → python3
