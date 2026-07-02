#!/bin/bash
# 一键运行数据处理全流程：清洗 + 分块
# 用法: bash scripts/process_HTMLdata.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROCESS_DIR="$PROJECT_ROOT/process"
# utils 位于 process/，入口 main.py 位于 process/src/，两者都需在 PYTHONPATH 中
export PYTHONPATH="$PROCESS_DIR:$PROCESS_DIR/src:$PYTHONPATH"

HTML_SOURCE="${HTML_SOURCE:-process/dataset/html_source}"
USE_VLLM="${USE_VLLM:-}"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "  数据处理全流程（清洗 + 分块）"
echo "  源目录: $HTML_SOURCE"
echo "=========================================="

echo ""
echo "▶ 运行清洗 + 分块..."
if [ -n "$USE_VLLM" ]; then
    python3 -m main --source-dir "$HTML_SOURCE" --use-vllm  # [Optimized] python → python3，兼容现代 macOS/Linux 无 python 别名的环境
else
    python3 -m main --source-dir "$HTML_SOURCE"  # [Optimized] python → python3
fi

echo ""
echo "✅ 数据处理完成，分块结果保存在 process/dataset/html_cleaned_block/"
