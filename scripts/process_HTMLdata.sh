#!/bin/bash
# 一键运行数据处理全流程：清洗 + 分块
# 用法: bash scripts/process_HTMLdata.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PROCESS_DIR="$PROJECT_ROOT/process"
# utils 位于 process/，入口 main.py 位于 process/src/，两者都需在 PYTHONPATH 中
export PYTHONPATH="$PROCESS_DIR:$PROCESS_DIR/src:$PYTHONPATH"

HTML_SOURCE="${HTML_SOURCE:-process/data}"
# 以下变量可覆盖默认的清洗/分块输出目录（默认自动推导为 {source-dir}_cleaned / {source-dir}_blocked）
TARGET_DIR="${TARGET_DIR:-}"
BLOCK_OUTPUT_DIR="${BLOCK_OUTPUT_DIR:-}"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "  数据处理全流程（清洗 + 分块）"
echo "  源目录: $HTML_SOURCE"
echo "=========================================="

# 构建命令行参数
MAIN_ARGS="--source-dir \"$HTML_SOURCE\""
if [ -n "$TARGET_DIR" ]; then
    MAIN_ARGS="$MAIN_ARGS --target-dir \"$TARGET_DIR\""
fi
if [ -n "$BLOCK_OUTPUT_DIR" ]; then
    MAIN_ARGS="$MAIN_ARGS --block-output-dir \"$BLOCK_OUTPUT_DIR\""
fi

echo ""
echo "▶ 运行清洗 + 分块..."
eval "python3 -m main $MAIN_ARGS"

echo ""
echo "✅ 数据处理完成"
echo "   清洗输出: ${TARGET_DIR:-${HTML_SOURCE}_cleaned}/"
echo "   分块输出: ${BLOCK_OUTPUT_DIR:-${HTML_SOURCE}_blocked}/"
