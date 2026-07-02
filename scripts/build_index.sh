#!/usr/bin/env bash
# [Optimized] 增强 build_index.sh：set -euo pipefail + Python 环境检查 + 依赖检查
# 构建 RAG 索引：读取 process/ 输出的知识块 JSON，写入向量库 + 关键词库。
# 默认使用本地降级后端（无需 Milvus/ES/TEI，开箱即用）。
#
# 用法:
#   bash scripts/build_index.sh
#   SOURCE_DIR=process/dataset/html_cleaned_block bash scripts/build_index.sh
#
#   # 切换到真实 Milvus/ES/TEI 后端
#   RAG_VECTOR_BACKEND=milvus RAG_KEYWORD_BACKEND=es RAG_EMBED_BACKEND=tei \
#       bash scripts/build_index.sh
set -euo pipefail

# ---- 颜色输出 ----
if [[ -t 1 ]] && command -v tput &>/dev/null; then
  RED=$(tput setaf 1); GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); CYAN=$(tput setaf 6); RESET=$(tput sgr0)
else
  RED=""; GREEN=""; YELLOW=""; CYAN=""; RESET=""
fi

info()  { echo "${CYAN}[INFO]${RESET}  $*"; }
ok()    { echo "${GREEN}[OK]${RESET}    $*"; }
fail()  { echo "${RED}[FAIL]${RESET}  $*"; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
info "工作目录: ${PROJECT_ROOT}"

# ====== Python 环境检查 ======
if ! command -v python3 &>/dev/null; then
  fail "未找到 python3，请先安装 Python 3.9+"
  exit 1
fi
PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ok "Python: ${PY_VERSION}"

# ====== 核心依赖检查 ======
info "检查核心依赖..."
MISSING=()
for dep in fastapi numpy jieba bs4; do
  if ! python3 -c "import ${dep}" 2>/dev/null; then
    MISSING+=("${dep}")
  fi
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
  fail "以下依赖缺失: ${MISSING[*]}"
  echo "  请执行: pip install -r requirements.txt"
  exit 1
fi
ok "核心依赖检查通过"

SOURCE_DIR="${SOURCE_DIR:-process/dataset/html_cleaned_block}"

echo "=========================================="
echo "  RAG 索引构建"
echo "  源目录: $SOURCE_DIR"
echo "  向量库后端: ${RAG_VECTOR_BACKEND:-local}"
echo "  关键词库后端: ${RAG_KEYWORD_BACKEND:-local}"
echo "=========================================="

if [[ ! -d "${SOURCE_DIR}" ]]; then
  fail "源目录不存在: ${SOURCE_DIR}"
  echo "  请先运行 bash scripts/process_HTMLdata.sh 生成知识块 JSON。"
  exit 1
fi

python3 scripts/build_index.py --source-dir "$SOURCE_DIR"
