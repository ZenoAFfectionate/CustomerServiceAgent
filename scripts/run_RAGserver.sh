#!/usr/bin/env bash
# ============================================================================
# run_RAGserver.sh — 一键启动 RAG 前后端服务
# ============================================================================
# 功能：环境自检（Python / 依赖 / 配置文件）→ 自动切换工作目录 → 启动
#       FastAPI 后端（rag/api/main.py，同时提供 /ui/ 前端页面）。
#
# 适用环境：macOS / Linux（Python 3.9+），RAG 后端默认使用本地降级实现，
#          无需 Milvus/ES/TEI 即可开箱运行；生产环境请通过 .env 切换后端。
#
# 用法：
#   bash scripts/run_RAGserver.sh              # 默认 0.0.0.0:8090
#   PORT=8888 bash scripts/run_RAGserver.sh     # 自定义端口
#   HOST=127.0.0.1 PORT=9000 bash scripts/run_RAGserver.sh
#   RELOAD=1 bash scripts/run_RAGserver.sh     # 开发模式，代码变更自动重载
#
# 权限：chmod +x scripts/run_RAGserver.sh  （或直接用 bash 前缀运行）
# ============================================================================
set -euo pipefail

# ---- 颜色输出（不支持时自动降级）----
if [[ -t 1 ]] && command -v tput &>/dev/null; then
  RED=$(tput setaf 1); GREEN=$(tput setaf 2); YELLOW=$(tput setaf 3); CYAN=$(tput setaf 6); RESET=$(tput sgr0)
else
  RED=""; GREEN=""; YELLOW=""; CYAN=""; RESET=""
fi

info()  { echo "${CYAN}[INFO]${RESET}  $*"; }
ok()    { echo "${GREEN}[OK]${RESET}    $*"; }
warn()  { echo "${YELLOW}[WARN]${RESET}  $*"; }
fail()  { echo "${RED}[FAIL]${RESET}  $*"; }

# ---- 切换到项目根目录 ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
info "工作目录: ${PROJECT_ROOT}"

# ====== 1. Python 环境检查 ======
if ! command -v python3 &>/dev/null; then
  fail "未找到 python3，请先安装 Python 3.9+（https://www.python.org/downloads/）"
  exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [[ "${PY_MAJOR}" -lt 3 ]] || [[ "${PY_MAJOR}" -eq 3 && "${PY_MINOR}" -lt 9 ]]; then
  fail "Python 版本过低（当前 ${PY_VERSION}），需要 3.9+，请升级。"
  exit 1
fi
ok "Python 版本: ${PY_VERSION}"

# ====== 2. 核心依赖完整性检查 ======
info "检查核心依赖..."
MISSING_DEPS=()
check_dep() {
  if ! python3 -c "import $1" 2>/dev/null; then
    MISSING_DEPS+=("$2")
  fi
}
check_dep fastapi    "fastapi>=0.104.0"
check_dep uvicorn    "uvicorn>=0.24.0"
check_dep pydantic   "pydantic>=2.0.0"
check_dep dotenv     "python-dotenv>=1.0.0"
check_dep numpy      "numpy>=1.24.0"
check_dep bs4        "beautifulsoup4>=4.12.0"
check_dep jieba      "jieba>=0.42.1"

if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
  warn "以下依赖缺失："
  for d in "${MISSING_DEPS[@]}"; do echo "    - ${d}"; done
  echo ""
  fail "依赖不完整，无法启动。请执行以下命令安装："
  echo ""
  echo "    pip install -r requirements.txt"
  echo ""
  exit 1
fi
ok "核心依赖检查通过"

# ====== 3. 配置文件存在性检查 ======
if [[ ! -f "config/config.json" ]]; then
  if [[ -f "config/config.example.json" ]]; then
    warn "config/config.json 不存在，从模板复制一份（含安全默认值，可直接启动）..."
    cp config/config.example.json config/config.json
    ok "已生成 config/config.json"
  else
    fail "config/config.json 和 config/config.example.json 均不存在，无法启动。"
    exit 1
  fi
fi

if [[ ! -f ".env" ]]; then
  if [[ -f ".env.example" ]]; then
    warn ".env 不存在，从 .env.example 复制一份（本地降级模式无需填写即可启动）..."
    cp .env.example .env
    ok "已生成 .env（默认全部后端为 local 降级模式，开箱即用）"
  else
    warn ".env 和 .env.example 均不存在，将使用代码内默认值（本地降级模式）。"
  fi
fi
ok "配置文件就绪"

# ====== 4. RAG 模块可导入性验证 ======
info "验证 rag/ 模块可导入..."
if ! python3 -c "from rag.api.main import app; print('routes:', len(app.routes))" 2>/dev/null; then
  fail "rag/ 模块导入失败，可能存在代码错误。请手动执行排查："
  echo ""
  echo "    python3 -c \"from rag.api.main import app\""
  echo ""
  exit 1
fi
ok "rag/ 模块导入成功"

# ====== 5. 启动服务 ======
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8090}"

echo ""
info "========================================"
info "  RAG 服务即将启动"
info "  地址:        http://${HOST}:${PORT}"
info "  Swagger UI:  http://localhost:${PORT}/docs"
info "  前端页面:    http://localhost:${PORT}/ui/"
info "  健康检查:    http://localhost:${PORT}/api/health"
info "========================================"
echo ""

if [[ "${RELOAD:-0}" == "1" ]]; then
  info "开发模式（--reload）已启用"
  exec python3 -m uvicorn rag.api.main:app --host "${HOST}" --port "${PORT}" --reload
else
  exec python3 -m uvicorn rag.api.main:app --host "${HOST}" --port "${PORT}"
fi
