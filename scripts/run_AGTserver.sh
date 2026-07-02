#!/usr/bin/env bash
# ============================================================================
# run_AGTserver.sh — 一键启动 Agent 服务（hello_agents SSE 流式输出）
# ============================================================================
# 功能：环境自检（Python / 依赖 / LLM 配置）→ 自动切换工作目录 → 启动
#       hello_agents 的 FastAPI SSE 流式输出服务端（agent/examples/fastapi_sse_server.py）。
#
# 适用环境：macOS / Linux（Python 3.10+，hello_agents 框架要求）。
#          启动前需在 .env 中配置 LLM_API_KEY / LLM_MODEL_ID / LLM_BASE_URL，
#          否则 Agent 实际调用 LLM 时会报错（服务本身可启动，但问答会失败）。
#
# 用法：
#   bash scripts/run_AGTserver.sh              # 默认 0.0.0.0:8000
#   PORT=8001 bash scripts/run_AGTserver.sh     # 自定义端口
#   RELOAD=1 bash scripts/run_AGTserver.sh     # 开发模式，代码变更自动重载
#
# 权限：chmod +x scripts/run_AGTserver.sh  （或直接用 bash 前缀运行）
#
# 说明：
#   当前 agent/ 与 rag/ 服务的工具化调用尚未接线（M4 待办），
#   本脚本启动的是 hello_agents 框架自带的 SSE 示例服务端，
#   用于验证框架可用性与体验 Agent 流式输出能力。
#   待 M4 完成后，此入口将替换为对接 rag/ 检索的业务 Agent 服务。
# ============================================================================
set -euo pipefail

# ---- 颜色输出 ----
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

# ====== 1. Python 环境检查（hello_agents 要求 3.10+）======
if ! command -v python3 &>/dev/null; then
  fail "未找到 python3，请先安装 Python 3.10+（https://www.python.org/downloads/）"
  exit 1
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [[ "${PY_MAJOR}" -lt 3 ]] || [[ "${PY_MAJOR}" -eq 3 && "${PY_MINOR}" -lt 10 ]]; then
  fail "Python 版本过低（当前 ${PY_VERSION}），hello_agents 框架要求 3.10+，请升级。"
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
check_dep openai     "openai>=1.0.0"
check_dep tiktoken   "tiktoken>=0.5.0"
check_dep networkx   "networkx>=3.0"
check_dep yaml       "pyyaml>=6.0"

if [[ ${#MISSING_DEPS[@]} -gt 0 ]]; then
  warn "以下依赖缺失："
  for d in "${MISSING_DEPS[@]}"; do echo "    - ${d}"; done
  echo ""
  fail "依赖不完整，无法启动。请执行以下命令安装："
  echo ""
  echo "    pip install -r requirements.txt"
  echo "    pip install -r agent/pyproject.toml  # 如有独立依赖声明"
  echo ""
  exit 1
fi
ok "核心依赖检查通过"

# ====== 3. hello_agents 模块可导入性验证 ======
info "验证 hello_agents 框架可导入..."
# agent/ 和 agent/src 都需在 sys.path 上（hello_agents.py 垫片 + src/ 包目录）
export PYTHONPATH="${PROJECT_ROOT}/agent:${PROJECT_ROOT}/agent/src:${PYTHONPATH:-}"
if ! python3 -c "from hello_agents import ReActAgent, HelloAgentsLLM, ToolRegistry; print('hello_agents OK')" 2>/dev/null; then
  fail "hello_agents 框架导入失败。可能原因："
  echo ""
  echo "    1. agent/hello_agents/ 目录不存在或文件不完整"
  echo "    2. 依赖缺失（见上方检查）"
  echo ""
  echo "  排查命令："
  echo "    PYTHONPATH=agent python3 -c 'import hello_agents; print(hello_agents.__file__)'"
  echo ""
  exit 1
fi
ok "hello_agents 框架导入成功"

# ====== 4. SSE 服务端入口文件存在性检查 ======
SSE_SERVER="agent/examples/fastapi_sse_server.py"
if [[ ! -f "${SSE_SERVER}" ]]; then
  fail "Agent SSE 服务端入口文件不存在: ${SSE_SERVER}"
  echo ""
  echo "  该文件是 hello_agents 框架自带的 FastAPI SSE 流式输出示例。"
  echo "  若已被移动或删除，请从 agent/ 仓库恢复。"
  echo ""
  exit 1
fi
ok "SSE 服务端入口就绪: ${SSE_SERVER}"

# ====== 5. LLM 配置检查（非阻塞警告）======
# hello_agents 通过 LLM_API_KEY / LLM_MODEL_ID / LLM_BASE_URL 自动检测 provider。
# 未配置时服务可启动（FastAPI 进程正常），但实际调用 Agent 时会因无 LLM 凭证而报错。
load_env_file() {
  local env_file="$1"
  if [[ -f "${env_file}" ]]; then
    while IFS='=' read -r key value || [[ -n "${key}" ]]; do
      # 跳过注释行和空行
      [[ "${key}" =~ ^[[:space:]]*# ]] && continue
      [[ -z "${key}" ]] && continue
      # 去除首尾空白
      key="${key//[[:space:]]/}"
      value="${value//\"/}"  # 去除引号
      # 仅在环境变量未设置时填充（不覆盖已有值）
      if [[ -z "${!key:-}" ]]; then
        export "${key}=${value}"
      fi
    done < "${env_file}"
  fi
}
load_env_file ".env"
load_env_file "agent/.env"

LLM_CONFIG_OK=1
if [[ -z "${LLM_API_KEY:-}" ]] || [[ "${LLM_API_KEY}" == "your-api-key-here" ]]; then
  warn "LLM_API_KEY 未配置（或仍为模板占位值）"
  LLM_CONFIG_OK=0
fi
if [[ -z "${LLM_MODEL_ID:-}" ]] || [[ "${LLM_MODEL_ID}" == "your-model-name" ]]; then
  warn "LLM_MODEL_ID 未配置（或仍为模板占位值）"
  LLM_CONFIG_OK=0
fi
if [[ -z "${LLM_BASE_URL:-}" ]] || [[ "${LLM_BASE_URL}" == "your-api-base-url" ]]; then
  warn "LLM_BASE_URL 未配置（或仍为模板占位值）"
  LLM_CONFIG_OK=0
fi

if [[ "${LLM_CONFIG_OK}" -eq 0 ]]; then
  echo ""
  warn "LLM 配置不完整，Agent 服务可启动但实际问答会失败。"
  echo "       请在 .env 中填写以下变量后重启："
  echo ""
  echo "         LLM_MODEL_ID=your-model-name        # 如 deepseek-chat / glm-4-flash"
  echo "         LLM_API_KEY=your-api-key-here       # 对应平台的 API 密钥"
  echo "         LLM_BASE_URL=your-api-base-url      # 如 https://api.deepseek.com/v1"
  echo ""
  echo "       模板参考 .env.example §7 节 或 agent/.env.example。"
  echo ""
  # 不 exit，允许用户先启动服务观察 /docs 页面，仅问答功能不可用
else
  ok "LLM 配置就绪（provider 将由框架自动检测）"
fi

# ====== 6. 启动服务 ======
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo ""
info "========================================"
info "  Agent 服务即将启动"
info "  地址:        http://${HOST}:${PORT}"
info "  Swagger UI:  http://localhost:${PORT}/docs"
info "  SSE 测试:    curl -N http://localhost:${PORT}/agent/stream \\"
info "               -X POST -H 'Content-Type: application/json' \\"
info "               -d '{\"input\": \"你好\"}'"
info "========================================"
echo ""

# 通过 uvicorn 启动 fastapi_sse_server（模块路径相对于 agent/ 目录）
cd "${PROJECT_ROOT}/agent"
if [[ "${RELOAD:-0}" == "1" ]]; then
  info "开发模式（--reload）已启用"
  exec python3 -m uvicorn examples.fastapi_sse_server:app --host "${HOST}" --port "${PORT}" --reload
else
  exec python3 -m uvicorn examples.fastapi_sse_server:app --host "${HOST}" --port "${PORT}"
fi
