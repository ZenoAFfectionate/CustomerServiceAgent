#!/usr/bin/env bash
# ============================================================================
# start_tei.sh — 一键启动 TEI (Text Embeddings Inference) 模型服务
# ============================================================================
# 功能：环境自检（Docker / docker compose）→ 启动 Embedding + Reranker 两个
#       TEI 容器 → 健康检查轮询 → 状态报告。
#
# 适用环境：需安装 Docker + NVIDIA GPU 驱动 + nvidia-container-toolkit。
#          无 GPU 环境下 TEI 容器无法启动，但 rag/ 会自动降级为本地实现。
#
# 启动依赖关系：
#   TEI 服务（本脚本）→ rag/ 检索服务（run_RAGserver.sh）
#   TEI 是 rag/ 的上游依赖，但 rag/ 设计了优雅降级：
#   - TEI 不可用时 rag/ 自动降级为本地哈希嵌入 + 余弦代理精排
#   - 因此可以先启动 rag/ 再启动 TEI，TEI 就绪后 rag/ 会自动开始使用真实服务
#
# 用法：
#   bash scripts/start_tei.sh              # 前台启动（日志输出到终端）
#   bash scripts/start_tei.sh --check      # 仅健康检查，不启动
#   bash scripts/start_tei.sh --down       # 停止 TEI 服务
#
# 权限：chmod +x scripts/start_tei.sh  （或直接用 bash 前缀运行）
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

COMPOSE_FILE="model/inference/docker-compose-tei.yml"
EMBED_URL="${TEI_EMBED_URL:-http://localhost:8080}"
RERANK_URL="${TEI_RERANK_URL:-http://localhost:8081}"

# ====== 1. Docker 环境检查 ======
if ! command -v docker &>/dev/null; then
  fail "未找到 docker，请先安装 Docker（https://docs.docker.com/get-docker/）"
  exit 1
fi
ok "Docker 已安装: $(docker --version)"

if ! docker compose version &>/dev/null 2>&1; then
  if command -v docker-compose &>/dev/null; then
    info "检测到旧版 docker-compose，将使用 docker-compose 命令"
    COMPOSE_CMD="docker-compose"
  else
    fail "docker compose 不可用。请安装 Docker Compose V2（docker compose 子命令）。"
    exit 1
  fi
else
  COMPOSE_CMD="docker compose"
fi
ok "Compose 命令: ${COMPOSE_CMD}"

# ====== 2. Compose 文件存在性检查 ======
if [[ ! -f "${COMPOSE_FILE}" ]]; then
  fail "Compose 文件不存在: ${COMPOSE_FILE}"
  echo ""
  echo "  请确认 model/inference/docker-compose-tei.yml 文件存在。"
  exit 1
fi
ok "Compose 文件就绪: ${COMPOSE_FILE}"

# ====== 3. NVIDIA GPU 可用性检查（非阻塞警告）======
if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
  warn "未检测到 NVIDIA Container Runtime。"
  echo "       TEI 的 cuda 镜像需要 GPU + nvidia-container-toolkit。"
  echo "       安装指南: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
  echo ""
  echo "       继续启动（容器可能因 GPU 不可用而失败）..."
fi

# ====== 4. 子命令分发 ======

# --- 健康检查函数 ---
health_check() {
  local service_name="$1"
  local url="$2"
  local max_retries="${3:-30}"
  local retry_interval=2

  info "健康检查: ${service_name} (${url}/health)"
  for i in $(seq 1 "${max_retries}"); do
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "${url}/health" 2>/dev/null || echo "000")
    if [[ "${http_code}" == "200" ]]; then
      ok "${service_name} 就绪 (尝试 ${i}/${max_retries})"
      return 0
    fi
    printf "\r  等待中... %d/%d (HTTP %s)" "${i}" "${max_retries}" "${http_code}"
    sleep "${retry_interval}"
  done
  echo ""
  fail "${service_name} 健康检查超时（${max_retries} 次尝试，每次间隔 ${retry_interval}s）"
  return 1
}

case "${1:-up}" in

  # --- 健康检查模式 ---
  --check)
    echo ""
    info "=== TEI 服务健康检查 ==="
    echo ""
    local_ok=0
    health_check "Embedding 服务" "${EMBED_URL}" 1 || local_ok=1
    health_check "Reranker 服务" "${RERANK_URL}" 1 || local_ok=1
    echo ""
    if [[ "${local_ok}" -eq 0 ]]; then
      ok "全部 TEI 服务健康"
    else
      fail "部分 TEI 服务不可用（rag/ 将自动降级为本地实现）"
    fi
    exit "${local_ok}"
    ;;

  # --- 停止模式 ---
  --down)
    echo ""
    info "停止 TEI 服务..."
    ${COMPOSE_CMD} -f "${COMPOSE_FILE}" down
    ok "TEI 服务已停止"
    exit 0
    ;;

  # --- 启动模式（默认） ---
  up|*)
    echo ""
    info "=== 启动 TEI 模型服务 ==="
    echo ""
    info "Embedding 模型: Qwen/Qwen3-Embedding-4B (端口 8080)"
    info "Reranker 模型:  Qwen/Qwen3-Reranker-4B (端口 8081)"
    info "Compose 文件:   ${COMPOSE_FILE}"
    echo ""

    # 启动容器（-d 后台运行）
    info "拉取镜像并启动容器（首次可能需要下载镜像，请耐心等待）..."
    ${COMPOSE_CMD} -f "${COMPOSE_FILE}" up -d

    echo ""
    info "=== 健康检查（等待模型加载） ==="
    echo ""
    health_check "Embedding 服务" "${EMBED_URL}" 60 || true
    health_check "Reranker 服务" "${RERANK_URL}" 60 || true

    echo ""
    info "========================================"
    info "  TEI 服务状态"
    info "  Embedding:  ${EMBED_URL}"
    info "  Reranker:   ${RERANK_URL}"
    info "  健康检查:    ${EMBED_URL}/health"
    info "               ${RERANK_URL}/health"
    info "========================================"
    echo ""
    info "停止服务:  bash scripts/start_tei.sh --down"
    info "检查状态: bash scripts/start_tei.sh --check"
    echo ""

    # 显示容器状态
    ${COMPOSE_CMD} -f "${COMPOSE_FILE}" ps
    ;;

esac
