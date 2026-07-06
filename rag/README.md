# 🔍 rag/ —— RAG 检索增强生成模块

> 按"能力"重新划分为 7 个子模块（检索 / 索引 / 知识库 / 生成 / 评估 / 观测 / 集成），
> 每个子模块职责单一、接口清晰，**任一组件可独立替换**而不影响其它模块。

---

## 📑 目录

- [模块定位](#模块定位)
- [核心设计：真实服务 vs 本地降级](#核心设计真实服务-vs-本地降级)
- [快速开始](#快速开始)
- [配置项](#配置项)
- [Python API 用法](#python-api-用法)
- [与 process/ 的对接](#与-process-的对接)
- [与 model/ 的对接](#与-model-的对接)
- [与 agent/ 的对接](#与-agent-的对接)
- [测试](#测试)
- [日志与可追溯性](#日志与可追溯性)
- [已知限制](#已知限制)

---

## 模块定位

![RAG 7 子模块架构图](../images/RAGstructure.png)

> 上图展示 `rag/` 的 7 个子模块（retrieval / indexing / knowledge_base /
> generation / evaluation / observability / integration）以及 pipeline.py 全流程
> 编排入口、双后端存储设计和观测层包裹的整体架构组织方式。

---

## 核心设计：真实服务 vs 本地降级

TODO.md 中规划的生产链路依赖 **Milvus + Elasticsearch + TEI（Embedding/Reranker）+ vLLM**。
但这些服务往往不是"随手就有"的，为了让本系统**开箱即用**（不部署任何外部服务也能跑通完整
RAG 全链路）并且**方便单测**（无需 mock 网络请求），每个关键组件都设计了两套后端：

| 组件 | 生产后端（`xxx_backend=...`） | 本地降级后端（默认） | 切换方式 |
|------|------------------------------|----------------------|---------|
| 向量库 | `milvus`（Milvus 2.x + HNSW/COSINE） | `local`（JSON 持久化 + numpy 暴力余弦检索） | `RAG_VECTOR_BACKEND=milvus` |
| 关键词库 | `es`（Elasticsearch + IK 分词 + `build_optimal_jieba_query`） | `local`（jieba 分词 + TF-IDF 余弦相似度） | `RAG_KEYWORD_BACKEND=es` |
| Embedding | `tei`（对接 `model/inference/tei_client.py`） | `local`（特征哈希 Hashing Trick，确定性、无需训练） | `RAG_EMBED_BACKEND=tei` |
| Reranker | `tei`（TEI 交叉编码精排） | `local`（Embedder 余弦相似度代理） | `RAG_RERANK_BACKEND=tei` |
| 生成 | `vllm`（OpenAI 兼容 Chat 接口，走 `config.json` 的 `vllm_api_url`） | `local`（抽取式摘录回答，不调用任何 LLM） | `RAG_GENERATION_BACKEND=vllm` |

**优雅降级**：即使显式配置为生产后端，服务连接失败时也会自动降级并打印告警日志（不会导致
请求 500 报错），例如：
- `indexing/embedding.py`：TEI 健康检查失败 → 自动切换本地哈希嵌入
- `retrieval/rerank.py`：TEI 不可用 → 按 DoD 要求跳过精排，直接返回融合结果
- `generation/llm_generation.py`：vLLM 不可用 → 降级为抽取式摘录回答

> ⚠️ 本地降级实现的检索精度**明显低于**真实的 Embedding/Reranker 模型（哈希嵌入是词袋近似，
> 非真实语义向量），仅用于演示、集成测试与无 GPU 环境下的快速验证。**生产部署务必切换到
> 真实后端**，切换前可运行 `python -m rag.integration.deployment` 做部署就绪自检。

---

## 快速开始

### 1. 安装依赖

```bash
cd CustomerServiceAgent
pip install -r requirements.txt
```

### 2. 直接启动（本地降级后端，零外部依赖）

```bash
bash scripts/run_RAGserver.sh
# 或: python -m uvicorn rag.api.main:app --host 0.0.0.0 --port 8090 --reload
```

打开浏览器访问：
- 前端页面：http://localhost:8090/ui/（内置「💬 问答」/「🔍 检索测试」/「📊 监控看板」三个标签页）
- Swagger UI：http://localhost:8090/docs
- 健康检查：http://localhost:8090/api/health
- 运维看板（原始 JSON）：http://localhost:8090/api/dashboard

在前端页面上传一份 `.txt`/`.md` 文档，即可在"问答"标签页体验检索增强问答；"检索测试"
标签页可查看原始检索结果与分数；"监控看板"标签页可视化展示 `/api/dashboard` 的请求量、
延迟分位数、后端使用分布、阈值告警与各阶段错误明细（每 10s 自动刷新，支持手动刷新）。
文档管理侧边栏点击某个文档还可查看其**版本历史**（历次导入的内容哈希/块数/时间，
对接 `GET /api/documents/{doc_id}/history`）。

### 3. 生产部署（切换到真实服务）

```bash
# 1. 部署 Milvus / Elasticsearch（自行部署或使用云服务）
# 2. 部署 TEI Embedding/Reranker（见 scripts/start_tei.sh 或 model/inference/docker-compose-tei.yml）
bash scripts/start_tei.sh  # 推荐：含健康检查
# 或手动: cd model/inference && docker compose -f docker-compose-tei.yml up -d && cd -

# 3. 配置环境变量（.env，参考 .env.example）
export MILVUS_HOST_DEV=127.0.0.1
export ES_HOST_DEV=127.0.0.1
export TEI_EMBED_URL=http://localhost:8080
export TEI_RERANK_URL=http://localhost:8081
export VLLM_API_URL=http://localhost:8011/v1/chat/completions

# 4. 切换后端
export RAG_VECTOR_BACKEND=milvus
export RAG_KEYWORD_BACKEND=es
export RAG_EMBED_BACKEND=tei
export RAG_RERANK_BACKEND=tei
export RAG_GENERATION_BACKEND=vllm

# 5. 部署就绪自检（聚合各组件 health_check，见 rag/integration/deployment.py）
python -m rag.integration.deployment

# 6. 构建索引（从 process/ 输出批量导入，按内容哈希增量同步）
bash scripts/build_index.sh

# 7. 启动服务
bash scripts/run_RAGserver.sh
```

---

## 配置项

全部配置见 `rag/config.py` 的 `RAG_CONFIG`，可通过环境变量覆盖：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `RAG_VECTOR_BACKEND` | `local` | `local` / `milvus` |
| `RAG_KEYWORD_BACKEND` | `local` | `local` / `es` |
| `RAG_EMBED_BACKEND` | `local` | `local` / `tei` |
| `RAG_RERANK_BACKEND` | `local` | `local` / `tei` |
| `RAG_GENERATION_BACKEND` | `local` | `local` / `vllm` |
| `RAG_TOP_K_RECALL` | `20` | 单路召回条数 |
| `RAG_TOP_K_FINAL` | `5` | 精排后返回条数 |
| `RAG_FUSION_METHOD` | `rrf` | `rrf` / `weighted` |
| `RAG_FUSION_WEIGHT_MILVUS` / `RAG_FUSION_WEIGHT_ES` | `0.5` / `0.5` | 加权融合权重 |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `500` / `50` | 通用文本分块大小/重叠 |
| `RAG_UPLOAD_MAX_SIZE_MB` | `20` | 上传文件大小上限 |
| `RAG_HALLUCINATION_APPEND_CAVEAT` | `false` | 生成关联度过低时是否在回答末尾追加提示语（见 `generation/hallucination_control.py`） |
| `RAG_DATA_DIR` | `rag/data` | 本地降级后端持久化目录 |
| `MILVUS_HOST_DEV` / `ES_HOST_DEV` / ... | 见 `.env.example` | 复用项目级 `config/config_loader.py` 配置 |

```bash
python -c "from rag.config import RAG_CONFIG; print(RAG_CONFIG)"
```

---

## Python API 用法

### 全流程编排（检索 / 问答）

```python
from rag.pipeline import retrieve, answer

# 仅检索，不生成答案
results = retrieve("广告限流后怎么办", top_k=5)
for r in results:
    print(r["score"], r["title"], r["page_url"])

# 端到端问答
result = answer("广告限流后怎么办")
print(result["answer"])
print(result["citations"])
```

### 知识库管理（推荐入口，替代直接调用 `indexing/`）

```python
from rag.knowledge_base import corpus_management, update_sync

# 导入 process/ 输出的知识块（含质量检查 + 版本记录）
meta = corpus_management.ingest_blocks(json.load(open("process/dataset/html_cleaned_block/xxx.json")))

# 导入普通文档（自动分块）
meta = corpus_management.ingest_upload("faq.txt", open("faq.txt", "rb").read())

# 目录级增量同步：内容未变化的文件自动跳过，避免重复索引
report = update_sync.sync_directory("process/dataset/html_cleaned_block")
print(report["ingested"], report["skipped_unchanged"])
```

### 评估

```python
from rag.evaluation.retrieval_eval import evaluate_retrieval
from rag.evaluation.generation_eval import evaluate_generation

cases = [{"query": "怎么退款", "relevant_ids": [1, 2]}]
print(evaluate_retrieval(cases, top_k=5))
print(evaluate_generation([{"query": "怎么退款"}]))
```

### 观测

```python
from rag.observability import monitoring
from rag.observability.alerting import check_alerts

snapshot = monitoring.snapshot()   # 延迟分位数 / 后端使用分布 / 错误率
print(check_alerts(snapshot))       # 超过阈值的告警列表
```

### 集成到 Agent

```python
from rag.integration.tool_usage import RAG_TOOL_SCHEMAS, dispatch_tool_call

# RAG_TOOL_SCHEMAS 可直接注册进支持 OpenAI Function Calling 风格的 Agent 框架
result = dispatch_tool_call("rag_answer", {"query": "退款政策是什么"})
```

---

## 与 process/ 的对接

- `rag/schema.py` 的字段定义**完全对齐** `process/` 的输出（`chunk_idx/page_name/title/
  page_url/text/html_content/block_path/summary/question/time`），因此 `process/` 产出的
  JSON 可通过 `POST /api/documents/ingest_blocks` 或 `corpus_management.ingest_blocks()`
  直接导入，无需转换。
- `rag/retrieval/hybrid_search.py` 的去重复用 `process/src/text_process.py` 的
  `deduplicate_ranked_blocks_pal`（TF-IDF + cosine，按时间保留最新版本）。
- `rag/indexing/es_index.py` 的关键词查询构建复用 `build_optimal_jieba_query`。
- `rag/retrieval/query_rewrite.py` 复用 `process/utils/llm_api.rewrite_query_vllm` 做多轮
  指代补全。
- `process/src/html_pruner.py` 的两阶段剪枝（`two_stage_prune`）尚未接入 `rag/pipeline.py`
  （见 TODO.md「深化方向 A」），后续可在精排后、送入生成前对 `html_content` 做剪枝以进一步
  压缩上下文 token。

---

## 与 model/ 的对接

- `rag/indexing/embedding.py` 与 `rag/retrieval/rerank.py` 的 `tei` 后端直接复用
  `model/inference/tei_client.py` 的 `get_tei_client()` 单例。
- `model/utils/build_dataset.py` 调用 `rag.retrieval.hybrid_search` 的
  `vector_search()`/`keyword_search()`/`fuse()` 做真实双模检索（见该文件顶部注释与 TODO.md T6）。

---

## 与 agent/ 的对接

`rag/integration/` 提供了将 RAG 能力接入 Agent 框架的适配层：

- `agent_integration.py`：`rag_retrieve_tool()`/`rag_answer_tool()`，入参出参均为普通
  dict、**不抛异常**（失败时返回 `{"ok": False, "error": ...}`），符合 Agent 工具函数的
  调用约定。
- `tool_usage.py`：`RAG_TOOL_SCHEMAS`（OpenAI Function Calling 风格的工具定义）+
  `dispatch_tool_call(name, arguments)`（按名称路由的最小调度器）。
- `api_integration.py`：如需把 RAG 的 HTTP 接口挂载进 Agent 所在的同一 FastAPI 进程
  （而不是独立部署），可用 `mount_rag_api(host_app, prefix="/rag/api")`。

Agent 侧（`agent/`）只需在其 `ToolRegistry` 中注册 `rag_retrieve_tool`/`rag_answer_tool`
（或直接使用 `RAG_TOOL_SCHEMAS` + `dispatch_tool_call`），即可让 `ReActAgent` 在多轮对话中
自主判断"是否需要检索"并调用，`rag/` 服务本身可独立开发、测试、部署、扩容。

---

## 测试

```bash
# 运行全部 rag/ 测试（纯本地降级后端，无需任何外部服务）
pytest tests/test_rag/ -v

# 运行全部项目测试（process/ + rag/）
pytest tests/ -v
```

测试覆盖（`tests/test_rag/` 按 rag/ 的 7 个子模块分文件组织，文件名前缀标注所属子模块）

---

## 已知限制

- 本地降级后端（`local`）不适合大规模生产知识库（`LocalVectorStore` 每次检索做全量暴力
  余弦搜索，`LocalKeywordStore` 每次检索重新拟合 TF-IDF），仅用于演示与中小规模场景。
- 本地哈希嵌入（`local_hash_embedding`）是词袋近似而非真实语义向量，语义理解能力弱于
  Qwen3-Embedding，例如无法识别"广告投放"≈"推广策略"这类同义改写。
- `/api/chat/stream` 当前是"生成完整答案后分段推送"，并非真正的逐 token 流式生成；
  切换到 `vllm` 后端并接入 OpenAI 流式 API 可获得真流式效果（接口契约已预留）。
- `two_stage_prune`（HTML 两阶段剪枝）尚未接入 `pipeline.py`，见「与 process/ 的对接」。
- `observability/monitoring.py` 是**进程内**滚动指标（有界内存窗口），多 worker/多副本部署下
  每个进程各自独立统计，不做跨进程汇总；生产环境如需全局视图，建议将
  `monitoring.snapshot()` 的输出定期上报到真实的可观测性平台。
- `evaluation/generation_eval.py` 的 groundedness 评分是词汇重叠的启发式代理指标，不等价于
  事实核查，仅用于捕捉"完全脱离上下文编造"的明显异常场景。
- Agent 侧（`agent/`）注册 `rag/integration/` 提供的工具函数到其 `ToolRegistry` 的接线代码
  尚待编写，是 M4 里程碑的核心待办（见 `TODO.md`）。
