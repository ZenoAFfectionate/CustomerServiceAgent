# RAG API 接口文档（Agent 友好版）

> **本文档专为自动化 Agent 调用优化**：字段定义无歧义、请求示例完整可直接复制执行、响应结构与错误码固定。
> 人类可读的交互式文档见 Swagger UI：`http://<host>:<port>/docs`；OpenAPI JSON 见 `rag/docs/openapi.json` 或运行时 `http://<host>:<port>/openapi.json`。

## 基本信息

| 项 | 值 |
|---|---|
| **Base URL** | `http://<host>:<port>/api`（默认 `http://localhost:8090/api`） |
| **协议** | HTTP/1.1，JSON 请求体（除文件上传为 `multipart/form-data`） |
| **鉴权** | 当前版本无鉴权（内网/演示用途），生产部署建议在反向代理层加鉴权 |
| **字符编码** | UTF-8 |
| **通用响应头** | `Content-Type: application/json`（除 `/chat/stream` 为 `text/event-stream`） |

---

## 接口列表总览

| # | 方法 | 路径 | 说明 |
|---|------|------|------|
| 1 | `GET`  | `/api/health` | 健康检查与服务统计 |
| 2 | `POST` | `/api/documents/upload` | 上传文档并建立索引 |
| 3 | `POST` | `/api/documents/ingest_blocks` | 批量导入已分块文档（如 process/ 输出） |
| 4 | `GET`  | `/api/documents` | 知识库文档列表 |
| 5 | `GET`  | `/api/documents/{doc_id}` | 文档详情 |
| 6 | `DELETE` | `/api/documents/{doc_id}` | 删除文档 |
| 7 | `POST` | `/api/retrieve` | 检索问答上下文（不生成答案） |
| 8 | `POST` | `/api/chat` | 检索增强问答（RAG QA） |
| 9 | `POST` | `/api/chat/stream` | 检索增强问答（SSE 流式输出） |

---

## 1. `GET /api/health`

**说明**：检查服务是否正常运行，并返回当前各组件（向量库/关键词库/嵌入/精排/生成）使用的后端类型与统计数据。

**请求参数**：无

**响应 200**：
```json
{
  "status": "ok",
  "stats": {
    "num_documents": 3,
    "num_vector_chunks": 42,
    "num_keyword_chunks": 42,
    "vector_backend": "local",
    "keyword_backend": "local",
    "embed_backend": "local",
    "rerank_backend": "local",
    "generation_backend": "local"
  }
}
```

**curl 示例**：
```bash
curl -X GET http://localhost:8090/api/health
```

---

## 2. `POST /api/documents/upload`

**说明**：上传单个文档文件，服务端自动解析、分块、向量化并写入向量库与关键词库。

**Content-Type**：`multipart/form-data`

**请求参数**：

| 参数 | 位置 | 类型 | 必填 | 说明 |
|------|------|------|:----:|------|
| `file` | form-data | file | ✅ | 待上传文件，支持扩展名 `.txt` `.md` `.html` `.htm` `.json` `.pdf`；`.json` 需为 `process/` 输出格式的文档块数组 |

**响应 200**（`DocumentUploadResponse`）：
```json
{
  "doc_id": "a1b2c3d4e5f60718",
  "filename": "ad_limit_faq.txt",
  "source": "upload",
  "num_chunks": 8,
  "chunk_ids": [0, 1, 2, 3, 4, 5, 6, 7],
  "created_at": "2026-07-02 10:12:33"
}
```

**curl 示例**：
```bash
curl -X POST http://localhost:8090/api/documents/upload \
  -F "file=@/path/to/ad_limit_faq.txt"
```

**错误**：
- `422 RAG_VALIDATION_ERROR`：文件类型不支持 / 文件内容为空 / 解析后无有效文本块
- `413 RAG_PAYLOAD_TOO_LARGE`：文件超过最大上传体积（默认 20MB，见 `RAG_UPLOAD_MAX_SIZE_MB`）

---

## 3. `POST /api/documents/ingest_blocks`

**说明**：直接导入结构化文档块数组（对接 `process/` 数据处理流水线输出的 JSON），跳过文档解析与分块步骤。

**Content-Type**：`application/json`

**请求体字段**（`IngestBlocksRequest`）：

| 字段 | 类型 | 必填 | 说明 | 示例 |
|------|------|:----:|------|------|
| `blocks` | array[object] | ✅ | 文档块数组，每个对象可含 `text`/`title`/`page_name`/`page_url`/`html_content`/`block_path`/`summary`/`question`/`time` 等字段（对齐 `process/` 输出，未知字段会被忽略，`text` 为必需内容字段） | 见下 |
| `filename` | string | ❌ | 来源标识（如原始文件名/页面名），默认 `"manual"` | `"htmlrag_page_001.json"` |

**请求示例**：
```json
{
  "filename": "htmlrag_page_001.json",
  "blocks": [
    {
      "chunk_idx": 0,
      "page_name": "广告限流规则",
      "title": "广告限流规则说明",
      "page_url": "https://help.example.com/ad-limit",
      "text": "当账户存在异常投放行为时，系统将自动限流，限流期间广告曝光量将下降 50%-80%。",
      "html_content": "<div><p>当账户存在异常投放行为时...</p></div>",
      "block_path": "html>body>div0>p",
      "summary": "介绍广告限流的触发条件与影响",
      "question": "什么情况下广告会被限流？",
      "time": "2026-06-01"
    }
  ]
}
```

**响应 200**：与 `/api/documents/upload` 相同结构（`source` 字段为 `"ingest_blocks"`）。

**curl 示例**：
```bash
curl -X POST http://localhost:8090/api/documents/ingest_blocks \
  -H "Content-Type: application/json" \
  -d '{"filename":"htmlrag_page_001.json","blocks":[{"text":"当账户存在异常投放行为时，系统将自动限流。","title":"广告限流规则"}]}'
```

**错误**：`422 RAG_VALIDATION_ERROR`（`blocks` 为空）

---

## 4. `GET /api/documents`

**说明**：返回知识库中全部已索引文档及其分块统计。

**请求参数**：无

**响应 200**（`DocumentListResponse`）：
```json
{
  "total": 1,
  "documents": [
    {
      "doc_id": "a1b2c3d4e5f60718",
      "filename": "ad_limit_faq.txt",
      "source": "upload",
      "num_chunks": 8,
      "chunk_ids": [0, 1, 2, 3, 4, 5, 6, 7],
      "created_at": "2026-07-02 10:12:33"
    }
  ]
}
```

**curl 示例**：
```bash
curl -X GET http://localhost:8090/api/documents
```

---

## 5. `GET /api/documents/{doc_id}`

**说明**：查询单个文档的详细信息。

**路径参数**：

| 参数 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `doc_id` | string | ✅ | 文档 ID（`/upload` 或 `/ingest_blocks` 响应中的 `doc_id`） |

**响应 200**：结构同上单个 `documents` 元素。

**curl 示例**：
```bash
curl -X GET http://localhost:8090/api/documents/a1b2c3d4e5f60718
```

**错误**：`404 RAG_NOT_FOUND`（文档不存在）

---

## 6. `DELETE /api/documents/{doc_id}`

**说明**：从向量库、关键词库与知识库登记表中同时删除该文档的全部分块。

**路径参数**：同上 `doc_id`。

**响应 200**（`DeleteResponse`）：
```json
{ "doc_id": "a1b2c3d4e5f60718", "deleted": true }
```

**curl 示例**：
```bash
curl -X DELETE http://localhost:8090/api/documents/a1b2c3d4e5f60718
```

**错误**：`404 RAG_NOT_FOUND`（文档不存在）

---

## 7. `POST /api/retrieve`

**说明**：对查询执行「向量检索 + 关键词检索 → 融合去重 → Reranker 精排」，仅返回检索到的文档块，**不生成答案**。适合 Agent 自行决定是否需要检索、以及如何使用检索结果（Tool-based 集成场景）。

**Content-Type**：`application/json`

**请求体字段**（`RetrieveRequest`）：

| 字段 | 类型 | 必填 | 说明 | 示例 |
|------|------|:----:|------|------|
| `query` | string | ✅ | 查询文本，非空 | `"广告限流后怎么办"` |
| `dialogue` | array[object] | ❌ | 多轮历史对话，用于 query 重写（指代补全）。每项：`{"speaker": "user"\|"bot", "text": "..."}` | `[{"speaker":"user","text":"我的广告为什么下不去了"},{"speaker":"bot","text":"可能是被限流了"}]` |
| `top_k` | integer | ❌ | 返回条数，范围 1-50，默认服务端配置（5） | `5` |

**请求示例**：
```json
{ "query": "广告限流后怎么办", "top_k": 5 }
```

**响应 200**（`RetrieveResponse`）：
```json
{
  "query": "广告限流后怎么办",
  "results": [
    {
      "global_chunk_idx": 3,
      "doc_id": "a1b2c3d4e5f60718",
      "chunk_idx": 3,
      "page_name": "广告限流规则",
      "title": "限流解除方式",
      "page_url": "https://help.example.com/ad-limit",
      "text": "限流通常持续 24-72 小时，可通过优化创意质量、降低出价频次申请人工复核解除。",
      "html_content": "",
      "block_path": "html>body>div0>p",
      "summary": "限流解除的常见方法",
      "question": "",
      "time": "2026-06-01",
      "score": 0.8231,
      "source_retriever": "reranked"
    }
  ],
  "latency_ms": 128.4
}
```

**字段说明**：`score` 越大越相关；`source_retriever` 取值 `milvus`/`es`/`fused`/`reranked`，标识该结果最后经过的处理阶段。

**curl 示例**：
```bash
curl -X POST http://localhost:8090/api/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query":"广告限流后怎么办","top_k":5}'
```

**错误**：`422 RAG_VALIDATION_ERROR`（`query` 为空）

---

## 8. `POST /api/chat`

**说明**：端到端检索增强问答（RAG QA）：检索 → 融合去重 → 精排 → 生成回答。无生成模型服务（`RAG_GENERATION_BACKEND=local`）时自动降级为**抽取式摘录回答**（直接摘录最相关片段，不调用 LLM），不会报错或返回空。

**Content-Type**：`application/json`

**请求体字段**（`ChatRequest`）：

| 字段 | 类型 | 必填 | 说明 | 示例 |
|------|------|:----:|------|------|
| `query` | string | ✅ | 用户问题，非空 | `"千川广告投放规则是什么"` |
| `dialogue` | array[object] | ❌ | 多轮历史对话，格式同 `/api/retrieve` | 见上 |
| `top_k` | integer | ❌ | 检索上下文条数，范围 1-20 | `5` |

**请求示例**：
```json
{ "query": "千川广告投放规则是什么" }
```

**响应 200**（`ChatResponse`）：
```json
{
  "query": "千川广告投放规则是什么",
  "rewritten_query": null,
  "answer": "根据知识库检索结果，为您找到以下 3 条相关信息：\n[1] 千川投放规则：...\n[2] ...\n（以上内容直接摘录自知识库，未经 LLM 润色；如需更完整的智能问答，请部署生成模型服务）",
  "citations": [
    { "index": 1, "page_url": "https://help.example.com/qianchuan", "block_path": "html>body>div0>p", "title": "千川投放规则", "score": 0.81 }
  ],
  "backend_used": "local",
  "contexts": [ /* 结构同 /api/retrieve 的 results 数组元素 */ ],
  "latency_ms": 156.2
}
```

**字段说明**：
- `rewritten_query`：仅当传入 `dialogue` 且重写结果与原始 `query` 不同时返回重写后的问题，否则为 `null`。
- `backend_used`：`"vllm"`（真实 LLM 生成）/ `"local"`（抽取式兜底）/ `"no_context"`（未检索到任何相关内容）。
- `citations[].index` 与 `answer` 正文中的 `[数字]` 标注对应。

**curl 示例**：
```bash
curl -X POST http://localhost:8090/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"千川广告投放规则是什么"}'
```

**错误**：`422 RAG_VALIDATION_ERROR`（`query` 为空）

---

## 9. `POST /api/chat/stream`

**说明**：与 `/api/chat` 相同的问答流程，但以 **Server-Sent Events (SSE)** 逐段返回答案文本，适合前端流式展示（注意：当前实现是"生成完整答案后分段推送"，并非逐 token 真流式；接口契约与未来切换到真流式 LLM 完全兼容）。

**Content-Type**：`application/json`（请求）/ `text/event-stream`（响应）

**请求体**：与 `/api/chat` 相同。

**SSE 事件序列**：

| event | data 结构 | 说明 |
|-------|-----------|------|
| `citations` | `[{"index":1,"page_url":...,"block_path":...,"title":...,"score":...}, ...]` | 最先推送，引用来源列表 |
| `answer` | `{"delta": "答案文本片段"}` | 多次推送，拼接 `delta` 即完整答案 |
| `done` | `{"backend_used": "local"}` | 结束标记 |

**curl 示例**：
```bash
curl -N -X POST http://localhost:8090/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"千川广告投放规则是什么"}'
```

**JS EventSource 消费示例**（需自行用 fetch + ReadableStream 解析，因 POST body 不支持原生 `EventSource`）：见 `rag/web/index.html` 的 `/api/chat` 调用方式（当前前端默认使用非流式 `/api/chat`）。

---

## 错误码汇总

| HTTP 状态码 | `error_code` | 触发场景 |
|:-----------:|---------------|----------|
| 404 | `RAG_NOT_FOUND` | 查询/删除不存在的 `doc_id` |
| 413 | `RAG_PAYLOAD_TOO_LARGE` | 上传文件超过大小限制 |
| 415 | `RAG_UNSUPPORTED_MEDIA_TYPE` | 不支持的文件格式（预留） |
| 422 | `RAG_VALIDATION_ERROR` | 请求参数校验失败（如 `query`/`blocks` 为空、文件类型不支持） |
| 422 | （FastAPI 原生） | Pydantic 请求体字段类型/约束校验失败（如 `top_k` 超出 1-50） |
| 500 | `RAG_INTERNAL_ERROR` | 服务器内部未捕获异常 |

**统一错误响应结构**：
```json
{ "error_code": "RAG_NOT_FOUND", "message": "文档不存在: xxxx", "detail": null }
```

---

## Agent 调用建议

1. **仅需检索不需要生成** → 调用 `/api/retrieve`，自行判断是否需要检索（Tool-based 集成，见 `TODO.md` 难点⑥方案 A）。
2. **需要完整问答** → 调用 `/api/chat`，`backend_used` 字段可用于判断答案是 LLM 生成还是抽取式兜底，据此决定是否需要二次处理。
3. **批量导入知识库** → 优先使用 `/api/documents/ingest_blocks`（直接对接 `process/` 输出，无需重新分块，保留 `html_content`/`block_path` 等结构化字段）；零散文档使用 `/api/documents/upload`。
4. **多轮对话** → 每轮调用时把历史对话通过 `dialogue` 字段传入（建议保留最近 3-6 轮），服务端会自动做指代补全重写。
5. **健康探活** → 部署后先调用 `/api/health` 确认 `stats.vector_backend`/`keyword_backend` 等是否为期望的后端（`local` 为无外部依赖的降级演示模式，生产环境应配置为 `milvus`/`es`/`tei`/`vllm`）。
