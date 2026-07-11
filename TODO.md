# TODO — CustomerServiceAgent 项目规划

> 最后更新：2026-07-02

---

## 🗺️ 里程碑路线图

| 里程碑 | 目标 | 状态 |
|:------:|------|:----:|
| **M1 数据地基** | HTML → 干净知识块（`process/` 全链路 + 单测） | ✅ 完成 |
| **M2 模型底座** | TEI 部署 Embedding/Reranker + SFT/DPO 训练代码 | 🔄 80%（待实机 GPU 验证） |
| **M3 检索打通** | `rag/` 索引 + 双模检索 + 融合 + 精排 + 生成 + FastAPI 服务 | ✅ 完成 |
| **M4 智能问答** | `agent/`（hello_agents 框架引入）+ RAG 工具注册 | 🔄 框架就位，业务集成待开发 |
| **M5 可度量** | 端到端评测体系 + A/B | ✅ 基础完成（Bitext 数据集验证召回率 93.5%） |
| **M6 生产化** | CI/CD + 容器化 + 可观测性 | 🔲 待规划 |

---

## ✅ 已完成工作概览

### process/ — HTML 数据处理与分块

完成了对标 HtmlRAG 论文的全链路 HTML 清洗与结构化分块：无损清洗（噪声标签移除、合并单元格展开、冗余包装展开、模板残留清除）→ Block Tree 语义分块（BFS 遍历 DOM、粒度双阈值控制、表格按行切分、混合内容分离）→ 两阶段块树剪枝算法实现（Embedding 粗剪 + Reranker 精剪，待接入 rag 检索链路）。中文电商领域适配了 jieba 词典与按字符计数。293 项单元测试全绿。

### model/ — 模型推理与训练

选定 TEI 框架部署 Qwen3-Embedding-4B（端口 8080）+ Qwen3-Reranker-4B（端口 8081），提供 Docker Compose 一键部署。实现了 Reranker 两阶段微调：SFT 监督微调（多级标注 0/1/2）+ DPO 偏好优化（chosen vs rejected）。训练数据构造脚本已接入 `rag.retrieval` 真实召回（Pointwise/Pairwise/DPO 三格式 + 困难负样本）。代码层面 100% 完成，待实机 GPU 部署验证。

### rag/ — RAG 检索增强生成服务

自研完整 RAG 框架：索引构建（文档解析/分块/向量化/双库写入）→ 双模检索（Milvus 向量 + ES 关键词）→ 融合去重（RRF/加权 + TF-IDF cosine）→ Reranker 精排 → 生成融合（vLLM/抽取式兜底 + Citation 引用）。每个组件均有"真实后端 + 本地降级"双实现，零外部依赖即可跑通全链路。提供 FastAPI 后端（文档管理/检索/问答/SSE 流式）+ 轻量前端 + Agent 友好 API 文档（OpenAPI + 纯文本）。已完成全组件代码审计：修复单例缓存缺失导致的并发丢更新、SSE 伪流式、CORS 反模式等 8 项 Bug + 5 项性能优化。439 项测试全绿。

### agent/ — 智能体框架引入

引入 hello_agents 生产级多智能体框架（16 项核心能力：ReAct/Reflection/Plan-Solve 多范式、ToolRegistry 工具调用、上下文工程、会话持久化、Skills 知识外化、TraceLogger 可观测性等），19 个自带测试套件已迁移至 `tests/test_agent/`。**与 `rag/` 服务的工具化集成代码尚未编写**，是 M4 的核心待办。

### tests/ — 测试体系

按模块重构为 `test_process/`（10 文件）+ `test_rag/`（13 文件）+ `test_agent/`（19 文件）三个子包，统一 `conftest.py` 注入 `sys.path`，无需手动设置 `PYTHONPATH`。`process/` + `rag/` 共 439 项测试全绿。

### 工程基础设施

- `config/` 双层配置（`config.json` + `.env`），环境变量优先级明确，`.env.example` 逐项注释
- `scripts/` 全套启动脚本：`run_RAGserver.sh`/`run_AGTserver.sh`/`start_tei.sh`（含环境自检+健康检查）、`build_index.sh`/`build_index.py`、`process_HTMLdata.sh`、`build_JIEBAdict.sh`、`export_openapi.py`
- `dataset/` 评测数据集：Bitext Customer Support（26,872 条 QA / 11 类别 27 意图），预处理脚本 + 知识库块 + 31 条评测用例（召回率 93.5%，平均延迟 574ms）
- 文档：主 `README.md`（含 Mermaid 架构图 + workflow 流程图 + Demo 展示）+ 各模块 README + `images/IMAGE_PROMPT.md` AI 生图提示词

---

## 📊 评测结果（Bitext 数据集，本地降级后端）

| 指标 | 值 |
|------|-----|
| 知识库导入 | 1,341 块 / 1.62s |
| 总召回率 | **93.5%**（29/31 命中） |
| 基础场景 | 96%（22/23） |
| 边界场景（极短 query） | 67%（2/3） |
| 多轮场景 | 100%（5/5） |
| 平均端到端延迟 | 574ms |
| 生成后端 | local（抽取式兜底） |

---

## 📦 候选评测数据集调研

> 当前使用 Bitext Customer Support（26,872 条 QA），适合 RAG 知识库构建与检索评测。
> 以下数据集规模更大、更适合 SFT/DPO 微调训练，可作为后续模型微调与端到端评测的补充。

### 1. Salesforce DialogStudio（推荐，最大最全）

| 属性 | 值 |
|------|-----|
| **HuggingFace** | [Salesforce/dialogstudio](https://huggingface.co/datasets/Salesforce/dialogstudio) |
| **总规模** | 52.6 GB，包含 80+ 个对话数据集的统一集合 |
| **许可证** | Apache 2.0（部分子数据集保留原始许可） |
| **语言** | 多语言（含中文子集如 BiTOD、DuRecDial） |
| **格式** | 统一 JSON Schema，含 `prompt`/`log`/`external knowledge`/`intent` 字段 |
| **适用场景** | SFT 指令微调 + 任务型对话 + 知识接地对话 |

**与本项目高度相关的子数据集**：

| 子集 | 规模 | 场景 | 与项目的关系 |
|------|------|------|------------|
| **Taskmaster 1/2/3** | ~50K 对话 | 电商订单、餐厅预订、机票 | 任务型对话，订单场景直接对口 |
| **MultiWOZ 2.2** | 10K 对话 | 餐厅/酒店/火车/景点预订 | 多领域任务型对话标杆 |
| **ABCD** | 10K 对话 | 客服对话（退货/换货/取消订单） | **最接近本项目客服场景** |
| **AirDialogue** | 402K 对话 | 航班预订客服 | 大规模客服对话 |
| **BANKING77** | 13K 对话 | 银行客服意图识别 | 意图分类训练 |
| **Taskmaster-3 (Reddit)** | 113K 对话 | Reddit 讨论帖 | 开放域对话 |

**优势**：DialogStudio 是目前最全面的统一对话数据集合集，包含 ABCD（客服对话）直接对口本项目。Apache 2.0 许可证，可直接用于训练。数据已统一格式化，含 prompt 字段可直接用于 SFT。

**用于 DPO**：DialogStudio 本身不含 chosen/rejected 偏好对，但可通过 LLM-as-Judge 对其多条回复自动构造偏好数据（项目已有 `model/utils/build_dataset.py` 的 DPO 构造能力）。

---

### 2. V1rtucious Ecom-Chatbot-Finetuning-Dataset

| 属性 | 值 |
|------|-----|
| **HuggingFace** | [V1rtucious/Ecom-Chatbot-Finetuning-Dataset](https://huggingface.co/datasets/V1rtucious/Ecom-Chatbot-Finetuning-Dataset) |
| **总规模** | 50,098 行（101 MB） |
| **许可证** | 未明确标注（需确认） |
| **语言** | English |
| **格式** | Parquet，含 `system`/`prompt`/`response`/`context`/`tools`/`difficulty`/`quality_score` 等 20 个字段 |
| **适用场景** | SFT 微调 + RAG 评测（含 `context.retrieved_docs` 检索上下文） |

**与 Bitext 的对比**：

| 维度 | Bitext | V1rtucious |
|------|--------|-----------|
| 规模 | 26,872 行 | **50,098 行** |
| 数据来源 | 合成数据（单一来源） | **5 个来源整合**（Amazon 评论 + ASOS + Bitext + 合成） |
| 字段丰富度 | 5 列 | **20 列**（含 `difficulty`/`quality_score`/`tools`/`history`） |
| 电商领域覆盖 | 11 类别 27 意图 | **16 个电商领域** + 11 个子意图 |
| 检索上下文 | 无 | **有**（`context.retrieved_docs`，天然适合 RAG 评测） |
| 对话历史 | 无 | **有**（`history` 字段，支持多轮） |
| 工具调用 | 无 | **有**（`response_type: tool_call`） |

**优势**：规模是 Bitext 的 2 倍，字段丰富度远超（含难度分级、质量评分、对话历史、工具调用、检索上下文），天然适合 RAG + Agent 联合评测。`context.retrieved_docs` 字段可直接用于评估检索召回质量。

---

### 3. UltraFeedback（DPO 偏好对首选）

| 属性 | 值 |
|------|-----|
| **HuggingFace** | [openbmb/UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) |
| **总规模** | 64K 条（prompt + 4 条 AI 回复 + 细粒度评分） |
| **许可证** | MIT |
| **格式** | 每条含 prompt + 4 条不同模型的回复 + 多维评分（instruction_following/truthfulness/honesty/helpfulness） |
| **适用场景** | **DPO 偏好对构造**（取最高分 vs 最低分作为 chosen/rejected）+ Reranker 训练 |

**与本项目的关系**：本项目已有 `model/trainer/reranker_dpo.py` 的 DPO 训练能力，UltraFeedback 可直接提供 chosen/rejected 偏好对用于训练。64K 条规模适中，多维细粒度评分（非简单二分）可用于构造多级标注（与本项目 0/1/2 三级标注思路一致）。MIT 许可证无使用限制。

---

### 4. LMSYS Chatbot Arena Conversations

| 属性 | 值 |
|------|-----|
| **HuggingFace** | [lmsys/chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) |
| **总规模** | 33K 条真实用户对话（含 pairwise 人类偏好） |
| **许可证** | CC-BY-NC（仅研究用途） |
| **格式** | 真实用户 prompt + 两个模型的回复 + 人类裁判选择 |
| **适用场景** | DPO 偏好训练（真实人类偏好标注，非 AI 合成） |

**优势**：真实用户对话 + 真实人类偏好标注（非 AI 合成），数据质量最高。但许可证为 CC-BY-NC（非商业），且场景为通用对话而非电商客服。

---

### 数据集选型建议

| 用途 | 推荐数据集 | 理由 |
|------|----------|------|
| **RAG 知识库 + 检索评测** | Bitext（当前） | 场景对口、格式简洁、开箱即用 |
| **SFT 微调训练** | DialogStudio → ABCD 子集 | 10K 客服对话，退货/换货/取消订单场景直接对口 |
| **DPO 偏好对训练** | UltraFeedback | 64K + MIT 许可 + 多维评分可直接构造 chosen/rejected |
| **端到端 RAG+Agent 联合评测** | V1rtucious | 50K 条 + 含 `retrieved_docs` 检索上下文 + 多轮历史 + 工具调用 |
| **真实偏好质量验证** | LMSYS Chatbot Arena | 真实人类偏好，用于最终质量校验 |

---

## 🔄 后续改进与完善方向

### P0 — 最高优先级

#### 1. RAG 与 Agent 集成（M4 阻塞项）

> 核心问题：如何将 `rag.pipeline` 检索能力注册给 `agent/`（hello_agents），使 Agent 能够按需调用？
> `agent/` 框架已就位（`ToolRegistry`/`Tool` 接口齐备），**集成代码尚未编写**，是当前唯一阻塞 M4 达成的工作。

**方案调研**（三种候选路径）：

- [ ] **A: Tool-based** — 将 `rag.pipeline.retrieve`/`answer` 封装为 `hello_agents.tools.base.Tool` 子类，通过 `ReActAgent` 的 function calling 按需调用（适合复杂问题，模型自主判断是否检索）
- [ ] **B: Middleware** — RAG 作为感知层中间件，每轮对话自动注入检索结果（适合高频确定性问答）
- [ ] **C: Hybrid** — 常见问题自动检索 + 复杂问题 Agent 主动调用（生产倾向，推荐）

**技术选型**：
- [ ] 基于 hello_agents 已有的 `ToolRegistry.register_tool()` 接口封装 RAG Tool
- [ ] 评估是否需要 `SkillLoader`（Skills 知识外化系统）作为替代/补充路径
- [ ] 实现 `agent/hello_agents/tools/builtin/rag_tool.py`（或项目侧自定义 Tool）→ 注册 → 用 `ReActAgent` 端到端测试

**触发策略优化**：
- [ ] 避免过度检索（拖慢+噪声）与漏检索（答非所问）
- [ ] 评估缓存层：query / embedding 结果缓存，降低重复计算

#### 2. 业务工具层（🥇 P0，最高价值）

> 客服智能体与普通 RAG 问答的本质区别：用户问"订单到哪了"，纯 RAG 只能答规则，智能体应真的去查物流。

- [ ] 业务 API 工具集：订单查询、物流跟踪、退款/退货、优惠券发放、账户状态
- [ ] 工具编排（ReAct / Plan-and-Solve，hello_agents 均已提供对应 Agent 实现）：自主决定"查订单→判断退款条件→发起退款"
- [ ] 参数槽填充（Slot Filling）：缺订单号时主动追问而非失败
- [ ] 落点：基于 hello_agents `ToolRegistry` 扩展自定义 Tool，RAG 检索工具只是其一，业务工具才是"客服"主体

#### 3. 安全与合规护栏（🥇 P0，🎯 数据优势）

> 已有 AI 管控/质检的数据与经验（违规判定标准），可复用做客服护栏。

- [ ] **输入护栏**：Prompt 注入检测、敏感/违规提问拦截（复用管控标准思路）
- [ ] **输出护栏**：回答合规校验、PII（手机号/身份证）自动脱敏
- [ ] **拒答边界**：超出知识库范围不硬编，明确说明"不知道"
- [ ] **越权防护**：退款等敏感操作二次确认/权限校验

#### 4. 对话理解层（🥇 P0）

> 复用 `llm_api.py` 已有的 Query 重写（多轮指代补全）与 `infer_chunk_category`（分类）思路向上扩展。

- [ ] **意图识别**：咨询/投诉/查询/闲聊/转人工 分流（复用分类思路，加 query 级意图分类）
- [ ] **情绪识别**：检测愤怒/焦虑 → 触发安抚话术或转人工（新增，客服场景刚需）
- [ ] **澄清反问**：query 模糊时主动反问，而非硬答（补充到 Query 重写下游）
- [ ] **多意图拆解**："查订单顺便退款"拆成两个子任务（Agent 编排配合）

---

### P1 — 高优先级

#### 5. 两阶段剪枝接入 rag 检索链路

> `html_pruner.py` 已实现论文 HtmlRAG 的核心贡献（Embedding 粗剪 + Reranker 精剪），29 项测试覆盖。

- [ ] 接入 `rag/pipeline.retrieve`：检索得到候选块后，对其 `html_content` 做两阶段剪枝再送 LLM
- [ ] 真实 embedding/reranker 服务下的剪枝质量评估（保留率、Faithfulness）
- [ ] 分块参数自动调优：`max_node_words` / `min_node_words` 针对电商语料网格搜索

#### 6. 真实后端联调

- [ ] 部署真实 Milvus 2.x + Elasticsearch 8.x + TEI + vLLM 服务
- [ ] 跑 `scripts/build_index.sh` + `pytest` 全链路验证
- [ ] 对比本地降级 vs 真实后端的性能（延迟 QPS）与精度（Recall@K / NDCG）差异

#### 7. 反馈与持续学习闭环（🥈 P1，🎯 独特护城河）

> 已有 DPO/SFT 训练 pipeline + QC 质检经验，天然适合做数据飞轮。

```
线上对话 → 用户点赞/点踩 + 自动质检（LLM-as-Judge）
    → 收集 Bad Case（答错/转人工/低分）
    → 回流构造训练样本（chosen/rejected）
    → 重训 Reranker（reranker_dpo.py 已就绪）
    → A/B 验证收益（四层评测）→ 上线
```

- [ ] 用户满意度采集（点赞/点踩 / CSAT）
- [ ] 会话自动质检（迁移 AI 管控 QC 经验 + LLM-as-Judge）
- [ ] Bad Case 挖掘 → DPO 偏好样本回流 → 重训 → A/B 验证

#### 8. 回答生成深化

- [ ] **忠实度/防幻觉**：强制 grounding 到检索结果（当前 Prompt 已约束，待评测量化）
- [ ] **客服话术风格**：品牌调性、礼貌用语、结构化回答（步骤/链接）
- [ ] **时效性排序**：利用 chunk 的 `time` 字段对规则类内容加权
- [ ] **真流式输出**：当前 SSE 为"生成完后分段推送"，待接入 vLLM 逐 token 流式 API
- [ ] **无答案兜底**：检索不到时澄清反问或明确"不知道"（当前已有兜底文案，待升级为主动澄清）

#### 9. 端到端评测体系完善

**四层评测体系**：

| 层级 | 评测对象 | 指标 | 测量方法 |
|------|---------|------|---------|
| Layer 1 | 数据处理质量 | 噪声残留率 / 内容保留率 / 压缩比 / 分块合理性 | 标注噪声，统计清洗后残留数 |
| Layer 2 | 检索召回 | Recall@K / MRR / NDCG@K / 延迟 P95 | query→gold_doc_id 评测集 |
| Layer 3 | 精排质量 | Top-1 命中率 / 排序改善幅度 | 评测集统计 |
| Layer 4 | 端到端回答 | Accuracy / Faithfulness / Relevance / 完整性 | 人工 / LLM-as-Judge |

- [ ] 采样 200-500 文档块，人工构造 query（事实/比较/多跳/拒答型）
- [ ] 标注 gold_doc_id + relevance 等级（0/1/2），JSONL 版本化管理
- [ ] 实现 `scripts/run_eval.py` + Markdown 对比报告生成器（ΔRecall/ΔNDCG/Δ延迟/ΔAccuracy）
- [ ] 评测结果版本化存储 `eval_results/YYYYMMDD_HHMMSS/`
- [ ] **每项修改的收益评估**：清洗优化→噪声↓Recall↑、换 Embedding→MRR↑、Reranker 微调→Top-1↑、剪枝接入→token↓Faithfulness↑、Agent 集成→Accuracy↑

#### 10. model/ 深化

- [ ] **Embedding 微调**（当前只微调 Reranker）：对比学习 + 领域内 query-doc 对
- [ ] TEI **实机部署验证**：吞吐、延迟、显存实测（补齐 M2 最后 20%）
- [ ] 模型量化/加速：INT8/FP8，降低服务成本
- [ ] SFT → DPO 全流程跑通并对比收益（用上述评测体系量化）

#### 11. rag/ 深化

- [ ] **上下文压缩**：结合剪枝，控制送入 LLM 的 token 量
- [ ] **查询理解**：意图识别、多路召回、query 改写多样化
- [ ] **缓存层**：query / embedding 结果缓存，降低重复计算

---

### P2 — 中长期

#### 12. 记忆与个性化

- [ ] **短期记忆**（对话历史）：hello_agents `context/history.py` 已提供 `HistoryManager`，待接入业务对话流
- [ ] **长期记忆**（用户画像、历史工单、VIP 等级 → 个性化话术与优先级）
- [ ] **会话摘要**（长对话自动压缩，防上下文爆炸）：hello_agents `context/` 已提供 Token 计数与自动截断/压缩能力

#### 13. 多渠道接入与多模态

- [ ] 多渠道接入（Web / 小程序 / 企微 / 飞书）
- [ ] 多模态：用户发商品截图 → 图片理解（VLM，hello_agents `skills/` 已提供 VLM 技能资源可复用）
- [ ] 在线指标：自助解决率、转人工率、首响时间、CSAT

#### 14. 工程化生产化（M6）

- [ ] **API 服务层**：`rag/api` 已提供 FastAPI 服务化，`agent/` 侧待补充服务化入口
  - [ ] 流式输出（SSE，已支持基础版）、限流、鉴权、多租户
  - [ ] 灰度发布与回滚
- [ ] **依赖分层**：拆分 `requirements/`（process 仅需 bs4/jieba/sklearn，不必装 torch/pymilvus；`agent/` 已独立维护 `agent/requirements.txt`）
  - [ ] `requirements-process.txt` / `requirements-model.txt` / `requirements-rag.txt`
- [ ] **CI/CD**：GitHub Actions / 蓝盾流水线自动跑 `pytest tests/`（含冒烟导入测试）
- [ ] **容器化**：整体 `docker-compose`（process + TEI + Milvus + ES + RAG API + Agent 服务一键起）
- [ ] **可观测性**：结构化日志、Prometheus 指标（延迟/QPS/错误率）、链路追踪（`agent/` 已提供 `TraceLogger` 基础能力）
- [ ] **配置校验**：启动时校验 config.json / .env 完整性，缺失即 fail-fast（脚本已实现基础版环境自检）

#### 15. 数据接入与知识库运营

> 当前假设 HTML 已存在于 `process/data/`，但"数据从哪来、如何更新"尚未规划。

- [ ] **数据接入**：爬虫 / 导出接口对接（电商平台帮助中心、规则文档）
- [ ] **增量抓取与去重**：定时抓取 + 内容指纹去重 + 变更检测
- [ ] **知识库更新流程**：抓取 → 清洗分块 → 索引更新（对接 `rag/indexing` 增量能力，`scripts/build_index.py` 已支持幂等增量导入）
- [ ] **数据版本管理**：知识块快照与版本号，支持回滚与对比
- [ ] **数据质量校验**：空块率、超长块率、乱码检测、时间戳缺失告警

---

### 优先级速览

| 优先级 | 方向 | 理由 |
|:------:|------|------|
| 🥇 P0 | RAG-Agent 集成（M4 阻塞项） | 打通检索能力与智能体决策，是当前最紧迫任务 |
| 🥇 P0 | 业务工具 + 意图/转人工 | 决定"是不是真客服"，价值最高 |
| 🥇 P0 | 安全护栏 | 上线合规底线，有数据优势 |
| 🥈 P1 | 质检 + Bad Case 回流闭环 | 独特护城河（DPO + QC 经验） |
| 🥈 P1 | 忠实度 + 剪枝接入 | 直接提升回答质量 |
| 🥈 P1 | 评测体系完善 + 真实后端联调 | 量化每项修改收益 |
| 🥉 P2 | 长期记忆 / 多渠道多模态 | 锦上添花，后期扩展 |
| 🥉 P2 | 工程化生产化 + 数据运营 | 生产上线必备 |

---

## ⚠️ 已知风险与技术债

- `model/` 训练脚本仅在语法/结构层面验证，尚未在真实 GPU + 数据上跑通
- `process/utils/config.py` 与 `config/config_loader.py` 存在重复配置加载逻辑，建议抽公共层
- `rag/` 本地降级后端不适合大规模生产知识库，生产必须切换 Milvus/ES
- `rag/api/chat/stream` 生成阶段非真正逐 token 流式（待接入 vLLM 流式 API）
- `agent/`（hello_agents）框架自带 3 项缺陷（`PlanAndSolveAgent` 导出别名缺失、`ReadOnlyFilter` 断言不一致、测试依赖真实 API Key），均为第三方框架问题
- `agent/` 与 `rag/` 尚未产生任何调用关系，M4 仅完成框架引入
