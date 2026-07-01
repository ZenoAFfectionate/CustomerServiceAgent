# TODO — CustomerServiceAgent 项目规划

> 最后更新：2026-07-01
>
> 本文件是项目的**单一事实源（Single Source of Truth）**，记录已完成工作、里程碑路线、待办事项、深化方向与技术债。

---

## 🗺️ 里程碑路线图（先看这里）

| 里程碑 | 目标 | 关键交付物 | 状态 |
|:------:|------|-----------|:----:|
| **M1 数据地基** | HTML → 干净知识块 | `process/` 全链路 + 单测 | ✅ 完成 |
| **M2 模型底座** | Embedding/Reranker 可部署可微调 | TEI 部署 + SFT/DPO 训练 | 🔄 80%（待实机部署验证） |
| **M3 检索打通** | 知识块 → 可检索 → 精排 | `rag/` 索引 + 双模检索 + Reranker | 🔲 待开发 |
| **M4 智能问答** | 检索 → Agent 多轮问答 | `agent/` + RAG 工具注册 | 🔲 待开发 |
| **M5 可度量** | 端到端可量化评测 + A/B | `eval/` 四层指标 + 报告 | 🔲 待设计 |
| **M6 生产化** | 服务化 + 可观测 + CI | API 服务 + 监控 + 流水线 | 🔲 待规划 |

> 关键路径：**M3（RAG）是当前最高优先级**，它同时是 M4、M5 的前置依赖。建议下一步集中攻坚 M3。

---

## ✅ 已完成

### 1. process/ — HTML 数据处理与分块

- [x] 从 HtmlRAG 项目迁移并重构为 `process/` 子模块，`utils/` 独立于 `src/`
- [x] 大清洗：移除 DB 插库（step3）、检索查询（step4）、API 服务、db_utils 等无关内容
- [x] 职责聚焦：仅负责"把杂乱 HTML 变成干净的结构化知识块"
- [x] 对标 HtmlRAG 论文：`block_path` 字段输出、`html_content` 结构保留、BFS 裸文本块逻辑
- [x] 噪声清洗：SVG/input/button/nav/aside/footer/head/title、隐藏元素、模板残留、UI 噪声
- [x] 冗余包装展开、空标签清理、表格 colspan/rowspan 展开、混合内容分离
- [x] 中文适配：jieba 电商领域词典、`zh_char` 按字符计数
- [x] 配置脱敏：`.env` + `config/config.json` 双层配置，无硬编码

### 2. model/ — 模型推理与训练

- [x] 推理框架选型：TEI vs vLLM vs Infinity vs Xinference → **选定 TEI**（支持 Qwen3、双 API、Flash Attention）
- [x] `model/inference/tei_client.py` — TEI 客户端（embed/embed_batch/rerank/rerank_scores/health_check）
- [x] `model/inference/docker-compose-tei.yml` — Embedding(8080) + Reranker(8081) 一键部署
- [x] `model/trainer/reranker_ft.py` — Reranker SFT 监督微调（CrossEncoder，多级标签归一化）
- [x] `model/trainer/reranker_dpo.py` — **DPO 偏好优化**（chosen vs rejected，参考模型冻结）
- [x] `model/utils/build_dataset.py` — 数据集构造（Pointwise/Pairwise/DPO **三格式** + 困难负样本 + 多级标注）
- [x] 模型选型：Qwen3-Embedding-4B + Qwen3-Reranker-4B（精度/速度平衡，MTEB 69.45 分）

### 3. tests/ — 测试体系

- [x] 测试迁移至项目级 `tests/`，conftest 路径自适应（无需手动设 PYTHONPATH）
- [x] **264 个测试用例全部通过**，覆盖 HTML 清洗、表格展开、Block Tree、文档块生成、去重、端到端
- [x] 新增 `test_bugfix_regression.py`：配置路径、UI 噪声、大集群去重、**异步/同步分块一致性**

### 4. 代码审查与质量修复（本轮迭代）

- [x] 全量代码审查（P0/P1/P2/P3 分级），实测复现关键缺陷
- [x] **P0 修复**：`config.py` 路径计算错误（模块无法导入）、`conftest.py` 路径（测试无法采集）、训练脚本 import 即执行训练
- [x] **P1 修复**：脚本 PYTHONPATH/模块名、数据集负采样越界、SFT 标签归一化、jieba 输出路径一致性、异步分块丢正文
- [x] **P2/P3 修复**：deque 去重、正则预编译、超时读配置、旧版 amp API、死代码清理

### 5. 项目架构与文档

- [x] 四模块架构：`process/` → `model/` → `rag/` → `agent/`
- [x] `config/` 统一配置（config_loader.py + config.json + .env）
- [x] `scripts/` 运行脚本（process_HTMLdata.sh、build_JIEBAdict.sh）
- [x] **README.md 已编写**（含徽章、目录导航、重点难点、架构图 overview.png、详细模块说明）
- [x] 各模块文档：`process/README.md` ✅、`model/README.md` ✅
- [x] AI 生图提示词 `images/IMAGE_PROMPT.md` + 架构图 `images/overview.png`

---

## 🔲 待完成 — 核心链路

### 1. RAG 模块搭建（M3，🔥 最高优先级）

> 目标：打通"数据预处理 → RAG 检索 → Agent 调用"全流程

- [ ] **rag/indexing/** — 索引构建
  - [ ] 读取 `process/` 输出的 JSON 文档块，向量化后写入 Milvus + ES
  - [ ] Schema 设计：`global_chunk_idx` 主键、`text`/`html_content`/`summary`/`block_path`/`time` 字段
  - [ ] 增量更新（新增/删除/覆盖），避免全量重建
- [ ] **rag/retrieval/** — 检索模块
  - [ ] Milvus 向量检索（对接 TEI Embedding）
  - [ ] Elasticsearch 关键词检索（IK 分词 + `build_optimal_jieba_query`）
  - [ ] 双模融合：**RRF（Reciprocal Rank Fusion）** 或加权融合（权重需可调 + A/B）
  - [ ] 融合去重（复用 `deduplicate_ranked_blocks_pal`）
  - [ ] Reranker 精排（对接 TEI Reranker）
- [ ] **rag/pipeline.py** — 全流程编排：query 重写 → 双模检索 → 融合去重 → 精排 → Top-K
- [ ] **rag/config.py** — RAG 专属配置（collection_name、index_name、top_k、融合权重）
- [ ] 回填 `model/utils/build_dataset.py` 中 `chunks = [res[0]]` 占位，接入真实检索

### 2. Agent 模块搭建（M4，高优先级）

> 目标：让 Pi-Agent 能够按需调用 RAG 能力

- [ ] **agent/core/** — Agent 主循环（感知 → 决策 → 行动 → 观察）+ 工具注册调度
- [ ] **agent/tools/** — RAG 检索工具、知识库问答工具、业务工具（可扩展）
- [ ] **agent/memory/** — 短期记忆（对话历史）+ 长期记忆（用户画像/偏好）
- [ ] **安全护栏**：拒答策略、越权拦截、敏感词过滤、检索失败降级兜底
- [ ] **编排策略**：ReAct / Plan-and-Execute 选型，多工具协同

### 3. 难点追踪：RAG 与 Agent 的集成方式（⭐ 项目最大挑战）

> 核心问题：如何将 RAG 检索能力注册给 Pi-Agent，使 Agent 能够按需调用？

- [ ] **方案调研**
  - [ ] A：Tool-based — RAG 封装为 Agent Tool，function calling 调用（复杂问题）
  - [ ] B：Middleware — RAG 作为感知层中间件，自动注入检索结果（高频确定性问答）
  - [ ] C：Hybrid — 常见问题自动检索 + 复杂问题主动调用（生产倾向）
- [ ] **技术选型**：LangChain Tool / ReAct / Function Calling 适配；评估 Pi-Agent 工具注册接口
- [ ] **实现与验证**：RAG Tool 封装 → 注册 → 测试触发率与准确率
- [ ] **触发策略优化**：避免过度检索（拖慢+噪声）与漏检索（答非所问）

---

## 📌 M3 · RAG 模块 —— 可执行开发清单（含验收标准 DoD）

> 本节把 M3 拆为**可直接照着开发**的任务，标注依赖关系、接口约定与验收标准（Definition of Done）。
> 建议执行顺序：**T0 → T1 → T2 → T3 → T4 → T5 → T6 → T7**（T1/T2 可并行）。

### 目标目录结构

```
rag/
├── __init__.py
├── config.py              # RAG 专属配置（collection/index/top_k/融合权重）
├── schema.py              # Milvus/ES schema 定义与字段映射
├── indexing/
│   ├── __init__.py
│   ├── embedder.py        # 文本 → 向量（对接 TEI Embedding）
│   ├── milvus_index.py    # Milvus 建表 / 写入 / 增量更新
│   └── es_index.py        # ES 建索引 / 写入 / 增量更新
├── retrieval/
│   ├── __init__.py
│   ├── milvus_search.py   # 向量检索
│   ├── es_search.py       # 关键词检索（复用 build_optimal_jieba_query）
│   ├── fusion.py          # RRF / 加权融合 + 去重
│   └── reranker.py        # Reranker 精排（对接 TEI Reranker）
├── pipeline.py            # 全流程编排入口
└── README.md
tests/
└── test_rag_*.py          # 各环节单测 + 端到端
```

### T0 · 脚手架与配置（依赖：无）

- [ ] 创建 `rag/` 包结构与 `rag/config.py`（读取 `config/config.json` 的 `env_config`：milvus_host/es_host/collection_name/index_name）
- [ ] 新增配置项：`top_k_recall`（单路召回数，默认 20）、`top_k_final`（精排后返回数，默认 5）、`fusion_method`（rrf/weighted）、`fusion_weights`
- [ ] `rag/schema.py` 定义文档块字段 ↔ Milvus/ES 字段映射（对齐 process 输出：`chunk_idx/page_name/title/page_url/text/html_content/block_path/summary/time`）
- **DoD**：`python -c "from rag.config import RAG_CONFIG; print(RAG_CONFIG)"` 正常输出；无硬编码地址。

### T1 · 索引构建 `rag/indexing/`（依赖：T0）

- [ ] `embedder.py`：封装 TEI Embedding，`embed_texts(texts) -> List[vec]`，复用 `model/inference/tei_client.py`
- [ ] `milvus_index.py`：`create_collection()` / `upsert_blocks(blocks)` / `delete_by_page(page_url)`
- [ ] `es_index.py`：`create_index()`（IK 分词 mapping）/ `bulk_index(blocks)` / `delete_by_page(page_url)`
- [ ] 增量更新：以 `page_url` 为粒度先删后插，避免全量重建
- [ ] `scripts/build_index.sh`：读取 `process/dataset/html_cleaned_block/*.json` → 写入 Milvus + ES
- **DoD**：给定示例 JSON 块，能成功写入并在两库中查到对应条数；重复运行同一文件不产生重复数据（幂等）。

### T2 · 双模检索 `rag/retrieval/`（依赖：T1）

- [ ] `milvus_search.py`：`search(query, top_k) -> List[block]`（query 经 TEI 向量化后 ANN 检索）
- [ ] `es_search.py`：`search(query, top_k) -> List[block]`（jieba 分词 → `build_optimal_jieba_query` → ES）
- [ ] 统一返回结构：`{block..., "score": float, "source": "milvus"/"es"}`
- **DoD**：对一条示例 query，两路各能返回 ≤ top_k 条带分数结果；分数单调、可排序。

### T3 · 融合去重 `rag/retrieval/fusion.py`（依赖：T2）

- [ ] 实现 **RRF**：`rrf_fuse(results_list, k=60) -> List[block]`
- [ ] 实现加权融合（可选）：按 `fusion_weights` 归一化分数相加
- [ ] 融合后复用 `deduplicate_ranked_blocks_pal` 去重
- **DoD**：两路结果融合后无重复 `chunk_idx`；单元测试验证 RRF 排序正确（构造已知输入→预期顺序）。

### T4 · Reranker 精排 `rag/retrieval/reranker.py`（依赖：T2）

- [ ] 封装 TEI Reranker：`rerank(query, blocks, top_k) -> List[block]`（复用 `tei_client.rerank`）
- [ ] 输入文档拼接策略：`title + summary + text`（截断到 Reranker max_length）
- [ ] 无 Reranker 服务时降级：跳过精排，直接返回融合结果并告警
- **DoD**：精排后返回 `top_k_final` 条，顺序按 rerank 分数降序；服务不可用时不崩溃。

### T5 · 全流程编排 `rag/pipeline.py`（依赖：T3、T4）

- [ ] `retrieve(query, dialogue=None) -> List[block]`：
      query 重写（复用 `llm_api.rewrite_query_*`，可选）→ 双模检索 → 融合去重 → Reranker 精排 → Top-K
- [ ] 全链路计时日志（各阶段耗时），失败降级（任一路失败仍能返回）
- **DoD**：`retrieve("如何处理广告限流")` 端到端返回 Top-K 带 `page_url`/`block_path` 的结果；单条 query 全链路有日志。

### T6 · 回填依赖（依赖：T5）

- [ ] 替换 `model/utils/build_dataset.py` 中 `chunks = [res[0]]` 占位，接入 `rag.retrieval` 真实召回
- **DoD**：`build_dataset` 能基于真实检索产出 ≥1 条 Pointwise/Pairwise/DPO 样本（非空）。

### T7 · 测试（依赖：T1~T5，可随开发同步）

- [ ] `test_rag_fusion.py`：RRF/加权融合、去重（纯逻辑，无需外部服务）
- [ ] `test_rag_schema.py`：字段映射正确性
- [ ] `test_rag_retrieval.py`：mock Milvus/ES/TEI，验证 pipeline 编排与降级逻辑
- [ ] （可选）集成测试：真实起 Milvus/ES/TEI 的 e2e（打 `@pytest.mark.integration`，CI 默认跳过）
- **DoD**：纯逻辑测试可在无外部依赖环境跑通并纳入 `pytest tests/`；`264 → 264+N` 全绿。

### 建议接口约定（供实现参考）

```python
# rag/pipeline.py
def retrieve(
    query: str,
    dialogue: list | None = None,   # 多轮历史，用于 query 重写
    top_k: int | None = None,        # 覆盖默认 top_k_final
) -> list[dict]:
    """返回精排后的 Top-K 文档块，每条含 text/html_content/page_url/block_path/score。"""
```

> ✅ 里程碑判定：当 T5 + T7 完成，M3 视为达成，即可解锁 M4（Agent 调用 `rag.pipeline.retrieve`）与 M5（评测接入真实检索）。

---

## 🚀 深化与优化方向（值得后续推进）

### A. process/ 深化 —— HtmlRAG 论文核心「两阶段剪枝」

> ✅ **已实现**：`process/src/html_pruner.py` 落地了论文 HtmlRAG 的核心贡献 **Block Tree Pruning（两阶段剪枝）**，并有 `tests/test_html_pruner.py`（29 项）全覆盖。这是查询时（query-time）操作，将由 `rag/` 在检索链路中调用。

- [x] **Embedding-based 粗剪枝**（`prune_by_embedding`）：粗粒度块树上用嵌入余弦相似度快速裁掉与 query 无关的块
- [x] **细粒度精剪枝**（`prune_by_reranker`）：细粒度块树上用 Reranker 交叉编码精细打分二次剪枝（论文「生成式细粒度剪枝」的可部署替代）
- [x] **贪心剪枝算法**（`greedy_prune_indices`）：token 预算内保留高分块，纯函数可单测
- [x] 剪枝后 HTML 保留结构作为送入 LLM 的上下文（对标论文"HTML 优于纯文本"）
- [x] 打分后端可注入 + 服务不可用时优雅降级（不剪枝）
- [ ] 接入 `rag/pipeline.retrieve`：检索得到候选块后，对其 `html_content` 做两阶段剪枝再送 LLM
- [ ] 真实 embedding/reranker 服务下的剪枝质量评估（保留率、Faithfulness）
- [ ] 分块参数自动调优：`max_node_words` / `min_node_words` 针对电商语料网格搜索

### B. model/ 深化

- [ ] **Embedding 微调**（当前只微调 Reranker）：对比学习 + 领域内 query-doc 对
- [ ] TEI **实机部署验证**：吞吐、延迟、显存实测（补齐 M2 最后 20%）
- [ ] 模型量化/加速：INT8/FP8，降低服务成本
- [ ] SFT → DPO 全流程跑通并对比收益（用 M5 评测体系量化）

### C. rag/ 深化

- [ ] **引用与溯源（Citation）**：回答标注来源 `page_url` / `block_path`，可信可查
- [ ] **上下文压缩**：结合 A 的剪枝，控制送入 LLM 的 token 量
- [ ] 查询理解：意图识别、多路召回、query 改写多样化
- [ ] 缓存层：query / embedding 结果缓存，降低重复计算

### D. agent/ 深化

- [ ] 多轮会话状态管理与并发隔离
- [ ] 工具调用可观测（每步 trace）、失败重试与降级
- [ ] 人工接管（Human-in-the-loop）与转人工策略

---

## 🤖 智能客服能力分层规划（从"问答机器人"到"智能客服体"）

> 抛开 RAG 检索与 Agent 框架本身，要让系统成为一个**真正智能的客服智能体**，还需在以下 9 层能力上补齐。
> 其中 **③业务工具、⑦安全护栏、⑥反馈闭环、①对话理解** 是把"问答机器人"升级为"智能客服体"的关键。

### 能力分层总览

```
   ⑧渠道  Web / 小程序 / 飞书 / 企微（多渠道接入）
   ⑦安全  敏感词 / Prompt注入 / PII脱敏 / 拒答边界    ← 贯穿全链路
   ─────────────────────────────────────────────
   ①理解  意图识别 · 情绪识别 · 澄清反问 · 会话状态
   ②检索  RAG（开发中）+ 两阶段剪枝 + 溯源 + 时效
   ③行动  业务工具：订单/物流/退款/优惠券（会办事）   ← 智能体的灵魂
   ④生成  忠实度 · 引用标注 · 客服话术 · 流式
   ⑤记忆  短期对话 + 长期用户画像 + 历史工单
   ─────────────────────────────────────────────
   ⑥闭环  满意度反馈 · 会话质检 · Bad Case 回流重训    ← 独特优势
   ⑨观测  解决率 / 转人工率 / 端到端评测（见 M5）
```

### ③ 业务工具层（🥇 P0，最高价值）

> 客服智能体与普通 RAG 问答的本质区别：用户问"订单到哪了"，纯 RAG 只能答规则，智能体应真的去查物流。

- [ ] 业务 API 工具集：订单查询、物流跟踪、退款/退货、优惠券发放、账户状态
- [ ] 工具编排（ReAct / Plan-and-Execute）：自主决定"查订单→判断退款条件→发起退款"
- [ ] 参数槽填充（Slot Filling）：缺订单号时主动追问而非失败
- [ ] 落点：`agent/tools/`，RAG 检索工具只是其一，业务工具才是"客服"主体

### ① 对话理解层（🥇 P0）

> 复用 `llm_api.py` 已有的 Query 重写（多轮指代补全）与 `infer_chunk_category`（分类）思路向上扩展。

| 能力 | 作用 | 与现有结合 |
|------|------|-----------|
| 意图识别 | 咨询/投诉/查询/闲聊/转人工 分流 | 复用分类思路，加 query 级意图分类 |
| 情绪识别 | 检测愤怒/焦虑 → 触发安抚话术或转人工 | 新增，客服场景刚需 |
| 澄清反问 | query 模糊时主动反问，而非硬答 | 补充到 Query 重写下游 |
| 多意图拆解 | "查订单顺便退款"拆成两个子任务 | Agent 编排配合 |

- [ ] 意图识别（分流咨询/投诉/查询/闲聊/转人工）
- [ ] 情绪识别（愤怒/焦虑 → 安抚话术或转人工）
- [ ] 澄清反问（模糊 query 主动反问）
- [ ] 多意图拆解（一句多诉求拆分为子任务）

### ⑦ 安全与合规护栏（🥇 P0，🎯 数据优势）

> 已有 AI 管控/质检的数据与经验（违规判定标准），可复用做客服护栏。

- [ ] 输入护栏：Prompt 注入检测、敏感/违规提问拦截（复用管控标准思路）
- [ ] 输出护栏：回答合规校验、PII（手机号/身份证）自动脱敏
- [ ] 拒答边界：超出知识库范围不硬编，明确说明
- [ ] 越权防护：退款等敏感操作二次确认/权限校验

### ⑥ 反馈与持续学习闭环（🥈 P1，🎯 独特护城河）

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

### ④ 回答生成层（🥈 P1）

- [ ] 忠实度/防幻觉：强制 grounding 到检索结果
- [ ] 引用标注（Citation）：标注 `page_url`/`block_path`（chunk 已带字段，天然支持）
- [ ] 客服话术风格：品牌调性、礼貌用语、结构化回答（步骤/链接）
- [ ] 流式输出（SSE 逐字返回）

### ② 检索深化（🥈 P1，详见「深化方向 A/C」）

- [x] 两阶段剪枝（HtmlRAG 论文核心，已实现于 `process/src/html_pruner.py`）；待接入 rag 检索链路
- [ ] 时效性排序：利用 chunk 的 `time` 字段对规则类内容加权
- [ ] 无答案兜底：检索不到时澄清反问或明确"不知道"

### ⑤ 记忆与个性化（🥉 P2）

- [ ] 短期记忆（对话历史）
- [ ] 长期记忆（用户画像、历史工单、VIP 等级 → 个性化话术与优先级）
- [ ] 会话摘要（长对话自动压缩，防上下文爆炸）

### ⑧ 渠道与 ⑨ 多模态（🥉 P2）

- [ ] 多渠道接入（Web / 小程序 / 企微 / 飞书）
- [ ] 多模态：用户发商品截图 → 图片理解（VLM）
- [ ] 在线指标：自助解决率、转人工率、首响时间、CSAT

### 优先级速览

| 优先级 | 方向 | 理由 |
|:------:|------|------|
| 🥇 P0 | ③业务工具 + ①意图/转人工 | 决定"是不是真客服"，价值最高 |
| 🥇 P0 | ⑦安全护栏 | 上线合规底线，有数据优势 |
| 🥈 P1 | ⑥质检 + Bad Case 回流闭环 | 独特护城河（DPO + QC 经验） |
| 🥈 P1 | ④忠实度 + 引用 + ②剪枝 | 直接提升回答质量 |
| 🥉 P2 | ⑤长期记忆 / ⑧多渠道多模态 | 锦上添花，后期扩展 |

> 💡 **一句话**：现有链路（process→model→rag）是扎实的「知识问答」；要变「智能客服体」，最该补的是 **③能办事、⑦守得住、⑥能进化**——后两者恰好复用你已有的管控数据与训练 pipeline，是差异化壁垒。

---

## 🏭 工程化与生产化（M6）

- [ ] **API 服务层**：用 `requirements.txt` 中已声明的 FastAPI 暴露 `/chat` `/retrieve` 接口
  - [ ] 流式输出（SSE）、限流、鉴权、多租户
  - [ ] 灰度发布与回滚
- [ ] **依赖分层**：拆分 `requirements/`（process 仅需 bs4/jieba/sklearn，不必装 torch/pymilvus）
  - [ ] `requirements-process.txt` / `requirements-model.txt` / `requirements-rag.txt`
- [ ] **CI/CD**：GitHub Actions / 蓝盾流水线自动跑 `pytest tests/`（含冒烟导入测试）
- [ ] **容器化**：整体 `docker-compose`（process + TEI + Milvus + ES + API 一键起）
- [ ] **可观测性**：结构化日志、Prometheus 指标（延迟/QPS/错误率）、链路追踪
- [ ] **配置校验**：启动时校验 config.json / .env 完整性，缺失即 fail-fast

---

## 🗄️ 数据接入与知识库运营

> 当前假设 HTML 已存在于 `dataset/html_source/`，但"数据从哪来、如何更新"尚未规划。

- [ ] **数据接入**：爬虫 / 导出接口对接（电商平台帮助中心、规则文档）
- [ ] **增量抓取与去重**：定时抓取 + 内容指纹去重 + 变更检测
- [ ] **知识库更新流程**：抓取 → 清洗分块 → 索引更新（对接 rag/indexing 增量能力）
- [ ] **数据版本管理**：知识块快照与版本号，支持回滚与对比
- [ ] **数据质量校验**：空块率、超长块率、乱码检测、时间戳缺失告警

---

## 📈 评测体系设计（M5，核心难点）

> 核心问题：如何构造一套可行的评价体系，量化评估项目效果，以及**每项修改的收益与回报**？

### 当前痛点

缺乏统一可量化评测，导致无法判断：清洗优化 / 换 Embedding / Reranker 微调 / Agent 集成到底带来正收益还是负收益，只能凭主观感觉，无法做 A/B 对比。

### 四层评测体系

```
Layer 1 数据处理质量（process/ 已可用）
    ↓
Layer 2 检索召回（rag/ 开发后可用）
    ↓
Layer 3 精排质量（model/ + rag/ 联合）
    ↓
Layer 4 端到端回答（agent/ 开发后可用）
```

**Layer 1 — 数据处理质量**

| 指标 | 含义 | 测量方法 |
|------|------|---------|
| 噪声残留率 | 清洗后残留噪声比例 | 标注噪声，统计清洗后残留数 |
| 内容保留率 | 有效内容保留比例 | 对比清洗前后有效文本量 |
| 压缩比 | 清洗后/原始体积 | `len(cleaned)/len(raw)` |
| 分块合理性 | 块大小分布 | 统计块大小 P50/P95/最小值 |
| block_path / html_content 覆盖率 | 关键字段非空比例 | `count(!="")/total` |

**Layer 2 — 检索召回**

| 指标 | 含义 | 测量方法 |
|------|------|---------|
| Recall@K | Top-K 命中正确文档比例 | query→gold_doc_id 评测集 |
| MRR | 正确文档排名倒数均值 | `mean(1/rank)` |
| NDCG@K | 归一化折损累积增益 | 按 relevance 分级计算 |
| 检索延迟 P95 / QPS | 性能 | 并发压测 |

**Layer 3 — 精排质量**

| 指标 | 含义 | 测量方法 |
|------|------|---------|
| Top-1 命中率 | 精排后第一名是否正确 | 评测集统计 |
| 排序改善幅度 | 精排前后 NDCG 提升 | `NDCG_after - NDCG_before` |
| Reranker 延迟 | 单次 rerank 耗时 | TEI 接口计时 |

**Layer 4 — 端到端回答**

| 指标 | 含义 | 测量方法 |
|------|------|---------|
| Answer Accuracy | 回答是否正确 | 人工 / LLM-as-Judge |
| Faithfulness | 是否忠实文档（无幻觉） | LLM-as-Judge |
| Relevance | 与问题相关度 | LLM-as-Judge 1-5 |
| 完整性 | 是否覆盖问题各方面 | LLM-as-Judge |
| 端到端延迟 / Agent 触发率 | 性能与策略 | 全链路计时 / 触发统计 |

### 评测数据集与 A/B 框架

- [ ] 采样 200-500 文档块，人工构造 query（事实/比较/多跳/拒答型）
- [ ] 标注 gold_doc_id + relevance 等级（0/1/2），JSONL 版本化管理
- [ ] 实现 `scripts/run_eval.py` + Markdown 对比报告生成器（ΔRecall/ΔNDCG/Δ延迟/ΔAccuracy）
- [ ] 评测结果版本化存储 `eval_results/YYYYMMDD_HHMMSS/`

### 每项修改的收益评估

| 修改类型 | 关注指标 | 期望收益 |
|---------|---------|---------|
| 清洗算法优化 | 噪声残留率↓、内容保留率↑ | 检索噪声减少，Recall↑ |
| 两阶段剪枝 | 上下文 token↓、Faithfulness↑ | 精度提升、成本降低 |
| 换用 Qwen3-Embedding | Recall@10↑、MRR↑ | 更好语义理解 |
| Reranker 微调（SFT→DPO） | Top-1 命中率↑、NDCG↑ | 排序更准 |
| 分块参数调整 | 分块合理性↑、Recall↑ | 更好粒度 |
| Agent RAG 集成 | Answer Accuracy↑、触发率↑ | 回答质量提升 |

---

## ⚠️ 已知风险与技术债

- [ ] `model/utils/build_dataset.py` 仍用 `chunks = [res[0]]` 占位，依赖 rag/ 检索就绪后替换
- [ ] `model/` 训练脚本尚未在真实 GPU + 数据上跑通（仅通过语法/结构/导入验证）
- [ ] `process/utils/config.py` 与 `config/config_loader.py` 存在重复配置加载逻辑，建议抽公共层
- [ ] `rewrite_query_*` / 摘要生成依赖 vLLM 服务，缺少服务不可用时的整体降级策略
- [ ] 缺少端到端冒烟测试（import 级）纳入 CI，防止重构再次引入路径断裂

---

## 📊 项目进度总览

| 模块 | 状态 | 进度 | 说明 |
|------|------|------|------|
| `process/` 数据处理 | ✅ 完成 | 100% | 293 测试通过；**两阶段剪枝已实现**（`html_pruner.py`），待 rag 链路接入 |
| `model/` 推理与训练 | 🔄 基础完成 | 80% | TEI 客户端 + SFT/DPO；待实机部署验证 |
| `rag/` 检索增强 | 🔲 待开发 | 0% | **当前最高优先级（M3）** |
| `agent/` 智能体 | 🔲 待开发 | 0% | 依赖 rag/ |
| `tests/` 单元测试 | 🔄 持续完善 | 68% | process 已覆盖（含剪枝），rag/agent 待补 |
| `eval/` 评测体系 | 🔲 待设计 | 10% | 四层指标已规划，待实现脚本与数据集 |
| 工程化/生产化 | 🔲 待规划 | 5% | Docker(TEI) 已有；API/CI/监控待做 |
| 数据接入运营 | 🔲 待规划 | 0% | 数据来源与增量更新未规划 |
| `config/` 配置管理 | ✅ 完成 | 100% | — |
| `scripts/` 运行脚本 | ✅ 完成 | 100% | — |
| `README.md` 项目文档 | ✅ 完成 | 90% | 主 README + process/model README 完成，rag/agent 待补 |
