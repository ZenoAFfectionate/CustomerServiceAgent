# 🤖 CustomerServiceAgent

> 🏢 面向企业级电商场景的智能客服系统 —— 打通「原始 HTML 知识」→「RAG 检索」→「Agent 问答」的全链路解决方案。

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="python"/>
  <img src="https://img.shields.io/badge/HtmlRAG-TheWebConf%202025-green.svg" alt="paper"/>
  <img src="https://img.shields.io/badge/inference-TEI-orange.svg" alt="tei"/>
  <img src="https://img.shields.io/badge/RAG-RAGFlow-purple.svg" alt="ragflow"/>
  <img src="https://img.shields.io/badge/Agent-Pi--Agent-red.svg" alt="pi-agent"/>
  <img src="https://img.shields.io/badge/license-research--only-lightgrey.svg" alt="license"/>
</p>

---

## 📑 目录导航

- [📖 项目简介](#-项目简介)
- [🎯 重点与难点](#-重点与难点核心看点)
- [📂 目录结构](#-目录结构)
- [🏗️ 技术架构详解](#️-技术架构详解)
  - [📦 process — HTML 数据处理与分块](#-process--html-数据处理与分块)
  - [🤖 model — 模型推理与训练](#-model--模型推理与训练)
  - [🔍 rag — RAG 检索增强生成](#-rag--rag-检索增强生成)
  - [💬 agent — 智能体](#-agent--智能体)
- [🚀 快速开始](#-快速开始)
- [🛠️ 技术栈](#️-技术栈)
- [📋 项目状态](#-项目状态)

---

## 📖 项目简介

许多企业沉淀了海量以 **HTML 网页**形式存在的运营知识（电商平台规则、帮助中心文档、产品政策、投放指南等）。这些网页有三大天然缺陷：

- **结构杂乱**：大量 `<div>` 深层嵌套、`<nav>`/`<footer>`/`<svg>` 等无内容标签、模板残留文本、隐藏元素；
- **信息密集**：复杂表格（含 `colspan`/`rowspan` 合并单元格）、图文混排，纯文本抽取会破坏语义；
- **难以检索**：直接切块会割裂标题与正文的从属关系，导致召回不准、答非所问。

如果把这些原始网页直接喂给大模型，会遇到 **上下文超长、噪声干扰、语义割裂** 三重问题，最终表现为「答不准、答不全、答非所问」。

**CustomerServiceAgent** 是一套端到端方案，完整覆盖从「原始 HTML 网页」到「用户精准问答」的全流程：

| 痛点 | 对应模块 | 解决思路 |
|------|---------|---------|
| HTML 结构杂乱、无法直接检索 | 📦 `process/` | 对标 **HtmlRAG** 论文做无损清洗 + Block Tree 语义分块，保留 HTML 结构 |
| 模型推理慢、部署复杂 | 🤖 `model/` | 用 **TEI** 框架一键部署 Qwen3 Embedding/Reranker，Flash Attention 加速 |
| 检索不精准、召回率低 | 🔍 `rag/` | 基于 **RAGFlow** 的向量 + 关键词双模检索 + Reranker 精排 |
| 客服机器人无法多轮对话 | 💬 `agent/` | 基于 **Pi-Agent** 的多轮问答，按需调用 RAG 工具 |

### 🔄 全流程架构图

![CustomerServiceAgent 全流程架构](images/overview.png)

#### 架构说明

上图展示了从「原始 HTML 网页」到「用户精准问答」的完整数据流，共经历 **四个核心模块**，数据自左向右单向流动：

```
原始 HTML → [process 清洗分块] → 结构化知识块 JSON
                                      ↓
                              [model 向量化 + 精排]
                                      ↓
用户提问 → [rag 双模检索 + 融合去重 + Reranker 精排] → Top-K 文档
                                      ↓
                              [agent 多轮对话 + 工具调用] → 精准回答
```

**① process — 数据处理层（🟦 沉稳蓝）**

原始 HTML 网页经 `main.py` 一键处理：先做**无损清洗**（移除 SVG/nav/script 等噪声标签、展开 `colspan`/`rowspan` 合并单元格、展开冗余嵌套包装），再做 **Block Tree 语义分块**（BFS 遍历 DOM 树、按 heading 层级拆分、表格按行切分、混合内容分离）。输出同时携带 `text`（纯文本）、`html_content`（保留 HTML 结构）与 `block_path`（块路径），为下游检索提供结构化输入。

**② model — 模型推理层（🟩 森绿色）**

采用 HuggingFace **TEI（Text Embeddings Inference）** 框架，以 Docker 一键部署两个推理服务：
- **Qwen3-Embedding-4B**（端口 8080）：将知识块与用户 query 向量化，支持动态批处理与 Flash Attention 加速
- **Qwen3-Reranker-4B**（端口 8081）：对检索候选文档做交叉编码精排，从 ~20 个候选中精选 Top-5

同时支持 SFT 监督微调 + DPO 偏好优化，可训练领域专属 Reranker。

**③ rag — 检索增强层（🟪 浅紫色）**

基于 **RAGFlow** 构建，执行"双模检索 → 融合去重 → 精排"三步链路：
- **向量检索**（Milvus）：语义匹配，"广告投放"能召回"推广策略"
- **关键词检索**（Elasticsearch + IK 分词）：精确匹配，"千川"能命中"巨量千川"
- **融合去重**：RRF 或加权融合两路结果，再用 TF-IDF + cosine 去除重复块
- **Reranker 精排**：TEI Reranker 对融合后的候选集做二次排序

**④ agent — 智能问答层（🟦 深沉蓝）**

基于 **Pi-Agent** 构建，核心是「感知 → 决策 → 行动 → 观察」循环：
- 接收用户问题后，先做 **Query 重写**（多轮对话指代补全："那个怎么办"→"广告限流后怎么办"）
- Agent 自主判断是否需要检索，通过 **function calling** 调用 RAG 工具获取知识
- 结合检索结果与对话记忆生成最终回答，支持多轮上下文与用户画像记忆

> 💡 **架构亮点**：各模块职责单一、接口清晰——`process` 输出标准 JSON 块，`model` 提供嵌入与排序 API，`rag` 消费前两者产出检索结果，`agent` 编排调用。任一模块可独立替换或升级，不影响其他层。

### 🎯 核心能力一览

| 能力 | 模块 | 一句话说明 |
|------|------|------|
| 🧹 HTML 清洗与分块 | `process/` | 移除噪声标签、展开合并单元格、BFS 语义分块、LLM 摘要生成 |
| 🤖 模型推理部署 | `model/` | TEI 部署 Qwen3-Embedding + Qwen3-Reranker，双 API 一体化 |
| 🎓 模型微调优化 | `model/` | SFT 监督微调（多级标注 0/1/2）+ DPO 偏好优化 |
| 🔍 RAG 检索增强 | `rag/` | 向量 + 关键词双模检索，融合去重后 Reranker 精排 |
| 💬 智能问答 | `agent/` | 多轮对话、Query 重写、RAG 工具按需调用 |

---

## 🎯 重点与难点（核心看点）

> 本节是理解本项目**技术含金量**的关键。下表按「实现难度 × 业务价值」排序，⭐ 越多代表越是核心攻坚点。

| 难点 | 所在模块 | 难度 | 状态 |
|------|---------|:----:|:----:|
| ① HTML 无损清洗与结构化分块（对标论文） | `process/` | ⭐⭐⭐⭐⭐ | ✅ |
| ② 复杂表格展开与图文混合内容分离 | `process/` | ⭐⭐⭐⭐ | ✅ |
| ③ 中文电商领域的分词与去重适配 | `process/` | ⭐⭐⭐ | ✅ |
| ④ Reranker 训练数据构造（困难负样本 + DPO） | `model/` | ⭐⭐⭐⭐ | ✅ |
| ⑤ 向量 + 关键词双模检索融合 | `rag/` | ⭐⭐⭐⭐ | 🔲 |
| ⑥ **RAG 如何注册给 Agent 按需调用** | `agent/` | ⭐⭐⭐⭐⭐ | 🔲 |
| ⑦ 端到端可量化评测体系 | 全局 | ⭐⭐⭐⭐ | 🔲 |

下面对最核心的几个难点展开说明。

### 🔥 难点①：HTML 无损清洗与结构化分块

这是整个项目的地基，直接对标 [HtmlRAG (TheWebConf 2025)](https://arxiv.org/abs/2411.02959) 论文实现。核心矛盾是——**既要移除噪声，又不能破坏语义结构**。

- **无损结构压缩**：多层单嵌套 `<div><div><p>文本</p></div></div>` 自底向上合并为 `<p>文本</p>`，同时**保留** `<h1>~<h6>` 标题层级（包装为 `hN_domain`），保证「标题—正文」从属关系不丢失。
- **Block Tree + 粒度控制**：用 **BFS** 遍历 DOM（对标论文 Algorithm 1），通过 `max_node_words` / `min_node_words` 双阈值控制块粒度——太大超上下文、太小丢语义，需要在拆分与合并之间动态权衡；对超阈值节点继续下钻，对裸文本（直接附属于节点、不在子标签内的文本）单独成块以避免信息丢失。
- **保留 HTML 结构（论文核心观点）**：输出同时携带 `text`（纯文本）、`html_content`（保留标签结构）与 `block_path`（如 `html>body>div0>p`，唯一标识块，可用于后续剪枝）。论文核心结论正是 **"HTML is Better Than Plain Text"**。

### 🔥 难点②：复杂表格展开与图文混合内容分离

电商知识库表格密集，是抽取质量的重灾区：

- **合并单元格展开**：将 `colspan` / `rowspan` 还原为标准矩阵，忽略 `0` 值占位单元格，保证每一行都能独立被检索命中；
- **按行切分且每块带表头**：长表格按行切分为多个块，**每个块都以表头开头**，使单块脱离上下文也可被理解；单行超长时强制独立成块；
- **混合内容分离**：同一区块内的「正文 + 表格」被分别提取为不同文档块（`_extract_mixed_content`），避免表格挤占正文语义。

### 🔥 难点④：Reranker 训练数据构造与 SFT → DPO

排序质量取决于训练数据质量，本项目在数据构造上做了三层设计：

- **多级相关性标注**：用 LLM 对候选文档打 `0/1/2`（无关 / 部分相关 / 高度相关），比单纯二分类信息量更大；
- **困难负样本**：从**同页面不同段落**采样——主题相近但内容不同，最能锻炼模型的区分力；辅以随机负样本作为基础对照；
- **两阶段训练**：先 **SFT** 学「绝对相关性分数」，再 **DPO** 在偏好对（chosen vs rejected）上学「相对排序」，无需显式奖励模型，收敛更稳。

### 🔥 难点⑥：RAG 如何注册给 Agent（本项目最大挑战，持续追踪）

> 这是打通「检索能力」与「智能体决策」的关键，也是目前仍在攻坚的核心问题。

需要回答：**Agent 何时、以何种方式调用 RAG？** 目前规划三条候选路径：

| 方案 | 思路 | 适用 |
|------|------|------|
| A. Tool-based | 将 RAG 检索封装为 Agent Tool，通过 function calling 调用 | 复杂问题、需模型自主判断 |
| B. Middleware | RAG 作为感知层中间件，自动注入检索结果 | 高频、确定性问答 |
| C. Hybrid | 常见问题自动检索 + 复杂问题 Agent 主动调用 | 生产综合场景（倾向） |

难点在于**触发策略**：过度检索会拖慢响应、引入噪声；漏检索则答非所问。详见 [TODO.md](TODO.md)。

### 🔥 难点⑦：端到端可量化评测体系

缺乏统一评测，就无法判断「每一次修改」到底带来了正收益还是负收益。规划了**四层评测体系**：

```
Layer 1 数据处理质量  →  噪声残留率 / 内容保留率 / 压缩比 / 分块合理性
Layer 2 检索召回质量  →  Recall@K / MRR / NDCG@K / 延迟 P95
Layer 3 精排质量      →  Top-1 命中率 / 排序改善幅度
Layer 4 端到端回答    →  Accuracy / Faithfulness / Relevance（LLM-as-Judge）
```

---

## 📂 目录结构

> 下述结构与仓库实际布局一致：`process/src/` 存放核心处理逻辑，`process/utils/` 存放通用工具（配置、LLM 客户端、分词词典）。

```
CustomerServiceAgent/
│
├── 📁 process/                           # HTML 数据处理与分块（仅负责"网页 → 知识块"）
│   ├── src/                              #   核心处理逻辑
│   │   ├── main.py                       #     全流程入口（清洗 + 分块，支持 --step 分步执行）
│   │   ├── html_utils.py                 #     HTML 清洗与 Block Tree 分块（对标 HtmlRAG）
│   │   ├── text_process_utils.py         #     文档块生成（block_path/html_content）+ 去重
│   │   └── html_pruner.py                #     两阶段块树剪枝（HtmlRAG 论文核心）
│   ├── utils/                            #   通用工具 / 基础设施
│   │   ├── config.py                     #     process 专属配置加载 + 日志
│   │   ├── llm_api.py                    #     LLM 摘要 / 问题生成 / Query 重写（vLLM/ChatGLM）
│   │   └── jieba_util.py                 #     jieba 电商领域自定义词典构建
│   ├── dataset/                          #   数据（词典、HTML 源、测试 HTML）
│   └── logs/                             #   运行日志
│
├── 📁 model/                             # 模型推理与训练
│   ├── inference/                        #   TEI 推理部署
│   │   ├── tei_client.py                 #     TEI 客户端（embed / rerank / health_check）
│   │   └── docker-compose-tei.yml        #     Docker Compose 一键部署
│   ├── trainer/                          #   模型微调
│   │   ├── reranker_ft.py                #     SFT 监督微调（CrossEncoder）
│   │   └── reranker_dpo.py               #     DPO 偏好优化
│   └── utils/                            #   训练数据工具
│       └── build_dataset.py              #     数据集构造（Pointwise/Pairwise/DPO 三格式）
│
├── 📁 rag/                               # RAG 检索增强生成（基于 RAGFlow，待开发）
├── 📁 agent/                             # 智能体（基于 Pi-Agent，待开发）
│
├── 📁 config/                            # 全局配置管理
│   ├── config_loader.py                  #   项目级配置加载器（供 model/ 等使用）
│   ├── config.json                       #   结构化配置（不入库）
│   └── config.example.json               #   配置模板
│
├── 📁 tests/                             # 单元测试（~245 个用例）
│   ├── conftest.py                       #   pytest 共享配置
│   ├── test_html_utils*.py               #   HTML 清洗与分块测试
│   ├── test_text_process_utils*.py       #   文本处理 / 去重测试
│   ├── test_jieba_util.py                #   jieba 词典测试
│   ├── test_algorithm_optimization.py    #   算法优化验证
│   └── test_algorithm_completeness.py    #   算法完整性补充
│
├── 📁 scripts/                           # 运行脚本
│   ├── process_HTMLdata.sh                #   数据处理流水线（清洗 → 分块）
│   ├── build_JIEBAdict.sh                 #   jieba 词典构建
│   ├── run_AGTserver.sh                   #   Agent 服务启动（待实现）
│   └── run_RAGserver.sh                   #   RAG 服务启动（待实现）
│
├── 📁 images/                            # 图片资源
│   ├── overview.png                      #   全流程架构图
│   └── IMAGE_PROMPT.md                   #   架构图 AI 生图提示词
│
├── 📄 .env.example                       # 环境变量模板
├── 📄 requirements.txt                   # Python 依赖
├── 📄 TODO.md                            # 项目规划与进度
└── 📄 README.md                          # 本文件
```

---

## 🏗️ 技术架构详解

### 📦 process — HTML 数据处理与分块

对标 [HtmlRAG](https://arxiv.org/abs/2411.02959)（TheWebConf 2025）论文实现，核心算法与论文一致。

> **职责边界**：只负责"把杂乱的 HTML 网页变成干净的结构化知识块"，**不涉及**数据库插库与检索查询（交给 `rag/`）。这种单一职责设计让 process 可独立测试、独立复用。

#### 阶段 1：HTML 清洗 🧹

| 清洗规则 | 说明 |
|---------|------|
| 噪声标签移除 | `script` / `style` / `svg` / `nav` / `aside` / `footer` / `head` / `title` / `meta` / `input` / `button` 等 |
| 隐藏元素移除 | `display:none` / `visibility:hidden` / `.hidden` class |
| 模板文本清除 | `{{PLACEHOLDER}}` 等模板残留（仅替换占位符，保留正常文本） |
| 空标签清理 | 完全空标签、仅含空白、仅含 `<br>` 的标签 |
| 表格展开 | `colspan` / `rowspan` 合并单元格 → 标准矩阵 |
| 冗余包装展开 | 多层单嵌套 `<div><div><p>` → `<p>`（自底向上迭代 3 轮） |
| 标题域包装 | `<h1>`~`<h6>` 按层级包装为 `<div class="hN_domain">` |
| 不可见字符清理 | 零宽字符、控制字符 |

#### 阶段 2：结构化分块 🧩

| 算法 | 说明 |
|------|------|
| Block Tree 构建 | BFS 广度优先遍历 DOM（对标论文 Algorithm 1） |
| 粒度控制 | `max_node_words`（最大词数）+ `min_node_words`（最小词数），中文按字符计 |
| 裸文本块 | 节点被拆分时，直接附属文本单独成块，避免信息丢失 |
| Heading 拆分 | 按 H1~H6 层级切为独立内容块（`_find_heading_parent` 穿透 wrapper） |
| 表格按行切分 | 每个块以表头开头，长行强制独立成块 |
| 混合内容分离 | 文本与表格分别提取为不同文档块 |
| UI 噪声过滤 | 进度条、导航文本（`0%`/`PROGRESS`/`目录` 等）不进入文档块 |

**输出格式**（每个文档块）：
```json
{
  "chunk_idx": 0,
  "page_name": "页面名称",
  "title": "段落标题",
  "page_url": "来源路径",
  "text": "纯文本内容",
  "html_content": "<div><p>保留 HTML 结构（论文核心：HTML 优于纯文本）</p></div>",
  "block_path": "html>body>div0>p",
  "summary": "LLM 生成的一句话摘要",
  "question": "代表性用户问题",
  "time": "文档时间戳"
}
```

> 💡 **中文适配要点**：`jieba_util.py` 会从 HTML 语料中提取高频短语（如"巨量千川""广告限流"）构建领域词典，显著提升分词与关键词检索的准确率；分块统计词数时按**字符**计数（`zh_char=True`）而非空格分词。

> 💡 **去重要点**：`deduplicate_ranked_blocks_pal` 用 TF-IDF + cosine 相似度识别重复块，再用**连通分量 BFS**（而非递归 DFS，避免大集群栈溢出）聚簇，簇内按 `time` 保留最新版本。

---

### 🤖 model — 模型推理与训练

#### 推理框架选型：TEI

经过对 TEI、vLLM、Infinity、Xinference 四个框架对比，最终选定 **[TEI (Text Embeddings Inference)](https://github.com/huggingface/text-embeddings-inference)**：

| 优势 | 说明 |
|------|------|
| ✅ 明确支持 Qwen3 | 官方支持 Qwen3-Embedding 与 Qwen3-Reranker |
| ✅ 双 API 一体化 | 同时提供 `/embed`（嵌入）与 `/rerank`（重排序） |
| ✅ HuggingFace 官方维护 | Apache-2.0，生产就绪 |
| ✅ 性能最优 | Flash Attention + 动态批处理，吞吐远超 sentence-transformers |
| ✅ 部署简单 | 轻量 Docker 镜像，无需图编译，启动快 |

#### 模型选型

Qwen3-Embedding / Qwen3-Reranker 系列提供 **0.6B / 4B / 8B** 三个尺寸，8B 版本在 MTEB 多语言榜单排名第一（70.58 分）。选型参考：

| 模型 | 用途 | 参数量 | 嵌入维度 | MTEB 多语言 | 推荐场景 |
|------|------|--------|---------|------------|---------|
| **Qwen3-Embedding-4B** | 文本嵌入 | 4B | 2560 | **69.45** | **生产推荐（精度/速度平衡）** |
| **Qwen3-Reranker-4B** | 重排序 | 4B | — | — | **生产推荐（精度/速度平衡）** |

#### 模型微调策略（两阶段）

```
Stage 1: SFT 监督微调（reranker_ft.py）
  📊 数据: reranker_qa_pointwise.jsonl（多级标注 0/1/2）
  🎯 目标: 学习绝对相关性分数
         │
         ▼
Stage 2: DPO 偏好优化（reranker_dpo.py）
  📊 数据: reranker_qa_dpo.jsonl（chosen vs rejected 偏好对）
  🎯 目标: 在 SFT 基础上学习相对排序偏好
```

| 策略 | Loss 函数 | 优势 |
|------|---------|------|
| SFT | BCE / MSE | 简单稳定，学习绝对分数 |
| DPO | DPO Loss | 偏好对齐好，无需显式奖励模型 |

#### 数据集构造改进

| 改进点 | 说明 |
|--------|------|
| 多级相关性标注 | LLM 对每个文档打 0/1/2 三级分数 |
| 困难负样本 | 同页面不同段落，主题相近但内容不同 |
| 随机负样本 | 明显无关文档，用于基础训练 |
| 三格式输出 | 同时产出 Pointwise / Pairwise / DPO 三种格式 |

---

### 🔍 rag — RAG 检索增强生成

基于 [RAGFlow](https://github.com/infiniflow/ragflow) 构建（🔲 待开发），设计检索链路：

```
用户问题
    │
    ▼
┌──────────────────────────────────────┐
│  Query 重写（多轮对话指代补全）        │
│  "那个怎么办" → "广告限流后怎么办"     │
└──────────┬───────────────────────────┘
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐ ┌─────────┐
│ Milvus  │ │   ES    │
│ 向量检索 │ │ 关键词  │
│ (语义)  │ │ (精确)  │
└────┬────┘ └────┬────┘
     └─────┬─────┘
           ▼
┌──────────────────┐
│  融合去重         │
│  (TF-IDF + 时间)  │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Reranker 精排    │
│  (Qwen3-Reranker) │
└────────┬─────────┘
         ▼
    Top-K 文档块
```

| 能力 | 技术 | 说明 |
|------|------|------|
| 🗄️ 向量检索 | Milvus 2.x + Qwen3-Embedding | 语义理解，"广告投放" 匹配 "推广策略" |
| 🔑 关键词检索 | Elasticsearch 8.x + IK 分词 | 精确匹配，"千川" 匹配 "巨量千川" |
| 🔄 融合去重 | TF-IDF + cosine 相似度 | 消除双模检索的重复结果 |
| 📊 Reranker 精排 | Qwen3-Reranker via TEI | 从 ~20 个候选中精选 Top-5 |

---

### 💬 agent — 智能体

基于 [Pi-Agent](https://github.com/zhipuai/Pi-Agent) 构建（🔲 待开发）：

| 能力 | 说明 |
|------|------|
| 🧠 核心循环 | 感知 → 决策 → 行动 → 观察 |
| 🔧 工具调用 | RAG 检索工具注册与调度，Agent 按需调用知识库 |
| 💬 多轮对话 | 对话历史记忆 + Query 重写（指代补全） |
| 📝 记忆管理 | 短期记忆（对话历史）+ 长期记忆（用户画像） |

---

## 🚀 快速开始

### 1️⃣ 安装依赖

```bash
pip install -r requirements.txt
```

### 2️⃣ 配置环境

```bash
cp config/config.example.json config/config.json
cp .env.example .env
# 编辑 .env：填写 DEEPSEEK_API_KEY、服务地址等
# 编辑 config.json：确认模型配置
```

> **配置优先级**：环境变量 > `.env` 文件 > `config/config.json` > 代码默认值

### 3️⃣ 部署模型服务

```bash
cd model/inference
docker compose -f docker-compose-tei.yml up -d

# 验证服务
curl http://localhost:8080/health   # Embedding 服务
curl http://localhost:8081/health   # Reranker 服务
```

### 4️⃣ 处理 HTML 数据

```bash
# 将 HTML 文件放入 process/dataset/html_source/
bash scripts/process_HTMLdata.sh

# 或使用 main.py 灵活控制（PYTHONPATH 需同时包含 process 与 process/src）
export PYTHONPATH=process:process/src

# 全流程（清洗 + 分块）
python -m main --source-dir process/dataset/html_source

# 仅清洗
python -m main --source-dir process/dataset/html_source --step clean

# 仅分块（需先完成清洗）
python -m main --html-dir process/dataset/html_cleaned --step block

# 使用 vLLM 远程摘要（无需本地加载 ChatGLM）
python -m main --source-dir process/dataset/html_source --use-vllm

# 输出：process/dataset/html_cleaned_block/*.json
```

### 5️⃣ 模型微调（可选）

```bash
# Step 1: 构造训练数据
PYTHONPATH=. python -m model.utils.build_dataset \
    --milvus-host 127.0.0.1 \
    --collection-name htmlrag_dev \
    --output-dir dataset/

# Step 2: SFT 监督微调
PYTHONPATH=. python -m model.trainer.reranker_ft

# Step 3: DPO 偏好优化（在 SFT 基础上）
PYTHONPATH=. python -m model.trainer.reranker_dpo
```

### 6️⃣ 运行测试

```bash
export PYTHONPATH=process
python -m pytest tests/ -v
```

---

## 🛠️ 技术栈

| 组件 | 技术 | 用途 |
|------|------|------|
| 🌐 HTML 解析 | BeautifulSoup4 | HTML 清洗与解析 |
| ✂️ 中文分词 | jieba | 关键词提取与领域词典 |
| 🚀 模型推理 | TEI (Text Embeddings Inference) | Embedding + Reranker 推理服务 |
| 📐 嵌入模型 | Qwen3-Embedding-4B | 文本向量化 |
| 📊 精排模型 | Qwen3-Reranker-4B | 文档重排序 |
| 🗄️ 向量数据库 | Milvus 2.x | 向量存储与 ANN 检索 |
| 🔍 搜索引擎 | Elasticsearch 8.x + IK 分词 | 关键词全文检索 |
| 🔧 RAG 框架 | [RAGFlow](https://github.com/infiniflow/ragflow) | RAG 检索增强生成 |
| 🤖 Agent 框架 | [Pi-Agent](https://github.com/zhipuai/Pi-Agent) | 智能体问答 |
| 🎓 训练框架 | sentence-transformers + 自研 DPO | Reranker 微调与偏好优化 |
| 📊 实验追踪 | Weights & Biases (wandb) | 训练指标可视化 |
| ⚙️ 环境管理 | python-dotenv | 环境变量加载 |
| 🧪 测试框架 | pytest | 单元测试 |

---

## 📋 项目状态

| 模块 | 状态 | 进度 | 说明 |
|------|------|------|------|
| 📦 process/ | ✅ 完成 | 100% | 对标 HtmlRAG 论文，单元测试覆盖 |
| 🤖 model/ | ✅ 基础完成 | 80% | TEI 客户端 + SFT/DPO 训练代码（待实际部署验证） |
| 🔍 rag/ | 🔲 待开发 | 0% | 基于 RAGFlow 集成 |
| 💬 agent/ | 🔲 待开发 | 0% | 基于 Pi-Agent 集成 |
| 🧪 tests/ | 🔄 持续完善 | 60% | process 已覆盖，rag/agent 待补 |
| 📊 评测体系 | 🔲 待设计 | 10% | 已规划四层指标，待实现评测脚本 |

详见 [TODO.md](TODO.md)。

---

## 📄 License

本项目仅供学习和研究使用。引用的开源项目（HtmlRAG、TEI、RAGFlow、Pi-Agent 等）请遵循各自许可证。
