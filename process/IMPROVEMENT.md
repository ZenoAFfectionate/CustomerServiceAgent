# 从分块数据到 RAG 评测 —— 完整指南

> 更新日期：2026-07-11

---

## 〇、数据全景

### 当前数据规模

| 指标 | 数值 |
|------|------|
| 原始 HTML 文件 | 2134 |
| 清洗后 HTML | 2134 |
| 分块 JSON | 2134 |
| 总 chunk 数 | 9081 |
| 平均每文件 chunk | 4.3 |
| chunk text 长度范围 | 4 ~ 67,245 字符 |
| chunk text 中位数 | 350 字符 |
| summary 覆盖率 | **100%**（全部由 vLLM 生成） |
| question 覆盖率 | **0%**（全部为空） |

### 目录结构（8 个一级分类）

```
process/data/
├── 抖音电商规则中心/            ← 原始HTML
├── 抖音电商规则中心_cleaned/    ← 清洗后HTML（保留html结构）
├── 抖音电商规则中心_blocked/    ← 分块JSON
│   ├── 规则总则/                  30+ 文件
│   ├── 精选联盟/
│   │   ├── 抖客推广/              7+ 文件
│   │   ├── 团长/
│   │   └── 基础规范/
│   ├── 营销推广/
│   │   ├── 基础营销规则/
│   │   ├── 营销工具/
│   │   ├── 大促活动专区/
│   │   └── 特色营销业务规则/
│   ├── 发货物流/
│   ├── 行业/                      各行各业规则
│   ├── 创作者管理/
│   ├── 公告专区/
│   ├── 规则动态/
│   └── 特色业务/
```

---

## 一、当前分块数据格式

每个 JSON 文件包含一个数组，每个元素是一个 chunk：

```json
[
  {
    "chunk_idx": 0,
    "page_name": "抖音电商规则总则",
    "title": "第一章 概述",
    "page_url": "process/data_cleaned/抖音电商规则中心/规则总则/抖音电商规则总则.html",
    "summary": "本总则确立了抖音电商生态的安全合规...",
    "question": "",
    "text": "第一章 概述1.1 目的及依据为了保障消费者的合法权益...",
    "html_content": "<div class=\"h1_domain\"><h1>第一章 概述</h1>...",
    "block_path": "h1_domain>h2_domain>p",
    "time": "2025-02-05 16:52:56"
  }
]
```

### 可用于评测构造的关键字段

| 字段 | 评测用途 | 质量评估 |
|------|---------|---------|
| `text` | query 生成的**核心输入**；提取 reference_answer | 覆盖充分 |
| `summary` | query 生成时的浓缩上下文；意图提示 | 100% 覆盖，质量好 |
| `title` | 标注 query 对应的知识章节；难度标签 | 结构清晰 |
| `html_content` | 从 HTML heading 推断知识骨架，用于多 chunk 关联 | 保留层级 |
| `page_name` | Layer 1 评测的 `expected_page_name` | 文件名即文档名 |
| `block_path` | 块在文档中的位置 | 辅助判断 chunk 粒度 |
| 目录层级 | `{一级}/{二级}/page_name` → 天然分类标注 | 人工命名，可靠 |

### ⭐ 关键结论：只需依赖 chunk 文件即可构造完整评测集

**构造评测集只需要 `_blocked/` 下的 chunk JSON 文件，不需要单独读取 HTML 文件。**

原因：**每个 chunk 的 JSON 里已经内嵌了 `html_content` 字段**——HTML 结构信息已经随 chunk 一起存好了。因此：

| 数据 | 是否必需 | 说明 |
|------|---------|------|
| **chunk 文件**（`_blocked/*.json`） | ✅ **必需（唯一输入）** | 含 `text` / `summary` / `title` / `html_content` / 目录路径，构造评测所需信息全部齐备 |
| **清洗 HTML**（`_cleaned/*.html`） | ❌ **不需要** | 其结构信息已内嵌进每个 chunk 的 `html_content` |
| **原始 HTML**（未处理） | ❌ **不需要** | 仅归档用途 |

**关于"整篇文档骨架"**：单个 chunk 的 `html_content` 只含它自己那一段。若做跨章节的高级 query（多跳/综合推理）需要完整文档骨架，**不必去读 HTML 文件**——因为 blocked JSON 本身就是**按文档分组**的（一个 JSON = 一篇文档的所有 chunk），把同一个 JSON 文件内的所有 chunk 聚合起来（拼接它们的 `html_content` / `block_path`）即可重建完整骨架。

> 一句话：**整个评测集构造流程，输入端只有 chunk JSON 一种文件。**

---

## 二、现有评测体系分析

项目已有**两套**评测路径：

### 2.1 标准评测路径（`rag/evaluation/`）

评测用例格式：

```json
[
  {
    "query": "广告限流规则是什么？",
    "relevant_ids": [0, 3, 7],
    "reference_answer": "广告限流是一种风控手段..."
  }
]
```

| 字段 | 必需 | 说明 |
|------|------|------|
| `query` | ✅ | 用户查询文本 |
| `relevant_ids` | 检索评测必需 | 对应的 `global_chunk_idx`（索引时分配） |
| `reference_answer` | 生成评测可选 | 参考答案，用于 lexical_f1 |

评测指标：Recall@K / Precision@K / MRR / NDCG@K / citation_coverage / groundedness / lexical_f1

**核心矛盾**：`relevant_ids` 是入库时动态分配的 `global_chunk_idx`，分块阶段无法预知。

### 2.2 自定义评测路径（`tests/experiment/run_eval.py` / `dataset/preprocess.py`）

评测用例格式（不依赖 `global_chunk_idx`）：

```json
[
  {
    "id": "basic_ORDER_track_order",
    "type": "basic",
    "query": "Where is my order?",
    "expected_category": "ORDER",
    "expected_intent": "track_order",
    "reference_answer": "Your order is on the way..."
  }
]
```

匹配逻辑：检查检索到的上下文块的 `category`/`intent` 标签是否与预期一致。

---

## 三、推荐方案：双层评测体系

### 3.1 设计思路

```
Layer 1: 语义标签评测（入库前即可用）
    ├── 不需要预知 global_chunk_idx
    ├── 基于 page_name / block_path / 目录层级匹配
    └── 评测"检索是否找到了正确的知识文档"

Layer 2: 精确块级评测（入库后可做）
    ├── 先入库 → 获取 global_chunk_idx 映射
    ├── 再通过 LLM 标注每个 query 应命中的具体块
    └── 评测标准 Recall@K / MRR / NDCG
```

### 3.2 推荐的目标格式

```json
[
  {
    "id": "eval_0001",
    "type": "basic",
    "query": "商家入驻抖音电商需要满足哪些资质要求？",
    "expected_page_name": "抖音电商规则总则",
    "expected_dir": "规则总则",
    "expected_title_contains": ["招商入驻", "资质"],
    "relevant_chunk_keys": [
      {"page_name": "抖音电商规则总则", "chunk_idx": 0}
    ],
    "relevant_ids": null,
    "reference_answer": "商家须满足《招商标准及入驻规范》...",
    "source_chunk_text": "4.1 招商入驻...",
    "category": "规则总则",
    "subcategory": "商家规定/招商入驻",
    "difficulty": "easy",
    "query_type": "direct",
    "generation_metadata": {
      "model": "Qwen3.5-2B",
      "prompt_version": "v1",
      "confidence": "high"
    }
  }
]
```

| 字段 | 用途 | 何时可用 |
|------|------|---------|
| `expected_page_name` / `expected_dir` | Layer 1 评测 | 分块阶段即可 |
| `relevant_chunk_keys` | `(page_name, chunk_idx)` 元组唯一标识块 | 分块阶段即可 |
| `relevant_ids` | 标准 Recall@K/MRR 评测（入库后回填） | 入库后 |
| `reference_answer` / `source_chunk_text` | 生成评测 + 可审计性 | 分块阶段即可 |
| `generation_metadata` | 追溯每条用例的生成来源 | 生成时记录 |

---

## 四、LLM 驱动的评测数据集构造方案（核心）

### 4.1 总体架构

整个方案分为 **5 个阶段**，其中 3 个阶段需要 LLM 参与：

```
                               Phase 1: 数据分析与采样（无 LLM）
                              ┌──────────────────────────────┐
  blocked JSON（唯一输入）    │  chunk扫描 → 统计分布        │
  ──────────────────────────▶ │  按策略采样 200-500 个chunk  │
  （html_content 已内嵌，     │  提取目录/标题/HTML层级      │
    无需单独读 HTML 文件）    └──────────┬───────────────────┘
                                         │ 采样后的 chunk 列表
                                         ▼
                               Phase 2: Query 生成（LLM 核心）
                              ┌──────────────────────────────┐
                               │  对每个采样chunk调用LLM     │
                               │  输入: text+summary+title   │
                               │  输出: 多条query+answer     │
                              └──────────┬───────────────────┘
                                         │ 原始 query 池 (~600-1500条)
                                         ▼
                               Phase 3: 质量过滤与去重（LLM辅助）
                              ┌──────────────────────────────┐
                               │  LLM自检 + embedding去重    │
                               │  难度标注 + 类型分类        │
                              └──────────┬───────────────────┘
                                         │ 清洗后的用例 (~400-800条)
                                         ▼
                               Phase 4: Hard Negative 构造（LLM）
                              ┌──────────────────────────────┐
                               │  生成易混淆的相似query      │
                               │  生成跨文档综合推理query    │
                              └──────────┬───────────────────┘
                                         │
                                         ▼
                               Phase 5: 组装 + 入库 + ID回填
                              ┌──────────────────────────────┐
                               │  导出为标准评测格式         │
                               │  入库 → 检索 → 回填ID       │
                               │  人工抽检 50 条 Gold Set    │
                              └──────────────────────────────┘
```

### 4.2 Phase 1: 智能采样策略

目标：从 9081 个 chunk 中采样 **200-500 个代表性子集**，覆盖知识面又控制 LLM 调用成本。

#### 4.2.1 分层采样（Stratified Sampling）

```
采样配额 = 500 chunks 总预算

一级分类（8个）× 每类 15-30 chunks = 120-240
    ├── 规则总则: 30 chunks（核心知识，多采样）
    ├── 精选联盟: 25 chunks
    ├── 营销推广: 30 chunks
    ├── 发货物流: 25 chunks
    ├── 行业:     30 chunks
    ├── 创作者管理: 25 chunks
    ├── 公告专区:  15 chunks
    ├── 规则动态:  15 chunks
    └── 特色业务:  15 chunks

二级分类内（每类的子目录均匀覆盖）:
    每个二级分类至少 3 chunks，最多 15 chunks

chunk 粒度补充采样: 60 chunks
    ├── 极小chunk (<100字): 15个 → 测试短文本检索
    ├── 中型chunk (500-2000字): 25个 → 主力测试
    └── 大chunk (>5000字): 20个 → 测试长文本理解

表格类 chunk 专项: 40 chunks
    表格按行切分后每行是一个独立 chunk，需要额外覆盖
```

#### 4.2.2 采样优先级打分

对每个 chunk 计算一个"采样价值分"，优先取高分：

```python
def chunk_sampling_score(chunk, dir_path):
    score = 0.0

    # 1. 内容质量（summary 非空 + text 长度适中）
    if 100 < len(chunk["text"]) < 4000:
        score += 2.0
    elif len(chunk["text"]) >= 4000:
        score += 1.0  # 大 chunk 有价值但 LLM 处理成本高
    else:
        score += 0.5  # 过短，信息量有限

    # 2. 标题信息丰富度（含具体主题词）
    title = chunk.get("title", "")
    if len(title) > 8 and any(kw in title for kw in
        ["规则", "规范", "管理", "标准", "流程", "条件", "要求"]):
        score += 1.5

    # 3. HTML 结构完整性（heading 层级保留得好 → 知识结构化好）
    if "h1_domain" in chunk.get("html_content", ""):
        score += 1.0
    if "h2_domain" in chunk.get("html_content", ""):
        score += 1.0

    # 4. 目录深度（深层目录通常承载更具体的知识）
    depth = dir_path.count(os.sep)
    if depth >= 2:
        score += 1.0

    return score
```

#### 4.2.3 HTML 结构辅助采样

利用 `html_content` 中的 heading 标签识别**知识骨架**：

```python
def extract_knowledge_skeleton(html_content):
    """
    从 HTML 中提取 heading 层级结构作为知识骨架。

    输入: <div class="h1_domain"><h1>商家规定</h1>
           <div class="h2_domain"><h2>招商入驻</h2>...
    输出: ["h1: 商家规定", "h2: 招商入驻", "h3: 资质要求"]
    """
    soup = BeautifulSoup(html_content, "html.parser")
    skeleton = []
    for tag in soup.find_all(["h1", "h2", "h3", "h4"]):
        skeleton.append(f"{tag.name}: {tag.get_text(strip=True)[:80]}")
    return skeleton
```

若多个 chunk 来自同一文档的不同 section（不同 heading），优先采样**每个 section 的首个 chunk**（通常包含该节的概述）。这样确保评测覆盖"跨 section 查询"场景。

### 4.3 Phase 2: LLM Query 生成（核心）

#### 4.3.1 两阶段 Prompt 设计

**为什么需要两阶段？**
- 一阶段直接输出可能产生格式不规范的 JSON，或者 query 质量不一致
- 两阶段将"创意生成"与"格式化+质量检查"分离，每个阶段任务简单且确定性高

##### Stage 1: Query 发散生成（Creativity）

```
╔══════════════════════════════════════════════════════════════╗
║                   SYSTEM PROMPT (Stage 1)                    ║
╠══════════════════════════════════════════════════════════════╣
║ 你是一个 RAG 评测数据集构建专家。你的任务是根据知识库中的   ║
║ 具体内容，模拟真实用户可能会提出的问题。                     ║
║                                                              ║
║ ## 核心原则                                                   ║
║ 1. 问题必须基于提供的知识内容，不可凭空编造                 ║
║ 2. 模拟真实用户的语言风格（口语化、不完整句、带错别字也可） ║
║ 3. 问题应该是知识能回答的（答案可从给定内容中推导或提取）   ║
║ 4. 多样化提问角度：不同用户会有不同的表达方式               ║
║                                                              ║
║ ## 问题类型分布                                               ║
║ - 直接查询型 (50%): 直接问"XX是什么""XX怎么操作"           ║
║ - 场景应用型 (30%): "我是商家，遇到XX情况该怎么办"         ║
║ - 对比推理型 (15%): "A和B有什么区别""XX适用于YY吗"         ║
║ - 反事实/边界型 (5%): "如果不满足XX条件会怎样"             ║
╚══════════════════════════════════════════════════════════════╝

USER INPUT:
---
知识来源文档: {page_name}
所属分类: {一级分类} > {二级分类}
章节标题: {title}
知识摘要: {summary}
知识正文（关键段落）:
{text[:1500]}  ← 截断到 1500 字符，控制 token 成本
---

请生成 3-5 个不同角度、不同难度的问题。要求：
- 每个问题独立一行
- 问题后标注 [难度: easy/medium/hard]
- easy: 问题关键词直接出现在正文中
- medium: 需要理解后概括回答
- hard: 需要综合多方信息或推理
```

##### Stage 2: 格式化 + 答案提取（Precision）

```
╔══════════════════════════════════════════════════════════════╗
║                   SYSTEM PROMPT (Stage 2)                    ║
╠══════════════════════════════════════════════════════════════╣
║ 你需要将上一阶段生成的问题格式化，并从知识正文中提取精确的   ║
║ 参考答案。同时做一轮质量自检。                               ║
║                                                              ║
║ ## 输出要求（严格 JSON 数组）                                ║
║ [                                                            ║
║   {                                                          ║
║     "query": "用户问题原文",                                  ║
║     "difficulty": "easy|medium|hard",                        ║
║     "query_type": "direct|scenario|comparison|boundary",     ║
║     "reference_answer": "基于知识正文的标准答案（100字内）", ║
║     "answerable": true/false,                                ║
║     "answer_source": "direct|inferred|partial",              ║
║     "keywords": ["关键词1", "关键词2"],                      ║
║     "quality_check": {                                       ║
║       "too_vague": false,                                    ║
║       "hallucinated": false,                                 ║
║       "duplicate_same_chunk": false                          ║
║     }                                                        ║
║   }                                                          ║
║ ]                                                            ║
║                                                              ║
║ ## 质量自检规则                                              ║
║ - too_vague: 问题是否过于模糊（如"怎么样""好不好"无上下文） ║
║ - hallucinated: 问题是否引用了正文中不存在的信息             ║
║ - answerable: 仅从给定正文是否能给出完整答案                 ║
╚══════════════════════════════════════════════════════════════╝

USER INPUT:
知识正文: {text[:2000]}
上一阶段生成的问题:
{stage1_questions}

请格式化以上问题并提取答案。
```

#### 4.3.2 多样性增强技术

为避免 LLM 生成的问题过于相似（模板化），在 Stage 1 中使用**多样性指令随机注入**：

```python
DIVERSITY_INSTRUCTIONS = [
    "请用口语化、带语气的风格提问（如'我想问下...'）",
    "请模拟一个焦急的商家提问（带紧迫感）",
    "请模拟一个新手商家（对平台规则不熟悉）",
    "请用简洁的关键词风格提问（如搜索引擎输入）",
    "请模拟一个遇到具体纠纷场景的商家",
    "请从消费者角度提问（而非商家角度）",
]

# 每个 chunk 随机选一条注入
import random
style_hint = random.choice(DIVERSITY_INSTRUCTIONS)
prompt = BASE_PROMPT + f"\n\n【风格要求】{style_hint}"
```

#### 4.3.3 HTML 结构增强的 Query 生成

当 chunk 来自一个大型文档（包含多个 section）时，额外利用 HTML 结构信息：

```
额外上下文:
本文档的完整章节结构:
  h1: 商家管理规定
    h2: 招商入驻
      h3: 资质要求  ← 当前 chunk 所在位置
      h3: 资费标准
      h3: 店铺类型
    h2: 商品管理
    h2: 履约规范

请基于以上知识骨架，额外生成 2 个跨章节的问题：
- 这些问题需要综合当前 chunk 的知识与相邻章节的知识
- 例如："商家入驻需要哪些资质？不同店铺类型的资费标准一样吗？"
```

这样从 **HTML 结构 + chunk 内容** 两个维度出发，LLM 可以生成更自然的综合查询。

### 4.4 Phase 3: 质量过滤与去重

#### 4.4.1 自动过滤 Pipeline

```python
def filter_eval_cases(raw_cases, strict=True):
    """多阶段过滤原始 LLM 生成的评测用例"""
    filtered = raw_cases

    # Stage 1: 格式校验
    filtered = [c for c in filtered if all(k in c for k in
        ["query", "difficulty", "reference_answer", "answerable"])]

    # Stage 2: 内容质量过滤
    filtered = [
        c for c in filtered
        if len(c["query"]) >= 5                          # 过短
        and len(c["query"]) <= 200                       # 过长
        and not c.get("quality_check", {}).get("hallucinated", False)
        and not c.get("quality_check", {}).get("too_vague", False)
    ]

    # Stage 3: query 文本去重（embedding 相似度）
    filtered = _dedup_by_embedding_similarity(filtered, threshold=0.85)

    # Stage 4: 答案可答性二次校验（LLM 自检）
    # 对标注为 answerable=false 的丢掉，对 uncertain 的二次确认

    return filtered
```

#### 4.4.2 Embedding 去重细节

##### 为什么需要 embedding 去重？

LLM 生成 query 时会产生大量**"字面不同、意思相同"的重复**。例如针对同一个"退款规则" chunk，不同批次可能生成：

```
"怎么退款？"
"如何申请退款？"
"退款流程是什么？"
"我想退款该怎么操作？"
```

这 4 句**字符串完全不同**，但语义几乎一模一样。若不去重，评测集会虚胖，且某些知识点被反复测、另一些没测到，导致指标失真。

##### 为什么传统"字符串去重"不够用？

传统的字符相似度去重（如 `difflib`）看的是**字面重合度**：

```
"怎么退款"  vs  "如何申请退款"
     ↓
字面只共享"退款"二字，字符相似度很低 → 被误判为"不重复" ❌
```

它抓不住"语义相同但用词不同"的情况。

##### embedding 去重的原理（3 步）

**核心思想：把每句 query 转成一个"语义向量"，比较向量距离，而不是比较文字。**

```
第1步：文本 → 向量（embedding）
    用 embedding 模型（项目里是 Qwen3-Embedding-4B）把每句话编码成
    一个高维向量（如 1024 维），向量代表"句子的含义"。
    意思越接近的句子，向量在空间里越接近——哪怕用词完全不同。

    "怎么退款？"        → [0.21, -0.05, 0.88, ...]
    "如何申请退款？"    → [0.19, -0.03, 0.91, ...]  ← 数值非常接近
    "大促活动怎么报名？" → [0.77,  0.40, 0.10, ...]  ← 数值差很远

第2步：算余弦相似度（cosine similarity）
    比较两个向量夹角的余弦值，范围 0~1：
      相似度 = 1.0 → 语义完全相同
      相似度 = 0.9 → 语义高度相似（大概率重复）
      相似度 = 0.3 → 语义无关

    "怎么退款" vs "如何申请退款"  → cos ≈ 0.92  ← 判为重复 ✅
    "怎么退款" vs "大促怎么报名"  → cos ≈ 0.25  ← 保留（不同问题）

第3步：设阈值，超过就丢弃（threshold=0.85）
    遍历每条新 query：
        若它与"已保留的某条 query"相似度 ≥ 0.85 → 判为重复，丢弃
        否则 → 保留为新的独立 query
```

> 一句话：**embedding 去重 = 用 AI 理解"意思"来去重，而不是死抠字面。**
> 它能把"怎么退款"和"如何申请退款"识别为同一个问题去掉一个，
> 让评测集里每条 query 都是真正独立的知识点。

##### 阈值调节

`threshold=0.85` 是可调的：

| 阈值 | 效果 | 适用场景 |
|------|------|---------|
| 调高（如 0.92） | 去重更宽松，保留更多"稍有差异"的 query | 想要更大更多样的评测集 |
| 默认（0.85） | 平衡 | 推荐起点 |
| 调低（如 0.75） | 去重更激进，只保留差异明显的 query | 想要精简、每条差异明显 |

建议先跑一批看实际效果再定阈值。

##### 参考实现

```python
def _dedup_by_embedding_similarity(cases, threshold=0.85):
    """
    使用本地 embedder 计算 query 之间的余弦相似度去重。
    注意：用文本 embedding 而非字符相似度——"怎么退款"和"如何申请退款"
    应该被识别为重复。
    """
    from rag.indexing.embedding import get_embedder
    embedder = get_embedder()
    queries = [c["query"] for c in cases]
    embeddings = embedder.embed_texts(queries)

    keep_idx = []
    keep_embeddings = []
    for i, emb in enumerate(embeddings):
        if not keep_embeddings:            # 第一条无条件保留
            keep_idx.append(i)
            keep_embeddings.append(emb)
            continue
        # 和所有"已保留"的 query 算相似度，取最大值
        sims = cosine_similarity([emb], keep_embeddings)[0]
        if max(sims) < threshold:          # 与所有已保留的都不够像 → 是新问题
            keep_idx.append(i)
            keep_embeddings.append(emb)
        # 否则：与某条已保留的高度相似 → 丢弃这条重复
    return [cases[i] for i in keep_idx]
```

> **复用提示**：项目里 `rag/evaluation/` 与 `text_process.py` 已经用到
> `sklearn.metrics.pairwise.cosine_similarity` 和 `get_embedder()`，
> 这里直接复用同一套 embedding 后端，无需额外引入依赖。

#### 4.4.3 LLM 自检 Prompt（用于 Stage 4）

```
你是评测数据集质量审核员。请检查以下(query, reference_answer, source_text)三元组：

Query: {query}
参考答案: {reference_answer}
源文本: {source_chunk_text}

请判断：
1. 参考答案是否确实可以从源文本中推导？（yes/no/partial）
2. Query 是否明确指向了源文本中的特定信息？（yes/no）
3. 是否存在"答案过于通用、缺乏具体信息"的问题？（yes/no）

输出JSON: {"derivable": "yes/no/partial", "specific": "yes/no", "too_generic": "yes/no"}
```

### 4.5 Phase 4: Hard Negative & 高级用例构造

这是 LLM 方案的核心增值点——不仅生成基础评测用例，还用 LLM 构造**高难度、高区分度**的用例。

#### 4.5.1 混淆 Query 生成（Hard Negative）

策略：基于正确 chunk A 的标题/主题，让 LLM 生成一个看起像在问 A 但实际在问 B 的 query。

```
你有两个知识块：
  块A（正确目标）: {chunk_A_title}: {chunk_A_summary}
  块B（干扰项）:   {chunk_B_title}: {chunk_B_summary}

请生成 2 个用户查询，要求：
- 查询的关键词与块B的标题高度重叠（让向量检索容易被B"骗走"）
- 但语义上查询真正想问的是块A的内容
- 这种查询测试的是检索系统能否"透过关键词表面理解真实意图"

示例：
  块A: "抖客API服务商准入及考核管理规则"（关于API接入的技术规范）
  块B: "抖客推广准入及准出管理规则"（关于抖客个人推广者的准入）
  生成query: "抖客API接入后，推广者还需要满足什么条件才能开始推广？"
  （关键词"抖客""API""推广"容易命中B，但真实需求是A中的技术准入）
```

#### 4.5.2 跨文档综合 Query（Multi-hop）

从同一一级分类的**不同文档**中各取 1 个 chunk，让 LLM 生成需要综合两个 chunk 的问题：

```
你有来自同一分类的两个知识块：
  块A [{dir_A}]: {chunk_A_content[:1000]}
  块B [{dir_B}]: {chunk_B_content[:1000]}

请生成一个需要综合两块知识才能完整回答的问题。
这个问题的答案必须分散在两块中，检索系统需要同时召回两者。

要求：question 的表述应该自然，不能暴露"需要查两个文档"的提示。
```

#### 4.5.3 多轮对话场景

从同一文档的不同 section 提取 2-3 个 chunk，构造连续追问：

```
从同一文档 [{page_name}] 的不同章节各取一段：
  第1轮 chunk: {section1_chunk}
  第2轮 chunk: {section2_chunk}

构造一个多轮对话评测用例：
- Turn 1: 用户问关于 section1 的问题
- Turn 2: 用户在得到回答后，追问关于 section2 的深入问题
- 追问应该自然地衔接上一轮的答案

输出：
{
  "id": "multiturn_{page_name}",
  "type": "multiturn",
  "turns": [
    {"query": "...", "expected_page_name": "...", "expected_chunk_key": {...}},
    {"query": "...", "expected_page_name": "...", "expected_chunk_key": {...}}
  ],
  "context_requirement": "turn2的query需要依赖turn1的上下文才能正确理解"
}
```

### 4.6 Phase 5: 组装与评测就绪

#### 4.6.1 Case 难度分层统计

| 难度 | 定义 | 占比目标 | 示例 |
|------|------|---------|------|
| **easy** | 单关键词直接命中，答案在单个 chunk 中 | 40% | "保证金是什么" |
| **medium** | 需要理解概括，可能跨 2-3 个 chunk | 40% | "入驻需要哪些资质" |
| **hard** | 需要多文档综合或推理 | 15% | "A 和 B 的准入条件有何异同" |
| **adversarial** | 故意使用易混淆的关键词 | 5% | 混淆 query、反事实场景 |

#### 4.6.2 输出格式集成

最终的 `eval_cases.json` 需要兼容项目现有的两套评测代码：

```python
def export_eval_cases(cases, format="universal"):
    """
    导出为不同格式：
    - "universal": 包含所有字段的完整格式（推荐存储格式）
    - "standard":  rag/evaluation/rag_e2e_eval.py 兼容格式
    - "custom":    tests/experiment/run_eval.py 兼容格式
    """
    if format == "standard":
        return [{
            "query": c["query"],
            "relevant_ids": c.get("relevant_ids", []),
            "reference_answer": c.get("reference_answer", ""),
        } for c in cases]
    if format == "custom":
        return [{
            "id": c["id"],
            "type": c["type"],
            "query": c["query"],
            "expected_category": c["category"],
            "expected_intent": c.get("subcategory", ""),
            "reference_answer": c.get("reference_answer", ""),
        } for c in cases]
    return cases  # universal
```

---

## 五、完整实现代码设计

### 5.1 文件结构

```
process/
├── eval/
│   ├── __init__.py                    ← 包初始化
│   ├── build_eval_dataset.py          ← 主入口脚本
│   ├── sampler.py                     ← Phase 1: 智能采样
│   ├── llm_query_generator.py         ← Phase 2: LLM query生成
│   ├── quality_filter.py              ← Phase 3: 质量过滤与去重
│   ├── hard_negative_builder.py       ← Phase 4: 高级用例构造
│   ├── eval_exporter.py              ← Phase 5: 格式导出
│   ├── prompts.py                     ← 所有 LLM prompt 模板
│   └── README.md                      ← 使用说明
```

### 5.2 核心类与函数签名

```python
# --- sampler.py ---

@dataclass
class SampledChunk:
    """采样后的 chunk 元数据"""
    chunk: dict                        # 原始 chunk 数据
    rel_path: str                      # 相对于 blocked_dir 的路径
    category: str                      # 一级分类
    subcategory: str                   # 二级分类
    sampling_score: float              # 采样优先级分
    knowledge_skeleton: list           # HTML 结构骨架

def sample_chunks(
    blocked_dir: str,
    cleaned_dir: str = None,           # 可选的清洗HTML目录，用于提取结构
    total_budget: int = 500,
    random_seed: int = 42,
) -> List[SampledChunk]:
    """分层采样 500 个 chunk，覆盖全部知识类别"""


# --- llm_query_generator.py ---

@dataclass
class GeneratedCase:
    """LLM 生成的一条评测用例"""
    query: str
    difficulty: str
    query_type: str
    reference_answer: str
    source_chunk: SampledChunk
    quality_check: dict
    generation_metadata: dict

def generate_queries_batch(
    chunks: List[SampledChunk],
    llm_api_url: str = "http://localhost:8011/v1/chat/completions",
    model: str = "Qwen/Qwen3.5-2B",
    queries_per_chunk: int = 3,
    batch_size: int = 8,
    use_html_structure: bool = True,
    diversity_injection: bool = True,
) -> List[GeneratedCase]:
    """
    批量调用 vLLM 为每个采样 chunk 生成评测 query。

    两阶段流程：
    1. 发散生成 → 3-5 个自然语言问题
    2. 格式化 + 答案提取 + 质量自检

    并发控制：每批 8 个 chunk 并行调用（避免打爆 vLLM）


def generate_queries_single(
    chunk: SampledChunk,
    llm_api_url: str,
    model: str,
    n_queries: int = 3,
) -> List[GeneratedCase]:
    """单 chunk 的完整两阶段 query 生成"""


# --- quality_filter.py ---

def filter_and_dedup(
    cases: List[GeneratedCase],
    embedding_threshold: float = 0.85,
    llm_self_check: bool = True,
    llm_api_url: str = None,
) -> List[GeneratedCase]:
    """多阶段质量过滤 + embedding去重 + LLM自检"""


# --- hard_negative_builder.py ---

def build_confusion_queries(
    cases: List[GeneratedCase],
    all_chunks_index: dict,             # (page_name, chunk_idx) → chunk
    n_confusion_per_category: int = 5,
    llm_api_url: str = None,
) -> List[GeneratedCase]:
    """构造混淆query用例"""

def build_multihop_queries(
    cases: List[GeneratedCase],
    all_chunks_index: dict,
    n_multihop: int = 20,
    llm_api_url: str = None,
) -> List[GeneratedCase]:
    """构造跨文档综合推理用例"""

def build_multiturn_dialogues(
    cases: List[GeneratedCase],
    all_chunks_index: dict,
    n_multiturn: int = 10,
    llm_api_url: str = None,
) -> List[GeneratedCase]:
    """构造多轮对话用例"""


# --- eval_exporter.py ---

def export_eval_cases(
    cases: List[GeneratedCase],
    output_path: str,
    format: str = "universal",
) -> str:
    """导出评测用例到文件"""

def resolve_global_ids(
    cases: List[GeneratedCase],
    blocked_dir: str,
) -> List[GeneratedCase]:
    """
    入库后回填 global_chunk_idx。
    流程：将 blocked JSON 全部导入索引 → 通过检索找到每个 query 对应的
          global_chunk_idx → 回填到 relevant_ids
    """


# --- prompts.py ---

STAGE1_QUERY_GENERATION = """..."""     # Phase 2 Stage 1 prompt
STAGE2_FORMAT_AND_ANSWER = """..."""    # Phase 2 Stage 2 prompt
SELF_CHECK_QUALITY = """..."""          # Phase 3 LLM 自检 prompt
CONFUSION_QUERY = """..."""             # Phase 4 混淆query prompt
MULTIHOP_QUERY = """..."""              # Phase 4 跨文档综合 prompt
MULTITURN_DIALOGUE = """..."""          # Phase 4 多轮对话 prompt
```

### 5.3 主入口脚本使用方式

```bash
# 完整流程（需要 vLLM 服务在运行）
PYTHONPATH=process:process/src python -m eval.build_eval_dataset \
    --blocked-dir process/data/抖音电商规则中心_blocked \
    --cleaned-dir process/data/抖音电商规则中心_cleaned \
    --output process/data/eval_cases.json \
    --sample-budget 500 \
    --queries-per-chunk 3 \
    --vllm-url http://localhost:8011/v1/chat/completions \
    --with-hard-negatives \
    --with-multihop \
    --with-multiturn \
    --dry-run-first        # 先 dry-run 10 个 chunk 看质量

# 仅采样 + 预览（不调用 LLM）
PYTHONPATH=process:process/src python -m eval.build_eval_dataset \
    --blocked-dir process/data/抖音电商规则中心_blocked \
    --sample-only \
    --sample-budget 20     # 少量快速验证采样策略

# Phase 5: ID 回填（入库后）
PYTHONPATH=process:process/src python -m eval.build_eval_dataset \
    --eval-path process/data/eval_cases.json \
    --blocked-dir process/data/抖音电商规则中心_blocked \
    --resolve-ids-only

# 人工审核模式：输出 Markdown 方便人工检查
PYTHONPATH=process:process/src python -m eval.build_eval_dataset \
    --eval-path process/data/eval_cases.json \
    --export-review-md process/data/eval_cases_review.md
```

---

## 六、Token 成本估算

### 6.1 成本分解

| 阶段 | LLM调用次数 | 每次输入 tokens | 每次输出 tokens | 总 tokens (估算) |
|------|-----------|----------------|----------------|-----------------|
| Phase 2 Stage 1 | 500 chunks | ~800 (text+summary+prompt) | ~200 | ~500K |
| Phase 2 Stage 2 | 500 chunks | ~1200 (text+stage1结果) | ~300 | ~750K |
| Phase 3 LLM自检 | ~300 (answerable=yes) | ~600 | ~100 | ~210K |
| Phase 4 混淆query | ~50 pairs | ~800 | ~200 | ~50K |
| Phase 4 跨文档 | ~20 pairs | ~1500 | ~300 | ~36K |
| Phase 4 多轮 | ~10 groups | ~2000 | ~400 | ~24K |
| **合计** | | | | **~1.6M tokens** |

### 6.2 成本优化

使用本地 vLLM (Qwen3.5-2B) → **零 API 费用**，仅 GPU 电费。

如果希望更快的生成速度：
- 将 `sample_budget` 降到 200（~640K tokens，约 10 分钟）
- 或使用 `--queries-per-chunk 2`（减少 33% 输出 tokens）

---

## 七、质量保障机制

### 7.1 自动校验

| 校验项 | 方法 | 时机 |
|--------|------|------|
| JSON 格式正确性 | `json.loads` | Phase 2 输出后 |
| query 非空 + 长度合理 | 规则检查 | Phase 3 |
| reference_answer 可回溯 | source_chunk_text 保留 | 全程 |
| 无信息幻觉 | LLM Stage 2 自检 `hallucinated` 字段 | Phase 2 |
| query 多样性 | embedding 相似度去重 | Phase 3 |
| 答案可答性 | LLM 自检 `answerable` 字段 | Phase 2 + 3 |
| 知识覆盖度 | 按一级/二级分类统计覆盖率 | Phase 5 |

### 7.2 人工抽检流程

```
从最终 400-800 条用例中随机抽 50 条，人工检查：

检查清单（每条扣0/1分，满分5分）:
□ query 是否自然（不像机器生成）         [1分]
□ query 是否确实指向标注的 chunk         [1分]
□ reference_answer 是否正确              [1分]
□ difficulty 标注是否合理                [1分]
□ 不存在信息幻觉                         [1分]

统计：平均分 < 4.0 → 调整 prompt 后重新生成
       平均分 >= 4.5 → 质量通过
```

### 7.3 Gold Standard 集合

从人工抽检通过的 50 条中选最可靠的 30 条作为 **Gold Set**，用途：
- 每次 prompt 调整后的回归测试（确保新 prompt 质量不低于旧版）
- 不同 embedding/reranker 模型的对比基准
- CI 中的 smoke test（每次代码变更后快速验证检索未退化）

---

## 八、Prompt 迭代优化策略

### 8.1 A/B 测试框架

```python
PROMPT_VARIANTS = {
    "v1_baseline": {  # 基础版
        "stage1": STAGE1_BASELINE,
        "stage2": STAGE2_BASELINE,
    },
    "v2_detailed": {  # 增强详细度
        "stage1": STAGE1_WITH_DETAIL_INSTRUCTION,
        "stage2": STAGE2_BASELINE,
    },
    "v3_roleplay": {  # 角色扮演增强
        "stage1": STAGE1_WITH_ROLEPLAY,
        "stage2": STAGE2_BASELINE,
    },
}

def ab_test_prompts(sample_chunks, variants, n_samples=30):
    """对同一批采样chunk用不同的prompt变体生成，比较质量"""
    results = {}
    for variant_name, prompts in variants.items():
        cases = generate_queries_batch(
            sample_chunks[:n_samples],
            prompt_overrides=prompts,
        )
        results[variant_name] = {
            "n_cases": len(cases),
            "avg_query_length": statistics.mean([len(c["query"]) for c in cases]),
            "unique_ratio": len(set(c["query"] for c in cases)) / len(cases),
            "hallucination_rate": sum(1 for c in cases
                if c.get("quality_check", {}).get("hallucinated"))
                / len(cases),
            "answerable_rate": sum(1 for c in cases
                if c.get("answerable"))
                / len(cases),
        }
    return results
```

### 8.2 问题诊断与改进循环

```
观察 LLM 输出 → 发现模式问题 → 调整 prompt → A/B 测试 → 确定最优版本

常见问题及修复：
  问题: query 太模板化（都像 "XX是什么"）
  修复: 增加 diversity_injection，或提高 Stage 1 的 temperature

  问题: reference_answer 直接从原文复制（太长）
  修复: Stage 2 增加长度限制 + "请用自己的话概括" 的指令

  问题: 大量 hallucinated=true（LLM编造了原文不存在的内容）
  修复: 提升 temperature=0（贪婪解码），在 Stage 1 中也强调"必须基于原文"
```

---

## 九、与现有评测体系的对接

### 9.1 对接 `rag/evaluation/retrieval_eval.py`

```python
from rag.evaluation.retrieval_eval import evaluate_retrieval

# relevant_ids 已回填后
retrieval_report = evaluate_retrieval(eval_cases, top_k=5)
# → {"recall@5": 0.82, "precision@5": 0.65, "mrr": 0.71, "ndcg@5": 0.78}
```

### 9.2 对接 `rag/evaluation/generation_eval.py`

```python
from rag.evaluation.generation_eval import evaluate_generation

# 只要有 reference_answer 即可（不需要 relevant_ids）
generation_report = evaluate_generation(eval_cases, top_k=5)
```

### 9.3 对接 `rag/evaluation/rag_e2e_eval.py`

```python
from rag.evaluation.rag_e2e_eval import run_e2e_eval
report = run_e2e_eval(eval_cases, top_k=5)
```

### 9.4 Layer 1 评测（入库前即可用，不依赖 global_chunk_idx）

```python
def layer1_retrieval_eval(eval_cases, top_k=5):
    """使用 page_name + directory 匹配的快速检索评测"""
    from rag import pipeline

    results = []
    for case in eval_cases:
        contexts = pipeline.retrieve(case["query"], top_k=top_k)

        # 三层匹配（逐层放松）：
        # L1: page_name 精确匹配
        hit_page = any(
            c.get("page_name") == case["expected_page_name"]
            for c in contexts
        )
        # L2: 目录路径前缀匹配（至少选了同一分类下的文档）
        hit_dir = any(
            c.get("page_url", "").startswith(case.get("expected_dir", ""))
            for c in contexts
        )
        # L3: 标题关键词匹配
        keywords = case.get("expected_title_contains", [])
        hit_title = any(
            all(kw in c.get("title", "") for kw in keywords)
            for c in contexts
        ) if keywords else False

        results.append({
            "id": case["id"],
            "query": case["query"],
            "hit_page": hit_page,
            "hit_dir": hit_dir,
            "hit_title": hit_title,
        })

    return {
        "page_accuracy": sum(r["hit_page"] for r in results) / len(results),
        "dir_accuracy": sum(r["hit_dir"] for r in results) / len(results),
        "title_accuracy": sum(r["hit_title"] for r in results) / len(results),
    }
```

---

## 十、最小可行方案（MVP）- 立即可执行

如果不想等待完整脚本，可以立即手动执行：

### 10.1 Step 1: 手动采样 + LLM 生成（1 小时）

```bash
# 从 9081 个 chunk 中随机采样 30 个，手工调用 LLM 生成 query
python3 -c "
import os, json, random

blocked = 'process/data/抖音电商规则中心_blocked'
all_chunks = []
for root, _, files in os.walk(blocked):
    for f in files:
        if f.endswith('.json'):
            with open(os.path.join(root, f)) as fp:
                for c in json.load(fp):
                    c['_file'] = f
                    c['_dir'] = os.path.relpath(root, blocked)
                    all_chunks.append(c)

# 分层采样 30 个
random.seed(42)
samples = random.sample(all_chunks, min(30, len(all_chunks)))

# 输出给 LLM 的 prompt
for i, c in enumerate(samples):
    print(f'--- Sample {i} ---')
    print(f'目录: {c[\"_dir\"]}')
    print(f'文档: {c[\"_file\"]}')
    print(f'标题: {c[\"title\"]}')
    print(f'摘要: {c[\"summary\"][:200]}')
    print(f'正文(前500字): {c[\"text\"][:500]}')
    print()
" > /tmp/eval_samples.txt
```

### 10.2 Step 2: 用 LLM（任何可用的模型）逐条生成

将 `/tmp/eval_samples.txt` 的内容喂给 LLM，使用 **第四节的 Stage 1 + Stage 2 Prompt** 生成评测用例。

### 10.3 Step 3: 组装 + 评测

```bash
# 导入块
SOURCE_DIR=process/data/抖音电商规则中心_blocked bash scripts/build_index.sh

# Layer 1 评测
python3 -c "
import json
from rag import pipeline

with open('process/data/eval_cases.json') as f:
    cases = json.load(f)

for c in cases:
    result = pipeline.retrieve(c['query'], top_k=5)
    hit = any(r.get('page_name') == c['expected_page_name'] for r in result)
    print(f\"{'✅' if hit else '❌'} {c['id']}: {c['query'][:60]}\")
"
```

---

## 十一、后续扩展

1. **自动标注工具**：Python 脚本实现 Phase 1-5 全自动流程
2. **持续评测 CI**：每次知识库更新后自动运行评测，检测召回率退化
3. **人工精标 Gold Set**：50 条人工校验的高质量用例作为基准
4. **Reranker 专项评测**：单独评测 HtmlRAG 两阶段剪枝前后的效果
5. **多语言 query**：生成中英混合、方言风格的 query 测试鲁棒性
6. **难度自动校准**：用实际检索结果反推难度标注是否准确

---

## 十二、关键风险与缓解

| 风险 | 影响 | 缓解策略 |
|------|------|---------|
| LLM 生成的 query 与真实用户行为差异大 | 评测结果不反映真实表现 | 对比人工标注 + 真实客服日志（如有）校准 |
| LLM 幻觉生成不存在于源chunk的query | 预期答案无法匹配 | Phase 2 Stage 2 自检 + Phase 3 LLM审核 |
| 采样偏差（某些类别未被充分覆盖） | 评测结论不全面 | 分层采样确保 8 大类全覆盖 |
| 评测集泄露到训练集 | 虚假高指标 | 评测集与模型训练数据严格隔离 |
| vLLM 服务不可用时阻塞流程 | 无法生成评测集 | 支持 fallback 到外部 API（如 Claude API） |
