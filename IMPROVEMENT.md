# Reranker 监督微调数据集构造方案

> 更新日期：2026-07-12
>
> 目标：利用 `process/data/抖音电商规则中心_blocked` 中的 9081 个 chunk 构造高质量 SFT 数据集，对 Qwen3-Reranker-4B 进行监督微调，提升在 HtmlRAG 两阶段剪枝中的精排效果。

---

## 〇、背景与问题分析

### 0.1 Reranker 在现有系统中的角色

```
用户 Query
    │
    ▼  Stage 1: Embedding 粗检（Bi-Encoder）
    │   从 9081 个 chunk 中召回 Top-K 候选（如 K=50）
    │   速度快但精度有限——只看 query 和 doc 各自的向量，无法捕捉细粒度交互
    │
    ▼  Stage 2: Reranker 精排（Cross-Encoder）  ← 本次微调目标
    │   对 K 个候选做 (query, doc) 联合编码，逐对打分
    │   保留 Top-K' 送入 LLM 生成回答（如 K'=5）
    │   速度慢但精度高——能看到 query 和 doc 的 token-level 交互
```

### 0.2 为什么要微调 Reranker？

| 问题 | 说明 |
|------|------|
| **领域偏移** | 基座 Qwen3-Reranker-4B 用通用语料训练，不理解"抖音电商规则"的术语和表达习惯（如"抖客""精选联盟""商品品质差"等） |
| **硬负样本盲区** | 通用 reranker 容易被关键词重叠欺骗——"抖客API准入"和"抖客推广准入"字面高度相似但内容完全不同，基座模型难以区分 |
| **长文本理解不足** | 电商规则 chunk 常包含复杂层级结构（条/款/项），基座模型可能无法准确定位答案所在 |
| **零样本局限** | 当前系统用 reranker 做 zero-shot 推理，未利用已有领域数据的监督信号 |

### 0.3 用户思路（正负样本）的可行性评估

**完全可行且是正确方向。** 正负样本构造是 reranker SFT 的核心范式：

- ✅ 正向样本（relevant pairs）教会模型"什么算相关"
- ✅ 负向样本（irrelevant pairs）教会模型"什么不算相关"
- ✅ **硬负样本（hard negatives）** 是 reranker 训练的**最大增值点**——教会模型区分"长得像但不相关"的文档

下文将对这一思路进行系统性扩充，并提供多种互补方案。

---

## 一、已有训练基础设施

项目已具备完整的 reranker 训练代码，无需从零搭建：

| 训练模块 | 文件 | 数据格式 | 用途 |
|---------|------|---------|------|
| **SFT（Pointwise）** | `model/trainer/reranker_ft.py` | `reranker_qa_pointwise.jsonl`：`{"query": "...", "doc": "...", "label": 0/1/2}` | 学习绝对相关性分数（标签归一化到 [0,1]） |
| **DPO（Preference）** | `model/trainer/reranker_dpo.py` | `reranker_qa_dpo.jsonl`：`{"query": "...", "chosen": "...", "rejected": "..."}` | 学习相对排序偏好（chosen > rejected） |

**推荐训练流程**：SFT 预训练 → DPO 精调（两者互补，SFT 打底学绝对相关性，DPO 在 SFT 基础上做偏好对齐）

---

## 二、核心思路：五阶段数据集构造方案

### 阶段总览

```
 blocked JSON（9081 chunks）
        │
        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  Phase 1: Query 生成（LLM 辅助）                              │
 │   每个 chunk → 多条自然语言 query（模拟真实用户提问）         │
 │   产出: ~15000-30000 条 (query, positive_chunk) 正向对        │
 ├──────────────────────────────────────────────────────────────┤
 │  Phase 2: 多级标签构造（含 Easy/Hard Negatives）              │
 │   为每个 query 标注 label=2/1/0 的 chunk                      │
 │   产出: reranker_qa_pointwise.jsonl（SFT 训练数据）           │
 ├──────────────────────────────────────────────────────────────┤
 │  Phase 3: 硬负样本专项挖掘（Hard Negative Mining）             │
 │   用 embedding 模型检索 + LLM 交叉验证生成高价值负样本        │
 │   产出: 增强版 pointwise 数据 + hard_negative 专项集          │
 ├──────────────────────────────────────────────────────────────┤
 │  Phase 4: 偏好对齐数据构造（DPO）                              │
 │   构造 (query, chosen, rejected) 三元组                       │
 │   产出: reranker_qa_dpo.jsonl（DPO 训练数据）                 │
 ├──────────────────────────────────────────────────────────────┤
 │  Phase 5: 质量验证与数据划分                                   │
 │   自动校验 + 人工抽检 Gold Set + train/val/test 划分          │
 └──────────────────────────────────────────────────────────────┘
```

---

## 三、Phase 1: LLM 驱动的 Query 生成

### 3.1 核心原则

- **一条 chunk 生成多条 query**：建议 3~5 条/chunk，覆盖不同问法和难度
- **模拟真实用户**：口语化、场景化、多样化（不要都是"XX是什么"）
- **基于 chunk 内容生成，不凭空编造**：确保 query 的答案确实在 chunk 中

### 3.2 Prompt 设计

#### Stage 1: Query 发散生成

```
你是一个电商客服领域的查询生成专家。根据以下抖音电商规则知识片段，
模拟真实商家/消费者/创作者可能提出的问题。

## 核心要求
1. 问题必须基于提供的知识内容，答案可在给定内容中找到
2. 模拟真实用户的语言风格——口语化、可能是半句话、带业务术语
3. 多样化提问角度，覆盖不同用户角色和使用场景
4. 每个问题标注难度

## 问题类型分布（每次生成 5 条）
- 直接查询型 (2条): "XX的规则是什么""XX需要什么条件"
- 场景应用型 (2条): "我是商家，遇到XX情况该怎么处理"
- 对比边界型 (1条): "A和B有什么区别""不做XX会怎样"

## 用户角色轮换（每批随机一种）
- 新手商家：对平台规则不熟悉，问题偏基础
- 老练商家：问题具体、涉及边缘场景
- 焦急商家：遇到罚款/封禁，语气紧迫
- 消费者：从购物体验角度提问
- 创作者/达人：从内容创作和推广角度提问

## 知识片段
文档名称: {page_name}
所属分类: {category} > {subcategory}
章节标题: {title}
内容摘要: {summary}
正文内容:
{text[:1500]}

请生成 5 个不同角度的问题，每行一个，格式：
问题文本 [难度: easy/medium/hard]
```

#### Stage 2: 格式化 + 答案提取 + 质量自检

```
你将 Stage 1 生成的问题格式化为 JSON，提取参考答案，做质量自检。

## 输入
知识正文: {text[:2000]}
Stage 1 生成的问题: {stage1_output}

## 输出（严格 JSON 数组）
[
  {
    "query": "用户问题原文",
    "difficulty": "easy|medium|hard",
    "query_type": "direct|scenario|comparison|boundary",
    "reference_answer": "基于知识正文的标准答案（100字内，用自己的话概括）",
    "answerable": true,
    "answer_source": "direct|inferred|partial",
    "keywords": ["关键词1", "关键词2"],
    "user_persona": "merchant_new|merchant_expert|merchant_urgent|consumer|creator",
    "quality_check": {
      "too_vague": false,
      "hallucinated": false,
      "answer_not_in_text": false
    }
  }
]

## 质量自检规则
- too_vague: 问题过于模糊（如"怎么样"无上下文）
- hallucinated: 问题引用了正文中不存在的信息
- answer_not_in_text: 仅从给定正文无法完整回答
```

### 3.3 多样性增强技术

```python
# 多样性注入——每批 query 随机选一种风格
DIVERSITY_HINTS = [
    "请用口语化语气提问",
    "请用电商行业术语丰富的表达提问",
    "请模拟用户从搜索引擎输入的简短关键词风格",
    "请模拟一个遇到紧急问题的商家（语气急迫）",
    "请模拟看了规则但仍然困惑的用户提问",
    "请用带方言口吻的表达方式提问",
    "请模拟对比两条规则后产生的困惑提问",
]

# 用户角色轮换
USER_PERSONAS = [
    "新手商家（对平台规则不熟悉，问题偏基础）",
    "老练商家（问题具体、涉及边缘场景和漏洞）",
    "焦急商家（刚收到罚款通知/店铺被封，语气紧迫）",
    "消费者（从购物体验和维权角度提问）",
    "创作者/达人（从内容创作和商品推广角度提问）",
    "供应商/服务商（从接入平台服务的B端角度提问）",
]
```

### 3.4 实现参数建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| 采样 chunk 数 | 2000~3000（约总体的 25%~33%） | 覆盖所有 15 个一级分类 |
| 每 chunk query 数 | 3~5 | 核心 chunk 多生成，边缘 chunk 少生成 |
| LLM | Qwen3.5-2B（本地 vLLM）或 Claude API | 本地零成本但质量略低，Claude 质量高但有 API 费用 |
| 预计总 query 数 | 8000~15000 | 取决于采样 chunk 数和 queries/chunk |
| 预计 token 消耗 | ~3M-5M tokens（本地 vLLM 零 API 费用） | |

---

## 四、Phase 2: 多级标签构造（核心）

### 4.1 标签体系设计

Reranker SFT 的核心是教会模型区分不同级别的相关性。建议使用三级标签：

| 标签 | 原始值 | 归一化值 | 含义 | 构造方式 |
|------|--------|---------|------|---------|
| **2（高度相关）** | 2 | 1.0 | 该 chunk 包含 query 的完整答案 | query 的来源 chunk（正样本） |
| **1（部分相关）** | 1 | 0.5 | 该 chunk 涉及相关话题但不能完整回答 | 同一文档的其他 chunk / 同一子类别的 chunk |
| **0（不相关）** | 0 | 0.0 | 该 chunk 与 query 无关 | 不同类别的随机 chunk / embedding 低分 chunk |

### 4.2 具体构造策略

#### 策略 A：基于文档结构的自动标注（无需 LLM）

```python
def auto_label_for_query(query_data, all_chunks_index):
    """
    利用已有的文档/目录结构自动构造多级标签。
    无需额外 LLM 调用，计算成本极低。
    """
    source_chunk = query_data["source_chunk"]
    source_page = source_chunk["page_name"]
    source_category = query_data["category"]

    samples = []

    # Label 2: 正样本（来源 chunk 本身 + 同一文档的相邻 chunk）
    samples.append({
        "query": query_data["query"],
        "doc": source_chunk["text"],
        "label": 2,
        "source": "source_chunk"
    })

    # 同一文档的其他 chunk（不同章节）
    same_page_chunks = all_chunks_index[source_page]
    for c in same_page_chunks:
        if c["chunk_idx"] != source_chunk["chunk_idx"]:
            samples.append({
                "query": query_data["query"],
                "doc": c["text"],
                "label": 1,  # 部分相关
                "source": "same_page_other_chunk"
            })

    # Label 0: 不同一级分类的随机 chunk（大概率不相关）
    diff_category_chunks = [
        c for c in all_chunks_index.values()
        if c["category"] != source_category
    ]
    for c in random.sample(diff_category_chunks, min(3, len(diff_category_chunks))):
        samples.append({
            "query": query_data["query"],
            "doc": c["text"],
            "label": 0,
            "source": "diff_category_random"
        })

    return samples
```

#### 策略 B：LLM 辅助的精细标注（高精度但需 LLM 调用）

对自动标注的边界样本（label=1 不确定是否真的部分相关），用 LLM 做二次确认：

```
你是一个相关性判断专家。请判断以下 (query, document) 对的相关性级别。

Query: {query}
Document: {doc_text[:1000]}

相关性定义：
- 2 (高度相关): 文档直接包含query的答案，能完整回答问题
- 1 (部分相关): 文档涉及相同主题/领域，但不能直接回答query
- 0 (不相关): 文档内容与query无关

请输出 JSON: {"label": 0|1|2, "reason": "一句话理由"}
```

### 4.3 正负样本比例控制

这是 reranker SFT 最关键的超参数之一：

| 比例 | 效果 | 适用场景 |
|------|------|---------|
| **正:负 = 1:3** | 模型学会有效排除不相关文档 | 推荐起点 |
| **正:负 = 1:5** | 模型更保守，倾向于打低分 | 候选池很大时（K=50） |
| **正:负 = 1:1** | 模型更平衡 | 候选池较小时（K=10） |

建议配置：

```
每个 query 的样本构成：
  - Label 2（正样本）: 1-2 条（来源 chunk ± 相邻 chunk）
  - Label 1（部分相关）: 2-3 条（同文档其他 chunk）
  - Label 0（负样本）: 5-8 条（不同类别随机 + 硬负样本）
      ├── 简单负样本 (3-4条): 完全不同类别的随机 chunk
      └── 硬负样本 (2-4条): embedding 相似但不相关的 chunk（Phase 3 生成）
```

---

## 五、Phase 3: 硬负样本专项挖掘（最大增值点 ⭐）

### 5.1 为什么硬负样本对 Reranker 至关重要？

Reranker 和 Embedding 模型的本质区别在于：

- **Embedding 模型（Bi-Encoder）**：query 和 doc 独立编码 → 对"关键词重叠但语义不同"的分辨力弱
- **Reranker（Cross-Encoder）**：query 和 doc 联合编码 → 有能力学习 token-level 的细粒度语义交互

硬负样本就是那些 **Embedding 模型容易给高分（向量检索排名靠前），但实际上与 query 不相关或不完全相关** 的 chunk。用这些样本训练 reranker，能教会模型"透过关键词表面理解真实意图"——这正是 reranker 存在的核心价值。

### 5.2 硬负样本挖掘方法

#### 方法 1: Embedding 相似度挖掘（自动，零 LLM 成本）

```python
def mine_hard_negatives_embedding(query, positive_chunk, all_chunks, embedder, top_k=20):
    """
    用 embedding 模型检索 top-k 相似 chunk，
    排除正样本后，取高分但不相关的作为硬负样本。

    原理：embedding 给高分说明"向量空间里离 query 很近"，
    这正是 reranker 需要学会"推翻"的候选。
    """
    query_vec = embedder.embed_query(query)
    scores = []
    for chunk in all_chunks:
        # 排除正样本本身
        if chunk["id"] == positive_chunk["id"]:
            continue
        sim = cosine_similarity(query_vec, chunk["embedding"])
        scores.append((sim, chunk))

    scores.sort(key=lambda x: -x[0])
    candidates = scores[:top_k]  # embedding top-20

    # 按与正样本的相似度分档
    hard_negatives = []
    for sim, chunk in candidates:
        # 同一文档但不是同一 chunk → label=1（部分相关，不是严格负样本）
        if chunk["page_name"] == positive_chunk["page_name"]:
            continue
        # 不同类别但 embedding 相似度高 → 硬负样本 label=0
        if chunk["category"] != positive_chunk["category"]:
            hard_negatives.append({
                "chunk": chunk,
                "embedding_sim": sim,
                "type": "hard_negative_cross_category"
            })

    return hard_negatives[:5]
```

#### 方法 2: 标题/关键词混淆挖掘（结构感知，零 LLM 成本）

```python
def mine_hard_negatives_title_confusion(positive_chunk, all_chunks_index):
    """
    找到与正样本 chunk 标题/关键词高度相似但内容不同的 chunk。

    典型场景：
      正样本: "抖客API服务商准入及考核管理规则"（API技术规范）
      硬负样本: "抖客推广准入及准出管理规则"（个人推广者准入）
      → 标题都有"抖客""准入"，但内容完全不同
    """
    import jieba

    pos_keywords = set(jieba.cut(positive_chunk["title"]))
    candidates = []

    for chunk in all_chunks_index.values():
        if chunk["page_name"] == positive_chunk["page_name"]:
            continue  # 跳过同一文档
        chunk_keywords = set(jieba.cut(chunk.get("title", "")))
        overlap = len(pos_keywords & chunk_keywords)
        if overlap >= 2:  # 至少共享 2 个关键词
            candidates.append((overlap, chunk))

    candidates.sort(key=lambda x: -x[0])
    return [c for _, c in candidates[:5]]
```

#### 方法 3: LLM 生成混淆 Query（高质量，需 LLM 调用）

这是最精细的硬负样本构造方式——让 LLM 基于两个相似但不同的 chunk，生成一个"看起来像在问 A 但实际在问 B"的 query：

```
你是一个对抗样本生成专家。你有两个内容相似但实质不同的知识块：

块A（正确答案来源）:
  标题: {chunk_A["title"]}
  内容: {chunk_A["text"][:1000]}

块B（干扰项）:
  标题: {chunk_B["title"]}
  内容: {chunk_B["text"][:1000]}

请生成 2 个用户查询，要求：
- 查询的关键词与块B的标题/术语高度重叠（容易让向量检索被B"骗走"）
- 但语义上查询真正想问的是块A的内容
- 查询应该是自然、合理的用户提问

示例：
  块A: "抖客API服务商准入及考核管理规则"（关于API接入的技术规范）
  块B: "抖客推广准入及准出管理规则"（关于个人推广者的准入条件）
  生成query: "抖客API接入后，推广者还需要满足什么条件才能开始推广？"
  → 关键词"抖客""推广""条件"容易命中B，但真实需求是A中的技术准入

输出 JSON: [{"query": "...", "true_chunk": "A", "confusion_chunk": "B", "confusion_reason": "..."}]
```

### 5.3 硬负样本的质量验证

```python
def validate_hard_negative(query, positive_chunk, hard_negative_chunk, llm_client):
    """
    用 LLM 验证硬负样本的质量：
    1. 硬负样本确实不能回答 query
    2. 硬负样本与 query 有足够的表面相似性（否则不是"硬"负样本）
    """
    prompt = f"""判断以下两个文档块对 query 的相关性：

Query: {query}

文档A（正样本候选）: {positive_chunk["text"][:500]}
文档B（硬负样本候选）: {hard_negative_chunk["text"][:500]}

请回答：
1. 文档A是否能回答query？(yes/no)
2. 文档B是否能回答query？(yes/no)
3. 文档B和query在主题/术语上是否有表面相似性？(yes/no)
4. 这是否是一个有效的硬负样本（B表面相似但不能回答问题）？(yes/no)

输出 JSON: {{"A_answers": "yes/no", "B_answers": "yes/no",
           "B_surface_similar": "yes/no", "valid_hard_negative": "yes/no"}}
"""
    result = llm_client(prompt)
    return result["valid_hard_negative"] == "yes"
```

---

## 六、Phase 4: DPO 偏好对齐数据构造

### 6.1 SFT vs DPO 的分工

| 维度 | SFT (Pointwise) | DPO (Preference) |
|------|----------------|------------------|
| 学什么 | 绝对相关性分数（这个 doc 有多相关？） | 相对排序偏好（A 和 B 哪个更相关？） |
| 数据格式 | `(query, doc, label)` | `(query, chosen, rejected)` |
| 适用场景 | 初步微调，建立领域相关性认知 | SFT 后的偏好精调 |
| 对数据的要求 | 需要准确的 label 标注 | 需要 chosen 明确优于 rejected |

### 6.2 DPO 数据构造策略

#### 策略 1: 基于多级标签自动构造（零额外成本）

直接从 Phase 2 的 pointwise 数据转换：

```python
def pointwise_to_dpo(pointwise_samples):
    """
    从 pointwise 数据自动生成 DPO 偏好对。

    规则：
    - (label=2, label=0) → chosen=label2, rejected=label0  ✅ 高置信度
    - (label=2, label=1) → chosen=label2, rejected=label1  ✅ 中等置信度
    - (label=1, label=0) → chosen=label1, rejected=label0  ⚠️ 低置信度（需 LLM 确认）
    """
    by_query = {}
    for s in pointwise_samples:
        by_query.setdefault(s["query"], []).append(s)

    dpo_pairs = []
    for query, samples in by_query.items():
        label2 = [s for s in samples if s["label"] == 2]
        label1 = [s for s in samples if s["label"] == 1]
        label0 = [s for s in samples if s["label"] == 0]

        # 高置信度: label=2 vs label=0
        for chosen in label2:
            for rejected in label0:
                dpo_pairs.append({
                    "query": query,
                    "chosen": chosen["doc"],
                    "rejected": rejected["doc"],
                    "confidence": "high"
                })

        # 中等置信度: label=2 vs label=1
        for chosen in label2:
            for rejected in label1:
                dpo_pairs.append({
                    "query": query,
                    "chosen": chosen["doc"],
                    "rejected": rejected["doc"],
                    "confidence": "medium"
                })

    return dpo_pairs
```

#### 策略 2: 同一 Query 下的文档对比较（LLM 辅助）

对同一 query，让 LLM 对两个候选 chunk 做精细对比：

```
对于以下 query，比较两个文档块的相关性：

Query: {query}

文档A:
标题: {chunk_A["title"]}
内容: {chunk_A["text"][:800]}

文档B:
标题: {chunk_B["title"]}
内容: {chunk_B["text"][:800]}

请判断哪个文档更能回答 query。考虑：
1. 哪个包含更直接、更完整的答案？
2. 哪个更聚焦于 query 所问的具体问题？
3. 如果两篇都不相关，请说明

输出 JSON:
{
  "better_doc": "A" | "B" | "tie" | "neither",
  "confidence": "high" | "medium" | "low",
  "reason": "一句话理由"
}
```

#### 策略 3: 难分文档对（最宝贵的 DPO 数据）

选择同一分类下、标题相似但内容焦点不同的两个 chunk 构造 DPO 对：

```python
def construct_challenging_dpo_pairs(all_chunks_index):
    """
    找到"难分"的文档对——它们在同一分类下，主题相近，
    但针对特定 query 只有一个真正相关。
    这种数据最能教会 reranker 做精细区分。
    """
    pairs = []
    for category, chunks in group_by_category(all_chunks_index):
        # 按标题关键词聚类
        clusters = cluster_by_title_keywords(chunks)
        for cluster in clusters:
            if len(cluster) >= 2:
                # 同一 cluster 内的文档主题最相近 → 最难区分
                for i in range(len(cluster)):
                    for j in range(i + 1, len(cluster)):
                        pairs.append((cluster[i], cluster[j], "same_cluster"))
    return pairs
```

---

## 七、Phase 5: 质量验证与数据划分

### 7.1 自动质量校验

```python
def validate_dataset(samples, format="pointwise"):
    """数据集自动质量校验。

    检查项:
    - JSON 格式正确性
    - query 非空且长度合理 (5 ≤ len ≤ 200)
    - doc 非空且长度合理 (≥ 20 字符)
    - label 值合法 (pointwise: 0/1/2, dpo: chosen ≠ rejected)
    - query-doc 对不重复
    - 各类别覆盖率 ≥ 80%
    - 标签分布均衡度 (label 0/1/2 比例接近预设)
    """
    checks = {
        "format_valid": True,
        "query_length_ok": True,
        "doc_length_ok": True,
        "label_valid": True,
        "no_duplicates": True,
        "category_coverage": 0.0,
        "label_distribution": {},
    }
    # ... 实现细节
    return checks
```

### 7.2 人工抽检 Gold Set

从最终数据集中随机抽 50~100 条，人工检查：

| 检查项 | 标准 |
|--------|------|
| Query 自然度 | 像真实用户提问，不模板化 |
| 正样本准确性 | Label=2 的 doc 确实能回答 query |
| 负样本合理性 | Label=0 的 doc 确实不能回答 query |
| 硬负样本质量 | 硬负样本确实"看起来相关但其实不相关" |
| 标签一致性 | 不同审核者对同一条的 label 判断一致 |

### 7.3 数据划分

```
dataset/reranker/
├── reranker_qa_pointwise_train.jsonl    # SFT 训练集 (80%)
├── reranker_qa_pointwise_val.jsonl      # SFT 验证集 (10%)
├── reranker_qa_pointwise_test.jsonl     # SFT 测试集 (10%)
├── reranker_qa_dpo_train.jsonl         # DPO 训练集 (80%)
├── reranker_qa_dpo_val.jsonl           # DPO 验证集 (10%)
├── reranker_qa_dpo_test.jsonl          # DPO 测试集 (10%)
└── gold_set.json                        # 人工标注 Gold Set (50~100条)
```

---

## 八、进阶方案：超越正负样本

除了基本的正负样本 SFT，还有以下互补方案可进一步提升 reranker 效果：

### 8.1 对比学习增强（Contrastive Fine-tuning）

在 SFT 中引入对比学习损失，让模型在 batch 内自动挖掘负样本：

```python
# 在标准 BCE Loss 基础上增加对比损失
def contrastive_reranker_loss(scores, labels, temperature=0.07):
    """
    scores: [batch_size] — 模型对每个 (query, doc) 的打分
    labels: [batch_size] — 0/1/2 标签

    思想：在同一个 batch 内，label=2 的样本应该得分显著高于 label=0 的样本
    """
    bce_loss = F.binary_cross_entropy_with_logits(scores, labels / 2.0)

    # 对比损失：正样本分数应高于所有负样本
    pos_mask = labels == 2
    neg_mask = labels == 0
    if pos_mask.any() and neg_mask.any():
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]
        # 正样本与每个负样本的 pairwise 对比
        logits = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        contrastive_loss = -F.logsigmoid(logits / temperature).mean()
        return bce_loss + 0.3 * contrastive_loss
    return bce_loss
```

### 8.2 课程学习（Curriculum Learning）

分阶段训练，从易到难：

```
Stage 1 (Easy): 只用 label=2（正样本）+ label=0（简单负样本：不同大类随机）
    目标：模型学会基本的"相关 vs 不相关"判断

Stage 2 (Medium): 加入 label=1（部分相关样本）+ embedding 挖掘的硬负样本
    目标：模型学会细粒度区分

Stage 3 (Hard): 全量硬负样本 + 混淆 query + DPO 偏好对
    目标：模型对难样本有最佳区分力
```

### 8.3 数据增强：Query 改写与回译

对已有 query 做多样性增强，扩展训练集：

```python
def augment_queries(queries, llm_client, n_variants=3):
    """
    对每条 query 生成 n_variants 个语义相同但表达不同的版本。
    - 同义词替换（如"规则"→"规定""要求"）
    - 句式变换（疑问句→陈述句、长句→短句）
    - 角色改写（商家视角→消费者视角）
    """
    prompt = f"""将以下用户查询改写为 {n_variants} 个语义相同但表达不同的版本。
要求：保持原意不变，但变换用词、句式、语气。

原始查询: {query}

输出 JSON: {{"variants": ["改写1", "改写2", "改写3"]}}
"""
    # 变体 query 对应的正样本/负样本标签不变
```

### 8.4 多任务学习

在 reranker 训练中同时优化相关性评分 + 分类（该 chunk 属于哪个类别）：

```python
# 多任务头设计
class MultiTaskReranker(nn.Module):
    def __init__(self, base_model):
        self.encoder = base_model
        self.relevance_head = nn.Linear(hidden_size, 1)     # 相关性打分（主任务）
        self.category_head = nn.Linear(hidden_size, 15)      # 类别分类（辅助任务）

    def forward(self, **inputs):
        hidden = self.encoder(**inputs).last_hidden_state[:, 0]
        relevance_score = self.relevance_head(hidden)
        category_logits = self.category_head(hidden)
        return relevance_score, category_logits

# 辅助任务可以帮助模型学习领域知识结构
# loss = BCE(relevance) + 0.1 * CrossEntropy(category)
```

---

## 九、与现有训练代码的集成

### 9.1 SFT 训练（Pointwise）

```bash
# Step 1: 生成 SFT 数据集（本次工作的核心）
PYTHONPATH=. python -m model.utils.build_reranker_sft_data \
    --blocked-dir process/data/抖音电商规则中心_blocked \
    --output dataset/reranker/reranker_qa_pointwise.jsonl \
    --sample-chunks 2000 \
    --queries-per-chunk 4 \
    --hard-negative-ratio 3 \
    --vllm-url http://localhost:8011/v1/chat/completions

# Step 2: SFT 训练（已有代码，直接使用）
PYTHONPATH=. python -m model.trainer.reranker_ft
# 训练数据路径: dataset/reranker_qa_pointwise.jsonl
# 模型输出: model/trained_reranker/
```

### 9.2 DPO 训练（Preference）

```bash
# Step 1: 从 pointwise 数据构造 DPO 偏好对
PYTHONPATH=. python -m model.utils.build_reranker_dpo_data \
    --pointwise-data dataset/reranker/reranker_qa_pointwise.jsonl \
    --output dataset/reranker/reranker_qa_dpo.jsonl \
    --llm-verification \
    --vllm-url http://localhost:8011/v1/chat/completions

# Step 2: DPO 训练（已有代码，直接使用）
PYTHONPATH=. python -m model.trainer.reranker_dpo
# 训练数据路径: dataset/reranker_qa_dpo.jsonl
# 模型输出: model/trained_reranker_dpo/
```

### 9.3 训练后评估

```bash
# 在 Gold Set 上对比基座模型 vs SFT 模型 vs SFT+DPO 模型
PYTHONPATH=. python -m tests.test_model.test_reranker_dpo
# 或编写专门的评估脚本，用 Gold Set 测 Recall@K / NDCG / MRR
```

---

## 十、推荐执行路径

### 10.1 最小可行方案（MVP，2~3 天）

如果时间有限，优先执行：

```
Day 1: Phase 1 (Query 生成) + Phase 2 (基于文档结构自动标注)
       ├── 采样 500 个 chunk，用 vLLM 生成 ~2000 条 query
       ├── 基于文档结构自动构造 label=2/1/0
       └── 产出 ~10000-15000 条 pointwise 训练数据

Day 2: Phase 3 (硬负样本挖掘)
       ├── 用 embedding 模型检索 top-20 候选 → 筛选硬负样本
       └── 增强 pointwise 数据

Day 3: SFT 训练 + 评估
       ├── 运行 reranker_ft.py 训练
       └── 在 Gold Set 上对比训练前后效果
```

### 10.2 完整方案（推荐，1~2 周）

```
Week 1:
  Day 1-2: Phase 1 (Query 生成，2000-3000 chunks)
  Day 3-4: Phase 2 (多级标签构造 + LLM 精细标注)
  Day 5:   Phase 3 (硬负样本挖掘)
  Day 6-7: Phase 5 (质量验证) + SFT 训练

Week 2:
  Day 1-2: Phase 4 (DPO 数据构造)
  Day 3-4: DPO 训练
  Day 5:   完整评估（基座 vs SFT vs SFT+DPO）
  Day 6-7: 分析与迭代（根据评估结果调整数据构造策略）
```

### 10.3 迭代优化建议

```
第一轮: MVP 跑通 → 看 SFT 后 reranker 是否有提升
第二轮: 如果提升不明显 → 重点加强硬负样本质量
第三轮: 如果硬负样本质量已很高 → 加入 DPO 做偏好精调
第四轮: 如果仍有瓶颈 → 考虑课程学习 / 对比学习增强
```

---

## 十一、数据构造工具实现清单

建议在 `model/utils/` 下新增以下工具脚本：

```
model/utils/
├── __init__.py
├── build_reranker_sft_data.py     # Phase 1+2+3 主脚本：query生成 + 多级标签 + 硬负样本
├── build_reranker_dpo_data.py     # Phase 4: DPO偏好对构造
├── reranker_data_prompts.py       # 所有 LLM prompt 模板
├── hard_negative_miner.py         # 硬负样本挖掘（embedding + 标题 + LLM三种方法）
├── data_validator.py              # Phase 5: 质量校验
└── README.md                      # 使用说明
```

### 核心 API 设计

```python
# build_reranker_sft_data.py 主入口
def build_sft_dataset(
    blocked_dir: str,              # _blocked 数据目录
    output_path: str,             # 输出 jsonl 路径
    sample_chunks: int = 2000,    # 采样 chunk 数
    queries_per_chunk: int = 4,   # 每 chunk 生成的 query 数
    hard_neg_ratio: float = 3.0,  # 负样本/正样本比例
    llm_api_url: str = None,      # LLM API（vLLM 或 Claude）
    model: str = "Qwen/Qwen3.5-2B",
    use_llm_labeling: bool = False,  # 是否用 LLM 做精细标注
    use_llm_confusion: bool = False, # 是否用 LLM 生成混淆 query
) -> str:
    """返回生成的 pointwise jsonl 文件路径"""
    ...

# hard_negative_miner.py
def mine_hard_negatives(
    query: str,
    positive_chunk: dict,
    all_chunks: List[dict],
    embedder,                       # embedding 模型
    method: str = "all",           # "embedding" | "title" | "llm" | "all"
    top_k: int = 5,
) -> List[dict]:
    """返回硬负样本列表"""
    ...
```

---

## 十二、常见问题与风险

| 问题 | 风险 | 缓解措施 |
|------|------|---------|
| LLM 生成的 query 与真实用户行为差距大 | 训练后的 reranker 在真实场景效果不佳 | 尽量用真实客服日志校准（如有）；增加 query 多样性注入；人工抽检自然度 |
| 正负样本比例失衡导致模型过保守/过激进 | reranker 全部打高分或全部打低分 | 严格按 1:3~1:5 控制比例；监控训练集标签分布 |
| 硬负样本不够"硬"（太容易区分） | 模型学不到精细区分能力 | 用 embedding 相似度筛选 top-10；LLM 验证硬负样本质量 |
| 数据泄露：测试集的 chunk 出现在训练集 | 评测指标虚高，上线效果差 | 按 page_name 做 train/val/test split（同一文档的所有 chunk 只在同一个 split） |
| SFT 后模型遗忘通用能力 | 在非电商领域的 reranker 任务上退化 | 保留 5%~10% 通用 reranker 数据混合训练；监控通用 benchmark |
| vLLM 不可用时阻塞数据构造 | 无法生成 query | 支持 fallback 到 Claude API 或 OpenAI API |

---

## 十三、总结

### 核心要点

1. **正负样本思路完全可行**——这是 reranker SFT 的标准范式，项目已有完整的训练代码（`reranker_ft.py` + `reranker_dpo.py`）可直接使用

2. **关键不是"有没有正负样本"，而是"负样本的质量"**——硬负样本（embedding 高分但不相关的 chunk）是 reranker SFT 的最大增值点

3. **推荐 SFT → DPO 两阶段训练**：SFT 打底学绝对相关性（pointwise），DPO 精调学相对排序偏好（pairwise）

4. **LLM 辅助数据构造是正确思路**——用本地 vLLM (Qwen3.5-2B) 零成本批量生成 query，LLM 辅助质量验证和混淆样本生成

5. **数据构造本身是迭代过程**：MVP 快速跑通 → 评估 → 针对性加强（硬负样本/DPO/课程学习）→ 再评估

### 数据构造优先级

```
Phase 1 (Query 生成) + Phase 2 (自动标注)  ← 必做，MVP 的基础
    │
    ├── Phase 3 (硬负样本挖掘)              ← 强烈推荐，最大的效果提升来源
    │
    ├── Phase 4 (DPO 偏好对)               ← 推荐，在 SFT 基础上进一步提升
    │
    └── 进阶方案（对比学习/课程学习等）     ← 可选，SFT+DPO 效果不够时再尝试
```
