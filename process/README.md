# Process — HTML 数据处理流水线

> 更新日期：2026-07-07

## 一、项目概述

本项目是 **CustomerServiceAgent** 的数据处理子模块，负责将飞书文档导出的原始 HTML 知识库清洗、结构化分块，并生成可供 RAG 检索与向量化的文档块 JSON。

### 处理流水线

```
原始 HTML（飞书导出）
    │
    ▼  Step 1: HTML 清洗（process/src/html_utils.py）
    去噪 → 飞书噪声预处理 → 语义块转换 → 域包装 → 表格展开
    │
    ▼  Step 2: 结构化分块（process/src/html_utils.py::build_block_tree）
    BFS 拆分 → heading 域分块 → 表格按行切分
    │
    ▼  Step 3: 文档块生成（process/src/text_process.py）
    摘要生成 → 问句生成 → JSON 持久化
    │
    ▼  Step 4: HtmlRAG 两阶段剪枝（process/src/html_pruner.py）
    Embedding 粗剪 → Reranker 精剪 → 保留 HTML 结构的精简上下文
    │
    ▼
    结构化 JSON 文档块 → 向量数据库 / ES / LLM 上下文
```

---

## 二、HTML 清洗详解

### 2.1 数据来源

所有原始 HTML 均为**飞书（Lark/Feishu）文档导出**的片段，具有以下特征：

- **极深嵌套**：每个语义块被 6-8 层 div 包装（heading → heading-block → heading → heading-content → zone-container → text-editor → ace-line → span）
- **大量 UI 噪声**：折叠按钮、SVG 图标、占位符、零宽字符、emoji 容器等
- **飞书特有 class**：`docx-*-block`、`zone-container`、`ace-line`、`heading-hN` 等
- **`data-block-type` 语义标识**：飞书用此属性标识块类型（heading1~6 / text / bullet / ordered / callout / quote_container / grid / table_cell）

### 2.2 清洗流程

```
原始 HTML
    │
    ▼  1. _clean_feishu_noise()     ← 飞书噪声预处理（新增）
    │     - 移除噪声元素（fold-wrapper / placeholder / emoji / SVG / bullet-dot 等）
    │     - 移除零宽字符（data-enter / data-zero-space）
    │     - 展平飞书包装层（zone-container / text-editor / heading-block 等，5 次迭代）
    │     - 展平仅含文本的 span 标签
    │
    ▼  2. 移除标准噪声标签           ← 通用清洗
    │     - script / style / svg / input / button / nav / aside / footer 等
    │
    ▼  3. 移除隐藏元素                ← 通用清洗
    │     - display:none / visibility:hidden / class 含 hidden
    │
    ▼  4. heading-hN class 转换       ← 飞书标题识别
    │     - heading-h1 class → data-block-type="heading1"
    │     （飞书噪声预处理后，内层 heading-hN 已被展平，
    │           仅外层 block div 保留原始 data-block-type）
    │
    ▼  5. 属性清除                    ← 通用清洗
    │     - 保留 data-block-type / colspan / rowspan
    │     - 移除其余所有属性
    │
    ▼  6. 空标签移除 + 注释移除       ← 通用清洗
    │
    ▼  7. 冗余包装合并（3 次迭代）     ← 通用清洗
    │     - 单子标签的父标签 → 用子标签替换
    │
    ▼  8. warp_domains()              ← 语义结构重建
    │     a. _convert_semantic_blocks()  ← 新增
    │        - data-block-type="text"            → <p>
    │        - data-block-type="bullet"          → <li>
    │        - data-block-type="ordered"         → <li>
    │        - data-block-type="callout"         → unwrap（保留内容）
    │        - data-block-type="quote_container" → <blockquote>
    │        - data-block-type="grid/grid_column" → unwrap
    │     b. _convert_headings()
    │        - data-block-type="headingN" → <hN>
    │        - 转换后清除子元素残留的 data-block-type（避免嵌套重复转换）
    │     c. 按标题层级包装 <div class="hN_domain">
    │     d. 包装表格 <div class="table_domain">
    │
    ▼  9. expand_table_spans()        ← 表格处理
    │     - 展开 colspan/rowspan 合并单元格
    │
    ▼  10. clean_xml() + clean_html_text()  ← 文本规范化
          - 移除 XML 声明 / DOCTYPE
          - 去除 markdown 块标记
          - 合并连续空行
```

### 2.3 清洗效果示例

**输入**（飞书原始 HTML）：
```html
<div class="block docx-heading1-block" data-block-type="heading1">
  <div class="heading-block"><div class="heading heading-h1 heading-block-align-">
    <div class="heading-content"><div class="zone-container text-editor hide-placeholder non-empty">
      <div class="ace-line" data-node="true" dir="auto">
        <span class="author-..." data-leaf="true" data-string="true">一、产品简介</span>
        <span data-enter="true" data-leaf="true" data-string="true">​</span>
      </div></div></div></div></div></div>
<div class="fold-wrapper can-fold fold-block-id-2">
  <div class="fold-wrapper can-fold fold-block-id-2 fold-handler-wrapper">
    <div class="fold-handler"><div class="svg-wrapper"><svg>...</svg></div></div>
  </div></div>
```

**输出**（清洗后）：
```html
<div class="h1_domain">
<h1>一、产品简介</h1>
...
</div>
```

### 2.4 关键设计决策

1. **飞书噪声预处理在通用清洗之前**：飞书 HTML 的嵌套深度远超通用清洗的处理能力。先移除噪声元素和展平包装层，使后续通用清洗更高效。

2. **5 次迭代展平**：飞书包装层可深达 6-8 层，需多次迭代才能完全展平。

3. **span 展平仅针对纯文本 span**：表格单元格内的 span 保留（可能影响表格解析），仅展平不含子标签的纯文本 span。

4. **heading 转换后清除子元素 data-block-type**：避免内外两层 div 都被转换为 `<hN>` 导致嵌套标题。

5. **`data-block-type` 语义转换**：将飞书的自定义属性转换为标准 HTML 标签（`<p>` / `<li>` / `<blockquote>` 等），使下游 RAG 能利用标准 HTML 结构。

---

## 三、结构化分块（Block Tree）

### 3.1 分块算法

`build_block_tree()` 使用 **BFS（广度优先）** 算法将清洗后的 HTML 拆分为语义块：

```
输入: 清洗后 HTML（含 hN_domain / table_domain 包装）

1. 计算总词数
2. 若总词数 < min_node_words 且 > 0：整页作为单个块
3. 若总词数 > max_node_words：BFS 拆分
   a. 从根节点开始，遍历每个子节点
   b. table / tbody / li → 作为整体保留
   c. 词数 < min_node_words → 回收到父节点裸文本
   d. 词数 > max_node_words → 继续递归拆分
   e. 词数在 [min, max] 之间 → 独立成块
   f. 裸文本 → 构造独立 Tag（避免与子块重复）
4. 若总词数在 [min, max] 之间：按 heading 拆分
```

### 3.2 分块参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_node_words` | 4096 | 每个块的最大词数（中文按字符计） |
| `min_node_words` | 48 | 每个块的最小词数 |
| `zh_char` | True | 是否按字符计数（适配中文） |

### 3.3 块路径（Block Path）

每个块附带从根标签到当前块的路径，如 `["h1_domain", "h2_domain", "p"]`，用于：
- 唯一标识块
- 在剪枝时提供标题上下文
- 在重建 HTML 时保持层次结构

---

## 四、HtmlRAG 两阶段剪枝

### 4.1 算法概述

对标论文 *HTML is Better Than Plain Text for Modeling Retrieved Knowledge in RAG Systems*（HtmlRAG）的核心贡献——**Two-Stage Block Tree Pruning**：

```
清洗后的 HTML
    │
    ▼  Stage 1（粗剪，Embedding-based）
    在粗粒度块树上，用文本嵌入余弦相似度快速裁掉无关块
    │
    ▼  Stage 2（精剪，Reranker-based）
    在细粒度块树上，用 Reranker 交叉编码精确打分后二次剪枝
    │
    ▼
    剪枝后的 HTML（保留结构）→ 送入 LLM
```

### 4.2 关键改进

1. **标题上下文感知打分**：在打分时为每个块附加标题上下文（如 `[h1 h2]`），使打分器能感知块所属的章节层级，提升相关性判断的准确性。

2. **保留 HTML 结构重建**：`rebuild_html_with_domains()` 将同一 heading domain 下的连续块包装在 `<div class="hN_domain">` 中，保留文档层次结构。

3. **贪心选块算法**：在词数预算内优先保留高分块；分数相同时优先保留文档靠前的块；即使预算不足以容纳任何块，也至少保留分数最高的 1 个块。

4. **优雅降级**：打分服务不可用时不崩溃，回退为「不剪枝」（保留全部块）。

### 4.3 打分后端

| 阶段 | 后端 | 接口 |
|------|------|------|
| Stage 1 | vLLM / TEI Embedding | `default_embed_fn` → POST `/v1/embeddings` |
| Stage 2 | TEI Reranker | `default_rerank_fn` → POST `/rerank` |

测试环境可注入 mock 打分器，无需真实起服务。

---

## 五、使用方法

所有输出统一存放在 `process/data/` 目录下（文件夹级后缀，保留子目录结构）：
- 清洗后：`{数据集名}_cleaned/`（子目录结构和文件名不变）
- 分块后：`{数据集名}_blocked/`（子目录结构不变，.html → .json）

### 5.1 全流程（清洗 + 分块 + vLLM 摘要）

```bash
PYTHONPATH=process python -m main \
    --source-dir process/data/抖音电商规则中心 \
    --step all
# 输出：process/data/抖音电商规则中心_cleaned/ + process/data/抖音电商规则中心_blocked/
```

### 5.2 仅清洗

```bash
PYTHONPATH=process python -m main \
    --source-dir process/data/抖音电商规则中心 \
    --step clean
# 输出：process/data/抖音电商规则中心_cleaned/
```

### 5.3 仅分块（需先完成清洗，需要 vLLM 服务在运行）

```bash
PYTHONPATH=process python -m main \
    --html-dir process/data/抖音电商规则中心_cleaned \
    --step block
# 输出：process/data/抖音电商规则中心_blocked/
```

### 5.4 HtmlRAG 剪枝（运行时检索阶段）

```python
from src.html_pruner import two_stage_prune

pruned_html = two_stage_prune(
    html=cleaned_html,
    query="用户问题",
    stage1_max_context_words=4096,
    stage2_max_context_words=2048,
)
```

---

## 六、评测数据集构造

> 完整方案（Prompt 模板、代码设计、成本估算等）详见 [IMPROVEMENT.md](IMPROVEMENT.md)，此处仅概述核心思路。

### 6.1 核心思想

**用大模型从分块数据"反向生成"评测集**：以每个 chunk 的内容作为标准答案素材，让 LLM 模拟真实用户提出可能的问题（query），从而自动得到 `(query, 正确 chunk, 参考答案)` 三元组，供 RAG 检索与生成评测使用。

**关键前提：只依赖 chunk 文件**（`_blocked/*.json`）。每个 chunk 已内嵌 `text` / `summary` / `title` / `html_content` 及目录分类信息，构造评测所需素材全部齐备，无需再读取 HTML 文件。

### 6.2 主要步骤

```
_blocked/*.json
    │
    ▼  Step 1: 智能采样（无 LLM）
    │   分层采样 200-500 个代表性 chunk，覆盖全部知识分类
    │
    ▼  Step 2: LLM 生成 query（核心）
    │   两阶段 Prompt：① 发散生成多角度问题 ② 格式化 + 提取参考答案
    │   每个 chunk 产出若干条 (query, reference_answer)
    │
    ▼  Step 3: 质量过滤与去重
    │   规则过滤 + LLM 自检（防幻觉/防模糊）+ embedding 语义去重
    │
    ▼  Step 4: 高级用例构造（可选）
    │   LLM 生成混淆 query（Hard Negative）、跨文档综合、多轮追问
    │
    ▼  Step 5: 组装 + 入库回填
        导出标准格式 → 入库检索 → 回填 global_chunk_idx → 人工抽检
```

### 6.3 双层评测设计

评测集同时支持两个层级，兼顾"立即可用"与"精确度量"：

| 层级 | 匹配依据 | 指标 | 前置条件 |
|------|---------|------|---------|
| **Layer 1** | `page_name` / 目录分类 | 文档级命中率 | 入库前即可用 |
| **Layer 2** | `global_chunk_idx` | Recall@K / MRR / NDCG | 需入库后回填 ID |

**桥梁字段** `relevant_chunk_keys`（用 `(page_name, chunk_idx)` 标识块）在分块阶段即可确定，入库后再映射为 `global_chunk_idx`，解决"精确 ID 需入库才分配"的矛盾。

### 6.4 与现有评测对接

产出的 `eval_cases.json` 兼容项目已有评测代码：

- `rag/evaluation/retrieval_eval.py` —— 检索指标（Recall/Precision/MRR/NDCG）
- `rag/evaluation/generation_eval.py` —— 生成指标（groundedness/引用覆盖/lexical_f1）
- `rag/evaluation/rag_e2e_eval.py` —— 端到端评测编排

---

## 七、配置说明

配置文件：`CustomerServiceAgent/config/config.json`

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `lang` | `zh` | 语言（影响词数计算方式） |
| `max_node_words_embed` | 4096 | 分块最大词数 |
| `min_node_words_embed` | 48 | 分块最小词数 |
| `llm_model` | `Qwen/Qwen3.5-2B` | vLLM 摘要模型 |
| `embed_model` | `Qwen/Qwen3-Embedding-4B` | 嵌入模型 |
| `rerank_model` | `Qwen/Qwen3-Reranker-4B` | 精排模型 |
| `vllm_api_url` | `http://localhost:8011/v1/chat/completions` | vLLM API |
| `embed_api_url` | `http://localhost:8010/v1/embeddings` | 嵌入 API |
| `vllm_timeout` | 60 | API 超时（秒） |

可通过 `.env` 文件或环境变量覆盖。

---

## 八、依赖

```
beautifulsoup4 >= 4.12
jieba >= 0.42
scikit-learn >= 1.3
numpy >= 1.24
aiohttp >= 3.9（异步摘要）
python-dotenv（环境变量加载）
playwright（爬虫浏览器自动化）
```

---

## 九、审查与修复记录

### 9.1 代码审查发现的问题与修复

| 编号 | 问题描述 | 严重性 | 修复方式 |
|------|----------|--------|---------|
| C1 | `block-comment` 在移除列表中，导致 callout 内容被误删 | **严重** | 从 `_FEISHU_REMOVE_PREFIXES` 移至 `_FEISHU_UNWRAP_PREFIXES` |
| C2 | `heading-block-align-` 在移除列表中，导致标题文本被误删 | **严重** | 从移除列表中删除（纯样式类，属性清洗时自动清除） |
| C3 | `local-comment-all-third-party` 在移除列表中，导致 callout 内容被误删 | **严重** | 从 `_FEISHU_REMOVE_PREFIXES` 移至 `_FEISHU_UNWRAP_PREFIXES` |
| C4 | span 展平过于激进，破坏普通 HTML 的 UI 噪声检测 | **中等** | 仅展平含 `data-leaf` 或 `data-string` 属性的飞书 span |
| C5 | `_convert_headings` 转换后子元素残留 `data-block-type` 导致嵌套标题 | **中等** | 转换后清除子元素的 `data-block-type` 属性 |
| C6 | `_extract_heading_context` 仅匹配 `_domain` 后缀，不匹配 `_section` | **轻微** | 增加 `_section` 后缀匹配 |
| C7 | decompose 后标签 attrs 变 None，后续迭代报 AttributeError | **中等** | 增加 `tag.attrs is None` 安全检查 |
| C8 | `test_config.py::test_data_dir_exists` 检查 `process/dataset`（不存在） | **轻微** | 改为检查父目录 `process/` 存在 |

### 9.2 测试覆盖

| 测试文件 | 测试数 | 覆盖范围 |
|---------|--------|---------|
| `tests/test_process/test_html_utils.py` | 16 | clean_html / expand_table_spans / build_block_tree / parse_time_tag |
| `tests/test_process/test_html_utils_extended.py` | 12 | 深层嵌套 / 混合 colspan/rowspan / 空标签 / 标题层级 |
| `tests/test_process/test_html_pruner.py` | 29 | cosine_similarity / greedy_prune / rebuild_html / embedding 剪枝 / reranker 剪枝 |
| `tests/test_process/test_feishu_cleaning.py` | **41** | **新增**：飞书噪声预处理 / 语义块转换 / domain 重建 / 标题上下文 / 剪枝集成 |
| `tests/test_process/test_algorithm_completeness.py` | 12 | BFS 裸文本 / block_path / JSON 持久化 |
| `tests/test_process/test_algorithm_optimization.py` | 12 | SVG/input/nav 清理 / 隐藏元素 / UI 噪声过滤 |
| `tests/test_process/test_bugfix_regression.py` | — | 路径计算 / 字典一致性 / 异步分块 / 去重 |
| **总计** | **326 passed, 11 skipped** | 全部通过 |

运行测试：
```bash
PYTHONPATH=process:process/src:tests python -m pytest tests/test_process/ -v
```
