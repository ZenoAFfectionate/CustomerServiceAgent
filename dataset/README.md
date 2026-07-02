# 📊 电商客服 RAG 评测数据集

> 基于公开开源数据集，用于 CustomerServiceAgent RAG 系统的知识库构建与端到端效果评测。

## 数据集来源

### 主数据集：Bitext Customer Support LLM Chatbot Training Dataset

| 属性 | 值 |
|------|-----|
| **HuggingFace** | [bitext/Bitext-customer-support-llm-chatbot-training-dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) |
| **总行数** | 26,872 条 QA 对 |
| **字段** | `flags`, `instruction`, `category`, `intent`, `response` |
| **类别数** | 11 个（ORDER, REFUND, PAYMENT, DELIVERY, SHIPPING, INVOICE, ACCOUNT, CONTACT, FEEDBACK, CANCEL, SUBSCRIPTION） |
| **意图数** | 27 个（cancel_order, track_refund, payment_issue, delivery_options 等） |
| **语言** | English |
| **格式** | Parquet（已自动转换，可通过 pandas/pyarrow 读取） |
| **版权** | © Bitext Innovations, 2024. 用于研究与测试用途。 |

### 场景覆盖

| Category | 典型场景 | 意图示例 |
|----------|---------|---------|
| **ORDER** | 下单、改单、取消订单 | `place_order`, `change_order`, `cancel_order` |
| **REFUND** | 退款政策、退款追踪 | `check_refund_policy`, `track_refund`, `get_refund` |
| **PAYMENT** | 支付方式、支付问题 | `check_payment_methods`, `payment_issue` |
| **DELIVERY** | 物流选项、送达时间 | `delivery_options`, `delivery_period` |
| **SHIPPING** | 收货地址管理 | `change_shipping_address`, `set_up_shipping_address` |
| **INVOICE** | 发票查询与获取 | `check_invoice`, `get_invoice` |
| **ACCOUNT** | 账户创建、编辑、删除 | `create_account`, `edit_account`, `delete_account` |
| **CONTACT** | 联系客服 | `contact_customer_service`, `contact_human_agent` |
| **FEEDBACK** | 投诉与评价 | `complaint`, `review` |
| **CANCEL** | 取消费用查询 | `check_cancellation_fee` |
| **SUBSCRIPTION** | 订阅管理 | `newsletter_subscription` |

## 文件结构

```
dataset/
├── raw/                                    # 原始数据（从 HuggingFace 下载）
│   └── Bitext_customer_support.parquet     #   Bitext 原始 Parquet 文件（26,872 行）
├── README.md                               # 本文件（数据集文档）
├── test_comprehensive.html                 # 测试用 HTML 文档
├── papers_interpretation.html               # 论文解读 HTML
├── user_dict.txt                           # jieba 自定义词典
└── user_dict.example.txt                   # 词典示例

# 预处理与评测代码已迁移至 tests/experiment/：
tests/experiment/
├── preprocess.py                           # 数据集预处理脚本（生成知识库块 + 评测用例集）
├── run_eval.py                             # 端到端评测脚本（导入知识库→运行31条用例→输出报告）
├── kb_blocks.json                          # [运行时生成] 知识库块 JSON（1,341 块）
├── eval_cases.json                         # [运行时生成] 评测用例集（31 条）
└── eval_results.json                       # [运行时生成] 评测结果明细
```

## 字段定义

### `kb_blocks.json`（知识库块）

每条知识块的字段与 `rag/schema.py` 的 `DocBlock` 兼容：

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | string | 知识块正文（"Customer Question: ... \nSupport Answer: ..."） |
| `title` | string | 标题（"{Category} - {Intent}"） |
| `page_name` | string | 页面名（"Customer Support - {Category}"） |
| `page_url` | string | 来源标识（"bitext://customer-support/{category}/{intent}"） |
| `summary` | string | 摘要（response 前 100 字符） |
| `question` | string | 原始用户指令 |
| `category` | string | 客服类别（ORDER/REFUND/PAYMENT 等） |
| `intent` | string | 具体意图（cancel_order/track_refund 等） |
| `block_path` | string | 块路径（"bitext>{category}>{intent}"） |
| `time` | string | 时间戳（空） |

### `eval_cases.json`（评测用例）

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | string | 用例 ID（"basic_ORDER_cancel_order" 等） |
| `type` | string | 类型：`basic`（基础）/ `boundary`（边界）/ `multiturn`（多轮） |
| `query` | string | 用户查询 |
| `dialogue` | array | 多轮场景的前序对话（仅 `multiturn` 类型有） |
| `expected_category` | string | 预期命中的类别 |
| `expected_intent` | string | 预期命中的意图 |
| `reference_answer` | string | 参考回答 |
| `description` | string | 用例描述 |

## 使用方法

### 1. 导入知识库到 RAG 系统

```python
import json
from rag.indexing import indexer

blocks = json.load(open("dataset/kb_blocks.json", encoding="utf-8"))
meta = indexer.ingest_blocks(blocks, filename="bitext_customer_support.json")
print(f"导入完成: {meta}")
```

或通过 API：

```bash
# 启动 RAG 服务后
python3 -c "
import json, requests
blocks = json.load(open('dataset/kb_blocks.json', encoding='utf-8'))
resp = requests.post('http://localhost:8090/api/documents/ingest_blocks',
    json={'blocks': blocks[:200], 'filename': 'bitext_sample.json'})
print(resp.json())
"
```

### 2. 运行评测

```python
import json
from rag.pipeline import retrieve

cases = json.load(open("dataset/eval_cases.json", encoding="utf-8"))
for case in cases:
    results = retrieve(case["query"], top_k=5)
    hit = any(case["expected_category"] in r.get("page_name", "") for r in results)
    print(f"{'✅' if hit else '❌'} {case['id']}: {case['query'][:50]}")
```

### 3. 重新生成

```bash
# 从原始数据重新生成知识库块和评测用例
python dataset/preprocess.py
```

## 许可证与引用

### 许可证

原始数据集 © Bitext Innovations, 2024。本数据集用于研究与测试用途。
预处理生成的 `kb_blocks.json` 和 `eval_cases.json` 派生自上述原始数据，
继承相同的使用限制。

### 引用

```bibtex
@misc{bitext2024customer,
  title={Bitext Customer Support LLM Chatbot Training Dataset},
  author={Bitext Innovations},
  year={2024},
  howpublished={\url{https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset}},
}
```
