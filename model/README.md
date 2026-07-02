# 🤖 model — 模型训练与推理

> Embedding / Reranker 模型的部署、训练和数据构造。

## 目录结构

```
model/
├── inference/                # 模型推理与部署
│   ├── tei_client.py         #   TEI 客户端（Embedding + Reranker 接口）
│   └── docker-compose-tei.yml #  TEI Docker Compose 部署配置
├── trainer/                  # 模型微调
│   └── reranker_ft.py        #   Reranker CrossEncoder 微调
├── utils/                    # 训练数据工具
│   └── build_dataset.py      #   Reranker 训练数据集构造（DeepSeek + 知识库采样）
└── README.md
```

## 推理框架选型：TEI

### 为什么选择 TEI？

| 框架 | Embedding | Reranker | Qwen3 支持 | 性能 | 成熟度 | 结论 |
|------|-----------|----------|-----------|------|--------|------|
| **TEI** | ✅ 原生 | ✅ 原生 | ✅ 明确支持 | ⭐⭐⭐⭐⭐ | 4.9k stars, HuggingFace 官方 | **最佳选择** |
| vLLM | ⚠️ 次要 | ❌ 不支持 | ⚠️ 仅生成 | ⭐⭐⭐ | 适合 LLM 生成 | 不适合 |
| Infinity | ✅ | ⚠️ 有限 | ⚠️ 不确定 | ⭐⭐⭐⭐ | 较新 | 备选 |
| Xinference | ✅ | ✅ | ⚠️ 需验证 | ⭐⭐⭐ | 通用但重 | 过重 |

**TEI 优势：**
- HuggingFace 官方维护，Apache-2.0 开源
- 明确支持 Qwen3-Embedding 和 Qwen3-Reranker
- 同时提供 `/embed`（嵌入）和 `/rerank`（重排序）两个 API
- Flash Attention + 动态批处理，吞吐量远超 sentence-transformers
- 轻量 Docker 镜像，启动快，无需图编译
- 内置 OpenTelemetry + Prometheus 监控

### 模型选型

| 模型 | 用途 | 参数量 | MTEB 多语言 | 嵌入维度 | 推荐场景 |
|------|------|--------|------------|---------|---------|
| Qwen3-Embedding-0.6B | 文本嵌入 | 0.6B | 64.33 | 1024 | 快速验证 / 资源受限 |
| **Qwen3-Embedding-4B** | 文本嵌入 | 4B | **69.45** | 2560 | **生产推荐（精度/速度平衡）** |
| Qwen3-Embedding-8B | 文本嵌入 | 8B | 70.58（榜单第一） | 4096 | 极致精度 |
| Qwen3-Reranker-0.6B | 重排序 | 0.6B | — | — | 快速验证 / 资源受限 |
| **Qwen3-Reranker-4B** | 重排序 | 4B | — | — | **生产推荐（精度/速度平衡）** |
| Qwen3-Reranker-8B | 重排序 | 8B | — | — | 极致精度 |

> 所有版本均支持 32K 上下文、100+ 语言、TEI 部署。8B 需 ~16GB VRAM，4B 需 ~8GB VRAM。

## 快速开始

### 1. 部署 TEI 服务

```bash
# 方式一：一键启动脚本（推荐，含环境检查 + 健康检查）
bash scripts/start_tei.sh

# 方式二：手动 Docker Compose
cd model/inference
docker compose -f docker-compose-tei.yml up -d

# 或单独部署 Embedding 模型
docker run --gpus all -p 8080:80 -v $PWD/data:/data \
    ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \
    --model-id Qwen/Qwen3-Embedding-4B

# 单独部署 Reranker 模型
docker run --gpus all -p 8081:80 -v $PWD/data:/data \
    ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \
    --model-id Qwen/Qwen3-Reranker-4B
```

### 2. 验证服务

```bash
# 健康检查
curl http://localhost:8080/health
curl http://localhost:8081/health

# 测试 Embedding
curl http://localhost:8080/embed \
    -X POST -H 'Content-Type: application/json' \
    -d '{"inputs": "你好世界"}'

# 测试 Reranker
curl http://localhost:8081/rerank \
    -X POST -H 'Content-Type: application/json' \
    -d '{"query": "什么是广告投放", "texts": ["广告投放是核心环节", "数据分析很重要"]}'
```

### 3. 使用 Python 客户端

```python
from model.inference.tei_client import TEIClient

client = TEIClient(
    embed_url="http://localhost:8080",
    rerank_url="http://localhost:8081",
)

# 获取嵌入向量
embedding = client.embed("你好世界")

# 批量嵌入
embeddings = client.embed_batch(["文本1", "文本2", "文本3"])

# 重排序
results = client.rerank("什么是广告投放", ["广告投放是核心环节", "数据分析很重要"])
# results: [{"index": 0, "score": 0.95}, {"index": 1, "score": 0.12}]
```

## 模型微调

### 1. 构造训练数据

```bash
# 需要先部署知识库（Milvus + ES），并设置 DEEPSEEK_API_KEY
PYTHONPATH=. python -m model.utils.build_dataset \
    --milvus-host 127.0.0.1 \
    --collection-name htmlrag_dev \
    --output-path dataset/reranker_qa_dataset.jsonl
```

### 2. 微调 Reranker

```bash
PYTHONPATH=. python -m model.trainer.reranker_ft
```

微调产物保存在 `model/trained_reranker/`，可直接用于 TEI 部署：

```bash
docker run --gpus all -p 8081:80 -v $PWD/model/trained_reranker/best_model:/data/model \
    ghcr.io/huggingface/text-embeddings-inference:cuda-1.9 \
    --model-id /data/model
```

## 环境变量

在 `.env` 中配置以下变量：

```bash
# TEI 服务地址
TEI_EMBED_URL=http://localhost:8080
TEI_RERANK_URL=http://localhost:8081

# DeepSeek API（训练数据构造用）
DEEPSEEK_API_KEY=sk-your-key-here
DEEPSEEK_BASE_URL=https://api.deepseek.com
```
