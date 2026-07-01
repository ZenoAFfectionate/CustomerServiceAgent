# -*- coding: utf-8 -*-
"""
Reranker 模型监督微调（SFT）模块。

使用 sentence-transformers CrossEncoder 对 reranker 模型进行微调，
基于多级相关性标注（0/1/2）的训练数据进行优化。

数据格式（reranker_qa_pointwise.jsonl）：
    {"query": "...", "doc": "...", "label": 0/1/2}
    label 会被归一化到 [0, 1]（0→0.0, 1→0.5, 2→1.0），
    以适配 CrossEncoder(num_labels=1) 默认的 BCEWithLogitsLoss。

与 DPO 的对比：
- SFT 适合从零开始微调，学习绝对相关性分数
- DPO 适合在 SFT 基础上做偏好对齐，学习相对排序
- 推荐流程：SFT 预训练 → DPO 精调

用法：
    PYTHONPATH=. python -m model.trainer.reranker_ft
"""
import os
import sys
import json

import torch
import wandb
from sklearn.model_selection import train_test_split
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, set_seed
from tqdm.auto import tqdm

# 添加项目根目录到 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from config.config_loader import CONFIG, logger, PROJECT_ROOT


# ======================== 训练参数 ========================
MODEL_NAME = CONFIG.get("rerank_model", "Qwen/Qwen3-Reranker-4B")
DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "reranker_qa_pointwise.jsonl")
SAVE_DIR = os.path.join(PROJECT_ROOT, "model", "trained_reranker")
PROJECT_NAME = "reranker-qwen3"
RUN_NAME = "qwen3-reranker-ft"
MAX_LENGTH = 512
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 2e-5
EARLY_STOP_PATIENCE = 1
USE_AMP = True
# 相关性标签最大值，用于把 0/1/2 归一化到 [0,1]
MAX_LABEL = 2.0


# ======================== 数据预处理 ========================

def truncate_to_max_length(tokenizer, query, doc, max_length=512):
    """截断 query+doc 到最大长度。"""
    tokens = tokenizer.encode_plus(query, doc, truncation=True, max_length=max_length)
    decoded = tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)
    parts = decoded.split(tokenizer.sep_token) if tokenizer.sep_token else [decoded]
    return parts[0], parts[1] if len(parts) > 1 else ""


def make_collate_fn(tokenizer, max_length):
    """构造 collate 函数（闭包捕获 tokenizer，避免全局依赖）。"""
    def _collate(batch):
        queries = [item.texts[0] for item in batch]
        docs = [item.texts[1] for item in batch]
        labels = torch.tensor([item.label for item in batch], dtype=torch.float)
        encoded = tokenizer(queries, docs, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
        encoded["labels"] = labels
        return encoded
    return _collate


def load_dataset(tokenizer, data_path, max_length):
    """加载 pointwise 数据，标签归一化到 [0,1]。"""
    if not os.path.isfile(data_path):
        raise FileNotFoundError(
            f"训练数据不存在: {data_path}\n"
            f"请先运行 `python -m model.utils.build_dataset` 生成 reranker_qa_pointwise.jsonl"
        )
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            q, d = truncate_to_max_length(tokenizer, item["query"], item["doc"], max_length)
            # 多级标签 0/1/2 → 归一化到 [0,1]，适配 BCEWithLogitsLoss
            norm_label = float(item["label"]) / MAX_LABEL
            data.append(InputExample(texts=[q, d], label=norm_label))
    logger.info(f"📊 加载训练样本: {len(data)} 条（标签已归一化到 [0,1]）")
    return data


def evaluate_on_val(model, val_data):
    """在验证集上评估模型准确率。"""
    model.model.eval()
    scorer = CrossEncoderClassificationEvaluator(
        sentence_pairs=[[ex.texts[0], ex.texts[1]] for ex in val_data],
        labels=[ex.label for ex in val_data],
        name="dev-set-eval"
    )
    return scorer(model)


# ======================== 主流程 ========================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate_fn = make_collate_fn(tokenizer, MAX_LENGTH)

    data = load_dataset(tokenizer, DATA_PATH, MAX_LENGTH)
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = CrossEncoder(model_name_or_path=MODEL_NAME, num_labels=1, max_length=MAX_LENGTH, device=device)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    best_acc = 0.0
    patience = 0

    for epoch in range(EPOCHS):
        logger.info(f"🔁 Epoch {epoch + 1}/{EPOCHS}")
        model.model.train()

        progress_bar = tqdm(train_dataloader, total=len(train_dataloader))
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                outputs = model.model(**batch)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            if step % 10 == 0:
                wandb.log({"train_loss": loss.item(), "epoch": epoch})

        # 验证 + 早停
        acc = evaluate_on_val(model, val_data)
        val_acc = acc["dev-set-eval_accuracy"]
        wandb.log({"val_acc": val_acc, "epoch": epoch})
        logger.info(f"✅ Epoch {epoch + 1} 验证准确率: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0
            best_dir = os.path.join(SAVE_DIR, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            model.save(best_dir)
            logger.info(f"📌 保存最佳模型 (val acc = {val_acc:.4f})")
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                logger.info("⏹️ 早停触发，结束训练")
                break

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save(SAVE_DIR)
    logger.info(f"✅ 最终模型保存至: {SAVE_DIR}")


if __name__ == "__main__":
    main()
