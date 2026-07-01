# -*- coding: utf-8 -*-
"""
Reranker 模型 DPO (Direct Preference Optimization) 训练模块。

使用偏好数据（chosen vs rejected）对 Reranker 模型进行偏好优化，
让模型学会在排序时偏好与 query 更相关的文档。

DPO 优势（相比 SFT）：
- 直接利用偏好数据，无需显式奖励模型
- 通过对比 chosen/rejected 的 log-prob 差值优化，学到相对排序而非绝对分数
- 比 pairwise margin loss 更稳定，收敛更快

数据格式（reranker_qa_dpo.jsonl）：
    {"query": "...", "chosen": "doc text...", "rejected": "doc text..."}

用法：
    PYTHONPATH=. python -m model.trainer.reranker_dpo
"""
import os
import sys
import json
import torch
import torch.nn.functional as F
import wandb
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed
from tqdm.auto import tqdm

# 添加项目根目录到 sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from config.config_loader import CONFIG, logger, PROJECT_ROOT


# ======================== 训练参数 ========================
MODEL_NAME = CONFIG.get("rerank_model", "Qwen/Qwen3-Reranker-4B")
DATA_PATH = os.path.join(PROJECT_ROOT, "dataset", "reranker_qa_dpo.jsonl")
SAVE_DIR = os.path.join(PROJECT_ROOT, "model", "trained_reranker_dpo")
PROJECT_NAME = "reranker-dpo"
RUN_NAME = "qwen3-reranker-dpo"
MAX_LENGTH = 512
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 5e-6
BETA = 0.1               # DPO 温度参数（β 越小偏好越强）
EARLY_STOP_PATIENCE = 2
USE_AMP = True
GRADIENT_CLIP = 1.0


# ======================== 环境初始化 ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================== 数据集 ========================

class PreferenceDataset(Dataset):
    """DPO 偏好数据集。

    每个样本包含 (query, chosen_doc, rejected_doc)，
    模型需要学会给 chosen_doc 打更高分。
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.data.append({
                    "query": item["query"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                })

        logger.info(f"📊 加载偏好数据: {len(self.data)} 条")

    def __len__(self):
        return len(self.data)

    def _encode_pair(self, query: str, doc: str) -> Dict:
        """编码 (query, doc) 对为模型输入。"""
        encoded = self.tokenizer(
            query, doc,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoded.items()}

    def __getitem__(self, idx):
        item = self.data[idx]
        chosen_inputs = self._encode_pair(item["query"], item["chosen"])
        rejected_inputs = self._encode_pair(item["query"], item["rejected"])
        return {
            "chosen": chosen_inputs,
            "rejected": rejected_inputs,
        }


def dpo_collate_fn(batch):
    """DPO batch collate 函数。"""
    chosen_batch = {}
    rejected_batch = {}
    for key in batch[0]["chosen"]:
        chosen_batch[key] = torch.stack([b["chosen"][key] for b in batch])
        rejected_batch[key] = torch.stack([b["rejected"][key] for b in batch])
    return {"chosen": chosen_batch, "rejected": rejected_batch}


# ======================== DPO Loss ========================

def compute_dpo_loss(
    policy_chosen_scores: torch.Tensor,
    policy_rejected_scores: torch.Tensor,
    reference_chosen_scores: torch.Tensor,
    reference_rejected_scores: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """计算 DPO 损失。

    L_DPO = -E[log σ(β · (Δπ_chosen - Δπ_rejected))]

    其中 Δπ = policy_score - reference_score

    Args:
        policy_chosen_scores: 策略模型对 chosen 的打分 [batch]
        policy_rejected_scores: 策略模型对 rejected 的打分 [batch]
        reference_chosen_scores: 参考模型对 chosen 的打分 [batch]
        reference_rejected_scores: 参考模型对 rejected 的打分 [batch]
        beta: DPO 温度参数

    Returns:
        DPO 损失值
    """
    # 计算偏好差值
    policy_log_ratio = policy_chosen_scores - policy_rejected_scores
    reference_log_ratio = reference_chosen_scores - reference_rejected_scores

    # DPO 损失
    logits = beta * (policy_log_ratio - reference_log_ratio)
    loss = -F.logsigmoid(logits).mean()

    # 计算准确率（chosen 是否确实得分更高）
    with torch.no_grad():
        acc = (policy_chosen_scores > policy_rejected_scores).float().mean()

    return loss, acc


# ======================== 训练辅助函数 ========================

def evaluate_dpo(model, ref_model, val_loader):
    """在验证集上评估 DPO 效果。"""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            chosen_scores = model(**{k: v.to(device) for k, v in batch["chosen"].items()}).logits.squeeze(-1)
            rejected_scores = model(**{k: v.to(device) for k, v in batch["rejected"].items()}).logits.squeeze(-1)

            ref_chosen = ref_model(**{k: v.to(device) for k, v in batch["chosen"].items()}).logits.squeeze(-1)
            ref_rejected = ref_model(**{k: v.to(device) for k, v in batch["rejected"].items()}).logits.squeeze(-1)

            loss, acc = compute_dpo_loss(chosen_scores, rejected_scores, ref_chosen, ref_rejected, BETA)
            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1

    return {
        "loss": total_loss / max(num_batches, 1),
        "acc": total_acc / max(num_batches, 1),
    }


# ======================== 主流程 ========================

def main():
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    logger.info(f"📦 加载策略模型: {MODEL_NAME}")
    policy_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, torch_dtype=torch.float16
    ).to(device)

    logger.info(f"📦 加载参考模型（冻结）: {MODEL_NAME}")
    reference_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=1, torch_dtype=torch.float16
    ).to(device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and device.type == "cuda")

    # 数据加载
    full_dataset = PreferenceDataset(DATA_PATH, tokenizer, MAX_LENGTH)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, collate_fn=dpo_collate_fn)
    val_dataloader = DataLoader(val_data, shuffle=False, batch_size=BATCH_SIZE, collate_fn=dpo_collate_fn)

    wandb.init(project=PROJECT_NAME, name=RUN_NAME, config={
        "model": MODEL_NAME, "beta": BETA, "lr": LEARNING_RATE,
        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "max_length": MAX_LENGTH,
    })

    best_acc = 0.0
    patience = 0

    for epoch in range(EPOCHS):
        policy_model.train()
        total_loss = 0.0
        total_acc = 0.0

        progress_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for step, batch in enumerate(progress_bar):
            chosen_inputs = {k: v.to(device) for k, v in batch["chosen"].items()}
            rejected_inputs = {k: v.to(device) for k, v in batch["rejected"].items()}

            with torch.amp.autocast("cuda", enabled=USE_AMP and device.type == "cuda"):
                policy_chosen = policy_model(**chosen_inputs).logits.squeeze(-1)
                policy_rejected = policy_model(**rejected_inputs).logits.squeeze(-1)

                # 参考模型打分（不计算梯度）
                with torch.no_grad():
                    ref_chosen = reference_model(**chosen_inputs).logits.squeeze(-1)
                    ref_rejected = reference_model(**rejected_inputs).logits.squeeze(-1)

                loss, acc = compute_dpo_loss(
                    policy_chosen, policy_rejected,
                    ref_chosen, ref_rejected,
                    beta=BETA,
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_acc += acc.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc.item():.3f}"})

            if step % 10 == 0:
                wandb.log({"train_loss": loss.item(), "train_acc": acc.item(), "epoch": epoch})

        # 验证
        val_metrics = evaluate_dpo(policy_model, reference_model, val_dataloader)
        avg_train_loss = total_loss / len(train_dataloader)
        avg_train_acc = total_acc / len(train_dataloader)

        wandb.log({
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["acc"],
            "avg_train_loss": avg_train_loss, "avg_train_acc": avg_train_acc,
            "epoch": epoch,
        })

        logger.info(f"✅ Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, train_acc={avg_train_acc:.3f}, "
                    f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['acc']:.3f}")

        # 早停
        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]
            patience = 0
            best_dir = os.path.join(SAVE_DIR, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            policy_model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            logger.info(f"📌 保存最佳模型 (val acc = {best_acc:.4f})")
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                logger.info("⏹️ 早停触发，结束训练")
                break

    os.makedirs(SAVE_DIR, exist_ok=True)
    policy_model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    logger.info(f"✅ 最终 DPO 模型保存至: {SAVE_DIR}")


if __name__ == "__main__":
    main()
