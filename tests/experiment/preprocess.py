#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据集预处理：将 Bitext Parquet 全量转换为 RAG 知识库块 + 全量评测用例集。

从 dataset/raw/Bitext_customer_support.parquet（26,872 条 QA 对）生成：
  - kb_blocks.json    — 全量知识块 JSON（去重后约 ~24,000 块，供 rag/indexing/indexer.ingest_blocks 导入）
  - eval_cases.json   — 全量评测用例集（每条原始数据均作为评测 case，覆盖全部 11 类别 27 意图）

设计原则：
  - 知识库与评测集分离：知识库包含全部去重数据，评测集同样覆盖全量
  - 按类别标注：每条 case 携带 expected_category + expected_intent，便于按场景统计召回率
  - 场景分层：basic（每条 instruction 逐一评测）/ boundary（极短/极长 query）/ multiturn（同类别连续追问）

用法：
  python tests/experiment/preprocess.py
  python tests/experiment/preprocess.py --input dataset/raw/Bitext_customer_support.parquet --output-dir tests/experiment/
"""
import argparse
import json
import os
import sys

import pandas as pd


def build_kb_blocks(df: pd.DataFrame) -> list:
    """将 Bitext 全量 QA 数据转换为 RAG 知识库块格式（去重）。

    策略：遍历全部 26,872 行，按 instruction 文本去重后构建知识块。
    不再限制每组 50 条上限，确保知识库覆盖全部数据。
    """
    blocks = []
    seen_texts = set()

    for idx, row in df.iterrows():
        instruction = str(row["instruction"]).strip()
        response = str(row["response"]).strip()
        category = str(row["category"])
        intent = str(row["intent"])

        if not instruction or len(instruction) < 3:
            continue

        # 去重：规范化后比较（去除模板占位符与大小写差异）
        normalized = instruction.lower()
        for placeholder in ["{{order number}}", "{{account type}}", "{{invoice number}}",
                            "{{product name}}", "{{refund id}}", "{{payment method}}"]:
            normalized = normalized.replace(placeholder.lower(), "")
        normalized = normalized.replace("{{", "").replace("}}", "").strip()

        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)

        text = f"Customer Question: {instruction}\nSupport Answer: {response}"
        blocks.append({
            "text": text,
            "title": f"{category} - {intent}",
            "page_name": f"Customer Support - {category}",
            "page_url": f"bitext://customer-support/{category}/{intent}",
            "summary": response[:120] + "..." if len(response) > 120 else response,
            "question": instruction,
            "category": category,
            "intent": intent,
            "block_path": f"bitext>{category}>{intent}",
            "time": "",
        })

    return blocks


def build_eval_cases(df: pd.DataFrame) -> list:
    """构建全量评测用例集。

    覆盖三类场景：
    1. basic：每条 instruction 逐一作为评测 query（去重后），预期命中其所属 category
    2. boundary：极短 query（≤10 字符）与极长 query（≥80 字符），验证检索鲁棒性
    3. multiturn：每个 category 取 2 条不同 intent 构造连续追问场景
    """
    eval_cases = []
    seen_queries = set()

    # ---------- 1. 全量基础场景 ----------
    for idx, row in df.iterrows():
        instruction = str(row["instruction"]).strip()
        if not instruction or len(instruction) < 3:
            continue

        # 去重：避免同一条 instruction 被评测多次
        query_key = instruction.lower().strip()
        if query_key in seen_queries:
            continue
        seen_queries.add(query_key)

        eval_cases.append({
            "id": f"basic_{len(eval_cases):05d}_{row['category']}_{row['intent']}",
            "type": "basic",
            "query": instruction,
            "expected_category": str(row["category"]),
            "expected_intent": str(row["intent"]),
            "reference_answer": str(row["response"]),
            "description": f"基础场景：{row['category']}/{row['intent']}",
        })

    # ---------- 2. 边界场景 ----------
    # 2a. 极短 query（≤10 字符）
    short_queries = df[df["instruction"].str.len() <= 10].drop_duplicates(subset=["instruction"])
    for _, row in short_queries.head(20).iterrows():
        query = str(row["instruction"]).strip()
        if not query:
            continue
        eval_cases.append({
            "id": f"boundary_short_{row['category']}_{row['intent']}_{len(eval_cases)}",
            "type": "boundary",
            "subtype": "short",
            "query": query,
            "expected_category": str(row["category"]),
            "expected_intent": str(row["intent"]),
            "reference_answer": str(row["response"]),
            "description": f"边界场景（极短 query ≤10 字符）：{row['category']}/{row['intent']}",
        })

    # 2b. 极长 query（≥80 字符）
    long_queries = df[df["instruction"].str.len() >= 80].drop_duplicates(subset=["instruction"])
    for _, row in long_queries.head(20).iterrows():
        query = str(row["instruction"]).strip()
        eval_cases.append({
            "id": f"boundary_long_{row['category']}_{row['intent']}_{len(eval_cases)}",
            "type": "boundary",
            "subtype": "long",
            "query": query,
            "expected_category": str(row["category"]),
            "expected_intent": str(row["intent"]),
            "reference_answer": str(row["response"]),
            "description": f"边界场景（极长 query ≥80 字符）：{row['category']}/{row['intent']}",
        })

    # ---------- 3. 多轮场景 ----------
    # 每个 category 取 2 条不同 intent 构造连续追问
    multi_turn_categories = ["ORDER", "REFUND", "DELIVERY", "PAYMENT", "ACCOUNT",
                             "SHIPPING", "INVOICE", "CANCEL", "FEEDBACK", "CONTACT", "SUBSCRIPTION"]
    for category in multi_turn_categories:
        group = df[df["category"] == category]
        intents = group["intent"].unique()
        if len(intents) >= 2:
            rows = []
            for intent in intents[:2]:
                intent_rows = group[group["intent"] == intent]
                if len(intent_rows) > 0:
                    rows.append(intent_rows.iloc[0])
            if len(rows) >= 2:
                eval_cases.append({
                    "id": f"multiturn_{category}",
                    "type": "multiturn",
                    "dialogue": [
                        {"role": "user", "content": str(rows[0]["instruction"])},
                        {"role": "assistant", "content": str(rows[0]["response"])},
                    ],
                    "query": str(rows[1]["instruction"]),
                    "expected_category": category,
                    "expected_intent": str(rows[1]["intent"]),
                    "reference_answer": str(rows[1]["response"]),
                    "description": f"多轮场景：{category} 连续追问（测试 query 重写能力）",
                })

    return eval_cases


def print_dataset_stats(df: pd.DataFrame, blocks: list, eval_cases: list):
    """打印数据集详细统计信息。"""
    print(f"\n{'=' * 70}")
    print(f"📊 数据集统计")
    print(f"{'=' * 70}")
    print(f"  原始数据:   {len(df):,} 行")
    print(f"  知识库块:   {len(blocks):,} 块（去重后）")
    print(f"  评测用例:   {len(eval_cases):,} 条")
    print(f"    - basic:     {len([c for c in eval_cases if c['type'] == 'basic']):,}")
    print(f"    - boundary:  {len([c for c in eval_cases if c['type'] == 'boundary']):,}")
    print(f"      - short:   {len([c for c in eval_cases if c.get('subtype') == 'short']):,}")
    print(f"      - long:    {len([c for c in eval_cases if c.get('subtype') == 'long']):,}")
    print(f"    - multiturn: {len([c for c in eval_cases if c['type'] == 'multiturn']):,}")
    print()
    print(f"  类别分布:")
    for cat, count in df["category"].value_counts().items():
        block_count = len([b for b in blocks if b["category"] == cat])
        print(f"    {cat:15s}: {count:6,} 原始 → {block_count:5,} 知识块")
    print()
    print(f"  意图分布:")
    for intent, count in df["intent"].value_counts().items():
        print(f"    {intent:30s}: {count:6,}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="预处理 Bitext 数据集为 RAG 知识库格式（全量）")
    parser.add_argument(
        "--input", default="dataset/raw/Bitext_customer_support.parquet",
        help="Bitext Parquet 文件路径",
    )
    parser.add_argument(
        "--output-dir", default="tests/experiment/",
        help="输出目录",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        print("   请先从 HuggingFace 下载：")
        print("   https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset")
        sys.exit(1)

    print(f"📖 读取数据集: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"   总行数: {len(df):,}  类别数: {df['category'].nunique()}  意图数: {df['intent'].nunique()}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n🔧 构建知识库块（全量去重）...")
    blocks = build_kb_blocks(df)
    kb_path = os.path.join(args.output_dir, "kb_blocks.json")
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, ensure_ascii=False, indent=2)
    print(f"   ✅ {len(blocks):,} 块 → {kb_path}")

    print("\n🔧 构建评测用例集（全量）...")
    eval_cases = build_eval_cases(df)
    eval_path = os.path.join(args.output_dir, "eval_cases.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_cases, f, ensure_ascii=False, indent=2)
    print(f"   ✅ {len(eval_cases):,} 条 → {eval_path}")

    print_dataset_stats(df, blocks, eval_cases)


if __name__ == "__main__":
    main()
