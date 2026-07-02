#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[Optimized] 电商客服 RAG 评测数据集预处理脚本。

将 Bitext Customer Support 数据集转换为 RAG 系统可直接导入的知识库格式。

输入：dataset/raw/Bitext_customer_support.parquet（26,872 条 QA 对）
输出：
  - dataset/kb_blocks.json     — 知识块 JSON（供 rag/indexing/indexer.ingest_blocks 导入）
  - dataset/eval_cases.json    — 评测用例集（20+ 条典型 QA，覆盖各场景）

处理逻辑：
  1. 按 category + intent 分组，每组取代表性的 instruction → response 构建知识块
  2. 去重（同 intent 下 instruction 文本相似度高的仅保留一条）
  3. 每个知识块附带 category/intent 标签，便于后续按场景分析召回率
  4. 从各 category 中抽取典型 case 构建评测集（基础/边界/多轮场景）

用法：
  python dataset/preprocess.py
  python dataset/preprocess.py --input dataset/raw/Bitext_customer_support.parquet --output-dir dataset/
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import pandas as pd


def build_kb_blocks(df: pd.DataFrame) -> list:
    """将 Bitext QA 数据转换为 RAG 知识库块格式。

    策略：按 (category, intent) 分组，每组取 instruction 作为知识块文本，
    response 作为补充说明，确保每个客服场景都有知识覆盖。
    """
    blocks = []
    seen_texts = set()  # 去重：相同 instruction 文本只保留一条

    grouped = df.groupby(["category", "intent"])
    for (category, intent), group in grouped:
        # 每组最多取 50 条（避免某些大组如 ACCOUNT 5986 条主导知识库）
        samples = group.head(50)
        for _, row in samples.iterrows():
            instruction = str(row["instruction"]).strip()
            response = str(row["response"]).strip()
            if not instruction or len(instruction) < 5:
                continue

            # 去重：规范化后比较
            normalized = instruction.lower().replace("{{order number}}", "").replace("{{", "").replace("}}", "").strip()
            if normalized in seen_texts:
                continue
            seen_texts.add(normalized)

            # 构建知识块文本：instruction + response 拼接
            text = f"Customer Question: {instruction}\nSupport Answer: {response}"

            block = {
                "text": text,
                "title": f"{category} - {intent}",
                "page_name": f"Customer Support - {category}",
                "page_url": f"bitext://customer-support/{category}/{intent}",
                "summary": response[:100] + "..." if len(response) > 100 else response,
                "question": instruction,
                "category": category,
                "intent": intent,
                "block_path": f"bitext>{category}>{intent}",
                "time": "",
            }
            blocks.append(block)

    return blocks


def build_eval_cases(df: pd.DataFrame, n_per_category: int = 3) -> list:
    """从各 category 抽取典型评测用例，覆盖基础/边界/多轮场景。

    选取策略：
    - 每个 category 取 n_per_category 条不同的 intent（确保场景多样性）
    - 构造多轮对话场景（同 category 下连续提问）
    - 包含边界 case（instruction 极短/极长、含模板占位符等）
    """
    eval_cases = []

    # 1. 基础场景：每 category 取不同 intent 的典型 case
    grouped = df.groupby("category")
    for category, group in grouped:
        intent_grouped = group.groupby("intent")
        intents = list(intent_grouped.groups.keys())
        for i, intent in enumerate(intents[:n_per_category]):
            row = intent_grouped.get_group(intent).iloc[0]
            eval_cases.append({
                "id": f"basic_{category}_{intent}",
                "type": "basic",
                "query": str(row["instruction"]),
                "expected_category": category,
                "expected_intent": intent,
                "reference_answer": str(row["response"]),
                "description": f"基础场景：{category}/{intent}",
            })

    # 2. 边界场景：极短 query
    shortest = df.loc[df["instruction"].str.len().nsmallest(3).index]
    for _, row in shortest.iterrows():
        eval_cases.append({
            "id": f"boundary_short_{row['category']}_{row['intent']}",
            "type": "boundary",
            "query": str(row["instruction"]),
            "expected_category": str(row["category"]),
            "expected_intent": str(row["intent"]),
            "reference_answer": str(row["response"]),
            "description": "边界场景：极短 query（验证检索鲁棒性）",
        })

    # 3. 多轮场景：模拟用户连续追问
    multi_turn_categories = ["ORDER", "REFUND", "DELIVERY", "PAYMENT", "ACCOUNT"]
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


def main():
    parser = argparse.ArgumentParser(description="预处理电商客服数据集为 RAG 知识库格式")
    parser.add_argument(
        "--input", default="dataset/raw/Bitext_customer_support.parquet",
        help="Bitext Parquet 文件路径",
    )
    parser.add_argument(
        "--output-dir", default="dataset/",
        help="输出目录",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        print("   请先下载：python dataset/preprocess.py --download")
        sys.exit(1)

    print(f"📖 读取数据集: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"   总行数: {len(df)}")
    print(f"   类别数: {df['category'].nunique()}")
    print(f"   意图数: {df['intent'].nunique()}")

    # 构建知识库块
    print("\n🔧 构建知识库块...")
    blocks = build_kb_blocks(df)
    print(f"   知识块数: {len(blocks)}（去重后）")

    kb_path = os.path.join(args.output_dir, "kb_blocks.json")
    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, ensure_ascii=False, indent=2)
    print(f"   ✅ 已写入: {kb_path}")

    # 构建评测用例
    print("\n🔧 构建评测用例集...")
    eval_cases = build_eval_cases(df)
    eval_path = os.path.join(args.output_dir, "eval_cases.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_cases, f, ensure_ascii=False, indent=2)
    print(f"   评测用例数: {len(eval_cases)}")
    print(f"     - 基础场景: {len([c for c in eval_cases if c['type'] == 'basic'])}")
    print(f"     - 边界场景: {len([c for c in eval_cases if c['type'] == 'boundary'])}")
    print(f"     - 多轮场景: {len([c for c in eval_cases if c['type'] == 'multiturn'])}")
    print(f"   ✅ 已写入: {eval_path}")

    # 输出统计摘要
    print("\n" + "=" * 60)
    print("📊 数据集预处理完成")
    print(f"   原始数据: {len(df)} 行")
    print(f"   知识库块: {len(blocks)} 块（写入 {kb_path}）")
    print(f"   评测用例: {len(eval_cases)} 条（写入 {eval_path}）")
    print(f"   覆盖类别: {df['category'].nunique()} 个")
    print(f"   覆盖意图: {df['intent'].nunique()} 个")
    print("=" * 60)
    print("\n💡 下一步：")
    print("   1. 导入知识库：")
    print("      python -c \"")
    print("        import json")
    print("        from rag.indexing import indexer")
    print("        blocks = json.load(open('dataset/kb_blocks.json', encoding='utf-8'))")
    print("        indexer.ingest_blocks(blocks, filename='bitext_customer_support.json')")
    print("      \"")
    print("   2. 运行评测：")
    print("      python dataset/preprocess.py --eval")


if __name__ == "__main__":
    main()
