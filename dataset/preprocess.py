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
  python dataset/preprocess.py --eval           # 对已生成的评测用例运行检索命中率评测（见 run_retrieval_eval）
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd

# 支持以脚本方式直接运行（python dataset/preprocess.py），保证 `--eval`
# 依赖的 `from rag import pipeline`（延迟导入）能找到项目根目录下的 rag/ 包
# ——脚本运行时 sys.path[0] 默认是脚本所在目录（dataset/），而非项目根目录。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


_PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}")


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

            # 去重：规范化后比较。【修复 L18】此前仅 replace("{{order number}}", "")
            # 再简单剥离 "{{"/"}}"，其他占位符（如 "{{product id}}"）移除花括号
            # 后会残留 "product id" 字面文本，导致"仅占位符不同"的近重复
            # instruction（如 "查询{{order number}}物流" vs "查询{{product id}}物流"）
            # 未被识别为重复而各自入库。改用正则统一移除全部 `{{...}}` 占位符。
            normalized = _PLACEHOLDER_RE.sub("", instruction.lower()).strip()
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
    # 【修复 L18】此前直接对原始 df 取每组第一行 `str(row["instruction"])`；
    # 若该行 instruction 恰好为 NaN（数据集缺陷），`str(nan)` 会生成字面量
    # 字符串 "nan" 作为评测 query，污染评测集且不易被发现。这里先过滤掉
    # instruction/response 为空的行，保证后续所有采样来源都是有效数据。
    df = df.dropna(subset=["instruction", "response"])
    df = df[df["instruction"].astype(str).str.strip() != ""]

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


def run_retrieval_eval(eval_cases: list, top_k: int = 5) -> dict:
    """对评测用例集运行检索命中率评测（修复审查报告 L17：`--eval` 此前在
    帮助文本中被引导使用，但 `main()` 从未定义该参数，实际运行会被
    argparse 报 `unrecognized arguments` 拒绝，功能完全不可用）。

    调用 `rag.pipeline.retrieve()` 对每条用例检索，检查返回上下文块的
    `category`/`intent` 标签（`build_kb_blocks` 写入知识块时附带）是否命中
    用例的 `expected_category`/`expected_intent`，统计命中率。

    仅评测 "basic"/"boundary" 类型用例——"multiturn" 用例依赖多轮 query
    重写能力，单轮 `retrieve()` 无法公平评测，予以跳过。

    Args:
        eval_cases: `build_eval_cases()` 产出的评测用例列表（或从磁盘加载的等价 JSON）
        top_k: 检索 top_k

    Returns:
        {"total_cases": int, "hit_count": int, "hit_rate": float,
         "per_case": [{"id", "query", "hit"}, ...]}
    """
    from rag import pipeline

    evaluable = [c for c in eval_cases if c.get("type") in ("basic", "boundary")]
    per_case = []
    hit_count = 0
    for case in evaluable:
        contexts = pipeline.retrieve(case["query"], top_k=top_k)
        hit = any(
            c.get("category") == case.get("expected_category")
            and c.get("intent") == case.get("expected_intent")
            for c in contexts
        )
        hit_count += int(hit)
        per_case.append({"id": case.get("id"), "query": case["query"], "hit": hit})

    total = len(evaluable)
    return {
        "total_cases": total,
        "hit_count": hit_count,
        "hit_rate": round(hit_count / total, 4) if total else 0.0,
        "per_case": per_case,
    }


def _run_eval_mode(args) -> None:
    """`--eval` 模式：加载已生成的评测用例集并运行检索命中率评测。"""
    eval_path = args.eval_path or os.path.join(args.output_dir, "eval_cases.json")
    if not os.path.exists(eval_path):
        print(f"❌ 评测用例文件不存在: {eval_path}")
        print("   请先运行 `python dataset/preprocess.py` 生成 eval_cases.json，"
              "并将 kb_blocks.json 导入 RAG 知识库后再运行 --eval。")
        sys.exit(1)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_cases = json.load(f)
    print(f"📖 加载评测用例: {len(eval_cases)} 条（来自 {eval_path}）")

    report = run_retrieval_eval(eval_cases, top_k=args.eval_top_k)
    print("\n" + "=" * 60)
    print("📊 检索命中率评测结果")
    print(f"   评测用例数（basic/boundary）: {report['total_cases']}")
    print(f"   命中数: {report['hit_count']}")
    print(f"   命中率: {report['hit_rate']:.2%}")
    print("=" * 60)
    for item in report["per_case"]:
        mark = "✅" if item["hit"] else "❌"
        print(f"   {mark} [{item['id']}] {item['query'][:40]}")


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
    # 【修复 L17】补全 --eval/--eval-path/--eval-top-k 参数定义（此前帮助
    # 文本引导使用 --eval 但从未定义，见 run_retrieval_eval 的说明）。
    parser.add_argument(
        "--eval", action="store_true",
        help="对已生成的评测用例集运行检索命中率评测，而非重新构建知识库块",
    )
    parser.add_argument(
        "--eval-path", default=None,
        help="评测用例 JSON 路径，默认取 <output-dir>/eval_cases.json",
    )
    parser.add_argument("--eval-top-k", type=int, default=5, help="--eval 模式下检索使用的 top_k")
    args = parser.parse_args()

    if args.eval:
        _run_eval_mode(args)
        return

    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        # 【修复 L17】此前提示"请先下载：python dataset/preprocess.py --download"，
        # 但该参数从未被定义/实现，运行即报错。改为明确的手动获取指引，不再
        # 承诺一个不存在的自动下载功能。
        print("   请手动下载 Bitext 客服数据集（Parquet 格式，含 instruction/response/"
              "category/intent 字段）并放置到该路径，例如：")
        print("     - HuggingFace: bitext/Bitext-customer-support-llm-chatbot-training-dataset")
        print("     - 或任意提供同等字段的客服问答数据集")
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
    print("        from rag.knowledge_base import corpus_management")
    print("        blocks = json.load(open('dataset/kb_blocks.json', encoding='utf-8'))")
    print("        corpus_management.ingest_blocks(blocks, filename='bitext_customer_support.json')")
    print("      \"")
    print("   2. 运行评测：")
    print("      python dataset/preprocess.py --eval")


if __name__ == "__main__":
    main()
