#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""端到端 RAG 全量评测脚本：知识库导入 → 全量评测用例 → 详细指标统计。

完整流程：
  1. 预处理 Bitext 数据集（全量 ~24,000 知识块 + 全量评测用例）
  2. 将知识库导入 RAG 系统（本地降级后端）
  3. 逐条运行全量评测用例，统计召回率、延迟、分场景/分类别命中率
  4. 输出详细评测报告（含每类别召回率、延迟分布、未命中 case 分析）+ JSON 明细

用法：
  python tests/experiment/run_eval.py
  python tests/experiment/run_eval.py --skip-ingest       # 跳过知识库导入
  python tests/experiment/run_eval.py --max-cases 200     # 限制评测用例数（快速验证）
"""
import argparse
import json
import os
import shutil
import sys
import time
from collections import defaultdict

# 确保项目根目录在 sys.path 上
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

_EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATASET_RAW = os.path.join(_PROJECT_ROOT, "dataset", "raw", "Bitext_customer_support.parquet")


def ensure_preprocessed():
    """确保 kb_blocks.json 和 eval_cases.json 已生成。"""
    kb_path = os.path.join(_EXPERIMENT_DIR, "kb_blocks.json")
    eval_path = os.path.join(_EXPERIMENT_DIR, "eval_cases.json")

    if os.path.exists(kb_path) and os.path.exists(eval_path):
        return kb_path, eval_path

    if not os.path.exists(_DATASET_RAW):
        print(f"❌ 原始数据集不存在: {_DATASET_RAW}")
        print("   请先从 HuggingFace 下载 Bitext 数据集：")
        print("   https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset")
        sys.exit(1)

    print("📦 预处理数据集（全量）...")
    from tests.experiment.preprocess import build_kb_blocks, build_eval_cases
    import pandas as pd

    df = pd.read_parquet(_DATASET_RAW)
    blocks = build_kb_blocks(df)
    eval_cases = build_eval_cases(df)

    with open(kb_path, "w", encoding="utf-8") as f:
        json.dump(blocks, f, ensure_ascii=False, indent=2)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_cases, f, ensure_ascii=False, indent=2)

    print(f"   ✅ {len(blocks):,} 知识块 + {len(eval_cases):,} 评测用例")
    return kb_path, eval_path


def reset_rag_state():
    """重置 RAG 单例与本地数据，保证干净的测试环境。"""
    data_dir = os.path.join(_PROJECT_ROOT, "tests", "_rag_test_data")
    if os.path.isdir(data_dir):
        shutil.rmtree(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    os.environ["RAG_DATA_DIR"] = data_dir

    import rag.indexing.metadata as metadata_mod
    import rag.indexing.embedding as embedding_mod
    from rag.indexing.store import reset_keyword_store, reset_vector_store
    metadata_mod._default_registry = None
    embedding_mod._default_embedder = None
    reset_vector_store()
    reset_keyword_store()


def ingest_kb(kb_path: str) -> dict:
    """将知识库块导入 RAG 索引（自动分批，每批 5000 条）。"""
    from rag.indexing import index_builder as indexer

    with open(kb_path, encoding="utf-8") as f:
        blocks = json.load(f)

    total = len(blocks)
    batch_size = 4000  # 留余量低于 5000 限制
    print(f"📥 导入 {total:,} 个知识块（分 {(total - 1) // batch_size + 1} 批，每批 {batch_size}）...")

    t0 = time.time()
    num_chunks = 0
    for i in range(0, total, batch_size):
        batch = blocks[i:i + batch_size]
        batch_num = i // batch_size + 1
        meta = indexer.ingest_blocks(batch, filename=f"bitext_customer_support_part{batch_num}.json")
        num_chunks += meta["num_chunks"]
        print(f"   批次 {batch_num}: +{meta['num_chunks']:,} 块 (累计 {num_chunks:,})")

    elapsed = time.time() - t0
    print(f"   ✅ 导入完成: {num_chunks:,} 块 / {elapsed:.2f}s")

    stats = indexer.get_stats()
    return {
        "ingest_time_s": round(elapsed, 2),
        "num_chunks": num_chunks,
        "stats": stats,
    }


def run_eval(eval_path: str, max_cases: int = None) -> tuple:
    """运行评测用例集，返回结果列表 + 统计信息。"""
    from rag.pipeline import answer

    with open(eval_path, encoding="utf-8") as f:
        cases = json.load(f)

    if max_cases and max_cases > 0:
        cases = cases[:max_cases]

    total = len(cases)
    print(f"\n🧪 运行 {total:,} 条评测用例...\n")

    results = []
    correct = 0
    total_latency = 0
    latency_list = []

    # 按类别统计
    category_stats = defaultdict(lambda: {"total": 0, "correct": 0, "latency_sum": 0})
    # 按场景统计
    type_stats = defaultdict(lambda: {"total": 0, "correct": 0, "latency_sum": 0})

    for i, case in enumerate(cases):
        query = case["query"]
        expected_cat = case["expected_category"]
        case_type = case["type"]

        t0 = time.time()
        resp = answer(query, top_k=5)
        latency = (time.time() - t0) * 1000
        total_latency += latency
        latency_list.append(latency)

        retrieved = resp["contexts"]
        hit = any(expected_cat in ctx.get("page_name", "") for ctx in retrieved)

        if hit:
            correct += 1

        # 更新统计
        category_stats[expected_cat]["total"] += 1
        category_stats[expected_cat]["latency_sum"] += latency
        if hit:
            category_stats[expected_cat]["correct"] += 1

        type_stats[case_type]["total"] += 1
        type_stats[case_type]["latency_sum"] += latency
        if hit:
            type_stats[case_type]["correct"] += 1

        # 进度输出（每 200 条或最后一条）
        if (i + 1) % 200 == 0 or i + 1 == total:
            status = "✅" if hit else "❌"
            print(f"  [{i+1:>5}/{total}] {status} [{case_type:8s}] "
                  f"recall_so_far={correct}/{i+1} ({correct/(i+1)*100:.1f}%) | "
                  f"avg_latency={total_latency/(i+1):.0f}ms | "
                  f"{query[:40]:40s}")
        elif i < 5:
            # 前 5 条详细输出
            status = "✅" if hit else "❌"
            answer_preview = resp["answer"][:60].replace("\n", " ")
            print(f"  [{i+1:>5}/{total}] {status} [{case_type:8s}] "
                  f"{latency:5.0f}ms | {query[:40]:40s} | {answer_preview}...")

        results.append({
            "id": case["id"],
            "type": case_type,
            "query": query,
            "hit": hit,
            "latency_ms": round(latency, 1),
            "num_retrieved": len(retrieved),
            "expected_category": expected_cat,
            "expected_intent": case.get("expected_intent", ""),
            "retrieved_categories": [ctx.get("page_name", "") for ctx in retrieved[:3]],
            "retrieved_scores": [round(ctx.get("score", 0), 4) for ctx in retrieved[:3]],
            "answer_preview": resp["answer"][:200],
            "backend_used": resp["backend_used"],
        })

    return results, correct, total_latency, latency_list, category_stats, type_stats


def print_report(results, correct, total_latency, latency_list, category_stats, type_stats, ingest_info):
    """输出详细评测报告。"""
    total = len(results)
    recall = correct / total * 100
    avg_latency = total_latency / total

    # 延迟分位数
    sorted_latency = sorted(latency_list)
    p50 = sorted_latency[int(len(sorted_latency) * 0.5)]
    p95 = sorted_latency[int(len(sorted_latency) * 0.95)]
    p99 = sorted_latency[min(int(len(sorted_latency) * 0.99), len(sorted_latency) - 1)]
    min_lat = sorted_latency[0]
    max_lat = sorted_latency[-1]

    print(f"\n{'=' * 80}")
    print(f"📊 全量评测结果汇总")
    print(f"{'=' * 80}")
    print(f"  评测用例总数: {total:,}")
    print(f"  命中数:       {correct:,}")
    print(f"  召回率:       {recall:.1f}%")
    print(f"  生成后端:     {results[0]['backend_used']}")
    print(f"  导入信息:     {ingest_info['num_chunks']:,} 块 / {ingest_info['ingest_time_s']}s")
    print()
    print(f"  延迟统计 (ms):")
    print(f"    平均:  {avg_latency:.0f}")
    print(f"    最小:  {min_lat:.0f}")
    print(f"    最大:  {max_lat:.0f}")
    print(f"    P50:   {p50:.0f}")
    print(f"    P95:   {p95:.0f}")
    print(f"    P99:   {p99:.0f}")
    print()

    # 按场景统计
    print(f"  按场景统计:")
    print(f"  {'场景':<12s} {'总数':>6s} {'命中':>6s} {'召回率':>8s} {'平均延迟':>8s}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for t in ["basic", "boundary", "multiturn"]:
        s = type_stats.get(t)
        if s and s["total"] > 0:
            t_recall = s["correct"] / s["total"] * 100
            t_avg_lat = s["latency_sum"] / s["total"]
            print(f"  {t:<12s} {s['total']:>6,} {s['correct']:>6,} {t_recall:>7.1f}% {t_avg_lat:>7.0f}ms")
    print()

    # 按类别统计
    print(f"  按类别统计:")
    print(f"  {'类别':<16s} {'总数':>6s} {'命中':>6s} {'召回率':>8s} {'平均延迟':>8s}")
    print(f"  {'-'*16} {'-'*6} {'-'*6} {'-'*8} {'-'*8}")
    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        if s["total"] > 0:
            c_recall = s["correct"] / s["total"] * 100
            c_avg_lat = s["latency_sum"] / s["total"]
            print(f"  {cat:<16s} {s['total']:>6,} {s['correct']:>6,} {c_recall:>7.1f}% {c_avg_lat:>7.0f}ms")
    print()

    # 未命中 case 分析（展示前 20 条）
    missed = [r for r in results if not r["hit"]]
    print(f"  未命中 case 数: {len(missed):,} (占 {len(missed)/total*100:.1f}%)")
    if missed:
        print(f"\n  未命中 case 示例（前 20 条）:")
        print(f"  {'ID':<40s} {'类别':<12s} {'查询':<40s}")
        print(f"  {'-'*40} {'-'*12} {'-'*40}")
        for r in missed[:20]:
            print(f"  {r['id']:<40s} {r['expected_category']:<12s} {r['query'][:40]:<40s}")

    print(f"\n{'=' * 80}")


def main():
    parser = argparse.ArgumentParser(description="端到端 RAG 全量评测")
    parser.add_argument("--skip-ingest", action="store_true", help="跳过知识库导入")
    parser.add_argument("--max-cases", type=int, default=None, help="限制评测用例数（快速验证）")
    args = parser.parse_args()

    # 1. 确保预处理产物
    kb_path, eval_path = ensure_preprocessed()

    # 2. 重置 RAG 状态
    print("🔧 初始化 RAG 环境...")
    reset_rag_state()

    # 3. 导入知识库
    if not args.skip_ingest:
        ingest_info = ingest_kb(kb_path)
    else:
        print("⏭️  跳过知识库导入（--skip-ingest）")
        ingest_info = {"ingest_time_s": 0, "num_chunks": 0}

    # 4. 运行评测
    results, correct, total_latency, latency_list, category_stats, type_stats = run_eval(
        eval_path, args.max_cases
    )

    # 5. 输出报告
    print_report(results, correct, total_latency, latency_list, category_stats, type_stats, ingest_info)

    # 6. 保存详细结果
    output_path = os.path.join(_EXPERIMENT_DIR, "eval_results.json")
    summary = {
        "total_cases": len(results),
        "correct": correct,
        "recall_rate": f"{correct / len(results) * 100:.1f}%",
        "avg_latency_ms": round(total_latency / len(results), 1),
        "latency_min_ms": round(min(latency_list), 1),
        "latency_max_ms": round(max(latency_list), 1),
        "latency_p50_ms": round(sorted(latency_list)[int(len(latency_list) * 0.5)], 1),
        "latency_p95_ms": round(sorted(latency_list)[int(len(latency_list) * 0.95)], 1),
        "latency_p99_ms": round(sorted(latency_list)[min(int(len(latency_list) * 0.99), len(latency_list) - 1)], 1),
        "backend_used": results[0]["backend_used"] if results else "unknown",
        "ingest_info": ingest_info,
        "category_stats": {
            cat: {
                "total": s["total"],
                "correct": s["correct"],
                "recall_rate": f"{s['correct'] / s['total'] * 100:.1f}%" if s["total"] > 0 else "N/A",
                "avg_latency_ms": round(s["latency_sum"] / s["total"], 1) if s["total"] > 0 else 0,
            }
            for cat, s in sorted(category_stats.items())
        },
        "type_stats": {
            t: {
                "total": s["total"],
                "correct": s["correct"],
                "recall_rate": f"{s['correct'] / s['total'] * 100:.1f}%" if s["total"] > 0 else "N/A",
                "avg_latency_ms": round(s["latency_sum"] / s["total"], 1) if s["total"] > 0 else 0,
            }
            for t, s in sorted(type_stats.items())
        },
        "missed_count": len([r for r in results if not r["hit"]]),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n📁 详细结果已保存: {output_path}")
    print(f"   摘要: {summary['total_cases']:,} 用例, 召回率 {summary['recall_rate']}, "
          f"平均 {summary['avg_latency_ms']}ms, P95 {summary['latency_p95_ms']}ms")


if __name__ == "__main__":
    main()
