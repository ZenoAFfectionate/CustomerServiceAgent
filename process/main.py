# -*- coding: utf-8 -*-
"""
HTML 数据处理全流程入口（清洗 + 分块）。

将原 step1_html_clean（HTML 批量清洗）与 step2_block_construct（结构化分块 +
LLM 摘要）合并为单文件，一次运行即可从原始 HTML 得到结构化 JSON 文档块。

流程：
    原始 HTML → Step 1 清洗（去噪、表格展开、域包装）→ Step 2 分块（Block
    Tree + 表格切分 + vLLM 摘要）→ 输出 JSON

输出结构（文件夹级后缀，保留子目录结构）：
    process/data/
    ├── {数据集名}/                ← 原始 HTML
    ├── {数据集名}_cleaned/        ← 清洗后（子目录结构不变，文件名不变）
    └── {数据集名}_blocked/        ← 分块后（子目录结构不变，.html → .json）

用法：
    # 全流程（清洗 + 分块）
    PYTHONPATH=process python -m main --source-dir process/data/抖音电商规则中心

    # 仅清洗
    PYTHONPATH=process python -m main --source-dir process/data/抖音电商规则中心 --step clean

    # 仅分块（需先完成清洗，需要 vLLM 服务在运行）
    PYTHONPATH=process python -m main --html-dir process/data/抖音电商规则中心_cleaned --step block
"""
import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from html_utils import process_html_file, build_block_tree, parse_time_tag
from text_process import generate_block_documents, save_doc_meta_to_block_dir
from utils.config import CONFIG, logger


# ======================== Step 1: HTML 批量清洗 ========================

def run_clean(source_dir: str, target_dir: str) -> list:
    """批量清洗 HTML 文件，保留子目录结构，输出到 target_dir。

    路径映射：source_dir/a/b/c.html → target_dir/a/b/c.html（文件名不变）
    _cleaned 后缀在文件夹级别（由调用方通过 target_dir 命名体现）。
    """
    os.makedirs(target_dir, exist_ok=True)

    cleaned_files = []
    for dirpath, _, filenames in os.walk(source_dir):
        rel_path = os.path.relpath(dirpath, source_dir)
        target_subdir = os.path.join(target_dir, rel_path) if rel_path != "." else target_dir
        os.makedirs(target_subdir, exist_ok=True)

        for filename in filenames:
            if filename.endswith(".html"):
                source_file = os.path.join(dirpath, filename)
                target_file = os.path.join(target_subdir, filename)
                try:
                    process_html_file(source_file, target_file)
                    cleaned_files.append((source_file, target_file))
                except Exception as e:
                    logger.error(f"❌ 清洗失败 {source_file}：{e}")

    logger.info(f"✅ 清洗完成，共处理 {len(cleaned_files)} 个 HTML 文件，输出至 {target_dir}")
    return cleaned_files


# ======================== Step 2: 结构化分块与摘要 ========================

def get_all_html_files(html_dir: str) -> list:
    """递归获取目录下所有 HTML 文件路径。"""
    html_files = []
    for root, _, files in os.walk(html_dir):
        for file in files:
            if file.lower().endswith(".html"):
                html_files.append(os.path.join(root, file))
    return html_files


def process_html_to_blocks(html_path: str, args) -> None:
    """处理单个 HTML 文件：提取 time → 分块 → 生成摘要（vLLM）→ 保存 JSON。

    输出到 block_root_dir（如 process/data/数据集名_blocked/），保留子目录结构。
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    logger.info(f"=== Processing: {html_path} ===")

    # 提取 <time> 标签
    time_value, remaining_html = parse_time_tag(html)

    # 清洗并构建结构树
    block_tree, _ = build_block_tree(
        remaining_html,
        max_node_words=args.max_node_words,
        min_node_words=args.min_node_words,
        zh_char=(args.lang == "zh"),
    )

    # 生成文档块元数据（含 vLLM 摘要）
    doc_meta = generate_block_documents(
        block_tree,
        max_node_words=args.max_node_words,
        page_url=os.path.relpath(html_path),
        time_value=time_value,
    )

    # 保存 JSON：保留子目录结构，.html → .json
    save_doc_meta_to_block_dir(
        doc_meta,
        html_path,
        html_root_dir=args.html_dir,
        block_root_dir=args.block_root_dir,
    )


def run_block(args) -> None:
    """批量分块：遍历 html_dir 下所有 HTML 文件，生成 JSON 文档块。"""
    html_files = get_all_html_files(args.html_dir)
    logger.info(f"📄 共发现 HTML 文件数: {len(html_files)}")

    for html_file in html_files:
        process_html_to_blocks(html_file, args)

    logger.info(f"✅ 全部文档块 JSON 已生成完毕，输出至 {args.block_root_dir}")


# ======================== 主入口 ========================

def main():
    parser = argparse.ArgumentParser(
        description="HTML 数据处理全流程（清洗 + 分块 + vLLM 摘要）"
    )

    # --- 路径参数 ---
    parser.add_argument(
        "--source-dir", type=str, default="process/data",
        help="原始 HTML 文件根目录（用于 --step all/clean），如 process/data/抖音电商规则中心",
    )
    parser.add_argument(
        "--target-dir", type=str, default="",
        help="清洗后输出目录（默认 {source-dir}_cleaned，与 source 同级）",
    )
    parser.add_argument(
        "--html-dir", type=str, default="",
        help="清洗后 HTML 目录（用于 --step block），如 process/data/数据集名_cleaned",
    )
    parser.add_argument(
        "--block-output-dir", type=str, default="",
        help="分块 JSON 输出目录（默认 {source-dir}_blocked，与 source 同级）",
    )

    # --- 流程控制 ---
    parser.add_argument(
        "--step", type=str, default="all",
        choices=["all", "clean", "block"],
        help="执行步骤：all=清洗+分块，clean=仅清洗，block=仅分块（默认 all）",
    )

    # --- 分块参数 ---
    parser.add_argument("--lang", type=str, default=CONFIG.get("lang", "zh"))
    parser.add_argument("--max-node-words", type=int, default=CONFIG.get("max_node_words_embed", 4096))
    parser.add_argument("--min-node-words", type=int, default=CONFIG.get("min_node_words_embed", 48))

    args = parser.parse_args()

    # --- Step 1: 清洗 ---
    if args.step in ("all", "clean"):
        target_dir = args.target_dir or (args.source_dir.rstrip("/") + "_cleaned")
        run_clean(args.source_dir, target_dir)
        # 全流程模式下自动更新 html_dir 指向清洗输出
        if args.step == "all":
            args.html_dir = target_dir

    # --- Step 2: 分块（需要 vLLM 服务在运行）---
    if args.step in ("all", "block"):
        # html_dir 默认值：source_dir_cleaned
        if not args.html_dir:
            args.html_dir = args.source_dir.rstrip("/") + "_cleaned"
        # block_root_dir：从 html_dir 推导（替换 _cleaned → _blocked）
        if args.block_output_dir:
            args.block_root_dir = args.block_output_dir
        else:
            html_dir = args.html_dir.rstrip("/")
            if html_dir.endswith("_cleaned"):
                args.block_root_dir = html_dir[:-len("_cleaned")] + "_blocked"
            else:
                args.block_root_dir = html_dir + "_blocked"
        run_block(args)


if __name__ == "__main__":
    main()
