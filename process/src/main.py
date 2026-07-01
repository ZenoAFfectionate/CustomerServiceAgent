# -*- coding: utf-8 -*-
"""
HTML 数据处理全流程入口（清洗 + 分块）。

将原 step1_html_clean（HTML 批量清洗）与 step2_block_construct（结构化分块 +
LLM 摘要）合并为单文件，一次运行即可从原始 HTML 得到结构化 JSON 文档块。

流程：
    原始 HTML → Step 1 清洗（去噪、表格展开、域包装）→ Step 2 分块（Block
    Tree + 表格切分 + LLM 摘要）→ 输出 JSON

用法：
    # 全流程（清洗 + 分块）
    PYTHONPATH=process python -m main --source-dir process/dataset/html_source

    # 仅清洗
    PYTHONPATH=process python -m main --source-dir process/dataset/html_source --step clean

    # 仅分块（需先完成清洗）
    PYTHONPATH=process python -m main --html-dir process/dataset/html_cleaned --step block

    # 使用 vLLM 远程摘要（无需本地加载 ChatGLM）
    PYTHONPATH=process python -m main --source-dir ... --use-vllm
"""
import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from html_utils import process_html_file, build_block_tree, parse_time_tag
from text_process_utils import generate_block_documents, save_doc_meta_to_block_dir
from utils.config import CONFIG, logger


# ======================== Step 1: HTML 批量清洗 ========================

def run_clean(source_dir: str, target_dir: str) -> list:
    """批量清洗 HTML 文件，输出到 target_dir，返回 (源路径, 目标路径) 列表。"""
    os.makedirs(target_dir, exist_ok=True)

    cleaned_files = []
    for dirpath, _, filenames in os.walk(source_dir):
        rel_path = os.path.relpath(dirpath, source_dir)
        target_subdir = os.path.join(target_dir, rel_path)
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


def process_html_to_blocks(html_path: str, args, summary_model=None, summary_tokenizer=None) -> None:
    """处理单个 HTML 文件：提取 time → 分块 → 生成摘要 → 保存 JSON。"""
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

    # 生成文档块元数据（含摘要）
    doc_meta = generate_block_documents(
        block_tree,
        max_node_words=args.max_node_words,
        page_url=os.path.relpath(html_path),
        summary_model=summary_model,
        summary_tokenizer=summary_tokenizer,
        time_value=time_value,
        use_vllm=args.use_vllm,
    )

    # 保存为 JSON 文件
    save_doc_meta_to_block_dir(
        doc_meta,
        html_path,
        html_root_dir=args.html_dir,
        block_root_dir=args.html_dir + "_block",
    )


def run_block(args, summary_model=None, summary_tokenizer=None) -> None:
    """批量分块：遍历 html_dir 下所有 HTML 文件，生成 JSON 文档块。"""
    html_files = get_all_html_files(args.html_dir)
    logger.info(f"📄 共发现 HTML 文件数: {len(html_files)}")

    for html_file in html_files:
        process_html_to_blocks(html_file, args, summary_model, summary_tokenizer)

    logger.info("✅ 全部文档块 JSON 已生成完毕。")


# ======================== 模型加载 ========================

def load_summary_model(model_name: str):
    """加载本地 ChatGLM 模型用于摘要生成（仅在 --use-vllm 未启用时调用）。"""
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().to(device)
    model.eval()
    logger.info(f"📦 已加载摘要模型: {model_name} (device={device})")
    return model, tokenizer


# ======================== 主入口 ========================

def main():
    parser = argparse.ArgumentParser(
        description="HTML 数据处理全流程（清洗 + 分块 + 摘要）"
    )

    # --- 路径参数 ---
    parser.add_argument(
        "--source-dir", type=str, default="dataset/html_source",
        help="原始 HTML 文件根目录（用于 --step all/clean）",
    )
    parser.add_argument(
        "--target-dir", type=str, default="",
        help="清洗后输出目录（默认 {source-dir}_cleaned）",
    )
    parser.add_argument(
        "--html-dir", type=str, default="dataset/html_cleaned",
        help="清洗后 HTML 目录（用于 --step block）",
    )

    # --- 流程控制 ---
    parser.add_argument(
        "--step", type=str, default="all",
        choices=["all", "clean", "block"],
        help="执行步骤：all=清洗+分块，clean=仅清洗，block=仅分块（默认 all）",
    )
    parser.add_argument(
        "--use-vllm", action="store_true", default=False,
        help="使用 vLLM 远程服务生成摘要（无需本地加载 ChatGLM 模型）",
    )

    # --- 分块参数 ---
    parser.add_argument("--lang", type=str, default=CONFIG.get("lang", "zh"))
    parser.add_argument("--max-node-words", type=int, default=CONFIG.get("max_node_words_embed", 4096))
    parser.add_argument("--min-node-words", type=int, default=CONFIG.get("min_node_words_embed", 48))
    parser.add_argument(
        "--summary-model", type=str, default=CONFIG.get("llm_model", "THUDM/glm-4-9b-chat"),
        help="本地摘要模型名（仅 --use-vllm 未启用时使用）",
    )

    args = parser.parse_args()

    # --- Step 1: 清洗 ---
    if args.step in ("all", "clean"):
        target_dir = args.target_dir or (args.source_dir.rstrip("/") + "_cleaned")
        run_clean(args.source_dir, target_dir)
        # 全流程模式下自动更新 html_dir 指向清洗输出
        if args.step == "all":
            args.html_dir = target_dir

    # --- Step 2: 分块 ---
    if args.step in ("all", "block"):
        summary_model = None
        summary_tokenizer = None

        # 非远程 vLLM 模式下加载本地 ChatGLM
        if not args.use_vllm:
            summary_model, summary_tokenizer = load_summary_model(args.summary_model)

        run_block(args, summary_model, summary_tokenizer)


if __name__ == "__main__":
    main()
