# -*- coding: utf-8 -*-
"""
HTML 文本高频短语提取与分词词典生成脚本

本模块用于批量提取中文 HTML 文本中的高频短语，支持 n-gram 统计、句级截断、冗余过滤等操作，最终生成适配 `jieba` 和 IK Analyzer 的自定义词典文件。

核心功能：
------------------------------------------------
1. 文本提取与清洗：
   - `extract_text_from_html`: 从单个 HTML 文件中提取纯文本内容。
   - `collect_all_html_texts`: 遍历目录，提取所有 HTML 文件文本。

2. 高频短语提取：
   - `extract_phrases_by_frequency`: 基于句级切分和滑动窗口方式提取 n-gram 词组，支持频率统计与长度控制。
   - `split_sentences`: 基于中文标点符号断句，避免跨句拼词。

3. 冗余词过滤策略：
   - `filter_keep_longest_only`: 仅保留最长不重叠短语（按包含关系过滤）。
   - `filter_by_freq_ratio`: 若短词是某长词子串，且频率相近，则视为冗余短词，过滤掉。

4. 分词词典构建与保存：
   - `save_to_jieba_dict`: 将提取结果保存为 `jieba` 词典格式：词语 频率 词性（n）。
   - 自动生成 IK Analyzer 所需的纯词条词典文件 `my_dict.dic`。

配置与用法：
------------------------------------------------
- 默认处理目录为 `"巨量千川知识库all"`，可在 `__main__` 中自定义。
- 支持指定 `ngram` 范围（如 2~12）与提取数量上限 `top_k`。
- 最终词典默认保存为：
  - `user_dict.txt`: jieba 格式（可用于中文分词增强）
  - `my_dict.dic`: IK 分词器词条字典（仅含词项）

"""



import os
import re
from collections import Counter
from bs4 import BeautifulSoup
from utils.config import logger

def extract_text_from_html(html_path):
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            text = soup.get_text()
            logger.debug(f"✔️ 成功提取文本：{html_path}（{len(text)} 字）")
            return text
    except Exception as e:
        logger.warning(f"❌ 读取失败: {html_path}, 原因: {e}")
        return ""

def collect_all_html_texts(root_dir):
    texts = []
    logger.info(f"🔍 正在遍历目录: {root_dir}")
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".html"):
                full_path = os.path.join(root, file)
                text = extract_text_from_html(full_path)
                if text:
                    texts.append(text)
    logger.info(f"📄 共收集到 {len(texts)} 个 HTML 文本")
    return texts

def split_sentences(text):
    # 中文及中英标点断句
    return re.split(r"[。！？；，,.!?;:、\n\r]+", text)

def is_chinese_word(word):
    return bool(word) and all('\u4e00' <= ch <= '\u9fff' for ch in word)


def extract_phrases_by_frequency(texts, ngram_range=(2, 5), top_k=500):
    counter = Counter()
    logger.info("📊 正在进行 n-gram 词频统计（按句截断，纯中文）...")

    for text in texts:
        sentences = split_sentences(text)
        for sentence in sentences:
            sentence = re.sub(r"[^\u4e00-\u9fff]", "", sentence)  # 仅保留中文
            for n in range(ngram_range[0], ngram_range[1] + 1):
                for i in range(len(sentence) - n + 1):
                    gram = sentence[i:i + n]
                    if is_chinese_word(gram):
                        counter[gram] += 1

    top_phrases = counter.most_common(top_k)
    logger.info(f"✅ 高频短语提取完毕（分句后），共返回前 {len(top_phrases)} 项")
    return top_phrases




def save_to_jieba_dict(phrases, output_path="user_dict.txt", default_freq=10000):
    logger.info(f"📝 正在保存词典到 {output_path} ...")
    with open(output_path, "w", encoding="utf-8") as f:
        for phrase, freq in phrases:
            f.write(f"{phrase} {freq or default_freq} n\n")
    logger.info(f"✅ 词典保存成功，可用于 jieba / IK 分词器")


def filter_keep_longest_only(phrases):
    logger.debug("正在过滤冗余词 ...")
    phrases_sorted = sorted(phrases, key=lambda x: (-len(x[0]), -x[1]))
    kept = []
    for phrase, freq in phrases_sorted:
        if any(phrase in longer for longer, _ in kept if phrase != longer):
            continue  # 是已保留词的子串，跳过
        kept.append((phrase, freq))
    return kept

def filter_by_freq_ratio(phrases, threshold=0.8):
    logger.debug("正在过滤冗余词 ...")
    phrases_sorted = sorted(phrases, key=lambda x: -len(x[0]))  # 先长词后短词
    phrase_map = {phrase: freq for phrase, freq in phrases}
    filtered = {}

    for phrase in phrase_map:
        if any(
            phrase in longer and phrase != longer and
            phrase_map.get(longer, 0) >= phrase_map[phrase] * threshold
            for longer in phrase_map
        ):
            continue  # 被某个更长词吸收，跳过
        filtered[phrase] = phrase_map[phrase]

    return list(filtered.items())



# ===== 主流程入口 =====
if __name__ == "__main__":
    import argparse
    from utils.config import DATA_DIR

    parser = argparse.ArgumentParser(description="构建 jieba 自定义词典")
    parser.add_argument("--html-dir", type=str, default="process/data",
                        help="HTML 源文件目录（用于提取高频短语）")
    parser.add_argument("--output", type=str, default=os.path.join(DATA_DIR, "user_dict.txt"),
                        help="输出词典路径（默认与 config.USER_DICT_PATH 一致）")
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=1000)
    args = parser.parse_args()

    texts = collect_all_html_texts(args.html_dir)
    phrases = extract_phrases_by_frequency(texts, ngram_range=(args.ngram_min, args.ngram_max), top_k=args.top_k)
    filtered_phrases = filter_by_freq_ratio(phrases)
    save_to_jieba_dict(filtered_phrases, output_path=args.output)

    # 额外生成 IK 分词器使用的纯词条词典
    ik_dict_path = args.output.replace(".txt", ".dic")
    with open(args.output, "r", encoding="utf-8") as fin, open(ik_dict_path, "w", encoding="utf-8") as fout:
        for line in fin:
            parts = line.strip().split()
            if parts:
                word = parts[0]
                fout.write(word + "\n")
    logger.info(f"✅ IK 分词器词典已生成: {ik_dict_path}（仅包含词条）")
