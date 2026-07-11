# -*- coding: utf-8 -*-
"""
文本处理与文档块生成模块。

负责从 HTML 清洗结果中提取结构化文档块，生成摘要与问句，
并提供查询构建、文本清洗与去重能力。

核心函数：
    - clean_text:                  移除非中英文数字字符
    - clean_invisible:             清除零宽与控制字符
    - extract_title_from_block:    从 HTML 块提取标题
    - build_optimal_jieba_query:   构建 ES bool 查询
    - deduplicate_ranked_blocks_pal: TF-IDF 去重（时间优先）
    - generate_block_documents:     同步生成文档块（含表格切分 + 摘要）
    - generate_block_documents_async: 异步批量生成文档块
    - save_doc_meta_to_block_dir:   保存文档块 JSON
"""

import os
import re
import json
import time
import asyncio
from datetime import datetime
from collections import defaultdict, deque
from difflib import SequenceMatcher
from typing import List

import jieba
import numpy as np
from bs4 import Tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils.llm_api import (
    generate_summary_vllm,
    generate_summary_vllm_async,
    generate_question_vllm_async,
)
from utils.config import CONFIG, logger, USER_DICT_PATH

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 初始化 jieba 分词器
_pure_tokenizer = jieba.Tokenizer(dictionary=jieba.DEFAULT_DICT)
if os.path.isfile(USER_DICT_PATH):
    jieba.load_userdict(USER_DICT_PATH)
else:
    logger.warning(f"自定义词典不存在: {USER_DICT_PATH}，使用 jieba 默认词典")

# 中文字符+英文+数字正则
_CLEAN_TEXT_RE = re.compile(r"[\u4e00-\u9fa5a-zA-Z0-9]+")
# 词数统计正则（用于表格行切分时的中文词数计算）
_WORD_COUNT_RE = re.compile(r"[\u4e00-\u9fa5a-zA-Z0-9]")
# 不可见字符正则
_INVISIBLE_RE = re.compile(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f]')
# UI 噪声文本模式（进度条、导航等），模块级预编译避免重复编译
_UI_NOISE_RE = re.compile(
    r'^(?:\d+%|PROGRESS|CONTENTS|尚未开始|目录|↑顶部|搜索\.\.\.|←上一页|下一页→)$',
    re.IGNORECASE
)


# ======================== 文本清洗工具 ========================

def clean_text(text: str) -> str:
    """移除文本中的特殊字符，仅保留中英文与数字。"""
    return "".join(_CLEAN_TEXT_RE.findall(text))


def jieba_cut_clean(text: str) -> list:
    """清洗后进行 jieba 分词。"""
    return list(_pure_tokenizer.cut(clean_text(text), HMM=False))


def clean_invisible(text: str) -> str:
    """清除零宽与控制类不可见字符。"""
    return _INVISIBLE_RE.sub("", text)


# ======================== 标题提取 ========================

def extract_title_from_block(tag: Tag) -> str:
    """从 HTML 块中提取第一个 heading 标签（h1~h6）作为标题。

    若不存在则回退为第一个非空文本。标题截断为 48 字符。
    过滤 UI 噪声文本（进度条、导航等）和前导数字序号。
    """
    for descendant in tag.descendants:
        if isinstance(descendant, Tag) and descendant.name:
            name = descendant.name.lower()
            if name.startswith("h") and len(name) == 2 and name[1].isdigit():
                title = descendant.get_text(separator="", strip=True)[:48]
                return _clean_title(title)

    for t in tag.stripped_strings:
        t = t.strip()
        if t and not _is_ui_noise_text(t):
            return _clean_title(t)[:48]
    return ""


def _is_ui_noise_text(text: str) -> bool:
    """判断文本是否为 UI 噪声（进度条、导航等）。"""
    return bool(_UI_NOISE_RE.match(text.strip()))


def _clean_title(title: str) -> str:
    """清理标题：去除前导数字序号（如 "1.2 "）、多余空白。"""
    # 去除前导的 "1.", "1.2 ", "0" 等数字序号
    title = re.sub(r'^[\d.]+\s*', '', title)
    return title.strip()[:48]


# ======================== ES 查询构建 ========================

def build_optimal_jieba_query(
    jieba_keywords: list,
    fields_config: dict,
    synonym_map: dict = None,
    use_phrase: bool = True,
    use_fuzzy: bool = True,
) -> dict:
    """构建 Elasticsearch bool 查询：精确匹配 + 模糊匹配 + 短语匹配 + 同义词扩展。

    Args:
        jieba_keywords: jieba 提取的关键词列表
        fields_config: 字段配置，如 {"title": {"boost": 5, "fuzzy": False}}
        synonym_map: 同义词字典，如 {"千川": ["千川", "巨量千川"]}
        use_phrase: 是否启用短语匹配
        use_fuzzy: 是否启用模糊匹配

    Returns:
        ES 查询 JSON 字典
    """
    should_clauses = []

    for word in jieba_keywords:
        synonyms = synonym_map.get(word, [word]) if synonym_map else [word]

        for field, config in fields_config.items():
            boost = config.get("boost", 1)
            synonym_queries = []

            # 精确匹配
            if synonyms:
                synonym_queries.append(
                    {"terms": {f"{field}.keyword": synonyms, "boost": boost * 1.2}}
                )

            # 模糊匹配
            if use_fuzzy and config.get("fuzzy", True):
                for syn in synonyms:
                    synonym_queries.append({
                        "match": {
                            field: {
                                "query": syn,
                                "fuzziness": "AUTO",
                                "boost": boost * 0.5,
                            }
                        }
                    })

            # 短语匹配
            if use_phrase and len(word) > 1:
                for syn in synonyms:
                    synonym_queries.append({
                        "match_phrase": {
                            field: {"query": syn, "slop": 2, "boost": boost * 0.8}
                        }
                    })

            if synonym_queries:
                should_clauses.append(
                    {"bool": {"should": synonym_queries, "minimum_should_match": 1}}
                )

    if not should_clauses:
        # 【修复 M10】关键词为空（如空 query、纯停用词、query 重写后返回空串）时，
        # 此前会生成 {"bool": {"should": [], "minimum_should_match": "30%"}} 这种
        # 退化查询，其行为依赖 ES 版本——可能被解释为全库召回（污染 rerank/剪枝
        # 预算）或零召回，均不理想。显式返回 match_none，保证行为可预测：无
        # 有效关键词时不召回任何结果（入口 /chat、/chat/stream 已对 query 做
        # 非空校验，此处主要覆盖 rewrite_query 返回空串或绕过 API 直调的场景）。
        return {
            "query": {"match_none": {}},
            "highlight": {
                "fields": {"*": {"pre_tags": ["<em>"], "post_tags": ["</em>"]}}
            },
        }

    return {
        "query": {"bool": {"should": should_clauses, "minimum_should_match": "30%"}},
        "highlight": {
            "fields": {"*": {"pre_tags": ["<em>"], "post_tags": ["</em>"]}}
        },
    }


# ======================== 检索结果去重 ========================

def parse_time(t: str) -> datetime:
    """解析时间字符串，失败返回 datetime.min。"""
    try:
        return datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return datetime.min


def str_sim(a: str, b: str) -> float:
    """计算两个字符串的相似度比率。"""
    return SequenceMatcher(None, a, b).ratio()


def deduplicate_ranked_blocks_pal(
    docs: list,
    threshold_content: float = 0.9,
    threshold_page_name: float = 0.6,
    threshold_content_strict: float = 0.97,
) -> list:
    """基于 TF-IDF + cosine 相似度去重，按时间优先保留最新版本。

    使用迭代式 BFS（替代递归 DFS）避免大集群栈溢出。

    Args:
        docs: 文档块列表
        threshold_content: 正文相似度阈值（配合 page_name 阈值判定"中等相似"重复）
        threshold_page_name: 页面名相似度阈值
        threshold_content_strict: 正文近乎完全相同的严格阈值。达到此阈值时无论
            page_name 是否相似都直接判重——修复"同一文档以不同文件名两次入库"
            （如 `投放计划说明.md` 与 `投放计划说明(2).md`）时，因 page_name 差异
            过大而漏判重复的问题（对应代码审查报告 M8）。

    Returns:
        去重后的文档块列表
    """
    n = len(docs)
    if n <= 1:
        return docs

    texts = [clean_text(doc.get("text", "")) for doc in docs]
    names = [clean_text(doc.get("page_name", "")) for doc in docs]
    times = [parse_time(doc.get("time", "")) for doc in docs]

    # 【修复 N8】此前用同一个 TfidfVectorizer 拟合 texts+names，让正文（长、
    # 词汇量大）与文件名共用词典，正文 IDF 压低文件名特有词权重，导致
    # sim_name 实为"字符集合相似度"而非真正的文件名语义相似度。改为对
    # texts 和 names 分别用独立的 TfidfVectorizer 拟合，各自维护独立 IDF。
    tfidf_text = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit(texts)
    sim_text = cosine_similarity(tfidf_text.transform(texts))

    # 文件名相似度：用独立的 vectorizer（独立 IDF），避免正文词汇干扰
    if not any(names):
        sim_name = np.ones((n, n))
    else:
        tfidf_name = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit(names)
        name_matrix = tfidf_name.transform(names)
        if name_matrix.nnz == 0:
            sim_name = np.ones((n, n))
        else:
            sim_name = cosine_similarity(name_matrix)

    # 上三角重复对。判定为重复的两种情形：
    #   1) 正文几乎完全相同（≥ threshold_content_strict）：无论 page_name 是否
    #      相似都直接判重（修复 M8：同内容异文件名漏判）；
    #   2) 正文相似度处于中等区间（[threshold_content, threshold_content_strict)）：
    #      仍要求 page_name 也相似，避免仅措辞相近但确系不同文档的内容被误判。
    triu_idx = np.triu_indices(n, k=1)
    strict_dup = sim_text[triu_idx] >= threshold_content_strict
    lenient_dup = (sim_text[triu_idx] >= threshold_content) & (sim_name[triu_idx] >= threshold_page_name)
    sim_mask = strict_dup | lenient_dup
    dup_pairs = list(zip(triu_idx[0][sim_mask], triu_idx[1][sim_mask]))

    # 构建重复簇：用图表示
    graph = defaultdict(set)
    for i, j in dup_pairs:
        graph[i].add(j)
        graph[j].add(i)

    # 迭代式 BFS 连通分量检测（避免递归栈溢出）
    visited = set()
    keep = set()

    for start in range(n):
        if start in visited:
            continue

        # BFS 遍历连通分量
        group = []
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            group.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        if len(group) == 1:
            keep.add(group[0])
        else:
            # 保留时间最新的
            keep.add(max(group, key=lambda x: times[x]))

    kept = sorted(keep)
    logger.info(f"✅ 原始 {n} 个块，重复对 {len(dup_pairs)}，去重后保留 {len(kept)}")
    return [docs[i] for i in kept]


# ======================== 文档块持久化 ========================

def save_doc_meta_to_block_dir(doc_meta: list, html_path: str, html_root_dir: str, block_root_dir: str) -> str:
    """将文档块 JSON 保存到与 HTML 相同的目录结构。

    路径映射：html_root/a/b/c.html → block_root/b/c.json
    """
    rel_path = os.path.relpath(html_path, html_root_dir)
    rel_json_path = os.path.splitext(rel_path)[0] + ".json"
    json_full_path = os.path.join(block_root_dir, rel_json_path)

    os.makedirs(os.path.dirname(json_full_path), exist_ok=True)

    with open(json_full_path, "w", encoding="utf-8") as f:
        json.dump(doc_meta, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ JSON 已保存：{json_full_path}")
    return json_full_path


# ======================== 共享辅助函数 ========================

def _row_to_text(row: Tag) -> str:
    """将表格行转换为文本（空格分隔各单元格 + 换行符）。"""
    return " ".join(cell.strip() for cell in row.stripped_strings) + "\n"


def _count_words_in_text(text: str) -> int:
    """计算文本中的中英文数字字符数（用于表格行词数统计）。"""
    return len(_WORD_COUNT_RE.findall(text))


def _generate_summary_and_question(
    text: str,
    page_url: str,
    gen_question: bool = False,
    **kwargs,  # backward compat: use_vllm, summary_model, summary_tokenizer (ignored)
) -> tuple:
    """生成摘要（vLLM），可选生成代表性问句。

    Returns:
        (summary, question) 字符串元组
    """
    summary = generate_summary_vllm(text, page_url)
    question = ""
    return summary, question


def _make_chunk_dict(
    chunk_idx: int,
    page_name: str,
    title: str,
    page_url: str,
    text: str,
    time_value: str,
    summary: str = "",
    question: str = "",
    block_path: str = "",
    html_content: str = "",
) -> dict:
    """构造标准文档块字典。

    Args:
        block_path: 从根标签到当前块的标签路径（如 "html>body>div0>p"），论文中用于唯一标识块。
        html_content: 保留 HTML 结构的块内容（论文核心观点：HTML 优于纯文本）。
    """
    return {
        "chunk_idx": chunk_idx,
        "page_name": page_name,
        "title": title,
        "page_url": page_url,
        "summary": summary,
        "question": question,
        "text": text,
        "html_content": html_content,
        "block_path": block_path,
        "time": time_value,
    }


def _extract_mixed_content(tag: Tag, title: str, max_node_words: int) -> list:
    """从混合内容块中分别提取文本和表格。

    遍历标签的直接子节点，将表格和非表格内容分开处理。
    返回 [(sub_tag_or_none, sub_title, text), ...] 列表。
    """
    results = []
    text_buffer = []

    def flush_text():
        """将缓冲的文本作为一个块输出。"""
        if text_buffer:
            text = clean_invisible(" ".join(text_buffer).replace("\x00", ""))
            if text:
                results.append((None, title, text))
            text_buffer.clear()

    for child in tag.children:
        if isinstance(child, Tag) and child.name == "table":
            # 遇到表格：先提交前面的文本
            flush_text()
            # 提交表格
            for text, row_start, row_end in _split_table_into_chunks(child, title, max_node_words):
                results.append((child, f"{title[:96]} 表格行{row_start}-{row_end}", text))
        elif isinstance(child, Tag) and child.find("table"):
            # 子标签中包含表格：递归处理
            flush_text()
            for sub_tag, sub_title, sub_text in _extract_mixed_content(child, title, max_node_words):
                if sub_text:
                    results.append((sub_tag, sub_title, sub_text))
        else:
            # 普通文本节点
            if isinstance(child, Tag):
                t = child.get_text()
            else:
                t = str(child)
            t = t.strip()
            if t:
                text_buffer.append(t)

    flush_text()
    # 后处理：将过小的文本片段（如表格间过渡文字）合并到相邻块中
    return _merge_tiny_text_fragments(results, max_node_words)


def _merge_tiny_text_fragments(results: list, max_node_words: int) -> list:
    """将过小的文本片段合并到相邻块中。

    flush_text() 可能在表格之间产生极短的文本片段
    （如"商家"、"附件二："、2-10 字符），这些碎片对 RAG 几乎无价值。
    这里将过小的纯文本片段与其相邻块合并。

    合并策略（最小修改原则）：
    - 找到 < 20 字符的纯文本片段（sub_tag 为 None）
    - 优先向前合并（追加到前一个块的 text 末尾）
    - 如果前面没有块，则向后合并（插入到后一个块的 text 开头）
    - 如果合并后超过 max_node_words，则保留原样
    """
    TINY_THRESHOLD = 20  # 小于此字符数视为碎片

    if len(results) <= 1:
        return results

    merged = []
    i = 0

    while i < len(results):
        sub_tag, sub_title, text = results[i]

        # 正常块（非碎片）直接保留
        if sub_tag is not None or len(text) >= TINY_THRESHOLD:
            merged.append(results[i])
            i += 1
            continue

        # --- 以下是碎片处理 ---
        handled = False

        # 策略 1：向前合并到 merged 中最后一个块
        if merged:
            prev = merged[-1]
            prev_text = prev[2]
            if len(prev_text) + len(text) + 1 <= max_node_words:
                merged[-1] = (prev[0], prev[1], prev_text + "\n" + text)
                handled = True

        # 策略 2：向后合并到下一个块（修改 results 原地，下一个迭代会处理）
        if not handled and i + 1 < len(results):
            next_item = results[i + 1]
            next_text = next_item[2]
            if len(next_text) + len(text) + 1 <= max_node_words:
                # 将碎片文本注入到下一个块的 text 开头
                results[i + 1] = (next_item[0], next_item[1], text + "\n" + next_text)
                # 跳过碎片：下一次迭代 i+1 会处理已注入文本的合并块
                handled = True

        # 策略 3：无法合并，保留原样
        if not handled:
            merged.append(results[i])

        i += 1

    return merged


def _split_table_into_chunks(
    table_tag: Tag,
    title: str,
    max_node_words: int,
) -> list:
    """将表格按行切分为多个文本块。

    返回 [(text, row_range_start, row_range_end), ...] 列表。
    每个块都以表头开头，便于独立理解。
    """
    rows = table_tag.find_all("tr")
    if not rows:
        return []

    # 提取表头（第一行或 thead 中的行）
    header_text = _row_to_text(rows[0])
    header_words = _count_words_in_text(header_text)

    # 只有一行表头时，直接返回
    if len(rows) == 1:
        text = clean_invisible(header_text.strip())
        if text:
            return [(text, 0, 0)]
        return []

    chunks = []
    current_text = header_text
    current_words = header_words
    row_range_start = 1

    for idx, row in enumerate(rows[1:], start=2):
        row_text = _row_to_text(row)
        row_words = _count_words_in_text(row_text)

        # 单行就超过 max_node_words 时，强制独立成块
        if row_words > max_node_words:
            # 先提交当前块
            if current_words > header_words:
                text = clean_invisible(current_text.strip())
                if text:
                    chunks.append((text, row_range_start, idx - 1))
            # 提交超长行（带表头）
            text = clean_invisible((header_text + row_text).strip())
            if text:
                chunks.append((text, idx, idx))
            # 重置
            current_text = header_text
            current_words = header_words
            row_range_start = idx + 1
        elif current_words + row_words > max_node_words:
            # 提交当前块
            text = clean_invisible(current_text.strip())
            if text:
                chunks.append((text, row_range_start, idx - 1))
            # 重置（新块以表头开头）
            current_text = header_text + row_text
            current_words = header_words + row_words
            row_range_start = idx
        else:
            current_text += row_text
            current_words += row_words

    # 提交最后一个块（仅当包含数据行时才提交，避免幽灵表头块）
    if current_words > header_words:
        text = clean_invisible(current_text.strip())
        if text:
            chunks.append((text, row_range_start, len(rows)))

    return chunks


# ======================== 文档块生成（同步） ========================

def generate_block_documents(
    block_tree: list,
    max_node_words: int,
    page_url: str = "unknown.html",
    time_value: str = "",
    gen_question: bool = False,
    **kwargs,  # backward compat: use_vllm, summary_model, summary_tokenizer (ignored)
) -> list:
    """生成结构化文档块，支持表格自动切分和 vLLM 摘要生成。

    输出包含 block_path（论文中的块路径，用于唯一标识和剪枝）和
    html_content（保留 HTML 结构，论文核心观点：HTML 优于纯文本）。

    Args:
        block_tree: build_block_tree 返回的块列表 [(Tag, path, is_leaf), ...]
        max_node_words: 每个块的最大词数
        page_url: 来源页面 URL
        time_value: 文档时间戳
        gen_question: 是否生成代表性问题

    Returns:
        文档块元数据列表
    """
    doc_meta = []
    chunk_idx = 0
    page_name = os.path.splitext(os.path.basename(page_url))[0]

    logger.info(f"📦 共提取块数：{len(block_tree)}")

    for block_tag, block_path, is_leaf in block_tree:
        title = extract_title_from_block(block_tag)
        path_str = ">".join(block_path) if block_path else ""
        html_content = str(block_tag) if isinstance(block_tag, Tag) else ""
        table_tag = block_tag if block_tag.name == "table" else block_tag.find("table")

        if table_tag is not None and table_tag is block_tag:
            # 标签本身就是表格：按行切分
            logger.debug(f"📊 表格类型，执行按行拼接切分")
            for text, row_start, row_end in _split_table_into_chunks(table_tag, title, max_node_words):
                summary, question = _generate_summary_and_question(
                    text, page_url, gen_question
                )
                doc_meta.append(_make_chunk_dict(
                    chunk_idx, page_name,
                    f"{title[:96]} 表格行{row_start}-{row_end}",
                    page_url, text, time_value, summary, question,
                    block_path=path_str, html_content=html_content,
                ))
                chunk_idx += 1
        else:
            # 混合内容或纯文本：检查是否包含嵌套表格
            nested_tables = block_tag.find_all("table")
            if nested_tables:
                # 混合内容块：分别处理表格前后的文本和表格本身
                for sub_tag, sub_title, sub_text in _extract_mixed_content(block_tag, title, max_node_words):
                    if not sub_text:
                        continue
                    summary, question = _generate_summary_and_question(
                        sub_text, page_url, gen_question
                    )
                    sub_html = str(sub_tag) if sub_tag and isinstance(sub_tag, Tag) else ""
                    doc_meta.append(_make_chunk_dict(
                        chunk_idx, page_name, sub_title[:128],
                        page_url, sub_text, time_value, summary, question,
                        block_path=path_str, html_content=sub_html,
                    ))
                    chunk_idx += 1
            else:
                # 纯文本类型
                text = clean_invisible(block_tag.get_text().replace("\x00", ""))
                if not text:
                    logger.debug("⚠️ 空内容，跳过")
                    continue

                summary, question = _generate_summary_and_question(
                    text, page_url, gen_question
                )
                doc_meta.append(_make_chunk_dict(
                    chunk_idx, page_name, title[:128],
                    page_url, text, time_value, summary, question,
                    block_path=path_str, html_content=html_content,
                ))
                chunk_idx += 1

    logger.info(f"✅ 所有块处理完毕，共生成 {len(doc_meta)} 条有效文档块")
    return doc_meta


# ======================== 文档块生成（异步） ========================

async def generate_block_documents_async(
    block_tree: list,
    max_node_words: int,
    page_url: str = "unknown.html",
    time_value: str = "",
    gen_question: bool = False,
    batch_size: int = 32,
    **kwargs,  # backward compat: use_vllm, summary_model, summary_tokenizer (ignored)
) -> list:
    """异步批量生成文档块，先切分再批量并发生成摘要。"""
    doc_meta = []
    chunk_idx = 0
    page_name = os.path.splitext(os.path.basename(page_url))[0]
    summary_tasks = []

    for block_tag, block_path, is_leaf in block_tree:
        title = extract_title_from_block(block_tag)
        path_str = ">".join(block_path) if block_path else ""
        html_content = str(block_tag) if isinstance(block_tag, Tag) else ""
        table_tag = block_tag if block_tag.name == "table" else block_tag.find("table")

        if table_tag is not None and table_tag is block_tag:
            # 标签本身就是表格：按行切分
            for text, row_start, row_end in _split_table_into_chunks(table_tag, title, max_node_words):
                doc_meta.append(_make_chunk_dict(
                    chunk_idx, page_name,
                    f"{title[:96]} 表格行{row_start}-{row_end}",
                    page_url, text, time_value,
                    block_path=path_str, html_content=html_content,
                ))
                summary_tasks.append((chunk_idx, text))
                chunk_idx += 1
        elif block_tag.find_all("table"):
            # 混合内容块：分别处理表格前后的文本和表格本身（与同步版一致）
            for sub_tag, sub_title, sub_text in _extract_mixed_content(block_tag, title, max_node_words):
                if not sub_text:
                    continue
                sub_html = str(sub_tag) if sub_tag and isinstance(sub_tag, Tag) else ""
                doc_meta.append(_make_chunk_dict(
                    chunk_idx, page_name, sub_title[:128],
                    page_url, sub_text, time_value,
                    block_path=path_str, html_content=sub_html,
                ))
                summary_tasks.append((chunk_idx, sub_text))
                chunk_idx += 1
        else:
            text = clean_invisible(block_tag.get_text().replace("\x00", ""))
            if not text:
                continue
            doc_meta.append(_make_chunk_dict(
                chunk_idx, page_name, title[:128],
                page_url, text, time_value,
                block_path=path_str, html_content=html_content,
            ))
            summary_tasks.append((chunk_idx, text))
            chunk_idx += 1

    # 批量并发生成摘要（与 gen_question=True 时并发生成 question，两者互不阻塞）
    logger.debug(f"{page_url} 任务准备完毕，开始分批并发生成 {len(summary_tasks)} 个摘要"
                 f"{'（含 question）' if gen_question else ''} ...")
    start = time.time()

    for i in range(0, len(summary_tasks), batch_size):
        batch = summary_tasks[i:i + batch_size]
        gather_tasks = [generate_summary_vllm_async(text, page_url) for _, text in batch]
        if gen_question:
            # 【修复 M1】异步路径此前接受 gen_question 参数却完全忽略，只回填
            # summary，question 永远为空，与同步版 `_generate_summary_and_question`
            # 产出结构不一致。这里并发生成 question，与摘要生成共用同一批次的
            # asyncio.gather，不额外增加往返轮次。
            gather_tasks += [generate_question_vllm_async(text, page_url) for _, text in batch]

        results = await asyncio.gather(*gather_tasks)
        n = len(batch)
        summaries, questions = results[:n], results[n:]

        for j, (meta_idx, _) in enumerate(batch):
            doc_meta[meta_idx]["summary"] = summaries[j]
            if gen_question:
                doc_meta[meta_idx]["question"] = questions[j]

    elapsed = time.time() - start
    logger.debug(f"{page_url} 所有块处理完毕，共生成 {len(doc_meta)} 条有效文档块, 耗时 {elapsed:.2f}")
    return doc_meta
