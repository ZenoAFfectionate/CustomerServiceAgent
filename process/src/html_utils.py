# -*- coding: utf-8 -*-
"""
HTML 清洗与结构化分块模块。

将原始 HTML 网页处理为高质量结构化输入，供下游 RAG 检索与向量化使用。

核心函数：
    - clean_html:          HTML 清洗主入口（去冗余 → 域包装 → 字符规范化）
    - build_block_tree:     将清洗后的 HTML 拆分为语义块
    - process_html_file:    单文件处理入口（清洗 + 表格展开 + time 保留）
    - expand_table_spans:   展开 colspan/rowspan 合并单元格
    - parse_time_tag:       提取起始 <time> 标签
"""

import re
import copy
import logging
from collections import deque
from typing import List, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag, Comment

logger = logging.getLogger("GlobalLogger")

# 表格保护标签集合（简化时保留 colspan/rowspan 属性）
_TABLE_PROTECTED_TAGS = frozenset({
    "table", "colgroup", "col", "thead", "tbody", "tr", "td", "th"
})
# 始终保留的属性
_ALWAYS_KEEP_ATTRS = frozenset({"data-block-type"})
# heading class 正则
_HEADING_CLASS_RE = re.compile(r"heading-h(\d)")
# 不可见字符正则
_INVISIBLE_RE = re.compile(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f]')
# 无内容价值的标签（在清洗时直接移除）
_NOISE_TAGS = frozenset({
    "script", "style", "svg", "input", "button", "select", "textarea",
    "option", "noscript", "iframe", "canvas", "video", "audio",
    "nav", "aside", "head", "title", "meta", "link", "footer",
})
# UI 噪声文本模式（进度条、导航等）
_UI_NOISE_RE = re.compile(
    r'^(?:\d+%|PROGRESS|CONTENTS|尚未开始|目录|↑顶部|搜索\.\.\.|←上一页|下一页→)$',
    re.IGNORECASE
)
# 模板残留文本模式
_TEMPLATE_TEXT_RE = re.compile(r'\{\{[^}]*PLACEHOLDER[^}]*\}\}', re.IGNORECASE)


# ======================== <time> 标签 ========================

_TIME_TAG_RE = re.compile(r"^\s*<time[^>]*?>(.*?)</time>", re.IGNORECASE | re.DOTALL)
_TIME_TAG_FULL_RE = re.compile(r"^\s*<time[^>]*?>.*?</time>", re.IGNORECASE | re.DOTALL)


def parse_time_tag(html: str) -> Tuple[str, str]:
    """提取 HTML 开头的 <time> 标签内容。

    Args:
        html: 原始 HTML 字符串

    Returns:
        (time_value, remaining_html)：时间值和剩余 HTML
    """
    match = _TIME_TAG_RE.match(html)
    if match:
        time_value = match.group(1).strip()
        remaining = html[match.end():].lstrip()
        return time_value, remaining
    return "", html


# ======================== HTML 清洗 ========================

def simplify_html_keep_table(soup: BeautifulSoup, keep_attr: bool = False) -> str:
    """保留表格结构的 HTML 简化。

    - 移除噪声标签（script/style/svg/input/button/nav/aside 等）
    - 移除隐藏元素（style="display:none" 或 class 含 hidden）
    - 移除模板残留文本（{{PLACEHOLDER}}）
    - 将 heading-hN class 转换为 data-block-type 属性
    - 清除非保护标签的多余属性
    - 移除空标签和 HTML 注释
    - 合并冗余包装标签
    """
    # 移除噪声标签（script, style, svg, input, button, nav, aside 等）
    for tag in soup(list(_NOISE_TAGS)):
        tag.decompose()

    # 移除隐藏元素
    _remove_hidden_elements(soup)

    # 移除模板残留文本（仅替换 {{PLACEHOLDER}} 模式，保留其余文本）
    for text_node in soup.find_all(string=_TEMPLATE_TEXT_RE.search):
        cleaned_text = _TEMPLATE_TEXT_RE.sub("", str(text_node))
        text_node.replace_with(cleaned_text)

    if not keep_attr:
        for tag in soup.find_all(True):
            # heading-hN class → data-block-type
            class_list = tag.get("class", [])
            if isinstance(class_list, str):
                class_list = class_list.split()
            for cls in class_list:
                m = _HEADING_CLASS_RE.match(cls)
                if m:
                    tag.attrs["data-block-type"] = f"heading{m.group(1)}"
                    break

            # 清除属性
            if tag.name in _TABLE_PROTECTED_TAGS:
                tag.attrs = {
                    k: v for k, v in tag.attrs.items()
                    if k in ("colspan", "rowspan") or k in _ALWAYS_KEEP_ATTRS
                }
            else:
                tag.attrs = {
                    k: v for k, v in tag.attrs.items() if k in _ALWAYS_KEEP_ATTRS
                }

    # 移除空标签（表格标签有子元素时保留）
    _remove_empty_tags(soup)

    # 移除 <a> 的 href
    for tag in soup.find_all("a"):
        tag.attrs.pop("href", None)

    # 移除 HTML 注释
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # 合并冗余包装标签（迭代多次，确保深层嵌套也被清理）
    for _ in range(3):
        _unwrap_redundant_wrappers(soup)

    # 再次清理可能因 unwrap 产生的空标签
    _remove_empty_tags(soup)

    # 清除空行
    return "\n".join(line for line in str(soup).split("\n") if line.strip())


def _remove_hidden_elements(soup: BeautifulSoup) -> None:
    """移除通过 CSS 隐藏的元素（display:none, visibility:hidden, class 含 hidden）。"""
    for tag in soup.find_all(True):
        style = tag.get("style", "")
        if "display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", ""):
            tag.decompose()
            continue
        class_list = tag.get("class", [])
        if isinstance(class_list, str):
            class_list = class_list.split()
        if any("hidden" in cls.lower() for cls in class_list):
            tag.decompose()


def _remove_empty_tags(soup: BeautifulSoup) -> None:
    """迭代移除所有无文本内容的空标签。

    处理以下情况：
    - 完全空的标签 <div></div>
    - 仅含空白的标签 <div>   </div>
    - 仅含 <br> 的标签 <div><br></div>
    - 嵌套空标签 <div><span></span></div>
    """
    changed = True
    while changed:
        changed = False
        for tag in soup.find_all():
            if tag.name in _TABLE_PROTECTED_TAGS and tag.contents:
                continue
            # 检查是否有实际文本内容
            text = tag.get_text().strip()
            if text:
                continue
            # 检查是否仅含 <br> 或空标签
            child_tags = [c for c in tag.contents if isinstance(c, Tag)]
            if not child_tags:
                # 纯空标签
                tag.decompose()
                changed = True
            elif all(c.name == "br" or (not c.get_text().strip() and not [g for g in c.contents if isinstance(g, Tag)]) for c in child_tags):
                # 仅含 <br> 或空子标签
                tag.decompose()
                changed = True


def _unwrap_redundant_wrappers(soup: BeautifulSoup) -> None:
    """合并冗余包装标签。

    当父标签只有一个子标签且文本内容一致时，用子标签替换父标签。
    优先处理最深层标签（自底向上），确保多层嵌套的 div 包装被完全展开。
    """
    # 自底向上处理：先收集所有候选标签，按深度排序
    candidates = [
        tag for tag in soup.find_all()
        if tag.name not in _TABLE_PROTECTED_TAGS
        and tag.name not in ("h1", "h2", "h3", "h4", "h5", "h6", "table")
        and not any(attr in tag.attrs for attr in _ALWAYS_KEEP_ATTRS)
    ]

    # 按深度从深到浅排序（先处理内层，再处理外层）
    candidates.sort(key=lambda t: -len(list(t.parents)))

    for tag in candidates:
        # tag 可能已被 decompose 或 unwrap
        if not tag.parent and tag is not soup:
            continue
        # 只有一个子标签时才考虑展开
        child_tags = [c for c in tag.contents if isinstance(c, Tag)]
        if len(child_tags) != 1:
            continue
        # 检查是否有裸文本（父标签有直接文本内容时不能展开）
        bare_text = "".join(
            str(c) for c in tag.contents
            if not isinstance(c, Tag)
        ).strip()
        if bare_text:
            continue
        # 文本内容一致检查（忽略空白）
        tag_text = re.sub(r"[\n\t ]+", "", tag.get_text())
        child_text = re.sub(r"[\n\t ]+", "", child_tags[0].get_text())
        if tag_text != child_text:
            continue
        tag.replace_with_children()


def warp_domains(html: str) -> str:
    """将 HTML 中的 <hX> 和 <table> 标签进行语义包装。

    1. 将 data-block-type="headingN" 转换为标准 <hN> 标签
    2. 根据标题层级包装 <div class="hN_domain">
    3. 对 <table> 外包一层 <div class="table_domain">
    """
    soup = BeautifulSoup(html, "html.parser")

    # Step 1: 转换 heading 标签
    _convert_headings(soup)

    # Step 2: 按标题层级包装
    _wrap_heading_domains(soup)

    # Step 3: 包装表格
    _wrap_table_domains(soup)

    return str(soup)


def _convert_headings(soup: BeautifulSoup) -> None:
    """将 data-block-type="headingN" 转换为 <hN>。"""
    block_map = {f"heading{i}": f"h{i}" for i in range(1, 7)}
    for tag in soup.find_all(attrs={"data-block-type": True}):
        block_type = tag.get("data-block-type")
        if block_type in block_map:
            new_tag = soup.new_tag(block_map[block_type])
            for child in list(tag.contents):
                new_tag.append(child.extract())
            tag.replace_with(new_tag)


def _get_heading_level(tag_name: str):
    """返回标题级别（1-6），非标题返回 None。"""
    if tag_name and tag_name.startswith("h") and tag_name[1:].isdigit():
        level = int(tag_name[1:])
        if 1 <= level <= 6:
            return level
    return None


def _wrap_heading_domains(soup: BeautifulSoup) -> None:
    """根据标题层级将内容包装到 <div class="hN_domain"> 中。"""
    body_nodes = list(soup.contents)
    soup.clear()

    has_heading = any(
        isinstance(n, Tag) and _get_heading_level(n.name) is not None
        for n in body_nodes
    )

    if not has_heading:
        # 无标题：整体包一层 isolated_domain
        wrapper = soup.new_tag("div", **{"class": "isolated_domain"})
        for node in body_nodes:
            wrapper.append(node)
        soup.append(wrapper)
        return

    # 有标题：按标题层级递归包装（迭代实现，避免栈溢出）
    for node in _process_heading_nodes(soup, body_nodes):
        soup.append(node)


def _process_heading_nodes(soup: BeautifulSoup, nodes: list) -> list:
    """迭代处理节点列表，按标题层级包装。"""
    result = []
    i = 0
    while i < len(nodes):
        node = nodes[i]
        level = _get_heading_level(node.name) if isinstance(node, Tag) else None

        if level is not None:
            # 找到同级或更高级标题为止
            children = [node]
            j = i + 1
            while j < len(nodes):
                next_node = nodes[j]
                next_level = _get_heading_level(next_node.name) if isinstance(next_node, Tag) else None
                if next_level is not None and next_level <= level:
                    break
                children.append(next_node)
                j += 1

            # 递归处理子节点
            wrapped = _process_heading_nodes(soup, children[1:])
            wrapper = soup.new_tag("div", **{"class": f"h{level}_domain"})
            wrapper.append(children[0])
            for c in wrapped:
                wrapper.append(c)
            result.append(wrapper)
            i = j
        else:
            result.append(node)
            i += 1
    return result


def _wrap_table_domains(soup: BeautifulSoup) -> None:
    """对所有 table 标签外包一层 <div class='table_domain'>。"""
    for table in soup.find_all("table"):
        if not table.find_parent("div", class_="table_domain"):
            wrapper = soup.new_tag("div", **{"class": "table_domain"})
            table.insert_before(wrapper)
            wrapper.append(table.extract())


def expand_table_spans(html: str) -> str:
    """展开 HTML 表格中的 colspan/rowspan 合并单元格，生成标准矩阵表格。

    忽略 rowspan=0 / colspan=0 的占位单元格。
    """
    soup = BeautifulSoup(html, "html.parser")

    for table in soup.find_all("table"):
        _expand_single_table(soup, table)

    return str(soup)


def _expand_single_table(soup: BeautifulSoup, table: Tag) -> None:
    """展开单个表格的合并单元格。"""
    rows = table.find_all("tr")
    if not rows:
        return

    grid = []       # grid[row][col] = cell
    max_cols = 0

    for row_idx, row in enumerate(rows):
        if len(grid) <= row_idx:
            grid.append([])

        col_idx = 0
        for cell in row.find_all(["td", "th"]):
            rowspan, colspan = _safe_int_attr(cell, "rowspan", 1), _safe_int_attr(cell, "colspan", 1)

            # 跳过无效的 0 值
            if rowspan == 0 or colspan == 0:
                continue

            # 找下一个空白列
            while col_idx < len(grid[row_idx]) and grid[row_idx][col_idx] is not None:
                col_idx += 1

            # 清除合并属性
            cell.attrs.pop("rowspan", None)
            cell.attrs.pop("colspan", None)

            # 填充网格
            for r in range(rowspan):
                target_row = row_idx + r
                while len(grid) <= target_row:
                    grid.append([])
                while len(grid[target_row]) < col_idx + colspan:
                    grid[target_row].append(None)
                for c in range(colspan):
                    if r == 0 and c == 0:
                        grid[target_row][col_idx + c] = cell
                    else:
                        grid[target_row][col_idx + c] = copy.copy(cell)

            col_idx += colspan
            max_cols = max(max_cols, col_idx)

    # 构建新表格
    new_table = soup.new_tag("table")
    for row_cells in grid:
        tr = soup.new_tag("tr")
        for cell in row_cells[:max_cols]:
            if cell is not None:
                tr.append(cell)
            else:
                empty = soup.new_tag("td")
                empty.string = ""
                tr.append(empty)
        new_table.append(tr)

    table.replace_with(new_table)


def _safe_int_attr(tag: Tag, attr: str, default: int = 1) -> int:
    """安全读取标签的整型属性值。"""
    try:
        return int(tag.get(attr, default))
    except (ValueError, TypeError):
        return default


def clean_xml(html: str) -> str:
    """移除 XML 声明和 DOCTYPE。"""
    html = re.sub(r"<\?xml.*?>", "", html)
    html = re.sub(r"<!DOCTYPE.*?>", "", html, flags=re.IGNORECASE)
    return html


def clean_html_text(html_content: str) -> str:
    """清理 HTML 文本中的 markdown 块标记、多余换行和空格。

    仅移除标签间的纯空白换行，不会合并跨标签的文本内容。
    """
    # 去除 markdown 风格的 ```html``` 块
    html_content = re.sub(r'\n*```html\n*', '', html_content)
    html_content = re.sub(r'\n*```\n*', '', html_content)

    # 移除标签间的纯空白换行符：>\n\n< → ><（仅当 > 和 < 之间全是空白时）
    html_content = re.sub(r'>\s*\n\s*<', '><', html_content)

    # 移除标签开头的多余空白（保留标签内文本的空格）
    html_content = re.sub(r'>\s+\n', '>\n', html_content)

    # 合并连续空行为单个换行
    html_content = re.sub(r'\n{3,}', '\n\n', html_content)

    return html_content


def clean_html(html: str, keep_att: bool = False) -> str:
    """HTML 清洗主入口：简化 → 域包装 → 去 XML 声明 → 文本规范化。"""
    soup = BeautifulSoup(html, "html.parser")
    html = simplify_html_keep_table(soup, keep_att)
    html = warp_domains(html)
    html = clean_xml(html)
    html = clean_html_text(html)
    return html


def process_html_file(source_path: str, target_path: str) -> str:
    """清洗单个 HTML 文件，保留 <time> 标签，写入目标路径。

    Args:
        source_path: 源 HTML 文件路径
        target_path: 输出文件路径

    Returns:
        清洗后的 HTML 字符串
    """
    with open(source_path, "r", encoding="utf-8") as f:
        html = f.read()

    # 提取 <time> 标签（复用 parse_time_tag 的正则）
    time_match = _TIME_TAG_FULL_RE.match(html)
    time_tag = ""
    remaining_html = html

    if time_match:
        time_tag = time_match.group(0).strip()
        remaining_html = html[time_match.end():].lstrip()

    # 清洗 + 展开表格
    simplified_html = clean_html(remaining_html, keep_att=False)
    simplified_html = expand_table_spans(simplified_html)

    # 将 <time> 标签补回开头
    final_html = time_tag + simplified_html

    # 写入目标路径
    with open(target_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    logger.info(f"✅ 已处理 {source_path} → {target_path}")
    logger.debug(f"原始长度: {len(html)}, 清理后长度: {len(final_html)}")
    return final_html


# ======================== HTML 分块 ========================

def _count_words(tag, zh_char: bool) -> int:
    """计算 Tag 的词数（zh_char=True 时为字符数）。"""
    text = tag.get_text()
    if zh_char:
        return len(text)
    return len(text.split())


def _count_str_words(s: str, zh_char: bool) -> int:
    """计算字符串的词数。"""
    if zh_char:
        return len(s)
    return len(s.split())


def _is_ui_noise(tag: Tag) -> bool:
    """判断标签是否为纯 UI 噪声（进度条、导航文本等）。

    检测规则：
    - 标签文本仅含 UI 噪声关键词（PROGRESS、CONTENTS、0% 等）
    - 标签所有 stripped_strings 都匹配 UI 噪声模式
    """
    strings = list(tag.stripped_strings)
    if not strings:
        return False
    # 如果所有文本都匹配 UI 噪声模式，则判定为噪声
    return all(_UI_NOISE_RE.match(s.strip()) for s in strings)


def _make_bare_text_tag(name: str, texts: list):
    """构造仅包含裸文本（不含已被单独处理为子块的子标签）的独立 Tag。

    `build_block_tree` 在 BFS 拆分节点时，若节点同时存在裸文本与被单独处理
    的子标签（各自已成块或已入队列），此前会把整棵 `tree`（其 `get_text()`
    会包含全部子标签文本）也作为独立块 append，导致父块与子块内容重复
    （审查报告 H4）。这里改为只用收集到的裸文本片段构造一个全新的最小 Tag，
    与已单独成块的子标签内容互不重叠。

    Args:
        name: 新 Tag 的标签名（沿用原节点名，仅用于路径展示，不影响语义）
        texts: 裸文本片段列表（NavigableString 原文 + M7 中回收的小子标签文本）

    Returns:
        仅含拼接后裸文本的新 Tag；裸文本为空时返回 None。
    """
    text = "".join(texts).strip()
    if not text:
        return None
    new_soup = BeautifulSoup("", "html.parser")
    new_tag = new_soup.new_tag(name or "div")
    new_tag.append(NavigableString(text))
    return new_tag


def build_block_tree(
    html: str,
    max_node_words: int = 512,
    min_node_words: int = 32,
    zh_char: bool = False,
) -> Tuple[list, str]:
    """将 HTML 分割成结构化语义块。

    Args:
        html: 清洗后的 HTML 字符串
        max_node_words: 每个块的最大词数
        min_node_words: 每个块的最小词数
        zh_char: 为 True 时按字符数计算（适配中文）

    Returns:
        (blocks, raw_html)：
        - blocks: [(Tag, path, is_leaf), ...]
        - raw_html: 原始 HTML 字符串
    """
    soup = BeautifulSoup(html, "html.parser")
    total_words = _count_words(soup, zh_char)

    if total_words < min_node_words:
        # 【修复 M7】此前整页词数不足 min_node_words 时直接返回空列表，短
        # 帮助页/零散小段落会被完全丢弃、永不进入知识库。只要页面确实有
        # 内容（total_words > 0），仍应保留为单个整页块，而非直接丢弃；
        # 仅当页面确实完全为空（total_words == 0）时才返回空列表。
        if total_words > 0:
            return [(soup, [], True)], str(soup)
        return [], str(soup)

    if total_words > max_node_words:
        # BFS 拆分：用 deque 替代 list.pop(0) 提升 O(1) 出队性能
        queue = deque([(soup, [])])
        target_trees = []

        while queue:
            tree, path = queue.popleft()

            # 统计子标签类型
            tag_children_count = {}
            for child in tree.contents:
                if isinstance(child, Tag):
                    tag_children_count[child.name] = tag_children_count.get(child.name, 0) + 1

            # 子标签序号计数器（用于区分同名标签）
            child_idx_map = {name: 0 for name in tag_children_count}
            bare_word_count = 0
            # 【修复 H4/M7】收集裸文本的实际内容（而非仅统计词数），供拆分
            # 结束后构造"仅含裸文本"的独立 Tag（而非整棵 tree），避免与已
            # 单独成块的子标签内容重复；同时把因词数不足被跳过的小子标签
            # 文本回收进来，避免小段落信息永久丢失。
            bare_texts = []

            for child in tree.contents:
                if isinstance(child, Tag):
                    # 表格/表体作为整体保留
                    if child.name in ("table", "tbody"):
                        if _count_words(child, zh_char) >= min_node_words:
                            target_trees.append((child, path + [child.name], True))
                        else:
                            # 【修复 M7】过小的表格不再直接丢弃，回收其文本到
                            # 父节点裸文本中。
                            text = child.get_text()
                            if text.strip():
                                bare_texts.append(text)
                                bare_word_count += _count_str_words(text, zh_char)
                        continue

                    # 跳过 UI 噪声块（进度条、导航文本等）
                    if _is_ui_noise(child):
                        continue

                    # 为同名子标签编号（div0, div1, ...）
                    if tag_children_count[child.name] > 1:
                        new_name = f"{child.name}{child_idx_map[child.name]}"
                        child_idx_map[child.name] += 1
                    else:
                        new_name = child.name

                    new_path = path + [new_name]
                    words = _count_words(child, zh_char)

                    if words < min_node_words:
                        # 【修复 M7】词数过少不单独成块，但文本不应直接丢弃：
                        # 回收到父节点的裸文本中，避免小段落信息永久丢失
                        # （此前既不计入裸文本也不单独成块，永久丢失）。
                        text = child.get_text()
                        if text.strip():
                            bare_texts.append(text)
                            bare_word_count += _count_str_words(text, zh_char)
                        continue
                    if words > max_node_words and len(new_path) < 64:
                        queue.append((child, new_path))
                    else:
                        target_trees.append((child, new_path, True))
                else:
                    # NavigableString
                    text = str(child)
                    bare_texts.append(text)
                    bare_word_count += _count_str_words(text, zh_char)

            # 纯文本节点：论文 Algorithm 1 规定，当节点被拆分时，
            # 节点的裸文本（直接附属于节点的文本，不在子标签中）应作为独立块。
            # 【修复 H4】此前无论哪种情形都把整棵 `tree`（含全部子标签文本）
            # append 为块，与已单独处理的子标签内容重复。现改为仅用收集到的
            # 裸文本片段构造一个全新的最小 Tag，与子块内容互不重叠。
            if bare_word_count >= min_node_words or (bare_word_count > 0 and tag_children_count):
                bare_tag = _make_bare_text_tag(tree.name if isinstance(tree, Tag) else "div", bare_texts)
                if bare_tag is not None:
                    target_trees.append((bare_tag, path, True))

        return target_trees, str(soup)

    # 总词数在 min~max 之间：检查是否需要按 heading 拆分
    soup_children = [c for c in soup.contents if isinstance(c, Tag)]

    # 如果只有一个子节点，检查它是否包含多个 heading
    if len(soup_children) == 1:
        only_child = soup_children[0]

        # 找到 heading 的直接父节点（可能需要向下穿透 wrapper 标签）
        heading_parent = _find_heading_parent(only_child)
        if heading_parent is not None:
            split_blocks = _split_by_headings(heading_parent, zh_char, min_node_words)
            if split_blocks:
                return split_blocks, str(soup)

        if _count_words(only_child, zh_char) >= min_node_words:
            return [(only_child, [only_child.name], True)], str(soup)
        return [], str(soup)

    if len(soup_children) > 1:
        # 多个顶级子节点：按 heading 拆分
        split_blocks = _split_by_headings(soup, zh_char, min_node_words)
        if split_blocks:
            return split_blocks, str(soup)

        # 无 heading：合并为一个块
        new_soup = BeautifulSoup("", "html.parser")
        new_tag = new_soup.new_tag("html")
        new_soup.append(new_tag)
        valid_children = []
        for child in soup_children:
            if _is_ui_noise(child):
                continue
            if _count_words(child, zh_char) >= min_node_words:
                new_tag.append(child)
                valid_children.append(child)
        if valid_children:
            return [(new_tag, ["html"], True)], str(soup)

    return [], str(soup)


def _find_heading_parent(node: Tag) -> Tag:
    """找到 heading 标签的直接父节点。

    如果 node 的直接子节点中有 heading，返回 node。
    否则向下穿透 wrapper 标签（div, body, html 等），找到包含 heading 的层级。
    """
    # 检查直接子节点中是否有 heading
    for child in node.children:
        if isinstance(child, Tag) and _get_heading_level(child.name) is not None:
            return node

    # 向下穿透 wrapper 标签
    for child in node.children:
        if isinstance(child, Tag) and child.name in ("div", "body", "html", "main", "section", "article"):
            result = _find_heading_parent(child)
            if result is not None:
                return result

    return None


def _split_by_headings(parent, zh_char: bool, min_node_words: int) -> list:
    """按 heading 标签将内容切分为多个独立块。

    每个 heading 及其后到下一个同级或更高级 heading 之前的内容归为一个块。
    过滤 UI 噪声子节点（进度条、导航文本等）。
    """
    # 先复制 children 列表，避免 extract() 修改正在迭代的列表
    all_children = list(parent.contents)
    blocks = []
    current_children = []
    current_heading = None

    for child in all_children:
        if isinstance(child, Tag) and _get_heading_level(child.name) is not None:
            # 遇到 heading：提交之前的块
            if current_children:
                block = _build_block_from_children(current_children)
                if block and _count_words(block, zh_char) >= min_node_words:
                    level = _get_heading_level(current_heading.name) if current_heading else 0
                    blocks.append((block, [f"h{level}_section"], True))
            # 开始新块
            current_children = [child]
            current_heading = child
        elif isinstance(child, Tag) and _is_ui_noise(child):
            # 跳过 UI 噪声
            continue
        else:
            current_children.append(child)

    # 提交最后一个块
    if current_children:
        block = _build_block_from_children(current_children)
        if block and _count_words(block, zh_char) >= min_node_words:
            level = _get_heading_level(current_heading.name) if current_heading else 0
            blocks.append((block, [f"h{level}_section"], True))

    return blocks


def _build_block_from_children(children: list) -> Tag:
    """从子节点列表构建一个新的包装标签（使用 copy 避免修改原 DOM）。"""
    wrapper = BeautifulSoup("", "html.parser").new_tag("div")
    for child in children:
        if isinstance(child, Tag):
            # 使用 copy 避免修改原 DOM
            wrapper.append(copy.copy(child))
        else:
            wrapper.append(copy.copy(child))
    return wrapper
