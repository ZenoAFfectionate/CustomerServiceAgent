# -*- coding: utf-8 -*-
"""
抖店规则中心爬虫核心逻辑（支持增量更新）。

工作流程：
    1. 启动 Playwright 无头浏览器，导航至规则中心页面。
    2. 通过 page.evaluate() 调用后端 API，获取分类菜单树。
    3. 递归遍历菜单树，收集所有叶子节点（具体分类）。
    4. 加载本地已有的 metadata.json（增量更新基准）。
    5. 分页获取每个分类下的文章列表，汇总为"当前线上全集"。
    6. 逐篇比对线上 update_at 与本地存储的时间戳：
       - 新文章（本地无）       → 爬取并保存
       - 已更新（线上时间戳更新） → 重新爬取并替换
       - 未变化（时间戳相同）    → 跳过，不发起页面请求
    7. 可选清理：线上已删除但本地仍存在的文章（--cleanup）。
    8. 保存更新后的 metadata.json。

用法：
    from crawler.crawler import RulesCrawler
    crawler = RulesCrawler()
    crawler.run()
"""
import os
import re
import json
import time
from datetime import datetime
from typing import Optional

from playwright.sync_api import sync_playwright, Page

from crawler import config as cfg


# ======================== 工具函数 ========================

def _sanitize_filename(name: str) -> str:
    """将标题清洗为安全的文件名。"""
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = name.strip().strip(".")
    if len(name) > 120:
        name = name[:120]
    return name


def _ts_to_str(ts: int) -> str:
    """Unix 时间戳 → 'YYYY-MM-DD HH:MM:SS' 字符串。"""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _now_str() -> str:
    """当前时间 → 'YYYY-MM-DD HH:MM:SS' 字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ======================== 正文提取 JS ========================

# 在浏览器中执行的 JS：提取正文 HTML 并清理 UI 噪音元素
_EXTRACT_JS = """
() => {
    const selectors = [
        '.eschool-doc-content-wrapper',
        '.editor-kit-container .ace-editor-wrapper',
        '.editor-kit-container',
        '.ace-editor-wrapper',
        '.article-content',
        '[class*="ace-editor"]',
        '[class*="rich-text"]',
    ];
    let container = null;
    for (const sel of selectors) {
        const el = document.querySelector(sel);
        if (el && el.innerText.trim().length > 50) {
            container = el;
            break;
        }
    }
    if (!container) {
        const aceLines = document.querySelectorAll('.ace-line');
        if (aceLines.length > 0) {
            container = aceLines[0].parentElement;
        }
    }
    if (!container) return null;

    const clone = container.cloneNode(true);
    const noiseSelectors = [
        '.ace-table-fullscreen-icon',
        '.ace-table-fullscreen-navbar',
        '.ace-table-fullscreen-mask',
        '[class*="fullscreen"]',
        '[class*="toolbar"]:not([class*="ace-line"])',
    ];
    for (const sel of noiseSelectors) {
        clone.querySelectorAll(sel).forEach(el => el.remove());
    }
    return clone.innerHTML;
}
"""


# ======================== 核心爬虫 ========================

class RulesCrawler:
    """抖店规则中心爬虫（支持增量更新）。

    Args:
        max_articles: 限制最大爬取文章数（0=不限制，调试用）。
        category_filter: 只爬取分类名模糊匹配此值的分类（空字符串=不过滤）。
        exclude_categories: 排除顶级分类名列表（如 ["历史规则/协议"]）。
        full_crawl: 强制全量爬取，忽略本地时间戳（默认 False）。
        cleanup: 清理线上已删除但本地仍存在的文章（默认 False）。
    """

    def __init__(self, max_articles: int = 0, category_filter: str = "",
                 exclude_categories: list[str] = None,
                 full_crawl: bool = False, cleanup: bool = False):
        self._playwright = None
        self._browser = None
        self._page: Optional[Page] = None
        # metadata: article_id -> entry dict（内存中用 dict 做 O(1) 查找）
        self._metadata: dict[str, dict] = {}
        self._max_articles = max_articles
        self._category_filter = category_filter
        self._exclude_categories = exclude_categories or []
        self._full_crawl = full_crawl
        self._cleanup = cleanup
        # 统计
        self._stats = {"new": 0, "updated": 0, "skipped": 0, "failed": 0, "deleted": 0}

    # ---------- 浏览器生命周期 ----------

    def _start(self):
        """启动 Playwright 与浏览器。"""
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=cfg.HEADLESS)
        self._page = self._browser.new_page()
        self._page.set_default_timeout(cfg.NAV_TIMEOUT_MS)
        print(f"[crawler] 浏览器已启动 (headless={cfg.HEADLESS})")

    def _close(self):
        """关闭浏览器与 Playwright。"""
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        print("[crawler] 浏览器已关闭")

    # ---------- 元数据管理 ----------

    def _load_existing_metadata(self):
        """加载本地已有的 metadata.json，构建 article_id → entry 字典。"""
        if not os.path.exists(cfg.METADATA_FILE):
            print("[crawler] 未找到现有 metadata.json，将进行全量爬取")
            return

        try:
            with open(cfg.METADATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 兼容旧版 list 格式和 dict 格式
            if isinstance(data, list):
                self._metadata = {m["article_id"]: m for m in data}
            elif isinstance(data, dict):
                self._metadata = data
            else:
                self._metadata = {}
            print(f"[crawler] 已加载现有元数据: {len(self._metadata)} 篇")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[crawler] metadata.json 解析失败 ({e})，将进行全量爬取")
            self._metadata = {}

    def _save_metadata(self):
        """将元数据保存为 metadata.json（list 格式，保持向后兼容）。"""
        data = list(self._metadata.values())
        with open(cfg.METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[crawler] 元数据已保存: {cfg.METADATA_FILE} (共 {len(data)} 篇)")

    # ---------- API 调用 ----------

    def _fetch_menu_tree(self) -> list[dict]:
        """获取规则中心的完整分类菜单树。"""
        url = f"{cfg.API_MENU_LIST}?node_id={cfg.ROOT_NODE_ID}&graphId={cfg.GRAPH_ID}"
        result = self._page.evaluate(
            """async (url) => {
                const resp = await fetch(url);
                return resp.json();
            }""",
            url,
        )
        if result.get("code") != 0:
            raise RuntimeError(f"菜单 API 返回异常: {result.get('msg')}")
        print(f"[crawler] 获取菜单树成功，顶级分类数: {len(result['data'])}")
        return result["data"]

    def _fetch_article_list(self, node_id: str) -> list[dict]:
        """分页获取某个分类节点下的全部文章列表。"""
        all_articles = []
        page_num = 1

        while True:
            url = (
                f"{cfg.API_ARTICLE_LIST}"
                f"?node_id={node_id}&page_size={cfg.PAGE_SIZE}&page={page_num}"
            )
            result = self._page.evaluate(
                """async (url) => {
                    const resp = await fetch(url);
                    return resp.json();
                }""",
                url,
            )

            if result.get("code") != 0:
                print(f"[crawler]  警告: node_id={node_id} page={page_num} 返回异常: {result.get('msg')}")
                break

            data = result.get("data", {})
            articles = data.get("articles", [])
            total = data.get("total", 0)

            all_articles.extend(articles)
            print(f"[crawler]  node_id={node_id}  page {page_num}: 获取 {len(articles)} 篇 (累计 {len(all_articles)}/{total})")

            if len(all_articles) >= total or not articles:
                break

            page_num += 1
            time.sleep(0.3)

        return all_articles

    def _fetch_all_articles(self, leaf_nodes: list[tuple[str, list[str]]]) -> dict[str, dict]:
        """
        遍历所有叶子分类，获取线上全部文章列表。

        返回: {article_id: {article原始字段 + "_category_path": [...]} }
        """
        all_articles: dict[str, dict] = {}

        for idx, (node_id, category_path) in enumerate(leaf_nodes, 1):
            category_str = "/".join(category_path)
            print(f"[crawler]  [{idx}/{len(leaf_nodes)}] 获取文章列表: {category_str}")

            articles = self._fetch_article_list(node_id)

            for art in articles:
                art_id = art["id"]
                # 附带分类路径信息
                art["_category_path"] = category_path
                art["_category_str"] = category_str
                all_articles[art_id] = art

        return all_articles

    # ---------- 菜单树遍历 ----------

    @staticmethod
    def _collect_leaf_nodes(menu_tree: list[dict], path: list[str] = None) -> list[tuple[str, list[str]]]:
        """
        递归遍历菜单树，返回所有叶子节点。

        返回: [(node_id, 分类路径), ...]
        例如: ("11674", ["商家管理", "招商入驻", "保证金管理"])
        """
        if path is None:
            path = []

        leaves = []
        for node in menu_tree:
            current_path = path + [node["name"]]
            children = node.get("menus")

            if children:
                leaves.extend(RulesCrawler._collect_leaf_nodes(children, current_path))
            else:
                leaves.append((node["id"], current_path))

        return leaves

    # ---------- 增量判断 ----------

    def _determine_action(self, article_id: str, online_update_at: int) -> str:
        """
        判断文章需要执行的操作。

        返回: "new" | "update" | "skip"
        """
        if self._full_crawl or article_id not in self._metadata:
            return "new" if article_id not in self._metadata else "update"

        local_update_at = self._metadata[article_id].get("update_at", 0)
        if online_update_at > local_update_at:
            return "update"
        return "skip"

    # ---------- 文章内容提取 ----------

    def _extract_article_html(self, article_id: str, update_time: str) -> Optional[str]:
        """
        导航至文章详情页，等待 .ace-line 渲染后提取正文 HTML。

        返回的 HTML 以 <time> 标签开头（与现有数据格式一致），后跟正文内容。
        提取失败返回 None。
        """
        article_url = cfg.ARTICLE_URL_TMPL.format(article_id=article_id)

        try:
            self._page.goto(article_url, wait_until="domcontentloaded", timeout=cfg.NAV_TIMEOUT_MS)
        except Exception as e:
            print(f"[crawler]    导航失败 {article_id}: {e}")

        # 等待正文元素出现
        try:
            self._page.wait_for_selector(".ace-line", timeout=cfg.NAV_TIMEOUT_MS)
        except Exception:
            try:
                self._page.wait_for_selector(
                    ".eschool-doc-content-wrapper, .article-content", timeout=5000
                )
            except Exception:
                print(f"[crawler]    未找到正文元素 {article_id}")
                return None

        self._page.wait_for_timeout(cfg.RENDER_WAIT_MS)

        content_html = self._page.evaluate(_EXTRACT_JS)

        if not content_html:
            print(f"[crawler]    正文提取为空 {article_id}")
            return None

        time_tag = f"<time>{update_time}</time>" if update_time else ""
        return f"{time_tag}{content_html}"

    def _extract_with_retry(self, article_id: str, update_time: str) -> Optional[str]:
        """提取文章 HTML，失败时重试一次。"""
        html = self._extract_article_html(article_id, update_time)
        if html is not None:
            return html

        print(f"[crawler]    重试中...")
        time.sleep(2)
        return self._extract_article_html(article_id, update_time)

    # ---------- 存储 ----------

    def _save_article_html(self, category_path: list[str], title: str,
                           html: str, article_id: str,
                           old_file_path: str = "") -> str:
        """
        保存文章 HTML 到 data/{category_path}/{title}.html。

        如果文章已存在且文件路径发生变化（分类或标题变更），
        自动删除旧文件，避免残留。

        Args:
            old_file_path: 旧的文件相对路径（增量更新时传入）。

        返回: 新的文件相对路径。
        """
        dir_path = os.path.join(cfg.OUTPUT_DIR, *category_path)
        os.makedirs(dir_path, exist_ok=True)

        filename = _sanitize_filename(title) + ".html"
        filepath = os.path.join(dir_path, filename)
        new_rel_path = os.path.relpath(filepath, cfg.OUTPUT_DIR)

        # 如果旧文件路径与新路径不同，删除旧文件
        if old_file_path and old_file_path != new_rel_path:
            old_abs = os.path.join(cfg.OUTPUT_DIR, old_file_path)
            if os.path.exists(old_abs):
                os.remove(old_abs)
                print(f"[crawler]    旧文件已删除: {old_file_path}")

        # 如果目标文件已存在（同名不同文章），加 article_id 后缀
        if os.path.exists(filepath) and old_file_path != new_rel_path:
            base, ext = os.path.splitext(filename)
            filepath = os.path.join(dir_path, f"{base}_{article_id}{ext}")
            new_rel_path = os.path.relpath(filepath, cfg.OUTPUT_DIR)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        return new_rel_path

    def _cleanup_deleted_articles(self, current_article_ids: set[str]):
        """清理线上已删除但本地仍存在的文章。"""
        deleted_ids = [aid for aid in self._metadata if aid not in current_article_ids]
        for aid in deleted_ids:
            entry = self._metadata[aid]
            filepath = os.path.join(cfg.OUTPUT_DIR, entry.get("file_path", ""))
            if os.path.exists(filepath):
                os.remove(filepath)
                print(f"[crawler]  已清理删除: {entry['title']} ({aid})")
            del self._metadata[aid]
        self._stats["deleted"] = len(deleted_ids)

    # ---------- 主流程 ----------

    def run(self):
        """执行增量爬取流程。"""
        mode = "全量" if self._full_crawl else "增量"
        print(f"[crawler] ===== 开始{mode}爬取抖店规则中心 =====")
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        try:
            self._start()

            # Step 1: 导航至规则中心页面
            print("[crawler] Step 1: 导航至规则中心页面...")
            self._page.goto(cfg.RULES_URL, wait_until="domcontentloaded", timeout=cfg.NAV_TIMEOUT_MS)
            self._page.wait_for_timeout(cfg.RENDER_WAIT_MS)

            # Step 2: 获取菜单树
            print("[crawler] Step 2: 获取分类菜单树...")
            menu_tree = self._fetch_menu_tree()

            # Step 3: 收集叶子节点
            leaf_nodes = self._collect_leaf_nodes(menu_tree)

            # 排除指定顶级分类（如"历史规则/协议"含上万篇归档内容）
            if self._exclude_categories:
                before = len(leaf_nodes)
                leaf_nodes = [
                    (nid, path) for nid, path in leaf_nodes
                    if not any(path[0] == ex for ex in self._exclude_categories)
                ]
                print(f"[crawler] 排除分类 {self._exclude_categories}: "
                      f"{before} → {len(leaf_nodes)} 个叶子分类")

            if self._category_filter:
                kw = self._category_filter
                leaf_nodes = [
                    (nid, path) for nid, path in leaf_nodes
                    if any(kw in p for p in path)
                ]
                print(f"[crawler] 分类过滤 '{kw}': 剩余 {len(leaf_nodes)} 个叶子分类")

            print(f"[crawler] Step 3: 共 {len(leaf_nodes)} 个叶子分类")

            # Step 4: 加载本地已有元数据
            print("[crawler] Step 4: 加载本地元数据...")
            self._load_existing_metadata()

            # Step 5: 获取线上全部文章列表
            print("[crawler] Step 5: 获取线上文章列表...")
            online_articles = self._fetch_all_articles(leaf_nodes)
            print(f"[crawler] 线上文章总数: {len(online_articles)}")

            # Step 6: 增量比对，确定需要爬取的文章
            print("[crawler] Step 6: 增量比对...")

            to_crawl: list[tuple[str, str, dict]] = []  # (action, article_id, article_info)
            for art_id, art_info in online_articles.items():
                action = self._determine_action(art_id, art_info.get("update_at", 0))
                if action == "skip":
                    self._stats["skipped"] += 1
                else:
                    to_crawl.append((action, art_id, art_info))

            new_count = sum(1 for a, _, _ in to_crawl if a == "new")
            update_count = sum(1 for a, _, _ in to_crawl if a == "update")
            print(f"[crawler]  新增: {new_count}  "
                  f"更新: {update_count}  "
                  f"跳过: {self._stats['skipped']}  "
                  f"待爬取: {len(to_crawl)}")

            # 应用 max_articles 限制
            if self._max_articles and len(to_crawl) > self._max_articles:
                to_crawl = to_crawl[:self._max_articles]
                print(f"[crawler]  按限制截取前 {self._max_articles} 篇")

            # Step 7: 逐篇爬取
            print(f"[crawler] Step 7: 开始爬取 ({len(to_crawl)} 篇)...")

            for idx, (action, art_id, art_info) in enumerate(to_crawl, 1):
                title = art_info["title"]
                category_path = art_info["_category_path"]
                category_str = art_info["_category_str"]
                update_ts = art_info.get("update_at", 0)
                update_time = _ts_to_str(update_ts) if update_ts else ""
                url = cfg.ARTICLE_URL_TMPL.format(article_id=art_id)

                action_label = "新增" if action == "new" else "更新"
                print(f"\n[crawler]  [{idx}/{len(to_crawl)}] [{action_label}] {title}")

                # 提取正文 HTML
                html = self._extract_with_retry(art_id, update_time)

                if html is None:
                    print(f"[crawler]    ✗ 提取失败，跳过")
                    self._stats["failed"] += 1
                    continue

                # 获取旧文件路径（用于清理）
                old_entry = self._metadata.get(art_id, {})
                old_file_path = old_entry.get("file_path", "")

                # 保存文件
                rel_path = self._save_article_html(
                    category_path=category_path,
                    title=title,
                    html=html,
                    article_id=art_id,
                    old_file_path=old_file_path,
                )

                # 更新元数据
                self._metadata[art_id] = {
                    "article_id": art_id,
                    "title": title,
                    "url": url,
                    "category": category_str,
                    "category_path": category_path,
                    "update_at": update_ts,
                    "update_time": update_time,
                    "create_time": _ts_to_str(art_info.get("create_at", 0)) if art_info.get("create_at") else "",
                    "view_count": art_info.get("view_count", 0),
                    "extra_tags": art_info.get("extra_tags", []),
                    "file_path": rel_path,
                    "crawled_at": _now_str(),
                }

                if action == "new":
                    self._stats["new"] += 1
                else:
                    self._stats["updated"] += 1

                time.sleep(cfg.REQUEST_INTERVAL_SEC)

            # Step 8: 可选清理已删除文章
            if self._cleanup:
                if self._category_filter:
                    print(f"\n[crawler] Step 8: 跳过清理（--category 过滤模式下不执行清理，避免误删）")
                else:
                    print(f"\n[crawler] Step 8: 清理已删除文章...")
                    self._cleanup_deleted_articles(set(online_articles.keys()))

            # Step 9: 保存元数据
            print(f"\n[crawler] Step 9: 保存元数据...")
            self._save_metadata()

            # 打印统计
            print(f"\n[crawler] ===== 爬取完成 =====")
            print(f"[crawler] 新增: {self._stats['new']}  "
                  f"更新: {self._stats['updated']}  "
                  f"跳过: {self._stats['skipped']}  "
                  f"失败: {self._stats['failed']}  "
                  f"删除: {self._stats['deleted']}")
            print(f"[crawler] 本地文章总数: {len(self._metadata)}")
            print(f"[crawler] 输出目录: {cfg.OUTPUT_DIR}")

        finally:
            self._close()
