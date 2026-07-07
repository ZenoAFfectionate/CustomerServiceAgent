# -*- coding: utf-8 -*-
"""
Support API 爬虫（巨量千川/巨量广告/巨量本地推）。

这三个平台共享相同的后端 API 结构：
    /support/backend/content/queryByNode?nodeIds={id}&num={n}&spaceId={sid}

工作流程：
    1. 导航至支持中心页面（加载安全 SDK）。
    2. 对每个已知 node_id，调用 queryByNode API 获取文章列表。
    3. 逐篇导航至文章详情页 /support/content/{contentId}，提取正文 HTML。
    4. 按分类存储到 process/data/{中心名}/ 下，支持增量更新。
"""
import os
import re
import json
import time
from datetime import datetime
from typing import Optional

from playwright.sync_api import sync_playwright, Page

from clawer import config as cfg
from clawer.centers import CenterConfig


# ======================== 工具函数 ========================

def _sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    name = name.strip().strip(".")
    if len(name) > 120:
        name = name[:120]
    return name


def _ts_to_str(ts: int) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ======================== 正文提取 JS ========================

_EXTRACT_JS = """
() => {
    const selectors = [
        '.eschool-doc-content-wrapper',
        '.editor-kit-container .ace-editor-wrapper',
        '.editor-kit-container',
        '.ace-editor-wrapper',
        '.article-content',
        '[class*="rich-text"]',
        '[class*="content-detail"]',
        '[class*="ace-editor"]',
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
        const aceLines = document.querySelectorAll('.ace-line, .docx-text-block, [class*="docx"]');
        if (aceLines.length > 0) {
            container = aceLines[0].parentElement;
        }
    }
    if (!container) return null;

    const clone = container.cloneNode(true);
    const noiseSelectors = [
        '.ace-table-fullscreen-icon', '.ace-table-fullscreen-navbar',
        '.ace-table-fullscreen-mask', '[class*="fullscreen"]',
        '[class*="toolbar"]:not([class*="ace-line"])',
        '[class*="feedback"]', '[class*="satisfaction"]',
    ];
    for (const sel of noiseSelectors) {
        clone.querySelectorAll(sel).forEach(el => el.remove());
    }
    return clone.innerHTML;
}
"""


# ======================== Support 爬虫 ========================

class SupportCrawler:
    """Support API 爬虫（巨量千川/巨量广告/巨量本地推）。

    Args:
        center: 中心配置。
        max_articles: 限制最大爬取文章数（0=不限制）。
        full_crawl: 强制全量爬取。
        cleanup: 清理已删除文章。
    """

    def __init__(self, center: CenterConfig, max_articles: int = 0,
                 full_crawl: bool = False, cleanup: bool = False):
        self._center = center
        self._playwright = None
        self._browser = None
        self._page: Optional[Page] = None
        self._metadata: dict[str, dict] = {}
        self._max_articles = max_articles
        self._full_crawl = full_crawl
        self._cleanup = cleanup
        self._stats = {"new": 0, "updated": 0, "skipped": 0, "failed": 0, "deleted": 0}
        # 输出目录：process/data/{中心名}/
        self._output_dir = os.path.join(cfg.PROCESS_ROOT, "data", center.output_dir)
        self._metadata_file = os.path.join(self._output_dir, "metadata.json")

    # ---------- 浏览器 ----------

    def _start(self):
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=cfg.HEADLESS)
        self._page = self._browser.new_page()
        self._page.set_default_timeout(cfg.NAV_TIMEOUT_MS)
        print(f"[{self._center.name}] 浏览器已启动")

    def _close(self):
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()
        print(f"[{self._center.name}] 浏览器已关闭")

    # ---------- 元数据 ----------

    def _load_metadata(self):
        if not os.path.exists(self._metadata_file):
            print(f"[{self._center.name}] 无现有元数据，全量爬取")
            return
        try:
            with open(self._metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._metadata = {m["article_id"]: m for m in data}
            else:
                self._metadata = data
            print(f"[{self._center.name}] 已加载元数据: {len(self._metadata)} 篇")
        except (json.JSONDecodeError, KeyError):
            self._metadata = {}

    def _save_metadata(self):
        data = list(self._metadata.values())
        with open(self._metadata_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[{self._center.name}] 元数据已保存: {len(data)} 篇")

    # ---------- API ----------

    def _fetch_articles_by_node(self, node_id: str) -> list[dict]:
        """调用 queryByNode API 获取某节点下的全部文章。"""
        all_articles = []
        # num=100 已验证可获取全部文章（API 最多返回 100 条）
        result = self._page.evaluate(
            """async (params) => {
                const url = `https://${params.host}/support/backend/content/queryByNode`
                    + `?nodeIds=${params.nodeId}&num=100&spaceId=${params.spaceId}`;
                const resp = await fetch(url);
                return resp.json();
            }""",
            {"host": self._center.host.replace("https://", ""),
             "nodeId": node_id, "spaceId": self._center.space_id},
        )

        if result.get("code") == 1 and isinstance(result.get("data"), list):
            all_articles = result["data"]

        return all_articles

    def _fetch_all_articles(self) -> dict[str, dict]:
        """获取所有节点的文章，返回 {contentId: article_info}。"""
        all_articles: dict[str, dict] = {}

        for idx, node_id in enumerate(self._center.node_ids, 1):
            print(f"[{self._center.name}]  [{idx}/{len(self._center.node_ids)}] "
                  f"获取节点 {node_id} 的文章列表...")
            articles = self._fetch_articles_by_node(node_id)
            print(f"[{self._center.name}]    获取 {len(articles)} 篇")

            for art in articles:
                content_id = str(art["contentId"])
                art["_node_id"] = node_id
                art["_node_index"] = idx
                all_articles[content_id] = art

        return all_articles

    # ---------- 增量判断 ----------

    def _determine_action(self, content_id: str, online_modify_time: int) -> str:
        if self._full_crawl or content_id not in self._metadata:
            return "new" if content_id not in self._metadata else "update"
        local_ts = self._metadata[content_id].get("modify_time_ts", 0)
        if online_modify_time > local_ts:
            return "update"
        return "skip"

    # ---------- 内容提取 ----------

    def _extract_article_html(self, content_id: str, modify_time_str: str) -> Optional[str]:
        article_url = f"{self._center.host}/support/content/{content_id}"

        try:
            self._page.goto(article_url, wait_until="domcontentloaded", timeout=cfg.NAV_TIMEOUT_MS)
        except Exception as e:
            print(f"[{self._center.name}]    导航失败 {content_id}: {e}")

        # 等待正文渲染
        try:
            self._page.wait_for_selector(
                ".ace-line, .docx-text-block, [class*='rich-text'], [class*='content-detail']",
                timeout=cfg.NAV_TIMEOUT_MS,
            )
        except Exception:
            print(f"[{self._center.name}]    未找到正文元素 {content_id}")
            return None

        self._page.wait_for_timeout(cfg.RENDER_WAIT_MS)

        content_html = self._page.evaluate(_EXTRACT_JS)

        if not content_html:
            print(f"[{self._center.name}]    正文提取为空 {content_id}")
            return None

        time_tag = f"<time>{modify_time_str}</time>" if modify_time_str else ""
        return f"{time_tag}{content_html}"

    def _extract_with_retry(self, content_id: str, modify_time_str: str) -> Optional[str]:
        html = self._extract_article_html(content_id, modify_time_str)
        if html is not None:
            return html
        print(f"[{self._center.name}]    重试中...")
        time.sleep(2)
        return self._extract_article_html(content_id, modify_time_str)

    # ---------- 存储 ----------

    def _save_article_html(self, title: str, html: str, content_id: str,
                           old_file_path: str = "") -> str:
        dir_path = self._output_dir
        os.makedirs(dir_path, exist_ok=True)

        filename = _sanitize_filename(title) + ".html"
        filepath = os.path.join(dir_path, filename)
        new_rel_path = os.path.relpath(filepath, self._output_dir)

        if old_file_path and old_file_path != new_rel_path:
            old_abs = os.path.join(self._output_dir, old_file_path)
            if os.path.exists(old_abs):
                os.remove(old_abs)

        if os.path.exists(filepath) and old_file_path != new_rel_path:
            base, ext = os.path.splitext(filename)
            filepath = os.path.join(dir_path, f"{base}_{content_id}{ext}")
            new_rel_path = os.path.relpath(filepath, self._output_dir)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

        return new_rel_path

    def _cleanup_deleted(self, current_ids: set[str]):
        deleted = [aid for aid in self._metadata if aid not in current_ids]
        for aid in deleted:
            entry = self._metadata[aid]
            fp = os.path.join(self._output_dir, entry.get("file_path", ""))
            if os.path.exists(fp):
                os.remove(fp)
                print(f"[{self._center.name}]  已清理: {entry['title']}")
            del self._metadata[aid]
        self._stats["deleted"] = len(deleted)

    # ---------- 主流程 ----------

    def run(self):
        mode = "全量" if self._full_crawl else "增量"
        print(f"\n[{self._center.name}] ===== 开始{mode}爬取 =====")
        os.makedirs(self._output_dir, exist_ok=True)

        try:
            self._start()

            # Step 1: 导航至中心页面
            print(f"[{self._center.name}] Step 1: 导航至中心页面...")
            self._page.goto(self._center.base_url, wait_until="domcontentloaded",
                            timeout=cfg.NAV_TIMEOUT_MS)
            self._page.wait_for_timeout(cfg.RENDER_WAIT_MS)

            # Step 2: 加载本地元数据
            print(f"[{self._center.name}] Step 2: 加载本地元数据...")
            self._load_metadata()

            # Step 3: 获取线上文章列表
            print(f"[{self._center.name}] Step 3: 获取线上文章列表...")
            online_articles = self._fetch_all_articles()
            print(f"[{self._center.name}] 线上文章总数: {len(online_articles)}")

            # Step 4: 增量比对
            print(f"[{self._center.name}] Step 4: 增量比对...")
            to_crawl: list[tuple[str, str, dict]] = []
            for cid, art in online_articles.items():
                action = self._determine_action(cid, art.get("modifyTime", 0))
                if action == "skip":
                    self._stats["skipped"] += 1
                else:
                    to_crawl.append((action, cid, art))

            new_cnt = sum(1 for a, _, _ in to_crawl if a == "new")
            upd_cnt = sum(1 for a, _, _ in to_crawl if a == "update")
            print(f"[{self._center.name}]  新增: {new_cnt}  更新: {upd_cnt}  "
                  f"跳过: {self._stats['skipped']}  待爬取: {len(to_crawl)}")

            if self._max_articles and len(to_crawl) > self._max_articles:
                to_crawl = to_crawl[:self._max_articles]
                print(f"[{self._center.name}]  按限制截取前 {self._max_articles} 篇")

            # Step 5: 逐篇爬取
            print(f"[{self._center.name}] Step 5: 开始爬取 ({len(to_crawl)} 篇)...")

            for idx, (action, cid, art) in enumerate(to_crawl, 1):
                title = art["name"]
                modify_ts = art.get("modifyTime", 0)
                modify_time_str = _ts_to_str(modify_ts) if modify_ts else ""
                url = f"{self._center.host}/support/content/{cid}"
                node_id = art.get("_node_id", "")

                label = "新增" if action == "new" else "更新"
                print(f"[{self._center.name}]  [{idx}/{len(to_crawl)}] [{label}] {title[:50]}")

                html = self._extract_with_retry(cid, modify_time_str)

                if html is None:
                    print(f"[{self._center.name}]    ✗ 提取失败")
                    self._stats["failed"] += 1
                    continue

                old_fp = self._metadata.get(cid, {}).get("file_path", "")
                rel_path = self._save_article_html(title, html, cid, old_fp)

                self._metadata[cid] = {
                    "article_id": cid,
                    "content_id": cid,
                    "title": title,
                    "url": url,
                    "node_id": node_id,
                    "modify_time_ts": modify_ts,
                    "modify_time": modify_time_str,
                    "create_time": _ts_to_str(art.get("createTime", 0)) if art.get("createTime") else "",
                    "file_path": rel_path,
                    "crawled_at": _now_str(),
                }

                if action == "new":
                    self._stats["new"] += 1
                else:
                    self._stats["updated"] += 1

                time.sleep(cfg.REQUEST_INTERVAL_SEC)

            # Step 6: 清理
            if self._cleanup:
                print(f"[{self._center.name}] Step 6: 清理已删除文章...")
                self._cleanup_deleted(set(online_articles.keys()))

            # Step 7: 保存元数据
            print(f"[{self._center.name}] Step 7: 保存元数据...")
            self._save_metadata()

            print(f"\n[{self._center.name}] ===== 完成 =====")
            print(f"[{self._center.name}] 新增: {self._stats['new']}  "
                  f"更新: {self._stats['updated']}  跳过: {self._stats['skipped']}  "
                  f"失败: {self._stats['failed']}  删除: {self._stats['deleted']}")
            print(f"[{self._center.name}] 本地文章总数: {len(self._metadata)}")

        finally:
            self._close()
