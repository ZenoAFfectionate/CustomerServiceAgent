# -*- coding: utf-8 -*-
"""
统一爬虫入口 — 一键爬取/更新所有可爬取的规则中心。

用法：
    cd CustomerServiceAgent
    PYTHONPATH=process python -m process.crawler.run_all              # 增量更新全部可爬取中心
    PYTHONPATH=process python -m process.crawler.run_all --full       # 强制全量爬取
    PYTHONPATH=process python -m process.crawler.run_all --cleanup    # 增量+清理已删除
    PYTHONPATH=process python -m process.crawler.run_all --only 抖店规则中心  # 只爬指定中心
    PYTHONPATH=process python -m process.crawler.run_all --max-articles 10    # 调试用

注意：
    巨量千川/巨量广告/巨量本地推 三个中心的正文内容需要登录后才能访问（文件预览机制），
    当前仅支持自动爬取"抖店规则中心"。其余中心的数据需从飞书/Lark手动导出。
"""
import os
import sys
import argparse

if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler import config as cfg
from crawler.centers import CENTERS, list_centers, list_crawlable_centers
from crawler.crawler import RulesCrawler
from crawler.support_crawler import SupportCrawler


def crawl_center(center, max_articles=0, full_crawl=False, cleanup=False):
    """爬取单个中心。"""
    if not center.crawlable:
        print(f"\n  ⚠ 跳过 '{center.name}'（不可自动爬取）")
        print(f"    原因: {center.note}")
        return False

    if center.api_type == "eschool":
        crawler = RulesCrawler(
            max_articles=max_articles,
            full_crawl=full_crawl,
            cleanup=cleanup,
            exclude_categories=center.exclude_categories,
        )
    else:
        crawler = SupportCrawler(
            center=center,
            max_articles=max_articles,
            full_crawl=full_crawl,
            cleanup=cleanup,
        )
    crawler.run()
    return True


def main():
    parser = argparse.ArgumentParser(description="统一爬虫 — 一键爬取所有规则中心")
    parser.add_argument(
        "--no-headless", action="store_true", default=False,
        help="显示浏览器窗口（调试用）",
    )
    parser.add_argument(
        "--max-articles", type=int, default=0,
        help="每个中心限制最大爬取文章数（调试用，0=不限制）",
    )
    parser.add_argument(
        "--only", type=str, default="",
        help=f"只爬取指定中心（可选: {', '.join(list_centers())}）",
    )
    parser.add_argument(
        "--full", action="store_true", default=False,
        help="强制全量爬取，忽略本地时间戳",
    )
    parser.add_argument(
        "--cleanup", action="store_true", default=False,
        help="清理线上已删除但本地仍存在的文章",
    )
    args = parser.parse_args()

    if args.no_headless:
        cfg.HEADLESS = False

    # 筛选要爬取的中心
    centers = CENTERS
    if args.only:
        centers = [c for c in CENTERS if c.name == args.only]
        if not centers:
            print(f"未找到中心 '{args.only}'，可选: {', '.join(list_centers())}")
            return

    print(f"{'='*60}")
    print(f"统一爬虫 — 共 {len(centers)} 个中心")
    crawlable = [c for c in centers if c.crawlable]
    not_crawlable = [c for c in centers if not c.crawlable]

    for c in crawlable:
        print(f"  ✅ {c.name} ({c.api_type})")
    for c in not_crawlable:
        print(f"  ❌ {c.name} — {c.note}")
    print(f"{'='*60}")

    if not crawlable:
        print("\n没有可自动爬取的中心。")
        return

    for i, center in enumerate(centers, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(centers)}] {center.name}")
        print(f"{'#'*60}")

        try:
            crawl_center(
                center,
                max_articles=args.max_articles,
                full_crawl=args.full,
                cleanup=args.cleanup,
            )
        except Exception as e:
            print(f"\n[ERROR] {center.name} 爬取失败: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"全部中心爬取完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
