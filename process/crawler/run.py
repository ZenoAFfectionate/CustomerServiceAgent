# -*- coding: utf-8 -*-
"""
抖店规则中心爬虫入口脚本。

用法：
    # 在项目根目录下执行（增量更新，默认行为）
    cd CustomerServiceAgent
    PYTHONPATH=process python -m process.crawler.run

    # 或直接在 crawler 目录下执行
    cd process/crawler
    python run.py

可选参数：
    --no-headless        显示浏览器窗口（调试用）
    --max-articles N     限制最大爬取文章数（调试用，0=不限制）
    --category "分类名"  只爬取指定分类（模糊匹配分类名，调试用）
    --exclude "分类1,分类2"  排除指定顶级分类（如"历史规则/协议"）
    --full               强制全量爬取，忽略本地时间戳
    --cleanup            清理线上已删除但本地仍存在的文章
"""
import os
import sys
import argparse

# 确保能以 `python run.py` 方式直接运行
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crawler import config as cfg
from crawler.crawler import RulesCrawler


def main():
    parser = argparse.ArgumentParser(description="抖店规则中心爬虫（支持增量更新）")
    parser.add_argument(
        "--no-headless", action="store_true", default=False,
        help="显示浏览器窗口（调试用）",
    )
    parser.add_argument(
        "--max-articles", type=int, default=0,
        help="限制最大爬取文章数（调试用，0=不限制）",
    )
    parser.add_argument(
        "--category", type=str, default="",
        help="只爬取指定分类（模糊匹配分类名，调试用）",
    )
    parser.add_argument(
        "--exclude", type=str, default="",
        help='排除指定顶级分类，逗号分隔（如 "历史规则/协议"）',
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

    # 解析排除列表
    exclude_list = [s.strip() for s in args.exclude.split(",") if s.strip()] if args.exclude else []

    crawler = RulesCrawler(
        max_articles=args.max_articles,
        category_filter=args.category,
        exclude_categories=exclude_list,
        full_crawl=args.full,
        cleanup=args.cleanup,
    )
    crawler.run()


if __name__ == "__main__":
    main()
