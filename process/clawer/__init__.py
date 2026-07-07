# -*- coding: utf-8 -*-
"""
抖店规则中心爬虫模块（支持增量更新）。

通过 Playwright 无头浏览器访问 https://school.jinritemai.com/doudian/web/rules，
调用后端 API 获取规则分类树与文章列表，再逐篇导航至文章详情页
提取渲染后的 HTML 内容，最终按分类目录结构存储到 process/data/抖店规则中心/ 下，
供后续 RAG 数据处理流水线（process/main.py）使用。

增量更新机制：
    首次运行 → 全量爬取所有文章，保存 metadata.json
    后续运行 → 比对线上 update_at 与本地时间戳，仅爬取新增/更新的文章，
               跳过未变化的文章，可选清理已删除的文章（--cleanup）

数据流：
    规则中心页面 → API 获取菜单树 → API 获取文章列表 → 增量比对 → 爬取变更文章 → 存储到 process/data/抖店规则中心/
"""
