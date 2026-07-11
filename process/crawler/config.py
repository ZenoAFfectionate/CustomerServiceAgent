# -*- coding: utf-8 -*-
"""
爬虫配置常量。
"""
import os

# ======================== 目标站点 ========================
BASE_HOST = "https://school.jinritemai.com"
RULES_URL = f"{BASE_HOST}/doudian/web/rules"
ARTICLE_URL_TMPL = f"{BASE_HOST}/doudian/web/article/{{article_id}}"

# 规则中心根 node_id（"规则中心"顶级节点）
ROOT_NODE_ID = "7236"
GRAPH_ID = "312"

# ======================== API 端点 ========================
API_MENU_LIST = f"{BASE_HOST}/api/eschool/v2/library/menu/list"
API_ARTICLE_LIST = f"{BASE_HOST}/api/eschool/v2/library/article/list"
API_ARTICLE_DETAIL = f"{BASE_HOST}/api/eschool/v2/library/article/detail"

# ======================== 爬取参数 ========================
# 文章列表每页条数（API 最大支持 20）
PAGE_SIZE = 20

# 页面导航后等待内容渲染的时间（毫秒）
RENDER_WAIT_MS = 2000

# 每篇文章之间的间隔（秒），避免请求过快被风控
REQUEST_INTERVAL_SEC = 0.5

# 请求超时（毫秒）
NAV_TIMEOUT_MS = 30000

# 浏览器无头模式（调试时可改为 False 以观察浏览器行为）
HEADLESS = True

# ======================== 存储路径 ========================
# crawler/ 目录本身
CRAWLER_DIR = os.path.dirname(os.path.abspath(__file__))

# process/ 目录（crawler 的上级）
PROCESS_ROOT = os.path.dirname(CRAWLER_DIR)

# 数据输出根目录：process/data/抖店规则中心/
# 与现有 process/data/ 下的其他数据源并列，互不干扰
OUTPUT_DIR = os.path.join(PROCESS_ROOT, "data", "抖店规则中心")

# 元数据文件路径
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")
