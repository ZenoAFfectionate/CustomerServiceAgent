# -*- coding: utf-8 -*-
"""
爬虫中心映射表。

定义所有规则/帮助中心及其配置，包括：
- 抖店规则中心（eschool API，可爬取 ✅）
- 巨量千川规则中心（support API，需登录 ❌）
- 巨量广告规则中心（support API，需登录 ❌）
- 巨量本地推帮助中心（support API，需登录 ❌）

两种 API 类型：
- eschool: school.jinritemai.com 使用的 /api/eschool/v2/library/* 系列 API
  内容直接渲染为 HTML（ace-line divs），无需登录即可爬取。
- support: qianchuan/ad.oceanengine/localads 使用的 /support/backend/content/queryByNode API
  文章列表可获取，但正文以文件预览（general__viewer）形式加载，未登录时预览失败。
  现有数据是从飞书/Lark 文档手动导出的，非从支持中心直接爬取。
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CenterConfig:
    """单个规则/帮助中心的配置。"""
    name: str                          # 中心名称（也是输出目录名）
    api_type: str                      # "eschool" 或 "support"
    base_url: str                      # 中心入口 URL
    host: str                          # API 主机域名
    output_dir: str                    # 输出子目录名
    crawlable: bool = True             # 是否可自动爬取（support 站点需登录）

    # eschool API 参数
    root_node_id: str = ""
    graph_id: str = ""

    # support API 参数
    space_id: str = ""
    node_ids: list[str] = field(default_factory=list)

    # 排除的顶级分类（仅 eschool API）
    exclude_categories: list[str] = field(default_factory=list)

    # 不可爬取时的说明
    note: str = ""


# ======================== 全部中心配置 ========================

CENTERS: list[CenterConfig] = [
    # 1. 抖店规则中心（eschool API，可自动爬取 ✅）
    CenterConfig(
        name="抖店规则中心",
        api_type="eschool",
        base_url="https://school.jinritemai.com/doudian/web/rules",
        host="https://school.jinritemai.com",
        output_dir="抖店规则中心",
        crawlable=True,
        root_node_id="7236",
        graph_id="312",
        exclude_categories=["历史规则/协议"],
    ),

    # 2. 巨量千川规则中心（support API，需登录 ❌）
    CenterConfig(
        name="巨量千川规则中心",
        api_type="support",
        base_url="https://qianchuan.jinritemai.com/support/?pageId=229",
        host="https://qianchuan.jinritemai.com",
        output_dir="巨量千川规则中心",
        crawlable=False,
        space_id="295",
        node_ids=[
            "27959839", "27959583", "27955743",
            "6649352450", "6428166146", "6635521026",
        ],
        note="正文以文件预览形式加载，未登录时预览失败。现有数据从飞书/Lark导出。",
    ),

    # 3. 巨量广告规则中心（support API，需登录 ❌）
    CenterConfig(
        name="巨量广告规则中心",
        api_type="support",
        base_url="https://ad.oceanengine.com/support/?pageId=297",
        host="https://ad.oceanengine.com",
        output_dir="巨量广告规则中心",
        crawlable=False,
        space_id="171",
        node_ids=[
            "27788063", "27789855", "27790111",
            "27790367", "287235599", "287235855",
        ],
        note="未登录时页面直接报错(onAuthError)。现有数据从飞书/Lark导出。",
    ),

    # 4. 巨量本地推帮助中心（support API，需登录 ❌）
    CenterConfig(
        name="巨量本地推帮助中心",
        api_type="support",
        base_url="https://localads.chengzijianzhan.cn/support/?pageId=305",
        host="https://localads.chengzijianzhan.cn",
        output_dir="巨量本地推帮助中心",
        crawlable=False,
        space_id="174",
        node_ids=[
            "1838594306", "1838592514", "1838594050",
            "1838594818", "1838594562",
        ],
        note="正文以文件预览形式加载，未登录时预览失败。现有数据从飞书/Lark导出。",
    ),
]


def get_center(name: str) -> Optional[CenterConfig]:
    """按名称获取中心配置。"""
    for c in CENTERS:
        if c.name == name:
            return c
    return None


def list_centers() -> list[str]:
    """列出所有中心名称。"""
    return [c.name for c in CENTERS]


def list_crawlable_centers() -> list[str]:
    """列出可自动爬取的中心名称。"""
    return [c.name for c in CENTERS if c.crawlable]
