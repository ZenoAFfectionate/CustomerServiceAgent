# 抖店规则中心爬虫（支持增量更新 + 多中心映射）

## 中心映射表

| 中心名 | API 类型 | 可爬取 | URL | 说明 |
|--------|----------|--------|-----|------|
| 抖店规则中心 | eschool | ✅ | school.jinritemai.com/doudian/web/rules | 内容渲染为 HTML，无需登录 |
| 巨量千川规则中心 | support | ❌ | qianchuan.jinritemai.com/support/?pageId=229 | 正文以文件预览加载，需登录 |
| 巨量广告规则中心 | support | ❌ | ad.oceanengine.com/support/?pageId=297 | 未登录直接报错(onAuthError) |
| 巨量本地推帮助中心 | support | ❌ | localads.chengzijianzhan.cn/support/?pageId=305 | 正文以文件预览加载，需登录 |

> **说明**：巨量千川/广告/本地推三个中心的正文内容通过文件预览组件（`general__viewer`）加载，
> 未登录状态下预览失败（"There was an error previewing the file"）。
> 现有 `process/data/` 下这三个中心的数据是从飞书/Lark 文档手动导出的，非从支持中心直接爬取。

## 文件结构

```
process/crawler/               # 爬虫代码
├── __init__.py               # 模块说明
├── config.py                 # 配置常量
├── centers.py                # 中心映射表（CenterConfig + CENTERS 列表）
├── crawler.py                # eschool API 爬虫（抖店规则中心）
├── support_crawler.py        # support API 爬虫（千川/广告/本地推，需登录）
├── run.py                    # 单中心入口（抖店规则中心）
├── run_all.py                # 统一入口（一键爬取所有可爬取中心）
└── README.md

process/data/抖店规则中心/      # 爬取数据输出目录
├── 规则总则/
├── 商家管理/
│   ├── 招商入驻/
│   └── ...
└── metadata.json             # 元数据（增量更新基准）
```

## 用法

```bash
cd CustomerServiceAgent

# 统一爬取（自动跳过不可爬取的中心）
PYTHONPATH=process python -m process.crawler.run_all

# 只爬取抖店规则中心
PYTHONPATH=process python -m process.crawler.run

# 强制全量爬取
PYTHONPATH=process python -m process.crawler.run --full

# 增量更新 + 清理已删除
PYTHONPATH=process python -m process.crawler.run --cleanup

# 调试：限制数量
PYTHONPATH=process python -m process.crawler.run --max-articles 30
```

## 增量更新机制

| 场景 | 行为 |
|------|------|
| 首次运行 | 全量爬取所有文章 |
| 文章未变化 | **跳过**（不打开浏览器页面） |
| 文章已更新 | **重新爬取**并替换 |
| 新增文章 | **爬取并保存** |
| 文章已删除 | 默认保留；`--cleanup` 时删除 |

比对依据：API 返回的 `update_at` 时间戳。未变化的文章不导航到详情页，大幅减少爬取时间。

## 抖店规则中心文章分布

| 类别 | 文章数 | 说明 |
|------|--------|------|
| 核心规则分类 | 611 | 规则总则、商家管理、创作者管理等 11 个分类 |
| 辅助内容 | 909 | 公告专区、规则动态、协议专区等 |
| 历史归档（已排除） | 11,575 | 历史规则/协议（`--exclude` 跳过） |
| **本次爬取目标** | **1,520** | 排除历史归档后的全部内容 |
