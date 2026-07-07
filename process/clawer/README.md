# 抖店规则中心爬虫（支持增量更新）

爬取 [抖店规则中心](https://school.jinritemai.com/doudian/web/rules) 的全部规则文档，
按分类目录结构存储为 HTML 文件到 `process/data/抖店规则中心/`，供后续 RAG 数据处理流水线使用。

## 文件结构

```
process/clawer/               # 爬虫代码
├── __init__.py               # 模块说明
├── config.py                 # 配置常量
├── crawler.py                # 核心爬虫逻辑（RulesCrawler 类）
├── run.py                    # 入口脚本
└── README.md

process/data/抖店规则中心/      # 爬取数据输出目录（运行后自动创建）
├── 规则总则/
│   └── 抖音电商规则总则.html
├── 商家管理/
│   ├── 招商入驻/
│   │   ├── 入驻与退出/
│   │   │   └── 招商标准及入驻规范.html
│   │   └── 保证金管理/
│   │       └── 保证金管理规范.html
│   └── ...
└── metadata.json             # 全部文章元数据（增量更新基准）
```

## 增量更新机制

| 场景 | 行为 |
|------|------|
| 首次运行 | 全量爬取所有文章，保存 metadata.json |
| 文章未变化（update_at 相同） | **跳过**，不发起页面请求 |
| 文章已更新（update_at 更新） | **重新爬取**，替换旧文件 |
| 新增文章 | **爬取并保存** |
| 文章已删除（线上不存在） | 默认保留；`--cleanup` 时删除 |

比对依据：API 返回的 `update_at` 时间戳。未变化的文章不会导航到详情页，大幅减少爬取时间。

## 用法

```bash
cd CustomerServiceAgent

# 增量更新（默认行为，只爬取有变化的文章）
PYTHONPATH=process python -m process.clawer.run

# 强制全量爬取（忽略本地时间戳，重新爬取所有文章）
PYTHONPATH=process python -m process.clawer.run --full

# 增量更新 + 清理已删除的文章
PYTHONPATH=process python -m process.clawer.run --cleanup

# 调试：限制爬取数量
PYTHONPATH=process python -m process.clawer.run --max-articles 30

# 调试：只爬取指定分类
PYTHONPATH=process python -m process.clawer.run --category "商家管理"

# 调试：显示浏览器窗口
PYTHONPATH=process python -m process.clawer.run --no-headless
```

## 输出格式

每篇文章保存为单个 HTML 文件：

```html
<time>2026-06-30 14:20:23</time>
<div class="ace-line heading-h1 ...">...</div>
<div class="ace-line ...">...</div>
```

`metadata.json` 记录每篇文章的完整元数据：

```json
{
  "article_id": "aHVWKjDmNiUv",
  "title": "品牌限售细则",
  "url": "https://school.jinritemai.com/doudian/web/article/aHVWKjDmNiUv",
  "category": "商家管理/招商入驻/入驻与退出",
  "category_path": ["商家管理", "招商入驻", "入驻与退出"],
  "update_at": 1782800423,
  "update_time": "2026-06-30 14:20:23",
  "create_time": "2022-06-07 15:12:00",
  "view_count": 1156159,
  "extra_tags": ["更新"],
  "file_path": "商家管理/招商入驻/入驻与退出/品牌限售细则.html",
  "crawled_at": "2026-07-07 11:30:00"
}
```
