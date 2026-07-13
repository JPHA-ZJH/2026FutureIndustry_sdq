# TechReport 量子信息新闻筛选

本文件夹包含两个脚本：

1. `code/01_crawl_techreport_news.py`
   - 从 `https://techreport.com/news/` 抓取新闻列表和正文。
   - 输出到主项目目录的 `data/techreport_news_raw_运行时间/`。

2. `code/02_filter_quantum_information_news.py`
   - 默认读取最新的 `data/techreport_news_raw_*/`。
   - 根据量子信息关键词筛选相关新闻。
   - 输出到 `量子信息/data/techreport_quantum_news_运行时间/`。

## 运行方式

先安装依赖：

```powershell
pip install -r .\量子信息\requirements.txt
```

第一步：抓取 TechReport 新闻。

```powershell
python .\量子信息\code\01_crawl_techreport_news.py --max-pages 50
```

如需只抓取某个日期之后的新闻：

```powershell
python .\量子信息\code\01_crawl_techreport_news.py --max-pages 50 --stop-date 2024-01-01
```

第二步：筛选量子信息相关新闻。

```powershell
python .\量子信息\code\02_filter_quantum_information_news.py
```

如果要指定某一次抓取结果：

```powershell
python .\量子信息\code\02_filter_quantum_information_news.py --input-dir .\data\techreport_news_raw_YYYYMMDD_HHMMSS
```

输出中包含 CSV、JSONL、Excel 汇总表和每篇文章的 txt 正文文件。
