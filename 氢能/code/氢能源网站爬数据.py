# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 22:02:21 2026

@author: Zz
"""

# -*- coding: utf-8 -*-
"""
爬取 Hydrogen Fuel News - Featured News / Hydrogen news 栏目
目标：从最新报道开始，爬取到指定截止日期为止，并汇总为 CSV / Excel

运行前安装：
pip install requests beautifulsoup4 pandas python-dateutil tqdm lxml openpyxl
"""

import re
import time
import random
from pathlib import Path
from urllib.parse import urljoin

import requests
import pandas as pd
from bs4 import BeautifulSoup
from dateutil.parser import parse
from tqdm import tqdm


# =========================
# 1. 参数设置
# =========================

BASE_URL = "https://www.hydrogenfuelnews.com/category/featured-news/"

# 爬到这个日期为止：包含该日期及之后的新闻
# 例如 "2024-01-01" 表示爬取 2024-01-01 至最新新闻
STOP_DATE = "2026-01-01"

# 请求间隔，避免访问过快
SLEEP_MIN = 1.0
SLEEP_MAX = 2.5

# 最多翻多少页，防止死循环
MAX_PAGES = 1000

OUT_DIR = Path("hydrogenfuelnews_output")
OUT_DIR.mkdir(exist_ok=True)

STOP_DATE = pd.to_datetime(STOP_DATE).date()


# =========================
# 2. 基础请求函数
# =========================

session = requests.Session()
session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
})


def get_soup(url, timeout=30):
    """请求网页并返回 BeautifulSoup 对象。"""
    resp = session.get(url, timeout=timeout)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def clean_text(text):
    """清理空白字符。"""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def sleep_random():
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


# =========================
# 3. 解析栏目页
# =========================

def parse_date_from_text(text):
    """
    从文本中解析日期。
    该站栏目页常见格式类似：
    June 29, 2026 Off
    May 19, 2026 0
    """
    text = clean_text(text)
    m = re.search(
        r"([A-Z][a-z]+ \d{1,2}, \d{4})",
        text
    )
    if not m:
        return None
    return parse(m.group(1)).date()


def parse_listing_page(url):
    """
    解析栏目页，返回：
    - posts: 当前页新闻列表
    - next_url: 下一页链接
    """
    soup = get_soup(url)

    posts = []

    # WordPress 主题通常每篇文章在 <article> 中
    articles = soup.select("article")

    # 如果主题结构变动，退回到 h2.entry-title 周边解析
    if not articles:
        articles = []
        for h2 in soup.select("h2"):
            if h2.find("a", href=True):
                articles.append(h2.parent)

    for article in articles:
        title_tag = article.select_one("h2.entry-title a, h2 a, .entry-title a")
        if not title_tag:
            continue

        title = clean_text(title_tag.get_text())
        link = urljoin(url, title_tag.get("href"))

        # 日期
        date_text = ""
        time_tag = article.select_one("time")
        if time_tag:
            date_text = time_tag.get("datetime") or time_tag.get_text()
            try:
                date = parse(date_text).date()
            except Exception:
                date = parse_date_from_text(article.get_text(" ", strip=True))
        else:
            date = parse_date_from_text(article.get_text(" ", strip=True))

        if date is None:
            continue

        # 作者
        author_tag = article.select_one(".author a, .byline a, a[rel='author']")
        author = clean_text(author_tag.get_text()) if author_tag else ""

        # 摘要
        excerpt_tag = article.select_one(".entry-summary, .entry-content, p")
        excerpt = clean_text(excerpt_tag.get_text()) if excerpt_tag else ""

        posts.append({
            "date": date,
            "title": title,
            "url": link,
            "author": author,
            "excerpt": excerpt,
        })

    # 下一页链接
    next_tag = soup.select_one("a.next, .nav-links a.next, a[rel='next']")
    next_url = urljoin(url, next_tag.get("href")) if next_tag else None

    # 兼容部分主题：分页中只有文字 Next
    if not next_url:
        for a in soup.find_all("a", href=True):
            if clean_text(a.get_text()).lower() == "next":
                next_url = urljoin(url, a["href"])
                break

    return posts, next_url


# =========================
# 4. 解析单篇文章
# =========================

def parse_article_page(url):
    """
    进入单篇新闻页，抓取正文、分类等信息。
    """
    soup = get_soup(url)

    title_tag = soup.select_one("h1.entry-title, h1")
    title = clean_text(title_tag.get_text()) if title_tag else ""

    # 日期
    date = None
    time_tag = soup.select_one("time")
    if time_tag:
        date_text = time_tag.get("datetime") or time_tag.get_text()
        try:
            date = parse(date_text).date()
        except Exception:
            date = None

    if date is None:
        date = parse_date_from_text(soup.get_text(" ", strip=True))

    # 作者
    author_tag = soup.select_one(".author a, .byline a, a[rel='author']")
    author = clean_text(author_tag.get_text()) if author_tag else ""

    # 正文
    content_node = soup.select_one(".entry-content, article .post-content, article")
    paragraphs = []

    if content_node:
        # 删除不需要的区域
        for bad in content_node.select(
            "script, style, nav, form, .sharedaddy, .jp-relatedposts, "
            ".post-navigation, .comments-area, .widget, .yarpp-related"
        ):
            bad.decompose()

        for p in content_node.find_all(["p", "h2", "h3", "li"]):
            txt = clean_text(p.get_text(" ", strip=True))
            if txt:
                paragraphs.append(txt)

    content = "\n".join(paragraphs)

    # 分类
    category_tags = soup.select(".cat-links a, a[rel='category tag']")
    categories = "; ".join(clean_text(a.get_text()) for a in category_tags)

    return {
        "article_title": title,
        "article_date": date,
        "article_author": author,
        "categories": categories,
        "content": content,
        "content_length": len(content),
    }


# =========================
# 5. 主程序：从最新爬到截止日期
# =========================

def crawl_news():
    all_posts = []
    seen_urls = set()

    page_url = BASE_URL
    page_num = 1
    stop = False

    while page_url and page_num <= MAX_PAGES and not stop:
        print(f"\n正在爬取栏目页 {page_num}: {page_url}")
        posts, next_url = parse_listing_page(page_url)

        if not posts:
            print("当前页没有解析到新闻，停止。")
            break

        for post in posts:
            post_date = post["date"]

            # 因为栏目页按时间倒序排列；
            # 如果已经早于截止日期，可以停止继续翻页
            if post_date < STOP_DATE:
                stop = True
                continue

            if post["url"] in seen_urls:
                continue

            seen_urls.add(post["url"])
            all_posts.append(post)

        print(f"当前累计新闻数：{len(all_posts)}")

        if stop:
            print(f"已到达截止日期 {STOP_DATE}，停止翻页。")
            break

        page_url = next_url
        page_num += 1
        sleep_random()

    return all_posts


def enrich_with_article_content(posts):
    """
    对栏目页得到的新闻列表，逐篇进入正文页抓取完整内容。
    """
    results = []

    for post in tqdm(posts, desc="抓取单篇正文"):
        row = post.copy()

        try:
            detail = parse_article_page(post["url"])
            row.update(detail)
        except Exception as e:
            row["error"] = str(e)
            row["content"] = ""

        results.append(row)
        sleep_random()

    return results


# =========================
# 6. 汇总与导出
# =========================

def safe_filename(text, max_len=80):
    """生成安全文件名。"""
    text = str(text)
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", "_", text).strip("_")
    return text[:max_len] if text else "untitled"


def export_results(rows):
    df = pd.DataFrame(rows)

    if df.empty:
        print("没有爬到结果。")
        return df

    # 日期标准化
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if "article_date" in df.columns:
        df["article_date"] = pd.to_datetime(df["article_date"], errors="coerce")

    # 排序
    df = df.sort_values("date", ascending=False).reset_index(drop=True)

    # 增加年月
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)

    # 正文长度
    df["content"] = df["content"].fillna("")
    df["content_length"] = df["content"].str.len()

    # =========================
    # 1. 完整正文单独保存为 txt
    # =========================

    text_dir = OUT_DIR / "article_texts"
    text_dir.mkdir(exist_ok=True)

    content_paths = []

    for i, row in df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "unknown_date"
        title = row.get("title", "") or row.get("article_title", "")
        filename = f"{date_str}_{i+1:04d}_{safe_filename(title)}.txt"
        path = text_dir / filename

        text = (
            f"Title: {title}\n"
            f"Date: {date_str}\n"
            f"Author: {row.get('author', '')}\n"
            f"URL: {row.get('url', '')}\n"
            f"\n"
            f"{row.get('content', '')}"
        )

        path.write_text(text, encoding="utf-8")
        content_paths.append(str(path))

    df["content_txt_path"] = content_paths

    # =========================
    # 2. 完整数据保存为 CSV 和 JSONL
    # =========================

    csv_path = OUT_DIR / "hydrogen_featured_news_raw_full.csv"
    jsonl_path = OUT_DIR / "hydrogen_featured_news_raw_full.jsonl"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    # =========================
    # 3. Excel 中只保留摘要，避免 32767 字符限制
    # =========================

    df_excel = df.copy()

    # Excel 不放完整正文，只放前 3000 字
    df_excel["content_preview"] = df_excel["content"].str.slice(0, 3000)

    # 删除完整正文列，避免 Excel 截断警告
    df_excel = df_excel.drop(columns=["content"], errors="ignore")

    # 月度数量汇总
    monthly = (
        df.groupby("month", as_index=False)
          .agg(news_count=("url", "count"))
          .sort_values("month", ascending=False)
    )

    # 作者汇总
    author_summary = (
        df.groupby("author", dropna=False, as_index=False)
          .agg(news_count=("url", "count"))
          .sort_values("news_count", ascending=False)
    )

    # 正文长度异常文章
    long_articles = (
        df[["date", "title", "url", "content_length", "content_txt_path"]]
        .sort_values("content_length", ascending=False)
    )

    xlsx_path = OUT_DIR / "hydrogen_featured_news_summary.xlsx"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_excel.to_excel(writer, sheet_name="raw_news_preview", index=False)
        monthly.to_excel(writer, sheet_name="monthly_count", index=False)
        author_summary.to_excel(writer, sheet_name="author_count", index=False)
        long_articles.to_excel(writer, sheet_name="content_length_check", index=False)

    print("\n已完成导出：")
    print("完整 CSV：", csv_path)
    print("完整 JSONL：", jsonl_path)
    print("Excel 汇总：", xlsx_path)
    print("完整正文 txt 文件夹：", text_dir)

    print("\n数据范围：")
    print("最早日期：", df["date"].min().date())
    print("最新日期：", df["date"].max().date())
    print("新闻数量：", len(df))

    print("\n正文长度最长的前 10 篇：")
    print(long_articles.head(10)[["date", "title", "content_length"]])

    return df


if __name__ == "__main__":
    posts = crawl_news()
    rows = enrich_with_article_content(posts)
    df = export_results(rows)