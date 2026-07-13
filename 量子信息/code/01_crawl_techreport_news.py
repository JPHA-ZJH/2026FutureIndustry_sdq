# -*- coding: utf-8 -*-
"""
Crawl news articles from TechReport and save the raw corpus.

Output folder:
    <project_root>/data/techreport_news_raw_<run_time>/

Install dependencies if needed:
    pip install requests beautifulsoup4 pandas python-dateutil lxml openpyxl tqdm
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse as parse_date
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_BASE = PROJECT_ROOT / "data"

BASE_URL = "https://techreport.com/"
NEWS_URL = "https://techreport.com/news/"

MAX_PAGES = 50
STOP_DATE = ""  # Example: "2024-01-01"; leave empty to crawl until MAX_PAGES.
SLEEP_MIN = 1.0
SLEEP_MAX = 2.5
REQUEST_TIMEOUT = 30


session = requests.Session()
session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
)


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def sleep_random() -> None:
    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))


def get_soup(url: str) -> BeautifulSoup:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


def safe_filename(text: str, max_len: int = 90) -> str:
    text = re.sub(r'[\\/:*?"<>|]', "_", str(text))
    text = re.sub(r"\s+", "_", text).strip("._ ")
    return (text or "untitled")[:max_len]


def parse_date_value(value: str | None):
    value = clean_text(value)
    if not value:
        return None
    try:
        return parse_date(value, fuzzy=True).date()
    except Exception:
        return None


def parse_date_from_node(node) -> object | None:
    time_tag = node.select_one("time")
    if time_tag:
        parsed = parse_date_value(time_tag.get("datetime") or time_tag.get_text(" ", strip=True))
        if parsed:
            return parsed

    text = node.get_text(" ", strip=True)
    patterns = [
        r"([A-Z][a-z]+ \d{1,2}, \d{4})",
        r"(\d{4}-\d{1,2}-\d{1,2})",
        r"(\d{1,2}/\d{1,2}/\d{4})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            parsed = parse_date_value(match.group(1))
            if parsed:
                return parsed
    return None


def is_techreport_article_url(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.netloc.endswith("techreport.com"):
        return False
    blocked_parts = {
        "author",
        "category",
        "tag",
        "page",
        "news",
        "about",
        "contact",
        "privacy-policy",
        "terms-of-use",
    }
    parts = [part for part in parsed.path.split("/") if part]
    return bool(parts) and parts[0] not in blocked_parts


def parse_listing_page(url: str) -> tuple[list[dict], str | None]:
    soup = get_soup(url)
    posts: list[dict] = []

    candidate_nodes = soup.select("article")
    if not candidate_nodes:
        candidate_nodes = [
            node
            for node in soup.select("main h2, main h3, .site-main h2, .site-main h3")
            if node.select_one("a[href]") or node.find_parent("article")
        ]

    seen_urls: set[str] = set()
    for node in candidate_nodes:
        link_tag = node.select_one("h2 a[href], h3 a[href], .entry-title a[href], a[href]")
        if not link_tag:
            continue

        article_url = urljoin(url, link_tag["href"]).split("#")[0]
        if article_url in seen_urls or not is_techreport_article_url(article_url):
            continue
        seen_urls.add(article_url)

        title = clean_text(link_tag.get_text(" ", strip=True))
        if not title:
            continue

        date = parse_date_from_node(node)
        author_tag = node.select_one(".author a, a[rel='author'], .byline a")
        excerpt_tag = node.select_one(".entry-summary, .excerpt, p")

        posts.append(
            {
                "date": date,
                "title": title,
                "url": article_url,
                "author": clean_text(author_tag.get_text(" ", strip=True)) if author_tag else "",
                "excerpt": clean_text(excerpt_tag.get_text(" ", strip=True)) if excerpt_tag else "",
                "source_listing_url": url,
            }
        )

    next_url = None
    next_tag = soup.select_one("a.next, .nav-links a.next, a[rel='next']")
    if next_tag and next_tag.get("href"):
        next_url = urljoin(url, next_tag["href"])
    else:
        for a_tag in soup.find_all("a", href=True):
            text = clean_text(a_tag.get_text(" ", strip=True)).lower()
            if text in {"next", "next page", "older posts", "older"}:
                next_url = urljoin(url, a_tag["href"])
                break

    return posts, next_url


def parse_article_page(url: str) -> dict:
    soup = get_soup(url)

    title_tag = soup.select_one("h1.entry-title, h1")
    date = parse_date_from_node(soup)

    author_tag = soup.select_one(".author a, a[rel='author'], .byline a")
    category_tags = soup.select(".cat-links a, a[rel='category tag'], .post-categories a")

    content_node = soup.select_one(
        ".entry-content, article .post-content, article .content, main article, article"
    )
    paragraphs: list[str] = []
    if content_node:
        for bad in content_node.select(
            "script, style, noscript, iframe, form, nav, aside, "
            ".sharedaddy, .jp-relatedposts, .related-posts, .post-navigation, "
            ".comments-area, .newsletter, .advertisement, .ad, .wp-block-buttons"
        ):
            bad.decompose()

        for text_node in content_node.find_all(["p", "h2", "h3", "li"]):
            text = clean_text(text_node.get_text(" ", strip=True))
            if text:
                paragraphs.append(text)

    content = "\n".join(paragraphs)
    return {
        "article_title": clean_text(title_tag.get_text(" ", strip=True)) if title_tag else "",
        "article_date": date,
        "article_author": clean_text(author_tag.get_text(" ", strip=True)) if author_tag else "",
        "categories": "; ".join(clean_text(tag.get_text(" ", strip=True)) for tag in category_tags),
        "content": content,
        "content_length": len(content),
    }


def crawl_listing(start_url: str, max_pages: int, stop_date: str) -> list[dict]:
    rows: list[dict] = []
    seen_urls: set[str] = set()
    page_url = start_url
    page_num = 1
    stop_day = pd.to_datetime(stop_date).date() if stop_date else None

    while page_url and page_num <= max_pages:
        print(f"Reading listing page {page_num}: {page_url}")
        posts, next_url = parse_listing_page(page_url)
        if not posts:
            print("No articles found on this listing page; stopping.")
            break

        reached_stop_date = False
        for post in posts:
            post_date = post.get("date")
            if stop_day and post_date and post_date < stop_day:
                reached_stop_date = True
                continue
            if post["url"] in seen_urls:
                continue
            seen_urls.add(post["url"])
            rows.append(post)

        print(f"Collected listing records: {len(rows)}")
        if reached_stop_date:
            print(f"Reached stop date {stop_day}; stopping pagination.")
            break

        page_url = next_url
        page_num += 1
        sleep_random()

    return rows


def enrich_articles(posts: Iterable[dict]) -> list[dict]:
    rows: list[dict] = []
    for post in tqdm(list(posts), desc="Reading article pages"):
        row = dict(post)
        try:
            detail = parse_article_page(post["url"])
            row.update(detail)
            if row.get("article_date") and not row.get("date"):
                row["date"] = row["article_date"]
            if row.get("article_author") and not row.get("author"):
                row["author"] = row["article_author"]
        except Exception as exc:
            row["error"] = str(exc)
            row["content"] = ""
            row["content_length"] = 0
        rows.append(row)
        sleep_random()
    return rows


def export_results(rows: list[dict], out_base: Path) -> Path:
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base / f"techreport_news_raw_{run_time}"
    text_dir = out_dir / "article_texts"
    text_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    if df.empty:
        (out_dir / "EMPTY.txt").write_text("No articles were collected.\n", encoding="utf-8")
        print(f"No data collected. Empty marker saved to: {out_dir}")
        return out_dir

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["article_date"] = pd.to_datetime(df.get("article_date"), errors="coerce")
    df["content"] = df.get("content", "").fillna("")
    df["content_length"] = df["content"].str.len()
    df = df.sort_values(["date", "title"], ascending=[False, True]).reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)

    content_paths: list[str] = []
    for index, row in df.iterrows():
        date_str = row["date"].strftime("%Y-%m-%d") if pd.notna(row["date"]) else "unknown_date"
        title = row.get("title") or row.get("article_title") or "untitled"
        txt_path = text_dir / f"{date_str}_{index + 1:04d}_{safe_filename(title)}.txt"
        txt_path.write_text(
            "\n".join(
                [
                    f"Title: {title}",
                    f"Date: {date_str}",
                    f"Author: {row.get('author', '')}",
                    f"Categories: {row.get('categories', '')}",
                    f"URL: {row.get('url', '')}",
                    "",
                    row.get("content", ""),
                ]
            ),
            encoding="utf-8",
        )
        content_paths.append(str(txt_path))

    df["content_txt_path"] = content_paths

    csv_path = out_dir / "techreport_news_raw_full.csv"
    jsonl_path = out_dir / "techreport_news_raw_full.jsonl"
    xlsx_path = out_dir / "techreport_news_summary.xlsx"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False, date_format="iso")

    preview = df.copy()
    preview["content_preview"] = preview["content"].str.slice(0, 3000)
    preview = preview.drop(columns=["content"], errors="ignore")
    monthly = (
        df.groupby("month", dropna=False, as_index=False)
        .agg(news_count=("url", "count"))
        .sort_values("month", ascending=False)
    )
    categories = (
        df.assign(categories=df["categories"].fillna(""))
        .assign(category=df["categories"].str.split("; "))
        .explode("category")
    )
    categories = (
        categories.groupby("category", dropna=False, as_index=False)
        .agg(news_count=("url", "count"))
        .sort_values("news_count", ascending=False)
    )

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        preview.to_excel(writer, sheet_name="raw_news_preview", index=False)
        monthly.to_excel(writer, sheet_name="monthly_count", index=False)
        categories.to_excel(writer, sheet_name="category_count", index=False)

    metadata = {
        "source": NEWS_URL,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "article_count": int(len(df)),
        "csv": str(csv_path),
        "jsonl": str(jsonl_path),
        "xlsx": str(xlsx_path),
        "article_texts": str(text_dir),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(df)} articles to: {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl TechReport news articles.")
    parser.add_argument("--start-url", default=NEWS_URL, help="Listing page to start from.")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES, help="Maximum listing pages.")
    parser.add_argument("--stop-date", default=STOP_DATE, help="Oldest date to include, e.g. 2024-01-01.")
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE, help="Base output folder.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    posts = crawl_listing(args.start_url, args.max_pages, args.stop_date)
    rows = enrich_articles(posts)
    export_results(rows, args.out_base)


if __name__ == "__main__":
    main()
