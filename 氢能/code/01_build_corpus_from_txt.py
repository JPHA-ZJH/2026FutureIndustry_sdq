from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "article_texts"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
CSV_PATH = OUT_DIR / "articles_corpus.csv"
JSONL_PATH = OUT_DIR / "articles_corpus.jsonl"

META_FIELDS = {
    "title": "title",
    "date": "date",
    "author": "author",
    "url": "url",
    "categories": "categories",
    "tags": "tags",
    "meta keywords": "meta_keywords",
    "meta_keywords": "meta_keywords",
    "meta keyword": "meta_keywords",
}


def read_text(path: Path) -> str:
    """尽量兼容不同编码读取 txt。"""
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def clean_space(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_article(path: Path) -> dict[str, object]:
    text = clean_space(read_text(path))
    lines = text.splitlines()
    meta = {field: "" for field in META_FIELDS.values()}
    content_start = 0

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            content_start = idx + 1
            # 通常第一段空行以后就是正文。
            if idx > 0:
                break
            continue

        match = re.match(r"^([A-Za-z _-]+):\s*(.*)$", stripped)
        if not match:
            content_start = idx
            break

        raw_key, value = match.groups()
        key = raw_key.strip().lower().replace("-", " ")
        if key in META_FIELDS:
            meta[META_FIELDS[key]] = value.strip()
            content_start = idx + 1
        else:
            content_start = idx
            break

    content = clean_space("\n".join(lines[content_start:]))
    date_value = pd.to_datetime(meta["date"], errors="coerce")

    if pd.notna(date_value):
        year = int(date_value.year)
        month = int(date_value.month)
        start_month = ((month - 1) // 2) * 2 + 1
        end_month = start_month + 1
        bi_month_period = f"{year}-{start_month:02d}_{end_month:02d}"
        quarter = f"{year}-Q{((month - 1) // 3) + 1}"
        date_text = date_value.strftime("%Y-%m-%d")
    else:
        year = ""
        month = ""
        bi_month_period = ""
        quarter = ""
        date_text = meta["date"]

    article_key = meta["url"] or path.name
    article_id = hashlib.md5(article_key.encode("utf-8")).hexdigest()

    return {
        "article_id": article_id,
        "filename": path.name,
        "title": meta["title"],
        "date": date_text,
        "year": year,
        "month": month,
        "bi_month_period": bi_month_period,
        "quarter": quarter,
        "author": meta["author"],
        "url": meta["url"],
        "categories": meta["categories"],
        "tags": meta["tags"],
        "meta_keywords": meta["meta_keywords"],
        "content": content,
        "content_length": len(content),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No txt files found in {RAW_DIR}")

    rows = [parse_article(path) for path in tqdm(files, desc="Building corpus")]
    df = pd.DataFrame(rows)
    df = df.sort_values(["date", "filename"], na_position="last").reset_index(drop=True)

    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    with JSONL_PATH.open("w", encoding="utf-8") as f:
        for row in df.to_dict(orient="records"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved {len(df)} articles to {CSV_PATH}")
    print(f"Saved JSONL to {JSONL_PATH}")


if __name__ == "__main__":
    main()
