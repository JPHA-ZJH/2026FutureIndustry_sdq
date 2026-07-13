# -*- coding: utf-8 -*-
"""
Filter TechReport news for quantum information related articles.

Input:
    By default, the newest <project_root>/data/techreport_news_raw_*/ folder.

Output folder:
    <project_root>/量子信息/data/techreport_quantum_news_<run_time>/
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_BASE = PROJECT_ROOT / "data"
DEFAULT_OUT_BASE = PROJECT_ROOT / "量子信息" / "data"

QUANTUM_KEYWORDS = [
    "quantum",
    "qubit",
    "qubits",
    "quantum computing",
    "quantum computer",
    "quantum computers",
    "quantum processor",
    "quantum chip",
    "quantum hardware",
    "quantum software",
    "quantum algorithm",
    "quantum algorithms",
    "quantum simulation",
    "quantum simulator",
    "quantum sensing",
    "quantum sensor",
    "quantum sensors",
    "quantum communication",
    "quantum communications",
    "quantum network",
    "quantum networks",
    "quantum internet",
    "quantum cryptography",
    "quantum encryption",
    "quantum key distribution",
    "qkd",
    "post-quantum",
    "post quantum",
    "quantum-safe",
    "quantum safe",
    "quantum-resistant",
    "quantum resistant",
    "superconducting qubit",
    "trapped ion",
    "ion trap",
    "photonic quantum",
    "topological quantum",
    "量子",
    "量子信息",
    "量子计算",
    "量子通信",
    "量子加密",
    "量子密钥分发",
    "量子网络",
    "量子互联网",
    "量子芯片",
    "量子传感",
]

TEXT_COLUMNS = [
    "title",
    "article_title",
    "excerpt",
    "categories",
    "content",
]


def clean_text(text: object) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def safe_filename(text: str, max_len: int = 90) -> str:
    text = re.sub(r'[\\/:*?"<>|]', "_", str(text))
    text = re.sub(r"\s+", "_", text).strip("._ ")
    return (text or "untitled")[:max_len]


def latest_raw_folder(base_dir: Path) -> Path:
    candidates = sorted(
        [path for path in base_dir.glob("techreport_news_raw_*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No techreport_news_raw_* folder found under {base_dir}")
    return candidates[0]


def load_raw_articles(input_dir: Path) -> pd.DataFrame:
    csv_path = input_dir / "techreport_news_raw_full.csv"
    jsonl_path = input_dir / "techreport_news_raw_full.jsonl"

    if csv_path.exists():
        return pd.read_csv(csv_path)

    if jsonl_path.exists():
        rows = []
        with jsonl_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)

    raise FileNotFoundError(f"No raw CSV or JSONL found in {input_dir}")


def build_search_text(row: pd.Series) -> str:
    values = []
    for column in TEXT_COLUMNS:
        if column in row.index:
            values.append(clean_text(row[column]))
    return "\n".join(values)


def matched_keywords(text: str, keywords: list[str]) -> list[str]:
    text_lower = text.lower()
    matches = []
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if re.search(rf"(?<![a-z0-9-]){re.escape(keyword_lower)}(?![a-z0-9-])", text_lower):
            matches.append(keyword)
    return sorted(set(matches), key=str.lower)


def classify_quantum_articles(df: pd.DataFrame, keywords: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    rows = []
    for _, row in df.iterrows():
        text = build_search_text(row)
        matches = matched_keywords(text, keywords)
        if matches:
            new_row = row.copy()
            new_row["matched_keywords"] = "; ".join(matches)
            new_row["match_count"] = len(matches)
            rows.append(new_row)

    if not rows:
        return pd.DataFrame(columns=list(df.columns) + ["matched_keywords", "match_count"])

    result = pd.DataFrame(rows)
    if "date" in result.columns:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        result = result.sort_values(["date", "match_count"], ascending=[False, False])
    else:
        result = result.sort_values("match_count", ascending=False)
    return result.reset_index(drop=True)


def write_text_copies(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    text_dir = out_dir / "article_texts"
    text_dir.mkdir(parents=True, exist_ok=True)

    text_paths = []
    for index, row in df.iterrows():
        date_value = pd.to_datetime(row.get("date"), errors="coerce")
        date_str = date_value.strftime("%Y-%m-%d") if pd.notna(date_value) else "unknown_date"
        title = clean_text(row.get("title") or row.get("article_title") or "untitled")
        out_path = text_dir / f"{date_str}_{index + 1:04d}_{safe_filename(title)}.txt"

        source_txt = clean_text(row.get("content_txt_path"))
        if source_txt and Path(source_txt).exists():
            shutil.copy2(source_txt, out_path)
        else:
            out_path.write_text(
                "\n".join(
                    [
                        f"Title: {title}",
                        f"Date: {date_str}",
                        f"Author: {clean_text(row.get('author'))}",
                        f"Categories: {clean_text(row.get('categories'))}",
                        f"URL: {clean_text(row.get('url'))}",
                        f"Matched keywords: {clean_text(row.get('matched_keywords'))}",
                        "",
                        clean_text(row.get("content")),
                    ]
                ),
                encoding="utf-8",
            )
        text_paths.append(str(out_path))

    result = df.copy()
    result["quantum_content_txt_path"] = text_paths
    return result


def export_results(df: pd.DataFrame, input_dir: Path, out_base: Path, keywords: list[str]) -> Path:
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base / f"techreport_quantum_news_{run_time}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not df.empty:
        df = write_text_copies(df, out_dir)

    csv_path = out_dir / "techreport_quantum_news.csv"
    jsonl_path = out_dir / "techreport_quantum_news.jsonl"
    xlsx_path = out_dir / "techreport_quantum_news_summary.xlsx"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False, date_format="iso")

    preview = df.copy()
    if "content" in preview.columns:
        preview["content_preview"] = preview["content"].fillna("").astype(str).str.slice(0, 3000)
        preview = preview.drop(columns=["content"], errors="ignore")

    monthly = pd.DataFrame(columns=["month", "news_count"])
    if not df.empty and "date" in df.columns:
        dated = df.copy()
        dated["date"] = pd.to_datetime(dated["date"], errors="coerce")
        dated["month"] = dated["date"].dt.to_period("M").astype(str)
        monthly = (
            dated.groupby("month", dropna=False, as_index=False)
            .agg(news_count=("url", "count"))
            .sort_values("month", ascending=False)
        )

    keyword_rows = []
    if not df.empty and "matched_keywords" in df.columns:
        for keyword_list in df["matched_keywords"].fillna(""):
            for keyword in [item.strip() for item in keyword_list.split(";") if item.strip()]:
                keyword_rows.append({"keyword": keyword})
    keyword_summary = pd.DataFrame(keyword_rows)
    if not keyword_summary.empty:
        keyword_summary = (
            keyword_summary.groupby("keyword", as_index=False)
            .agg(news_count=("keyword", "count"))
            .sort_values("news_count", ascending=False)
        )
    else:
        keyword_summary = pd.DataFrame(columns=["keyword", "news_count"])

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        preview.to_excel(writer, sheet_name="quantum_news_preview", index=False)
        monthly.to_excel(writer, sheet_name="monthly_count", index=False)
        keyword_summary.to_excel(writer, sheet_name="keyword_count", index=False)

    metadata = {
        "source_raw_folder": str(input_dir),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "article_count": int(len(df)),
        "keywords": keywords,
        "csv": str(csv_path),
        "jsonl": str(jsonl_path),
        "xlsx": str(xlsx_path),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(df)} quantum information articles to: {out_dir}")
    return out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter TechReport articles for quantum information.")
    parser.add_argument("--input-dir", type=Path, default=None, help="Raw crawl folder to filter.")
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE, help="Base output folder.")
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=None,
        help="Optional custom keywords. If omitted, the built-in quantum information list is used.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir or latest_raw_folder(RAW_DATA_BASE)
    keywords = args.keywords if args.keywords else QUANTUM_KEYWORDS

    raw_df = load_raw_articles(input_dir)
    quantum_df = classify_quantum_articles(raw_df, keywords)
    export_results(quantum_df, input_dir, args.out_base, keywords)


if __name__ == "__main__":
    main()
