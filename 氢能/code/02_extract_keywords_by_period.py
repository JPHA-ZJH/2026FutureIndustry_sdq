from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "articles_corpus.csv"
OUT_DIR = PROJECT_ROOT / "output" / "keyword_results"
BIMONTH_PATH = OUT_DIR / "keywords_by_bimonth.csv"
QUARTER_PATH = OUT_DIR / "keywords_by_quarter.csv"

TEXT_COLUMNS = ["title", "tags", "meta_keywords", "content"]
MAX_KEYWORDS_PER_PERIOD = 120

# 研究重点词：这些词即使较短也优先保留。
PROTECTED_KEYWORDS = {
    "hydrogen production",
    "electrolysis",
    "electrolyzer",
    "electrolyser",
    "fuel cell",
    "fuel cells",
    "hydrogen storage",
    "liquid hydrogen",
    "ammonia",
    "methanol",
    "hydrogen refueling station",
    "hydrogen refuelling station",
    "hydrogen pipeline",
    "hydrogen aviation",
    "hydrogen aircraft",
    "hydrogen truck",
    "hydrogen trucks",
    "shipping",
    "steel",
    "hydrogen turbine",
    "power generation",
    "green hydrogen",
    "blue hydrogen",
    "e fuel",
    "synthetic fuel",
}

TECH_PATTERN = re.compile(
    r"\b("
    r"hydrogen|fuel cell|electrolys|electrolyz|ammonia|methanol|efuel|e fuel|"
    r"synthetic fuel|storage|carrier|pipeline|refuel|refuelling|refueling|"
    r"aviation|aircraft|truck|shipping|steel|turbine|power generation|"
    r"production|infrastructure|transport|decarbon|industrial|liquid"
    r")\b",
    re.I,
)

CUSTOM_STOP_WORDS = {
    "said",
    "says",
    "new",
    "news",
    "read",
    "article",
    "company",
    "companies",
    "market",
    "project",
    "projects",
    "use",
    "uses",
    "using",
    "based",
    "including",
    "according",
    "million",
    "billion",
    "year",
    "years",
    "today",
    "future",
    "world",
}


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("e-fuel", "e fuel").replace("e-fuels", "e fuels")
    text = text.replace("fuel-cell", "fuel cell").replace("zero-emission", "zero emission")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_relevant_keyword(keyword: str) -> bool:
    if keyword in PROTECTED_KEYWORDS:
        return True
    if len(keyword) <= 2:
        return False
    if TECH_PATTERN.search(keyword):
        return True
    return False


def keyword_frequency(texts: pd.Series, keyword: str) -> tuple[int, int]:
    pattern = re.compile(rf"(?<!\w){re.escape(keyword)}(?!\w)")
    total = 0
    docs = 0
    for text in texts:
        matches = pattern.findall(text)
        if matches:
            docs += 1
            total += len(matches)
    return total, docs


def extract_for_period(df: pd.DataFrame, period_col: str) -> pd.DataFrame:
    rows = []
    periods = sorted(x for x in df[period_col].dropna().unique() if str(x).strip())

    for period in periods:
        period_df = df[df[period_col] == period].copy()
        texts = period_df["analysis_text"]
        doc_total = len(period_df)
        if doc_total == 0:
            continue

        min_df = 1 if doc_total < 5 else 2
        vectorizer = TfidfVectorizer(
            lowercase=False,
            stop_words=list(ENGLISH_STOP_WORDS | CUSTOM_STOP_WORDS),
            ngram_range=(1, 4),
            min_df=min_df,
            max_df=0.95,
            token_pattern=r"(?u)\b[a-z][a-z0-9]+\b",
        )

        try:
            matrix = vectorizer.fit_transform(texts)
        except ValueError:
            continue

        features = vectorizer.get_feature_names_out()
        scores = matrix.sum(axis=0).A1
        ranked = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)

        kept = 0
        seen = Counter()
        for keyword, score in ranked:
            keyword = keyword.strip()
            if not is_relevant_keyword(keyword):
                continue

            # 避免同一长短词簇完全挤占榜单。
            root = keyword.split()[0]
            if seen[root] >= 20 and keyword not in PROTECTED_KEYWORDS:
                continue
            seen[root] += 1

            frequency, document_count = keyword_frequency(texts, keyword)
            if frequency == 0:
                continue

            rows.append(
                {
                    "period": period,
                    "keyword": keyword,
                    "tfidf_score": round(float(score), 6),
                    "frequency": int(frequency),
                    "document_count": int(document_count),
                    "document_share": round(document_count / doc_total, 6),
                }
            )
            kept += 1
            if kept >= MAX_KEYWORDS_PER_PERIOD:
                break

    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Corpus not found: {CORPUS_PATH}. Run 01_build_corpus_from_txt.py first.")

    df = pd.read_csv(CORPUS_PATH).fillna("")
    df["analysis_text"] = (
        df[TEXT_COLUMNS].astype(str).agg(" ".join, axis=1).map(normalize_text)
    )

    bimonth = extract_for_period(df, "bi_month_period")
    quarter = extract_for_period(df, "quarter")

    bimonth.to_csv(BIMONTH_PATH, index=False, encoding="utf-8-sig")
    quarter.to_csv(QUARTER_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved bimonth keywords to {BIMONTH_PATH}")
    print(f"Saved quarter keywords to {QUARTER_PATH}")


if __name__ == "__main__":
    main()
