from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "output" / "keyword_results" / "keywords_by_bimonth.csv"
OUT_DIR = PROJECT_ROOT / "output" / "keyword_results"
TREND_PATH = OUT_DIR / "keyword_trends_bimonth.csv"
EMERGING_PATH = OUT_DIR / "emerging_keywords_bimonth.csv"
DECLINING_PATH = OUT_DIR / "declining_keywords_bimonth.csv"

SMOOTH = 1e-6
LOOKBACK_PERIODS = 3
MIN_RECENT_DOCUMENT_COUNT = 2


def period_sort_key(period: str) -> tuple[int, int]:
    year, months = period.split("-", 1)
    start_month = months.split("_", 1)[0]
    return int(year), int(start_month)


def build_full_grid(df: pd.DataFrame) -> pd.DataFrame:
    periods = sorted(df["period"].dropna().unique(), key=period_sort_key)
    keywords = sorted(df["keyword"].dropna().unique())
    grid = pd.MultiIndex.from_product([periods, keywords], names=["period", "keyword"]).to_frame(index=False)
    full = grid.merge(df, on=["period", "keyword"], how="left")
    for col in ["tfidf_score", "frequency", "document_count", "document_share"]:
        full[col] = full[col].fillna(0)
    full["period_order"] = full["period"].map({period: i for i, period in enumerate(periods)})
    return full


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Keyword table not found: {IN_PATH}. Run 02_extract_keywords_by_period.py first.")

    df = pd.read_csv(IN_PATH).fillna("")
    if df.empty:
        raise ValueError("keywords_by_bimonth.csv is empty.")

    full = build_full_grid(df)
    full = full.sort_values(["keyword", "period_order"]).reset_index(drop=True)
    full["document_share_change"] = full.groupby("keyword")["document_share"].diff().fillna(0)
    full["tfidf_score_change"] = full.groupby("keyword")["tfidf_score"].diff().fillna(0)
    full["document_share_growth_rate"] = (
        (full["document_share"] + SMOOTH)
        / (full.groupby("keyword")["document_share"].shift(1).fillna(0) + SMOOTH)
        - 1
    )
    full.to_csv(TREND_PATH, index=False, encoding="utf-8-sig")

    periods = sorted(df["period"].dropna().unique(), key=period_sort_key)
    recent_period = periods[-1]
    previous_periods = periods[-(LOOKBACK_PERIODS + 1) : -1]
    recent = full[full["period"] == recent_period].copy()
    previous = full[full["period"].isin(previous_periods)].copy()

    baseline = (
        previous.groupby("keyword", as_index=False)
        .agg(
            baseline_document_share=("document_share", "mean"),
            baseline_document_count=("document_count", "mean"),
            baseline_tfidf_score=("tfidf_score", "mean"),
        )
    )

    scored = recent.merge(baseline, on="keyword", how="left").fillna(0)
    scored["recent_period"] = recent_period
    scored["baseline_periods"] = ", ".join(previous_periods)
    scored["document_share_delta"] = scored["document_share"] - scored["baseline_document_share"]
    scored["growth_rate"] = (
        (scored["document_share"] + SMOOTH)
        / (scored["baseline_document_share"] + SMOOTH)
        - 1
    )

    common_cols = [
        "recent_period",
        "baseline_periods",
        "keyword",
        "tfidf_score",
        "document_count",
        "document_share",
        "baseline_tfidf_score",
        "baseline_document_count",
        "baseline_document_share",
        "document_share_delta",
        "growth_rate",
    ]

    emerging = scored[
        (scored["document_count"] >= MIN_RECENT_DOCUMENT_COUNT)
        & (scored["document_share_delta"] > 0)
    ].sort_values(["growth_rate", "document_share_delta", "document_share"], ascending=False)

    declining = scored[
        (scored["baseline_document_count"] >= MIN_RECENT_DOCUMENT_COUNT)
        & (scored["document_share_delta"] < 0)
    ].sort_values(["document_share_delta", "baseline_document_share"], ascending=[True, False])

    emerging[common_cols].to_csv(EMERGING_PATH, index=False, encoding="utf-8-sig")
    declining[common_cols].to_csv(DECLINING_PATH, index=False, encoding="utf-8-sig")

    print(f"Saved keyword trends to {TREND_PATH}")
    print(f"Saved emerging keywords to {EMERGING_PATH}")
    print(f"Saved declining keywords to {DECLINING_PATH}")


if __name__ == "__main__":
    main()
