from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PATENT_FILE, FIRM_YEAR_FILE, FIRM_FILE, CITY_FILE, PROVINCE_FILE,
    CLEAN_DIR, CLEAN_PATENT_FILE, CLEAN_FIRM_YEAR_FILE, CLEAN_FIRM_FILE,
    AUDIT_DIR, DROP_EXACT_DUPLICATES,
)
from scripts.utils import (
    ensure_dirs, build_city_province_mapping, add_first_applicant_fields,
    add_location_fields, join_unique, mode_or_blank,
)


def load_raw():
    patents = pd.read_csv(PATENT_FILE, low_memory=False)
    raw_firm_year = pd.read_csv(FIRM_YEAR_FILE, low_memory=False)
    raw_firms = pd.read_csv(FIRM_FILE, low_memory=False)
    cities = pd.read_csv(CITY_FILE)
    provinces = pd.read_csv(PROVINCE_FILE)
    return patents, raw_firm_year, raw_firms, cities, provinces


def prepare_patents(patents: pd.DataFrame, city_mapping: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    p = patents.copy()
    original_rows = len(p)
    p = p[p.get("is_quantum_patent", 1).fillna(0).astype(int).eq(1)].copy()
    quantum_rows = len(p)

    exact_duplicates = int(p.duplicated(keep=False).sum())
    if DROP_EXACT_DUPLICATES:
        p = p.drop_duplicates().copy()

    p = add_first_applicant_fields(p)
    p = add_location_fields(p, city_mapping)
    p["year"] = pd.to_numeric(p.get("year", p.get("公开公告年份")), errors="coerce").astype("Int64")
    p["quantum_score"] = pd.to_numeric(p.get("quantum_score"), errors="coerce")
    p["被引证次数_num"] = pd.to_numeric(p.get("被引证次数"), errors="coerce")
    p["main_category"] = p.get("main_category", "").fillna("未分类").astype(str).str.strip().replace("", "未分类")
    p["国内省级地区"] = p["省级地区"].where(p["是否中国境内申请人"].eq(1), "境外")

    # 用于审计近似重复，不自动删除。
    basic_cols = [c for c in ["专利名称", "第一申请人", "year", "IPC主分类号"] if c in p.columns]
    near_duplicate_rows = int(p.duplicated(basic_cols, keep=False).sum()) if basic_cols else 0

    audit = {
        "raw_patent_rows": int(original_rows),
        "quantum_patent_rows_before_exact_dedup": int(quantum_rows),
        "exact_duplicate_rows_flagged": exact_duplicates,
        "clean_patent_rows": int(len(p)),
        "near_duplicate_rows_basic_key": near_duplicate_rows,
        "first_applicant_reextracted_rows": int(p["第一申请人提取是否变化"].sum()),
        "type_review_rows": int(p["主体类型需复核"].sum()),
        "type_review_share": float(p["主体类型需复核"].mean()),
        "unmapped_domestic_city_rows": int(p["城市映射状态"].eq("中国城市待映射").sum()),
        "missing_city_rows": int(p["城市映射状态"].eq("缺少城市信息").sum()),
        "years": sorted(p["year"].dropna().astype(int).unique().tolist()),
        "technology_categories": sorted(p["main_category"].dropna().unique().tolist()),
        "domestic_patent_rows": int(p["是否中国境内申请人"].eq(1).sum()),
        "foreign_patent_rows": int(p["是否中国境内申请人"].eq(0).sum()),
        "citation_coverage": float(p["被引证次数_num"].notna().mean()),
    }
    return p, audit


def rebuild_firm_tables(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = p[p["第一申请人"].fillna("").astype(str).str.strip().ne("")].copy()
    group_year = ["第一申请人", "主体类型", "省级地区", "第一申请人城市标准化", "year"]
    fy = (
        usable.groupby(group_year, dropna=False)
        .agg(
            quantum_patent_count=("专利名称", "size"),
            quantum_score_sum=("quantum_score", "sum"),
            quantum_score_mean=("quantum_score", "mean"),
            main_categories=("main_category", join_unique),
            main_quantum_category=("main_category", mode_or_blank),
            type_review_count=("主体类型需复核", "sum"),
        )
        .reset_index()
        .sort_values(["省级地区", "第一申请人", "year"])
    )

    group_firm = ["第一申请人", "主体类型", "省级地区", "第一申请人城市标准化"]
    f = (
        usable.groupby(group_firm, dropna=False)
        .agg(
            first_year=("year", "min"),
            last_year=("year", "max"),
            active_years=("year", "nunique"),
            quantum_patent_count=("专利名称", "size"),
            quantum_score_sum=("quantum_score", "sum"),
            quantum_score_mean=("quantum_score", "mean"),
            main_categories=("main_category", join_unique),
            main_quantum_category=("main_category", mode_or_blank),
            type_review_count=("主体类型需复核", "sum"),
        )
        .reset_index()
        .sort_values(["quantum_patent_count", "active_years"], ascending=False)
    )
    return fy, f


def write_audits(p: pd.DataFrame, raw_fy: pd.DataFrame, raw_f: pd.DataFrame, clean_fy: pd.DataFrame, clean_f: pd.DataFrame, audit: dict):
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit.update({
        "raw_firm_year_rows": int(len(raw_fy)),
        "raw_firm_rows": int(len(raw_f)),
        "rebuilt_firm_year_rows": int(len(clean_fy)),
        "rebuilt_firm_rows": int(len(clean_f)),
        "raw_firm_year_patent_sum": int(pd.to_numeric(raw_fy.get("quantum_patent_count"), errors="coerce").sum()),
        "raw_firm_patent_sum": int(pd.to_numeric(raw_f.get("quantum_patent_count"), errors="coerce").sum()),
        "rebuilt_firm_year_patent_sum": int(clean_fy["quantum_patent_count"].sum()),
        "rebuilt_firm_patent_sum": int(clean_f["quantum_patent_count"].sum()),
    })
    (AUDIT_DIR / "data_cleaning_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    city_audit = (
        p.groupby(["第一申请人城市_原值", "第一申请人城市标准化", "省级地区", "城市映射状态"], dropna=False)
        .size().rename("专利数").reset_index().sort_values("专利数", ascending=False)
    )
    city_audit.to_csv(AUDIT_DIR / "city_normalization_audit.csv", index=False, encoding="utf-8-sig")
    p[p["城市映射状态"].eq("中国城市待映射")][["第一申请人城市_原值", "第一申请人城市标准化"]] \
        .value_counts().rename("专利数").reset_index().to_csv(AUDIT_DIR / "unmapped_domestic_cities.csv", index=False, encoding="utf-8-sig")

    type_audit = (
        p.groupby(["第一申请人类型", "主体类型", "主体类型判断依据", "主体类型需复核"], dropna=False)
        .size().rename("专利数").reset_index().sort_values("专利数", ascending=False)
    )
    type_audit.to_csv(AUDIT_DIR / "applicant_type_audit.csv", index=False, encoding="utf-8-sig")
    review_cols = [c for c in ["第一申请人", "第一申请人类型", "申请人", "申请人类型", "主体类型", "主体类型判断依据"] if c in p.columns]
    p[p["主体类型需复核"]][review_cols].drop_duplicates().to_csv(
        AUDIT_DIR / "applicant_type_review_list.csv", index=False, encoding="utf-8-sig"
    )

    mismatch = p[p["第一申请人提取是否变化"]][["申请人", "第一申请人_原值", "第一申请人"]].drop_duplicates()
    mismatch.to_csv(AUDIT_DIR / "first_applicant_reextraction_audit.csv", index=False, encoding="utf-8-sig")

    years = "、".join(map(str, audit["years"]))
    lines = [
        "# 数据清洗与审计结果", "",
        f"- 原始专利记录：{audit['raw_patent_rows']:,}条。",
        f"- 清洗后量子专利记录：{audit['clean_patent_rows']:,}条。",
        f"- 数据年份：{years}。",
        f"- 第一申请人重新提取后发生变化：{audit['first_applicant_reextracted_rows']:,}条。",
        f"- 第一申请人类型仍需人工复核：{audit['type_review_rows']:,}条，占{audit['type_review_share']:.2%}。",
        f"- 中国城市待映射：{audit['unmapped_domestic_city_rows']:,}条；缺少城市信息：{audit['missing_city_rows']:,}条。",
        f"- 被引证次数字段覆盖率：{audit['citation_coverage']:.2%}。本报告默认不把被引指标作为主分析依据。",
        "",
        "## 本版本的两项关键修正", "",
        "1. 城市名称先统一去除常见后缀，例如“上海市”和“上海”统一为“上海”；省份映射使用全国地级行政区参考表，而不是只使用当前样本中出现的城市。未映射城市单独输出，不进行主观猜测。",
        "2. 合作专利中的“申请人类型”可能包含所有申请人的类型。代码重新提取第一申请人，并优先依据第一申请人名称识别唯一类型；名称不能判断时才使用原字段中的单一类型，仍无法判断的标记为“待复核”。",
        "",
        "## 报告写作约束", "",
        "- 技术内部分析仅使用main_category/main_quantum_category，不使用相关性等级，也不分析产业链环节。",
        "- 省际比较使用清洗后的省级地区。未映射记录不得被擅自分配到省份。",
        "- 企业分析只使用主体类型为“企业”的记录；高校和科研院所另行分析；“待复核”主体不进入企业排名。",
        "- 专利授权数量反映创新产出和布局，不等同于产值、市场份额、融资规模或产品商业化程度。",
        "- 最新年份可能受到授权滞后和数据库更新进度影响，趋势判断需保守。",
    ]
    (AUDIT_DIR / "data_cleaning_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dirs(CLEAN_DIR, AUDIT_DIR)
    patents, raw_fy, raw_f, cities, provinces = load_raw()
    city_mapping = build_city_province_mapping(cities, provinces)
    city_mapping.to_csv(CLEAN_DIR / "city_province_mapping_used.csv", index=False, encoding="utf-8-sig")
    p, audit = prepare_patents(patents, city_mapping)
    fy, f = rebuild_firm_tables(p)
    p.to_csv(CLEAN_PATENT_FILE, index=False, encoding="utf-8-sig")
    fy.to_csv(CLEAN_FIRM_YEAR_FILE, index=False, encoding="utf-8-sig")
    f.to_csv(CLEAN_FIRM_FILE, index=False, encoding="utf-8-sig")
    write_audits(p, raw_fy, raw_f, fy, f, audit)
    print(f"清洗完成：{CLEAN_PATENT_FILE}")
    print(f"重建企业—年份表：{CLEAN_FIRM_YEAR_FILE}")
    print(f"重建主体汇总表：{CLEAN_FIRM_FILE}")


if __name__ == "__main__":
    main()
