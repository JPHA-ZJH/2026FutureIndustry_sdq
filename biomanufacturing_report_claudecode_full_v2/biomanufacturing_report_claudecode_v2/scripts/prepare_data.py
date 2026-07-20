from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PATENT_FILE, FIRM_YEAR_FILE, FIRM_FILE, KEYWORD_FILE, CITY_FILE, PROVINCE_FILE,
    CLEAN_DIR, CLEAN_PATENT_FILE, CLEAN_FIRM_YEAR_FILE, CLEAN_FIRM_FILE, CLEAN_KEYWORD_FILE,
    AUDIT_DIR, DROP_EXACT_DUPLICATES, TECH_ROUTE_MAP,
)
from scripts.utils import (
    ensure_dirs, build_city_province_mapping, add_first_applicant_fields,
    add_location_fields, join_unique, mode_or_blank,
)


def load_raw():
    patents = pd.read_csv(PATENT_FILE, low_memory=False)
    raw_firm_year = pd.read_csv(FIRM_YEAR_FILE, low_memory=False)
    raw_firms = pd.read_csv(FIRM_FILE, low_memory=False)
    keywords = pd.read_csv(KEYWORD_FILE, low_memory=False)
    cities = pd.read_csv(CITY_FILE)
    provinces = pd.read_csv(PROVINCE_FILE)
    return patents, raw_firm_year, raw_firms, keywords, cities, provinces


def prepare_patents(patents: pd.DataFrame, city_mapping: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    p = patents.copy()
    original_rows = len(p)
    flag = pd.to_numeric(p.get("is_biomanufacturing_patent", 1), errors="coerce").fillna(0).astype(int)
    p = p[flag.eq(1)].copy()
    relevant_rows = len(p)

    exact_duplicates = int(p.duplicated(keep=False).sum())
    if DROP_EXACT_DUPLICATES:
        p = p.drop_duplicates().copy()

    p = add_first_applicant_fields(p)
    p = add_location_fields(p, city_mapping)
    p["year"] = pd.to_numeric(p.get("year", p.get("公开公告年份")), errors="coerce").astype("Int64")
    p["biomanufacturing_score"] = pd.to_numeric(p.get("biomanufacturing_score"), errors="coerce")
    p["evidence_score"] = pd.to_numeric(p.get("evidence_score"), errors="coerce")
    p["被引证次数_num"] = pd.to_numeric(p.get("被引证次数"), errors="coerce")
    p["main_category"] = p.get("main_category", "").fillna("未分类").astype(str).str.strip().replace("", "未分类")
    p["main_sub_category"] = p.get("main_sub_category", "").fillna("未分类").astype(str).str.strip().replace("", "未分类")
    p["报告技术路线"] = p["main_category"].map(TECH_ROUTE_MAP).fillna("其他/待核验")
    p["国内省级地区"] = p["省级地区"].where(p["是否中国境内申请人"].eq(1), "境外")

    basic_cols = [c for c in ["专利名称", "第一申请人", "year", "IPC主分类号"] if c in p.columns]
    near_duplicate_rows = int(p.duplicated(basic_cols, keep=False).sum()) if basic_cols else 0

    audit = {
        "raw_patent_rows": int(original_rows),
        "biomanufacturing_patent_rows_before_exact_dedup": int(relevant_rows),
        "exact_duplicate_rows_flagged": exact_duplicates,
        "clean_patent_rows": int(len(p)),
        "near_duplicate_rows_basic_key": near_duplicate_rows,
        "first_applicant_reextracted_rows": int(p["第一申请人提取是否变化"].sum()),
        "type_review_rows": int(p["主体类型需复核"].sum()),
        "type_review_share": float(p["主体类型需复核"].mean()),
        "unmapped_domestic_city_rows": int(p["城市映射状态"].eq("中国城市待映射").sum()),
        "missing_city_rows": int(p["城市映射状态"].eq("缺少城市信息").sum()),
        "years": sorted(p["year"].dropna().astype(int).unique().tolist()),
        "raw_technology_categories": sorted(p["main_category"].dropna().unique().tolist()),
        "report_technology_routes": sorted(p["报告技术路线"].dropna().unique().tolist()),
        "domestic_patent_rows": int(p["是否中国境内申请人"].eq(1).sum()),
        "foreign_patent_rows": int(p["是否中国境内申请人"].eq(0).sum()),
        "citation_coverage": float(p["被引证次数_num"].notna().mean()),
        "medical_institution_rows": int(p["主体类型"].eq("医疗机构").sum()),
    }
    return p, audit


def prepare_keywords(keywords: pd.DataFrame) -> pd.DataFrame:
    k = keywords.copy()
    k["技术领域"] = k.get("技术领域", "").fillna("未分类").astype(str).str.strip().replace("", "未分类")
    k["报告技术路线"] = k["技术领域"].map(TECH_ROUTE_MAP).fillna("其他/待核验")
    k["核心程度得分"] = pd.to_numeric(k.get("核心程度得分"), errors="coerce")
    return k.drop_duplicates().copy()


def rebuild_firm_tables(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    usable = p[p["第一申请人"].fillna("").astype(str).str.strip().ne("")].copy()
    group_year = ["第一申请人", "主体类型", "省级地区", "第一申请人城市标准化", "year"]
    fy = (
        usable.groupby(group_year, dropna=False)
        .agg(
            biomanufacturing_patent_count=("专利名称", "size"),
            biomanufacturing_score_sum=("biomanufacturing_score", "sum"),
            biomanufacturing_score_mean=("biomanufacturing_score", "mean"),
            main_categories=("main_category", join_unique),
            main_sub_categories=("main_sub_category", join_unique),
            report_tech_routes=("报告技术路线", join_unique),
            firm_main_category=("main_category", mode_or_blank),
            firm_main_sub_category=("main_sub_category", mode_or_blank),
            firm_main_report_tech_route=("报告技术路线", mode_or_blank),
            type_review_count=("主体类型需复核", "sum"),
        )
        .reset_index()
        .sort_values(["省级地区", "第一申请人", "year"])
    )

    group_firm = ["第一申请人", "主体类型", "省级地区", "第一申请人城市标准化"]
    f = (
        usable.groupby(group_firm, dropna=False)
        .agg(
            first_year=("year", "min"), last_year=("year", "max"), active_years=("year", "nunique"),
            biomanufacturing_patent_count=("专利名称", "size"),
            biomanufacturing_score_sum=("biomanufacturing_score", "sum"),
            biomanufacturing_score_mean=("biomanufacturing_score", "mean"),
            main_categories=("main_category", join_unique),
            main_sub_categories=("main_sub_category", join_unique),
            report_tech_routes=("报告技术路线", join_unique),
            firm_main_category=("main_category", mode_or_blank),
            firm_main_sub_category=("main_sub_category", mode_or_blank),
            firm_main_report_tech_route=("报告技术路线", mode_or_blank),
            type_review_count=("主体类型需复核", "sum"),
        )
        .reset_index()
        .sort_values(["biomanufacturing_patent_count", "active_years"], ascending=False)
    )
    return fy, f


def write_audits(p, raw_fy, raw_f, clean_fy, clean_f, keywords, audit):
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    audit.update({
        "raw_firm_year_rows": int(len(raw_fy)),
        "raw_firm_rows": int(len(raw_f)),
        "rebuilt_firm_year_rows": int(len(clean_fy)),
        "rebuilt_firm_rows": int(len(clean_f)),
        "raw_firm_year_patent_sum": int(pd.to_numeric(raw_fy.get("biomanufacturing_patent_count"), errors="coerce").sum()),
        "raw_firm_patent_sum": int(pd.to_numeric(raw_f.get("biomanufacturing_patent_count"), errors="coerce").sum()),
        "rebuilt_firm_year_patent_sum": int(clean_fy["biomanufacturing_patent_count"].sum()),
        "rebuilt_firm_patent_sum": int(clean_f["biomanufacturing_patent_count"].sum()),
        "keyword_rows": int(len(keywords)),
    })
    (AUDIT_DIR / "data_cleaning_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    city_audit = p.groupby(["第一申请人城市_原值", "第一申请人城市标准化", "省级地区", "城市映射状态"], dropna=False).size().rename("专利数").reset_index().sort_values("专利数", ascending=False)
    city_audit.to_csv(AUDIT_DIR / "city_normalization_audit.csv", index=False, encoding="utf-8-sig")
    p[p["城市映射状态"].eq("中国城市待映射")][["第一申请人城市_原值", "第一申请人城市标准化"]].value_counts().rename("专利数").reset_index().to_csv(AUDIT_DIR / "unmapped_domestic_cities.csv", index=False, encoding="utf-8-sig")

    type_audit = p.groupby(["第一申请人类型", "主体类型", "主体类型判断依据", "主体类型需复核"], dropna=False).size().rename("专利数").reset_index().sort_values("专利数", ascending=False)
    type_audit.to_csv(AUDIT_DIR / "applicant_type_audit.csv", index=False, encoding="utf-8-sig")
    review_cols = [c for c in ["第一申请人", "第一申请人类型", "申请人", "申请人类型", "主体类型", "主体类型判断依据"] if c in p.columns]
    p[p["主体类型需复核"]][review_cols].drop_duplicates().to_csv(AUDIT_DIR / "applicant_type_review_list.csv", index=False, encoding="utf-8-sig")
    p[p["第一申请人提取是否变化"]][["申请人", "第一申请人_原值", "第一申请人"]].drop_duplicates().to_csv(AUDIT_DIR / "first_applicant_reextraction_audit.csv", index=False, encoding="utf-8-sig")

    route_map_audit = p[["main_category", "报告技术路线"]].drop_duplicates().sort_values(["报告技术路线", "main_category"])
    route_map_audit.to_csv(AUDIT_DIR / "technology_route_mapping_audit.csv", index=False, encoding="utf-8-sig")
    keyword_summary = keywords.groupby(["报告技术路线", "技术领域"], dropna=False).agg(关键词数=("关键词", "nunique"), 平均得分=("核心程度得分", "mean")).reset_index()
    keyword_summary.to_csv(AUDIT_DIR / "keyword_dictionary_audit.csv", index=False, encoding="utf-8-sig")

    years = "、".join(map(str, audit["years"]))
    lines = [
        "# 生物制造数据清洗与审计结果", "",
        f"- 原始专利记录：{audit['raw_patent_rows']:,}条。",
        f"- 清洗后生物制造相关专利：{audit['clean_patent_rows']:,}条。",
        f"- 数据年份：{years}。",
        f"- 第一申请人重新提取后发生变化：{audit['first_applicant_reextracted_rows']:,}条。",
        f"- 第一申请人类型仍需人工复核：{audit['type_review_rows']:,}条，占{audit['type_review_share']:.2%}。",
        f"- 中国城市待映射：{audit['unmapped_domestic_city_rows']:,}条；缺少城市信息：{audit['missing_city_rows']:,}条。",
        f"- 医疗机构专利：{audit['medical_institution_rows']:,}条，单独分类，不并入高校、科研院所或企业。",
        f"- 被引证次数字段覆盖率：{audit['citation_coverage']:.2%}，默认不作为主分析指标。",
        f"- 关键词词典：{audit['keyword_rows']:,}条。", "",
        "## 本版本的关键清洗规则", "",
        "1. 城市统一去除常见后缀，例如“上海市”和“上海”统一为“上海”；省份映射使用全国地级行政区参考表，未映射值单独输出。",
        "2. 合作专利的原始类型可能包含所有申请人的类型。代码重新提取第一申请人，并依据第一申请人名称判断唯一类型。",
        "3. 生物制造数据中医院等主体数量较多，本项目将其单列为“医疗机构”，避免混入高校、科研院所或企业。",
        "4. 原始13类技术方向被映射为7类报告技术路线，以提高咨询报告可读性；原始分类仍保留在清洗数据和附表中。", "",
        "## 报告写作约束", "",
        "- 技术分析围绕报告技术路线和原始main_category展开，不使用相关性高低作为正文分类。",
        "- 不分析industry_segment所代表的产业链环节。",
        "- 企业分析仅使用主体类型为“企业”的记录；高校、科研院所和医疗机构分别统计。",
        "- 第五节省际主体结构比较，核心图表以企业、高校、科研院所三类专利为分母计算占比；医疗机构及其他主体不进入该占比，但在附表中保留。",
        "- 专利授权数据不等同于产值、市场份额、融资规模或商业化水平。",
        "- 最新年份可能受授权滞后和数据库更新进度影响，趋势判断应保守。",
    ]
    (AUDIT_DIR / "data_cleaning_audit.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dirs(CLEAN_DIR, AUDIT_DIR)
    patents, raw_fy, raw_f, keywords, cities, provinces = load_raw()
    city_mapping = build_city_province_mapping(cities, provinces)
    city_mapping.to_csv(CLEAN_DIR / "city_province_mapping_used.csv", index=False, encoding="utf-8-sig")
    p, audit = prepare_patents(patents, city_mapping)
    k = prepare_keywords(keywords)
    fy, f = rebuild_firm_tables(p)
    p.to_csv(CLEAN_PATENT_FILE, index=False, encoding="utf-8-sig")
    fy.to_csv(CLEAN_FIRM_YEAR_FILE, index=False, encoding="utf-8-sig")
    f.to_csv(CLEAN_FIRM_FILE, index=False, encoding="utf-8-sig")
    k.to_csv(CLEAN_KEYWORD_FILE, index=False, encoding="utf-8-sig")
    write_audits(p, raw_fy, raw_f, fy, f, k, audit)
    print(f"清洗完成：{CLEAN_PATENT_FILE}")
    print(f"重建主体—年份表：{CLEAN_FIRM_YEAR_FILE}")
    print(f"重建主体汇总表：{CLEAN_FIRM_FILE}")


if __name__ == "__main__":
    main()
