from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CLEAN_PATENT_FILE, CLEAN_FIRM_YEAR_FILE, CLEAN_FIRM_FILE, CLEAN_KEYWORD_FILE,
    OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, AUDIT_DIR,
    SHANGHAI, COMPARE_REGIONS, FOCUS_SUBJECT_TYPES, REPORT_TECH_ROUTES, RAW_TECH_CATEGORIES,
    TOP_SHANGHAI_ENTERPRISES, TOP_SHANGHAI_RESEARCH,
    SHANGHAI_CASE_COUNT, OTHER_CASE_COUNT, MAX_OTHER_CASES_PER_PROVINCE,
)
from scripts.utils import ensure_dirs, concentration_metrics, set_plot_style

REGION_COLORS = {
    "上海": "#B23A3A", "北京": "#3E648D", "安徽": "#438B70",
    "江苏": "#D18B38", "浙江": "#8267A4", "广东": "#4D8EAE",
}
ROUTE_COLORS = {
    "生物设计、读写与自动化工具": "#3E6E9E",
    "酶与蛋白质工程": "#4C9474",
    "细胞工厂与菌株工程": "#B57A39",
    "生物过程与规模化": "#8B5C8E",
    "原料与低碳路线": "#789447",
    "生物制造产品与应用": "#C3564A",
    "前沿制造与基础支撑": "#6E7781",
    "其他/待核验": "#B8B8B8",
}
SUBJECT_COLORS = {"企业": "#B23A3A", "高校": "#3E648D", "科研院所": "#4C9474"}


def read_clean_data():
    p = pd.read_csv(CLEAN_PATENT_FILE, low_memory=False)
    fy = pd.read_csv(CLEAN_FIRM_YEAR_FILE, low_memory=False)
    f = pd.read_csv(CLEAN_FIRM_FILE, low_memory=False)
    k = pd.read_csv(CLEAN_KEYWORD_FILE, low_memory=False)
    for df in (p, fy, f):
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return p, fy, f, k


def domestic_patents(p):
    return p[p["是否中国境内申请人"].eq(1)].copy()


def mapped_domestic_patents(p):
    d = domestic_patents(p)
    return d[d["省级地区"].notna() & d["省级地区"].astype(str).str.strip().ne("")].copy()


def reorder_route(df, col="技术路线"):
    out = df.copy()
    order = {v: i for i, v in enumerate(REPORT_TECH_ROUTES)}
    out["_order"] = out[col].map(order).fillna(99)
    return out.sort_values(["_order", col]).drop(columns="_order")


def nationwide_technology_tables(p):
    d = domestic_patents(p)
    annual = d.groupby(["year", "报告技术路线"], dropna=False).agg(
        专利数=("专利名称", "size"), 创新主体数=("第一申请人", "nunique")
    ).reset_index().rename(columns={"报告技术路线": "技术路线"})
    annual["企业专利数"] = d[d["主体类型"].eq("企业")].groupby(["year", "报告技术路线"]).size().reindex(
        pd.MultiIndex.from_frame(annual[["year", "技术路线"]].rename(columns={"技术路线": "报告技术路线"}))
    ).fillna(0).to_numpy(dtype=int)
    annual["当年全国专利占比"] = annual["专利数"] / annual.groupby("year")["专利数"].transform("sum")
    annual = reorder_route(annual)

    total = d.groupby("报告技术路线", dropna=False).agg(
        专利数=("专利名称", "size"), 创新主体数=("第一申请人", "nunique")
    ).reset_index().rename(columns={"报告技术路线": "技术路线"})
    ent_counts = d[d["主体类型"].eq("企业")].groupby("报告技术路线").size()
    ent_firms = d[d["主体类型"].eq("企业")].groupby("报告技术路线")["第一申请人"].nunique()
    total["企业专利数"] = total["技术路线"].map(ent_counts).fillna(0).astype(int)
    total["企业数"] = total["技术路线"].map(ent_firms).fillna(0).astype(int)
    total["专利占比"] = total["专利数"] / total["专利数"].sum()
    total = reorder_route(total)

    raw = d.groupby("main_category", dropna=False).agg(
        专利数=("专利名称", "size"), 创新主体数=("第一申请人", "nunique")
    ).reset_index().rename(columns={"main_category": "原始技术方向"})
    raw["专利占比"] = raw["专利数"] / raw["专利数"].sum()
    raw_order = {v: i for i, v in enumerate(RAW_TECH_CATEGORIES)}
    raw["_order"] = raw["原始技术方向"].map(raw_order).fillna(99)
    raw = raw.sort_values(["_order", "原始技术方向"]).drop(columns="_order")
    return annual, total, raw


def region_tables(p):
    d = mapped_domestic_patents(p)
    base = d.groupby("省级地区").agg(专利数=("专利名称", "size"), 创新主体数=("第一申请人", "nunique")).reset_index()
    for typ, prefix in [("企业", "企业"), ("高校", "高校"), ("科研院所", "科研院所"), ("医疗机构", "医疗机构")]:
        t = d[d["主体类型"].eq(typ)].groupby("省级地区").agg(**{f"{prefix}专利数": ("专利名称", "size"), f"{prefix}数": ("第一申请人", "nunique")}).reset_index()
        base = base.merge(t, on="省级地区", how="left")
    count_cols = [c for c in base.columns if c.endswith("专利数") or c.endswith("数")]
    base[count_cols] = base[count_cols].fillna(0)
    base["全国专利占比"] = base["专利数"] / base["专利数"].sum()
    base["专利数全国排名"] = base["专利数"].rank(method="min", ascending=False).astype(int)
    base["企业专利占比"] = base["企业专利数"] / base["专利数"].replace(0, np.nan)
    base = base.sort_values(["专利数", "创新主体数"], ascending=False)

    annual = d.groupby(["省级地区", "year"]).agg(专利数=("专利名称", "size"), 创新主体数=("第一申请人", "nunique")).reset_index()
    ent = d[d["主体类型"].eq("企业")].groupby(["省级地区", "year"]).agg(企业专利数=("专利名称", "size"), 企业数=("第一申请人", "nunique")).reset_index()
    annual = annual.merge(ent, on=["省级地区", "year"], how="left").fillna({"企业专利数": 0, "企业数": 0})
    annual["当年全国专利占比"] = annual["专利数"] / annual.groupby("year")["专利数"].transform("sum")
    annual["当年专利排名"] = annual.groupby("year")["专利数"].rank(method="min", ascending=False).astype(int)
    return base, annual


def region_technology_comparison(p, applicant_types=None):
    d = mapped_domestic_patents(p)
    d = d[d["省级地区"].isin(COMPARE_REGIONS)].copy()
    if applicant_types:
        d = d[d["主体类型"].isin(applicant_types)]
    out = d.groupby(["省级地区", "报告技术路线"]).agg(
        专利数=("专利名称", "size"), 创新主体数=("第一申请人", "nunique")
    ).reset_index().rename(columns={"报告技术路线": "技术路线"})
    out["地区内专利占比"] = out["专利数"] / out.groupby("省级地区")["专利数"].transform("sum")
    return reorder_route(out)


def shanghai_technology_tables(p):
    sh = mapped_domestic_patents(p)
    sh = sh[sh["省级地区"].eq(SHANGHAI)]
    total = sh.groupby("报告技术路线").agg(专利数=("专利名称", "size"), 创新主体数=("第一申请人", "nunique")).reset_index().rename(columns={"报告技术路线": "技术路线"})
    ent = sh[sh["主体类型"].eq("企业")].groupby("报告技术路线").size()
    total["企业专利数"] = total["技术路线"].map(ent).fillna(0).astype(int)
    total["专利占比"] = total["专利数"] / total["专利数"].sum()
    annual = sh.groupby(["year", "报告技术路线"]).agg(专利数=("专利名称", "size"), 创新主体数=("第一申请人", "nunique")).reset_index().rename(columns={"报告技术路线": "技术路线"})
    annual["当年上海专利占比"] = annual["专利数"] / annual.groupby("year")["专利数"].transform("sum")
    return reorder_route(total), reorder_route(annual)


def enterprise_tables(p, fy, f):
    ent_f = f[f["主体类型"].eq("企业") & f["省级地区"].notna()].copy()
    ent_fy = fy[fy["主体类型"].eq("企业") & fy["省级地区"].notna()].copy()
    region = ent_f.groupby("省级地区").agg(
        企业数=("第一申请人", "nunique"), 企业专利数=("biomanufacturing_patent_count", "sum"),
        企业平均专利数=("biomanufacturing_patent_count", "mean"),
        多年活跃企业数=("active_years", lambda x: int((x >= 2).sum())),
        三年及以上活跃企业数=("active_years", lambda x: int((x >= 3).sum())),
    ).reset_index()
    region["多年活跃企业占比"] = region["多年活跃企业数"] / region["企业数"].replace(0, np.nan)

    rows = []
    for reg, g in ent_f.groupby("省级地区"):
        row = {"省级地区": reg, "企业数": g["第一申请人"].nunique(), "企业专利数": int(g["biomanufacturing_patent_count"].sum())}
        row.update(concentration_metrics(g.groupby("第一申请人")["biomanufacturing_patent_count"].sum()))
        rows.append(row)
    concentration = pd.DataFrame(rows)

    annual = ent_fy.groupby(["省级地区", "year"]).agg(活跃企业数=("第一申请人", "nunique"), 企业专利数=("biomanufacturing_patent_count", "sum")).reset_index()
    entrants = ent_f.groupby(["省级地区", "first_year"]).size().rename("新进入企业数").reset_index().rename(columns={"first_year": "year"})
    annual = annual.merge(entrants, on=["省级地区", "year"], how="left").fillna({"新进入企业数": 0})
    annual["新进入企业占比"] = annual["新进入企业数"] / annual["活跃企业数"].replace(0, np.nan)

    def rename_rank(df):
        return df.rename(columns={
            "biomanufacturing_patent_count": "专利数", "active_years": "活跃年份数", "first_year": "最早年份", "last_year": "最晚年份",
            "firm_main_report_tech_route": "主要技术路线", "report_tech_routes": "技术路线覆盖", "firm_main_category": "主要原始技术方向",
            "第一申请人城市标准化": "城市",
        })

    sh_rank = rename_rank(ent_f[ent_f["省级地区"].eq(SHANGHAI)].copy()).sort_values(["专利数", "活跃年份数", "最晚年份"], ascending=False)
    sh_rank = sh_rank[["第一申请人", "城市", "专利数", "活跃年份数", "最早年份", "最晚年份", "主要技术路线", "主要原始技术方向", "技术路线覆盖"]]
    sh_cases = sh_rank.head(SHANGHAI_CASE_COUNT).copy()

    candidates = ent_f[~ent_f["省级地区"].eq(SHANGHAI)].sort_values(["biomanufacturing_patent_count", "active_years", "last_year"], ascending=False)
    selected, province_counter = [], {}
    for _, row in candidates.iterrows():
        province = row["省级地区"]
        if province_counter.get(province, 0) >= MAX_OTHER_CASES_PER_PROVINCE:
            continue
        selected.append(row)
        province_counter[province] = province_counter.get(province, 0) + 1
        if len(selected) >= OTHER_CASE_COUNT:
            break
    other_cases = pd.DataFrame(selected)
    if not other_cases.empty:
        other_cases = rename_rank(other_cases)
        other_cases = other_cases[["第一申请人", "省级地区", "城市", "专利数", "活跃年份数", "最早年份", "最晚年份", "主要技术路线", "主要原始技术方向", "技术路线覆盖"]]
    return region, concentration, annual, sh_rank, sh_cases, other_cases


def enterprise_technology_tables(p):
    ent = mapped_domestic_patents(p)
    ent = ent[ent["主体类型"].eq("企业")]
    sh = ent[ent["省级地区"].eq(SHANGHAI)]
    sh_table = sh.groupby("报告技术路线").agg(企业专利数=("专利名称", "size"), 企业数=("第一申请人", "nunique")).reset_index().rename(columns={"报告技术路线": "技术路线"})
    sh_table["企业专利占比"] = sh_table["企业专利数"] / sh_table["企业专利数"].sum()
    return reorder_route(sh_table), region_technology_comparison(p, ["企业"])


def research_tables(p, f):
    d = mapped_domestic_patents(p)
    res_types = ["高校", "科研院所"]
    res_p = d[d["主体类型"].isin(res_types)].copy()
    res_f = f[f["主体类型"].isin(res_types) & f["省级地区"].notna()].copy()

    sh_rank = res_f[res_f["省级地区"].eq(SHANGHAI)].rename(columns={
        "biomanufacturing_patent_count": "专利数", "active_years": "活跃年份数", "first_year": "最早年份", "last_year": "最晚年份",
        "firm_main_report_tech_route": "主要技术路线", "firm_main_category": "主要原始技术方向", "report_tech_routes": "技术路线覆盖", "第一申请人城市标准化": "城市",
    }).sort_values(["专利数", "活跃年份数"], ascending=False)
    sh_rank = sh_rank[["第一申请人", "主体类型", "城市", "专利数", "活跃年份数", "最早年份", "最晚年份", "主要技术路线", "主要原始技术方向", "技术路线覆盖"]]

    sh_res = res_p[res_p["省级地区"].eq(SHANGHAI)]
    sh_tech = sh_res.groupby(["主体类型", "报告技术路线"]).agg(专利数=("专利名称", "size"), 主体数=("第一申请人", "nunique")).reset_index().rename(columns={"报告技术路线": "技术路线"})
    sh_tech["类型内专利占比"] = sh_tech["专利数"] / sh_tech.groupby("主体类型")["专利数"].transform("sum")

    region = res_p.groupby(["省级地区", "主体类型"]).agg(专利数=("专利名称", "size"), 主体数=("第一申请人", "nunique")).reset_index()
    region_wide = region.pivot_table(index="省级地区", columns="主体类型", values=["专利数", "主体数"], aggfunc="sum", fill_value=0)
    region_wide.columns = [f"{typ}{metric}" for metric, typ in region_wide.columns]
    region_wide = region_wide.reset_index()
    region_wide["高校院所专利数"] = region_wide.get("高校专利数", 0) + region_wide.get("科研院所专利数", 0)
    region_wide["高校院所数"] = region_wide.get("高校主体数", 0) + region_wide.get("科研院所主体数", 0)

    compare_tech = region_technology_comparison(p, res_types)

    sh_all = d[d["省级地区"].eq(SHANGHAI) & d["主体类型"].isin(FOCUS_SUBJECT_TYPES)].copy()
    sh_compare = sh_all.groupby(["主体类型", "报告技术路线"]).agg(专利数=("专利名称", "size"), 主体数=("第一申请人", "nunique")).reset_index().rename(columns={"报告技术路线": "技术路线"})
    sh_compare["类型内专利占比"] = sh_compare["专利数"] / sh_compare.groupby("主体类型")["专利数"].transform("sum")
    return sh_rank, reorder_route(sh_tech), region_wide, compare_tech, reorder_route(sh_compare)


def region_subject_type_share(p):
    d = mapped_domestic_patents(p)
    compare_all = d[d["省级地区"].isin(COMPARE_REGIONS)].copy()
    total_all = compare_all.groupby("省级地区").size().rename("地区全部专利数")
    focus = compare_all[compare_all["主体类型"].isin(FOCUS_SUBJECT_TYPES)].copy()
    out = focus.groupby(["省级地区", "主体类型"]).size().rename("专利数").reset_index()
    focus_total = out.groupby("省级地区")["专利数"].transform("sum")
    out["三类主体内部占比"] = out["专利数"] / focus_total
    out["占地区全部专利比重"] = out["专利数"] / out["省级地区"].map(total_all)
    out["三类主体专利合计"] = focus_total
    return out


def keyword_tables(k):
    summary = k.groupby("报告技术路线", dropna=False).agg(
        关键词数=("关键词", "nunique"), 原始技术领域数=("技术领域", "nunique"), 平均核心程度得分=("核心程度得分", "mean")
    ).reset_index().rename(columns={"报告技术路线": "技术路线"})
    return reorder_route(summary)


def three_table_check(p, fy, f):
    return pd.DataFrame([
        {"检查项": "清洗后专利明细行数", "数值": len(p)},
        {"检查项": "重建主体—年份表专利数合计", "数值": int(fy["biomanufacturing_patent_count"].sum())},
        {"检查项": "重建主体汇总表专利数合计", "数值": int(f["biomanufacturing_patent_count"].sum())},
        {"检查项": "主体—年份表行数", "数值": len(fy)},
        {"检查项": "主体汇总表行数", "数值": len(f)},
    ])


def save_tables(p, fy, f, k):
    national_annual, national_total, raw_tech = nationwide_technology_tables(p)
    region_summary, region_annual = region_tables(p)
    compare_summary = region_summary[region_summary["省级地区"].isin(COMPARE_REGIONS)].copy()
    compare_summary["_order"] = compare_summary["省级地区"].map({r: i for i, r in enumerate(COMPARE_REGIONS)})
    compare_summary = compare_summary.sort_values("_order").drop(columns="_order")
    compare_annual = region_annual[region_annual["省级地区"].isin(COMPARE_REGIONS)].copy()
    sh_tech, sh_tech_annual = shanghai_technology_tables(p)
    region_tech = region_technology_comparison(p)
    ent_region, ent_conc, ent_annual, sh_ent_rank, sh_cases, other_cases = enterprise_tables(p, fy, f)
    sh_ent_tech, ent_tech_compare = enterprise_technology_tables(p)
    sh_res_rank, sh_res_tech, res_region, res_tech_compare, sh_subject_tech = research_tables(p, f)
    subject_share = region_subject_type_share(p)
    keyword_summary = keyword_tables(k)
    check = three_table_check(p, fy, f)

    tables = {
        "01_全国技术路线年度变化.csv": national_annual,
        "02_全国技术路线总体结构.csv": national_total,
        "03_全国原始技术方向结构.csv": raw_tech,
        "04_全国省级地区概览.csv": region_summary,
        "05_上海与重点省市总体比较.csv": compare_summary,
        "06_上海与重点省市年度比较.csv": compare_annual,
        "07_上海技术路线总体结构.csv": sh_tech,
        "08_上海技术路线年度变化.csv": sh_tech_annual,
        "09_重点省市技术路线比较.csv": region_tech,
        "10_重点省市企业总体比较.csv": ent_region[ent_region["省级地区"].isin(COMPARE_REGIONS)],
        "11_重点省市企业集中度比较.csv": ent_conc[ent_conc["省级地区"].isin(COMPARE_REGIONS)],
        "12_重点省市企业年度变化.csv": ent_annual[ent_annual["省级地区"].isin(COMPARE_REGIONS)],
        "13_上海企业排名.csv": sh_ent_rank.head(TOP_SHANGHAI_ENTERPRISES),
        "14_上海企业技术路线.csv": sh_ent_tech,
        "15_重点省市企业技术路线比较.csv": ent_tech_compare,
        "16_上海高校科研院所排名.csv": sh_res_rank.head(TOP_SHANGHAI_RESEARCH),
        "17_上海高校科研院所技术路线.csv": sh_res_tech,
        "18_重点省市高校科研院所总体比较.csv": res_region[res_region["省级地区"].isin(COMPARE_REGIONS)],
        "19_重点省市高校科研院所技术路线比较.csv": res_tech_compare,
        "20_上海企业高校科研院所技术结构对照.csv": sh_subject_tech,
        "21_重点省市企业高校科研院所专利占比.csv": subject_share,
        "22_上海代表性企业候选.csv": sh_cases,
        "23_其他省市代表性企业候选.csv": other_cases,
        "24_关键词词典技术路线概览.csv": keyword_summary,
        "25_三表一致性检查.csv": check,
    }
    for name, df in tables.items():
        df.to_csv(TABLE_DIR / name, index=False, encoding="utf-8-sig")
    return tables


def clean_axes(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.7, alpha=0.22)
    ax.set_axisbelow(True)


def save_fig(fig, filename):
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(tables):
    set_plot_style()

    annual = tables["01_全国技术路线年度变化.csv"]
    pivot = annual.pivot_table(index="year", columns="技术路线", values="专利数", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(11.2, 6.3))
    for route in REPORT_TECH_ROUTES:
        if route in pivot.columns:
            ax.plot(pivot.index, pivot[route], marker="o", linewidth=2.1, label=route, color=ROUTE_COLORS.get(route))
    ax.set_title("全国生物制造各技术路线发明授权专利数量变化")
    ax.set_xlabel("年份"); ax.set_ylabel("专利数（件）")
    ax.set_xticks(pivot.index, [str(int(y)) for y in pivot.index])
    ax.legend(frameon=False, ncol=2, loc="upper left")
    clean_axes(ax); save_fig(fig, "01_全国技术路线年度趋势.png")

    share = annual.pivot_table(index="year", columns="技术路线", values="当年全国专利占比", aggfunc="sum", fill_value=0)
    share = share.reindex(columns=[c for c in REPORT_TECH_ROUTES if c in share.columns])
    fig, ax = plt.subplots(figsize=(11.2, 6.3))
    ax.stackplot(share.index, *[share[c] for c in share.columns], labels=share.columns, colors=[ROUTE_COLORS[c] for c in share.columns], alpha=0.9)
    ax.set_title("全国生物制造专利技术结构变化"); ax.set_xlabel("年份"); ax.set_ylabel("当年专利占比")
    ax.set_xticks(share.index, [str(int(y)) for y in share.index]); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    clean_axes(ax); save_fig(fig, "02_全国技术结构年度变化.png")

    comp_year = tables["06_上海与重点省市年度比较.csv"]
    fig, ax = plt.subplots(figsize=(10.8, 6.1))
    for region in COMPARE_REGIONS:
        t = comp_year[comp_year["省级地区"].eq(region)].sort_values("year")
        ax.plot(t["year"], t["专利数"], marker="o", linewidth=2.1, label=region, color=REGION_COLORS.get(region))
    years = sorted(pd.to_numeric(comp_year["year"], errors="coerce").dropna().astype(int).unique())
    ax.set_xticks(years, [str(y) for y in years]); ax.set_title("上海与重点省市生物制造专利年度变化")
    ax.set_xlabel("年份"); ax.set_ylabel("专利数（件）"); ax.legend(frameon=False, ncol=3)
    clean_axes(ax); save_fig(fig, "03_重点省市专利年度变化.png")

    comp = tables["05_上海与重点省市总体比较.csv"]
    x = np.arange(len(comp)); width = 0.37
    fig, ax = plt.subplots(figsize=(10.6, 6.0))
    ax.bar(x-width/2, comp["专利数"], width, label="专利数", color="#496A8A")
    ax.bar(x+width/2, comp["创新主体数"], width, label="创新主体数", color="#B88B4A")
    ax.set_xticks(x, comp["省级地区"]); ax.set_title("上海与重点省市生物制造创新规模比较"); ax.set_ylabel("数量"); ax.legend(frameon=False)
    clean_axes(ax); save_fig(fig, "04_重点省市总体创新规模.png")

    sh_year = comp_year[comp_year["省级地区"].eq(SHANGHAI)].sort_values("year")
    x = np.arange(len(sh_year)); width = 0.38
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ax.bar(x-width/2, sh_year["专利数"], width, label="专利数", color="#B23A3A")
    ax.bar(x+width/2, sh_year["创新主体数"], width, label="创新主体数", color="#C99A4B")
    ax.set_xticks(x, sh_year["year"].astype(int).astype(str)); ax.set_title("上海生物制造创新规模年度变化"); ax.set_ylabel("数量"); ax.legend(frameon=False)
    clean_axes(ax); save_fig(fig, "05_上海创新规模年度变化.png")

    rank = tables["13_上海企业排名.csv"].head(15).sort_values("专利数")
    fig, ax = plt.subplots(figsize=(10.0, 7.0))
    ax.barh(rank["第一申请人"], rank["专利数"], color="#B23A3A")
    ax.set_title("上海生物制造企业发明授权专利数量排名"); ax.set_xlabel("专利数（件）")
    clean_axes(ax, "x"); save_fig(fig, "06_上海企业专利排名.png")

    ent = tables["10_重点省市企业总体比较.csv"].copy()
    ent["_order"] = ent["省级地区"].map({r:i for i,r in enumerate(COMPARE_REGIONS)}); ent = ent.sort_values("_order")
    x=np.arange(len(ent)); width=.37
    fig, ax=plt.subplots(figsize=(10.6,6.0))
    ax.bar(x-width/2, ent["企业专利数"], width, label="企业专利数", color="#B23A3A")
    ax.bar(x+width/2, ent["企业数"], width, label="企业数", color="#D2A04C")
    ax.set_xticks(x, ent["省级地区"]); ax.set_title("重点省市生物制造企业创新规模比较"); ax.set_ylabel("数量"); ax.legend(frameon=False)
    clean_axes(ax); save_fig(fig, "07_重点省市企业规模比较.png")

    conc = tables["11_重点省市企业集中度比较.csv"].copy()
    conc["_order"] = conc["省级地区"].map({r:i for i,r in enumerate(COMPARE_REGIONS)}); conc=conc.sort_values("_order")
    fig, ax=plt.subplots(figsize=(9.8,5.8))
    ax.bar(conc["省级地区"], conc["CR5"], color=[REGION_COLORS.get(x,"#777") for x in conc["省级地区"]])
    ax.set_title("重点省市生物制造企业专利CR5比较"); ax.set_ylabel("前五家企业专利占比"); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    clean_axes(ax); save_fig(fig, "08_重点省市企业CR5比较.png")

    sh_ent = tables["14_上海企业技术路线.csv"].sort_values("企业专利数")
    fig, ax=plt.subplots(figsize=(10.0,6.3))
    ax.barh(sh_ent["技术路线"], sh_ent["企业专利数"], color=[ROUTE_COLORS.get(x,"#777") for x in sh_ent["技术路线"]])
    ax.set_title("上海企业生物制造技术路线分布"); ax.set_xlabel("企业专利数（件）")
    clean_axes(ax,"x"); save_fig(fig, "09_上海企业技术路线.png")

    comp_tech=tables["15_重点省市企业技术路线比较.csv"]
    heat=comp_tech.pivot_table(index="省级地区",columns="技术路线",values="地区内专利占比",aggfunc="sum",fill_value=0)
    heat=heat.reindex(index=COMPARE_REGIONS,columns=[x for x in REPORT_TECH_ROUTES if x in heat.columns])
    fig, ax=plt.subplots(figsize=(12.0,5.7))
    im=ax.imshow(heat.values,aspect="auto",cmap="YlGnBu",vmin=0,vmax=max(0.01,float(heat.values.max())))
    ax.set_xticks(range(len(heat.columns)),heat.columns,rotation=30,ha="right"); ax.set_yticks(range(len(heat.index)),heat.index)
    ax.set_title("重点省市企业生物制造技术结构比较")
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            ax.text(j,i,f"{heat.iloc[i,j]:.0%}",ha="center",va="center",fontsize=8,color="black")
    fig.colorbar(im,ax=ax,fraction=.025,pad=.02,format=PercentFormatter(1.0))
    save_fig(fig,"10_重点省市企业技术结构热力图.png")

    res=tables["16_上海高校科研院所排名.csv"].head(15).sort_values("专利数")
    fig, ax=plt.subplots(figsize=(10.0,7.0))
    colors=["#3E648D" if x=="高校" else "#4C9474" for x in res["主体类型"]]
    ax.barh(res["第一申请人"],res["专利数"],color=colors)
    ax.set_title("上海高校和科研院所生物制造专利排名"); ax.set_xlabel("专利数（件）")
    clean_axes(ax,"x"); save_fig(fig,"11_上海高校科研院所专利排名.png")

    sh_compare=tables["20_上海企业高校科研院所技术结构对照.csv"]
    pivot2=sh_compare.pivot_table(index="技术路线",columns="主体类型",values="类型内专利占比",aggfunc="sum",fill_value=0).reindex(REPORT_TECH_ROUTES).fillna(0)
    fig, ax=plt.subplots(figsize=(11.0,6.2))
    x=np.arange(len(pivot2)); width=.25
    for idx,typ in enumerate(FOCUS_SUBJECT_TYPES):
        if typ in pivot2.columns:
            ax.bar(x+(idx-1)*width,pivot2[typ],width,label=typ,color=SUBJECT_COLORS[typ])
    ax.set_xticks(x,pivot2.index,rotation=25,ha="right"); ax.set_title("上海企业、高校与科研院所技术结构对照"); ax.set_ylabel("各主体类型内部专利占比")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0)); ax.legend(frameon=False)
    clean_axes(ax); save_fig(fig,"12_上海企业高校科研院所技术结构对照.png")

    sh_ta=tables["08_上海技术路线年度变化.csv"]
    pv=sh_ta.pivot_table(index="year",columns="技术路线",values="专利数",aggfunc="sum",fill_value=0)
    fig, ax=plt.subplots(figsize=(11.2,6.3))
    for route in REPORT_TECH_ROUTES:
        if route in pv.columns:
            ax.plot(pv.index,pv[route],marker="o",linewidth=2.0,label=route,color=ROUTE_COLORS[route])
    ax.set_title("上海生物制造技术路线年度变化"); ax.set_xlabel("年份"); ax.set_ylabel("专利数（件）")
    ax.set_xticks(pv.index,[str(int(y)) for y in pv.index]); ax.legend(frameon=False,ncol=2)
    clean_axes(ax); save_fig(fig,"13_上海技术路线年度变化.png")

    # 用户指定：第五节（四）展示各地区企业、高校、科研院所专利占比差异。
    subject=tables["21_重点省市企业高校科研院所专利占比.csv"]
    subject_pv=subject.pivot_table(index="省级地区",columns="主体类型",values="三类主体内部占比",aggfunc="sum",fill_value=0).reindex(COMPARE_REGIONS).fillna(0)
    fig, ax=plt.subplots(figsize=(10.8,6.1))
    left=np.zeros(len(subject_pv))
    for typ in FOCUS_SUBJECT_TYPES:
        values=subject_pv.get(typ,pd.Series(0,index=subject_pv.index)).to_numpy()
        ax.barh(subject_pv.index,values,left=left,label=typ,color=SUBJECT_COLORS[typ])
        for i,(l,v) in enumerate(zip(left,values)):
            if v>=0.045:
                ax.text(l+v/2,i,f"{v:.0%}",ha="center",va="center",fontsize=9,color="white" if typ=="企业" else "black")
        left+=values
    ax.set_xlim(0,1); ax.xaxis.set_major_formatter(PercentFormatter(1.0)); ax.set_xlabel("三类主体专利占比")
    ax.set_title("重点省市企业、高校与科研院所生物制造专利结构差异")
    ax.invert_yaxis()
    ax.legend(frameon=False,ncol=3,loc="upper center",bbox_to_anchor=(0.5,-0.12))
    clean_axes(ax,"x"); save_fig(fig,"14_重点省市企业高校科研院所专利占比.png")

    res_region=tables["18_重点省市高校科研院所总体比较.csv"].copy()
    res_region["_order"]=res_region["省级地区"].map({r:i for i,r in enumerate(COMPARE_REGIONS)}); res_region=res_region.sort_values("_order")
    x=np.arange(len(res_region)); width=.38
    fig, ax=plt.subplots(figsize=(10.6,6.0))
    ax.bar(x-width/2,res_region.get("高校专利数",0),width,label="高校专利数",color=SUBJECT_COLORS["高校"])
    ax.bar(x+width/2,res_region.get("科研院所专利数",0),width,label="科研院所专利数",color=SUBJECT_COLORS["科研院所"])
    ax.set_xticks(x,res_region["省级地区"]); ax.set_title("重点省市高校与科研院所生物制造专利规模比较"); ax.set_ylabel("专利数（件）"); ax.legend(frameon=False)
    clean_axes(ax); save_fig(fig,"15_重点省市高校科研院所规模比较.png")


def write_summary(tables):
    comp=tables["05_上海与重点省市总体比较.csv"]
    sh=comp[comp["省级地区"].eq(SHANGHAI)].iloc[0]
    sh_routes=tables["07_上海技术路线总体结构.csv"].sort_values("专利数",ascending=False)
    sh_ent=tables["10_重点省市企业总体比较.csv"]
    sh_ent=sh_ent[sh_ent["省级地区"].eq(SHANGHAI)].iloc[0]
    sh_res=tables["18_重点省市高校科研院所总体比较.csv"]
    sh_res=sh_res[sh_res["省级地区"].eq(SHANGHAI)].iloc[0]
    latest_year=int(tables["01_全国技术路线年度变化.csv"]["year"].max())
    prev_year=latest_year-1
    annual=tables["06_上海与重点省市年度比较.csv"]
    sh_latest=annual[(annual["省级地区"].eq(SHANGHAI)) & (annual["year"].eq(latest_year))]["专利数"].sum()
    sh_prev=annual[(annual["省级地区"].eq(SHANGHAI)) & (annual["year"].eq(prev_year))]["专利数"].sum()
    warning = sh_prev>0 and sh_latest < sh_prev*0.65
    top_routes="、".join(sh_routes.head(3)["技术路线"].tolist())
    lines=[
        "# 生物制造分析结果摘要", "",
        f"- 当前数据覆盖至{latest_year}年，上海共有{int(sh['专利数']):,}件生物制造相关发明授权专利，涉及{int(sh['创新主体数']):,}个创新主体。",
        f"- 上海在国内省级地区中的专利数量排名为第{int(sh['专利数全国排名'])}位，全国占比为{sh['全国专利占比']:.2%}。",
        f"- 上海企业主体{int(sh_ent['企业数']):,}家，企业专利{int(sh_ent['企业专利数']):,}件；高校和科研院所合计专利{int(sh_res['高校院所专利数']):,}件。",
        f"- 上海专利数量较多的技术路线为：{top_routes}。",
        f"- 最新年份完整性提示：{'最新年份数量明显低于上一年，可能存在授权滞后或数据更新不完整，不宜直接解释为趋势转弱。' if warning else '未触发机械式异常阈值，但仍需结合数据更新时间审慎解释最新年份。'}",
        "- 第五节省际比较应重点使用图14，展示上海、北京、安徽、江苏、浙江和广东的企业、高校、科研院所专利占比差异。图中占比以三类主体专利合计为分母，不包含医疗机构、个人和机关团体。",
        "- 被引证次数覆盖不足，不作为主要质量判断依据；biomanufacturing_score仅为识别得分，不代表技术先进性或商业价值。",
    ]
    (OUTPUT_DIR/"analysis_summary.md").write_text("\n".join(lines),encoding="utf-8")


def write_manifest(tables):
    rows=[]
    for name,df in tables.items(): rows.append({"类型":"表格","文件":f"outputs/tables/{name}","行数":len(df)})
    for path in sorted(FIGURE_DIR.glob("*.png")): rows.append({"类型":"图形","文件":f"outputs/figures/{path.name}","行数":""})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR/"result_manifest.csv",index=False,encoding="utf-8-sig")


def main():
    ensure_dirs(OUTPUT_DIR,TABLE_DIR,FIGURE_DIR,AUDIT_DIR)
    p,fy,f,k=read_clean_data()
    tables=save_tables(p,fy,f,k)
    make_figures(tables)
    write_summary(tables)
    write_manifest(tables)
    print(f"分析表格已生成：{TABLE_DIR}")
    print(f"图形已生成：{FIGURE_DIR}")


if __name__=="__main__":
    main()
