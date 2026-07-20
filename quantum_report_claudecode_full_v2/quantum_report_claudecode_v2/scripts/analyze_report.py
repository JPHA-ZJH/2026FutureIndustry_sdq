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
    CLEAN_PATENT_FILE, CLEAN_FIRM_YEAR_FILE, CLEAN_FIRM_FILE,
    OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, AUDIT_DIR,
    SHANGHAI, COMPARE_REGIONS, TECH_CATEGORIES,
    TOP_SHANGHAI_ENTERPRISES, TOP_SHANGHAI_RESEARCH,
    SHANGHAI_CASE_COUNT, OTHER_CASE_COUNT, MAX_OTHER_CASES_PER_PROVINCE,
)
from scripts.utils import ensure_dirs, concentration_metrics, set_plot_style, join_unique

COLORS = {
    "上海": "#B33A3A", "北京": "#365C8D", "安徽": "#3B8C6E",
    "江苏": "#D08C35", "浙江": "#8064A2", "广东": "#4C8DAE",
}
TECH_COLORS = {
    "量子计算": "#355F8D", "量子通信与安全": "#2E8B72", "量子传感": "#C9892B",
    "量子计算硬件": "#8B5E83", "量子基础概念": "#6E7781", "未分类": "#B8B8B8",
}


def read_clean_data():
    p = pd.read_csv(CLEAN_PATENT_FILE, low_memory=False)
    fy = pd.read_csv(CLEAN_FIRM_YEAR_FILE, low_memory=False)
    f = pd.read_csv(CLEAN_FIRM_FILE, low_memory=False)
    for df in (p, fy, f):
        if "year" in df.columns:
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return p, fy, f


def domestic_patents(p: pd.DataFrame) -> pd.DataFrame:
    return p[p["是否中国境内申请人"].eq(1)].copy()


def mapped_domestic_patents(p: pd.DataFrame) -> pd.DataFrame:
    d = domestic_patents(p)
    return d[d["省级地区"].notna() & d["省级地区"].ne("")].copy()


def reorder_tech(df: pd.DataFrame, col: str = "技术方向") -> pd.DataFrame:
    order = {v: i for i, v in enumerate(TECH_CATEGORIES)}
    out = df.copy()
    out["_order"] = out[col].map(order).fillna(99)
    return out.sort_values(["_order", col]).drop(columns="_order")


def nationwide_technology_tables(p: pd.DataFrame):
    d = domestic_patents(p)
    annual = (
        d.groupby(["year", "main_category"], dropna=False)
        .agg(
            专利数=("专利名称", "size"),
            创新主体数=("第一申请人", "nunique"),
            企业专利数=("主体类型", lambda x: x.eq("企业").sum()),
            企业数=("第一申请人", lambda x: d.loc[x.index].loc[d.loc[x.index, "主体类型"].eq("企业"), "第一申请人"].nunique()),
        )
        .reset_index().rename(columns={"main_category": "技术方向"})
    )
    annual["当年全国专利占比"] = annual["专利数"] / annual.groupby("year")["专利数"].transform("sum")
    annual = reorder_tech(annual)

    total = (
        d.groupby("main_category", dropna=False)
        .agg(
            专利数=("专利名称", "size"),
            创新主体数=("第一申请人", "nunique"),
            企业专利数=("主体类型", lambda x: x.eq("企业").sum()),
            企业数=("第一申请人", lambda x: d.loc[x.index].loc[d.loc[x.index, "主体类型"].eq("企业"), "第一申请人"].nunique()),
        )
        .reset_index().rename(columns={"main_category": "技术方向"})
    )
    total["专利占比"] = total["专利数"] / total["专利数"].sum()
    total = reorder_tech(total)
    return annual, total


def region_tables(p: pd.DataFrame):
    d = mapped_domestic_patents(p)
    summary = (
        d.groupby("省级地区")
        .agg(
            专利数=("专利名称", "size"),
            创新主体数=("第一申请人", "nunique"),
            企业专利数=("主体类型", lambda x: x.eq("企业").sum()),
            企业数=("第一申请人", lambda x: d.loc[x.index].loc[d.loc[x.index, "主体类型"].eq("企业"), "第一申请人"].nunique()),
            高校院所专利数=("主体类型", lambda x: x.isin(["高校", "科研院所"]).sum()),
            高校院所数=("第一申请人", lambda x: d.loc[x.index].loc[d.loc[x.index, "主体类型"].isin(["高校", "科研院所"]), "第一申请人"].nunique()),
        )
        .reset_index().sort_values(["专利数", "创新主体数"], ascending=False)
    )
    summary["全国专利占比"] = summary["专利数"] / summary["专利数"].sum()
    summary["专利数全国排名"] = summary["专利数"].rank(method="min", ascending=False).astype(int)
    summary["企业专利占比"] = summary["企业专利数"] / summary["专利数"].replace(0, np.nan)

    annual = (
        d.groupby(["省级地区", "year"])
        .agg(
            专利数=("专利名称", "size"),
            创新主体数=("第一申请人", "nunique"),
            企业专利数=("主体类型", lambda x: x.eq("企业").sum()),
            企业数=("第一申请人", lambda x: d.loc[x.index].loc[d.loc[x.index, "主体类型"].eq("企业"), "第一申请人"].nunique()),
        )
        .reset_index()
    )
    annual["当年全国专利占比"] = annual["专利数"] / annual.groupby("year")["专利数"].transform("sum")
    annual["当年专利排名"] = annual.groupby("year")["专利数"].rank(method="min", ascending=False).astype(int)
    return summary, annual


def region_technology_comparison(p: pd.DataFrame, applicant_types: list[str] | None = None):
    d = mapped_domestic_patents(p)
    d = d[d["省级地区"].isin(COMPARE_REGIONS)].copy()
    if applicant_types:
        d = d[d["主体类型"].isin(applicant_types)]
    out = (
        d.groupby(["省级地区", "main_category"])
        .agg(专利数=("专利名称", "size"), 创新主体数=("第一申请人", "nunique"))
        .reset_index().rename(columns={"main_category": "技术方向"})
    )
    out["地区内专利占比"] = out["专利数"] / out.groupby("省级地区")["专利数"].transform("sum")
    return reorder_tech(out)


def shanghai_technology_tables(p: pd.DataFrame):
    d = mapped_domestic_patents(p)
    sh = d[d["省级地区"].eq(SHANGHAI)].copy()
    total = (
        sh.groupby("main_category")
        .agg(专利数=("专利名称", "size"), 创新主体数=("第一申请人", "nunique"), 企业专利数=("主体类型", lambda x: x.eq("企业").sum()))
        .reset_index().rename(columns={"main_category": "技术方向"})
    )
    total["专利占比"] = total["专利数"] / total["专利数"].sum()
    annual = (
        sh.groupby(["year", "main_category"])
        .agg(专利数=("专利名称", "size"), 创新主体数=("第一申请人", "nunique"))
        .reset_index().rename(columns={"main_category": "技术方向"})
    )
    annual["当年上海专利占比"] = annual["专利数"] / annual.groupby("year")["专利数"].transform("sum")
    return reorder_tech(total), reorder_tech(annual)


def enterprise_tables(p: pd.DataFrame, fy: pd.DataFrame, f: pd.DataFrame):
    ent_f = f[(f["主体类型"].eq("企业")) & f["省级地区"].notna()].copy()
    ent_fy = fy[(fy["主体类型"].eq("企业")) & fy["省级地区"].notna()].copy()

    region = (
        ent_f.groupby("省级地区")
        .agg(
            企业数=("第一申请人", "nunique"),
            企业专利数=("quantum_patent_count", "sum"),
            企业平均专利数=("quantum_patent_count", "mean"),
            多年活跃企业数=("active_years", lambda x: (x >= 2).sum()),
            三年及以上活跃企业数=("active_years", lambda x: (x >= 3).sum()),
        )
        .reset_index()
    )
    region["多年活跃企业占比"] = region["多年活跃企业数"] / region["企业数"].replace(0, np.nan)

    concentration_rows = []
    for region_name, g in ent_f.groupby("省级地区"):
        row = {"省级地区": region_name, "企业数": g["第一申请人"].nunique(), "企业专利数": int(g["quantum_patent_count"].sum())}
        row.update(concentration_metrics(g.groupby("第一申请人")["quantum_patent_count"].sum()))
        concentration_rows.append(row)
    concentration = pd.DataFrame(concentration_rows)

    annual = (
        ent_fy.groupby(["省级地区", "year"])
        .agg(活跃企业数=("第一申请人", "nunique"), 企业专利数=("quantum_patent_count", "sum"))
        .reset_index()
    )
    entrants = (
        ent_f.groupby(["省级地区", "first_year"]).size().rename("新进入企业数").reset_index().rename(columns={"first_year": "year"})
    )
    annual = annual.merge(entrants, on=["省级地区", "year"], how="left").fillna({"新进入企业数": 0})
    annual["新进入企业占比"] = annual["新进入企业数"] / annual["活跃企业数"].replace(0, np.nan)

    sh_rank = ent_f[ent_f["省级地区"].eq(SHANGHAI)].copy()
    sh_rank = sh_rank.sort_values(["quantum_patent_count", "active_years", "last_year"], ascending=False)
    sh_rank = sh_rank.rename(columns={
        "quantum_patent_count": "专利数", "active_years": "活跃年份数",
        "first_year": "最早年份", "last_year": "最晚年份",
        "main_quantum_category": "主要技术方向", "main_categories": "技术方向覆盖",
        "第一申请人城市标准化": "城市",
    })
    sh_rank = sh_rank[["第一申请人", "城市", "专利数", "活跃年份数", "最早年份", "最晚年份", "主要技术方向", "技术方向覆盖"]]

    sh_detail = ent_f[ent_f["省级地区"].eq(SHANGHAI)].copy()
    sh_detail = sh_detail.rename(columns={
        "quantum_patent_count": "专利数", "active_years": "活跃年份数", "first_year": "最早年份", "last_year": "最晚年份",
        "main_quantum_category": "主要技术方向", "main_categories": "技术方向覆盖", "第一申请人城市标准化": "城市",
    })
    sh_cases = sh_detail.sort_values(["专利数", "活跃年份数", "最晚年份"], ascending=False).head(SHANGHAI_CASE_COUNT)
    sh_cases = sh_cases[["第一申请人", "城市", "专利数", "活跃年份数", "最早年份", "最晚年份", "主要技术方向", "技术方向覆盖"]]

    # 全国其他省份企业候选：按专利积累排序，同时避免单一省份包揽全部案例。
    candidates = ent_f[~ent_f["省级地区"].eq(SHANGHAI)].copy()
    candidates = candidates.sort_values(["quantum_patent_count", "active_years", "last_year"], ascending=False)
    selected = []
    province_counter: dict[str, int] = {}
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
        other_cases = other_cases.rename(columns={
            "quantum_patent_count": "专利数", "active_years": "活跃年份数", "first_year": "最早年份", "last_year": "最晚年份",
            "main_quantum_category": "主要技术方向", "main_categories": "技术方向覆盖", "第一申请人城市标准化": "城市",
        })
        other_cases = other_cases[["第一申请人", "省级地区", "城市", "专利数", "活跃年份数", "最早年份", "最晚年份", "主要技术方向", "技术方向覆盖"]]

    return region, concentration, annual, sh_rank, sh_cases, other_cases


def enterprise_technology_tables(p: pd.DataFrame):
    d = mapped_domestic_patents(p)
    ent = d[d["主体类型"].eq("企业")].copy()
    sh = ent[ent["省级地区"].eq(SHANGHAI)]
    sh_table = (
        sh.groupby("main_category")
        .agg(企业专利数=("专利名称", "size"), 企业数=("第一申请人", "nunique"))
        .reset_index().rename(columns={"main_category": "技术方向"})
    )
    sh_table["企业专利占比"] = sh_table["企业专利数"] / sh_table["企业专利数"].sum()
    sh_table = reorder_tech(sh_table)
    compare = region_technology_comparison(p, ["企业"])
    return sh_table, compare


def research_tables(p: pd.DataFrame, f: pd.DataFrame):
    research_types = ["高校", "科研院所"]
    d = mapped_domestic_patents(p)
    res_p = d[d["主体类型"].isin(research_types)].copy()
    res_f = f[f["主体类型"].isin(research_types) & f["省级地区"].notna()].copy()

    sh_rank = res_f[res_f["省级地区"].eq(SHANGHAI)].copy()
    sh_rank = sh_rank.rename(columns={
        "quantum_patent_count": "专利数", "active_years": "活跃年份数", "first_year": "最早年份", "last_year": "最晚年份",
        "main_quantum_category": "主要技术方向", "main_categories": "技术方向覆盖", "第一申请人城市标准化": "城市",
    }).sort_values(["专利数", "活跃年份数"], ascending=False)
    sh_rank = sh_rank[["第一申请人", "主体类型", "城市", "专利数", "活跃年份数", "最早年份", "最晚年份", "主要技术方向", "技术方向覆盖"]]

    sh_res_p = res_p[res_p["省级地区"].eq(SHANGHAI)]
    sh_tech = (
        sh_res_p.groupby(["主体类型", "main_category"])
        .agg(专利数=("专利名称", "size"), 主体数=("第一申请人", "nunique"))
        .reset_index().rename(columns={"main_category": "技术方向"})
    )
    sh_tech["类型内专利占比"] = sh_tech["专利数"] / sh_tech.groupby("主体类型")["专利数"].transform("sum")

    region = (
        res_p.groupby("省级地区")
        .agg(高校院所专利数=("专利名称", "size"), 高校院所数=("第一申请人", "nunique"), 高校专利数=("主体类型", lambda x: x.eq("高校").sum()), 科研院所专利数=("主体类型", lambda x: x.eq("科研院所").sum()))
        .reset_index()
    )
    compare_tech = region_technology_comparison(p, research_types)

    # 上海企业与科研主体的技术结构对照。
    sh_all = d[d["省级地区"].eq(SHANGHAI) & d["主体类型"].isin(["企业", "高校", "科研院所"])].copy()
    sh_all["主体组"] = np.where(sh_all["主体类型"].eq("企业"), "企业", "高校院所")
    sh_compare = (
        sh_all.groupby(["主体组", "main_category"])
        .agg(专利数=("专利名称", "size"), 主体数=("第一申请人", "nunique"))
        .reset_index().rename(columns={"main_category": "技术方向"})
    )
    sh_compare["组内专利占比"] = sh_compare["专利数"] / sh_compare.groupby("主体组")["专利数"].transform("sum")
    return sh_rank, reorder_tech(sh_tech), region, reorder_tech(compare_tech), reorder_tech(sh_compare)


def three_table_check(p: pd.DataFrame, fy: pd.DataFrame, f: pd.DataFrame):
    return pd.DataFrame([
        {"检查项": "清洗后专利明细行数", "数值": len(p)},
        {"检查项": "重建企业—年份表专利数合计", "数值": int(fy["quantum_patent_count"].sum())},
        {"检查项": "重建主体汇总表专利数合计", "数值": int(f["quantum_patent_count"].sum())},
        {"检查项": "企业—年份表行数", "数值": len(fy)},
        {"检查项": "主体汇总表行数", "数值": len(f)},
    ])


def save_tables(p: pd.DataFrame, fy: pd.DataFrame, f: pd.DataFrame):
    national_annual, national_total = nationwide_technology_tables(p)
    region_summary, region_annual = region_tables(p)
    compare_summary = region_summary[region_summary["省级地区"].isin(COMPARE_REGIONS)].copy()
    compare_summary["_order"] = compare_summary["省级地区"].map({r: i for i, r in enumerate(COMPARE_REGIONS)})
    compare_summary = compare_summary.sort_values("_order").drop(columns="_order")
    compare_annual = region_annual[region_annual["省级地区"].isin(COMPARE_REGIONS)].copy()
    sh_tech, sh_tech_annual = shanghai_technology_tables(p)
    region_tech = region_technology_comparison(p)
    ent_region, ent_conc, ent_annual, sh_ent_rank, sh_cases, other_cases = enterprise_tables(p, fy, f)
    sh_ent_tech, ent_tech_compare = enterprise_technology_tables(p)
    sh_res_rank, sh_res_tech, res_region, res_tech_compare, sh_ent_res_compare = research_tables(p, f)
    check = three_table_check(p, fy, f)

    tables = {
        "01_全国技术方向年度变化.csv": national_annual,
        "02_全国技术方向总体结构.csv": national_total,
        "03_全国省级地区概览.csv": region_summary,
        "04_上海与重点省市总体比较.csv": compare_summary,
        "05_上海与重点省市年度比较.csv": compare_annual,
        "06_上海技术方向总体结构.csv": sh_tech,
        "07_上海技术方向年度变化.csv": sh_tech_annual,
        "08_重点省市技术方向比较.csv": region_tech,
        "09_重点省市企业总体比较.csv": ent_region[ent_region["省级地区"].isin(COMPARE_REGIONS)],
        "10_重点省市企业集中度比较.csv": ent_conc[ent_conc["省级地区"].isin(COMPARE_REGIONS)],
        "11_重点省市企业年度变化.csv": ent_annual[ent_annual["省级地区"].isin(COMPARE_REGIONS)],
        "12_上海企业排名.csv": sh_ent_rank.head(TOP_SHANGHAI_ENTERPRISES),
        "13_上海企业技术方向.csv": sh_ent_tech,
        "14_重点省市企业技术方向比较.csv": ent_tech_compare,
        "15_上海高校院所排名.csv": sh_res_rank.head(TOP_SHANGHAI_RESEARCH),
        "16_上海高校院所技术方向.csv": sh_res_tech,
        "17_重点省市高校院所总体比较.csv": res_region[res_region["省级地区"].isin(COMPARE_REGIONS)],
        "18_重点省市高校院所技术方向比较.csv": res_tech_compare,
        "19_上海企业与高校院所技术结构对照.csv": sh_ent_res_compare,
        "20_上海代表性企业候选.csv": sh_cases,
        "21_其他省市代表性企业候选.csv": other_cases,
        "22_三表一致性检查.csv": check,
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


def make_figures(tables: dict):
    set_plot_style()

    # 1 全国技术路线年度趋势
    annual = tables["01_全国技术方向年度变化.csv"]
    pivot = annual.pivot_table(index="year", columns="技术方向", values="专利数", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for tech in TECH_CATEGORIES:
        if tech in pivot.columns:
            ax.plot(pivot.index, pivot[tech], marker="o", linewidth=2.2, label=tech, color=TECH_COLORS.get(tech))
    ax.set_title("全国量子科技各技术方向发明授权专利数量变化")
    ax.set_xlabel("年份")
    ax.set_ylabel("专利数（件）")
    ax.set_xticks(pivot.index, [str(int(y)) for y in pivot.index])
    ax.legend(frameon=False, ncol=3, loc="upper left")
    clean_axes(ax)
    save_fig(fig, "01_全国技术方向年度趋势.png")

    # 2 全国技术结构变化
    share = annual.pivot_table(index="year", columns="技术方向", values="当年全国专利占比", aggfunc="sum", fill_value=0)
    share = share.reindex(columns=[c for c in TECH_CATEGORIES if c in share.columns])
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.stackplot(share.index, *[share[c] for c in share.columns], labels=share.columns,
                 colors=[TECH_COLORS.get(c) for c in share.columns], alpha=0.9)
    ax.set_title("全国量子科技专利技术结构变化")
    ax.set_xlabel("年份")
    ax.set_ylabel("当年专利占比")
    ax.set_xticks(share.index, [str(int(y)) for y in share.index])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    clean_axes(ax)
    save_fig(fig, "02_全国技术结构年度变化.png")

    # 3 重点省市年度专利变化
    comp_year = tables["05_上海与重点省市年度比较.csv"]
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for region in COMPARE_REGIONS:
        t = comp_year[comp_year["省级地区"].eq(region)].sort_values("year")
        ax.plot(t["year"], t["专利数"], marker="o", linewidth=2.2, label=region, color=COLORS.get(region))
    ax.set_title("上海与重点省市量子信息专利年度变化")
    ax.set_xlabel("年份")
    ax.set_ylabel("专利数（件）")
    years = sorted(pd.to_numeric(comp_year["year"], errors="coerce").dropna().astype(int).unique())
    ax.set_xticks(years, [str(y) for y in years])
    ax.legend(frameon=False, ncol=3)
    clean_axes(ax)
    save_fig(fig, "03_重点省市专利年度变化.png")

    # 4 重点省市总体创新规模
    comp = tables["04_上海与重点省市总体比较.csv"]
    x = np.arange(len(comp))
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    width = 0.37
    ax.bar(x - width/2, comp["专利数"], width, label="专利数", color="#496A8A")
    ax.bar(x + width/2, comp["创新主体数"], width, label="创新主体数", color="#B88B4A")
    ax.set_xticks(x, comp["省级地区"])
    ax.set_title("上海与重点省市量子信息创新规模比较")
    ax.set_ylabel("数量")
    ax.legend(frameon=False)
    clean_axes(ax)
    save_fig(fig, "04_重点省市总体创新规模.png")

    # 5 上海年度专利和主体
    sh_year = comp_year[comp_year["省级地区"].eq(SHANGHAI)].sort_values("year")
    x = np.arange(len(sh_year))
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    width = 0.38
    ax.bar(x - width/2, sh_year["专利数"], width, label="专利数", color="#B33A3A")
    ax.bar(x + width/2, sh_year["创新主体数"], width, label="创新主体数", color="#C99A4B")
    ax.set_xticks(x, sh_year["year"].astype(int))
    ax.set_title("上海量子信息专利与创新主体年度变化")
    ax.set_ylabel("数量")
    ax.legend(frameon=False)
    clean_axes(ax)
    save_fig(fig, "05_上海创新规模年度变化.png")

    # 6 上海企业排名
    rank = tables["12_上海企业排名.csv"].head(15).sort_values("专利数")
    fig, ax = plt.subplots(figsize=(10.5, max(5.8, len(rank)*0.42)))
    bars = ax.barh(rank["第一申请人"], rank["专利数"], color="#B33A3A")
    ax.set_title("上海量子信息企业发明授权专利数量排名")
    ax.set_xlabel("专利数（件）")
    clean_axes(ax, "x")
    for bar, value in zip(bars, rank["专利数"]):
        ax.text(value, bar.get_y()+bar.get_height()/2, f" {int(value)}", va="center", fontsize=8.5)
    save_fig(fig, "06_上海企业专利排名.png")

    # 7 重点省市企业规模
    ent = tables["09_重点省市企业总体比较.csv"].copy()
    ent["_order"] = ent["省级地区"].map({r:i for i,r in enumerate(COMPARE_REGIONS)})
    ent = ent.sort_values("_order")
    x = np.arange(len(ent))
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    width = 0.37
    ax.bar(x-width/2, ent["企业专利数"], width, label="企业专利数", color="#446B8C")
    ax.bar(x+width/2, ent["企业数"], width, label="企业数", color="#6E9D83")
    ax.set_xticks(x, ent["省级地区"])
    ax.set_title("上海与重点省市量子信息企业创新规模比较")
    ax.set_ylabel("数量")
    ax.legend(frameon=False)
    clean_axes(ax)
    save_fig(fig, "07_重点省市企业规模比较.png")

    # 8 企业集中度
    conc = tables["10_重点省市企业集中度比较.csv"].copy()
    conc["_order"] = conc["省级地区"].map({r:i for i,r in enumerate(COMPARE_REGIONS)})
    conc = conc.sort_values("_order")
    fig, ax = plt.subplots(figsize=(9.5, 5.7))
    bars = ax.bar(conc["省级地区"], conc["CR5"], color=[COLORS.get(r, "#777") for r in conc["省级地区"]])
    ax.set_title("重点省市量子信息企业前五位专利集中度")
    ax.set_ylabel("CR5")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    clean_axes(ax)
    for bar, value in zip(bars, conc["CR5"]):
        ax.text(bar.get_x()+bar.get_width()/2, value, f"{value:.1%}", ha="center", va="bottom", fontsize=8.5)
    save_fig(fig, "08_重点省市企业CR5比较.png")

    # 9 上海企业技术方向
    sh_ent_tech = tables["13_上海企业技术方向.csv"].sort_values("企业专利数")
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    bars = ax.barh(sh_ent_tech["技术方向"], sh_ent_tech["企业专利数"], color=[TECH_COLORS.get(t, "#777") for t in sh_ent_tech["技术方向"]])
    ax.set_title("上海企业量子信息技术方向分布")
    ax.set_xlabel("企业专利数（件）")
    clean_axes(ax, "x")
    for bar, value in zip(bars, sh_ent_tech["企业专利数"]):
        ax.text(value, bar.get_y()+bar.get_height()/2, f" {int(value)}", va="center", fontsize=8.5)
    save_fig(fig, "09_上海企业技术方向.png")

    # 10 重点省市企业技术热力图
    tech = tables["14_重点省市企业技术方向比较.csv"]
    pivot = tech.pivot_table(index="省级地区", columns="技术方向", values="地区内专利占比", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(index=COMPARE_REGIONS, columns=[c for c in TECH_CATEGORIES if c in pivot.columns]).fillna(0)
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlGnBu", vmin=0)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("重点省市企业量子信息技术结构比较")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if value > max(0.25, pivot.values.max()*0.58) else "black")
    fig.colorbar(im, ax=ax, fraction=0.028, pad=0.025, label="地区内企业专利占比")
    save_fig(fig, "10_重点省市企业技术结构热力图.png")

    # 11 上海高校院所排名
    research_rank = tables["15_上海高校院所排名.csv"].head(15).sort_values("专利数")
    fig, ax = plt.subplots(figsize=(10.5, max(5.8, len(research_rank)*0.42)))
    colors = research_rank["主体类型"].map({"高校":"#4D7398", "科研院所":"#5A8D72"}).fillna("#777")
    bars = ax.barh(research_rank["第一申请人"], research_rank["专利数"], color=colors)
    ax.set_title("上海高校和科研院所量子信息专利数量排名")
    ax.set_xlabel("专利数（件）")
    clean_axes(ax, "x")
    for bar, value in zip(bars, research_rank["专利数"]):
        ax.text(value, bar.get_y()+bar.get_height()/2, f" {int(value)}", va="center", fontsize=8.5)
    save_fig(fig, "11_上海高校院所专利排名.png")

    # 12 上海企业与高校院所技术结构
    er = tables["19_上海企业与高校院所技术结构对照.csv"]
    pivot = er.pivot_table(index="技术方向", columns="主体组", values="组内专利占比", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex([c for c in TECH_CATEGORIES if c in pivot.index])
    x = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    width = 0.38
    ax.bar(x-width/2, pivot.get("企业", pd.Series(0, index=pivot.index)), width, label="企业", color="#B33A3A")
    ax.bar(x+width/2, pivot.get("高校院所", pd.Series(0, index=pivot.index)), width, label="高校院所", color="#4D7398")
    ax.set_xticks(x, pivot.index, rotation=18)
    ax.set_title("上海企业与高校院所量子信息技术结构对照")
    ax.set_ylabel("组内专利占比")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(frameon=False)
    clean_axes(ax)
    save_fig(fig, "12_上海企业与高校院所技术结构对照.png")

    # 13 上海技术方向年度变化
    sh_annual = tables["07_上海技术方向年度变化.csv"]
    pivot = sh_annual.pivot_table(index="year", columns="技术方向", values="专利数", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for tech in TECH_CATEGORIES:
        if tech in pivot.columns:
            ax.plot(pivot.index, pivot[tech], marker="o", linewidth=2.1, label=tech, color=TECH_COLORS.get(tech))
    ax.set_title("上海量子信息各技术方向专利数量变化")
    ax.set_xlabel("年份")
    ax.set_ylabel("专利数（件）")
    ax.set_xticks(pivot.index, [str(int(y)) for y in pivot.index])
    ax.legend(frameon=False, ncol=3)
    clean_axes(ax)
    save_fig(fig, "13_上海技术方向年度变化.png")


def write_analysis_summary(p: pd.DataFrame, tables: dict):
    years = sorted(pd.to_numeric(p["year"], errors="coerce").dropna().astype(int).unique())
    national = tables["02_全国技术方向总体结构.csv"]
    compare = tables["04_上海与重点省市总体比较.csv"]
    sh_ent = tables["12_上海企业排名.csv"].head(10)
    sh_research = tables["15_上海高校院所排名.csv"].head(10)
    sh_cases = tables["20_上海代表性企业候选.csv"]
    other_cases = tables["21_其他省市代表性企业候选.csv"]

    lines = ["# 供咨询报告写作使用的结构化分析摘要", ""]
    lines += ["## 数据口径", "", f"数据覆盖{years[0]}—{years[-1]}年，使用发明授权专利，以第一申请人作为主要创新主体。省市和主体类型均使用清洗后的字段。", ""]
    lines += ["## 全国技术方向总体结构", "", national.to_markdown(index=False), ""]
    lines += ["## 上海与重点省市总体比较", "", compare.to_markdown(index=False), ""]
    lines += ["## 上海企业排名（前10位）", "", sh_ent.to_markdown(index=False), ""]
    lines += ["## 上海高校院所排名（前10位）", "", sh_research.to_markdown(index=False), ""]
    lines += ["## 上海代表性企业候选", "", sh_cases.to_markdown(index=False), ""]
    lines += ["## 其他省市代表性企业候选", "", other_cases.to_markdown(index=False), ""]
    lines += [
        "## 强制写作边界", "",
        "- 技术内部分析只围绕五个技术方向，不使用相关性等级，不分析产业链环节。",
        "- 企业案例表是基于专利积累形成的候选，不等同于政府认定的龙头企业。企业产品、营收、融资和市场地位必须另查权威来源。",
        "- 最新年份专利授权量可能受到授权滞后和数据库更新进度影响，不应仅凭最后一年波动判断产业景气变化。",
        "- 报告必须严格按照docs/report_outline.md的章节和小节顺序撰写。",
    ]
    (OUTPUT_DIR / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_result_manifest(tables: dict):
    rows = []
    for name, df in tables.items():
        rows.append({"文件": name, "行数": len(df), "主要用途": "详见docs/indicator_dictionary.md"})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "result_manifest.csv", index=False, encoding="utf-8-sig")


def main():
    ensure_dirs(OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, AUDIT_DIR)
    p, fy, f = read_clean_data()
    tables = save_tables(p, fy, f)
    make_figures(tables)
    write_analysis_summary(p, tables)
    write_result_manifest(tables)
    print("统计分析与绘图完成。")
    print(f"表格：{TABLE_DIR}")
    print(f"图形：{FIGURE_DIR}")
    print(f"分析摘要：{OUTPUT_DIR / 'analysis_summary.md'}")


if __name__ == "__main__":
    main()
