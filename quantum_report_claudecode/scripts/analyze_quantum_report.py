from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    PATENT_FILE, FIRM_YEAR_FILE, FIRM_FILE, CITY_PROVINCE_FILE,
    OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, AUDIT_DIR,
    SHANGHAI, COMPARE_REGIONS, STRICT_CORE_LABELS, CORE_LABELS,
    CITATION_VALID_COVERAGE, TOP_ENTITY_N, TOP_OTHER_ENTERPRISE_N,
)
from scripts.utils import (
    ensure_dirs, set_plot_style, add_applicant_classification, add_province,
    concentration_metrics, join_unique,
)

PALETTE = ["#235789", "#F1A208", "#4F8A8B", "#C1292E", "#6B5B95", "#5B8C5A"]


def read_data():
    patents = pd.read_csv(PATENT_FILE, low_memory=False)
    firm_year = pd.read_csv(FIRM_YEAR_FILE, low_memory=False)
    firms = pd.read_csv(FIRM_FILE, low_memory=False)
    mapping = pd.read_csv(CITY_PROVINCE_FILE)
    return patents, firm_year, firms, mapping


def prepare_patents(patents: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    p = patents.copy()
    p["year"] = pd.to_numeric(p["year"], errors="coerce").astype("Int64")
    p["被引证次数_num"] = pd.to_numeric(p.get("被引证次数"), errors="coerce")
    p = add_applicant_classification(p)
    p = add_province(p, mapping)
    p["是否严格核心"] = p["relevance"].isin(STRICT_CORE_LABELS).astype(int)
    p["是否核心口径"] = p["relevance"].isin(CORE_LABELS).astype(int)
    p["是否PNT候选"] = p["relevance"].eq("量子计量/PNT候选").astype(int)
    return p


def build_audit(p: pd.DataFrame, fy: pd.DataFrame, f: pd.DataFrame) -> dict:
    years = sorted(p["year"].dropna().astype(int).unique().tolist())
    citation_coverage = float(p["被引证次数_num"].notna().mean())
    audit = {
        "patent_rows": int(len(p)),
        "firm_year_rows": int(len(fy)),
        "firm_rows": int(len(f)),
        "years": years,
        "year_count": len(years),
        "single_year_test": len(years) < 2,
        "unique_applicants": int(p["第一申请人"].nunique()),
        "unmapped_city_rows": int((p["省级地区"] == "待映射").sum()),
        "unmapped_cities": sorted(p.loc[p["省级地区"] == "待映射", "第一申请人城市"].dropna().unique().tolist()),
        "ambiguous_type_rows": int(p["主体类型需复核"].sum()),
        "ambiguous_type_share": float(p["主体类型需复核"].mean()),
        "citation_non_null": int(p["被引证次数_num"].notna().sum()),
        "citation_coverage": citation_coverage,
        "citation_metrics_allowed": citation_coverage >= CITATION_VALID_COVERAGE,
        "potential_duplicate_rows_basic_key": int(
            p.duplicated(["专利名称", "第一申请人", "year", "IPC主分类号"], keep=False).sum()
        ),
        "all_region_values": sorted(p["第一申请人地区"].dropna().astype(str).unique().tolist()),
    }
    return audit


def region_summary(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, g in p.groupby("省级地区", dropna=False):
        clear_enterprises = g[g["主体大类"] == "企业"]
        rows.append({
            "省级地区": region,
            "量子相关专利数": len(g),
            "创新主体数": g["第一申请人"].nunique(),
            "明确企业数": clear_enterprises["第一申请人"].nunique(),
            "企业专利数": len(clear_enterprises),
            "企业专利占比": len(clear_enterprises) / len(g) if len(g) else np.nan,
            "严格核心专利数": int(g["是否严格核心"].sum()),
            "严格核心专利占比": g["是否严格核心"].mean(),
            "核心口径专利数": int(g["是否核心口径"].sum()),
            "PNT候选专利数": int(g["是否PNT候选"].sum()),
            "高相关专利占比": g["relevance"].eq("高相关").mean(),
            "量子识别得分均值": pd.to_numeric(g["quantum_score"], errors="coerce").mean(),
            "被引字段覆盖率": g["被引证次数_num"].notna().mean(),
        })
    return pd.DataFrame(rows).sort_values(["量子相关专利数", "创新主体数"], ascending=False)


def yearly_summary(p: pd.DataFrame) -> pd.DataFrame:
    return (
        p.groupby(["省级地区", "year"], dropna=False)
         .agg(
            量子相关专利数=("专利名称", "size"),
            创新主体数=("第一申请人", "nunique"),
            明确企业数=("第一申请人", lambda x: p.loc[x.index].loc[p.loc[x.index, "主体大类"].eq("企业"), "第一申请人"].nunique()),
            企业专利数=("主体大类", lambda x: x.eq("企业").sum()),
            严格核心专利数=("是否严格核心", "sum"),
            PNT候选专利数=("是否PNT候选", "sum"),
         )
         .reset_index()
    )


def technology_table(p: pd.DataFrame, field: str, region: str | None = None) -> pd.DataFrame:
    g = p if region is None else p[p["省级地区"] == region]
    out = (
        g.groupby(field, dropna=False)
         .agg(
            专利数=("专利名称", "size"),
            创新主体数=("第一申请人", "nunique"),
            明确企业数=("第一申请人", lambda x: g.loc[x.index].loc[g.loc[x.index, "主体大类"].eq("企业"), "第一申请人"].nunique()),
            企业专利数=("主体大类", lambda x: x.eq("企业").sum()),
            高相关专利数=("relevance", lambda x: x.eq("高相关").sum()),
            严格核心专利数=("是否严格核心", "sum"),
            PNT候选专利数=("是否PNT候选", "sum"),
            量子识别得分均值=("quantum_score", "mean"),
         )
         .reset_index()
         .rename(columns={field: "类别"})
         .sort_values("专利数", ascending=False)
    )
    total = out["专利数"].sum()
    out["专利占比"] = out["专利数"] / total if total else np.nan
    return out


def comparison_technology(p: pd.DataFrame, field: str) -> pd.DataFrame:
    g = p[p["省级地区"].isin(COMPARE_REGIONS)].copy()
    counts = pd.crosstab(g["省级地区"], g[field])
    counts = counts.reindex(COMPARE_REGIONS).fillna(0).astype(int)
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    rows = []
    for region in counts.index:
        for category in counts.columns:
            rows.append({
                "省级地区": region,
                "类别": category,
                "专利数": int(counts.loc[region, category]),
                "地区内占比": shares.loc[region, category],
            })
    return pd.DataFrame(rows)


def entity_ranking(p: pd.DataFrame, region: str, applicant_class: str | None = None) -> pd.DataFrame:
    g = p[p["省级地区"] == region].copy()
    if applicant_class:
        g = g[g["主体大类"] == applicant_class]
    if g.empty:
        return pd.DataFrame()
    result = (
        g.groupby(["第一申请人", "主体大类", "第一申请人城市"], dropna=False)
         .agg(
            专利数=("专利名称", "size"),
            活跃年份数=("year", "nunique"),
            最早年份=("year", "min"),
            最晚年份=("year", "max"),
            高相关专利数=("relevance", lambda x: x.eq("高相关").sum()),
            中相关专利数=("relevance", lambda x: x.eq("中相关").sum()),
            严格核心专利数=("是否严格核心", "sum"),
            PNT候选专利数=("是否PNT候选", "sum"),
            量子识别得分均值=("quantum_score", "mean"),
            主要技术方向=("main_category", lambda x: x.value_counts().index[0] if len(x) else ""),
            技术方向覆盖=("main_category", join_unique),
            产业链环节覆盖=("chain_position", join_unique),
            被引字段覆盖率=("被引证次数_num", lambda x: x.notna().mean()),
            被引次数合计=("被引证次数_num", "sum"),
         )
         .reset_index()
         .sort_values(["专利数", "高相关专利数", "量子识别得分均值"], ascending=False)
    )
    return result


def other_top_enterprises(p: pd.DataFrame, n: int) -> pd.DataFrame:
    g = p[(p["省级地区"] != SHANGHAI) & (p["主体大类"] == "企业")]
    if g.empty:
        return pd.DataFrame()
    result = (
        g.groupby(["第一申请人", "省级地区", "第一申请人城市"], dropna=False)
         .agg(
            专利数=("专利名称", "size"),
            活跃年份数=("year", "nunique"),
            高相关专利数=("relevance", lambda x: x.eq("高相关").sum()),
            严格核心专利数=("是否严格核心", "sum"),
            PNT候选专利数=("是否PNT候选", "sum"),
            主要技术方向=("main_category", lambda x: x.value_counts().index[0] if len(x) else ""),
            技术方向覆盖=("main_category", join_unique),
            产业链环节覆盖=("chain_position", join_unique),
            量子识别得分均值=("quantum_score", "mean"),
         )
         .reset_index()
         .sort_values(["专利数", "高相关专利数", "量子识别得分均值"], ascending=False)
         .head(n)
    )
    return result


def enterprise_concentration(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region in COMPARE_REGIONS:
        g = p[(p["省级地区"] == region) & (p["主体大类"] == "企业")]
        counts = g.groupby("第一申请人").size()
        row = {"省级地区": region, "明确企业数": int(counts.size), "企业专利数": int(counts.sum())}
        row.update(concentration_metrics(counts))
        rows.append(row)
    return pd.DataFrame(rows)


def entrant_persistence(p: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = sorted(p["year"].dropna().astype(int).unique())
    if len(years) < 2:
        return pd.DataFrame(), pd.DataFrame()
    ent = p[p["主体大类"] == "企业"].copy()
    first = ent.groupby(["省级地区", "第一申请人"])["year"].min().rename("首次年份")
    active = ent.groupby(["省级地区", "第一申请人"])["year"].nunique().rename("活跃年份数")
    firm_panel = pd.concat([first, active], axis=1).reset_index()
    annual = ent.groupby(["省级地区", "year"])["第一申请人"].nunique().rename("活跃企业数").reset_index()
    entrants = firm_panel.groupby(["省级地区", "首次年份"]).size().rename("新进入企业数").reset_index().rename(columns={"首次年份": "year"})
    annual = annual.merge(entrants, on=["省级地区", "year"], how="left").fillna({"新进入企业数": 0})
    annual["新进入企业占比"] = annual["新进入企业数"] / annual["活跃企业数"].replace(0, np.nan)
    return firm_panel, annual


def save_tables(p: pd.DataFrame, fy: pd.DataFrame, f: pd.DataFrame, audit: dict):
    region = region_summary(p)
    compare = region[region["省级地区"].isin(COMPARE_REGIONS)].copy()
    compare["排序"] = compare["省级地区"].map({r: i for i, r in enumerate(COMPARE_REGIONS)})
    compare = compare.sort_values("排序").drop(columns="排序")
    yearly = yearly_summary(p)
    sh_cat = technology_table(p, "main_category", SHANGHAI)
    sh_chain = technology_table(p, "chain_position", SHANGHAI)
    comp_cat = comparison_technology(p, "main_category")
    comp_chain = comparison_technology(p, "chain_position")
    sh_entities = entity_ranking(p, SHANGHAI).head(TOP_ENTITY_N)
    sh_enterprises = entity_ranking(p, SHANGHAI, "企业")
    sh_research = pd.concat([
        entity_ranking(p, SHANGHAI, "高校"),
        entity_ranking(p, SHANGHAI, "科研院所"),
    ], ignore_index=True).sort_values("专利数", ascending=False)
    other_ent = other_top_enterprises(p, TOP_OTHER_ENTERPRISE_N)
    concentration = enterprise_concentration(p)
    firm_panel, entrants = entrant_persistence(p)

    tables = {
        "01_全国地区概览.csv": region,
        "02_上海与重点省市比较.csv": compare,
        "03_地区年度趋势.csv": yearly,
        "04_上海技术方向.csv": sh_cat,
        "05_上海产业链环节.csv": sh_chain,
        "06_重点省市技术方向比较.csv": comp_cat,
        "07_重点省市产业链比较.csv": comp_chain,
        "08_上海创新主体排名.csv": sh_entities,
        "09_上海企业排名.csv": sh_enterprises,
        "10_上海高校院所排名.csv": sh_research,
        "11_其他省市代表性企业.csv": other_ent,
        "12_企业集中度比较.csv": concentration,
        "13_企业持续性明细.csv": firm_panel,
        "14_新进入企业年度统计.csv": entrants,
    }
    for filename, df in tables.items():
        df.to_csv(TABLE_DIR / filename, index=False, encoding="utf-8-sig")

    # Cross-check the derived firm tables without treating their 'firm_relevance' label as enterprise identity.
    check = pd.DataFrame([
        {"检查项": "专利明细is_quantum_patent合计", "数值": int(p["is_quantum_patent"].sum())},
        {"检查项": "企业年度表quantum_patent_count合计", "数值": int(pd.to_numeric(fy["quantum_patent_count"], errors="coerce").sum())},
        {"检查项": "企业汇总表quantum_patent_count合计", "数值": int(pd.to_numeric(f["quantum_patent_count"], errors="coerce").sum())},
    ])
    check.to_csv(TABLE_DIR / "15_三表一致性检查.csv", index=False, encoding="utf-8-sig")

    return tables


def barh_chart(df, label_col, value_col, title, filename, n=15):
    if df.empty:
        return
    plot = df.head(n).sort_values(value_col)
    fig, ax = plt.subplots(figsize=(9, max(4.8, 0.42 * len(plot) + 1.2)))
    bars = ax.barh(plot[label_col], plot[value_col], color=PALETTE[0])
    ax.set_title(title, pad=14)
    ax.set_xlabel(value_col)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    for bar, value in zip(bars, plot[value_col]):
        ax.text(value, bar.get_y() + bar.get_height()/2, f" {value:g}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def make_figures(p: pd.DataFrame, tables: dict):
    set_plot_style()
    compare = tables["02_上海与重点省市比较.csv"]
    x = np.arange(len(compare))
    fig, ax1 = plt.subplots(figsize=(10, 5.8))
    width = 0.36
    ax1.bar(x - width/2, compare["量子相关专利数"], width, label="专利数", color=PALETTE[0])
    ax1.bar(x + width/2, compare["创新主体数"], width, label="创新主体数", color=PALETTE[1])
    ax1.set_xticks(x, compare["省级地区"])
    ax1.set_title("上海与重点省市量子信息创新规模比较")
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_重点省市创新规模比较.png", bbox_inches="tight")
    plt.close(fig)

    sh_cat = tables["04_上海技术方向.csv"]
    barh_chart(sh_cat, "类别", "专利数", "上海量子信息专利技术方向分布", "02_上海技术方向.png")
    sh_chain = tables["05_上海产业链环节.csv"]
    barh_chart(sh_chain, "类别", "专利数", "上海量子信息专利产业链环节分布", "03_上海产业链环节.png")

    sh_ent = tables["09_上海企业排名.csv"]
    barh_chart(sh_ent, "第一申请人", "专利数", "上海量子信息企业专利数量排名", "04_上海企业排名.png", n=12)

    # Stacked relevance structure.
    g = p[p["省级地区"].isin(COMPARE_REGIONS)]
    rel = pd.crosstab(g["省级地区"], g["relevance"]).reindex(COMPARE_REGIONS).fillna(0)
    rel = rel.div(rel.sum(axis=1).replace(0, np.nan), axis=0)
    order = ["高相关", "中相关", "低相关/待复核", "量子计量/PNT候选"]
    rel = rel.reindex(columns=[c for c in order if c in rel.columns])
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bottom = np.zeros(len(rel))
    for i, col in enumerate(rel.columns):
        vals = rel[col].values
        ax.bar(rel.index, vals, bottom=bottom, label=col, color=PALETTE[i % len(PALETTE)])
        bottom += vals
    ax.set_title("重点省市量子专利相关性结构")
    ax.set_ylabel("占比")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_重点省市相关性结构.png", bbox_inches="tight")
    plt.close(fig)

    # Technology share heatmap.
    comp_cat = tables["06_重点省市技术方向比较.csv"]
    pivot = comp_cat.pivot(index="省级地区", columns="类别", values="地区内占比").reindex(COMPARE_REGIONS).fillna(0)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues", vmin=0)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=28, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("重点省市量子信息技术结构比较")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if value > pivot.values.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="地区内专利占比")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_重点省市技术结构热力图.png", bbox_inches="tight")
    plt.close(fig)

    # Applicant structure for Shanghai and comparison regions.
    structure = pd.crosstab(g["省级地区"], g["主体大类"]).reindex(COMPARE_REGIONS).fillna(0)
    structure = structure.div(structure.sum(axis=1).replace(0, np.nan), axis=0)
    cols = [c for c in ["企业", "高校", "科研院所", "机关团体", "个人", "其他", "混合/待核验"] if c in structure.columns]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bottom = np.zeros(len(structure))
    for i, col in enumerate(cols):
        vals = structure[col].values
        ax.bar(structure.index, vals, bottom=bottom, label=col, color=PALETTE[i % len(PALETTE)])
        bottom += vals
    ax.set_title("重点省市量子专利第一申请人类型结构")
    ax.set_ylabel("专利占比")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "07_重点省市主体类型结构.png", bbox_inches="tight")
    plt.close(fig)

    # Trend only when multiple years are available.
    years = sorted(p["year"].dropna().unique())
    if len(years) >= 2:
        trend = tables["03_地区年度趋势.csv"]
        trend = trend[trend["省级地区"].isin(COMPARE_REGIONS)]
        fig, ax = plt.subplots(figsize=(10, 5.8))
        for i, region in enumerate(COMPARE_REGIONS):
            t = trend[trend["省级地区"] == region].sort_values("year")
            ax.plot(t["year"], t["量子相关专利数"], marker="o", linewidth=2, label=region, color=PALETTE[i])
        ax.set_title("重点省市量子信息专利年度变化")
        ax.set_xlabel("年份")
        ax.set_ylabel("专利数")
        ax.grid(linestyle="--", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, ncol=3)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "08_重点省市年度趋势.png", bbox_inches="tight")
        plt.close(fig)


def write_audit_files(audit: dict, p: pd.DataFrame):
    (AUDIT_DIR / "data_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    years_text = "、".join(map(str, audit["years"])) if audit["years"] else "无"
    lines = [
        "# 数据审计结果",
        "",
        f"- 专利明细记录：{audit['patent_rows']:,} 条",
        f"- 企业—年份记录：{audit['firm_year_rows']:,} 条",
        f"- 企业/主体汇总记录：{audit['firm_rows']:,} 条",
        f"- 当前年份：{years_text}",
        f"- 是否为单年测试数据：{'是' if audit['single_year_test'] else '否'}",
        f"- 第一申请人数量：{audit['unique_applicants']:,} 个",
        f"- 城市未映射记录：{audit['unmapped_city_rows']:,} 条",
        f"- 主体类型需复核记录占比：{audit['ambiguous_type_share']:.1%}",
        f"- 被引字段非空：{audit['citation_non_null']:,} 条，覆盖率 {audit['citation_coverage']:.1%}",
        f"- 当前是否允许将被引作为主分析指标：{'是' if audit['citation_metrics_allowed'] else '否'}",
        f"- 基础键潜在重复记录：{audit['potential_duplicate_rows_basic_key']:,} 条（仅提示，不自动删除）",
        "",
        "## 强制写作约束",
        "",
        "1. 当前仅有一个年份时，不得写增长率、趋势、持续创新、新进入企业或阶段演进结论。",
        "2. 被引字段覆盖率未达到70%时，不得将平均被引、高被引专利作为主要质量证据。",
        "3. `firm_relevance`表示主体的量子相关等级，不表示该主体一定是企业。",
        "4. `第一申请人地区`若均为“中国”，省际比较只能依据`第一申请人城市`映射；必须披露未映射城市。",
        "5. 主体类型混合值需要名称辅助识别，报告中应使用“明确识别为企业”的保守口径。",
        "6. 专利数据可反映技术创新布局，不能直接证明营业收入、市场份额、融资、产品落地或产业规模。",
    ]
    if audit["unmapped_cities"]:
        lines += ["", "## 待补充城市映射", "", "、".join(audit["unmapped_cities"])]
    (AUDIT_DIR / "data_audit.md").write_text("\n".join(lines), encoding="utf-8")

    type_check = (
        p.groupby(["第一申请人类型", "主体大类", "主体类型分类依据", "主体类型需复核"], dropna=False)
         .size().rename("记录数").reset_index().sort_values("记录数", ascending=False)
    )
    type_check.to_csv(AUDIT_DIR / "applicant_type_audit.csv", index=False, encoding="utf-8-sig")

    unmapped = p[p["省级地区"] == "待映射"]["第一申请人城市"].value_counts().rename_axis("第一申请人城市").reset_index(name="记录数")
    unmapped.to_csv(AUDIT_DIR / "unmapped_cities.csv", index=False, encoding="utf-8-sig")


def write_analysis_summary(audit: dict, tables: dict):
    compare = tables["02_上海与重点省市比较.csv"]
    sh = compare[compare["省级地区"] == SHANGHAI]
    sh_cat = tables["04_上海技术方向.csv"]
    sh_chain = tables["05_上海产业链环节.csv"]
    sh_ent = tables["09_上海企业排名.csv"].head(3)
    other = tables["11_其他省市代表性企业.csv"]

    lines = ["# 供报告写作使用的结构化分析摘要", ""]
    lines += ["## 数据适用范围", ""]
    if audit["single_year_test"]:
        lines.append(f"当前测试数据仅覆盖{audit['years'][0]}年，只能用于截面比较和代码验证；正式多年数据更新后，年度趋势和企业持续性模块会自动启用。")
    else:
        lines.append(f"当前数据覆盖{audit['years'][0]}—{audit['years'][-1]}年，可开展年度趋势、进入和持续性分析。")
    lines.append(f"被引字段覆盖率为{audit['citation_coverage']:.1%}，因此当前{'可以' if audit['citation_metrics_allowed'] else '不可以'}将被引指标作为主要结论依据。")

    lines += ["", "## 上海与重点省市", ""]
    lines.append(compare.to_markdown(index=False))
    lines += ["", "## 上海技术方向", "", sh_cat.to_markdown(index=False)]
    lines += ["", "## 上海产业链环节", "", sh_chain.to_markdown(index=False)]
    lines += ["", "## 上海专利口径下的前三家代表性企业候选", "", sh_ent.to_markdown(index=False)]
    lines += ["", "## 其他省市代表性企业候选", "", other.to_markdown(index=False)]
    lines += ["", "## 写作注意", "",
              "- 上述企业仅是依据当前专利数据选出的案例候选，不等于政府认定的龙头企业。",
              "- 企业基本情况、产品、融资、市场和应用案例必须另行查找权威来源并记录出处。",
              "- 任何省市优劣判断都应同时展示指标、年份与口径，避免将单年专利授权量等同于产业综合实力。"]
    (OUTPUT_DIR / "analysis_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    ensure_dirs(OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, AUDIT_DIR)
    patents, firm_year, firms, mapping = read_data()
    p = prepare_patents(patents, mapping)
    audit = build_audit(p, firm_year, firms)
    write_audit_files(audit, p)
    tables = save_tables(p, firm_year, firms, audit)
    make_figures(p, tables)
    write_analysis_summary(audit, tables)
    print("分析完成。")
    print(f"审计结果：{AUDIT_DIR}")
    print(f"表格结果：{TABLE_DIR}")
    print(f"图形结果：{FIGURE_DIR}")
    print(f"写作摘要：{OUTPUT_DIR / 'analysis_summary.md'}")


if __name__ == "__main__":
    main()
