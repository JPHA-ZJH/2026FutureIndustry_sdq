"""Run only the figure generation part, using already-computed tables."""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
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
    path = FIGURE_DIR / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    ensure_dirs(FIGURE_DIR)
    set_plot_style()

    # Load pre-computed tables
    def load_csv(name):
        return pd.read_csv(TABLE_DIR / name)

    compare = load_csv("02_上海与重点省市比较.csv")
    sh_cat = load_csv("04_上海技术方向.csv")
    sh_chain = load_csv("05_上海产业链环节.csv")
    sh_ent = load_csv("09_上海企业排名.csv")
    comp_cat = load_csv("06_重点省市技术方向比较.csv")
    yearly = load_csv("03_地区年度趋势.csv")
    entrants_df = load_csv("14_新进入企业年度统计.csv")
    sh_yearly_cat = load_csv("15_上海年度技术方向.csv")
    sh_ent_yearly = load_csv("17_上海头部企业年度趋势.csv")

    # Load patent data for relevance and applicant structure
    mapping = pd.read_csv(CITY_PROVINCE_FILE)
    patents = pd.read_csv(PATENT_FILE, low_memory=False)
    p = add_applicant_classification(patents)
    p = add_province(p, mapping)
    p["year"] = pd.to_numeric(p["year"], errors="coerce").astype("Int64")
    p["是否严格核心"] = p["relevance"].isin(STRICT_CORE_LABELS).astype(int)
    p["是否PNT候选"] = p["relevance"].eq("量子计量/PNT候选").astype(int)

    # ==================== FIGURE 01 ====================
    print("Generating Figure 01...")
    x = np.arange(len(compare))
    fig, ax1 = plt.subplots(figsize=(10, 5.8))
    width = 0.36
    ax1.bar(x - width/2, compare["量子相关专利数"], width, label="专利数", color=PALETTE[0])
    ax1.bar(x + width/2, compare["创新主体数"], width, label="创新主体数", color=PALETTE[1])
    ax1.set_xticks(x, compare["省级地区"])
    ax1.set_title("上海与重点省市量子信息创新规模比较（2021-2025累计）")
    ax1.grid(axis="y", linestyle="--", alpha=0.25)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_重点省市创新规模比较.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 01")

    # ==================== FIGURE 02 ====================
    print("Generating Figure 02...")
    barh_chart(sh_cat, "类别", "专利数", "上海量子信息专利技术方向分布（2021-2025累计）", "02_上海技术方向.png")

    # ==================== FIGURE 03 ====================
    print("Generating Figure 03...")
    barh_chart(sh_chain, "类别", "专利数", "上海量子信息专利产业链环节分布（2021-2025累计）", "03_上海产业链环节.png")

    # ==================== FIGURE 04 ====================
    print("Generating Figure 04...")
    barh_chart(sh_ent, "第一申请人", "专利数", "上海量子信息企业专利数量排名（2021-2025累计）", "04_上海企业排名.png", n=12)

    # ==================== FIGURE 05 ====================
    print("Generating Figure 05...")
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
    ax.set_title("重点省市量子专利相关性结构（2021-2025累计）")
    ax.set_ylabel("占比")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_重点省市相关性结构.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 05")

    # ==================== FIGURE 06 ====================
    print("Generating Figure 06...")
    pivot = comp_cat.pivot(index="省级地区", columns="类别", values="地区内占比").reindex(COMPARE_REGIONS).fillna(0)
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues", vmin=0)
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=28, ha="right")
    ax.set_yticks(range(len(pivot.index)), pivot.index)
    ax.set_title("重点省市量子信息技术结构比较（2021-2025累计）")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8,
                    color="white" if value > pivot.values.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="地区内专利占比")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "06_重点省市技术结构热力图.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 06")

    # ==================== FIGURE 07 ====================
    print("Generating Figure 07...")
    structure = pd.crosstab(g["省级地区"], g["主体大类"]).reindex(COMPARE_REGIONS).fillna(0)
    structure = structure.div(structure.sum(axis=1).replace(0, np.nan), axis=0)
    cols = [c for c in ["企业", "高校", "科研院所", "机关团体", "个人", "其他", "混合/待核验"] if c in structure.columns]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bottom = np.zeros(len(structure))
    for i, col in enumerate(cols):
        vals = structure[col].values
        ax.bar(structure.index, vals, bottom=bottom, label=col, color=PALETTE[i % len(PALETTE)])
        bottom += vals
    ax.set_title("重点省市量子专利第一申请人类型结构（2021-2025累计）")
    ax.set_ylabel("专利占比")
    ax.set_ylim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "07_重点省市主体类型结构.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 07")

    # ==================== FIGURE 08 ====================
    print("Generating Figure 08: Yearly trends...")
    trend = yearly[yearly["省级地区"].isin(COMPARE_REGIONS)]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for i, region in enumerate(COMPARE_REGIONS):
        t = trend[trend["省级地区"] == region].sort_values("year")
        ax.plot(t["year"], t["量子相关专利数"], marker="o", linewidth=2, label=region, color=PALETTE[i])
    ax.set_title("重点省市量子信息专利年度变化（2021-2023）")
    ax.set_xlabel("年份")
    ax.set_ylabel("专利数")
    ax.grid(linestyle="--", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "08_重点省市年度趋势.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 08")

    # ==================== FIGURE 09: Shanghai multi-indicator ====================
    print("Generating Figure 09: Shanghai multi-indicator...")
    sh_trend = trend[trend["省级地区"] == SHANGHAI].sort_values("year")
    if not sh_trend.empty:
        fig, ax1 = plt.subplots(figsize=(10, 5.8))
        ax1.bar(sh_trend["year"] - 0.15, sh_trend["量子相关专利数"], 0.3,
                label="专利总数", color=PALETTE[0])
        ax1.bar(sh_trend["year"] + 0.15, sh_trend["企业专利数"], 0.3,
                label="企业专利数", color=PALETTE[3])
        ax2 = ax1.twinx()
        ax2.plot(sh_trend["year"], sh_trend["创新主体数"], marker="s", linewidth=2,
                 label="创新主体数", color=PALETTE[1])
        ax2.plot(sh_trend["year"], sh_trend["明确企业数"], marker="D", linewidth=2,
                 label="明确企业数", color=PALETTE[2])
        ax1.set_title("上海量子信息创新年度变化（2021-2023）")
        ax1.set_xlabel("年份")
        ax1.set_ylabel("专利数")
        ax2.set_ylabel("主体/企业数")
        ax1.set_xticks(sh_trend["year"])
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, ncol=2,
                   loc="upper center", bbox_to_anchor=(0.5, -0.12))
        ax1.spines[["top", "right"]].set_visible(False)
        ax1.grid(axis="y", linestyle="--", alpha=0.2)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "09_上海多维年度趋势.png", bbox_inches="tight")
        plt.close(fig)
        print("  Saved: 09")

    # ==================== FIGURE 10: 6-province subplots ====================
    print("Generating Figure 10: 6-province subplots...")
    trend_mapped = trend[trend["省级地区"].isin(COMPARE_REGIONS)]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for i, region in enumerate(COMPARE_REGIONS):
        ax = axes[i // 3][i % 3]
        t = trend_mapped[trend_mapped["省级地区"] == region].sort_values("year")
        ax.bar(t["year"] - 0.2, t["量子相关专利数"], 0.35, label="专利总数", color=PALETTE[0])
        ax.bar(t["year"] + 0.15, t["企业专利数"], 0.35, label="企业专利", color=PALETTE[3])
        ax.set_title(region, fontsize=12)
        ax.set_xticks(t["year"].astype(int))
        ax.tick_params(labelsize=8)
        if i >= 3:
            ax.set_xlabel("年份", fontsize=9)
        if i % 3 == 0:
            ax.set_ylabel("专利数", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[0]),
                plt.Rectangle((0, 0), 1, 1, color=PALETTE[3])]
    fig.legend(handles, ["专利总数", "企业专利"], frameon=False, ncol=2,
                loc="upper center", bbox_to_anchor=(0.5, -0.02), fontsize=10)
    fig.suptitle("重点省市量子信息专利年度变化（分省，2021-2023）", y=1.01, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "10_重点省市分省年度趋势.png", bbox_inches="tight")
    plt.close(fig)
    print("  Saved: 10")

    # ==================== FIGURE 11: Shanghai new entrants ====================
    print("Generating Figure 11: Enterprise dynamics...")
    sh_entrants = entrants_df[entrants_df["省级地区"] == SHANGHAI].sort_values("year")
    if not sh_entrants.empty and len(sh_entrants) >= 2:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(sh_entrants["year"], sh_entrants["活跃企业数"], color=PALETTE[0], label="活跃企业数")
        ax.bar(sh_entrants["year"], sh_entrants["新进入企业数"], color=PALETTE[1], label="新进入企业数")
        ax.set_title("上海量子信息企业进入与活跃度变化（2021-2023）")
        ax.set_xlabel("年份")
        ax.set_ylabel("企业数")
        ax.set_xticks(sh_entrants["year"].astype(int))
        ax.legend(frameon=False, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.2)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "11_上海企业进入动态.png", bbox_inches="tight")
        plt.close(fig)
        print("  Saved: 11")

    # ==================== FIGURE 12: Top enterprises over years ====================
    print("Generating Figure 12: Top enterprise trends...")
    if not sh_ent_yearly.empty:
        top5 = sh_ent_yearly.groupby("企业")["专利数"].sum().nlargest(5).index.tolist()
        fig, ax = plt.subplots(figsize=(10, 5.5))
        for i, firm in enumerate(top5):
            tf = sh_ent_yearly[sh_ent_yearly["企业"] == firm].sort_values("年份")
            ax.plot(tf["年份"], tf["专利数"], marker="o", linewidth=2,
                    label=firm[:12], color=PALETTE[i])
        ax.set_title("上海头部量子企业专利年度变化（2021-2023）")
        ax.set_xlabel("年份")
        ax.set_ylabel("专利数")
        ax.set_xticks(sorted(sh_ent_yearly["年份"].unique()))
        ax.legend(frameon=False, ncol=2, fontsize=8,
                  loc="upper center", bbox_to_anchor=(0.5, -0.15))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.2)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "12_上海头部企业年度趋势.png", bbox_inches="tight")
        plt.close(fig)
        print("  Saved: 12")

    # ==================== FIGURE 13: Tech direction yearly changes ====================
    print("Generating Figure 13: Tech direction changes...")
    if not sh_yearly_cat.empty:
        top_cats = sh_yearly_cat.groupby("类别")["专利数"].sum().nlargest(4).index.tolist()
        cat_data = sh_yearly_cat[sh_yearly_cat["类别"].isin(top_cats)]
        years_list = sorted(cat_data["年份"].unique())
        x_pos = np.arange(len(years_list))
        n_cats = len(top_cats)
        bar_width = 0.8 / n_cats
        fig, ax = plt.subplots(figsize=(10, 5.8))
        for i, cat in enumerate(top_cats):
            vals = []
            for y in years_list:
                row = cat_data[(cat_data["年份"] == y) & (cat_data["类别"] == cat)]
                vals.append(row["专利数"].values[0] if len(row) > 0 else 0)
            offset = (i - (n_cats - 1) / 2) * bar_width
            ax.bar(x_pos + offset, vals, bar_width * 0.9, label=cat, color=PALETTE[i])
        ax.set_title("上海量子信息技术方向年度变化（2021-2023）")
        ax.set_xlabel("年份")
        ax.set_ylabel("专利数")
        ax.set_xticks(x_pos, years_list)
        ax.legend(frameon=False, ncol=2, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", alpha=0.2)
        fig.tight_layout()
        fig.savefig(FIGURE_DIR / "13_上海技术方向年度变化.png", bbox_inches="tight")
        plt.close(fig)
        print("  Saved: 13")

    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    main()
