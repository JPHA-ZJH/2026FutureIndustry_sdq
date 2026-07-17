import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def choose_chinese_font() -> str:
    preferred = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans CN",
        "PingFang SC", "WenQuanYi Micro Hei", "Arial Unicode MS"
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            return name
    return "DejaVu Sans"


def set_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [choose_chinese_font()]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 220
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9


def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def classify_applicant(name, raw_type):
    """Return (主体大类, 分类依据, 是否需复核).

    The source type is used first, but obvious conflicts between the source type and
    applicant name are conservatively flagged instead of being forced into "企业".
    """
    name = clean_text(name)
    raw = clean_text(raw_type)

    company_terms = ["有限公司", "有限责任公司", "股份公司", "股份有限公司", "集团公司", "公司"]
    research_terms = ["科学院", "研究所", "研究院", "实验室"]
    school_terms = ["大学", "学院", "学校"]

    has_company = any(t in name for t in company_terms)
    has_research = any(t in name for t in research_terms)
    has_school = any(t in name for t in school_terms) and "科学院" not in name

    if raw == "企业":
        if has_research and not has_company:
            return "混合/待核验", "原字段为企业但名称疑似科研机构", True
        return "企业", "原始申请人类型", False
    if raw == "学校":
        if has_company:
            return "混合/待核验", "原字段为学校但名称疑似企业", True
        return "高校", "原始申请人类型", False
    if raw == "科研单位":
        if has_company:
            return "混合/待核验", "原字段为科研单位但名称疑似企业", True
        return "科研院所", "原始申请人类型", False
    if raw == "个人":
        return "个人", "原始申请人类型", False
    if raw == "机关团体":
        return "机关团体", "原始申请人类型", False
    if raw == "其他":
        return "其他", "原始申请人类型", False

    # Mixed or missing source types: use applicant-name cues, but keep a review flag.
    if has_company:
        return "企业", "名称辅助识别", True
    if has_research:
        return "科研院所", "名称辅助识别", True
    if has_school:
        return "高校", "名称辅助识别", True
    return "混合/待核验", "无法可靠识别", True


def add_applicant_classification(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rows = out.apply(
        lambda r: classify_applicant(r.get("第一申请人", ""), r.get("第一申请人类型", r.get("first_applicant_types", ""))),
        axis=1,
        result_type="expand",
    )
    rows.columns = ["主体大类", "主体类型分类依据", "主体类型需复核"]
    return pd.concat([out, rows], axis=1)


def add_province(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["第一申请人城市"] = out["第一申请人城市"].map(clean_text)
    mapping = mapping.drop_duplicates("第一申请人城市")
    out = out.merge(mapping, on="第一申请人城市", how="left")
    # Future files may directly store province-like values in the city field.
    province_like = out["第一申请人城市"].str.endswith(("省", "市", "自治区", "特别行政区"), na=False)
    direct_values = out["第一申请人城市"].where(province_like)
    municipalities = {"上海市", "北京市", "天津市", "重庆市"}
    out.loc[out["省级地区"].isna() & direct_values.isin(municipalities), "省级地区"] = direct_values
    out["省级地区"] = out["省级地区"].fillna("待映射")
    return out


def safe_share(numerator, denominator):
    if denominator in (0, None) or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def join_unique(values: Iterable) -> str:
    result = []
    for value in values:
        if pd.isna(value):
            continue
        for item in re.split(r"[；;]", str(value)):
            item = item.strip()
            if item and item not in result:
                result.append(item)
    return "；".join(result)


def concentration_metrics(counts: pd.Series) -> dict:
    counts = pd.to_numeric(counts, errors="coerce").dropna()
    counts = counts[counts > 0].sort_values(ascending=False)
    total = counts.sum()
    if total == 0:
        return {"CR1": np.nan, "CR3": np.nan, "CR5": np.nan, "CR10": np.nan, "HHI": np.nan}
    shares = counts / total
    return {
        "CR1": shares.head(1).sum(),
        "CR3": shares.head(3).sum(),
        "CR5": shares.head(5).sum(),
        "CR10": shares.head(10).sum(),
        "HHI": (shares ** 2).sum(),
    }


def write_markdown_table(df: pd.DataFrame, path: Path, title: str = "") -> None:
    lines = []
    if title:
        lines += [f"# {title}", ""]
    if df.empty:
        lines.append("无可用数据。")
    else:
        lines.append(df.to_markdown(index=False))
    path.write_text("\n".join(lines), encoding="utf-8")
