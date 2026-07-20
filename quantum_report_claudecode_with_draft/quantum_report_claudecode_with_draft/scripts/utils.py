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
    """Map city names to provinces with robust handling of "市" suffix variants.

    The mapping file uses city names WITH "市" (e.g. "上海市", "合肥市"),
    but patent data may have city names WITHOUT "市" (e.g. "上海", "合肥").
    This function normalises both sides by stripping the trailing "市" before merging,
    so "上海"/"上海市" → "上海" → "上海市", "合肥"/"合肥市" → "合肥" → "安徽省".
    """
    out = df.copy()
    out["第一申请人城市"] = out["第一申请人城市"].map(clean_text)

    # ----- normalised key: strip trailing "市" from both sides -----
    mapping_norm = mapping.drop_duplicates("第一申请人城市").copy()
    mapping_norm["_city_key"] = mapping_norm["第一申请人城市"].str.replace(r"市$", "", regex=True)
    # In case different cities normalise to the same key (e.g. "吉林市" → "吉林"),
    # keep the first mapping row (城市→省级 是一对一的).
    mapping_norm = mapping_norm.drop_duplicates("_city_key")

    out["_city_key"] = out["第一申请人城市"].str.replace(r"市$", "", regex=True)
    out = out.merge(mapping_norm[["_city_key", "省级地区"]], on="_city_key", how="left")

    # ----- fallback 1: municipalities written without "市" -----
    # e.g. "上海" → "上海市", "北京" → "北京市"
    MUNICIPALITIES = {"上海": "上海市", "北京": "北京市", "天津": "天津市", "重庆": "重庆市"}
    still_na = out["省级地区"].isna()
    out.loc[still_na, "省级地区"] = (
        out.loc[still_na, "_city_key"].map(MUNICIPALITIES)
    )

    # ----- fallback 2: abbreviated prefecture names → full prefecture -----
    # e.g. "凉山" → "凉山彝族自治州" → "四川省"
    PREFECTURE_ABBREV = {
        "凉山": "凉山彝族自治州", "阿坝": "阿坝藏族羌族自治州",
        "甘孜": "甘孜藏族自治州", "延边": "延边朝鲜族自治州",
        "恩施": "恩施土家族苗族自治州", "湘西": "湘西土家族苗族自治州",
        "黔西南": "黔西南布依族苗族自治州", "黔东南": "黔东南苗族侗族自治州",
        "黔南": "黔南布依族苗族自治州", "楚雄": "楚雄彝族自治州",
        "红河": "红河哈尼族彝族自治州", "文山": "文山壮族苗族自治州",
        "西双版纳": "西双版纳傣族自治州", "大理": "大理白族自治州",
        "德宏": "德宏傣族景颇族自治州", "临夏": "临夏回族自治州",
        "甘南": "甘南藏族自治州", "海北": "海北藏族自治州",
        "黄南": "黄南藏族自治州", "海南": "海南藏族自治州",
        "果洛": "果洛藏族自治州", "玉树": "玉树藏族自治州",
        "海西": "海西蒙古族藏族自治州", "昌吉": "昌吉回族自治州",
        "博尔塔拉": "博尔塔拉蒙古自治州", "巴音郭楞": "巴音郭楞蒙古自治州",
        "克孜勒苏": "克孜勒苏柯尔克孜自治州", "伊犁": "伊犁哈萨克自治州",
        "澳门": "澳门特别行政区",
    }
    still_na = out["省级地区"].isna()
    full_names = out.loc[still_na, "_city_key"].map(PREFECTURE_ABBREV)
    # Map these full names to provinces via the normalised mapping
    full_to_prov = dict(zip(
        mapping_norm["_city_key"].str.replace(r"市$", "", regex=True),
        mapping_norm["省级地区"]
    ))
    out.loc[still_na, "省级地区"] = full_names.map(full_to_prov)

    # ----- fallback 3: province names without "省" suffix -----
    province_pattern = out["第一申请人城市"].str.endswith(
        ("省", "自治区", "特别行政区"), na=False
    )
    still_na = out["省级地区"].isna() & province_pattern
    out.loc[still_na, "省级地区"] = out.loc[still_na, "第一申请人城市"]

    # ----- fallback 4: try exact original merge for anything still unmatched -----
    still_na = out["省级地区"].isna()
    if still_na.any():
        orig_map = mapping_norm[["第一申请人城市", "省级地区"]].rename(
            columns={"第一申请人城市": "_orig_city"}
        )
        unmatched = out.loc[still_na, ["第一申请人城市"]].reset_index()
        matched = unmatched.merge(orig_map, left_on="第一申请人城市", right_on="_orig_city", how="left")
        out.loc[still_na, "省级地区"] = matched["省级地区"].values

    out["省级地区"] = out["省级地区"].fillna("待映射")
    out = out.drop(columns=["_city_key"], errors="ignore")
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
