from __future__ import annotations

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


def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_separators(text: str) -> str:
    return re.sub(r"\s*[;；]\s*", "；", clean_text(text))


def first_item(value) -> str:
    text = normalize_separators(value)
    return text.split("；", 1)[0].strip() if text else ""


def normalize_city(value) -> str:
    """统一城市写法。省份映射依赖全国城市参考表，未识别值进入审计清单。"""
    city = clean_text(value)
    if not city:
        return ""
    city = re.sub(r"\s+", "", city)
    aliases = {
        "北京市": "北京", "北京": "北京", "上海市": "上海", "上海": "上海",
        "天津市": "天津", "天津": "天津", "重庆市": "重庆", "重庆": "重庆",
        "香港特别行政区": "香港", "香港": "香港", "澳门特别行政区": "澳门", "澳门": "澳门",
    }
    if city in aliases:
        return aliases[city]
    if city.endswith("市") and len(city) > 2:
        return city[:-1]
    return city


def normalize_province(value) -> str:
    province = clean_text(value)
    if not province:
        return ""
    province = re.sub(r"\s+", "", province)
    aliases = {
        "北京市": "北京", "北京": "北京", "上海市": "上海", "上海": "上海",
        "天津市": "天津", "天津": "天津", "重庆市": "重庆", "重庆": "重庆",
        "内蒙古自治区": "内蒙古", "内蒙古": "内蒙古", "广西壮族自治区": "广西", "广西": "广西",
        "西藏自治区": "西藏", "西藏": "西藏", "宁夏回族自治区": "宁夏", "宁夏": "宁夏",
        "新疆维吾尔自治区": "新疆", "新疆": "新疆", "香港特别行政区": "香港", "香港": "香港",
        "澳门特别行政区": "澳门", "澳门": "澳门", "台湾省": "台湾", "台湾": "台湾",
    }
    if province in aliases:
        return aliases[province]
    for suffix in ("省", "市"):
        if province.endswith(suffix) and len(province) > 2:
            return province[:-1]
    return province


def build_city_province_mapping(cities: pd.DataFrame, provinces: pd.DataFrame) -> pd.DataFrame:
    cities = cities.copy()
    provinces = provinces.copy()
    cities["provinceCode"] = pd.to_numeric(cities["provinceCode"], errors="coerce").astype("Int64")
    provinces["code"] = pd.to_numeric(provinces["code"], errors="coerce").astype("Int64")
    provinces["省级地区"] = provinces["name"].map(normalize_province)
    mapping = cities.merge(provinces[["code", "省级地区"]], left_on="provinceCode", right_on="code", how="left")
    mapping["城市标准化"] = mapping["name"].map(normalize_city)
    mapping = mapping[(mapping["城市标准化"] != "") & (~mapping["城市标准化"].isin(["市辖区", "县", "省直辖县级行政区划", "自治区直辖县级行政区划"]))]
    mapping = mapping[["城市标准化", "省级地区"]].drop_duplicates()
    # 补充直辖县级市、自治州简称及参考表中常见缺项。
    # 未来出现的新城市仍会进入未映射清单，不会被强行猜测。
    supplement_pairs = {
        "北京": "北京", "上海": "上海", "天津": "天津", "重庆": "重庆",
        "香港": "香港", "澳门": "澳门", "台湾": "台湾", "台湾省": "台湾",
        "台北": "台湾", "新北": "台湾", "桃园": "台湾", "台中": "台湾", "台南": "台湾",
        "高雄": "台湾", "新竹": "台湾", "新竹县": "台湾", "基隆": "台湾", "嘉义": "台湾",
        "凉山": "四川", "阿坝": "四川", "甘孜": "四川",
        "延边": "吉林", "大兴安岭": "黑龙江", "兴安": "内蒙古",
        "石河子": "新疆", "阿拉尔": "新疆", "五家渠": "新疆", "铁门关": "新疆",
        "图木舒克": "新疆", "克孜勒苏": "新疆", "伊犁": "新疆", "巴音": "新疆",
        "巴音郭楞": "新疆", "阿勒泰": "新疆", "阿克苏": "新疆",
        "楚雄": "云南", "文山": "云南", "大理": "云南", "红河": "云南", "西双版纳": "云南", "迪庆": "云南",
        "恩施": "湖北", "仙桃": "湖北", "神农架": "湖北",
        "万宁": "海南", "琼海": "海南", "文昌": "海南", "澄迈": "海南", "定安": "海南",
        "屯昌": "海南", "陵水黎族自治县": "海南", "白沙黎族自治县": "海南",
        "黔东南": "贵州", "黔南": "贵州", "黔西南": "贵州",
        "济源": "河南", "甘南": "甘肃",
    }
    supplements = pd.DataFrame({"城市标准化": list(supplement_pairs), "省级地区": list(supplement_pairs.values())})
    return pd.concat([mapping, supplements], ignore_index=True).drop_duplicates("城市标准化")


def infer_first_applicant_type(name, raw_type) -> tuple[str, str, bool]:
    """依据第一申请人名称给出唯一主体类型，解决合作专利混合类型问题。"""
    name = clean_text(name)
    raw = clean_text(raw_type)
    base_name = re.split(r"[（(]", name, maxsplit=1)[0].strip() or name

    strong_enterprise = [
        r"有限公司", r"有限责任公司", r"股份有限公司", r"股份公司", r"有限合伙", r"合伙企业",
        r"Corporation", r"Company", r"Co\.?[, ]?Ltd", r"Ltd\.?", r"Inc\.?", r"LLC", r"GmbH", r"S\.A\.?", r"PLC", r"株式会社", r"会社",
    ]
    generic_enterprise = [r"公司", r"集团", r"企业", r"厂$", r"药业$", r"生物科技$", r"生物技术$"]
    medical = [
        r"医院", r"卫生院", r"妇幼保健院", r"疾病预防控制中心", r"疾控中心", r"血液中心", r"医疗中心",
        r"门诊部", r"诊所", r"疗养院", r"医学中心", r"Health System", r"Medical Center", r"Hospital",
    ]
    school = [r"大学", r"学院", r"学校", r"高等专科学校", r"职业技术学院", r"University", r"College", r"School of"]
    research = [
        r"中国科学院", r"农业科学院", r"医学科学院", r"科学院", r"研究所", r"研究院", r"研究设计院", r"实验室", r"科学中心",
        r"研究中心", r"研发中心", r"科研中心", r"工程中心", r"创新中心", r"技术中心", r"检测院", r"勘察院", r"试验站", r"研究局", r"研究机构",
        r"Institute", r"Laboratory", r"Laboratories", r"Research Center", r"Research Centre",
    ]
    government = [
        r"人民政府", r"委员会", r"管理局", r"公安局", r"气象局", r"税务局", r"财政局", r"农业农村局", r"市场监督管理局", r"海关",
        r"生产力促进中心", r"环境监测中心", r"技术推广站", r"畜牧兽医工作站", r"森林公园", r"国有.*林场",
        r"厅$", r"部$", r"署$", r"协会", r"学会", r"联盟", r"联合会", r"部队", r"军区", r"解放军",
    ]

    def matches(text, patterns):
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)

    # “研究院有限公司”应归企业；大学附属医院应归医疗机构。
    if matches(base_name, strong_enterprise):
        return "企业", "第一申请人名称规则", False
    if matches(base_name, medical):
        return "医疗机构", "第一申请人名称规则", False
    if matches(base_name, research):
        return "科研院所", "第一申请人名称规则", False
    if matches(base_name, school) and "科学院" not in base_name:
        return "高校", "第一申请人名称规则", False
    if matches(base_name, government):
        return "机关团体", "第一申请人名称规则", False
    if matches(base_name, generic_enterprise):
        return "企业", "第一申请人名称规则", False

    if matches(name, strong_enterprise):
        return "企业", "第一申请人完整名称规则", False
    if matches(name, medical):
        return "医疗机构", "第一申请人完整名称规则", False
    if matches(name, research):
        return "科研院所", "第一申请人完整名称规则", False
    if matches(name, school) and "科学院" not in name:
        return "高校", "第一申请人完整名称规则", False
    if matches(name, government):
        return "机关团体", "第一申请人完整名称规则", False
    if matches(name, generic_enterprise):
        return "企业", "第一申请人完整名称规则", False

    raw_parts = [x.strip() for x in re.split(r"[,，;；]", raw) if x.strip()]
    raw_unique = list(dict.fromkeys(raw_parts))
    raw_map = {"企业": "企业", "学校": "高校", "科研单位": "科研院所", "机关团体": "机关团体", "个人": "个人", "其他": "其他"}
    if len(raw_unique) == 1 and raw_unique[0] in raw_map:
        return raw_map[raw_unique[0]], "原字段单一类型兜底", False
    if "个人" in raw_unique and len(name) <= 12:
        return "个人", "原字段个人类型兜底", True
    # 始终给出唯一类型；无法可靠归入主要类型的主体列为“其他机构”，并保留复核标记。
    return "其他机构", "名称和原字段均无法唯一判断", True


def add_first_applicant_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "申请人" in out.columns:
        derived = out["申请人"].map(first_item)
        existing = out.get("第一申请人", pd.Series("", index=out.index)).map(clean_text)
        out["第一申请人_原值"] = existing
        out["第一申请人"] = derived.where(derived.ne(""), existing)
        out["第一申请人提取是否变化"] = out["第一申请人"].ne(existing) & existing.ne("")
    else:
        existing = out.get("第一申请人", pd.Series("", index=out.index)).map(clean_text)
        out["第一申请人_原值"] = existing
        out["第一申请人"] = existing
        out["第一申请人提取是否变化"] = False

    raw_type_col = "第一申请人类型" if "第一申请人类型" in out.columns else "first_applicant_types"
    raw_types = out.get(raw_type_col, pd.Series("", index=out.index))
    inferred = [infer_first_applicant_type(n, t) for n, t in zip(out["第一申请人"], raw_types)]
    inferred_df = pd.DataFrame(inferred, columns=["主体类型", "主体类型判断依据", "主体类型需复核"], index=out.index)
    return pd.concat([out, inferred_df], axis=1)


def add_location_fields(df: pd.DataFrame, city_mapping: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["第一申请人城市_原值"] = out.get("第一申请人城市", pd.Series("", index=out.index)).map(clean_text)
    out["第一申请人城市标准化"] = out["第一申请人城市_原值"].map(normalize_city)
    out = out.merge(city_mapping, left_on="第一申请人城市标准化", right_on="城市标准化", how="left")
    out.drop(columns=["城市标准化"], inplace=True, errors="ignore")

    raw_region = out.get("第一申请人地区", pd.Series("", index=out.index)).map(clean_text)
    domestic_labels = ["中国", "中国大陆", "中华人民共和国", ""]
    domestic = raw_region.isin(domestic_labels) & out["第一申请人城市标准化"].ne("")
    direct_province = raw_region.map(normalize_province)
    province_names = ["北京", "天津", "上海", "重庆", "河北", "山西", "内蒙古", "辽宁", "吉林", "黑龙江", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南", "湖北", "湖南", "广东", "广西", "海南", "四川", "贵州", "云南", "西藏", "陕西", "甘肃", "青海", "宁夏", "新疆", "香港", "澳门", "台湾"]
    direct_is_province = direct_province.isin(province_names) & ~raw_region.isin(domestic_labels)
    out.loc[direct_is_province, "省级地区"] = direct_province[direct_is_province]

    foreign = ~raw_region.isin(domestic_labels) & ~direct_is_province
    out.loc[foreign, "省级地区"] = raw_region[foreign]
    out["是否中国境内申请人"] = (~foreign).astype(int)
    out["城市映射状态"] = np.select(
        [foreign, direct_is_province, domestic & out["省级地区"].notna(), domestic & out["省级地区"].isna()],
        ["境外申请人", "地区字段直接识别", "城市映射成功", "中国城市待映射"],
        default="缺少城市信息",
    )
    return out


def join_unique(values: Iterable) -> str:
    result = []
    for value in values:
        if pd.isna(value):
            continue
        for item in re.split(r"[；;,，]", str(value)):
            item = item.strip()
            if item and item not in result:
                result.append(item)
    return "；".join(result)


def mode_or_blank(series: pd.Series) -> str:
    vals = series.dropna().astype(str)
    vals = vals[vals.str.strip().ne("")]
    return vals.value_counts().index[0] if not vals.empty else ""


def concentration_metrics(counts: pd.Series) -> dict:
    counts = pd.to_numeric(counts, errors="coerce").dropna()
    counts = counts[counts > 0].sort_values(ascending=False)
    total = counts.sum()
    if total == 0:
        return {"CR1": np.nan, "CR3": np.nan, "CR5": np.nan, "CR10": np.nan, "HHI": np.nan}
    shares = counts / total
    return {"CR1": float(shares.head(1).sum()), "CR3": float(shares.head(3).sum()), "CR5": float(shares.head(5).sum()), "CR10": float(shares.head(10).sum()), "HHI": float((shares ** 2).sum())}


def choose_chinese_font() -> str:
    candidates = ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"]
    for path in candidates:
        try:
            if Path(path).exists():
                font_manager.fontManager.addfont(path)
        except Exception:
            pass
    preferred = ["Microsoft YaHei", "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans CN", "SimHei", "PingFang SC", "WenQuanYi Micro Hei", "Noto Serif CJK SC", "Arial Unicode MS"]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in installed:
            return name
    return "DejaVu Sans"


def set_plot_style() -> None:
    plt.rcParams.update({
        "font.sans-serif": [choose_chinese_font()], "axes.unicode_minus": False,
        "figure.dpi": 140, "savefig.dpi": 240, "axes.titleweight": "bold", "axes.titlesize": 14,
        "axes.labelsize": 10.5, "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
