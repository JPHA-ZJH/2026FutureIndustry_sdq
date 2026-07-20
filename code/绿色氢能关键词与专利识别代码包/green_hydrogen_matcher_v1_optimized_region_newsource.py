# -*- coding: utf-8 -*-
"""绿色氢能专利匹配、技术分类与1—5分核心程度评分模块。"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd


def R(category, sub_category, terms, score, match_type, segment, source):
    return {"category": category, "sub_category": sub_category, "terms": terms,
            "score": score, "match_type": match_type,
            "industry_segment": segment, "source_type": source}


# 中文术语为主体；ALK、PEM、AEM、SOEC、LOHC、PtX等是中文技术文本中常见缩写，
# 直接保留，不另建一套重复的英文翻译词典。
GREEN_HYDROGEN_RULES: List[Dict] = [
    R("基础与系统集成", "核心概念", ["绿色氢能", "绿氢", "可再生氢", "可再生能源氢", "可再生能源制氢", "清洁低碳氢", "零碳氢", "低碳氢能"], 5, "core", "系统集成", "附件报告直接词/国家能源局"),
    R("基础与系统集成", "电氢耦合", ["电氢耦合", "电-氢耦合", "电力氢能耦合", "新能源制氢", "风电制氢", "光伏制氢", "风光制氢", "风光氢一体化", "源网荷储氢", "源网荷储氢一体化", "荷随源动", "柔性制氢"], 5, "core", "系统集成", "附件报告/国家能源局/资本研报"),
    R("基础与系统集成", "电转氢与PtX", ["电转氢", "Power-to-Hydrogen", "P2H", "电转燃料", "Power-to-X", "PtX", "电制燃料", "电制氨醇", "绿色氢氨醇", "绿氢氨醇"], 5, "core", "系统集成", "IEA/IRENA/资本研报"),
    R("基础与系统集成", "制储输用一体化", ["氢能制储输用", "制储输用一体化", "制氢储氢输氢用氢", "制储运加用", "制储运用一体化", "氢能全产业链", "氢能全链条"], 5, "core", "系统集成", "附件报告/国家能源局"),
    R("制氢", "电解水制氢", ["电解水制氢", "水电解制氢", "可再生能源电解水制氢", "绿电制氢", "电解制氢", "高压电解制氢", "差压电解制氢"], 5, "core", "制氢", "附件报告/国家能源局/IEA"),
    R("制氢", "碱性电解", ["碱性水电解", "碱性电解水", "碱性电解槽", "碱性制氢", "ALK电解槽", "ALK制氢", "AWE电解槽", "方形电解槽", "圆形电解槽", "加压碱性电解"], 5, "core", "制氢", "附件报告/资本研报"),
    R("制氢", "PEM电解", ["质子交换膜电解水", "质子交换膜制氢", "质子交换膜电解槽", "PEM电解槽", "PEM制氢", "PEMWE", "聚合物电解质膜电解"], 5, "core", "制氢", "附件报告/资本研报"),
    R("制氢", "AEM电解", ["阴离子交换膜电解水", "阴离子交换膜制氢", "阴离子交换膜电解槽", "AEM电解槽", "AEM制氢", "AEMWE"], 5, "core", "制氢", "附件报告/资本研报"),
    R("制氢", "高温固体氧化物电解", ["固体氧化物电解制氢", "固体氧化物电解槽", "高温水蒸气电解", "高温电解制氢", "SOEC制氢", "SOEC电解槽", "可逆固体氧化物电池", "rSOC"], 5, "core", "制氢", "附件报告/资本研报"),
    R("制氢", "复合与海水路线", ["碱性-PEM混合制氢", "碱性PEM混合制氢", "混合电解制氢", "海水电解制氢", "海水直接制氢", "海上制氢", "深远海制氢", "废水电解制氢"], 4, "core", "制氢", "附件报告/国家能源局/行业研究"),
    R("制氢", "前沿制氢", ["光电化学制氢", "光催化制氢", "太阳能光解水", "太阳能热化学制氢", "热化学循环制氢", "生物光解水制氢", "人工光合作用制氢", "PEC制氢"], 4, "core", "制氢", "IEA/科研机构补充"),
    R("制氢关键材料部件", "电解槽与电堆", ["制氢电解槽", "制氢电解堆", "电解槽电堆", "电解小室", "电解单元", "电解槽极框", "电解槽极板", "电解槽端板", "电解槽拉杆", "电解槽密封"], 4, "contextual", "关键材料与装备", "附件报告/资本研报"),
    R("制氢关键材料部件", "隔膜与交换膜", ["电解槽隔膜", "PPS隔膜", "复合隔膜", "离子交换膜", "质子交换膜", "阴离子交换膜", "氢氧根交换膜", "全氟磺酸膜", "膜电极组件", "膜电极", "MEA"], 4, "contextual", "关键材料与装备", "附件报告/资本研报"),
    R("制氢关键材料部件", "电催化材料", ["析氢催化剂", "析氧催化剂", "HER催化剂", "OER催化剂", "水电解催化剂", "制氢电催化剂", "铱基催化剂", "铂基催化剂", "镍铁催化剂", "非贵金属电催化剂", "催化层"], 4, "contextual", "关键材料与装备", "资本研报/科研机构补充"),
    R("制氢关键材料部件", "多孔传输与双极板", ["多孔传输层", "PTL", "气体扩散层", "GDL", "钛毡", "镍网", "泡沫镍", "双极板", "钛双极板", "金属双极板", "集流体", "扩散层"], 3, "contextual", "关键材料与装备", "附件报告/资本研报"),
    R("制氢关键材料部件", "制氢系统辅机", ["制氢整流器", "可控硅整流", "IGBT整流", "电解液循环", "气液分离器", "氢氧分离器", "氢气纯化器", "氢气干燥器", "去离子水系统", "制氢BOP", "制氢撬装系统"], 3, "contextual", "关键材料与装备", "附件报告/行业研究"),
    R("制氢运行控制", "宽负荷与波动适应", ["宽负荷制氢", "宽功率制氢", "动态制氢", "波动电源适应", "快速启停电解槽", "电解槽快速响应", "低负荷运行", "超额功率运行", "制氢负荷调节"], 4, "contextual", "制氢", "附件报告/资本研报"),
    R("制氢运行控制", "集群与能量管理", ["电解槽集群控制", "制氢集群控制", "电解槽群控", "电解槽功率分配", "制氢能量管理", "制氢功率预测", "风光功率预测制氢", "氢储能调度", "电氢协同调度"], 4, "contextual", "系统集成", "附件报告/科研机构补充"),
    R("制氢运行控制", "性能与寿命", ["电解槽电流密度", "电解槽能耗", "电解槽效率", "法拉第效率", "气体交叉渗透", "氢氧互串", "电解槽衰减", "电解槽寿命", "电极活化", "电解槽故障诊断"], 3, "contextual", "制氢", "附件报告/行业研究"),
    R("储氢", "高压气态储氢", ["高压气态储氢", "压缩氢储存", "35MPa储氢", "70MPa储氢", "储氢瓶", "储氢气瓶", "储氢瓶组", "储氢容器", "储氢罐", "管束式集装箱", "长管拖车"], 4, "contextual", "储氢", "附件报告/IEA"),
    R("储氢", "复合材料储氢瓶", ["III型储氢瓶", "IV型储氢瓶", "三型储氢瓶", "四型储氢瓶", "碳纤维缠绕储氢瓶", "塑料内胆储氢瓶", "铝内胆储氢瓶", "车载高压储氢系统"], 4, "core", "储氢", "附件报告/资本研报"),
    R("储氢", "液氢与低温储运", ["液氢", "氢液化", "氢气液化", "液氢储罐", "液氢容器", "车载液氢", "低温液氢", "仲正氢转化", "正仲氢转化", "液氢蒸发损失", "BOG氢气", "零蒸发液氢"], 4, "core", "储氢", "附件报告/IEA"),
    R("储氢", "地下储氢", ["地下储氢", "盐穴储氢", "盐洞储氢", "枯竭油气藏储氢", "含水层储氢", "岩穴储氢", "大规模地下储氢"], 4, "core", "储氢", "附件报告/IEA"),
    R("储氢", "固态与吸附储氢", ["固态储氢", "金属氢化物储氢", "镁基储氢", "稀土储氢合金", "配位氢化物", "硼氢化物储氢", "储氢合金", "吸附储氢", "多孔材料储氢", "MOF储氢", "碳材料储氢"], 4, "contextual", "储氢", "附件报告/科研机构补充"),
    R("储氢", "化学载体储氢", ["液态有机储氢", "液态有机氢载体", "LOHC", "有机液体储氢", "甲基环己烷储氢", "二苄基甲苯储氢", "氨储氢", "甲醇储氢", "氨硼烷储氢", "甲酸储氢"], 4, "core", "储氢", "IEA/资本研报"),
    R("输氢", "管道输氢", ["纯氢管道", "输氢管道", "氢气长输管道", "高压输氢管道", "跨区域输氢", "氢气管网", "氢能管网", "海底输氢管道"], 4, "core", "输氢", "附件报告/IEA"),
    R("输氢", "掺氢与分离", ["天然气掺氢", "管道掺氢", "掺氢天然气", "氢气掺混", "掺氢比例", "掺氢管输", "氢气分离", "氢气提纯", "掺氢分离", "氢气膜分离", "变压吸附提氢", "PSA提氢"], 3, "core", "输氢", "附件报告/行业研究"),
    R("输氢", "压缩与输送装备", ["氢气压缩机", "隔膜压缩机", "液驱压缩机", "离子液体压缩机", "氢气膨胀机", "氢气阀门", "氢用阀门", "氢气流量计", "氢气调压器", "氢气密封"], 3, "contextual", "输氢", "附件报告/资本研报"),
    R("加氢基础设施", "加氢站", ["加氢站", "制氢加氢一体站", "油氢合建站", "气氢合建站", "液氢加氢站", "撬装加氢站", "加氢母站", "加氢子站", "加氢机", "加氢枪", "加氢口", "预冷加氢", "加氢站顺序控制盘"], 3, "core", "加氢", "附件报告/行业研究"),
    R("氢能转化利用", "燃料电池", ["氢燃料电池", "质子交换膜燃料电池", "PEMFC", "固体氧化物燃料电池", "SOFC", "碱性燃料电池", "AFC", "可逆燃料电池", "再生燃料电池", "燃料电池电堆", "燃料电池系统"], 4, "core", "用氢", "附件报告/IEA"),
    R("氢能转化利用", "燃料电池材料部件", ["燃料电池膜电极", "燃料电池催化剂", "燃料电池双极板", "燃料电池气体扩散层", "燃料电池质子膜", "燃料电池空压机", "燃料电池氢循环泵", "燃料电池引射器", "燃料电池BOP"], 3, "contextual", "用氢", "附件报告/资本研报"),
    R("氢能转化利用", "氢燃烧与动力", ["氢燃气轮机", "掺氢燃气轮机", "纯氢燃烧", "掺氢燃烧", "氢气燃烧器", "氢内燃机", "氢能发动机", "氢锅炉", "氢氧燃烧", "氢氨燃烧", "氨氢混燃"], 4, "core", "用氢", "附件报告/行业研究"),
    R("氢能转化利用", "发电储能与热电联供", ["氢能发电", "氢储能发电", "氢电转换", "氢能热电联供", "燃料电池热电联供", "氢能长时储能", "电氢电转换", "氢能备用电源", "氢能微电网"], 3, "core", "用氢", "附件报告/IEA"),
    R("工业脱碳应用", "氢冶金", ["氢冶金", "氢基冶金", "氢还原炼铁", "氢基直接还原铁", "富氢高炉", "高炉富氢", "氢基竖炉", "直接还原铁", "DRI", "氢基熔融还原"], 4, "core", "用氢", "附件报告/国家能源局/资本研报"),
    R("工业脱碳应用", "绿色合成氨", ["绿氨", "绿色合成氨", "可再生氨", "电制氨", "绿氢合成氨", "柔性合成氨", "可再生能源合成氨", "氨氢耦合", "氨裂解制氢", "绿氨燃料"], 4, "core", "氢基衍生品", "附件报告/国家能源局/资本研报"),
    R("工业脱碳应用", "绿色甲醇与电燃料", ["绿色甲醇", "绿醇", "电制甲醇", "可再生甲醇", "绿氢制甲醇", "二氧化碳加氢制甲醇", "绿色航煤", "电制航煤", "合成航空燃料", "电制燃料", "e-fuel", "eSAF"], 4, "core", "氢基衍生品", "附件报告/IEA/资本研报"),
    R("工业脱碳应用", "炼化与化工", ["绿氢炼化", "可再生氢炼化", "绿氢加氢", "清洁氢炼化", "低碳氢化工", "氢气化工原料替代", "绿氢耦合煤化工"], 3, "contextual", "用氢", "附件报告/国家能源局"),
    R("交通应用", "氢能交通", ["燃料电池汽车", "氢燃料电池汽车", "氢能重卡", "氢燃料重卡", "氢能公交", "氢能船舶", "氢动力船舶", "氢能航空", "氢动力无人机", "氢能轨道交通"], 2, "core", "终端应用", "附件报告/IEA"),
    R("安全材料与标准", "氢安全", ["氢气泄漏检测", "氢泄漏传感器", "氢火焰探测", "氢气爆炸抑制", "氢气通风安全", "氢安全监测", "氢气风险评估", "氢气安全阀", "氢气放散", "氢氧安全联锁"], 3, "contextual", "安全与服务", "附件报告/行业研究"),
    R("安全材料与标准", "氢相容与氢脆", ["氢脆", "抗氢脆", "氢致开裂", "氢渗透", "氢相容性", "氢环境材料", "输氢管线钢", "氢气渗漏", "氢致疲劳", "氢用密封材料"], 3, "contextual", "安全与服务", "附件报告/科研机构补充"),
    R("安全材料与标准", "标准检测认证", ["氢气品质检测", "氢气纯度检测", "燃料氢质量", "氢能标准", "绿氢认证", "可再生氢认证", "氢碳足迹", "氢生命周期评价", "氢能计量", "氢气在线分析"], 2, "contextual", "安全与服务", "附件报告/IRENA/行业研究"),
]

NEGATIVE_RULES = [
    {"terms": ["煤制氢", "焦炉煤气制氢", "天然气制氢", "甲烷蒸汽重整", "SMR制氢", "工业副产氢", "灰氢"], "penalty": 5},
    {"terms": ["过氧化氢", "双氧水", "加氢油", "催化加氢脱硫", "氢化植物油", "氢原子钟", "氢键"], "penalty": 5},
]

HYDROGEN_CONTEXT_TERMS = ["氢能", "氢气", "制氢", "储氢", "输氢", "用氢", "加氢站", "绿氢", "可再生氢", "电解水", "氢燃料电池", "绿氨", "绿色甲醇"]
GREEN_CONTEXT_TERMS = ["绿色", "可再生", "绿电", "风电", "光伏", "风光", "低碳", "零碳", "清洁能源", "新能源", "海上风电"]
EXPLICIT_GREEN_TERMS = ["绿色氢能", "绿氢", "可再生氢", "可再生能源制氢", "绿电制氢", "风电制氢", "光伏制氢", "风光制氢", "电解水制氢", "绿色合成氨", "绿氨", "绿色甲醇", "绿醇"]


def safe_text(v):
    return "" if pd.isna(v) else str(v)


def normalize_text(text):
    return safe_text(text).lower().replace("－", "-").replace("—", "-").replace("–", "-").replace("（", "(").replace("）", ")")


def term_to_regex(term):
    esc = re.escape(term.strip().lower()).replace(r"\ ", r"[\s\-]*")
    if re.search(r"[\u4e00-\u9fff]", term):
        return esc
    return r"(?<![a-z0-9])" + esc.replace(r"\-", r"[\s\-]*") + r"(?![a-z0-9])"


def compile_terms(terms):
    return [(t, re.compile(term_to_regex(t), re.I)) for t in terms if t.strip()]


def compile_rules(rules):
    return [dict(x, patterns=compile_terms(x.get("terms", []))) for x in rules]


COMPILED_RULES = compile_rules(GREEN_HYDROGEN_RULES)
COMPILED_NEGATIVE = compile_rules(NEGATIVE_RULES)
HYDROGEN_PATTERNS = compile_terms(HYDROGEN_CONTEXT_TERMS)
GREEN_PATTERNS = compile_terms(GREEN_CONTEXT_TERMS)
EXPLICIT_PATTERNS = compile_terms(EXPLICIT_GREEN_TERMS)
INDEPENDENT_PATTERNS = compile_terms([t for r in GREEN_HYDROGEN_RULES if r["match_type"] == "core" for t in r["terms"]])
CONTEXTUAL_PATTERNS = compile_terms([t for r in GREEN_HYDROGEN_RULES if r["match_type"] != "core" for t in r["terms"]])


def _has(text, patterns):
    return any(p.search(text) for _, p in patterns)


def _union(patterns):
    return re.compile("|".join(p.pattern for _, p in patterns) or r"(?!)", re.I)


INDEPENDENT_REGEX, CONTEXTUAL_REGEX = _union(INDEPENDENT_PATTERNS), _union(CONTEXTUAL_PATTERNS)
HYDROGEN_REGEX, GREEN_REGEX = _union(HYDROGEN_PATTERNS), _union(GREEN_PATTERNS)


def _top(d):
    return max(d.items(), key=lambda x: (x[1], x[0]))[0] if d else ""


def score_one_patent_text(text: str) -> Dict:
    text = normalize_text(text)
    h_ctx, green_ctx, explicit = _has(text, HYDROGEN_PATTERNS), _has(text, GREEN_PATTERNS), _has(text, EXPLICIT_PATTERNS)
    strong = any(any(p.search(text) for _, p in r["patterns"]) for r in COMPILED_RULES if r["match_type"] == "core")
    matched, core_hits, context_hits, inactive = [], [], [], []
    term_scores, cats, subs, segments = {}, defaultdict(int), defaultdict(int), defaultdict(int)
    sources, raw, maximum = set(), 0, 0
    for r in COMPILED_RULES:
        hits = [t for t, p in r["patterns"] if p.search(text)]
        if not hits:
            continue
        valid = r["match_type"] == "core" or strong or (h_ctx and green_ctx)
        if not valid:
            inactive.extend(hits)
            continue
        s = int(r["score"]); raw += s; maximum = max(maximum, s)
        cats[r["category"]] += s; subs[(r["category"], r["sub_category"])] += s; segments[r["industry_segment"]] += s
        sources.add(r["source_type"]); matched.extend(hits)
        (core_hits if r["match_type"] == "core" else context_hits).extend(hits)
        for t in hits: term_scores[t] = max(term_scores.get(t, 0), s)
    negatives = [t for r in COMPILED_NEGATIVE for t, p in r["patterns"] if p.search(text)]
    score = maximum
    if negatives and not explicit:
        score = 0
    elif negatives:
        score = max(0, score - 1)
    main_cat = _top(cats)
    sub_pool = {k: v for k, v in subs.items() if k[0] == main_cat}
    main_sub = max(sub_pool.items(), key=lambda x: (x[1], x[0][1]))[0][1] if sub_pool else ""
    return {
        "green_hydrogen_score_raw": raw, "green_hydrogen_score": score, "core_score": score,
        "max_matched_keyword_score": maximum, "matched_terms": "；".join(sorted(set(matched))),
        "matched_core_terms": "；".join(sorted(set(core_hits))), "matched_context_terms": "；".join(sorted(set(context_hits))),
        "matched_term_scores": "；".join(f"{t}:{term_scores[t]}" for t in sorted(term_scores)),
        "inactive_terms_no_context": "；".join(sorted(set(inactive))), "negative_terms": "；".join(sorted(set(negatives))),
        "main_category": main_cat, "main_sub_category": main_sub, "industry_segment": _top(segments),
        "category_scores": json.dumps(dict(cats), ensure_ascii=False, sort_keys=True),
        "subcategory_scores": json.dumps({f"{a}/{b}": v for (a, b), v in subs.items()}, ensure_ascii=False, sort_keys=True),
        "source_types": "；".join(sorted(sources)), "has_hydrogen_context": int(h_ctx), "has_green_context": int(green_ctx),
    }


def make_text_series(df, cn_abs_col="摘要 (中文)", en_abs_col=None, extra_text_cols=None):
    cols = [c for c in ([cn_abs_col, en_abs_col] + list(extra_text_cols or [])) if c and c in df.columns]
    return (df[cols].fillna("").astype(str).agg(" ".join, axis=1).map(normalize_text) if cols
            else pd.Series([""] * len(df), index=df.index, dtype="object"))


def _join_unique(s):
    vals = [v.strip() for x in s.dropna().astype(str) for v in x.split("；") if v.strip()]
    return "；".join(sorted(set(vals)))


def _mode(s):
    vals = [str(v) for v in s.dropna() if str(v).strip()]
    return Counter(vals).most_common(1)[0][0] if vals else ""


def summarize_green_hydrogen_firms(patents, firm_col="第一申请人", year_col="year", region_col=None, firm_type_col=None):
    if patents.empty or firm_col not in patents.columns: return pd.DataFrame(), pd.DataFrame()
    d = patents[patents[firm_col].notna()].copy(); d[firm_col] = d[firm_col].astype(str).str.strip(); d = d[d[firm_col] != ""]
    regions = ([region_col] if isinstance(region_col, str) else list(region_col or [])); regions = [c for c in regions if c in d.columns]
    gy, gf = [firm_col] + regions + [year_col], [firm_col] + regions
    common = dict(green_hydrogen_patent_count=("is_green_hydrogen_patent", "sum"), green_hydrogen_score_sum=("green_hydrogen_score", "sum"),
                  green_hydrogen_score_mean=("green_hydrogen_score", "mean"), green_hydrogen_score_max=("green_hydrogen_score", "max"),
                  evidence_score_sum=("green_hydrogen_score_raw", "sum"), evidence_score_mean=("green_hydrogen_score_raw", "mean"),
                  main_categories=("main_category", _join_unique), main_sub_categories=("main_sub_category", _join_unique),
                  industry_segments=("industry_segment", _join_unique), matched_terms=("matched_terms", _join_unique))
    ya, fa = common.copy(), dict(first_year=(year_col, "min"), last_year=(year_col, "max"), **common)
    if firm_type_col and firm_type_col in d.columns: ya["first_applicant_types"] = (firm_type_col, _join_unique); fa["first_applicant_types"] = (firm_type_col, _join_unique)
    fy = d.groupby(gy, dropna=False).agg(**ya).reset_index(); f = d.groupby(gf, dropna=False).agg(**fa).reset_index()
    for src, dst in [("main_category", "firm_main_category"), ("main_sub_category", "firm_main_sub_category"), ("industry_segment", "firm_main_industry_segment")]:
        dom = d.groupby(gf, dropna=False)[src].agg(_mode).reset_index().rename(columns={src: dst}); f = f.merge(dom, on=gf, how="left")
    return fy, f


def tag_green_hydrogen_patents(df, cn_abs_col="摘要 (中文)", en_abs_col=None, firm_col="第一申请人", year_col="year",
                               region_col=None, firm_type_col=None, extra_text_cols=None, split_firms=False,
                               firm_sep_regex=r"[;；,，、|/]+", coarse_screen=True, progress_every=10000, min_score=1):
    data = df.copy(); texts = make_text_series(data, cn_abs_col, en_abs_col, extra_text_cols)
    if coarse_screen:
        mask = texts.str.contains(INDEPENDENT_REGEX, na=False) | (texts.str.contains(HYDROGEN_REGEX, na=False) & texts.str.contains(GREEN_REGEX, na=False) & texts.str.contains(CONTEXTUAL_REGEX, na=False))
        data, texts = data.loc[mask].copy(), texts.loc[mask]
    start, results = time.time(), []
    for i, text in enumerate(texts, 1):
        results.append(score_one_patent_text(text))
        if progress_every and (i % progress_every == 0 or i == len(data)):
            elapsed = time.time() - start; speed = i / elapsed if elapsed else 0; remain = (len(data)-i)/speed if speed else 0
            print(f"已处理 {i:,}/{len(data):,} 条候选，已用 {elapsed/60:.1f} 分钟，预计剩余 {remain/60:.1f} 分钟")
    scored = pd.DataFrame(results, index=data.index, columns=list(score_one_patent_text("").keys()))
    tagged = pd.concat([data, scored], axis=1); tagged["is_green_hydrogen_patent"] = (tagged["green_hydrogen_score"] >= min_score).astype(int)
    if split_firms and firm_col in tagged.columns:
        tagged["_firm"] = tagged[firm_col].fillna("").astype(str).str.split(firm_sep_regex); tagged = tagged.explode("_firm"); tagged[firm_col] = tagged["_firm"].str.strip(); tagged = tagged[tagged[firm_col] != ""].drop(columns="_firm")
    formal = tagged[tagged["is_green_hydrogen_patent"] == 1].copy()
    fy, f = summarize_green_hydrogen_firms(formal, firm_col, year_col, region_col, firm_type_col)
    return tagged, formal, fy, f


def export_keyword_dictionary():
    rows = []
    for r in GREEN_HYDROGEN_RULES:
        for t in r["terms"]:
            rows.append({"关键词": t, "技术领域": r["category"], "细分方向": r["sub_category"], "产业板块": r["industry_segment"],
                         "核心程度得分": r["score"], "匹配类型": r["match_type"],
                         "上下文要求": "无需上下文" if r["match_type"] == "core" else "需同时出现氢能与绿色能源上下文",
                         "来源类型": r["source_type"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print(f"绿色氢能词典包含 {len(export_keyword_dictionary()):,} 个关键词/缩写。")
