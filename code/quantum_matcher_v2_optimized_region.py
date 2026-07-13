# -*- coding: utf-8 -*-
"""
Quantum patent matcher v2 optimized
- Distinguishes core, conditional, supply_chain, application, time_frequency, negative terms.
- Generic optics/materials/application terms do not create quantum patents alone.
- Adds a conservative quantum metrology / time-frequency / PNT candidate pool.
- Removes overly broad time-frequency terms such as "时间尺度" and bare "原子时".
- Designed for Chinese patent title/abstract/claims DataFrame.
"""

import re
import time
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. Keyword rules
# ---------------------------------------------------------------------
# match_type:
#   core          : strong quantum term; can support relevance alone
#   conditional   : valid only when strong core context co-occurs
#   supply_chain  : used to identify upstream position only after core context exists
#   application   : application term; valid only with core context
#   time_frequency: atomic clock / time-frequency / PNT candidate; does not enter core pool by default
#   negative      : ambiguous/exclusion term; used for penalty / review

QUANTUM_RULES: List[Dict] = [
    # ---- core: general and computing ----
    {"category": "量子基础概念", "sub_category": "总称", "terms_cn": ["量子技术", "量子科技", "量子信息", "量子信息技术"], "terms_en": ["quantum technology", "quantum technologies", "quantum information"], "score": 5, "match_type": "core", "source_type": "PDF直接词"},
    {"category": "量子计算", "sub_category": "计算系统", "terms_cn": ["量子计算", "量子计算机", "量子处理器", "量子芯片", "量子计算平台"], "terms_en": ["quantum computing", "quantum computer", "quantum processor", "quantum chip", "quantum computing platform"], "score": 5, "match_type": "core", "source_type": "PDF直接词/PDF支持扩展"},
    {"category": "量子计算", "sub_category": "量子比特", "terms_cn": ["量子比特", "物理量子比特", "逻辑量子比特"], "terms_en": ["qubit", "qubits", "physical qubit", "logical qubit"], "score": 5, "match_type": "core", "source_type": "PDF直接词"},
    {"category": "量子计算", "sub_category": "量子门与线路", "terms_cn": ["量子门", "量子线路", "量子电路", "门保真度", "量子门保真度"], "terms_en": ["quantum gate", "quantum gates", "quantum circuit", "quantum circuits", "gate fidelity", "quantum gate fidelity"], "score": 5, "match_type": "core", "source_type": "PDF直接词"},
    {"category": "量子计算", "sub_category": "纠错与容错", "terms_cn": ["量子纠错", "量子误差校正", "量子错误校正", "容错量子计算", "可纠错量子计算"], "terms_en": ["quantum error correction", "fault-tolerant quantum computing", "error-corrected quantum computation"], "score": 5, "match_type": "core", "source_type": "PDF直接词/PDF支持扩展"},
    {"category": "量子计算", "sub_category": "量子模拟", "terms_cn": ["量子模拟", "量子仿真", "量子模拟器"], "terms_en": ["quantum simulation", "quantum simulator"], "score": 5, "match_type": "core", "source_type": "PDF直接词"},
    {"category": "量子计算", "sub_category": "算法", "terms_cn": ["Shor算法", "肖尔算法", "量子算法", "量子并行"], "terms_en": ["Shor's algorithm", "Shor algorithm", "quantum algorithm", "quantum algorithms", "quantum parallelism"], "score": 4, "match_type": "core", "source_type": "PDF直接词"},

    # ---- core: hardware routes ----
    {"category": "量子计算硬件", "sub_category": "离子阱", "terms_cn": ["离子阱", "俘获离子", "囚禁离子"], "terms_en": ["trapped ion", "trapped ions", "ion trap", "ion traps"], "score": 5, "match_type": "core", "source_type": "PDF直接词"},
    {"category": "量子计算硬件", "sub_category": "超导", "terms_cn": ["超导量子比特", "超导量子电路", "超导量子芯片"], "terms_en": ["superconducting qubit", "superconducting qubits"], "score": 5, "match_type": "core", "source_type": "PDF直接词/PDF支持扩展"},
    {"category": "量子计算硬件", "sub_category": "中性原子", "terms_cn": ["中性原子量子", "冷原子量子", "里德堡原子量子", "原子阵列量子"], "terms_en": ["neutral atom quantum", "Rydberg atom quantum", "atomic array quantum"], "score": 5, "match_type": "core", "source_type": "PDF直接词/PDF支持扩展"},
    {"category": "量子计算硬件", "sub_category": "硅自旋", "terms_cn": ["硅自旋量子比特", "自旋量子比特", "硅基量子比特"], "terms_en": ["silicon spin qubit", "spin qubit", "spin qubits"], "score": 4, "match_type": "core", "source_type": "PDF直接词/PDF支持扩展"},
    {"category": "量子计算硬件", "sub_category": "光量子", "terms_cn": ["光量子比特", "光子量子比特", "光量子计算", "光子量子计算"], "terms_en": ["photonic qubit", "photonic qubits", "photonic quantum computing"], "score": 4, "match_type": "core", "source_type": "PDF直接词/PDF支持扩展"},
    {"category": "量子计算硬件", "sub_category": "拓扑", "terms_cn": ["拓扑量子比特", "拓扑量子计算"], "terms_en": ["topological qubit", "topological qubits", "topological quantum computing"], "score": 4, "match_type": "core", "source_type": "PDF直接词/PDF支持扩展"},

    # ---- core: communication/security ----
    {"category": "量子通信与安全", "sub_category": "量子通信", "terms_cn": ["量子通信", "量子安全通信", "量子保密通信"], "terms_en": ["quantum communication", "quantum communications", "quantum-secure communication"], "score": 5, "match_type": "core", "source_type": "PDF直接词/PDF支持扩展"},
    {"category": "量子通信与安全", "sub_category": "QKD", "terms_cn": ["量子密钥分发", "量子密钥分配", "量子密钥"], "terms_en": ["quantum key distribution", "QKD"], "score": 5, "match_type": "core", "source_type": "PDF直接词"},
    {"category": "量子通信与安全", "sub_category": "量子网络", "terms_cn": ["量子网络", "量子互联网", "量子中继", "量子中继器", "量子转导", "分布式量子计算"], "terms_en": ["quantum network", "quantum networking", "quantum internet", "quantum repeater", "quantum repeaters", "quantum transduction", "distributed quantum computing"], "score": 5, "match_type": "core", "source_type": "PDF直接词/PDF支持扩展"},
    {"category": "量子通信与安全", "sub_category": "后量子密码", "terms_cn": ["后量子密码", "后量子加密", "抗量子密码", "抗量子加密", "抗量子攻击"], "terms_en": ["post-quantum cryptography", "post-quantum encryption", "quantum-resistant cryptography", "quantum-resistant encryption"], "score": 5, "match_type": "core", "source_type": "PDF直接词/PDF支持扩展"},

    # ---- core: sensing ----
    {"category": "量子传感", "sub_category": "总称", "terms_cn": ["量子传感", "量子传感器", "量子测量", "量子精密测量", "量子磁力计"], "terms_en": ["quantum sensing", "quantum sensor", "quantum sensors", "quantum measurement", "quantum metrology", "quantum magnetometer"], "score": 5, "match_type": "core", "source_type": "PDF直接词/PDF支持扩展"},
    {"category": "量子传感", "sub_category": "NV色心", "terms_cn": ["金刚石NV色心", "NV色心"], "terms_en": ["NV center", "nitrogen-vacancy center"], "score": 5, "match_type": "core", "source_type": "专家补充词"},
    {"category": "量子传感", "sub_category": "冷原子干涉", "terms_cn": ["冷原子干涉仪", "冷原子干涉", "原子干涉仪", "原子干涉", "磁光阱", "磁光阱真空腔", "MOT"], "terms_en": ["cold-atom interferometer", "cold atom interferometer", "atom interferometer", "atomic interferometer", "matter-wave interferometer", "magneto-optical trap", "MOT"], "score": 5, "match_type": "core", "source_type": "专家补充词"},
    {"category": "量子传感", "sub_category": "时间频率/PNT", "terms_cn": ["氢原子钟", "铷原子钟", "铯原子钟", "冷原子钟", "光钟", "原子钟", "原子钟频率", "原子钟频率驾驭", "原子钟联合守时", "原子钟守时", "原子钟授时", "原子钟时间同步", "原子钟频标", "光学频率传递", "光频传递", "光纤频率传递", "时间频率传递", "时间频率传输", "时间频率比对", "频率驾驭", "频率标准", "频率基准", "原子频标", "原子时标", "国际原子时", "原子时间尺度"], "terms_en": ["hydrogen maser", "rubidium atomic clock", "cesium atomic clock", "caesium atomic clock", "cold atom clock", "optical atomic clock", "optical clock", "atomic clock", "atomic frequency standard", "optical frequency transfer", "optical frequency dissemination", "time-frequency transfer", "time frequency transfer", "time-frequency comparison", "atomic time scale", "international atomic time"], "score": 3, "match_type": "time_frequency", "source_type": "专家补充词", "note": "广义量子传感/量子计量/PNT候选；默认不进入核心量子企业池；不使用裸词'时间尺度'和'原子时'以降低误判"},

    # ---- conditional: bare scientific words ----
    {"category": "量子基础概念", "sub_category": "基本原理", "terms_cn": ["量子叠加", "叠加态", "量子纠缠", "量子态", "量子相干"], "terms_en": ["quantum superposition", "quantum entanglement", "quantum state", "quantum coherence"], "score": 4, "match_type": "core", "source_type": "PDF直接词"},
    {"category": "量子基础概念", "sub_category": "需共现基本原理", "terms_cn": [], "terms_en": ["superposition", "entanglement"], "score": 2, "match_type": "conditional", "source_type": "PDF直接词", "note": "bare English word is too broad; requires strong quantum context"},
    {"category": "量子计算", "sub_category": "需共现保真度", "terms_cn": ["保真度"], "terms_en": ["fidelity"], "score": 2, "match_type": "conditional", "source_type": "PDF直接词", "note": "only valid with qubit/quantum gate/quantum circuit context"},
    {"category": "量子传感", "sub_category": "原子钟", "terms_cn": ["原子钟", "光钟", "便携式原子钟", "量子钟"], "terms_en": ["atomic clock", "atomic clocks", "optical clock"], "score": 2, "match_type": "conditional", "source_type": "PDF直接词/PDF支持扩展"},

    # ---- supply chain: do not identify quantum alone ----
    {"category": "上游材料与核心部件", "sub_category": "低温与真空", "terms_cn": ["稀释制冷机", "低温制冷机", "低温恒温器", "超高真空"], "terms_en": ["dilution refrigerator", "cryogenic refrigerator", "cryostat", "ultra-high vacuum"], "score": 3, "match_type": "supply_chain", "source_type": "PDF直接词/PDF支持扩展"},
    {"category": "上游材料与核心部件", "sub_category": "光电器件", "terms_cn": ["单光子探测器", "集成量子光子", "集成量子光子学", "微波-光学转导", "微波光学转导"], "terms_en": ["single photon detector", "single-photon detector", "integrated quantum photonics", "microwave-to-optical transduction", "microwave optical transduction"], "score": 3, "match_type": "supply_chain", "source_type": "PDF直接词/PDF支持扩展"},
    {"category": "上游材料与核心部件", "sub_category": "泛光电器件", "terms_cn": ["激光器", "光调制器", "电光晶体", "声光晶体"], "terms_en": ["laser", "lasers", "optical modulator", "optical modulators", "electro-optic crystal", "acousto-optic crystal"], "score": 1, "match_type": "supply_chain", "source_type": "PDF直接词/PDF支持扩展", "note": "very broad; only records supply-chain evidence when core quantum context exists"},
    {"category": "上游材料与核心部件", "sub_category": "关键材料", "terms_cn": ["高纯铝", "稀土金属"], "terms_en": ["high purity aluminum", "rare earth metals"], "score": 1, "match_type": "supply_chain", "source_type": "PDF直接词"},

    # ---- applications: valid only with core quantum context ----
    {"category": "量子应用", "sub_category": "密码与安全", "terms_cn": ["公钥加密破解", "非对称密码", "大数分解", "RSA破解", "RSA-2048"], "terms_en": ["public-key encryption", "asymmetric cryptography", "factoring large numbers", "RSA-2048"], "score": 2, "match_type": "application", "source_type": "PDF直接词/PDF支持扩展"},
    {"category": "量子应用", "sub_category": "材料与化学", "terms_cn": ["量子化学", "材料模拟", "新材料设计", "高温超导", "催化剂设计"], "terms_en": ["quantum chemistry", "materials simulation", "materials design", "high-temperature superconductivity", "catalyst design"], "score": 2, "match_type": "application", "source_type": "PDF直接词/PDF支持扩展"},
    {"category": "量子应用", "sub_category": "医药与生命科学", "terms_cn": ["药物发现", "蛋白质相互作用", "溶剂相互作用"], "terms_en": ["drug discovery", "protein interactions", "solvent interactions"], "score": 1, "match_type": "application", "source_type": "PDF直接词/PDF支持扩展"},
    {"category": "量子应用", "sub_category": "金融与机器学习", "terms_cn": ["量子机器学习", "量子优化", "投资组合优化", "金融组合优化"], "terms_en": ["quantum machine learning", "quantum optimization", "portfolio optimization", "financial portfolio optimization"], "score": 1, "match_type": "application", "source_type": "PDF直接词/PDF支持扩展"},
]

NEGATIVE_RULES: List[Dict] = [
    {"category": "易混淆材料词", "terms_cn": ["量子点", "量子点发光", "量子点显示", "量子点膜"], "terms_en": ["quantum dot", "quantum dots", "quantum dot display"], "penalty": 4},
    {"category": "易混淆半导体词", "terms_cn": ["量子阱", "多量子阱", "量子线"], "terms_en": ["quantum well", "quantum wells", "multiple quantum well", "quantum wire"], "penalty": 4},
    {"category": "易混淆光电指标", "terms_cn": ["量子效率", "外量子效率", "内量子效率", "量子产率"], "terms_en": ["quantum efficiency", "external quantum efficiency", "internal quantum efficiency", "quantum yield"], "penalty": 4},
    {"category": "英文修辞排除", "terms_cn": [], "terms_en": ["quantum leap"], "penalty": 8},
]

STRONG_CONTEXT_TERMS_CN = [
    "量子技术", "量子科技", "量子信息", "量子计算", "量子计算机", "量子处理器", "量子芯片", "量子比特",
    "量子门", "量子线路", "量子电路", "量子纠错", "量子模拟", "量子仿真", "量子算法",
    "量子通信", "量子密钥", "量子网络", "量子互联网", "量子中继", "量子转导", "后量子", "抗量子",
    "量子传感", "量子测量", "量子精密测量", "量子磁力计", "金刚石NV色心", "NV色心",
    "离子阱", "俘获离子", "囚禁离子", "超导量子", "光量子", "光子量子", "拓扑量子",
    "冷原子干涉仪", "冷原子干涉", "原子干涉仪", "原子干涉", "磁光阱", "磁光阱真空腔", "金刚石NV", "NV色心",
]
STRONG_CONTEXT_TERMS_EN = [
    "quantum technology", "quantum information", "quantum computing", "quantum computer", "quantum processor",
    "quantum chip", "qubit", "quantum gate", "quantum circuit", "quantum error correction", "quantum simulation",
    "quantum algorithm", "quantum communication", "quantum key", "QKD", "quantum network", "quantum internet",
    "quantum repeater", "quantum transduction", "post-quantum", "quantum-resistant", "quantum sensing", "quantum sensor",
    "quantum measurement", "quantum metrology", "NV center", "nitrogen-vacancy center", "trapped ion", "ion trap",
    "superconducting qubit", "photonic qubit", "topological qubit",
    "cold atom interferometer", "cold-atom interferometer", "atom interferometer", "atomic interferometer", "matter-wave interferometer", "magneto-optical trap",
]

TIMEFREQ_CONTEXT_TERMS_CN = [
    "氢原子钟", "铷原子钟", "铯原子钟", "冷原子钟", "光钟", "原子钟",
    "原子钟频率", "原子钟频率驾驭", "原子钟联合守时", "原子钟守时", "原子钟授时", "原子钟时间同步", "原子钟频标",
    "光学频率传递", "光频传递", "光纤频率传递", "时间频率传递", "时间频率传输", "时间频率比对",
    "频率驾驭", "频率标准", "频率基准", "原子频标", "原子时标", "国际原子时", "原子时间尺度",
]
TIMEFREQ_CONTEXT_TERMS_EN = [
    "hydrogen maser", "rubidium atomic clock", "cesium atomic clock", "caesium atomic clock", "cold atom clock",
    "optical atomic clock", "optical clock", "atomic clock", "atomic frequency standard",
    "optical frequency transfer", "optical frequency dissemination", "time-frequency transfer", "time frequency transfer",
    "time-frequency comparison", "atomic time scale", "international atomic time",
]


# ---------------------------------------------------------------------
# 2. Regex helpers
# ---------------------------------------------------------------------
def safe_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def normalize_text(text: str) -> str:
    text = safe_text(text).lower()
    text = text.replace("－", "-").replace("—", "-").replace("–", "-")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("，", ",").replace("；", ";")
    return text


def term_to_regex(term: str) -> Optional[str]:
    term = term.strip()
    if not term:
        return None
    has_cn = bool(re.search(r"[\u4e00-\u9fff]", term))
    if has_cn:
        return re.escape(term.lower())
    t = re.escape(term.lower())
    t = t.replace(r"\ ", r"[\s\-]+")
    t = t.replace(r"\-", r"[\s\-]+")
    return r"(?<![a-z0-9])" + t + r"(?![a-z0-9])"


def compile_terms(terms: List[str]) -> List[Tuple[str, re.Pattern]]:
    out = []
    for term in terms:
        rgx = term_to_regex(term)
        if rgx:
            out.append((term, re.compile(rgx, flags=re.IGNORECASE)))
    return out


def compile_rules(rules: List[Dict]) -> List[Dict]:
    compiled = []
    for item in rules:
        new = item.copy()
        new["patterns"] = compile_terms(item.get("terms_cn", []) + item.get("terms_en", []))
        compiled.append(new)
    return compiled


COMPILED_RULES = compile_rules(QUANTUM_RULES)
COMPILED_NEGATIVE = compile_rules(NEGATIVE_RULES)
CONTEXT_PATTERNS = compile_terms(STRONG_CONTEXT_TERMS_CN + STRONG_CONTEXT_TERMS_EN)
TIMEFREQ_PATTERNS = compile_terms(TIMEFREQ_CONTEXT_TERMS_CN + TIMEFREQ_CONTEXT_TERMS_EN)
COARSE_REGEX = re.compile("|".join([p.pattern for _, p in CONTEXT_PATTERNS]), flags=re.IGNORECASE)
COARSE_TIMEFREQ_REGEX = re.compile("|".join([p.pattern for _, p in TIMEFREQ_PATTERNS]), flags=re.IGNORECASE)


def has_strong_context(text: str) -> bool:
    return any(p.search(text) for _, p in CONTEXT_PATTERNS)


def build_patent_text(row: pd.Series, cn_abs_col="摘要 (中文)", en_abs_col="摘要 (英文)", extra_cols=None) -> str:
    parts = [safe_text(row.get(cn_abs_col, "")), safe_text(row.get(en_abs_col, ""))]
    if extra_cols:
        for c in extra_cols:
            parts.append(safe_text(row.get(c, "")))
    return normalize_text(" ".join(parts))


# ---------------------------------------------------------------------
# 3. Scoring
# ---------------------------------------------------------------------
def infer_chain_position(category_scores: Dict[str, int], subcategory_scores: Dict[Tuple[str, str], int], matched_core_terms: List[str]) -> str:
    cats = {c for c, s in category_scores.items() if s > 0}
    subcats = {sub for (cat, sub), s in subcategory_scores.items() if s > 0}

    # prioritize actual quantum application categories over generic supply-chain evidence
    if "时间频率/PNT" in subcats:
        # Atomic clock / time-frequency patents are kept as a separate broad quantum metrology candidate pool.
        if not any(s in subcats for s in ["总称", "NV色心", "冷原子干涉"]):
            return "量子计量/时间频率/PNT候选"
    if "量子通信与安全" in cats:
        return "量子通信与安全"
    if "量子传感" in cats:
        return "量子传感器/精密测量"
    if any(s in subcats for s in ["离子阱", "超导", "中性原子", "硅自旋", "光量子", "拓扑"]):
        return "量子计算硬件/量子芯片"
    if any(s in subcats for s in ["纠错与容错", "算法", "量子模拟"]):
        return "量子软件/算法/模拟"
    if "量子计算" in cats:
        return "量子计算系统"
    if "量子应用" in cats:
        return "量子应用场景"
    if "上游材料与核心部件" in cats and matched_core_terms:
        return "上游材料与核心部件"
    return "其他量子相关/待复核"


def infer_main_category(category_scores: Dict[str, int], subcategory_scores: Dict[Tuple[str, str], int]) -> str:
    """Use domain priority rather than pure max score to avoid QKD/quantum sensing being overridden by generic terms."""
    if not category_scores:
        return ""
    subcats = {sub for (cat, sub), s in subcategory_scores.items() if s > 0}
    cats = {cat for cat, s in category_scores.items() if s > 0}

    if any(s in subcats for s in ["QKD", "量子通信", "量子网络", "后量子密码"]):
        return "量子通信与安全"
    if any(s in subcats for s in ["NV色心", "冷原子干涉", "时间频率/PNT", "总称", "原子钟"]):
        return "量子传感"
    if any(s in subcats for s in ["离子阱", "超导", "中性原子", "硅自旋", "光量子", "拓扑"]):
        return "量子计算硬件"
    if any(s in subcats for s in ["纠错与容错", "算法", "量子模拟"]):
        return "量子计算"
    if "量子计算" in cats:
        return "量子计算"
    if "量子基础概念" in cats:
        return "量子基础概念"
    if "量子应用" in cats:
        return "量子应用"
    if "上游材料与核心部件" in cats:
        return "上游材料与核心部件"
    return max(category_scores.items(), key=lambda x: x[1])[0]


def score_one_patent_text(text: str) -> Dict:
    text = normalize_text(text)
    strong_context = has_strong_context(text)

    matched_core_terms, matched_conditional_terms, matched_supply_terms, matched_application_terms = [], [], [], []
    matched_timefreq_terms = []
    inactive_terms, negative_terms = [], []
    category_scores = defaultdict(int)
    subcategory_scores = defaultdict(int)
    source_types = set()
    penalty = 0
    core_score_value = 0

    for item in COMPILED_RULES:
        hits = [term for term, pattern in item["patterns"] if pattern.search(text)]
        if not hits:
            continue
        mt = item.get("match_type", "core")
        valid = (mt == "core") or (mt == "time_frequency") or (mt in {"conditional", "supply_chain", "application"} and strong_context)

        if not valid:
            inactive_terms.extend(hits)
            continue

        score = item.get("score", 0)
        category = item.get("category", "")
        sub_category = item.get("sub_category", "")
        category_scores[category] += score
        subcategory_scores[(category, sub_category)] += score
        source_types.add(item.get("source_type", ""))

        if mt == "core":
            matched_core_terms.extend(hits)
            core_score_value += score
        elif mt == "conditional":
            matched_conditional_terms.extend(hits)
        elif mt == "supply_chain":
            matched_supply_terms.extend(hits)
        elif mt == "application":
            matched_application_terms.extend(hits)
        elif mt == "time_frequency":
            matched_timefreq_terms.extend(hits)

    for item in COMPILED_NEGATIVE:
        hits = [term for term, pattern in item["patterns"] if pattern.search(text)]
        if hits:
            negative_terms.extend(hits)
            penalty += item.get("penalty", 0)

    core_score = core_score_value
    total_score_raw = sum(category_scores.values())
    total_score = max(0, total_score_raw - (penalty if not matched_core_terms else max(0, penalty - 2)))

    main_category = infer_main_category(category_scores, subcategory_scores)

    chain_position = infer_chain_position(category_scores, subcategory_scores, matched_core_terms)

    core_count = len(set(matched_core_terms))
    if core_count >= 2 and total_score >= 8:
        relevance = "高相关"
    elif core_count >= 1 and total_score >= 5:
        relevance = "中相关"
    elif core_count >= 1:
        relevance = "低相关/待复核"
    elif matched_timefreq_terms:
        relevance = "量子计量/PNT候选"
    elif matched_supply_terms or matched_application_terms or matched_conditional_terms:
        relevance = "供应链/应用待复核"
    elif negative_terms:
        relevance = "不相关/易误判"
    else:
        relevance = "不相关"

    all_terms = matched_core_terms + matched_conditional_terms + matched_supply_terms + matched_application_terms + matched_timefreq_terms
    return {
        "quantum_score_raw": total_score_raw,
        "quantum_score": total_score,
        "core_score": core_score,
        "matched_terms": "；".join(sorted(set(all_terms))),
        "matched_core_terms": "；".join(sorted(set(matched_core_terms))),
        "matched_conditional_terms": "；".join(sorted(set(matched_conditional_terms))),
        "matched_supply_terms": "；".join(sorted(set(matched_supply_terms))),
        "matched_application_terms": "；".join(sorted(set(matched_application_terms))),
        "matched_timefreq_terms": "；".join(sorted(set(matched_timefreq_terms))),
        "inactive_terms_no_context": "；".join(sorted(set(inactive_terms))),
        "negative_terms": "；".join(sorted(set(negative_terms))),
        "main_category": main_category,
        "chain_position": chain_position,
        "category_scores": dict(category_scores),
        "source_types": "；".join(sorted(x for x in source_types if x)),
        "relevance": relevance,
    }


# ---------------------------------------------------------------------
# 4. DataFrame interface
# ---------------------------------------------------------------------
def make_text_series(df: pd.DataFrame, cn_abs_col="摘要 (中文)", en_abs_col="摘要 (英文)", extra_text_cols=None) -> pd.Series:
    cols = [cn_abs_col, en_abs_col]
    if extra_text_cols:
        cols += extra_text_cols
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series([""] * len(df), index=df.index)
    return df[existing].fillna("").astype(str).agg(" ".join, axis=1).map(normalize_text)


def tag_quantum_patents(
    df: pd.DataFrame,
    cn_abs_col="摘要 (中文)",
    en_abs_col="摘要 (英文)",
    firm_col="第一申请人",
    year_col="year",
    region_col=None,
    extra_text_cols=None,
    split_firms=False,
    firm_sep_regex=r"[;；,，、|/]+",
    coarse_screen=True,
    progress_every=10000,
    include_review=False,
    include_timefreq=False,
):
    """
    Returns: patent_tagged, quantum_patents, firm_year_quantum, firm_quantum.
    By default quantum_patents includes only high/medium/low core-related patents.
    Set include_review=True to include supply-chain/application review cases.
    Set include_timefreq=True to include quantum metrology / time-frequency / PNT candidates in quantum_patents and firm summaries.
    Set region_col="申请人省市代码" to make firm_year_quantum grouped by firm-region-year and firm_quantum grouped by firm-region.
    """
    data = df.copy()
    text_series = make_text_series(data, cn_abs_col, en_abs_col, extra_text_cols)

    if coarse_screen:
        # Keep both strict quantum-core candidates and broader atomic-clock/time-frequency candidates.
        # Time-frequency candidates are NOT counted as formal quantum patents unless include_timefreq=True.
        core_mask = text_series.str.contains(COARSE_REGEX, regex=True, na=False)
        timefreq_mask = text_series.str.contains(COARSE_TIMEFREQ_REGEX, regex=True, na=False)
        candidate_mask = core_mask | timefreq_mask
        data = data.loc[candidate_mask].copy()
        text_series = text_series.loc[candidate_mask]

    total_n = len(data)
    results = []
    start = time.time()
    for i, text in enumerate(text_series, start=1):
        results.append(score_one_patent_text(text))
        if progress_every and (i % progress_every == 0 or i == total_n):
            elapsed = time.time() - start
            speed = i / elapsed if elapsed > 0 else 0
            remain = (total_n - i) / speed if speed > 0 else 0
            print(f"已处理 {i:,}/{total_n:,} 条，占比 {i/total_n:.2%}，已用 {elapsed/60:.1f} 分钟，预计剩余 {remain/60:.1f} 分钟")

    scores = pd.DataFrame(results, index=data.index)
    patent_tagged = pd.concat([data, scores], axis=1)

    relevant_labels = ["高相关", "中相关", "低相关/待复核"]
    if include_review:
        relevant_labels.append("供应链/应用待复核")
    if include_timefreq:
        relevant_labels.append("量子计量/PNT候选")
    patent_tagged["is_quantum_patent"] = patent_tagged["relevance"].isin(relevant_labels).astype(int)

    if split_firms and firm_col in patent_tagged.columns:
        patent_tagged[firm_col] = patent_tagged[firm_col].fillna("").astype(str)
        patent_tagged["_firm_list"] = patent_tagged[firm_col].str.split(firm_sep_regex)
        patent_tagged = patent_tagged.explode("_firm_list")
        patent_tagged[firm_col] = patent_tagged["_firm_list"].str.strip()
        patent_tagged = patent_tagged[patent_tagged[firm_col] != ""].copy()
        patent_tagged.drop(columns=["_firm_list"], inplace=True)

    quantum_patents = patent_tagged[patent_tagged["is_quantum_patent"] == 1].copy()

    if quantum_patents.empty or firm_col not in quantum_patents.columns:
        return patent_tagged, quantum_patents, pd.DataFrame(), pd.DataFrame()

    # 企业所在地口径：
    # 如果传入 region_col="申请人省市代码"，则按“第一申请人-所在地-年份”分组。
    # 这样同一个第一申请人在不同地区出现时，会被分开计算。
    group_cols_year = [firm_col, year_col]
    group_cols_firm = [firm_col]
    if region_col and region_col in quantum_patents.columns:
        group_cols_year = [firm_col, region_col, year_col]
        group_cols_firm = [firm_col, region_col]

    def join_unique(x):
        vals = []
        for s in x.dropna().astype(str):
            vals.extend([v for v in s.split("；") if v])
        return "；".join(sorted(set(vals)))

    firm_year_quantum = quantum_patents.groupby(group_cols_year, dropna=False).agg(
        quantum_patent_count=("is_quantum_patent", "sum"),
        quantum_score_sum=("quantum_score", "sum"),
        quantum_score_mean=("quantum_score", "mean"),
        main_categories=("main_category", lambda x: "；".join(sorted(set(x.dropna())))),
        chain_positions=("chain_position", lambda x: "；".join(sorted(set(x.dropna())))),
        matched_terms=("matched_terms", join_unique),
        high_related_count=("relevance", lambda x: (x == "高相关").sum()),
        medium_related_count=("relevance", lambda x: (x == "中相关").sum()),
        low_related_count=("relevance", lambda x: (x == "低相关/待复核").sum()),
        timefreq_candidate_count=("relevance", lambda x: (x == "量子计量/PNT候选").sum()),
    ).reset_index()

    firm_quantum = quantum_patents.groupby(group_cols_firm, dropna=False).agg(
        first_year=(year_col, "min"),
        last_year=(year_col, "max"),
        quantum_patent_count=("is_quantum_patent", "sum"),
        quantum_score_sum=("quantum_score", "sum"),
        quantum_score_mean=("quantum_score", "mean"),
        main_categories=("main_category", lambda x: "；".join(sorted(set(x.dropna())))),
        chain_positions=("chain_position", lambda x: "；".join(sorted(set(x.dropna())))),
        matched_terms=("matched_terms", join_unique),
        high_related_count=("relevance", lambda x: (x == "高相关").sum()),
        medium_related_count=("relevance", lambda x: (x == "中相关").sum()),
        low_related_count=("relevance", lambda x: (x == "低相关/待复核").sum()),
        timefreq_candidate_count=("relevance", lambda x: (x == "量子计量/PNT候选").sum()),
    ).reset_index()

    firm_main_position = quantum_patents.groupby(group_cols_firm)["chain_position"].agg(lambda x: Counter([i for i in x if i]).most_common(1)[0][0]).reset_index().rename(columns={"chain_position": "main_chain_position"})
    firm_main_category = quantum_patents.groupby(group_cols_firm)["main_category"].agg(lambda x: Counter([i for i in x if i]).most_common(1)[0][0]).reset_index().rename(columns={"main_category": "main_quantum_category"})
    firm_quantum = firm_quantum.merge(firm_main_position, on=group_cols_firm, how="left").merge(firm_main_category, on=group_cols_firm, how="left")

    firm_quantum["firm_relevance"] = np.select(
        [
            (firm_quantum["high_related_count"] >= 2) | (firm_quantum["quantum_score_sum"] >= 20),
            (firm_quantum["high_related_count"] >= 1) | (firm_quantum["quantum_score_sum"] >= 10),
            (firm_quantum["medium_related_count"] >= 1) | (firm_quantum["quantum_patent_count"] >= 1),
        ],
        ["核心量子企业", "较高相关企业", "候选/待复核企业"],
        default="不相关",
    )

    return patent_tagged, quantum_patents, firm_year_quantum, firm_quantum


if __name__ == "__main__":
    print("Import this module and call tag_quantum_patents(df, ...).")
