# -*- coding: utf-8 -*-
"""
Quantum information patent tagging.

The module keeps the original public function names while adding:
- match_type: core / conditional / supply_chain / negative
- co-occurrence rules for generic terms
- progress reporting for large DataFrames
- patent-level hit columns by match type
- firm-level dominant quantum category and chain position
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


CORE_COOCCUR_TERMS = [
    "量子", "量子信息", "量子科技", "量子技术", "量子计算", "量子通信", "量子密钥",
    "量子传感", "量子测量", "量子精密测量", "量子态", "量子纠缠", "量子叠加",
    "量子比特", "量子芯片", "量子门", "量子线路", "量子电路", "量子纠错",
    "后量子", "抗量子", "单光子", "光量子", "离子阱", "俘获离子", "囚禁离子",
    "中性原子", "里德堡原子", "冷原子", "超导量子", "自旋量子比特",
    "nv色心", "金刚石nv色心", "量子磁力计", "原子干涉仪",
    "quantum", "qubit", "qubits", "qkd", "post-quantum", "post quantum",
    "quantum-resistant", "quantum resistant", "single photon", "trapped ion",
    "ion trap", "neutral atom", "rydberg", "superconducting qubit",
    "spin qubit", "nv center", "nitrogen-vacancy",
]


def kw(
    category,
    sub_category,
    terms_cn=None,
    terms_en=None,
    score=0,
    source_type="专家规则",
    match_type="conditional",
    cooccur_required=False,
    cooccur_terms=None,
    penalty=0,
    note="",
):
    return {
        "category": category,
        "sub_category": sub_category,
        "terms_cn": terms_cn or [],
        "terms_en": terms_en or [],
        "score": score,
        "source_type": source_type,
        "match_type": match_type,
        "cooccur_required": cooccur_required,
        "cooccur_terms": cooccur_terms if cooccur_terms is not None else CORE_COOCCUR_TERMS,
        "penalty": penalty,
        "note": note,
    }


QUANTUM_KEYWORDS = [
    kw(
        "量子基础概念", "总称",
        ["量子技术", "量子科技", "量子信息", "量子信息技术"],
        ["quantum technology", "quantum technologies", "quantum information"],
        5, "政策/报告核心词", "core", False,
        note="直接指向量子信息技术体系。",
    ),
    kw(
        "量子基础概念", "基本原理",
        ["量子叠加", "叠加态", "量子纠缠", "量子态", "量子相干"],
        ["quantum superposition", "quantum entanglement", "quantum state", "quantum coherence"],
        4, "技术核心词", "core", False,
        note="带量子限定的基本原理词可直接计分。",
    ),
    kw(
        "量子基础概念", "英文泛化原理",
        [],
        ["superposition", "entanglement", "coherence"],
        2, "易泛化英文词", "conditional", True,
        note="英文单词常用于数学、信号、材料语境，需与核心词共现。",
    ),
    kw(
        "量子计算", "计算系统",
        ["量子计算", "量子计算机", "量子处理器", "量子芯片", "量子计算平台"],
        ["quantum computing", "quantum computer", "quantum processor", "quantum chip", "quantum computing platform"],
        5, "技术核心词", "core", False,
        note="直接指向量子计算整机、平台或处理器。",
    ),
    kw(
        "量子计算", "量子比特",
        ["量子比特", "物理量子比特", "逻辑量子比特"],
        ["qubit", "qubits", "physical qubit", "logical qubit"],
        5, "技术核心词", "core", False,
        note="量子计算最核心对象。",
    ),
    kw(
        "量子计算", "量子门与线路",
        ["量子门", "量子线路", "量子电路", "门保真度"],
        ["quantum gate", "quantum gates", "quantum circuit", "quantum circuits", "gate fidelity"],
        5, "技术核心词", "core", False,
        note="直接指向量子线路与门操作。",
    ),
    kw(
        "量子计算", "保真度",
        ["保真度"],
        ["fidelity"],
        2, "条件词", "conditional", True,
        note="保真度也常用于通信、音视频、建模，需与量子计算/量子态等共现。",
    ),
    kw(
        "量子计算", "纠错与容错",
        ["量子纠错", "量子误差校正", "量子错误校正", "容错量子计算", "可纠错量子计算"],
        ["quantum error correction", "fault-tolerant quantum computing", "error-corrected quantum computation"],
        5, "技术核心词", "core", False,
        note="直接指向可扩展量子计算关键技术。",
    ),
    kw(
        "量子计算", "纠错泛化词",
        ["误差校正", "错误校正"],
        ["error correction"],
        2, "条件词", "conditional", True,
        note="通用工程术语，需与量子核心词共现。",
    ),
    kw(
        "量子计算", "量子存储",
        ["量子存储", "量子内存", "量子记忆"],
        ["quantum memory", "quantum memories"],
        4, "技术核心词", "core", False,
        note="量子网络/计算关键模块。",
    ),
    kw(
        "量子计算", "量子模拟",
        ["量子模拟", "量子仿真", "量子模拟器"],
        ["quantum simulation", "quantum simulator", "quantum simulators"],
        5, "技术核心词", "core", False,
        note="直接指向量子计算应用或专用量子模拟器。",
    ),
    kw(
        "量子计算", "算法",
        ["Shor算法", "肖尔算法", "量子算法", "量子并行"],
        ["Shor's algorithm", "Shor algorithm", "quantum algorithm", "quantum algorithms", "quantum parallelism"],
        4, "技术核心词", "core", False,
        note="明确量子算法词。",
    ),
    kw(
        "量子计算硬件", "离子阱",
        ["离子阱", "俘获离子", "囚禁离子"],
        ["trapped ion", "trapped ions", "ion trap", "ion traps"],
        5, "硬件路线核心词", "core", False,
        note="主流量子计算硬件路线。",
    ),
    kw(
        "量子计算硬件", "超导",
        ["超导量子比特", "超导量子电路", "超导量子芯片"],
        ["superconducting qubit", "superconducting qubits"],
        5, "硬件路线核心词", "core", False,
        note="主流量子计算硬件路线。",
    ),
    kw(
        "量子计算硬件", "超导电路",
        ["超导电路"],
        ["superconducting circuit", "superconducting circuits"],
        3, "条件硬件词", "conditional", True,
        note="可能是普通超导电子电路，需与量子/量子比特共现。",
    ),
    kw(
        "量子计算硬件", "中性原子",
        ["中性原子", "原子阵列", "里德堡原子", "冷原子阵列"],
        ["neutral atom", "neutral atoms", "atom array", "atomic array", "Rydberg atom", "Rydberg atoms"],
        4, "硬件路线词", "conditional", True,
        note="冷原子/原子阵列也可能用于基础物理仪器，需量子语境共现。",
    ),
    kw(
        "量子计算硬件", "硅自旋",
        ["硅自旋量子比特", "自旋量子比特", "硅基量子比特"],
        ["silicon spin qubit", "silicon spin qubits", "spin qubit", "spin qubits"],
        5, "硬件路线核心词", "core", False,
        note="明确量子比特路线。",
    ),
    kw(
        "量子计算硬件", "光量子",
        ["光量子比特", "光子量子比特", "光量子计算", "光子量子计算"],
        ["photonic qubit", "photonic qubits", "photonic quantum computing", "optical qubit", "optical qubits"],
        5, "硬件路线核心词", "core", False,
        note="明确光量子计算。",
    ),
    kw(
        "量子计算硬件", "拓扑",
        ["拓扑量子比特", "拓扑量子计算"],
        ["topological qubit", "topological qubits", "topological quantum computing"],
        4, "硬件路线核心词", "core", False,
        note="明确拓扑量子计算。",
    ),
    kw(
        "量子通信与安全", "量子通信",
        ["量子通信", "量子安全通信", "量子保密通信"],
        ["quantum communication", "quantum communications", "quantum-secure communication"],
        5, "技术核心词", "core", False,
        note="直接指向量子通信产业方向。",
    ),
    kw(
        "量子通信与安全", "QKD",
        ["量子密钥分发", "量子密钥分配", "量子密钥"],
        ["quantum key distribution", "QKD"],
        5, "技术核心词", "core", False,
        note="量子通信最典型产业化方向。",
    ),
    kw(
        "量子通信与安全", "量子网络",
        ["量子网络", "量子互联网", "量子数据中心", "分布式量子计算"],
        ["quantum network", "quantum networking", "quantum internet", "quantum data center", "distributed quantum computing"],
        5, "技术核心词", "core", False,
        note="直接指向量子网络。",
    ),
    kw(
        "量子通信与安全", "量子中继与转导",
        ["量子中继", "量子中继器", "量子转导", "微波光学转导", "微波-光学转导"],
        ["quantum repeater", "quantum repeaters", "quantum transduction", "microwave-to-optical transduction"],
        5, "技术核心词", "core", False,
        note="量子网络关键环节。",
    ),
    kw(
        "量子通信与安全", "安全原理",
        ["不可克隆定理", "量子不可克隆", "量子隐形传态", "纠缠交换"],
        ["no-cloning theorem", "no cloning theorem", "quantum teleportation", "entanglement swapping"],
        4, "技术核心词", "core", False,
        note="明确量子通信/信息理论概念。",
    ),
    kw(
        "量子通信与安全", "后量子密码",
        ["后量子密码", "后量子加密", "抗量子密码", "抗量子加密", "抗量子攻击"],
        ["post-quantum cryptography", "post-quantum encryption", "quantum-resistant cryptography", "quantum-resistant encryption"],
        5, "技术核心词", "core", False,
        note="与量子威胁相关的信息安全方向，纳入量子信息产业外延。",
    ),
    kw(
        "量子传感", "总称",
        ["量子传感", "量子传感器", "量子测量", "量子精密测量"],
        ["quantum sensing", "quantum sensor", "quantum sensors", "quantum measurement", "quantum metrology"],
        5, "技术核心词", "core", False,
        note="直接指向量子精密测量。",
    ),
    kw(
        "量子传感", "导航与惯性测量",
        ["量子陀螺仪", "量子加速度计", "量子惯性导航"],
        ["quantum gyroscope", "quantum accelerometer"],
        5, "技术核心词", "core", False,
        note="明确量子传感装备。",
    ),
    kw(
        "量子传感", "导航泛化词",
        ["无GPS导航", "无全球定位导航"],
        ["GPS-denied navigation", "inertial navigation"],
        2, "条件词", "conditional", True,
        note="惯导泛化严重，必须与量子传感或原子干涉等共现。",
    ),
    kw(
        "量子传感", "重力与地球物理",
        ["量子重力仪", "量子重力梯度仪", "量子重力测量"],
        ["quantum gravimeter", "quantum gravimeters"],
        5, "技术核心词", "core", False,
        note="明确量子传感装备。",
    ),
    kw(
        "量子传感", "重力泛化词",
        ["重力梯度仪"],
        ["gravity gradiometer", "gravity gradiometers"],
        2, "条件词", "conditional", True,
        note="未带量子限定时需共现。",
    ),
    kw(
        "量子传感", "原子钟",
        ["量子钟"],
        ["quantum clock"],
        5, "技术核心词", "core", False,
        note="明确量子计量。",
    ),
    kw(
        "量子传感", "原子钟条件词",
        ["原子钟", "光钟", "便携式原子钟"],
        ["atomic clock", "atomic clocks", "optical clock"],
        2, "条件词", "conditional", True,
        note="原子钟可属于计量装备，但不必然是量子信息企业，需共现。",
    ),
    kw(
        "量子传感", "成像",
        ["量子成像", "量子增强成像", "量子显微", "量子增强显微"],
        ["quantum imaging", "quantum-enhanced imaging", "quantum microscopy", "quantum-enhanced microscopy"],
        4, "技术核心词", "core", False,
        note="明确量子增强成像。",
    ),
    kw(
        "量子传感", "传感补充",
        ["冷原子干涉仪", "原子磁力计", "金刚石NV色心", "NV色心", "量子磁力计"],
        ["cold-atom interferometer", "cold atom interferometer", "atomic magnetometer", "NV center", "nitrogen-vacancy center", "quantum magnetometer"],
        4, "专家补充词", "core", False,
        note="量子精密测量常见技术路线。",
    ),
    kw(
        "上游材料与核心部件", "低温与真空",
        ["稀释制冷机", "低温制冷机", "低温系统", "极低温", "超低温", "超高真空"],
        ["dilution refrigerator", "cryogenic refrigerator", "cryogenic system", "cryogenic environment", "ultra-high vacuum"],
        2, "产业链支撑词", "supply_chain", True,
        note="可判断上游支撑，但不能单独判为量子信息专利。",
    ),
    kw(
        "上游材料与核心部件", "光电器件",
        ["单光子探测器", "集成量子光子", "集成量子光子学"],
        ["single photon detector", "single-photon detector", "integrated quantum photonics"],
        3, "产业链支撑词", "supply_chain", True,
        note="量子通信/光量子重要部件，但仍需量子语境确认。",
    ),
    kw(
        "上游材料与核心部件", "普通光电器件",
        ["激光器", "光调制器"],
        ["laser", "lasers", "optical modulator", "optical modulators"],
        1, "产业链支撑词", "supply_chain", True,
        note="普通光电专利极多，只可在核心词共现时辅助判断链条位置。",
    ),
    kw(
        "上游材料与核心部件", "关键材料",
        ["电光晶体", "声光晶体", "高纯铝", "稀土金属"],
        ["electro-optic crystal", "acousto-optic crystal", "high purity aluminum", "rare earth metals"],
        1, "产业链支撑词", "supply_chain", True,
        note="材料支撑词必须与量子核心词共现。",
    ),
    kw(
        "量子应用", "密码与安全",
        ["非对称密码", "大数分解", "RSA破解", "RSA-2048"],
        ["factoring large numbers", "RSA-2048"],
        2, "条件词", "conditional", True,
        note="只有与量子算法/后量子/抗量子共现时才计入。",
    ),
    kw(
        "量子应用", "材料与化学",
        ["量子化学"],
        ["quantum chemistry"],
        2, "条件词", "conditional", True,
        note="量子化学既可能是普通计算化学，也可能是量子计算应用，需共现。",
    ),
    kw(
        "量子应用", "泛化应用",
        ["材料模拟", "新材料设计", "高温超导", "催化剂设计", "药物发现", "蛋白质相互作用", "溶剂相互作用", "投资组合优化", "金融组合优化"],
        ["materials simulation", "materials design", "high-temperature superconductivity", "catalyst design", "drug discovery", "protein interactions", "portfolio optimization"],
        1, "条件词", "conditional", True,
        note="这些是量子计算潜在应用场景，不能单独识别为量子信息专利。",
    ),
    kw(
        "量子应用", "量子机器学习与优化",
        ["量子机器学习", "量子优化"],
        ["quantum machine learning", "quantum optimization"],
        4, "技术核心词", "core", False,
        note="带量子限定的算法应用词。",
    ),
]


NEGATIVE_KEYWORDS = [
    kw(
        "误判控制词", "量子点显示/材料",
        ["量子点", "量子点发光", "量子点显示", "量子点膜"],
        ["quantum dot", "quantum dots", "quantum dot display"],
        0, "误判控制", "negative", False, penalty=4,
        note="通常对应显示、光伏、发光材料；无核心词时应排除。",
    ),
    kw(
        "误判控制词", "量子阱/普通半导体",
        ["量子阱", "多量子阱", "量子线"],
        ["quantum well", "quantum wells", "multiple quantum well", "quantum wire"],
        0, "误判控制", "negative", False, penalty=4,
        note="普通半导体结构高频词，需降权或复核。",
    ),
    kw(
        "误判控制词", "光电指标",
        ["量子效率", "外量子效率", "内量子效率", "量子产率"],
        ["quantum efficiency", "external quantum efficiency", "internal quantum efficiency", "quantum yield"],
        0, "误判控制", "negative", False, penalty=4,
        note="常见于普通光电器件、材料和光伏。",
    ),
    kw(
        "误判控制词", "普通加工/显示语境",
        ["激光切割", "激光焊接", "激光熔覆", "激光雷达", "量子点电视", "发光二极管"],
        ["laser cutting", "laser welding", "laser cladding", "lidar", "quantum dot tv", "light emitting diode"],
        0, "误判控制", "negative", False, penalty=3,
        note="普通制造、显示或光电语境，提示人工复核。",
    ),
    kw(
        "误判控制词", "英文修辞",
        [],
        ["quantum leap"],
        0, "误判控制", "negative", False, penalty=8,
        note="英文修辞用法，基本应排除。",
    ),
]

# Backward-compatible name used by older notebooks.
AMBIGUOUS_OR_NEGATIVE_KEYWORDS = NEGATIVE_KEYWORDS


def safe_text(x):
    if pd.isna(x):
        return ""
    return str(x)


def normalize_text(text):
    text = safe_text(text).lower()
    replacements = {
        "－": "-", "—": "-", "–": "-", "‑": "-",
        "（": "(", "）": ")", "，": ",", "；": ";",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def build_patent_text(row, cn_abs_col="中文摘要", en_abs_col="英文摘要", extra_cols=None):
    parts = [safe_text(row.get(cn_abs_col, "")), safe_text(row.get(en_abs_col, ""))]
    if extra_cols:
        parts.extend(safe_text(row.get(c, "")) for c in extra_cols)
    return normalize_text(" ".join(parts))


def term_to_regex(term):
    term = safe_text(term).strip()
    if not term:
        return None
    has_cn = bool(re.search(r"[\u4e00-\u9fff]", term))
    if has_cn:
        return re.escape(term.lower())
    t = re.escape(term.lower())
    t = t.replace(r"\ ", r"[\s\-]+").replace(r"\-", r"[\s\-]+")
    return r"(?<![a-z0-9])" + t + r"(?![a-z0-9])"


def compile_keyword_patterns(keyword_list):
    compiled = []
    for item in keyword_list:
        patterns = []
        for term in item.get("terms_cn", []) + item.get("terms_en", []):
            rgx = term_to_regex(term)
            if rgx:
                patterns.append((term, re.compile(rgx, flags=re.IGNORECASE)))
        new_item = item.copy()
        new_item["patterns"] = patterns
        compiled.append(new_item)
    return compiled


def compile_negative_patterns(negative_list):
    return compile_keyword_patterns(negative_list)


def _compile_cooccur_patterns(terms):
    return [(t, re.compile(term_to_regex(t), flags=re.IGNORECASE)) for t in terms if term_to_regex(t)]


POS_PATTERNS = compile_keyword_patterns(QUANTUM_KEYWORDS)
NEG_PATTERNS = compile_negative_patterns(NEGATIVE_KEYWORDS)
CORE_COOCCUR_PATTERNS = _compile_cooccur_patterns(CORE_COOCCUR_TERMS)


def _join_terms(terms):
    return "；".join(sorted(set(t for t in terms if t)))


def _matched_item_terms(text, item):
    return [term for term, pattern in item["patterns"] if pattern.search(text)]


def _has_cooccur(text, item, core_hit_terms):
    if core_hit_terms:
        return True
    terms = item.get("cooccur_terms") or CORE_COOCCUR_TERMS
    for term, pattern in _compile_cooccur_patterns(terms):
        if pattern.search(text):
            return True
    return False


def score_one_patent_text(text):
    text = normalize_text(text)

    all_positive_hits = []
    core_terms = []
    conditional_terms = []
    supply_chain_terms = []
    negative_terms = []
    skipped_conditional_terms = []
    skipped_supply_chain_terms = []
    category_scores = defaultdict(int)
    subcategory_scores = defaultdict(int)
    source_types = set()
    match_type_scores = defaultdict(int)

    positive_hits_by_item = []
    for item in POS_PATTERNS:
        hit_terms = _matched_item_terms(text, item)
        if not hit_terms:
            continue
        positive_hits_by_item.append((item, hit_terms))
        all_positive_hits.extend(hit_terms)
        if item.get("match_type") == "core":
            core_terms.extend(hit_terms)

    for item, hit_terms in positive_hits_by_item:
        match_type = item.get("match_type", "conditional")
        if item.get("cooccur_required") and not _has_cooccur(text, item, core_terms):
            if match_type == "supply_chain":
                skipped_supply_chain_terms.extend(hit_terms)
            else:
                skipped_conditional_terms.extend(hit_terms)
            continue

        score = int(item.get("score", 0))
        category = item["category"]
        sub_category = item["sub_category"]
        category_scores[category] += score
        subcategory_scores[(category, sub_category)] += score
        match_type_scores[match_type] += score
        source_types.add(item.get("source_type", ""))

        if match_type == "core":
            core_terms.extend(hit_terms)
        elif match_type == "supply_chain":
            supply_chain_terms.extend(hit_terms)
        else:
            conditional_terms.extend(hit_terms)

    penalty = 0
    for item in NEG_PATTERNS:
        hit_terms = _matched_item_terms(text, item)
        if hit_terms:
            negative_terms.extend(hit_terms)
            penalty += int(item.get("penalty", 0))

    raw_score = sum(category_scores.values())
    core_score = match_type_scores.get("core", 0)
    supply_score = match_type_scores.get("supply_chain", 0)
    conditional_score = match_type_scores.get("conditional", 0)
    has_core = core_score >= 4 or bool(core_terms)
    only_support = raw_score > 0 and core_score == 0

    adjusted_score = raw_score
    if negative_terms:
        adjusted_score = max(0, adjusted_score - penalty)
    if only_support:
        adjusted_score = min(adjusted_score, 2)

    if category_scores:
        main_category = max(category_scores.items(), key=lambda x: (x[1], x[0]))[0]
    else:
        main_category = ""

    chain_position = infer_chain_position(category_scores, subcategory_scores)

    if not has_core and (negative_terms or skipped_conditional_terms or skipped_supply_chain_terms):
        relevance = "不相关"
    elif adjusted_score >= 10 and has_core and core_score >= 5 and penalty <= 3:
        relevance = "高相关"
    elif adjusted_score >= 6 and has_core:
        relevance = "中相关"
    elif adjusted_score >= 4 and has_core:
        relevance = "低相关/待复核"
    else:
        relevance = "不相关"

    return {
        "quantum_score_raw": raw_score,
        "quantum_score": adjusted_score,
        "core_score": core_score,
        "conditional_score": conditional_score,
        "supply_chain_score": supply_score,
        "negative_penalty": penalty,
        "matched_terms": _join_terms(all_positive_hits),
        "core_terms": _join_terms(core_terms),
        "conditional_terms": _join_terms(conditional_terms),
        "supply_chain_terms": _join_terms(supply_chain_terms),
        "negative_terms": _join_terms(negative_terms),
        "skipped_conditional_terms": _join_terms(skipped_conditional_terms),
        "skipped_supply_chain_terms": _join_terms(skipped_supply_chain_terms),
        "main_category": main_category,
        "chain_position": chain_position,
        "category_scores": dict(category_scores),
        "source_types": _join_terms(source_types),
        "relevance": relevance,
    }


def infer_chain_position(category_scores, subcategory_scores):
    if not category_scores:
        return ""

    subcats = {sub for (cat, sub), score in subcategory_scores.items() if score > 0}
    cats = {cat for cat, score in category_scores.items() if score > 0}

    if {
        "离子阱", "超导", "超导电路", "中性原子", "硅自旋", "光量子", "拓扑"
    } & subcats:
        return "量子计算硬件/量子芯片"
    if {"纠错与容错", "纠错泛化词", "算法", "量子模拟", "量子机器学习与优化"} & subcats:
        return "量子软件/算法/模拟"
    if "量子通信与安全" in cats:
        return "量子通信与安全"
    if "量子传感" in cats:
        return "量子传感器/精密测量"
    if "量子计算" in cats:
        return "量子计算系统"
    if "上游材料与核心部件" in cats:
        return "上游材料与核心部件"
    if "量子应用" in cats:
        return "量子应用场景"
    return "其他量子相关"


CORE_QUANTUM_REGEX = re.compile(
    r"量子(?!点|阱|线|效率|产率)|"
    r"quantum(?![\s\-]*(dot|dots|well|wells|wire|efficiency|yield|leap))|"
    r"qubit|qkd|post[\s\-]+quantum|quantum[\s\-]+resistant|"
    r"shor'?s?[\s\-]+algorithm|"
    r"离子阱|俘获离子|囚禁离子|里德堡原子|冷原子|"
    r"单光子探测|nv色心|金刚石nv色心|原子干涉仪|"
    r"trapped[\s\-]+ion|ion[\s\-]+trap|rydberg|single[\s\-]+photon|nv[\s\-]+center",
    flags=re.IGNORECASE,
)


def make_text_series(df, cn_abs_col="中文摘要", en_abs_col="英文摘要", extra_text_cols=None):
    cols = [cn_abs_col, en_abs_col]
    if extra_text_cols:
        cols += extra_text_cols
    existing_cols = [c for c in cols if c in df.columns]
    if not existing_cols:
        return pd.Series([""] * len(df), index=df.index)
    return (
        df[existing_cols]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .map(normalize_text)
    )


def coarse_filter_quantum_patents(
    df,
    cn_abs_col="中文摘要",
    en_abs_col="英文摘要",
    extra_text_cols=None,
):
    text_series = make_text_series(df, cn_abs_col, en_abs_col, extra_text_cols)
    return text_series.str.contains(CORE_QUANTUM_REGEX, regex=True, na=False)


def _mode_nonempty(values):
    items = [i for i in values if isinstance(i, str) and i.strip()]
    return Counter(items).most_common(1)[0][0] if items else ""


def tag_quantum_patents(
    df,
    cn_abs_col="中文摘要",
    en_abs_col="英文摘要",
    firm_col="企业名称",
    year_col="year",
    extra_text_cols=None,
    split_firms=False,
    firm_sep_regex=r"[;；,，、|/]+",
    progress_every=10000,
):
    data = df.copy()

    data["_patent_text_for_quantum"] = data.apply(
        lambda row: build_patent_text(
            row,
            cn_abs_col=cn_abs_col,
            en_abs_col=en_abs_col,
            extra_cols=extra_text_cols,
        ),
        axis=1,
    )

    score_rows = []
    total = len(data)
    for n, text in enumerate(data["_patent_text_for_quantum"], start=1):
        score_rows.append(score_one_patent_text(text))
        if progress_every and n % progress_every == 0:
            print(f"量子专利识别进度：{n}/{total}")

    scores = pd.DataFrame(score_rows, index=data.index)
    existing_score_cols = [c for c in scores.columns if c in data.columns]
    if existing_score_cols:
        data = data.drop(columns=existing_score_cols)
    if "is_quantum_patent" in data.columns:
        data = data.drop(columns=["is_quantum_patent"])
    patent_tagged = pd.concat([data, scores], axis=1)
    patent_tagged["is_quantum_patent"] = patent_tagged["relevance"].isin(
        ["高相关", "中相关", "低相关/待复核"]
    ).astype(int)

    if split_firms and firm_col in patent_tagged.columns:
        patent_tagged[firm_col] = patent_tagged[firm_col].fillna("").astype(str)
        patent_tagged["_firm_list"] = patent_tagged[firm_col].str.split(firm_sep_regex)
        patent_tagged = patent_tagged.explode("_firm_list")
        patent_tagged[firm_col] = patent_tagged["_firm_list"].str.strip()
        patent_tagged = patent_tagged[patent_tagged[firm_col] != ""].copy()
        patent_tagged.drop(columns=["_firm_list"], inplace=True)

    quantum_patents = patent_tagged[patent_tagged["is_quantum_patent"] == 1].copy()

    if len(quantum_patents) == 0 or firm_col not in quantum_patents.columns:
        return patent_tagged, quantum_patents, pd.DataFrame(), pd.DataFrame()

    agg_common = dict(
        quantum_patent_count=("is_quantum_patent", "sum"),
        quantum_score_sum=("quantum_score", "sum"),
        quantum_score_mean=("quantum_score", "mean"),
        core_score_sum=("core_score", "sum"),
        supply_chain_score_sum=("supply_chain_score", "sum"),
        high_related_count=("relevance", lambda x: (x == "高相关").sum()),
        medium_related_count=("relevance", lambda x: (x == "中相关").sum()),
        low_related_count=("relevance", lambda x: (x == "低相关/待复核").sum()),
        main_categories=("main_category", lambda x: "；".join(sorted(set(x.dropna()) - {""}))),
        chain_positions=("chain_position", lambda x: "；".join(sorted(set(x.dropna()) - {""}))),
        matched_terms=("matched_terms", lambda x: "；".join(sorted(set("；".join(x.dropna()).split("；")) - {""}))),
        core_terms=("core_terms", lambda x: "；".join(sorted(set("；".join(x.dropna()).split("；")) - {""}))),
        supply_chain_terms=("supply_chain_terms", lambda x: "；".join(sorted(set("；".join(x.dropna()).split("；")) - {""}))),
        negative_terms=("negative_terms", lambda x: "；".join(sorted(set("；".join(x.dropna()).split("；")) - {""}))),
    )

    if year_col in quantum_patents.columns:
        firm_year_quantum = (
            quantum_patents.groupby([firm_col, year_col], dropna=False)
            .agg(**agg_common)
            .reset_index()
        )
    else:
        firm_year_quantum = pd.DataFrame()

    first_last = {}
    if year_col in quantum_patents.columns:
        first_last = {
            "first_year": (year_col, "min"),
            "last_year": (year_col, "max"),
        }

    firm_quantum = (
        quantum_patents.groupby(firm_col, dropna=False)
        .agg(**first_last, **agg_common)
        .reset_index()
    )

    firm_main_position = (
        quantum_patents.groupby(firm_col)["chain_position"]
        .agg(_mode_nonempty)
        .reset_index()
        .rename(columns={"chain_position": "main_chain_position"})
    )
    firm_main_category = (
        quantum_patents.groupby(firm_col)["main_category"]
        .agg(_mode_nonempty)
        .reset_index()
        .rename(columns={"main_category": "main_quantum_category"})
    )
    firm_quantum = firm_quantum.merge(firm_main_position, on=firm_col, how="left")
    firm_quantum = firm_quantum.merge(firm_main_category, on=firm_col, how="left")

    firm_quantum["firm_relevance"] = np.select(
        [
            (firm_quantum["high_related_count"] >= 2)
            | ((firm_quantum["high_related_count"] >= 1) & (firm_quantum["core_score_sum"] >= 12)),
            (firm_quantum["high_related_count"] >= 1)
            | (firm_quantum["medium_related_count"] >= 2)
            | (firm_quantum["core_score_sum"] >= 10),
            (firm_quantum["quantum_patent_count"] >= 1),
        ],
        ["核心量子企业", "较高相关企业", "候选/待复核企业"],
        default="不相关",
    )

    return patent_tagged, quantum_patents, firm_year_quantum, firm_quantum


if __name__ == "__main__":
    # Example:
    # df = pd.read_csv("your_patent_file.csv", low_memory=False)
    # candidate_mask = coarse_filter_quantum_patents(
    #     df,
    #     cn_abs_col="摘要 (中文)",
    #     en_abs_col="摘要 (英文)",
    #     extra_text_cols=["标题 (中文)", "标题 (英文)", "独立权利要求"],
    # )
    # patent_tagged, quantum_patents, firm_year_quantum, firm_quantum = tag_quantum_patents(
    #     df.loc[candidate_mask].copy(),
    #     cn_abs_col="摘要 (中文)",
    #     en_abs_col="摘要 (英文)",
    #     firm_col="第一申请人",
    #     year_col="year",
    #     extra_text_cols=["标题 (中文)", "标题 (英文)", "技术功效句", "独立权利要求", "首权翻译", "首项权利要求"],
    #     progress_every=10000,
    # )
    pass
