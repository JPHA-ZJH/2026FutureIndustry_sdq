# -*- coding: utf-8 -*-
"""
生物制造专利匹配、技术分类与核心程度评分模块。

设计原则
--------
1. 词典以中文术语为主体，保留中文专利和中文研报中常见的英文缩写或
   技术名（DBTL、CRISPR、MAGE、TALEN、PHA、PLA、SAF、CAR-T、
   iPSC、eBio、AI 等），不重复维护一套完整英文翻译词典。
2. “生物制造、合成生物学、细胞工厂、DBTL”等专用词可独立触发。
3. 基因编辑、发酵、酶、蛋白、细胞培养以及具体产品等通用词，必须处于
   生物工程与生产制造上下文中才生效，避免把普通生物医药、食品发酵、
   诊断检测或基础研究专利全部纳入。
4. 最终 biomanufacturing_score 为1—5分的技术核心程度；
   biomanufacturing_score_raw 单独保留有效规则的证据累计分。
5. 不输出“高相关/低相关/待复核”等离散相关性标签。

主要依据
--------
- 深企投产业研究院《2024生物制造行业研究报告》；
- Stanford Emerging Technology Review 2026 标注页39—55（生物技术与
  合成生物学章节；用户指定范围39—57内的实质相关内容）；
- 中国科学院天津工业生物技术研究所公开研究方向与科研进展；
- 公开资本市场研报对产业链、DBTL、菌株构建、发酵放大与分离纯化的补充。
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd


# ---------------------------------------------------------------------
# 1. 关键词规则
# ---------------------------------------------------------------------
# score：技术在生物制造体系中的核心程度，1（外围应用）—5（核心范式）。
# match_type：
#   core        强专用词，可独立匹配；
#   frontier    明确的前沿生物制造词，可独立匹配；
#   contextual  通用技术词，需生物工程+生产制造上下文；
#   process     工艺词，需生物工程+生产制造上下文；
#   product     产品/应用词，需生物工程+生产制造上下文；
#   support     安全、装备等支撑词，需生物制造上下文。

BIOMANUFACTURING_RULES: List[Dict] = [
    # ---- 核心概念与工程范式 ----
    {"category": "生物制造基础", "sub_category": "核心概念", "terms": ["生物制造", "工业生物制造", "先进生物制造", "绿色生物制造", "低碳生物制造", "合成生物制造", "微生物制造", "生物法制造", "生物制造技术"], "score": 5, "match_type": "core", "industry_segment": "技术平台", "source_type": "两份报告直接词"},
    {"category": "生物制造基础", "sub_category": "合成生物学", "terms": ["合成生物学", "工程生物学", "工业合成生物学", "合成生物技术", "合成生物平台", "合成生物制造平台", "SynBio"], "score": 5, "match_type": "core", "industry_segment": "技术平台", "source_type": "两份报告直接词/公开研报补充"},
    {"category": "生物制造基础", "sub_category": "工业生物技术", "terms": ["工业生物技术", "工业生物工程", "工业微生物技术", "工业微生物工程", "生物过程工程"], "score": 5, "match_type": "core", "industry_segment": "技术平台", "source_type": "2024报告/中科院公开方向"},
    {"category": "生物制造基础", "sub_category": "DBTL工程闭环", "terms": ["设计-构建-测试-学习", "设计—构建—测试—学习", "设计构建测试学习", "DBTL", "DBTL循环", "DBTL闭环", "生物工程闭环", "设计构建工作循环"], "score": 5, "match_type": "core", "industry_segment": "技术平台", "source_type": "两份报告直接词/公开资本研报补充"},
    {"category": "生物制造基础", "sub_category": "生物铸造与自动化平台", "terms": ["生物铸造厂", "生物铸造平台", "Biofoundry", "合成生物自动化平台", "自动化生物制造平台", "自动化菌种构建平台"], "score": 5, "match_type": "core", "industry_segment": "技术平台", "source_type": "科研机构/资本研报补充"},

    # ---- 计算设计、虚拟细胞与生成式生物学 ----
    {"category": "计算设计与AI", "sub_category": "虚拟细胞", "terms": ["虚拟细胞", "全细胞模型", "数字细胞", "细胞数字孪生", "细胞机理模型", "工业底盘细胞模型", "机理模型与人工智能双驱动"], "score": 5, "match_type": "frontier", "industry_segment": "底层工具", "source_type": "SETR/中科院公开进展", "source_detail": "SETR 2026标注页41；中国科学院天津工业生物技术研究所虚拟细胞项目", "source_url": "https://www.tib.cas.cn/xwdt/mtsm/t_8142486.html"},
    {"category": "计算设计与AI", "sub_category": "生成式生物学", "terms": ["生成式生物学", "生成生物学", "基因组基础模型", "基因组大模型", "DNA基础模型", "生物基础模型", "Evo 2", "Evo2"], "score": 5, "match_type": "frontier", "industry_segment": "底层工具", "source_type": "SETR直接词/科研机构补充", "source_detail": "SETR 2026标注页42"},
    {"category": "计算设计与AI", "sub_category": "蛋白与抗体基础模型", "terms": ["蛋白质基础模型", "蛋白质大模型", "抗体基础模型", "抗体生成模型"], "score": 4, "match_type": "contextual", "industry_segment": "底层工具", "source_type": "SETR直接词/科研机构补充", "source_detail": "SETR 2026标注页41—42；需生产制造上下文以排除一般药物发现"},
    {"category": "计算设计与AI", "sub_category": "生物计算设计", "terms": ["计算生物学设计", "菌种计算设计", "菌种从头设计", "细胞工厂计算设计", "计算辅助生物设计", "生物系统计算设计", "生物CAD", "代谢途径计算设计", "生物线路计算设计"], "score": 4, "match_type": "contextual", "industry_segment": "底层工具", "source_type": "中科院/公开资本研报补充", "source_url": "https://tib.cas.cn/rcdw/rczp/zyjszp/t_8227670.html"},
    {"category": "计算设计与AI", "sub_category": "AI蛋白与酶设计", "terms": ["AI蛋白质设计", "人工智能蛋白质设计", "AI酶设计", "酶智能设计", "蛋白质从头设计", "抗体从头设计", "生成式蛋白质设计", "蛋白质序列生成", "蛋白结构预测", "酶功能预测"], "score": 4, "match_type": "contextual", "industry_segment": "底层工具", "source_type": "SETR/科研机构/资本研报补充"},
    {"category": "计算设计与AI", "sub_category": "多组学与系统建模", "terms": ["系统生物学", "定量生物学", "多组学分析", "基因组学", "转录组学", "蛋白质组学", "代谢组学", "代谢网络模型", "代谢调控网络", "基因调控网络", "基因型-表型", "基因型－表型", "生物知识图谱"], "score": 4, "match_type": "contextual", "industry_segment": "底层工具", "source_type": "中科院/公开资本研报补充"},

    # ---- DNA/RNA读写编辑与合成工具 ----
    {"category": "DNA/RNA读写编辑", "sub_category": "基因测序", "terms": ["基因测序", "DNA测序", "高通量测序", "下一代测序", "纳米孔测序", "长读长测序", "单分子测序", "便携式DNA测序", "NGS"], "score": 4, "match_type": "contextual", "industry_segment": "底层工具", "source_type": "两份报告直接词/资本研报补充"},
    {"category": "DNA/RNA读写编辑", "sub_category": "基因合成", "terms": ["基因合成", "DNA合成", "DNA从头合成", "寡核苷酸合成", "酶促DNA合成", "大规模DNA合成", "超高密度DNA合成", "DNA组装", "基因组装", "基因组合成", "合成基因组", "人工合成基因组"], "score": 4, "match_type": "contextual", "industry_segment": "底层工具", "source_type": "两份报告直接词/科研机构补充"},
    {"category": "DNA/RNA读写编辑", "sub_category": "基因与基因组编辑", "terms": ["基因编辑", "基因组编辑", "CRISPR", "CRISPR/Cas", "CRISPR-Cas9", "碱基编辑", "先导编辑", "TALEN", "MAGE", "多重基因组编辑", "基因敲除", "基因敲入", "定点突变", "定向突变"], "score": 4, "match_type": "contextual", "industry_segment": "底层工具", "source_type": "2024报告/资本研报补充"},
    {"category": "DNA/RNA读写编辑", "sub_category": "RNA工程", "terms": ["RNA调控装置", "RNA开关", "核糖开关", "RNA编辑", "合成RNA", "非编码RNA调控", "mRNA设计", "转录后调控", "翻译调控"], "score": 4, "match_type": "contextual", "industry_segment": "底层工具", "source_type": "2024报告/支持扩展"},
    {"category": "DNA/RNA读写编辑", "sub_category": "DNA存储与新型信息材料", "terms": ["DNA数据存储", "DNA信息存储", "合成DNA存储", "DNA存储介质", "DNA编码存储", "DNA折纸", "DNA纳米结构"], "score": 3, "match_type": "contextual", "industry_segment": "应用产品", "source_type": "SETR直接词", "source_detail": "SETR 2026标注页44、48—49"},

    # ---- 生物元件、线路与模块 ----
    {"category": "生物元件与线路", "sub_category": "标准化生物元件", "terms": ["生物元件", "标准化生物元件", "生物零件", "BioBrick", "BioBricks", "元件库", "表达元件库", "启动子库", "终止子库", "核糖体结合位点", "RBS元件"], "score": 4, "match_type": "contextual", "industry_segment": "底层工具", "source_type": "2024报告/资本研报补充"},
    {"category": "生物元件与线路", "sub_category": "基因线路", "terms": ["基因线路", "合成基因线路", "遗传线路", "基因回路", "生物逻辑门", "基因开关", "遗传开关", "基因振荡器", "生物振荡器", "动态调控线路", "感应调控线路"], "score": 4, "match_type": "contextual", "industry_segment": "技术平台", "source_type": "2024报告/支持扩展"},
    {"category": "生物元件与线路", "sub_category": "模块化装配", "terms": ["生物模块", "基因模块", "模块化装配", "标准化装配", "DNA模块组装", "多基因组装", "组合生物合成", "组合式途径构建"], "score": 4, "match_type": "contextual", "industry_segment": "技术平台", "source_type": "2024报告/支持扩展"},

    # ---- 底盘细胞、菌株与群落工程 ----
    {"category": "底盘细胞与菌株工程", "sub_category": "底盘细胞", "terms": ["底盘细胞", "工业底盘细胞", "微生物底盘", "底盘菌株", "模式底盘", "通用底盘细胞", "非模式底盘", "最小细胞", "人工合成细胞", "合成细胞"], "score": 5, "match_type": "core", "industry_segment": "技术平台", "source_type": "两份报告/中科院/资本研报补充"},
    {"category": "底盘细胞与菌株工程", "sub_category": "工业菌种创制", "terms": ["工业菌种创制", "工业菌株创制", "工程菌株构建", "工程菌构建", "重组菌构建", "生产菌株构建", "高产菌株", "高效生产菌株", "细胞工厂菌株", "定制菌种", "菌种设计定制"], "score": 4, "match_type": "contextual", "industry_segment": "技术平台", "source_type": "中科院/资本研报补充", "source_url": "https://tib.cas.cn/kxyj/lyfx/202410/t20241031_7412324.html"},
    {"category": "底盘细胞与菌株工程", "sub_category": "菌种选育与诱变", "terms": ["菌种选育", "菌株选育", "微生物育种", "诱变育种", "复合诱变", "ARTP", "常压室温等离子体诱变", "基因组改组", "核糖体工程", "全基因组诱变"], "score": 4, "match_type": "contextual", "industry_segment": "技术平台", "source_type": "工业生物技术/资本研报补充"},
    {"category": "底盘细胞与菌株工程", "sub_category": "底盘物种", "terms": ["大肠杆菌底盘", "酿酒酵母底盘", "毕赤酵母底盘", "枯草芽孢杆菌底盘", "谷氨酸棒状杆菌底盘", "链霉菌底盘", "黑曲霉底盘", "米曲霉底盘", "微藻底盘", "光合细胞工厂"], "score": 4, "match_type": "contextual", "industry_segment": "技术平台", "source_type": "2024报告/资本研报补充"},
    {"category": "底盘细胞与菌株工程", "sub_category": "基因组精简与细胞优化", "terms": ["基因组精简", "基因组简化", "最小基因组", "非必需基因删除", "底盘细胞优化", "底盘优化", "宿主细胞优化", "细胞资源重分配", "前体供给强化", "辅因子工程"], "score": 4, "match_type": "contextual", "industry_segment": "技术平台", "source_type": "科研论文/中科院补充"},
    {"category": "底盘细胞与菌株工程", "sub_category": "耐受与适应性进化", "terms": ["菌株耐受性", "产物耐受性", "底物耐受性", "毒性耐受", "耐高温菌株", "耐酸菌株", "耐溶剂菌株", "适应性实验室进化", "适应性进化", "ALE育种", "代谢负担优化"], "score": 4, "match_type": "contextual", "industry_segment": "技术平台", "source_type": "科研机构/资本研报补充"},
    {"category": "底盘细胞与菌株工程", "sub_category": "合成群落与共培养", "terms": ["合成微生物群落", "人工微生物群落", "微生物群落工程", "微生物共培养", "多菌种共培养", "合成菌群", "微生物联合体", "分工代谢", "群落代谢工程"], "score": 4, "match_type": "contextual", "industry_segment": "技术平台", "source_type": "SETR/科研机构补充"},

    # ---- 代谢工程与细胞工厂 ----
    {"category": "代谢工程与细胞工厂", "sub_category": "细胞工厂", "terms": ["细胞工厂", "微生物细胞工厂", "工程细胞工厂", "人造细胞工厂", "高效细胞工厂", "全细胞催化工厂", "细胞工厂构建", "细胞工厂优化"], "score": 5, "match_type": "core", "industry_segment": "技术平台", "source_type": "两份报告/中科院直接词"},
    {"category": "代谢工程与细胞工厂", "sub_category": "代谢工程", "terms": ["代谢工程", "代谢途径工程", "代谢通路改造", "代谢通路重构", "合成代谢途径", "异源代谢途径", "生物合成途径", "代谢途径优化", "途径组装", "途径平衡"], "score": 4, "match_type": "contextual", "industry_segment": "技术平台", "source_type": "两份报告/中科院补充"},
    {"category": "代谢工程与细胞工厂", "sub_category": "代谢流与动态调控", "terms": ["代谢流分析", "代谢通量分析", "代谢流重定向", "碳流重定向", "动态代谢调控", "静态代谢调控", "反馈抑制解除", "旁路代谢敲除", "竞争途径敲除", "产物通路强化"], "score": 4, "match_type": "contextual", "industry_segment": "技术平台", "source_type": "科研机构/支持扩展"},
    {"category": "代谢工程与细胞工厂", "sub_category": "表达系统", "terms": ["高效表达系统", "异源表达系统", "重组表达系统", "蛋白高效表达", "异源蛋白表达", "分泌表达", "组成型表达", "诱导表达", "无诱导表达", "表达宿主", "表达盒"], "score": 4, "match_type": "contextual", "industry_segment": "技术平台", "source_type": "中科院/资本研报补充"},
    {"category": "代谢工程与细胞工厂", "sub_category": "运输与分泌工程", "terms": ["转运蛋白工程", "膜转运工程", "产物外排", "产物分泌", "分泌途径优化", "信号肽优化", "跨膜转运", "底物摄取强化"], "score": 4, "match_type": "contextual", "industry_segment": "技术平台", "source_type": "科研机构/支持扩展"},

    # ---- 酶工程、蛋白质工程与生物催化 ----
    {"category": "酶与蛋白质工程", "sub_category": "酶工程", "terms": ["酶工程", "工业酶工程", "酶分子改造", "酶定向改造", "酶活性改造", "酶稳定性改造", "酶理性设计", "酶半理性设计", "酶库构建", "工业酶创制"], "score": 4, "match_type": "contextual", "industry_segment": "技术平台", "source_type": "2024报告/中科院/资本研报补充"},
    {"category": "酶与蛋白质工程", "sub_category": "定向进化", "terms": ["定向进化", "酶定向进化", "蛋白质定向进化", "错误易感PCR", "易错PCR", "DNA改组", "DNA shuffling", "饱和突变", "迭代饱和突变", "突变体文库"], "score": 4, "match_type": "contextual", "industry_segment": "技术平台", "source_type": "科研机构/资本研报补充"},
    {"category": "酶与蛋白质工程", "sub_category": "蛋白质工程", "terms": ["蛋白质工程", "蛋白分子设计", "蛋白序列优化", "蛋白结构改造", "蛋白折叠优化", "重组蛋白工程", "多肽工程", "抗体工程", "融合蛋白设计"], "score": 4, "match_type": "contextual", "industry_segment": "技术平台", "source_type": "2024报告/SETR/中科院补充"},
    {"category": "酶与蛋白质工程", "sub_category": "生物催化", "terms": ["生物催化", "酶催化合成", "酶法合成", "全细胞催化", "级联酶催化", "多酶级联", "酶促转化", "生物转化", "手性生物催化", "辅酶再生"], "score": 4, "match_type": "contextual", "industry_segment": "生产制造", "source_type": "2024报告/中科院/资本研报补充"},
    {"category": "酶与蛋白质工程", "sub_category": "酶固定化与反应体系", "terms": ["固定化酶", "酶固定化", "固定化细胞", "酶膜反应器", "连续酶催化", "非水相酶催化", "酶反应器", "酶循环使用"], "score": 3, "match_type": "process", "industry_segment": "生产制造", "source_type": "工业生物技术支持扩展"},

    # ---- DBTL测试、筛选和实验自动化 ----
    {"category": "测试筛选与自动化", "sub_category": "高通量构建与筛选", "terms": ["高通量筛选", "超高通量筛选", "高通量菌株筛选", "高通量构建", "自动化高通量筛选", "菌株自动化筛选", "酶高通量筛选", "突变体筛选", "阳性转化子筛选", "表型筛选"], "score": 4, "match_type": "contextual", "industry_segment": "底层工具", "source_type": "中科院/资本研报补充"},
    {"category": "测试筛选与自动化", "sub_category": "微流控与单细胞筛选", "terms": ["微流控筛选", "液滴微流控", "单细胞筛选", "单细胞分选", "荧光激活细胞分选", "FACS", "生物传感器筛选", "代谢物生物传感器", "微液滴筛选"], "score": 4, "match_type": "contextual", "industry_segment": "底层工具", "source_type": "资本研报/科研机构补充"},
    {"category": "测试筛选与自动化", "sub_category": "实验室自动化", "terms": ["实验室自动化", "自动化移液", "自动化培养", "机器人实验平台", "自动化实验工作站", "自动化菌株改造", "无人化生物实验室", "云实验室"], "score": 4, "match_type": "contextual", "industry_segment": "底层工具", "source_type": "资本研报/科研机构补充"},
    {"category": "测试筛选与自动化", "sub_category": "生物制造数据", "terms": ["生物制造数据库", "工业菌种数据库", "工业酶数据库", "生物制造数据中心", "生物实验数据闭环", "数据-模型-应用飞轮", "生物制造数据飞轮"], "score": 4, "match_type": "core", "industry_segment": "底层工具", "source_type": "中科院公开方向", "source_url": "https://tib.cas.cn/rcdw/rczp/zyjszp/t_8227670.html"},

    # ---- 发酵、生物反应与规模化 ----
    {"category": "生物过程与规模化", "sub_category": "发酵工程", "terms": ["发酵工程", "工业发酵", "生物发酵", "微生物发酵", "精准发酵", "精密发酵", "深层发酵", "固态发酵", "液态发酵", "补料分批发酵", "连续发酵", "发酵生产"], "score": 4, "match_type": "process", "industry_segment": "生产制造", "source_type": "2024报告/资本研报补充"},
    {"category": "生物过程与规模化", "sub_category": "生物反应器", "terms": ["生物反应器", "发酵罐", "细胞培养反应器", "一次性生物反应器", "连续流生物反应器", "气升式生物反应器", "光生物反应器", "膜生物反应器", "自动化生物反应器"], "score": 3, "match_type": "process", "industry_segment": "生产制造", "source_type": "两份报告/支持扩展"},
    {"category": "生物过程与规模化", "sub_category": "过程控制与优化", "terms": ["发酵过程控制", "生物过程控制", "发酵工艺优化", "培养基优化", "补料策略", "溶氧控制", "pH控制", "通气搅拌控制", "在线发酵监测", "软测量", "过程分析技术", "PAT技术"], "score": 3, "match_type": "process", "industry_segment": "生产制造", "source_type": "资本研报补充"},
    {"category": "生物过程与规模化", "sub_category": "工艺放大与中试", "terms": ["生物工艺放大", "发酵放大", "规模放大", "工业化放大", "中试放大", "中试发酵", "中试平台", "放大培养", "规模化培养", "吨级发酵", "千吨级发酵"], "score": 4, "match_type": "process", "industry_segment": "生产制造", "source_type": "中科院/资本研报补充"},
    {"category": "生物过程与规模化", "sub_category": "下游分离纯化", "terms": ["发酵液分离", "发酵液纯化", "分离纯化", "下游纯化", "下游分离", "细胞破碎", "膜分离", "膜过滤", "超滤浓缩", "离子交换纯化", "层析纯化", "结晶纯化", "萃取纯化", "脱色除杂"], "score": 3, "match_type": "process", "industry_segment": "生产制造", "source_type": "资本研报补充"},
    {"category": "生物过程与规模化", "sub_category": "连续制造与过程强化", "terms": ["连续生物制造", "连续生物加工", "连续培养", "连续流生物催化", "生物过程强化", "集成生物工艺", "原位产物分离", "反应分离耦合"], "score": 4, "match_type": "process", "industry_segment": "生产制造", "source_type": "科研机构/支持扩展"},

    # ---- 原料路线、生物炼制与低碳制造 ----
    {"category": "原料与低碳路线", "sub_category": "生物炼制", "terms": ["生物炼制", "综合生物炼制", "木质纤维素生物炼制", "生物质炼制", "生物质高值化", "生物质梯级利用", "生物质全组分利用"], "score": 4, "match_type": "core", "industry_segment": "生产制造", "source_type": "2024报告/公开资本研报补充"},
    {"category": "原料与低碳路线", "sub_category": "非粮生物质", "terms": ["非粮生物质", "非粮碳源", "秸秆糖", "木质纤维素", "纤维素糖", "半纤维素糖", "农业废弃物糖化", "农林废弃物", "秸秆原料化", "生物质碳源平台"], "score": 3, "match_type": "contextual", "industry_segment": "原料层", "source_type": "2024报告/资本研报补充"},
    {"category": "原料与低碳路线", "sub_category": "一碳生物制造", "terms": ["一碳生物制造", "C1生物制造", "一碳生物转化", "二氧化碳生物转化", "二氧化碳生物合成", "甲醇生物转化", "甲烷生物转化", "合成气生物转化", "工业尾气发酵", "气体发酵", "食气微生物", "食气梭菌"], "score": 4, "match_type": "contextual", "industry_segment": "原料层", "source_type": "2024报告/中科院/资本研报补充"},
    {"category": "原料与低碳路线", "sub_category": "电生物合成", "terms": ["电生物合成", "电驱生物合成", "电化学-生物耦合", "电化学生物耦合", "电能驱动生物制造", "eBio", "电合成生物学"], "score": 5, "match_type": "frontier", "industry_segment": "前沿制造", "source_type": "SETR直接词", "source_detail": "SETR 2026标注页48"},
    {"category": "原料与低碳路线", "sub_category": "人工碳循环与固碳", "terms": ["人工碳循环", "生物固碳", "微生物固碳", "光合固碳", "非光合固碳", "碳固定途径", "碳捕获生物转化", "乙酰辅酶A合成途径", "还原甘氨酸途径"], "score": 4, "match_type": "contextual", "industry_segment": "原料层", "source_type": "SETR/中科院补充"},

    # ---- 无细胞、分布式、原位与组织制造 ----
    {"category": "前沿生物制造", "sub_category": "无细胞生物制造", "terms": ["无细胞生物制造", "无细胞蛋白质合成", "无细胞合成生物学", "无细胞表达系统", "无细胞生物合成", "细胞自由系统", "CFPS"], "score": 5, "match_type": "frontier", "industry_segment": "前沿制造", "source_type": "SETR直接词/科研机构补充"},
    {"category": "前沿生物制造", "sub_category": "分布式与按需制造", "terms": ["分布式生物制造", "按需生物制造", "现场生物制造", "原位生物制造", "便携式生物制造", "本地化生物制造", "去中心化生物制造", "生物制造网络"], "score": 5, "match_type": "frontier", "industry_segment": "前沿制造", "source_type": "SETR直接词", "source_detail": "SETR 2026标注页42—45、49"},
    {"category": "前沿生物制造", "sub_category": "活体与嵌入式生物技术", "terms": ["活体生物材料", "工程活体材料", "活体功能材料", "活体治疗菌", "工程益生菌", "工程微生物递送", "嵌入式生物技术", "环境响应生物系统", "生物传感植物"], "score": 4, "match_type": "contextual", "industry_segment": "前沿制造", "source_type": "SETR直接词/支持扩展"},
    {"category": "前沿生物制造", "sub_category": "3D生物打印", "terms": ["3D生物打印", "三维生物打印", "组织生物打印", "器官生物打印", "细胞打印", "生物墨水", "高密度细胞打印", "血管化组织打印", "SWIFT生物打印"], "score": 4, "match_type": "frontier", "industry_segment": "前沿制造", "source_type": "SETR直接词", "source_detail": "SETR 2026标注页47—48"},
    {"category": "前沿生物制造", "sub_category": "组织与器官工程制造", "terms": ["工程组织制造", "组织工程制造", "人工组织制造", "器官尺度制造", "可灌注组织", "血管化组织", "植入级组织", "组织构建体", "类器官制造"], "score": 4, "match_type": "contextual", "industry_segment": "前沿制造", "source_type": "SETR直接词/支持扩展"},
    {"category": "前沿生物制造", "sub_category": "干细胞与细胞规模制造", "terms": ["诱导多能干细胞", "iPSC", "干细胞规模化培养", "干细胞制造", "细胞规模化制造", "高密度细胞培养", "自动化细胞培养", "细胞聚集体制造"], "score": 3, "match_type": "contextual", "industry_segment": "前沿制造", "source_type": "SETR直接词/支持扩展"},

    # ---- 生物材料、化学品、能源、食品、农业、医药产品 ----
    {"category": "产品与应用", "sub_category": "生物基材料", "terms": ["生物基材料", "生物基聚合物", "生物基可降解材料", "生物基聚酰胺", "生物基聚酯", "微生物合成材料", "菌丝材料", "蛛丝蛋白材料", "重组蛋白材料", "生物基纤维"], "score": 2, "match_type": "product", "industry_segment": "应用产品", "source_type": "2024报告/资本研报补充"},
    {"category": "产品与应用", "sub_category": "可降解聚合物", "terms": ["聚乳酸", "PLA", "聚羟基脂肪酸酯", "PHA", "聚己二酸丁二醇酯", "生物基PEF", "聚呋喃二甲酸乙二醇酯", "生物尼龙", "聚戊二胺"], "score": 2, "match_type": "product", "industry_segment": "应用产品", "source_type": "2024报告/资本研报补充"},
    {"category": "产品与应用", "sub_category": "生物基化学品", "terms": ["生物基化学品", "平台化合物生物合成", "生物法长链二元酸", "生物基戊二胺", "生物基丁二酸", "生物基1,3-丙二醇", "生物基1,4-丁二醇", "生物基呋喃二甲酸", "生物基有机酸", "生物基芳香族化合物"], "score": 2, "match_type": "product", "industry_segment": "应用产品", "source_type": "2024报告/资本研报补充"},
    {"category": "产品与应用", "sub_category": "平台化合物", "terms": ["戊二酸", "丁二酸", "琥珀酸", "戊二胺", "1,4-丁二醇", "1,3-丙二醇", "呋喃二甲酸", "FDCA", "长链二元酸", "月桂二酸", "巴西酸", "衣康酸", "乳酸单体"], "score": 2, "match_type": "product", "industry_segment": "应用产品", "source_type": "2024报告直接词/资本研报补充"},
    {"category": "产品与应用", "sub_category": "氨基酸与有机酸发酵产品", "terms": ["赖氨酸", "谷氨酸", "苏氨酸", "蛋氨酸", "甲硫氨酸", "丙氨酸", "精氨酸", "柠檬酸", "葡萄糖酸", "苹果酸", "富马酸", "丙酮酸", "丙酸", "氨基酸发酵", "有机酸发酵"], "score": 1, "match_type": "product", "industry_segment": "应用产品", "source_type": "2024报告直接词"},
    {"category": "产品与应用", "sub_category": "天然产物与精细化学品", "terms": ["天然产物生物合成", "萜类生物合成", "黄酮生物合成", "生物碱生物合成", "甾体生物合成", "青蒿素生物合成", "虾青素生物合成", "麦角硫因生物合成", "依克多因生物合成", "红没药醇生物合成"], "score": 3, "match_type": "product", "industry_segment": "应用产品", "source_type": "2024报告/资本研报补充"},
    {"category": "产品与应用", "sub_category": "高值活性分子", "terms": ["青蒿素", "紫杉醇", "虾青素", "麦角硫因", "依克多因", "四氢嘧啶", "红没药醇", "红景天苷", "5-羟基色氨酸", "熊果苷", "NMN", "辅酶Q10", "核苷酸", "肌醇"], "score": 2, "match_type": "product", "industry_segment": "应用产品", "source_type": "2024报告直接词/资本研报补充"},
    {"category": "产品与应用", "sub_category": "生物能源与燃料", "terms": ["生物燃料", "燃料乙醇", "纤维素乙醇", "生物柴油", "生物天然气", "生物甲醇", "生物丁醇", "可持续航空燃料", "生物航空煤油", "SAF", "微生物产氢"], "score": 2, "match_type": "product", "industry_segment": "应用产品", "source_type": "2024报告直接词"},
    {"category": "产品与应用", "sub_category": "食品与营养", "terms": ["细胞培养肉", "培养肉", "替代蛋白", "微生物蛋白", "单细胞蛋白", "人造蛋白", "微生物油脂", "精准发酵乳蛋白", "合成乳蛋白", "人乳低聚糖", "HMO", "新型食品原料", "功能性食品原料"], "score": 2, "match_type": "product", "industry_segment": "应用产品", "source_type": "2024报告/资本研报补充"},
    {"category": "产品与应用", "sub_category": "工业酶制剂", "terms": ["工业酶制剂", "饲料酶", "食品酶制剂", "洗涤酶", "纺织酶", "造纸酶", "科研酶试剂", "蛋白酶制剂", "脂肪酶制剂", "纤维素酶制剂", "糖苷酶制剂"], "score": 2, "match_type": "product", "industry_segment": "应用产品", "source_type": "2024报告直接词/资本研报补充"},
    {"category": "产品与应用", "sub_category": "生物医药制造", "terms": ["重组蛋白药物", "重组多肽药物", "抗体药物生产", "疫苗生物制造", "细胞治疗产品制造", "基因治疗载体生产", "CAR-T细胞制备", "通用型CAR-T", "重组胶原蛋白", "医药中间体生物合成", "原料药生物制造"], "score": 2, "match_type": "product", "industry_segment": "应用产品", "source_type": "两份报告直接词/支持扩展"},
    {"category": "产品与应用", "sub_category": "重组药物与生物活性原料", "terms": ["重组胰岛素", "重组生长激素", "重组干扰素", "重组白细胞介素", "重组抗体", "单克隆抗体", "抗生素发酵", "维生素发酵", "胶原蛋白", "透明质酸", "肝素", "血液制品"], "score": 1, "match_type": "product", "industry_segment": "应用产品", "source_type": "2024报告直接词"},
    {"category": "产品与应用", "sub_category": "农业生物制造", "terms": ["农业合成生物学", "生物育种", "工程微生物菌剂", "合成生物肥料", "生物固氮", "微生物肥料", "生物农药", "饲料蛋白生物制造", "作物代谢工程"], "score": 2, "match_type": "product", "industry_segment": "应用产品", "source_type": "2024报告/资本研报补充"},
    {"category": "产品与应用", "sub_category": "环境修复与资源化", "terms": ["工程菌环境修复", "合成生物环境修复", "微生物资源化", "废弃物生物转化", "污染物生物降解", "塑料生物降解", "微生物降解塑料", "生物法废气资源化", "生物法废水资源化"], "score": 2, "match_type": "product", "industry_segment": "应用产品", "source_type": "2024报告/支持扩展"},
    {"category": "产品与应用", "sub_category": "生物电子与含能材料", "terms": ["生物制造电子材料", "生物合成电子材料", "生物制造含能材料", "生物合成含能材料", "生物制造推进剂", "生物合成磁性材料", "生物合成光纤", "DNA组装电子器件"], "score": 3, "match_type": "product", "industry_segment": "前沿制造", "source_type": "SETR直接词", "source_detail": "SETR 2026标注页48—49"},
    {"category": "产品与应用", "sub_category": "消费品与个护原料", "terms": ["化妆品活性原料生物合成", "个护原料生物合成", "透明质酸发酵", "胶原蛋白生物制造", "生物表面活性剂", "香料生物合成", "色素生物合成", "功能糖生物制造"], "score": 1, "match_type": "product", "industry_segment": "应用产品", "source_type": "2024报告/资本研报补充"},

    # ---- 安全、标准、装备与质量支撑 ----
    {"category": "安全质量与支撑", "sub_category": "生物安全与生物安保", "terms": ["生物制造安全", "合成生物安全", "工程菌生物安全", "生物安保", "生物遏制", "遗传防火墙", "营养缺陷型遏制", "自杀开关", "基因防逃逸", "环境释放风险", "镜像生命", "镜像微生物"], "score": 3, "match_type": "support", "industry_segment": "安全支撑", "source_type": "SETR直接词/支持扩展", "source_detail": "SETR 2026标注页50—52"},
    {"category": "安全质量与支撑", "sub_category": "质量控制与标准化", "terms": ["生物制造质量控制", "生物工艺质量控制", "菌株质量标准", "细胞库质量控制", "种子批系统", "主细胞库", "工作细胞库", "生物制造标准化", "批次一致性", "工艺验证"], "score": 3, "match_type": "support", "industry_segment": "安全支撑", "source_type": "工业生物技术支持扩展"},
    {"category": "安全质量与支撑", "sub_category": "生物制造装备", "terms": ["生物合成仪", "基因组装仪", "DNA合成仪", "自动化发酵设备", "菌种筛选设备", "高通量培养设备", "生物工艺装备", "生物制造中试装备"], "score": 3, "match_type": "support", "industry_segment": "底层工具", "source_type": "2024报告/资本研报补充"},
]


NEGATIVE_RULES: List[Dict] = [
    {"category": "传统食品发酵", "terms": ["白酒酿造", "白酒发酵", "啤酒酿造", "啤酒发酵", "葡萄酒酿造", "葡萄酒发酵", "黄酒酿造", "黄酒发酵", "酱油酿造", "酱油发酵", "食醋酿造", "食醋发酵", "泡菜发酵", "酸奶发酵", "面包发酵", "酒曲", "酿酒酵母制酒"], "penalty": 5},
    {"category": "纯诊断检测", "terms": ["仅用于诊断", "诊断试剂盒", "核酸检测试剂盒", "免疫检测试剂盒", "病理诊断", "影像诊断"], "penalty": 4},
    {"category": "非生物路线", "terms": ["纯化学合成", "石油化工路线", "天然提取法", "植物直接提取", "矿物加工"], "penalty": 4},
]


# 上下文词本身不计分，只用于使通用技术规则生效。
BIOENGINEERING_CONTEXT_TERMS = [
    "生物制造", "合成生物", "工业生物", "工程生物", "细胞工厂", "底盘细胞", "工程菌",
    "工程菌株", "重组菌", "生产菌株", "代谢工程", "酶工程", "蛋白质工程", "基因工程",
    "基因编辑", "基因组编辑", "基因合成", "DNA合成", "合成DNA", "基因组装", "生物合成", "生物催化", "酶催化", "全细胞催化", "微生物",
    "菌株", "细胞培养", "发酵", "无细胞", "生物反应器", "生物炼制", "生物转化",
]

PRODUCTION_CONTEXT_TERMS = [
    "制造", "生产", "制备", "合成", "发酵", "培养", "表达", "催化", "转化", "生物反应",
    "反应器", "发酵罐", "产率", "得率", "产量", "生产强度", "高产", "规模化", "工业化",
    "中试", "放大", "纯化", "分离", "工艺", "过程控制", "目标产物", "目标产品", "产物",
    "原料", "碳源", "底物", "代谢途径", "生产路线", "细胞工厂",
]

EXPLICIT_ENGINEERED_TERMS = [
    "生物制造", "合成生物", "工程生物", "工业生物", "细胞工厂", "底盘细胞", "DBTL",
    "工程菌", "重组菌", "代谢工程", "基因编辑", "基因合成", "生物催化", "酶工程",
    "无细胞生物制造", "分布式生物制造", "电生物合成", "生物铸造",
]


# ---------------------------------------------------------------------
# 2. 文本、正则和上下文工具
# ---------------------------------------------------------------------
def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value)


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
    escaped = re.escape(term.lower())
    if has_cn:
        return escaped.replace(r"\ ", r"\s*")
    escaped = escaped.replace(r"\ ", r"[\s\-]*").replace(r"\-", r"[\s\-]*")
    return r"(?<![a-z0-9])" + escaped + r"(?![a-z0-9])"


def compile_terms(terms: Sequence[str]) -> List[Tuple[str, re.Pattern]]:
    compiled = []
    for term in terms:
        regex = term_to_regex(term)
        if regex:
            compiled.append((term, re.compile(regex, flags=re.IGNORECASE)))
    return compiled


def compile_rules(rules: Sequence[Dict]) -> List[Dict]:
    compiled = []
    for item in rules:
        new_item = item.copy()
        new_item["patterns"] = compile_terms(item.get("terms", []))
        compiled.append(new_item)
    return compiled


def _make_union_regex(patterns: Sequence[Tuple[str, re.Pattern]]) -> re.Pattern:
    body = "|".join(pattern.pattern for _, pattern in patterns) or r"(?!)"
    return re.compile(body, flags=re.IGNORECASE)


COMPILED_RULES = compile_rules(BIOMANUFACTURING_RULES)
COMPILED_NEGATIVE = compile_rules(NEGATIVE_RULES)
BIOENGINEERING_PATTERNS = compile_terms(BIOENGINEERING_CONTEXT_TERMS)
PRODUCTION_PATTERNS = compile_terms(PRODUCTION_CONTEXT_TERMS)
EXPLICIT_ENGINEERED_PATTERNS = compile_terms(EXPLICIT_ENGINEERED_TERMS)

INDEPENDENT_TERMS = []
CONTEXTUAL_TERMS = []
for _rule in BIOMANUFACTURING_RULES:
    if _rule.get("match_type") in {"core", "frontier"}:
        INDEPENDENT_TERMS.extend(_rule.get("terms", []))
    else:
        CONTEXTUAL_TERMS.extend(_rule.get("terms", []))

INDEPENDENT_REGEX = _make_union_regex(compile_terms(sorted(set(INDEPENDENT_TERMS))))
CONTEXTUAL_REGEX = _make_union_regex(compile_terms(sorted(set(CONTEXTUAL_TERMS))))
BIOENGINEERING_REGEX = _make_union_regex(BIOENGINEERING_PATTERNS)
PRODUCTION_REGEX = _make_union_regex(PRODUCTION_PATTERNS)


def _has_any(text: str, patterns: Sequence[Tuple[str, re.Pattern]]) -> bool:
    return any(pattern.search(text) for _, pattern in patterns)


def _top_key(score_dict: Dict[str, int]) -> str:
    if not score_dict:
        return ""
    return max(score_dict.items(), key=lambda item: (item[1], item[0]))[0]


# ---------------------------------------------------------------------
# 3. 单条专利评分
# ---------------------------------------------------------------------
def score_one_patent_text(text: str) -> Dict:
    text = normalize_text(text)
    bioengineering_context = _has_any(text, BIOENGINEERING_PATTERNS)
    production_context = _has_any(text, PRODUCTION_PATTERNS)
    explicit_engineered = _has_any(text, EXPLICIT_ENGINEERED_PATTERNS)

    strong_core = False
    for item in COMPILED_RULES:
        if item.get("match_type") not in {"core", "frontier"}:
            continue
        if any(pattern.search(text) for _, pattern in item["patterns"]):
            strong_core = True
            break

    matched_terms: List[str] = []
    matched_core_terms: List[str] = []
    matched_context_terms: List[str] = []
    inactive_terms: List[str] = []
    matched_term_scores: Dict[str, int] = {}
    category_scores: Dict[str, int] = defaultdict(int)
    subcategory_scores: Dict[Tuple[str, str], int] = defaultdict(int)
    segment_scores: Dict[str, int] = defaultdict(int)
    source_types = set()
    source_details = set()
    source_urls = set()
    total_raw = 0
    max_keyword_score = 0

    for item in COMPILED_RULES:
        hits = [term for term, pattern in item["patterns"] if pattern.search(text)]
        if not hits:
            continue

        match_type = item.get("match_type", "contextual")
        valid = (
            match_type in {"core", "frontier"}
            or strong_core
            or (bioengineering_context and production_context)
        )
        if not valid:
            inactive_terms.extend(hits)
            continue

        score = int(item.get("score", 0))
        category = item.get("category", "")
        sub_category = item.get("sub_category", "")
        segment = item.get("industry_segment", "")

        # 同一规则的多个同义词只加一次，避免堆词抬高累计证据分。
        total_raw += score
        category_scores[category] += score
        subcategory_scores[(category, sub_category)] += score
        segment_scores[segment] += score
        max_keyword_score = max(max_keyword_score, score)
        source_types.add(item.get("source_type", ""))
        source_details.add(item.get("source_detail", ""))
        source_urls.add(item.get("source_url", ""))

        matched_terms.extend(hits)
        if match_type in {"core", "frontier"}:
            matched_core_terms.extend(hits)
        else:
            matched_context_terms.extend(hits)
        for term in hits:
            matched_term_scores[term] = max(score, matched_term_scores.get(term, 0))

    negative_terms = []
    for item in COMPILED_NEGATIVE:
        hits = [term for term, pattern in item["patterns"] if pattern.search(text)]
        negative_terms.extend(hits)

    # 最终分反映技术类型的最高核心程度，证据多少由raw分表达。
    total_score = max_keyword_score

    # 传统酿造、纯诊断或非生物路线只有在出现明确工程化生物制造词时才保留。
    if negative_terms and not explicit_engineered:
        total_score = 0
    elif negative_terms:
        total_score = max(0, total_score - 1)

    main_category = _top_key(dict(category_scores))
    main_sub_category = ""
    if subcategory_scores and main_category:
        same_category = {
            pair: score
            for pair, score in subcategory_scores.items()
            if pair[0] == main_category
        }
        if same_category:
            main_pair = max(
                same_category.items(),
                key=lambda item: (item[1], item[0][1]),
            )[0]
            main_sub_category = main_pair[1]
    industry_segment = _top_key(dict(segment_scores))

    return {
        "biomanufacturing_score_raw": total_raw,
        "biomanufacturing_score": total_score,
        "core_score": total_score,
        "max_matched_keyword_score": max_keyword_score,
        "matched_terms": "；".join(sorted(set(matched_terms))),
        "matched_core_terms": "；".join(sorted(set(matched_core_terms))),
        "matched_context_terms": "；".join(sorted(set(matched_context_terms))),
        "matched_term_scores": "；".join(
            f"{term}:{matched_term_scores[term]}" for term in sorted(matched_term_scores)
        ),
        "inactive_terms_no_context": "；".join(sorted(set(inactive_terms))),
        "negative_terms": "；".join(sorted(set(negative_terms))),
        "main_category": main_category,
        "main_sub_category": main_sub_category,
        "industry_segment": industry_segment,
        "category_scores": json.dumps(dict(category_scores), ensure_ascii=False, sort_keys=True),
        "subcategory_scores": json.dumps(
            {f"{cat}/{sub}": value for (cat, sub), value in subcategory_scores.items()},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "source_types": "；".join(sorted(value for value in source_types if value)),
        "source_details": "；".join(sorted(value for value in source_details if value)),
        "source_urls": "；".join(sorted(value for value in source_urls if value)),
        "has_bioengineering_context": int(bioengineering_context),
        "has_production_context": int(production_context),
    }


# ---------------------------------------------------------------------
# 4. DataFrame接口与企业汇总
# ---------------------------------------------------------------------
def make_text_series(
    df: pd.DataFrame,
    cn_abs_col: Optional[str] = "摘要 (中文)",
    en_abs_col: Optional[str] = None,
    extra_text_cols: Optional[Sequence[str]] = None,
) -> pd.Series:
    """合并实际存在的文本列；允许中文文本中的英文缩写和技术名。"""
    columns = [column for column in [cn_abs_col, en_abs_col] if column]
    if extra_text_cols:
        columns.extend(column for column in extra_text_cols if column)
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[existing].fillna("").astype(str).agg(" ".join, axis=1).map(normalize_text)


def _existing_region_cols(
    region_col: Optional[Union[str, Sequence[str]]],
    df: pd.DataFrame,
) -> List[str]:
    if region_col is None:
        return []
    columns = [region_col] if isinstance(region_col, str) else list(region_col)
    return [column for column in columns if column in df.columns]


def _join_unique_semicolon(series: pd.Series) -> str:
    values = []
    for text in series.dropna().astype(str):
        values.extend(value.strip() for value in text.split("；") if value.strip())
    return "；".join(sorted(set(values)))


def _most_common_nonempty(series: pd.Series) -> str:
    values = [str(value) for value in series.dropna() if str(value).strip()]
    return Counter(values).most_common(1)[0][0] if values else ""


def summarize_biomanufacturing_firms(
    patents: pd.DataFrame,
    firm_col: str = "第一申请人",
    year_col: str = "year",
    region_col: Optional[Union[str, Sequence[str]]] = None,
    firm_type_col: Optional[str] = None,
):
    """按第一申请人—地区—城市—年份及第一申请人—地区—城市汇总。"""
    if patents.empty or firm_col not in patents.columns:
        return pd.DataFrame(), pd.DataFrame()
    if year_col not in patents.columns:
        raise KeyError(f"缺少年份列：{year_col}")

    data = patents.copy()
    data = data[data[firm_col].notna()].copy()
    data[firm_col] = data[firm_col].astype(str).str.strip()
    data = data[data[firm_col] != ""].copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    region_cols = _existing_region_cols(region_col, data)
    group_cols_year = [firm_col] + region_cols + [year_col]
    group_cols_firm = [firm_col] + region_cols

    common_aggs = dict(
        biomanufacturing_patent_count=("is_biomanufacturing_patent", "sum"),
        biomanufacturing_score_sum=("biomanufacturing_score", "sum"),
        biomanufacturing_score_mean=("biomanufacturing_score", "mean"),
        biomanufacturing_score_max=("biomanufacturing_score", "max"),
        evidence_score_sum=("biomanufacturing_score_raw", "sum"),
        evidence_score_mean=("biomanufacturing_score_raw", "mean"),
        core_score_mean=("core_score", "mean"),
        core_score_max=("core_score", "max"),
        main_categories=("main_category", lambda x: "；".join(sorted(set(x.dropna()) - {""}))),
        main_sub_categories=("main_sub_category", lambda x: "；".join(sorted(set(x.dropna()) - {""}))),
        industry_segments=("industry_segment", lambda x: "；".join(sorted(set(x.dropna()) - {""}))),
        matched_terms=("matched_terms", _join_unique_semicolon),
    )
    year_aggs = common_aggs.copy()
    firm_aggs = dict(first_year=(year_col, "min"), last_year=(year_col, "max"), **common_aggs)
    if firm_type_col and firm_type_col in data.columns:
        year_aggs["first_applicant_types"] = (firm_type_col, _join_unique_semicolon)
        firm_aggs["first_applicant_types"] = (firm_type_col, _join_unique_semicolon)

    firm_year = data.groupby(group_cols_year, dropna=False).agg(**year_aggs).reset_index()
    firm = data.groupby(group_cols_firm, dropna=False).agg(**firm_aggs).reset_index()

    for source_col, target_col in [
        ("main_category", "firm_main_category"),
        ("main_sub_category", "firm_main_sub_category"),
        ("industry_segment", "firm_main_industry_segment"),
    ]:
        dominant = (
            data.groupby(group_cols_firm, dropna=False)[source_col]
            .agg(_most_common_nonempty)
            .reset_index()
            .rename(columns={source_col: target_col})
        )
        firm = firm.merge(dominant, on=group_cols_firm, how="left")

    return firm_year, firm


def tag_biomanufacturing_patents(
    df: pd.DataFrame,
    cn_abs_col: Optional[str] = "摘要 (中文)",
    en_abs_col: Optional[str] = None,
    firm_col: str = "第一申请人",
    year_col: str = "year",
    region_col: Optional[Union[str, Sequence[str]]] = None,
    firm_type_col: Optional[str] = None,
    extra_text_cols: Optional[Sequence[str]] = None,
    split_firms: bool = False,
    firm_sep_regex: str = r"[;；,，、|/]+",
    coarse_screen: bool = True,
    progress_every: int = 10000,
    min_score: int = 1,
):
    """返回tagged、正式专利、企业—年份汇总和企业汇总四个DataFrame。"""
    data = df.copy()
    text_series = make_text_series(data, cn_abs_col, en_abs_col, extra_text_cols)

    if coarse_screen:
        independent_mask = text_series.str.contains(INDEPENDENT_REGEX, regex=True, na=False)
        contextual_mask = (
            text_series.str.contains(BIOENGINEERING_REGEX, regex=True, na=False)
            & text_series.str.contains(PRODUCTION_REGEX, regex=True, na=False)
            & text_series.str.contains(CONTEXTUAL_REGEX, regex=True, na=False)
        )
        candidate_mask = independent_mask | contextual_mask
        data = data.loc[candidate_mask].copy()
        text_series = text_series.loc[candidate_mask]

    total = len(data)
    results = []
    start = time.time()
    for index, text in enumerate(text_series, start=1):
        results.append(score_one_patent_text(text))
        if progress_every and total and (index % progress_every == 0 or index == total):
            elapsed = time.time() - start
            speed = index / elapsed if elapsed else 0
            remain = (total - index) / speed if speed else 0
            print(
                f"已处理 {index:,}/{total:,} 条候选，占比 {index/total:.2%}，"
                f"已用 {elapsed/60:.1f} 分钟，预计剩余 {remain/60:.1f} 分钟"
            )

    score_columns = list(score_one_patent_text("").keys())
    score_frame = pd.DataFrame(results, index=data.index, columns=score_columns)
    patent_tagged = pd.concat([data, score_frame], axis=1)
    patent_tagged["is_biomanufacturing_patent"] = (
        patent_tagged["biomanufacturing_score"] >= min_score
    ).astype(int)

    if split_firms and firm_col in patent_tagged.columns:
        patent_tagged[firm_col] = patent_tagged[firm_col].fillna("").astype(str)
        patent_tagged["_firm_list"] = patent_tagged[firm_col].str.split(firm_sep_regex)
        patent_tagged = patent_tagged.explode("_firm_list")
        patent_tagged[firm_col] = patent_tagged["_firm_list"].str.strip()
        patent_tagged = patent_tagged[patent_tagged[firm_col] != ""].copy()
        patent_tagged.drop(columns=["_firm_list"], inplace=True)

    formal_patents = patent_tagged[
        patent_tagged["is_biomanufacturing_patent"] == 1
    ].copy()
    firm_year, firm = summarize_biomanufacturing_firms(
        formal_patents,
        firm_col=firm_col,
        year_col=year_col,
        region_col=region_col,
        firm_type_col=firm_type_col,
    )
    return patent_tagged, formal_patents, firm_year, firm


def export_keyword_dictionary() -> pd.DataFrame:
    """将规则展开为一行一个关键词，便于审查和导出CSV。"""
    rows = []
    for rule in BIOMANUFACTURING_RULES:
        for term in rule.get("terms", []):
            rows.append(
                {
                    "关键词": term,
                    "技术领域": rule.get("category", ""),
                    "细分方向": rule.get("sub_category", ""),
                    "产业板块": rule.get("industry_segment", ""),
                    "核心程度得分": rule.get("score", 0),
                    "匹配类型": rule.get("match_type", ""),
                    "上下文要求": (
                        "无需上下文"
                        if rule.get("match_type") in {"core", "frontier"}
                        else "需生物工程与生产制造上下文"
                    ),
                    "来源类型": rule.get("source_type", ""),
                    "来源说明": rule.get("source_detail", "") or "用户提供的两份生物制造相关报告",
                    "来源链接": rule.get("source_url", ""),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("请导入本模块并调用 tag_biomanufacturing_patents(df, ...)。")
    print(f"当前词典包含 {len(export_keyword_dictionary()):,} 个关键词/缩写。")
