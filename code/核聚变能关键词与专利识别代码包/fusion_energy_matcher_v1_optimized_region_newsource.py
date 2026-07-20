# -*- coding: utf-8 -*-
"""核聚变能专利匹配、技术路线分类与1—5分核心程度评分模块。"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List

import pandas as pd


def R(category, sub_category, terms, score, match_type, segment, source):
    return {"category": category, "sub_category": sub_category, "terms": terms,
            "score": score, "match_type": match_type, "industry_segment": segment,
            "source_type": source}


# ITER、DEMO、CFETR、EAST、ICF、HTS、REBCO、NBI等是中文聚变文献常用简称，
# 保留这些简称，但不为每个中文词机械复制英文同义词。
FUSION_ENERGY_RULES: List[Dict] = [
    R("基础与反应物理", "核心概念", ["核聚变", "可控核聚变", "受控核聚变", "核聚变能", "聚变能源", "聚变能发电", "核聚变发电", "聚变电站", "聚变堆", "聚变反应堆", "热核聚变", "人造太阳"], 5, "core", "聚变装置与系统", "附件论文/ITER/IAEA/资本研报"),
    R("基础与反应物理", "燃烧等离子体", ["燃烧等离子体", "聚变点火", "聚变增益", "能量增益因子", "聚变Q值", "劳逊判据", "三乘积", "自持燃烧", "阿尔法粒子自加热", "聚变燃烧控制"], 5, "core", "聚变科学", "附件论文/ITER/IAEA"),
    R("基础与反应物理", "聚变反应路线", ["氘氚聚变", "D-T聚变", "氘氘聚变", "D-D聚变", "氘氦三聚变", "D-He3聚变", "质子硼聚变", "p-B11聚变", "硼质子聚变", "无中子聚变", "低中子聚变"], 5, "core", "聚变科学", "附件论文/IAEA/科研论文"),
    R("基础与反应物理", "聚变燃料", ["聚变燃料", "氘氚燃料", "氘氚混合物", "重氢燃料", "氦三燃料", "硼十一燃料", "聚变燃料丸", "聚变靶丸"], 4, "core", "燃料循环", "附件论文/ITER/IAEA"),
    R("磁约束路线", "托卡马克", ["托卡马克", "托克马克", "Tokamak", "环形磁约束聚变", "全超导托卡马克", "紧凑型托卡马克", "先进托卡马克", "稳态托卡马克", "EAST装置", "CFETR", "ITER", "DEMO聚变堆", "SPARC聚变", "BEST聚变装置"], 5, "core", "聚变装置与系统", "附件论文/ITER/中科院/资本研报"),
    R("磁约束路线", "球形托卡马克", ["球形托卡马克", "球状托卡马克", "紧凑球形托卡马克", "低环径比托卡马克", "STEP聚变堆", "MAST-U"], 5, "core", "聚变装置与系统", "附件论文/UKAEA/资本研报"),
    R("磁约束路线", "仿星器", ["仿星器", "仿星器聚变", "螺旋器", "准轴对称仿星器", "准等磁仿星器", "模块化线圈仿星器", "W7-X", "LHD仿星器"], 5, "core", "聚变装置与系统", "附件论文/IAEA/资本研报"),
    R("磁约束路线", "磁镜与反场构型", ["磁镜聚变", "磁镜装置", "串联磁镜", "开端磁约束", "反场箍缩", "反向场箍缩", "RFP聚变", "场反位形", "场反构型", "FRC聚变", "球马克", "Spheromak"], 5, "core", "聚变装置与系统", "附件论文/IAEA/行业研究"),
    R("惯性约束路线", "激光惯性约束", ["惯性约束聚变", "激光聚变", "激光惯性约束聚变", "ICF聚变", "直接驱动聚变", "间接驱动聚变", "中心点火", "快点火", "冲击点火", "国家点火装置", "NIF聚变"], 5, "core", "聚变装置与系统", "附件论文/IAEA/科研机构"),
    R("惯性约束路线", "靶物理与靶制备", ["聚变靶", "聚变靶丸", "聚变冷冻靶丸", "聚变微球", "氘氚冰层", "聚变靶丸注入", "聚变靶丸跟踪", "聚变黑腔靶", "聚变黑腔", "聚变烧蚀层", "聚变内爆对称性", "聚变靶压缩", "内爆聚变"], 4, "core", "惯性聚变关键部件", "附件论文/科研机构补充"),
    R("惯性约束路线", "驱动器", ["聚变驱动激光器", "高功率聚变激光", "聚变高能激光驱动", "聚变激光束匀滑", "聚变激光脉冲整形", "聚变激光频率转换", "重离子驱动聚变", "离子束聚变驱动", "Z箍缩驱动聚变"], 4, "core", "惯性聚变关键部件", "科研机构/资本研报"),
    R("磁惯性与其他路线", "磁化靶聚变", ["磁化靶聚变", "磁化目标聚变", "磁惯性聚变", "磁惯性约束聚变", "MTF聚变", "等离子体内衬聚变", "液态金属内衬压缩"], 5, "core", "聚变装置与系统", "附件论文/IAEA行业数据库"),
    R("磁惯性与其他路线", "Z箍缩与等离子焦点", ["Z箍缩聚变", "Z-pinch聚变", "剪切流稳定Z箍缩", "等离子体箍缩", "致密等离子体焦点", "稠密等离子体焦点", "DPF聚变", "脉冲聚变"], 5, "core", "聚变装置与系统", "附件论文/科研资料"),
    R("磁惯性与其他路线", "惯性静电与束靶", ["惯性静电约束聚变", "IEC聚变", "静电约束聚变", "Fusor", "聚变中子管", "束靶聚变", "碰撞束聚变", "非热聚变"], 4, "core", "聚变装置与系统", "附件论文/IAEA行业数据库"),
    R("磁体与超导系统", "超导磁体系统", ["聚变超导磁体", "聚变磁体系统", "高场聚变磁体", "强磁场聚变", "环向场线圈", "极向场线圈", "中心螺线管", "校正场线圈", "TF线圈", "PF线圈", "CS线圈", "磁体馈线"], 4, "core", "主机关键系统", "附件论文/ITER/资本研报"),
    R("磁体与超导系统", "高温超导磁体", ["聚变高温超导磁体", "高温超导聚变磁体", "REBCO聚变磁体", "REBCO带材磁体", "稀土钡铜氧聚变磁体", "20T聚变磁体", "无绝缘聚变磁体", "可拆卸聚变磁体"], 5, "core", "主机关键系统", "附件论文/MIT-CFS论文/资本研报"),
    R("磁体与超导系统", "低温超导导体", ["Nb3Sn聚变磁体", "NbTi聚变磁体", "铌三锡超导线", "铌钛超导线", "管内电缆导体", "CICC导体", "超导股线", "超导缆线", "超导接头"], 4, "contextual", "关键材料与部件", "ITER/资本研报"),
    R("磁体与超导系统", "磁体保护与低温", ["聚变磁体失超保护", "聚变磁体淬火", "失超探测", "磁体机械应力", "超导线圈预紧", "聚变磁体低温系统", "超临界氦冷却", "磁体低温恒温器", "聚变磁体电源"], 4, "contextual", "主机关键系统", "ITER/科研论文"),
    R("等离子体加热与驱动", "中性束注入", ["聚变中性束注入", "聚变中性束加热", "聚变中性束电流驱动", "NBI加热", "聚变负离子中性束", "聚变中性束注入器", "聚变负离子束源", "聚变中性束中和器"], 4, "core", "主机关键系统", "附件论文/ITER"),
    R("等离子体加热与驱动", "电子回旋", ["电子回旋共振加热", "电子回旋加热", "电子回旋电流驱动", "ECRH", "ECCD", "毫米波等离子体加热", "聚变回旋管", "兆瓦级回旋管"], 4, "core", "主机关键系统", "附件论文/ITER"),
    R("等离子体加热与驱动", "离子回旋与低杂波", ["离子回旋共振加热", "离子回旋加热", "ICRH", "ICRF", "低杂波电流驱动", "低杂波加热", "LHCD", "低杂波天线", "射频聚变加热"], 4, "core", "主机关键系统", "附件论文/ITER"),
    R("等离子体加热与驱动", "欧姆与燃料注入", ["欧姆加热", "感应电流驱动", "聚变气体注入", "聚变燃料加注", "高速弹丸注入", "冷冻弹丸注入", "氘氚丸注入", "燃料丸注入器"], 3, "contextual", "主机关键系统", "ITER/科研机构"),
    R("等离子体控制与诊断", "平衡与输运控制", ["聚变等离子体平衡控制", "等离子体位形控制", "等离子体形状控制", "等离子体位置控制", "等离子体电流控制", "等离子体剖面控制", "聚变输运模拟", "托卡马克输运", "边界等离子体"], 4, "core", "控制诊断", "附件论文/ITER/科研机构"),
    R("等离子体控制与诊断", "不稳定性与破裂", ["聚变等离子体破裂", "托卡马克破裂", "破裂缓解系统", "大规模气体注入", "碎片弹丸注入", "逃逸电子抑制", "磁流体不稳定性", "撕裂模", "电阻壁模", "锯齿振荡", "边缘局域模", "ELM抑制"], 4, "core", "控制诊断", "附件论文/ITER"),
    R("等离子体控制与诊断", "诊断测量", ["聚变等离子体诊断", "托卡马克诊断", "汤姆孙散射诊断", "干涉仪诊断", "偏振仪诊断", "软X射线诊断", "轫致辐射诊断", "辐射热计", "磁探针诊断", "电子温度诊断", "离子温度诊断", "等离子体密度诊断"], 4, "contextual", "控制诊断", "附件论文/ITER"),
    R("等离子体控制与诊断", "聚变中子与粒子诊断", ["聚变中子诊断", "氘氚中子诊断", "中子通量监测", "中子谱仪", "聚变产物诊断", "阿尔法粒子诊断", "快离子诊断", "中子相机", "伽马谱聚变诊断"], 4, "core", "控制诊断", "附件论文/ITER"),
    R("等离子体控制与诊断", "智能控制与数字模型", ["聚变人工智能控制", "聚变等离子体机器学习", "托卡马克机器学习", "聚变数字孪生", "聚变PINN", "物理信息神经网络聚变", "聚变集成仿真", "聚变多物理场模拟", "等离子体实时控制"], 4, "core", "控制诊断", "附件论文/科研机构补充"),
    R("真空室与真空系统", "真空室", ["聚变真空室", "托卡马克真空室", "聚变双层真空室", "托卡马克D形真空室", "聚变真空室扇区", "聚变真空室港口", "聚变真空室焊接", "聚变真空容器", "聚变安全包容屏障"], 4, "core", "主机关键系统", "附件论文/ITER/资本研报"),
    R("真空室与真空系统", "真空与低温容器", ["聚变真空系统", "托卡马克真空系统", "低温泵聚变", "聚变真空泵", "聚变检漏", "聚变杜瓦", "托卡马克杜瓦", "聚变低温恒温器", "聚变冷屏", "低温真空绝热"], 3, "contextual", "配套系统", "附件论文/ITER/资本研报"),
    R("等离子体面对部件", "第一壁", ["聚变第一壁", "托卡马克第一壁", "聚变第一壁组件", "聚变第一壁装甲", "聚变第一壁冷却", "面向等离子体第一壁", "聚变钨第一壁", "聚变第一壁热负荷"], 4, "core", "关键材料与部件", "附件论文/ITER/资本研报"),
    R("等离子体面对部件", "偏滤器", ["聚变偏滤器", "托卡马克偏滤器", "偏滤器靶板", "偏滤器盒", "偏滤器穹顶", "钨偏滤器", "液态金属偏滤器", "偏滤器脱靶", "偏滤器热流", "Divertor"], 4, "core", "关键材料与部件", "附件论文/ITER/资本研报"),
    R("等离子体面对部件", "高热流部件", ["聚变高热流部件", "面向等离子体部件", "PFC部件", "钨铜模块", "钨铜连接", "CuCrZr热沉", "单体钨块", "钨铠甲", "高热流测试", "等离子体材料相互作用"], 4, "contextual", "关键材料与部件", "ITER/科研机构/资本研报"),
    R("包层与能量转换", "增殖包层", ["聚变增殖包层", "氚增殖包层", "产氚包层", "聚变堆包层", "测试包层模块", "TBM包层", "氚增殖比", "TBR", "氚自持", "包层中子学"], 5, "core", "燃料循环与能量转换", "附件论文/ITER/UKAEA/资本研报"),
    R("包层与能量转换", "固态增殖剂包层", ["固态增殖剂包层", "陶瓷增殖剂", "氦冷固态增殖包层", "HCPB包层", "水冷陶瓷增殖包层", "锂陶瓷小球", "偏钛酸锂", "正硅酸锂", "中子倍增材料", "铍中子倍增"], 4, "core", "燃料循环与能量转换", "ITER/UKAEA/科研机构"),
    R("包层与能量转换", "液态金属与熔盐包层", ["液态金属增殖包层", "锂铅包层", "铅锂共晶", "LiPb包层", "HCLL包层", "WCLL包层", "DCLL包层", "液态锂包层", "熔盐增殖包层", "FLiBe包层", "氟锂铍熔盐"], 4, "core", "燃料循环与能量转换", "ITER/UKAEA/资本研报"),
    R("包层与能量转换", "热工水力与发电循环", ["聚变包层冷却", "包层热工水力", "聚变堆氦冷却", "聚变堆水冷却", "聚变堆熔盐冷却", "聚变热交换器", "聚变蒸汽发生器", "聚变布雷顿循环", "聚变朗肯循环", "聚变热电转换"], 3, "contextual", "燃料循环与能量转换", "附件论文/ITER/UKAEA"),
    R("燃料循环", "氚增殖与提取", ["聚变氚增殖", "锂六产氚", "氚增殖材料", "氚提取系统", "包层氚提取", "氚渗透", "氚渗透屏障", "氚滞留", "氚回收", "氚迁移"], 5, "core", "燃料循环与能量转换", "附件论文/ITER/UKAEA"),
    R("燃料循环", "氚处理与同位素分离", ["聚变氚处理", "氚燃料循环", "氘氚燃料循环", "氢同位素分离", "低温精馏氚", "氚浓缩", "氚纯化", "氚水处理", "除氚系统", "去氚系统", "氚监测", "氚计量"], 5, "core", "燃料循环与能量转换", "附件论文/ITER/资本研报"),
    R("聚变材料", "低活化结构材料", ["聚变低活化钢", "低活化铁素体马氏体钢", "RAFM钢", "CLAM钢", "EUROFER钢", "F82H钢", "聚变结构材料", "低活化材料", "纳米析出强化钢", "ODS钢聚变"], 4, "core", "关键材料与部件", "附件论文/ITER/资本研报"),
    R("聚变材料", "难熔与复合材料", ["聚变钨材料", "聚变用钨合金", "钨基第一壁", "钨纤维增强钨", "钨铜复合材料", "铜铬锆合金聚变", "SiC/SiC聚变", "碳化硅复合材料聚变", "钒合金聚变"], 4, "core", "关键材料与部件", "附件论文/ITER/资本研报"),
    R("聚变材料", "辐照损伤", ["聚变中子辐照", "14MeV中子辐照", "聚变材料辐照", "位移损伤", "dpa损伤", "氦脆", "氢氦协同效应", "辐照肿胀", "辐照蠕变", "聚变材料活化", "中子嬗变"], 4, "contextual", "关键材料与部件", "附件论文/IFMIF/科研机构"),
    R("电源与脉冲功率", "线圈与加热电源", ["聚变线圈电源", "托卡马克磁体电源", "聚变脉冲电源", "聚变电源系统", "聚变整流电源", "聚变高压电源", "聚变射频电源", "聚变电源保护", "无功补偿聚变"], 3, "core", "配套系统", "附件论文/ITER/资本研报"),
    R("电源与脉冲功率", "储能与脉冲成形", ["聚变脉冲储能", "托卡马克储能", "聚变飞轮储能", "聚变电容储能", "脉冲成形网络聚变", "聚变磁体放电电阻", "聚变电网冲击抑制"], 3, "core", "配套系统", "附件论文/资本研报"),
    R("低温冷却与辅助系统", "低温冷却", ["聚变低温系统", "聚变氦制冷", "托卡马克低温系统", "聚变低温工厂", "聚变冷却水系统", "托卡马克冷却", "聚变热排出", "聚变余热排出"], 3, "core", "配套系统", "附件论文/ITER"),
    R("制造运维与机器人", "精密制造与检测", ["聚变装置精密制造", "聚变堆增材制造", "聚变部件3D打印", "聚变真空焊接", "聚变部件无损检测", "聚变部件热等静压", "聚变部件连接", "聚变装置装配"], 3, "contextual", "工程制造与服务", "附件论文/资本研报"),
    R("制造运维与机器人", "遥操作与维护", ["聚变远程维护", "聚变遥操作", "聚变遥处理", "聚变热室", "托卡马克检修机器人", "偏滤器维护机器人", "真空室内机器人", "聚变远程切割", "聚变远程焊接"], 3, "core", "工程制造与服务", "附件论文/ITER/UKAEA"),
    R("安全可靠性与监管", "聚变安全", ["聚变安全分析", "聚变堆事故分析", "聚变核安全", "聚变纵深防御", "聚变放射性包容", "聚变衰变热", "聚变放射性粉尘", "聚变安全屏障", "聚变源项"], 3, "core", "安全与服务", "附件论文/ITER/UKAEA"),
    R("安全可靠性与监管", "RAMI与许可", ["聚变RAMI", "聚变可靠性", "聚变可用性", "聚变可维护性", "聚变可检查性", "聚变质量保证", "聚变堆许可", "聚变监管", "聚变标准", "聚变核级鉴定"], 3, "core", "安全与服务", "附件论文/ITER"),
    R("应用与示范", "示范与商业化", ["聚变示范堆", "聚变工程实验堆", "聚变实验堆", "聚变原型堆", "聚变商业堆", "聚变示范电站", "聚变并网发电", "聚变商业化"], 4, "core", "终端应用", "附件论文/IAEA/资本研报"),
    R("应用与示范", "非电应用", ["聚变工业供热", "聚变制氢", "聚变中子源", "聚变材料试验中子源", "聚变同位素生产", "聚变航天推进", "聚变推进器", "聚变废物嬗变"], 2, "core", "终端应用", "附件论文/IAEA/科研机构"),
    # 下列为必须经过核聚变上下文门控的通用词。上下文确认后仍按其技术重要性评分。
    R("磁体与超导系统", "上下文通用磁体与导体", ["超导磁体", "高温超导磁体", "高场磁体", "REBCO带材", "REBCO磁体", "Nb3Sn导体", "NbTi导体", "超导线圈", "超导接头", "失超保护"], 4, "contextual", "主机关键系统", "ITER/科研论文/资本研报"),
    R("真空室与真空系统", "上下文通用真空与低温", ["真空室", "真空容器", "D形真空室", "真空泵", "低温泵", "杜瓦", "低温恒温器", "冷屏", "氦制冷机"], 4, "contextual", "主机关键系统", "ITER/资本研报"),
    R("等离子体面对部件", "上下文通用面对等离子体材料", ["第一壁", "偏滤器", "钨靶板", "钨合金", "钨铜复合材料", "CuCrZr", "高热流部件", "液态金属壁"], 4, "contextual", "关键材料与部件", "ITER/科研机构/资本研报"),
    R("等离子体控制与诊断", "上下文通用加热诊断控制", ["中性束注入", "电子回旋加热", "离子回旋加热", "低杂波电流驱动", "回旋管", "汤姆孙散射", "干涉仪", "磁探针", "等离子体实时控制"], 4, "contextual", "控制诊断", "ITER/科研机构"),
    R("制造运维与机器人", "上下文通用工程制造运维", ["真空焊接", "精密焊接", "增材制造", "热等静压", "无损检测", "远程维护", "遥操作", "检修机器人", "多物理场模拟"], 3, "contextual", "工程制造与服务", "ITER/UKAEA/资本研报"),
]

NEGATIVE_RULES = [
    {"terms": ["数据融合", "信息融合", "图像融合", "传感器融合", "特征融合", "多模态融合", "细胞融合", "脊柱融合", "骨融合", "核融合图像", "媒体融合"], "penalty": 5},
    {"terms": ["核裂变反应堆", "压水堆", "沸水堆", "快中子反应堆", "钠冷快堆", "钍基熔盐堆", "核燃料组件"], "penalty": 5},
]

FUSION_CONTEXT_TERMS = ["核聚变", "聚变能", "聚变堆", "聚变反应堆", "托卡马克", "仿星器", "惯性约束聚变", "磁约束聚变", "聚变等离子体", "氘氚聚变", "CFETR", "ITER"]
EXPLICIT_TERMS = FUSION_CONTEXT_TERMS + ["球形托卡马克", "磁化靶聚变", "Z箍缩聚变", "聚变增殖包层", "聚变氚燃料循环"]


def safe_text(v): return "" if pd.isna(v) else str(v)


def normalize_text(text):
    return safe_text(text).lower().replace("－", "-").replace("—", "-").replace("–", "-").replace("（", "(").replace("）", ")")


def term_to_regex(term):
    esc = re.escape(term.strip().lower()).replace(r"\ ", r"[\s\-]*")
    return esc if re.search(r"[\u4e00-\u9fff]", term) else r"(?<![a-z0-9])" + esc.replace(r"\-", r"[\s\-]*") + r"(?![a-z0-9])"


def compile_terms(terms): return [(t, re.compile(term_to_regex(t), re.I)) for t in terms if t.strip()]
def compile_rules(rules): return [dict(x, patterns=compile_terms(x.get("terms", []))) for x in rules]


COMPILED_RULES, COMPILED_NEGATIVE = compile_rules(FUSION_ENERGY_RULES), compile_rules(NEGATIVE_RULES)
FUSION_PATTERNS, EXPLICIT_PATTERNS = compile_terms(FUSION_CONTEXT_TERMS), compile_terms(EXPLICIT_TERMS)
INDEPENDENT_PATTERNS = compile_terms([t for r in FUSION_ENERGY_RULES if r["match_type"] == "core" for t in r["terms"]])
CONTEXTUAL_PATTERNS = compile_terms([t for r in FUSION_ENERGY_RULES if r["match_type"] != "core" for t in r["terms"]])


def _has(text, pats): return any(p.search(text) for _, p in pats)
def _union(pats): return re.compile("|".join(p.pattern for _, p in pats) or r"(?!)", re.I)
def _top(d): return max(d.items(), key=lambda x: (x[1], x[0]))[0] if d else ""


INDEPENDENT_REGEX, CONTEXTUAL_REGEX, FUSION_REGEX = _union(INDEPENDENT_PATTERNS), _union(CONTEXTUAL_PATTERNS), _union(FUSION_PATTERNS)


def score_one_patent_text(text: str) -> Dict:
    text = normalize_text(text); fusion_ctx, explicit = _has(text, FUSION_PATTERNS), _has(text, EXPLICIT_PATTERNS)
    strong = any(any(p.search(text) for _, p in r["patterns"]) for r in COMPILED_RULES if r["match_type"] == "core")
    matched, core_hits, context_hits, inactive = [], [], [], []
    term_scores, cats, subs, segments = {}, defaultdict(int), defaultdict(int), defaultdict(int)
    sources, raw, maximum = set(), 0, 0
    for r in COMPILED_RULES:
        hits = [t for t, p in r["patterns"] if p.search(text)]
        if not hits: continue
        if not (r["match_type"] == "core" or strong or fusion_ctx): inactive.extend(hits); continue
        s = int(r["score"]); raw += s; maximum = max(maximum, s)
        cats[r["category"]] += s; subs[(r["category"], r["sub_category"])] += s; segments[r["industry_segment"]] += s; sources.add(r["source_type"])
        matched.extend(hits); (core_hits if r["match_type"] == "core" else context_hits).extend(hits)
        for t in hits: term_scores[t] = max(term_scores.get(t, 0), s)
    negatives = [t for r in COMPILED_NEGATIVE for t, p in r["patterns"] if p.search(text)]
    score = maximum
    if negatives and not explicit: score = 0
    elif negatives: score = max(0, score - 1)
    main_cat = _top(cats); pool = {k: v for k, v in subs.items() if k[0] == main_cat}
    main_sub = max(pool.items(), key=lambda x: (x[1], x[0][1]))[0][1] if pool else ""
    return {"fusion_energy_score_raw": raw, "fusion_energy_score": score, "core_score": score,
            "max_matched_keyword_score": maximum, "matched_terms": "；".join(sorted(set(matched))),
            "matched_core_terms": "；".join(sorted(set(core_hits))), "matched_context_terms": "；".join(sorted(set(context_hits))),
            "matched_term_scores": "；".join(f"{t}:{term_scores[t]}" for t in sorted(term_scores)),
            "inactive_terms_no_context": "；".join(sorted(set(inactive))), "negative_terms": "；".join(sorted(set(negatives))),
            "main_category": main_cat, "main_sub_category": main_sub, "industry_segment": _top(segments),
            "category_scores": json.dumps(dict(cats), ensure_ascii=False, sort_keys=True),
            "subcategory_scores": json.dumps({f"{a}/{b}": v for (a, b), v in subs.items()}, ensure_ascii=False, sort_keys=True),
            "source_types": "；".join(sorted(sources)), "has_fusion_context": int(fusion_ctx)}


def make_text_series(df, cn_abs_col="摘要 (中文)", en_abs_col=None, extra_text_cols=None):
    cols = [c for c in ([cn_abs_col, en_abs_col] + list(extra_text_cols or [])) if c and c in df.columns]
    return (df[cols].fillna("").astype(str).agg(" ".join, axis=1).map(normalize_text) if cols else pd.Series([""]*len(df), index=df.index, dtype="object"))


def _join_unique(s):
    vals = [v.strip() for x in s.dropna().astype(str) for v in x.split("；") if v.strip()]
    return "；".join(sorted(set(vals)))


def _mode(s):
    vals = [str(v) for v in s.dropna() if str(v).strip()]
    return Counter(vals).most_common(1)[0][0] if vals else ""


def summarize_fusion_energy_firms(patents, firm_col="第一申请人", year_col="year", region_col=None, firm_type_col=None):
    if patents.empty or firm_col not in patents.columns: return pd.DataFrame(), pd.DataFrame()
    d = patents[patents[firm_col].notna()].copy(); d[firm_col] = d[firm_col].astype(str).str.strip(); d = d[d[firm_col] != ""]
    regions = ([region_col] if isinstance(region_col, str) else list(region_col or [])); regions = [c for c in regions if c in d.columns]
    gy, gf = [firm_col] + regions + [year_col], [firm_col] + regions
    common = dict(fusion_energy_patent_count=("is_fusion_energy_patent", "sum"), fusion_energy_score_sum=("fusion_energy_score", "sum"),
                  fusion_energy_score_mean=("fusion_energy_score", "mean"), fusion_energy_score_max=("fusion_energy_score", "max"),
                  evidence_score_sum=("fusion_energy_score_raw", "sum"), evidence_score_mean=("fusion_energy_score_raw", "mean"),
                  main_categories=("main_category", _join_unique), main_sub_categories=("main_sub_category", _join_unique),
                  industry_segments=("industry_segment", _join_unique), matched_terms=("matched_terms", _join_unique))
    ya, fa = common.copy(), dict(first_year=(year_col, "min"), last_year=(year_col, "max"), **common)
    if firm_type_col and firm_type_col in d.columns: ya["first_applicant_types"] = (firm_type_col, _join_unique); fa["first_applicant_types"] = (firm_type_col, _join_unique)
    fy = d.groupby(gy, dropna=False).agg(**ya).reset_index(); f = d.groupby(gf, dropna=False).agg(**fa).reset_index()
    for src, dst in [("main_category", "firm_main_category"), ("main_sub_category", "firm_main_sub_category"), ("industry_segment", "firm_main_industry_segment")]:
        dom = d.groupby(gf, dropna=False)[src].agg(_mode).reset_index().rename(columns={src: dst}); f = f.merge(dom, on=gf, how="left")
    return fy, f


def tag_fusion_energy_patents(df, cn_abs_col="摘要 (中文)", en_abs_col=None, firm_col="第一申请人", year_col="year",
                              region_col=None, firm_type_col=None, extra_text_cols=None, split_firms=False,
                              firm_sep_regex=r"[;；,，、|/]+", coarse_screen=True, progress_every=10000, min_score=1):
    data = df.copy(); texts = make_text_series(data, cn_abs_col, en_abs_col, extra_text_cols)
    if coarse_screen:
        mask = texts.str.contains(INDEPENDENT_REGEX, na=False) | (texts.str.contains(FUSION_REGEX, na=False) & texts.str.contains(CONTEXTUAL_REGEX, na=False))
        data, texts = data.loc[mask].copy(), texts.loc[mask]
    start, results = time.time(), []
    for i, text in enumerate(texts, 1):
        results.append(score_one_patent_text(text))
        if progress_every and (i % progress_every == 0 or i == len(data)):
            elapsed = time.time()-start; speed = i/elapsed if elapsed else 0; remain = (len(data)-i)/speed if speed else 0
            print(f"已处理 {i:,}/{len(data):,} 条候选，已用 {elapsed/60:.1f} 分钟，预计剩余 {remain/60:.1f} 分钟")
    scored = pd.DataFrame(results, index=data.index, columns=list(score_one_patent_text("").keys()))
    tagged = pd.concat([data, scored], axis=1); tagged["is_fusion_energy_patent"] = (tagged["fusion_energy_score"] >= min_score).astype(int)
    if split_firms and firm_col in tagged.columns:
        tagged["_firm"] = tagged[firm_col].fillna("").astype(str).str.split(firm_sep_regex); tagged = tagged.explode("_firm"); tagged[firm_col] = tagged["_firm"].str.strip(); tagged = tagged[tagged[firm_col] != ""].drop(columns="_firm")
    formal = tagged[tagged["is_fusion_energy_patent"] == 1].copy(); fy, f = summarize_fusion_energy_firms(formal, firm_col, year_col, region_col, firm_type_col)
    return tagged, formal, fy, f


def export_keyword_dictionary():
    return pd.DataFrame([{"关键词": t, "技术领域": r["category"], "细分方向": r["sub_category"], "产业板块": r["industry_segment"],
                          "核心程度得分": r["score"], "匹配类型": r["match_type"],
                          "上下文要求": "无需上下文" if r["match_type"] == "core" else "需核聚变上下文",
                          "来源类型": r["source_type"]} for r in FUSION_ENERGY_RULES for t in r["terms"]])


if __name__ == "__main__":
    print(f"核聚变能词典包含 {len(export_keyword_dictionary()):,} 个关键词/缩写。")
