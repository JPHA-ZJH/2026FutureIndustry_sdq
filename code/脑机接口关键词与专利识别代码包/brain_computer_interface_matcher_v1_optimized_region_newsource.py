# -*- coding: utf-8 -*-
"""脑机接口专利匹配、技术分类与1—5分核心程度评分模块。"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List

import pandas as pd


def R(category, sub_category, terms, score, match_type, segment, source):
    return {"category": category, "sub_category": sub_category, "terms": terms,
            "score": score, "match_type": match_type,
            "industry_segment": segment, "source_type": source}


# 以中文术语为主体；BCI、EEG、ECoG、SEEG、LFP、SSVEP、P300、DBS等
# 是中文脑机接口报告和专利中常见缩写，直接保留，不另建重复英文词典。
BCI_RULES: List[Dict] = [
    R("基础概念与系统范式", "领域锚点", ["脑机接口", "脑-机接口", "脑计算机接口", "脑-计算机接口", "脑机器接口", "脑-机器接口", "脑机交互", "脑-机交互", "脑机通信", "脑机智能", "BCI技术"], 1, "core", "系统与平台", "两份附件直接词/中国信通院/工信部"),
    R("基础概念与系统范式", "核心系统", ["脑机接口系统", "脑计算机接口系统", "脑机器接口系统", "脑机交互系统", "脑机通信系统", "BCI系统", "闭环脑机系统", "双向脑机系统", "脑机一体化系统"], 5, "core", "系统与平台", "两份附件直接词/中国信通院/工信部"),
    R("基础概念与系统范式", "神经接口与神经工程", ["脑神经接口", "中枢神经接口", "皮层神经接口", "神经接口系统", "神经工程接口", "神经电子接口", "神经-机器接口", "神经机器接口", "神经假体接口"], 4, "contextual", "系统与平台", "附件报告/FDA/科研机构"),
    R("基础概念与系统范式", "脑控与意念交互", ["脑控技术", "脑控系统", "脑控设备", "脑控指令", "脑控交互", "意念控制", "意念操控", "意图驱动控制", "思维控制外设", "神经意图控制"], 5, "core", "系统与平台", "两份附件直接词/资本研报"),
    R("基础概念与系统范式", "双向与闭环脑机接口", ["双向脑机接口", "闭环脑机接口", "闭环神经接口", "双向神经接口", "脑机闭环", "脑机双向交互", "感知-调控闭环", "感知调控一体化", "采集刺激闭环", "记录刺激闭环", "神经感知反馈闭环"], 5, "core", "系统与平台", "中国信通院2025/科研机构"),
    R("基础概念与系统范式", "脑脊与脑脑接口", ["脑脊接口", "脑-脊接口", "脑脊髓接口", "脑-脊髓接口", "脑脑接口", "脑-脑接口", "脑对脑接口", "群体脑机接口", "多人脑机接口", "超脑网络"], 5, "core", "系统与平台", "中国信通院/资本研报/科研机构"),
    R("基础概念与系统范式", "发展阶段与融合", ["交互式脑感知", "感知式脑调控", "脑感知与脑调控融合", "脑感知调控融合", "脑机深度协同", "人机智能融合", "碳基硅基融合", "神经信息闭环交互"], 5, "core", "系统与平台", "中国信通院2025直接词"),

    R("接口植入路线", "侵入式", ["侵入式脑机接口", "有创脑机接口", "全植入脑机接口", "完全植入式脑机接口", "颅内脑机接口", "皮层内脑机接口", "脑内植入接口", "植入式BCI", "侵入式BCI"], 5, "core", "系统与平台", "两份附件/中国信通院/资本研报"),
    R("接口植入路线", "半侵入与微创", ["半侵入式脑机接口", "半植入式脑机接口", "微创脑机接口", "微创植入脑机接口", "硬膜外脑机接口", "硬膜下脑机接口", "皮层表面脑机接口", "脑膜表面脑机接口", "微创BCI"], 5, "core", "系统与平台", "中国信通院/资本研报"),
    R("接口植入路线", "介入式与血管内", ["介入式脑机接口", "血管介入脑机接口", "血管内脑机接口", "脑血管内电极", "血管内神经接口", "支架电极脑机接口", "脑静脉窦电极", "Stentrode", "血管内BCI"], 5, "core", "系统与平台", "中国信通院/资本研报/科研机构"),
    R("接口植入路线", "非侵入式与可穿戴", ["非侵入式脑机接口", "非侵入脑机接口", "无创脑机接口", "无创BCI", "非侵入式BCI", "可穿戴脑机接口", "头戴式脑机接口", "耳戴式脑机接口", "便携式脑机接口"], 5, "core", "系统与平台", "两份附件/中国信通院/资本研报"),
    R("接口植入路线", "混合脑机接口", ["混合脑机接口", "多模态脑机接口", "混合BCI", "多模态BCI", "EEG-fNIRS脑机接口", "EEG-fMRI脑机接口", "EEG-眼电混合接口", "EEG-肌电混合接口", "脑肌接口", "脑眼接口"], 5, "core", "系统与平台", "两份附件/科研机构"),

    R("脑感知-电信号", "头皮脑电EEG", ["脑机接口脑电", "BCI脑电", "脑电脑机接口", "EEG脑机接口", "脑电信号脑控", "头皮脑电接口", "高密度脑电接口", "无线脑电脑机接口", "移动脑电脑机接口"], 5, "core", "采集感知", "两份附件/中国信通院/科研机构"),
    R("脑感知-电信号", "皮层与深部电信号", ["皮层脑电接口", "ECoG脑机接口", "颅内脑电脑机接口", "SEEG脑机接口", "局部场电位脑机接口", "LFP脑机接口", "单神经元脑机接口", "Spike脑机接口", "神经元放电脑机接口"], 5, "core", "采集感知", "两份附件/中国信通院/科研机构"),
    R("脑感知-电信号", "信号类型通用词", ["脑电信号", "头皮脑电", "颅内脑电", "皮层脑电", "脑皮层电位", "局部场电位", "单元放电", "神经元动作电位", "神经尖峰", "诱发电位", "事件相关电位"], 4, "contextual", "采集感知", "两份附件直接词"),
    R("脑感知-磁光超声", "脑磁MEG", ["脑磁脑机接口", "MEG脑机接口", "脑磁图脑机接口", "SQUID脑磁图", "超导量子干涉脑磁图", "原子磁力计脑磁图", "光泵磁力计脑磁图", "OPM-MEG", "可穿戴脑磁图", "无液氦脑磁图"], 4, "core", "采集感知", "中国信通院2025/科研机构"),
    R("脑感知-磁光超声", "功能近红外", ["fNIRS脑机接口", "功能近红外脑机接口", "近红外脑机接口", "功能近红外光谱脑机接口", "EEG-fNIRS融合", "近红外脑功能成像接口", "血氧脑机接口", "血红蛋白浓度脑机接口"], 4, "core", "采集感知", "两份附件/中国信通院"),
    R("脑感知-磁光超声", "功能磁共振", ["fMRI脑机接口", "功能磁共振脑机接口", "实时功能磁共振神经反馈", "rt-fMRI神经反馈", "BOLD脑机接口", "功能磁共振意图解码"], 4, "core", "采集感知", "两份附件/科研机构"),
    R("脑感知-磁光超声", "功能超声", ["功能超声脑机接口", "超快功能超声成像", "fUSI脑机接口", "功能超声神经成像", "超声脑信号感知", "血流动力学脑信号感知", "微型超声脑接口", "超声采集调控芯片"], 4, "core", "采集感知", "中国信通院2025/科研机构"),
    R("脑感知-磁光超声", "多模态成像与感知", ["多模态脑信号采集", "脑电脑磁融合", "脑电近红外融合", "脑电磁共振融合", "电磁光超声脑感知", "多模态神经成像", "同步脑电近红外", "同步EEG-fNIRS"], 4, "core", "采集感知", "中国信通院/科研机构"),

    R("神经电极与探针", "犹他阵列与微针", ["犹他电极阵列", "犹他阵列", "Utah Array", "皮层内微电极阵列", "微针神经电极", "微电极阵列脑机接口", "三维微电极阵列", "高密度微电极阵列", "穿刺式神经电极"], 4, "core", "上游核心器件", "两份附件/中国信通院/FDA"),
    R("神经电极与探针", "Neuropixels与神经探针", ["Neuropixels", "Neuropixel电极", "神经像素探针", "硅基神经探针", "高密度神经探针", "多通道神经探针", "微纳神经探针", "单神经元记录探针", "CMOS神经探针"], 4, "core", "上游核心器件", "两份附件/科研机构"),
    R("神经电极与探针", "柔性与超柔性电极", ["脑机柔性电极", "脑机超柔性电极", "柔性微丝电极", "超柔性微丝电极", "神经微丝电极", "柔性神经电极阵列", "超薄神经电极", "网状神经电极", "蛇形神经电极", "可拉伸神经电极", "自适应脑组织电极"], 4, "core", "上游核心器件", "中国信通院/资本研报/科研机构"),
    R("神经电极与探针", "通用电极器件词", ["柔性电极", "超柔性电极", "微丝电极", "微电极阵列", "高密度电极阵列", "多通道电极阵列", "植入电极", "皮层电极", "深部电极", "神经探针"], 4, "contextual", "上游核心器件", "附件报告/中国信通院/上下文补充"),
    R("神经电极与探针", "皮层表面电极", ["ECoG电极阵列", "皮层脑电电极", "皮层表面微电极", "硬膜下电极阵列", "硬膜外电极阵列", "脑膜电极", "薄膜皮层电极", "高密度ECoG", "微ECoG", "μECoG"], 4, "core", "上游核心器件", "两份附件/中国信通院"),
    R("神经电极与探针", "深部与立体脑电电极", ["SEEG电极", "立体脑电电极", "深部脑电电极", "深脑记录电极", "脑深部微电极", "深部刺激记录电极", "深脑双向电极", "深部神经探针"], 4, "contextual", "上游核心器件", "两份附件/科研机构"),
    R("神经电极与探针", "无创干湿电极", ["脑机干电极", "脑电干电极", "干电极", "微针干电极", "入耳式脑电电极", "耳脑电电极", "凝胶脑电电极", "湿式脑电电极", "湿电极", "盐水脑电电极", "主动脑电电极", "半干式脑电电极", "脑电帽电极", "脑电帽"], 4, "contextual", "上游核心器件", "两份附件/中国信通院/上下文补充"),
    R("神经电极与探针", "电极界面与性能", ["神经电极阻抗", "电极脑组织贴合", "脑电极信噪比", "电极电荷注入容量", "神经电极电荷密度", "电极接触阻抗", "脑电极通道密度", "神经电极长期稳定性", "神经电极漂移", "神经电极封装"], 3, "contextual", "上游核心器件", "附件报告/FDA/科研机构"),

    R("电极材料与微纳制造", "导电与贵金属材料", ["脑电极铂铱", "神经电极铂铱合金", "神经电极铂黑", "脑电极金", "脑电极银氯化银", "Ag/AgCl脑电极", "铱氧化物神经电极", "钛氮神经电极", "多孔铂神经电极"], 4, "contextual", "上游材料", "两份附件/科研机构"),
    R("电极材料与微纳制造", "碳与导电聚合物", ["脑电极石墨烯", "神经电极石墨烯", "碳纳米管脑电极", "碳纤维神经电极", "PEDOT:PSS神经电极", "导电聚合物神经电极", "聚吡咯神经电极", "有机电化学晶体管神经接口", "OECT神经接口"], 4, "contextual", "上游材料", "附件报告/科研机构"),
    R("电极材料与微纳制造", "柔性基底与水凝胶", ["神经电极聚酰亚胺", "神经电极聚对二甲苯", "Parylene神经电极", "神经电极水凝胶", "导电水凝胶脑电极", "细菌纤维素脑电极", "丝素蛋白神经电极", "柔性塑料神经探针", "可降解神经电极", "脑组织模量匹配材料"], 4, "contextual", "上游材料", "中国信通院2025/科研机构"),
    R("电极材料与微纳制造", "生物相容与界面涂层", ["神经电极生物相容性", "脑植入电极生物相容", "抗炎神经电极涂层", "抗胶质瘢痕涂层", "神经电极药物洗脱", "脑电极抗蛋白吸附", "神经界面组织整合", "神经电极表面修饰"], 4, "contextual", "上游材料", "FDA/科研机构/附件报告"),
    R("电极材料与微纳制造", "MEMS与微纳加工", ["脑机接口MEMS", "神经电极MEMS", "神经探针微纳加工", "脑电极光刻", "神经电极电子束曝光", "神经电极等离子刻蚀", "神经电极薄膜沉积", "神经探针晶圆级制造", "神经电极微纳封装"], 3, "contextual", "上游制造", "两份附件/资本研报"),

    R("脑机芯片与电路", "采集模拟前端", ["脑信号采集芯片", "脑电采集芯片", "神经信号采集芯片", "神经记录芯片", "神经模拟前端", "神经AFE", "低噪声神经放大器", "脑电前置放大器", "神经信号模数转换器", "神经ADC"], 4, "core", "上游核心器件", "两份附件/工信部/资本研报"),
    R("脑机芯片与电路", "高通道采集", ["高通道脑信号芯片", "高密度神经采集芯片", "千通道神经记录", "多通道脑电采集芯片", "通道复用神经芯片", "高通量脑机芯片", "脑信号通道管理", "脑电同步采集芯片"], 4, "core", "上游核心器件", "中国信通院/工信部/资本研报"),
    R("脑机芯片与电路", "刺激与记录一体", ["神经采集刺激一体化芯片", "脑信号采集刺激芯片", "神经记录刺激芯片", "闭环神经调控芯片", "双向神经接口芯片", "脑机接口SoC", "神经接口片上系统", "感知计算调控一体芯片"], 5, "core", "上游核心器件", "中国信通院/工信部/资本研报"),
    R("脑机芯片与电路", "低功耗与无线通信", ["超低功耗脑机芯片", "低功耗神经处理芯片", "无线神经接口芯片", "植入式无线脑机芯片", "脑信号通信芯片", "神经数据无线传输", "脑机接口遥测", "反向散射神经接口", "植入体无线供能", "经皮无线供电"], 4, "core", "上游核心器件", "中国信通院/工信部/FDA"),
    R("脑机芯片与电路", "存算传一体与神经形态", ["脑机存算一体", "脑机计算存储传输一体化", "忆阻器脑电解码", "忆阻器神经接口", "神经形态脑机芯片", "类脑解码芯片", "片上脑信号解码", "边缘脑信号处理", "脑机端侧计算"], 4, "core", "上游核心器件", "两份附件/中国信通院/工信部"),
    R("脑机芯片与电路", "封装可靠性", ["脑机植入体气密封装", "神经植入芯片封装", "陶瓷气密封装脑机", "植入体馈通", "脑机接口柔性互连", "植入芯片热管理", "神经芯片防水封装", "脑机植入体小型化"], 3, "contextual", "上游制造", "FDA/资本研报/科研机构"),

    R("信号采集与传输", "采集系统", ["脑机信号采集系统", "脑电采集系统BCI", "神经信号记录系统", "颅内神经记录系统", "高密度脑电系统", "多导脑电采集", "脑电同步器", "神经数据采集器", "脑机采集终端"], 4, "core", "中游采集分析", "两份附件/中国信通院"),
    R("信号采集与传输", "放大滤波与抗干扰", ["脑信号放大", "脑电低噪声放大", "神经信号滤波", "脑电陷波", "脑电共模抑制", "神经信号伪迹抑制", "刺激伪迹消除", "脑电运动伪迹消除", "脑电工频干扰抑制", "脑电信号质量评估"], 3, "contextual", "中游采集分析", "两份附件/工信部"),
    R("信号采集与传输", "压缩同步与高速传输", ["脑信号压缩", "神经数据压缩", "脑电无损压缩", "脑信号时间同步", "多通道神经数据同步", "脑信号高速传输", "神经数据低时延传输", "脑机无线通信协议", "脑机接口通信协议", "脑机数据链路"], 3, "contextual", "中游采集分析", "两份附件/工信部"),

    R("信号预处理与特征", "预处理与伪迹去除", ["脑机信号预处理", "脑电信号预处理", "脑电去噪", "脑电伪迹去除", "眼电伪迹去除", "肌电伪迹去除", "脑电基线校正", "脑电重参考", "独立成分分析脑电", "ICA脑电", "小波脑电去噪"], 3, "contextual", "中游算法", "两份附件/科研论文"),
    R("信号预处理与特征", "空间滤波与源定位", ["脑电空间滤波", "共空间模式", "CSP脑电", "滤波器组共空间模式", "FBCSP", "xDAWN脑电", "脑电源定位", "皮层源定位", "逆问题脑电", "波束形成脑磁", "头模型脑电"], 4, "contextual", "中游算法", "两份附件/科研论文"),
    R("信号预处理与特征", "时频与连接特征", ["脑电时频分析", "脑电频谱特征", "事件相关去同步", "事件相关同步", "ERD/ERS", "脑电功率谱密度", "脑网络功能连接", "脑电相位同步", "脑电相干性", "神经振荡特征", "脑电图特征提取"], 3, "contextual", "中游算法", "两份附件/科研机构"),
    R("信号预处理与特征", "尖峰排序", ["神经尖峰排序", "Spike sorting", "神经元信号分选", "Kilosort", "模板匹配尖峰", "在线尖峰检测", "单神经元信号提取", "动作电位分类", "神经放电聚类"], 4, "core", "中游算法", "五道口报告直接词/科研机构"),

    R("脑机实验范式", "P300与ERP", ["P300脑机接口", "P300拼写器", "P300字符输入", "事件相关电位脑机接口", "ERP脑机接口", "奇异刺激脑机接口", "oddball脑机接口"], 5, "core", "中游算法", "两份附件/科研论文"),
    R("脑机实验范式", "SSVEP与视觉诱发", ["SSVEP脑机接口", "稳态视觉诱发电位脑机接口", "视觉诱发电位脑机接口", "VEP脑机接口", "视觉编码脑机接口", "频率编码SSVEP", "相位编码SSVEP", "视觉闪烁脑机接口"], 5, "core", "中游算法", "两份附件/资本研报"),
    R("脑机实验范式", "运动想象与感觉运动节律", ["运动想象脑机接口", "MI脑机接口", "感觉运动节律脑机接口", "SMR脑机接口", "运动意图脑机接口", "运动想象解码", "左右手运动想象", "运动皮层意图解码"], 5, "core", "中游算法", "两份附件/资本研报"),
    R("脑机实验范式", "慢皮层电位与听觉", ["慢皮层电位脑机接口", "SCP脑机接口", "听觉脑机接口", "听觉诱发电位脑机接口", "稳态听觉诱发电位", "ASSR脑机接口", "耳蜗脑机反馈"], 4, "core", "中游算法", "五道口报告/科研论文"),
    R("脑机实验范式", "言语情感与被动式", ["言语脑机接口", "语音脑机接口", "情感脑机接口", "注意力脑机接口", "被动式脑机接口", "主动式脑机接口", "反应式脑机接口", "异步脑机接口", "自定步速脑机接口"], 5, "core", "中游算法", "两份附件/科研机构"),

    R("神经解码与智能算法", "运动意图解码", ["脑信号运动意图解码", "神经运动解码", "运动皮层解码", "手势意图解码", "抓握意图解码", "步态意图解码", "连续运动轨迹解码", "光标运动解码", "脑控运动分类"], 4, "core", "中游算法", "两份附件/资本研报"),
    R("神经解码与智能算法", "言语与语言解码", ["脑信号语音解码", "神经语音解码", "皮层语音解码", "ECoG语音解码", "脑信号汉语言解码", "脑信号文字解码", "神经手写解码", "意念打字", "脑控打字", "脑信号语音合成", "失语沟通解码"], 5, "core", "中游算法", "中国信通院/资本研报/科研机构"),
    R("神经解码与智能算法", "视觉语义与状态解码", ["脑信号视觉重建", "脑活动图像重建", "神经视觉解码", "脑信号语义解码", "脑状态解码", "认知状态识别", "疲劳脑电识别", "情绪脑电识别", "注意力脑电识别", "意识状态脑机评估"], 4, "contextual", "中游算法", "两份附件/中国信通院"),
    R("神经解码与智能算法", "机器学习与深度学习", ["脑机机器学习", "脑电支持向量机", "SVM脑电分类", "脑电卷积神经网络", "CNN脑电", "脑电循环神经网络", "RNN脑电", "脑电Transformer", "神经信号图神经网络", "脑电自编码器", "脑信号强化学习", "支持向量机", "SVM", "卷积神经网络", "CNN", "循环神经网络", "RNN", "长短期记忆网络", "LSTM", "Transformer", "图神经网络", "自编码器", "深度学习", "机器学习", "强化学习"], 4, "contextual", "中游算法", "两份附件/科研论文/上下文补充"),
    R("神经解码与智能算法", "迁移泛化与少校准", ["脑电迁移学习", "脑机域适应", "跨被试脑电解码", "被试独立脑电分类", "免校准脑机接口", "少校准脑机接口", "脑电小样本学习", "脑电自监督学习", "脑机在线自适应", "跨会话脑电解码", "脑机算法可解释性"], 4, "core", "中游算法", "中国信通院/科研论文"),
    R("神经解码与智能算法", "实时与边缘解码", ["实时脑电解码", "在线脑信号分类", "低时延神经解码", "脑机端到端解码", "脑机边缘推理", "片上神经解码", "脑机实时反馈算法", "神经解码器自适应"], 4, "core", "中游算法", "两份附件/工信部"),

    R("神经编码与感觉反馈", "感觉编码", ["神经感觉编码", "脑刺激信息编码", "神经刺激编码", "人工感觉编码", "触觉神经编码", "视觉神经编码", "听觉神经编码", "感觉替代编码", "外部信息神经编码"], 4, "core", "中游编解码", "中国信通院2025/科研机构"),
    R("神经编码与感觉反馈", "皮层微刺激与人工感觉", ["皮层内微刺激", "ICMS触觉反馈", "体感皮层刺激", "运动皮层刺激", "视觉皮层刺激", "人工触觉反馈", "人工本体感觉", "双向神经假体", "感觉反馈脑机接口", "闭环触觉脑机接口"], 5, "core", "中游编解码", "中国信通院/FDA/科研机构"),
    R("神经编码与感觉反馈", "神经反馈", ["脑机神经反馈", "EEG神经反馈", "fNIRS神经反馈", "实时神经反馈", "闭环神经反馈", "多模态神经反馈", "虚拟现实神经反馈", "脑状态反馈训练", "神经反馈自适应调控"], 4, "core", "中游编解码", "两份附件/中国信通院"),

    R("神经调控-电刺激", "深部脑刺激", ["闭环深部脑刺激", "自适应深部脑刺激", "感知式深部脑刺激", "脑电感知DBS", "闭环DBS", "aDBS", "记录刺激一体脑起搏器", "双向脑起搏器", "脑深部刺激反馈控制"], 5, "core", "调控与治疗", "中国信通院2025/FDA/资本研报"),
    R("神经调控-电刺激", "脊髓与外周刺激闭环", ["脑机脊髓电刺激", "脑控脊髓刺激", "闭环脊髓电刺激", "脑机功能性电刺激", "脑控功能性电刺激", "BCI-FES", "脑机外周神经刺激", "脑控肌肉电刺激", "意图驱动电刺激"], 5, "core", "调控与治疗", "中国信通院/资本研报/科研机构"),
    R("神经调控-电刺激", "无创经颅电刺激", ["脑机经颅直流电刺激", "闭环tDCS", "EEG-tDCS闭环", "脑机经颅交流电刺激", "闭环tACS", "EEG-tACS闭环", "经颅随机噪声刺激脑机", "tRNS脑机", "高精度经颅电刺激"], 4, "core", "调控与治疗", "中国信通院2025/科研机构"),
    R("神经调控-磁光超声", "经颅磁刺激", ["脑机经颅磁刺激", "EEG-TMS闭环", "闭环经颅磁刺激", "实时脑电引导TMS", "重复经颅磁刺激脑机", "rTMS脑机", "θ爆发刺激脑机", "TBS脑机", "模式化磁刺激", "pTMS"], 4, "core", "调控与治疗", "中国信通院2025"),
    R("神经调控-磁光超声", "超声神经调控", ["脑机超声调控", "闭环超声神经调控", "经颅聚焦超声神经调控", "低强度聚焦超声脑调控", "LIFU脑调控", "经颅超声刺激", "TUS神经调控", "影像引导超声脑调控", "超声血脑屏障调控"], 4, "core", "调控与治疗", "中国信通院2025/科研机构"),
    R("神经调控-磁光超声", "光调控与光遗传", ["脑机光调控", "闭环光遗传调控", "光遗传脑机接口", "经颅光生物调节", "tPBM脑调控", "近红外脑调控", "多波长脑刺激", "光遗传神经反馈", "光电一体神经接口"], 4, "core", "调控与治疗", "两份附件/中国信通院"),
    R("神经调控-磁光超声", "多模态组合调控", ["多模态脑调控", "电磁组合脑刺激", "电光神经调控", "磁声脑调控", "脑电引导神经调控", "个性化闭环脑调控", "自适应神经调控策略", "多参数脑刺激优化"], 4, "core", "调控与治疗", "中国信通院2025/科研机构"),

    R("植入手术与辅助设备", "手术机器人", ["脑机接口手术机器人", "神经电极植入机器人", "脑电极植入机器人", "神经探针植入机器人", "缝纫机式植入机器人", "脑机缝合机器人", "自动电极植入", "血管避让电极植入", "微米级神经植入", "亚微米植入控制"], 4, "core", "上游植入服务", "两份附件/中国信通院/工信部"),
    R("植入手术与辅助设备", "导航定位与影像", ["脑机立体定向植入", "神经导航电极植入", "脑区靶点定位", "脑机术中导航", "植入区域实时成像", "脑血管三维重建植入", "功能磁共振引导脑机", "脑机机器人路径规划", "电极植入轨迹规划"], 4, "contextual", "上游植入服务", "五道口报告/工信部/FDA"),
    R("植入手术与辅助设备", "微创递送", ["脑机微创递送", "颅骨微缝电极植入", "微孔电极植入", "注射式神经电极", "可展开神经电极", "血管内支架电极递送", "脑静脉窦电极递送", "柔性微丝植入", "神经电极拔除装置"], 4, "core", "上游植入服务", "中国信通院/资本研报/科研机构"),

    R("脑控外设与人机交互", "输入通信", ["脑控光标", "脑控键盘", "脑控拼写器", "脑机字符输入", "脑控中文输入", "脑控语音合成", "脑机辅助沟通", "脑控计算机", "意念输入系统"], 3, "core", "下游应用", "两份附件/资本研报"),
    R("脑控外设与人机交互", "机器人与假肢", ["脑控机械臂", "脑控机器人", "脑控仿生手", "脑控假肢", "脑机神经义肢", "脑控灵巧手", "脑控外骨骼", "脑机外骨骼", "脑控康复机器人", "意念控制机械手"], 3, "core", "下游应用", "两份附件/FDA/资本研报"),
    R("脑控外设与人机交互", "移动与环境控制", ["脑控轮椅", "脑控无人机", "脑控无人车", "脑控移动机器人", "脑控智能家居", "意念控制家电", "脑控辅助驾驶", "脑控物联网", "脑机环境控制"], 2, "core", "下游应用", "两份附件/科研机构"),
    R("脑控外设与人机交互", "虚拟现实与游戏", ["脑机虚拟现实", "脑机增强现实", "BCI-VR", "BCI-AR", "脑控游戏", "意念游戏控制", "沉浸式脑机交互", "神经反馈虚拟现实", "脑机元宇宙"], 2, "core", "下游应用", "两份附件/中国信通院"),

    R("医疗康复应用", "运动功能恢复", ["脑机运动康复", "脑机卒中康复", "脑机偏瘫康复", "脑机脊髓损伤康复", "脑机瘫痪康复", "脑机手功能康复", "运动想象康复训练", "脑机步态康复", "脑机神经功能重建"], 3, "core", "下游应用", "两份附件/资本研报"),
    R("医疗康复应用", "交流与功能替代", ["脑机失语康复", "脑机渐冻症沟通", "ALS脑机沟通", "闭锁综合征脑机接口", "脑机辅助交流", "脑机神经功能替代", "脑机感觉功能替代", "脑机运动功能替代"], 3, "core", "下游应用", "两份附件/FDA/资本研报"),
    R("医疗康复应用", "神经精神疾病", ["脑机帕金森治疗", "脑机震颤治疗", "脑机癫痫治疗", "脑机抑郁症治疗", "脑机强迫症治疗", "脑机成瘾戒除", "脑机自闭症干预", "脑机阿尔茨海默治疗", "脑机疼痛调控", "脑机意识障碍评估"], 2, "core", "下游应用", "两份附件/中国信通院"),
    R("医疗康复应用", "感觉神经假体", ["脑机视觉假体", "皮层视觉假体", "脑机人工耳蜗", "闭环人工耳蜗", "脑机听觉假体", "脑机触觉假体", "感觉神经假体", "视皮层假体", "人工感觉重建"], 3, "contextual", "下游应用", "两份附件/FDA/资本研报"),

    R("消费工业与研究应用", "状态监测", ["脑机疲劳监测", "脑电驾驶疲劳监测", "脑机注意力监测", "脑机情绪监测", "脑机睡眠监测", "脑机压力监测", "脑机认知负荷评估", "脑机作业状态监测", "脑纹身份认证"], 2, "core", "下游应用", "两份附件/中国信通院"),
    R("消费工业与研究应用", "教育娱乐与增强", ["脑机专注力训练", "脑机认知训练", "脑机记忆增强", "脑机教育", "脑机体育训练", "脑机艺术创作", "脑机娱乐", "脑机消费头环", "脑机睡眠改善"], 2, "core", "下游应用", "两份附件/中国信通院"),
    R("消费工业与研究应用", "工业交通与航天", ["脑机工业安全", "脑机高危作业", "脑机矿井作业监测", "脑机驾驶监测", "脑机智能座舱", "脑机航空航天", "脑机航天员状态", "脑机制造协同", "脑机人因工程"], 2, "core", "下游应用", "两份附件/工信部/资本研报"),
    R("消费工业与研究应用", "科研工具与平台", ["脑机接口科研平台", "BCI2000", "BCILAB", "脑机开源平台", "神经数据分析平台", "脑机算法云平台", "脑机临床转化平台", "脑机测试验证平台", "脑机数据集平台"], 3, "core", "支撑平台", "两份附件/中国信通院"),
    R("消费工业与研究应用", "外围配套与服务", ["脑机接口外壳", "脑机接口支架", "脑机接口连接器", "脑机接口充电装置", "脑机接口数据展示", "脑机接口用户界面", "脑机接口运维", "脑机接口培训", "脑机设备收纳", "脑机设备佩戴辅助"], 1, "core", "外围配套", "产业链上下文补充"),

    R("安全伦理标准与临床", "植入安全与可靠性", ["脑机植入安全", "神经植入体长期安全", "脑机植入感染", "神经电极胶质瘢痕", "神经电极免疫反应", "植入体机械失效", "神经接口信号漂移", "脑植入体取出", "脑机植入可靠性"], 3, "contextual", "安全与服务", "五道口报告/FDA/科研机构"),
    R("安全伦理标准与临床", "电气热与无线安全", ["脑机电荷安全", "神经刺激电荷平衡", "神经刺激组织损伤", "脑植入体温升", "脑机无线暴露", "植入式脑机电磁兼容", "脑机接口抗干扰", "脑机电池安全"], 3, "contextual", "安全与服务", "FDA/工信部"),
    R("安全伦理标准与临床", "数据安全与神经伦理", ["脑机数据安全", "脑数据隐私", "神经数据隐私", "脑机网络安全", "脑信号加密", "脑电身份认证", "神经权利", "神经伦理", "认知自由", "心理隐私", "脑机知情同意"], 3, "core", "安全与服务", "两份附件/中国信通院"),
    R("安全伦理标准与临床", "标准测试与评价", ["脑机接口标准", "脑机接口测试", "脑机接口性能评价", "脑机信息传输率", "BCI信息传输率", "脑机准确率评价", "脑机数据集质量", "脑电数据集评价", "脑机互操作性", "脑机接口协议标准"], 3, "core", "安全与服务", "中国信通院/工信部/政策文件"),
    R("安全伦理标准与临床", "临床试验与医疗器械", ["脑机接口临床试验", "植入式BCI临床", "脑机接口医疗器械", "脑机接口注册检验", "脑机接口非临床测试", "脑机可行性研究", "脑机关键性试验", "脑机全生命周期评价", "神经植入器械临床评价"], 3, "core", "安全与服务", "中国信通院/FDA/资本研报"),
]


# 这些词常与脑机接口共享字面形式，但本身不能证明属于脑机接口产业。
NEGATIVE_RULES = [
    {"terms": ["通用人机接口", "用户界面接口", "计算机网络接口", "机箱接口", "显示接口", "软件接口", "应用程序接口", "API接口"], "penalty": 5},
    {"terms": ["仅用于脑电诊断", "常规脑电图诊断", "睡眠脑电图检查", "癫痫脑电图诊断", "单纯脑电监护", "普通脑电图机"], "penalty": 4},
    {"terms": ["深度神经网络接口", "卷积神经网络接口", "人工神经网络接口", "网络接口控制器"], "penalty": 5},
]

BCI_CONTEXT_TERMS = ["脑机接口", "脑-机接口", "脑计算机接口", "脑机器接口", "脑机交互", "脑机通信", "BCI", "脑控", "意念控制", "神经机器接口", "脑脊接口", "脑脑接口", "闭环神经接口"]
NEURAL_SIGNAL_CONTEXT_TERMS = ["脑电", "脑信号", "神经信号", "神经活动", "皮层电位", "ECoG", "SEEG", "EEG", "LFP", "脑磁", "fNIRS", "神经元放电", "动作电位"]
INTERACTION_CONTEXT_TERMS = ["意图", "脑控", "控制外部", "控制设备", "人机交互", "解码", "编解码", "分类识别", "指令", "反馈", "闭环", "自适应调控", "神经假体", "功能替代", "功能恢复"]
EXPLICIT_TERMS = ["脑机接口", "脑-机接口", "脑计算机接口", "脑机器接口", "脑机交互", "脑控", "意念控制", "双向脑机接口", "闭环脑机接口", "脑脊接口", "脑脑接口"]


def safe_text(v): return "" if pd.isna(v) else str(v)


def normalize_text(text):
    return safe_text(text).lower().replace("－", "-").replace("—", "-").replace("–", "-").replace("（", "(").replace("）", ")")


def term_to_regex(term):
    esc = re.escape(term.strip().lower()).replace(r"\ ", r"[\s\-]*")
    return esc if re.search(r"[\u4e00-\u9fff]", term) else r"(?<![a-z0-9])" + esc.replace(r"\-", r"[\s\-]*") + r"(?![a-z0-9])"


def compile_terms(terms): return [(t, re.compile(term_to_regex(t), re.I)) for t in terms if t.strip()]
def compile_rules(rules): return [dict(x, patterns=compile_terms(x.get("terms", []))) for x in rules]
def _has(text, pats): return any(p.search(text) for _, p in pats)
def _union(pats): return re.compile("|".join(p.pattern for _, p in pats) or r"(?!)", re.I)
def _top(d): return max(d.items(), key=lambda x: (x[1], x[0]))[0] if d else ""


COMPILED_RULES, COMPILED_NEGATIVE = compile_rules(BCI_RULES), compile_rules(NEGATIVE_RULES)
BCI_PATTERNS, SIGNAL_PATTERNS = compile_terms(BCI_CONTEXT_TERMS), compile_terms(NEURAL_SIGNAL_CONTEXT_TERMS)
INTERACTION_PATTERNS, EXPLICIT_PATTERNS = compile_terms(INTERACTION_CONTEXT_TERMS), compile_terms(EXPLICIT_TERMS)
INDEPENDENT_PATTERNS = compile_terms([t for r in BCI_RULES if r["match_type"] == "core" for t in r["terms"]])
CONTEXTUAL_PATTERNS = compile_terms([t for r in BCI_RULES if r["match_type"] != "core" for t in r["terms"]])
INDEPENDENT_REGEX, CONTEXTUAL_REGEX = _union(INDEPENDENT_PATTERNS), _union(CONTEXTUAL_PATTERNS)
BCI_REGEX, SIGNAL_REGEX, INTERACTION_REGEX = _union(BCI_PATTERNS), _union(SIGNAL_PATTERNS), _union(INTERACTION_PATTERNS)


def score_one_patent_text(text: str) -> Dict:
    text = normalize_text(text)
    bci_ctx, signal_ctx = _has(text, BCI_PATTERNS), _has(text, SIGNAL_PATTERNS)
    interaction_ctx, explicit = _has(text, INTERACTION_PATTERNS), _has(text, EXPLICIT_PATTERNS)
    strong = any(any(p.search(text) for _, p in r["patterns"]) for r in COMPILED_RULES if r["match_type"] == "core")
    matched, core_hits, context_hits, inactive = [], [], [], []
    term_scores, cats, subs, segments = {}, defaultdict(int), defaultdict(int), defaultdict(int)
    sources, raw, maximum = set(), 0, 0
    for r in COMPILED_RULES:
        hits = [t for t, p in r["patterns"] if p.search(text)]
        if not hits: continue
        valid = r["match_type"] == "core" or strong or bci_ctx or (signal_ctx and interaction_ctx)
        if not valid:
            inactive.extend(hits); continue
        s = int(r["score"]); raw += s; maximum = max(maximum, s)
        cats[r["category"]] += s; subs[(r["category"], r["sub_category"])] += s; segments[r["industry_segment"]] += s
        sources.add(r["source_type"]); matched.extend(hits)
        (core_hits if r["match_type"] == "core" else context_hits).extend(hits)
        for t in hits: term_scores[t] = max(term_scores.get(t, 0), s)
    negatives = [t for r in COMPILED_NEGATIVE for t, p in r["patterns"] if p.search(text)]
    score = maximum
    if negatives and not explicit: score = 0
    elif negatives: score = max(0, score - 1)
    main_cat = _top(cats); pool = {k: v for k, v in subs.items() if k[0] == main_cat}
    main_sub = max(pool.items(), key=lambda x: (x[1], x[0][1]))[0][1] if pool else ""
    return {"brain_computer_interface_score_raw": raw, "brain_computer_interface_score": score,
            "core_score": score, "max_matched_keyword_score": maximum,
            "matched_terms": "；".join(sorted(set(matched))), "matched_core_terms": "；".join(sorted(set(core_hits))),
            "matched_context_terms": "；".join(sorted(set(context_hits))),
            "matched_term_scores": "；".join(f"{t}:{term_scores[t]}" for t in sorted(term_scores)),
            "inactive_terms_no_context": "；".join(sorted(set(inactive))), "negative_terms": "；".join(sorted(set(negatives))),
            "main_category": main_cat, "main_sub_category": main_sub, "industry_segment": _top(segments),
            "category_scores": json.dumps(dict(cats), ensure_ascii=False, sort_keys=True),
            "subcategory_scores": json.dumps({f"{a}/{b}": v for (a, b), v in subs.items()}, ensure_ascii=False, sort_keys=True),
            "source_types": "；".join(sorted(sources)), "has_bci_context": int(bci_ctx),
            "has_neural_signal_context": int(signal_ctx), "has_interaction_context": int(interaction_ctx)}


def make_text_series(df, cn_abs_col="摘要 (中文)", en_abs_col=None, extra_text_cols=None):
    cols = [c for c in ([cn_abs_col, en_abs_col] + list(extra_text_cols or [])) if c and c in df.columns]
    return (df[cols].fillna("").astype(str).agg(" ".join, axis=1).map(normalize_text)
            if cols else pd.Series([""] * len(df), index=df.index, dtype="object"))


def _join_unique(s):
    vals = [v.strip() for x in s.dropna().astype(str) for v in x.split("；") if v.strip()]
    return "；".join(sorted(set(vals)))


def _mode(s):
    vals = [str(v) for v in s.dropna() if str(v).strip()]
    return Counter(vals).most_common(1)[0][0] if vals else ""


def summarize_brain_computer_interface_firms(patents, firm_col="第一申请人", year_col="year", region_col=None, firm_type_col=None):
    if patents.empty or firm_col not in patents.columns: return pd.DataFrame(), pd.DataFrame()
    d = patents[patents[firm_col].notna()].copy(); d[firm_col] = d[firm_col].astype(str).str.strip(); d = d[d[firm_col] != ""]
    regions = ([region_col] if isinstance(region_col, str) else list(region_col or [])); regions = [c for c in regions if c in d.columns]
    gy, gf = [firm_col] + regions + [year_col], [firm_col] + regions
    common = dict(brain_computer_interface_patent_count=("is_brain_computer_interface_patent", "sum"),
                  brain_computer_interface_score_sum=("brain_computer_interface_score", "sum"),
                  brain_computer_interface_score_mean=("brain_computer_interface_score", "mean"),
                  brain_computer_interface_score_max=("brain_computer_interface_score", "max"),
                  evidence_score_sum=("brain_computer_interface_score_raw", "sum"),
                  evidence_score_mean=("brain_computer_interface_score_raw", "mean"),
                  main_categories=("main_category", _join_unique), main_sub_categories=("main_sub_category", _join_unique),
                  industry_segments=("industry_segment", _join_unique), matched_terms=("matched_terms", _join_unique))
    ya, fa = common.copy(), dict(first_year=(year_col, "min"), last_year=(year_col, "max"), **common)
    if firm_type_col and firm_type_col in d.columns:
        ya["first_applicant_types"] = (firm_type_col, _join_unique); fa["first_applicant_types"] = (firm_type_col, _join_unique)
    fy = d.groupby(gy, dropna=False).agg(**ya).reset_index(); f = d.groupby(gf, dropna=False).agg(**fa).reset_index()
    for src, dst in [("main_category", "firm_main_category"), ("main_sub_category", "firm_main_sub_category"), ("industry_segment", "firm_main_industry_segment")]:
        dom = d.groupby(gf, dropna=False)[src].agg(_mode).reset_index().rename(columns={src: dst}); f = f.merge(dom, on=gf, how="left")
    return fy, f


def tag_brain_computer_interface_patents(df, cn_abs_col="摘要 (中文)", en_abs_col=None, firm_col="第一申请人", year_col="year",
                                         region_col=None, firm_type_col=None, extra_text_cols=None, split_firms=False,
                                         firm_sep_regex=r"[;；,，、|/]+", coarse_screen=True, progress_every=10000, min_score=1):
    data = df.copy(); texts = make_text_series(data, cn_abs_col, en_abs_col, extra_text_cols)
    if coarse_screen:
        mask = texts.str.contains(INDEPENDENT_REGEX, na=False) | texts.str.contains(BCI_REGEX, na=False) | (
            texts.str.contains(SIGNAL_REGEX, na=False) & texts.str.contains(INTERACTION_REGEX, na=False) & texts.str.contains(CONTEXTUAL_REGEX, na=False))
        data, texts = data.loc[mask].copy(), texts.loc[mask]
    start, results = time.time(), []
    for i, text in enumerate(texts, 1):
        results.append(score_one_patent_text(text))
        if progress_every and (i % progress_every == 0 or i == len(data)):
            elapsed = time.time()-start; speed = i/elapsed if elapsed else 0; remain = (len(data)-i)/speed if speed else 0
            print(f"已处理 {i:,}/{len(data):,} 条候选，已用 {elapsed/60:.1f} 分钟，预计剩余 {remain/60:.1f} 分钟")
    scored = pd.DataFrame(results, index=data.index, columns=list(score_one_patent_text("").keys()))
    tagged = pd.concat([data, scored], axis=1)
    tagged["is_brain_computer_interface_patent"] = (tagged["brain_computer_interface_score"] >= min_score).astype(int)
    if split_firms and firm_col in tagged.columns:
        tagged["_firm"] = tagged[firm_col].fillna("").astype(str).str.split(firm_sep_regex); tagged = tagged.explode("_firm")
        tagged[firm_col] = tagged["_firm"].str.strip(); tagged = tagged[tagged[firm_col] != ""].drop(columns="_firm")
    formal = tagged[tagged["is_brain_computer_interface_patent"] == 1].copy()
    fy, f = summarize_brain_computer_interface_firms(formal, firm_col, year_col, region_col, firm_type_col)
    return tagged, formal, fy, f


def export_keyword_dictionary():
    return pd.DataFrame([{"关键词": t, "技术领域": r["category"], "细分方向": r["sub_category"],
                          "产业板块": r["industry_segment"], "核心程度得分": r["score"],
                          "匹配类型": r["match_type"],
                          "上下文要求": "无需上下文" if r["match_type"] == "core" else "需脑机接口上下文，或脑/神经信号与交互解码上下文同时成立",
                          "来源类型": r["source_type"]} for r in BCI_RULES for t in r["terms"]])


if __name__ == "__main__":
    print(f"脑机接口词典包含 {len(export_keyword_dictionary()):,} 个关键词/缩写。")
