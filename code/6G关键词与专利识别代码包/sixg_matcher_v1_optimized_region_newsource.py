# -*- coding: utf-8 -*-
"""6G专利匹配、技术分类与1—5分核心程度评分模块。"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List

import pandas as pd


def R(category, sub_category, terms, score, match_type, segment, source):
    return {"category": category, "sub_category": sub_category, "terms": terms,
            "score": score, "match_type": match_type,
            "industry_segment": segment, "source_type": source}


# 以中文科技文本中的实际写法为主，保留6G、IMT-2030、RIS、ISAC、NTN等常见缩写，
# 不另建与中文词条机械重复的整套英文词典。
SIXG_RULES: List[Dict] = [
    R("基础概念与标准体系", "领域锚点", ["6G", "6G技术", "第六代移动通信", "第六代移动通信技术", "第六代蜂窝通信", "下一代移动通信6G", "IMT-2030", "IMT 2030", "Beyond 5G", "B5G", "Pre-6G"], 1, "core", "系统与平台", "两份附件/ITU/3GPP"),
    R("基础概念与标准体系", "核心系统", ["6G通信系统", "6G移动通信系统", "6G无线通信系统", "6G蜂窝通信系统", "6G网络系统", "6G无线系统", "IMT-2030系统", "第六代移动通信系统", "6G端到端系统", "6G融合通信系统"], 5, "core", "系统与平台", "两份附件/IMT-2030推进组"),
    R("基础概念与标准体系", "总体愿景", ["6G万物智联", "6G数字孪生", "6G人机物智慧互联", "6G虚实共生", "6G泛在互联", "6G智慧内生", "6G性能卓越", "6G绿色节能", "6G安全可信", "6G通信感知计算智能融合", "通感算智融合"], 5, "core", "系统与平台", "两份附件/IMT-2030推进组"),
    R("基础概念与标准体系", "标准与版本", ["6G标准", "6G标准化", "6G技术规范", "6G候选技术", "IMT-2030框架", "IMT-2030候选技术", "ITU-R M.2160", "Release 20 6G", "Rel-20 6G", "Release 21 6G", "Rel-21 6G", "6G RIT", "6G无线接口技术提案"], 4, "core", "标准与试验", "TDIA2025/ITU/3GPP"),

    R("场景与关键能力", "沉浸式通信", ["6G沉浸式通信", "沉浸式通信IC", "沉浸式云XR", "6G云XR", "6G扩展现实", "6G全息通信", "6G多感官通信", "6G感官互联", "临场通信", "多感官XR"], 3, "core", "下游应用", "两份附件/ITU"),
    R("场景与关键能力", "超可靠低时延", ["6G超可靠低时延通信", "6G极其可靠通信", "HRLLC", "6G亚毫秒时延", "6G七个九可靠性", "6G确定性低时延", "6G高可靠通信", "6G超低时延通信"], 4, "core", "中游网络", "两份附件/ITU"),
    R("场景与关键能力", "海量通信", ["6G海量通信", "6G超大规模连接", "6G海量机器通信", "6G千万级连接", "Massive Communication 6G", "6G千亿物联", "6G泛在物联", "6G海量接入"], 3, "core", "中游网络", "两份附件/ITU"),
    R("场景与关键能力", "泛在连接", ["6G泛在连接", "6G全域覆盖", "6G全球无缝覆盖", "6G立体覆盖", "Ubiquitous Connectivity 6G", "6G地理全覆盖", "6G空天地海覆盖", "6G按需覆盖"], 4, "core", "中游网络", "两份附件/ITU"),
    R("场景与关键能力", "通信与AI", ["6G通信与人工智能融合", "6G AI与通信", "AIAC", "AI and Communication 6G", "6G普惠智能", "6G智能服务", "6G移动智能", "6G泛在智能"], 4, "core", "中游网络", "两份附件/ITU/3GPP"),
    R("场景与关键能力", "通信与感知", ["6G通信感知融合", "6G通信感知一体化", "6G通感一体化", "6G通感融合", "6G ISAC", "IMT-2030通感", "6G感知服务"], 5, "core", "中游空口", "两份附件/ITU/IMT-2030推进组"),
    R("场景与关键能力", "性能指标", ["Tbps峰值速率", "太比特每秒通信", "Gbps体验速率", "亚毫秒级时延", "千万级连接", "七个九可靠性", "厘米级感知精度", "毫米级感知精度", "每立方米连接密度", "超高流量密度"], 3, "contextual", "中游系统", "两份附件/ITU"),

    R("频谱与传播", "太赫兹通信", ["6G太赫兹通信", "6G太赫兹系统", "6G THz通信", "IMT-2030太赫兹", "6G亚太赫兹通信", "6G次太赫兹通信"], 5, "core", "中游空口", "两份附件/ITU"),
    R("频谱与传播", "太赫兹通用词", ["太赫兹通信", "太赫兹无线传输", "THz通信", "亚太赫兹通信", "次太赫兹通信", "Sub-THz通信", "毫米波太赫兹融合", "0.1至10THz", "100GHz以上通信", "D波段通信", "G波段通信"], 4, "contextual", "中游空口", "两份附件/ITU/科研论文"),
    R("频谱与传播", "中高频融合", ["6G中高频融合", "6G多频段融合", "6G全频谱融合", "低中高全频谱", "厘米波毫米波太赫兹融合", "Sub-6GHz与太赫兹协同", "中频段与高频段协同", "多频段协同传输", "频谱动态聚合", "跨频段联合传输"], 4, "contextual", "中游空口", "两份附件/IMT-2030推进组"),
    R("频谱与传播", "可见光与光无线", ["6G可见光通信", "6G光无线通信", "可见光通信", "VLC通信", "LiFi", "光无线通信", "自由空间光通信", "FSO通信", "红外无线通信", "紫外无线通信", "水下可见光通信", "光射频融合通信"], 4, "contextual", "中游空口", "工信部推进组报告/科研论文"),
    R("频谱与传播", "信道测量与建模", ["6G信道模型", "6G信道测量", "太赫兹信道测量", "太赫兹信道建模", "高频信道探测", "空天地信道模型", "星地信道模型", "近场信道建模", "非平稳信道建模", "稀疏多径信道", "分子吸收损耗", "雨衰建模", "遮挡传播模型", "三维空间信道模型"], 4, "contextual", "中游算法", "两份附件/ITU"),
    R("频谱与传播", "近场与新传播特性", ["近场通信感知", "电磁近场传输", "球面波信道", "近场波束聚焦", "近场波束赋形", "空间非平稳性", "波束偏斜", "波束分裂效应", "远近场联合传输", "瑞利距离扩展", "超大孔径近场"], 4, "contextual", "中游空口", "科研机构/公开论文"),

    R("增强型无线空口", "统一调制编码", ["6G统一编译码架构", "6G调制编码", "统一信道编码", "统一编译码", "自适应调制编码6G", "多场景编码", "短码长高可靠编码", "高吞吐量译码", "低复杂度译码"], 4, "contextual", "中游空口", "工信部推进组报告"),
    R("增强型无线空口", "信道编码", ["6G极化码", "Polar码6G", "准循环LDPC", "QC-LDPC", "6G LDPC", "喷泉码无线", "稀疏图码", "神经信道编码", "语义信道编码", "短包编码", "无率码6G"], 4, "contextual", "中游空口", "工信部推进组报告/科研论文"),
    R("增强型无线空口", "新波形", ["6G新型波形", "变换域波形", "OTFS", "正交时频空间调制", "AFDM", "仿射频分复用", "FTN传输", "超奈奎斯特传输", "SEFDM", "SEFFM", "OVXDM", "滤波器组多载波", "FBMC", "通感一体波形"], 4, "contextual", "中游空口", "工信部推进组报告/科研论文"),
    R("增强型无线空口", "新型多址", ["6G新型多址", "非正交多址", "NOMA", "速率分拆多址", "RSMA", "稀疏码多址", "SCMA", "图样分割多址", "PDMA", "多用户共享接入", "MUSA", "免授权接入", "Grant-free接入", "无蜂窝随机接入", "AI多址接入"], 4, "contextual", "中游空口", "工信部推进组报告/科研论文"),
    R("增强型无线空口", "超大规模MIMO", ["6G超大规模MIMO", "超大规模MIMO", "XL-MIMO", "极大规模MIMO", "超大规模天线阵列", "超大规模口径阵列", "ELAA", "连续孔径阵列", "超密集天线阵列", "分布式超大规模MIMO", "超大孔径阵列", "近场MIMO"], 4, "contextual", "上游设备与中游空口", "工信部推进组报告/ITU"),
    R("增强型无线空口", "无蜂窝与分布式MIMO", ["6G无蜂窝网络", "无蜂窝大规模MIMO", "Cell-free Massive MIMO", "用户中心无蜂窝网络", "分布式MIMO", "协作多点传输", "分布式接入点", "用户中心虚拟小区", "去小区化接入", "无定形网络"], 4, "contextual", "中游接入网", "工信部推进组报告/科研论文"),
    R("增强型无线空口", "波束管理", ["6G波束管理", "太赫兹波束赋形", "太赫兹波束跟踪", "高频波束对准", "多波束协同", "智能波束扫描", "近场波束训练", "波束预测", "波束失败恢复", "混合波束赋形", "全息波束赋形"], 4, "contextual", "中游空口", "两份附件/科研论文"),
    R("增强型无线空口", "带内全双工", ["6G带内全双工", "带内全双工", "In-band Full Duplex", "IBFD", "同频同时收发", "全双工基站", "全双工中继", "射频自干扰抵消", "数字域自干扰消除", "模拟域自干扰抑制", "大规模天线全双工"], 4, "contextual", "中游空口", "工信部推进组报告"),
    R("增强型无线空口", "智能反射与协作传输", ["协作传输6G", "网络协作波束赋形", "分布式协作传输", "联合发送接收", "联合传输接收点", "多连接协同", "多点联合调度", "端到端链路协同"], 3, "contextual", "中游空口", "科研机构/公开论文"),
    R("增强型无线空口", "通用空口技术词", ["波束赋形", "波束跟踪", "波束对准", "波束扫描", "大规模MIMO", "MIMO", "信道估计", "干扰抑制", "功率控制", "资源调度", "链路自适应", "混合预编码", "多用户检测"], 4, "contextual", "中游空口", "两份附件/上下文补充"),

    R("新物理维度传输", "智能超表面", ["6G智能超表面", "6G可重构智能表面", "6G RIS", "6G IRS", "IMT-2030 RIS"], 5, "core", "上游器件与中游空口", "两份附件/IMT-2030推进组"),
    R("新物理维度传输", "智能超表面通用词", ["智能超表面", "可重构智能表面", "可重构智能反射表面", "RIS", "IRS", "可编程超表面", "编码超表面", "数字编码超表面", "STAR-RIS", "透射反射智能表面", "有源智能超表面", "无源智能超表面", "全息超表面"], 4, "contextual", "上游器件与中游空口", "工信部推进组报告/科研机构"),
    R("新物理维度传输", "RIS控制与信道估计", ["RIS相位优化", "RIS波束赋形", "RIS信道估计", "级联信道估计", "RIS单元控制", "RIS辅助定位", "RIS辅助感知", "RIS辅助太赫兹", "RIS辅助MIMO", "RIS反射系数优化", "RIS部署优化", "超表面电磁调控"], 4, "contextual", "中游算法", "IMT-2030推进组/科研论文"),
    R("新物理维度传输", "轨道角动量", ["6G轨道角动量", "轨道角动量通信", "OAM通信", "涡旋电磁波通信", "OAM模态复用", "OAM波束", "涡旋波束", "螺旋相位波束", "OAM模态分选", "OAM天线", "OAM量子态", "太赫兹OAM"], 4, "contextual", "中游空口", "工信部推进组报告/科研论文"),
    R("新物理维度传输", "智能全息无线电", ["6G智能全息无线电", "智能全息无线电", "IHR", "全息无线电", "射频全息", "射频空间谱全息", "全息空间波场合成", "全息MIMO", "Holographic MIMO", "连续孔径全息天线", "全息干涉通信"], 4, "contextual", "上游设备与中游空口", "工信部推进组报告/科研机构"),
    R("新物理维度传输", "微波光子融合", ["6G微波光子", "微波光子天线阵列", "相干光上变频", "光生太赫兹", "光电融合太赫兹", "光子辅助毫米波", "微波光子信号处理", "射频光子集成", "光域波束赋形", "光域信号处理"], 4, "contextual", "上游器件", "工信部推进组报告/科研机构"),

    R("通信感知一体化", "通感系统与架构", ["通信感知一体化", "通感一体化", "通信感知融合", "通感融合", "ISAC", "JCAS", "雷达通信一体化", "通信雷达一体化", "通信定位感知一体化", "通感算一体化", "通感算智一体化"], 4, "contextual", "中游空口", "两份附件/ITU/3GPP"),
    R("通信感知一体化", "通感波形与资源", ["通感一体化波形", "通信感知联合波形", "通感波束赋形", "通感资源分配", "通感频谱共享", "通信雷达共存", "通感联合预编码", "通感功率分配", "通感干扰管理", "通感参考信号", "通感帧结构"], 5, "contextual", "中游空口", "工信部推进组报告/IMT-2030推进组"),
    R("通信感知一体化", "无线感知", ["无线通信感知", "无线环境感知", "通信信号感知", "基站感知", "蜂窝网络感知", "无源无线感知", "人体动作无线感知", "呼吸心跳无线感知", "目标检测通信信号", "无线成像", "射频成像", "微多普勒感知"], 4, "contextual", "中游算法", "两份附件/科研论文"),
    R("通信感知一体化", "高精度定位", ["6G高精度定位", "通感一体定位", "厘米级无线定位", "毫米级无线定位", "三维定位6G", "高频段定位", "太赫兹定位", "RIS高精度定位", "定位通信融合", "定位感知融合", "姿态感知定位"], 4, "contextual", "中游算法", "两份附件/ITU"),
    R("通信感知一体化", "感知辅助通信", ["感知辅助通信", "环境感知辅助波束", "感知辅助接入", "感知辅助资源调度", "感知辅助信道估计", "感知信息反馈通信", "环境地图辅助通信", "无线环境地图", "信道知识地图", "CKM"], 4, "contextual", "中游算法", "工信部推进组报告/科研机构"),

    R("AI原生与语义通信", "AI原生网络", ["6G AI原生网络", "6G内生智能网络", "6G智慧内生网络", "AI原生空口", "内生智能空口", "AI原生架构", "网络内生智能", "无线内生智能", "AI-native 6G", "原生智能无线网络"], 5, "core", "中游网络", "两份附件/IMT-2030推进组/3GPP"),
    R("AI原生与语义通信", "AI赋能空口", ["AI赋能空口", "无线AI", "AI-RAN", "神经接收机", "神经网络接收机", "学习型收发机", "端到端学习通信", "AI信道估计", "AI波束管理", "AI资源调度", "AI调制识别", "深度学习物理层", "强化学习无线资源管理"], 4, "contextual", "中游算法", "两份附件/AI-RAN/科研论文"),
    R("AI原生与语义通信", "分布式学习", ["6G联邦学习", "无线联邦学习", "空口联邦学习", "分层联邦学习", "去中心化联邦学习", "分割学习无线", "边缘协同学习", "群智协同学习", "多智能体无线", "分布式机器学习通信", "隐私保护联邦学习"], 4, "contextual", "中游算法", "工信部推进组报告/科研机构"),
    R("AI原生与语义通信", "语义通信", ["6G语义通信", "语义通信", "Semantic Communication", "语义编码", "语义解码", "语义传输", "语义信息论", "语义噪声", "语义相似度", "多模态语义通信", "生成式语义通信", "知识图谱语义通信"], 5, "contextual", "中游算法", "工信部推进组报告/科研机构"),
    R("AI原生与语义通信", "任务与目标导向通信", ["任务导向通信", "面向任务通信", "目标导向通信", "意图驱动通信", "任务语义通信", "任务级传输", "价值语义通信", "通信计算联合推理", "边缘推理传输", "面向机器通信", "机器语义通信"], 5, "contextual", "中游算法", "科研机构/公开论文"),
    R("AI原生与语义通信", "空中计算", ["空中计算", "空口计算", "Over-the-Air Computation", "AirComp", "无线聚合计算", "模拟空中聚合", "联邦学习空中聚合", "计算通信融合", "函数计算无线传输", "多址信道计算"], 4, "contextual", "中游算法", "科研机构/公开论文"),
    R("AI原生与语义通信", "网络自智", ["6G网络自智", "网络自运维", "网络自检测", "网络自修复", "网络自优化", "网络自演进", "意图驱动网络", "零接触运维", "闭环网络自动化", "自治网络等级", "认知网络管理"], 4, "contextual", "中游网络", "工信部推进组报告/科研机构"),
    R("AI原生与语义通信", "网络服务AI", ["Network for AI", "6G智算服务", "网络使能AI", "泛在AI服务", "分布式AI推理网络", "端边云AI协同", "AI模型无线传输", "大模型通信网络", "具身智能6G连接", "智能体通信网络"], 4, "contextual", "中游网络", "TDIA2025/科研机构"),
    R("AI原生与语义通信", "通用AI算法词", ["深度学习", "机器学习", "强化学习", "联邦学习", "迁移学习", "自监督学习", "卷积神经网络", "CNN", "循环神经网络", "RNN", "长短期记忆网络", "LSTM", "Transformer", "图神经网络", "多智能体强化学习"], 4, "contextual", "中游算法", "两份附件/上下文补充"),

    R("新型网络架构", "分布式自治网络", ["6G分布式自治网络", "分布式自治网络", "6G去中心化网络", "去中心化核心网", "用户中心网络架构", "场景定制化网络", "需求驱动网络架构", "自生长网络", "自演进网络架构", "轻量化接入网", "柔性接入网架构"], 4, "contextual", "中游网络", "工信部推进组报告/IMT-2030推进组"),
    R("新型网络架构", "云原生与服务化", ["6G云原生网络", "6G服务化架构", "云原生核心网6G", "服务化RAN", "服务化空口", "微服务核心网", "无状态网络功能", "容器化网元", "网络功能服务化", "按需编排网络", "网络能力开放6G"], 4, "contextual", "中游网络", "TDIA2025/IMT-2030推进组"),
    R("新型网络架构", "开放可编程网络", ["6G开放网络", "6G开放无线接入网", "Open RAN 6G", "O-RAN 6G", "可编程无线网络", "软件定义空口", "软件定义无线网络", "SDN 6G", "NFV 6G", "网络可编程数据面", "开放接口6G"], 4, "contextual", "中游网络", "TDIA2025/3GPP/资本研报"),
    R("新型网络架构", "网络切片与专网", ["6G网络切片", "6G端到端切片", "智能网络切片", "切片按需编排", "跨域网络切片", "卫星地面融合切片", "算网切片", "通感切片", "场景定制切片", "6G行业专网"], 3, "contextual", "中游网络", "两份附件/科研机构"),
    R("新型网络架构", "数字孪生网络", ["6G数字孪生网络", "数字孪生网络", "DTN", "网络数字孪生", "无线网络数字孪生", "物理网络孪生体", "孪生网络闭环优化", "网络实时映射", "数字孪生网络建模", "数字孪生网络仿真", "网络预测控制"], 4, "contextual", "中游网络", "工信部推进组报告/科研机构"),
    R("新型网络架构", "确定性网络", ["6G确定性网络", "确定性网络", "确定性无线网络", "有界时延网络", "有界抖动", "有界丢包", "端到端确定性", "确定性资源预留", "确定性服务保护", "跨域确定性", "时间敏感网络6G", "TSN 6G"], 4, "contextual", "中游网络", "工信部推进组报告"),
    R("新型网络架构", "算力感知与算网融合", ["6G算力网络", "算力感知网络", "移动算力网络", "算网融合6G", "算网一体", "通信计算融合", "通信计算存储一体", "算力路由", "算力资源度量", "算力资源编排", "算力服务调度", "端边云网协同", "云边端网融合"], 4, "contextual", "中游网络", "工信部推进组报告/IMT-2030推进组"),
    R("新型网络架构", "数据服务与数据面", ["6G数据服务", "6G数据面", "网络数据服务", "数据驱动网络架构", "数据编排网络", "数据织网6G", "数据随路计算", "网络数据治理", "数据闭环网络", "数据服务化架构", "通信数据空间"], 4, "contextual", "中游网络", "TDIA2025/IMT-2030推进组"),
    R("新型网络架构", "近域与体域网络", ["6G近域网络", "近域通信网络", "设备到设备6G", "D2D 6G", "侧行链路6G", "体域网6G", "纳米通信网络", "片间无线通信", "芯粒无线互联", "片上无线网络", "个人域智能网络"], 3, "contextual", "中游网络", "TDIA2025/科研机构"),
    R("新型网络架构", "通用网络技术词", ["云计算", "边缘计算", "移动边缘计算", "网络切片", "软件定义网络", "网络功能虚拟化", "云原生网络", "微服务架构", "数字孪生", "算力调度", "服务化架构", "可编程网络", "自治网络", "确定性通信"], 3, "contextual", "中游网络", "两份附件/上下文补充"),

    R("空天地海一体化", "融合网络架构", ["6G空天地一体化", "6G空天地海一体化", "6G星地融合", "6G天地一体", "6G非地面网络", "6G NTN", "空天地一体化网络6G", "星地一体融合组网", "SAGIN 6G", "空天地海融合网络"], 5, "core", "中游网络与基础设施", "两份附件/ITU/资本研报"),
    R("空天地海一体化", "非地面网络", ["非地面网络", "NTN", "卫星地面融合网络", "星地融合网络", "天地一体化网络", "空天地一体化网络", "低轨卫星通信网络", "中低轨卫星组网", "高中低轨协同", "卫星蜂窝融合", "卫星移动通信融合"], 4, "contextual", "中游网络与基础设施", "两份附件/3GPP"),
    R("空天地海一体化", "手机直连卫星", ["6G手机直连卫星", "手机直连卫星", "终端直连卫星", "卫星直连终端", "Direct-to-Device", "D2D卫星通信", "卫星直连普通手机", "星地融合终端", "卫星物联网终端", "卫星宽带终端"], 3, "contextual", "下游终端", "TDIA2025/资本研报"),
    R("空天地海一体化", "星间与星地链路", ["星间激光通信", "星间高速光链路", "星地激光通信", "卫星太赫兹链路", "星地毫米波链路", "星间路由", "星间链路切换", "星地链路协同", "星间星地联合传输", "多星协同传输"], 4, "contextual", "上游设备与中游网络", "工信部推进组报告/科研机构"),
    R("空天地海一体化", "星载网络与载荷", ["6G星载核心网", "星载基站", "星载移动通信载荷", "再生式卫星载荷", "透明转发卫星载荷", "星上在轨计算", "星载边缘计算", "星上网络功能", "星载路由器", "星载波束形成", "卫星数字透明处理器"], 4, "contextual", "上游设备", "两份附件/资本研报"),
    R("空天地海一体化", "卫星接入与移动性", ["卫星随机接入", "星地融合空口", "卫星波束跳变", "卫星波束切换", "卫星多普勒补偿", "卫星时延补偿", "卫星频偏估计", "卫星移动性管理", "卫星寻址路由", "卫星星历辅助通信", "多卫星切换"], 4, "contextual", "中游空口与网络", "工信部推进组报告/3GPP"),
    R("空天地海一体化", "空基与海洋网络", ["高空平台通信", "HAPS通信", "临近空间通信", "无人机基站", "空基通信网络", "海洋通信网络6G", "海空天通信", "海上宽带通信6G", "机载移动通信", "低空通信感知网络", "低空智联网"], 3, "contextual", "基础设施与应用", "两份附件/资本研报"),

    R("芯片器件与材料", "太赫兹收发器件", ["太赫兹收发芯片", "太赫兹收发机", "太赫兹发射机", "太赫兹接收机", "太赫兹混频器", "太赫兹倍频器", "太赫兹放大器", "太赫兹振荡器", "太赫兹探测器", "太赫兹调制器", "太赫兹变频电路", "太赫兹功率放大器"], 4, "contextual", "上游核心器件", "工信部推进组报告/ITU/资本研报"),
    R("芯片器件与材料", "太赫兹芯片材料", ["磷化铟太赫兹", "InP太赫兹", "锗硅太赫兹", "SiGe太赫兹", "氮化镓太赫兹", "GaN太赫兹", "CMOS太赫兹芯片", "石墨烯太赫兹器件", "二维材料太赫兹", "光子太赫兹芯片", "硅光太赫兹", "太赫兹集成电路"], 4, "contextual", "上游材料与器件", "工信部推进组报告/科研机构"),
    R("芯片器件与材料", "高频射频前端", ["6G射频前端", "亚太赫兹射频前端", "D波段射频前端", "毫米波太赫兹前端", "高频功率放大器", "高频低噪声放大器", "宽带混频器", "高速数模转换器", "高速模数转换器", "射频收发芯片6G", "高频本振芯片", "宽带频率合成器"], 4, "contextual", "上游核心器件", "两份附件/资本研报"),
    R("芯片器件与材料", "基带与AI芯片", ["6G基带芯片", "6G通信芯片", "6G调制解调芯片", "6G原型芯片", "无线AI芯片", "神经网络基带芯片", "片上AI接收机", "低功耗6G基带", "高速基带信号处理芯片", "通感一体芯片", "通信计算融合芯片", "纳米光子芯片"], 4, "contextual", "上游核心器件", "两份附件/工信部推进组报告/资本研报"),
    R("芯片器件与材料", "天线与阵列", ["太赫兹天线", "太赫兹天线阵列", "超大规模阵列天线", "连续孔径天线", "全息天线", "透镜天线太赫兹", "片上天线太赫兹", "封装天线", "AiP天线", "相控阵天线6G", "超宽带高频天线", "可重构天线6G"], 4, "contextual", "上游核心器件", "工信部推进组报告/资本研报"),
    R("芯片器件与材料", "智能超表面器件", ["RIS单元", "超表面单元", "可调超材料单元", "PIN二极管超表面", "变容二极管超表面", "MEMS超表面", "液晶超表面", "石墨烯超表面", "超表面控制器", "RIS控制芯片", "超表面阵列板", "可编程电磁表面"], 4, "contextual", "上游核心器件", "IMT-2030推进组/科研机构"),
    R("芯片器件与材料", "光通信器件", ["可见光通信LED", "高速Micro-LED通信", "激光二极管可见光通信", "可见光光电探测器", "自由空间光收发机", "硅光收发芯片", "光子集成收发器", "微波光子芯片", "相干光无线收发", "光电融合前端"], 4, "contextual", "上游核心器件", "工信部推进组报告/科研机构"),
    R("芯片器件与材料", "封装测试与互连", ["6G芯片先进封装", "太赫兹芯片封装", "高频低损耗封装", "天线封装一体化", "晶圆级射频封装", "异质集成射频芯片", "射频芯粒", "Chiplet射频", "高频互连", "太赫兹探针测试", "毫米波太赫兹测试接口"], 3, "contextual", "上游制造", "资本研报/科研机构"),
    R("芯片器件与材料", "高频材料", ["6G高频材料", "低损耗高频覆铜板", "毫米波太赫兹基板", "液晶聚合物天线材料", "LCP高频材料", "PTFE高频基板", "低介电常数材料", "低介质损耗材料", "超材料电磁表面", "高频封装材料", "射频陶瓷材料"], 3, "contextual", "上游材料", "资本研报/科研机构"),
    R("芯片器件与材料", "通用通信硬件词", ["通信芯片", "基带芯片", "射频前端", "功率放大器", "低噪声放大器", "模数转换器", "数模转换器", "相控阵天线", "天线阵列", "通信模组", "收发机", "基站设备", "核心网设备", "光模块"], 3, "contextual", "上游核心器件", "两份附件/资本研报/上下文补充"),

    R("绿色通信与能量", "绿色网络", ["6G绿色通信", "6G绿色网络", "6G节能网络", "6G低碳网络", "网络能效优化6G", "碳感知网络", "基站智能节能6G", "端到端能耗优化", "零比特零瓦特", "按需唤醒网络", "通信算力协同节能"], 3, "contextual", "中游网络", "两份附件/ITU"),
    R("绿色通信与能量", "无源物联网", ["6G无源物联网", "Ambient IoT 6G", "环境物联网6G", "无源蜂窝物联网", "反向散射通信6G", "环境反向散射", "零功耗物联网终端", "无电池通信终端", "超低功耗标签通信", "蜂窝无源标签"], 4, "contextual", "上游终端与下游应用", "TDIA2025/科研机构"),
    R("绿色通信与能量", "能量采集与无线供能", ["6G无线能量传输", "无线信息能量同传", "SWIPT", "无线供能通信", "射频能量采集", "太赫兹无线供能", "RIS辅助无线供能", "能量波束赋形", "通信感知供能一体", "信息能量联合传输"], 4, "contextual", "中游空口", "工信部推进组报告/科研论文"),

    R("安全可信与隐私", "内生安全", ["6G内生安全", "6G安全内生", "网络内生安全6G", "6G可信内生", "6G安全架构", "6G韧性安全", "安全与网络一体化设计", "自适应网络防御6G", "6G全生命周期安全"], 5, "core", "安全与服务", "工信部推进组报告/IMT-2030推进组"),
    R("安全可信与隐私", "多模信任", ["6G多模信任", "多模信任架构", "中心化去中心化混合信任", "跨域信任6G", "零信任6G", "动态信任评估", "异构网络信任管理", "第三方信任网络", "可信身份6G"], 4, "contextual", "安全与服务", "工信部推进组报告"),
    R("安全可信与隐私", "隐私计算与数据安全", ["6G隐私计算", "无线隐私计算", "6G数据隐私", "联邦学习隐私保护", "差分隐私无线网络", "安全多方计算6G", "同态加密通信网络", "可信执行环境6G", "数据流转监测6G", "网络数据脱敏"], 4, "contextual", "安全与服务", "工信部推进组报告/科研机构"),
    R("安全可信与隐私", "物理层安全", ["6G物理层安全", "无线物理层安全", "安全波束赋形", "人工噪声保密通信", "秘密速率优化", "信道密钥生成", "射频指纹认证", "太赫兹保密通信", "RIS辅助物理层安全", "抗窃听无线传输"], 4, "contextual", "安全与服务", "科研机构/公开论文"),
    R("安全可信与隐私", "密码与量子安全", ["6G后量子密码", "后量子密码通信网络", "6G量子安全密码", "量子密钥分发6G", "QKD星地融合", "逼近香农一次一密", "6G密钥安全分发", "高通量通信加密", "区块链6G网络", "6G分布式账本"], 3, "contextual", "安全与服务", "工信部推进组报告/科研机构"),

    R("测试验证与仪器", "信道与射频测试", ["6G信道探测仪", "太赫兹信道探测器", "太赫兹信号源", "太赫兹频谱分析仪", "太赫兹矢量网络分析", "高频空口测试", "6G射频一致性测试", "D波段测试系统", "太赫兹OTA测试", "近场天线测试6G"], 4, "contextual", "上游仪器仪表", "两份附件/资本研报"),
    R("测试验证与仪器", "原型与试验平台", ["6G原型系统", "6G试验网", "6G测试床", "6G仿真平台", "6G公共试验平台", "6G研发试验装置", "6G技术验证平台", "6G系统级验证", "6G外场试验", "6G先导试验", "智启6G平台"], 4, "core", "标准与试验", "TDIA2025/IMT-2030推进组"),
    R("测试验证与仪器", "数字仿真与评估", ["6G系统仿真", "6G链路级仿真", "6G网络级仿真", "6G硬件在环", "6G数字孪生测试", "6G候选技术评估", "IMT-2030性能评估", "6G互操作测试", "6G端到端测试", "6G多场景测试"], 3, "contextual", "标准与试验", "TDIA2025/ITU/3GPP"),

    R("终端设备与基础设施", "基站与核心网设备", ["6G基站", "6G微基站", "6G分布式基站", "6G核心网", "6G无线接入网", "6G承载网", "6G回传设备", "6G前传设备", "6G室内分布系统", "6G边缘节点", "6G云化基站"], 4, "core", "上游设备与基础设施", "TDIA2025/资本研报"),
    R("终端设备与基础设施", "终端与模组", ["6G终端", "6G手机", "6G通信模组", "6G射频模组", "6G卫星终端", "6G物联网模组", "6G可穿戴终端", "6G工业终端", "6G车载终端", "6G XR终端", "6G通感终端"], 3, "core", "上游终端", "TDIA2025/资本研报"),
    R("终端设备与基础设施", "光纤承载与数据中心", ["6G光承载网", "6G前传光模块", "6G回传光模块", "6G相干光通信", "6G超高速光互联", "6G数据中心互联", "6G全光网络", "6G光纤无线融合", "6G太赫兹光纤融合", "6G算力中心连接"], 3, "contextual", "上游设备与基础设施", "资本研报/ITU-T"),

    R("应用与业务", "全息与多感官交互", ["6G全息视频", "6G全息会议", "6G全息远程呈现", "6G多感官交互", "6G触觉互联网", "6G触觉通信", "6G嗅觉通信", "6G味觉通信", "6G情感交互", "6G脑机交互", "6G智慧交互"], 2, "core", "下游应用", "工信部推进组报告/ITU"),
    R("应用与业务", "数字孪生与工业", ["6G工业数字孪生", "6G智慧工厂", "6G工业互联网", "6G无线工业总线", "6G柔性制造", "6G远程控制", "6G机器人协作", "6G人机物协同生产", "6G智能电网", "6G矿山通信"], 2, "core", "下游应用", "两份附件/资本研报"),
    R("应用与业务", "交通与低空", ["6G车联网", "6G自动驾驶", "6G车路云一体化", "6G智能交通", "6G铁路通信", "6G低空经济", "6G无人机通信", "6G无人机监管", "6G低空智联网", "6G航空通信"], 2, "core", "下游应用", "两份附件/资本研报"),
    R("应用与业务", "医疗教育与公共服务", ["6G远程医疗", "6G远程手术", "6G智慧医疗", "6G沉浸式教育", "6G技能培训", "6G应急通信", "6G灾害救援", "6G环境监测", "6G智慧城市", "6G普惠服务"], 2, "core", "下游应用", "工信部推进组报告/ITU"),
    R("应用与业务", "泛在物联与智能体", ["6G物联网", "6G智能物联网", "6G机器通信", "6G机器人网络", "6G具身智能", "6G智能体互联", "6G群体智能", "6G数字人", "6G环境物联网", "6G海量传感器网络"], 2, "core", "下游应用", "两份附件/TDIA2025"),
    R("应用与业务", "外围配套与服务", ["6G设备外壳", "6G设备支架", "6G设备连接器", "6G设备电源适配器", "6G数据展示界面", "6G网络培训", "6G设备运维工具", "6G设备包装", "6G终端保护套", "6G项目管理平台"], 1, "core", "外围配套", "产业链上下文补充"),
]


NEGATIVE_RULES = [
    R("排除", "重量剂量", ["重量为6g", "重6g", "6g重量", "6g剂量", "加入6g", "称取6g"], 0, "negative", "排除", "消歧"),
    R("排除", "非6G代际", ["仅支持5G", "仅适用于5G", "4G/5G网络", "5G专网", "5G-A专网", "NR专网", "LTE网络优化"], 0, "negative", "排除", "消歧"),
    R("排除", "其他接口", ["Wi-Fi 6GHz", "6GHz Wi-Fi", "6GHz无线局域网", "6GHz频段WLAN"], 0, "negative", "排除", "消歧"),
]

SIXG_CONTEXT_TERMS = ["6G", "第六代移动通信", "第六代蜂窝通信", "IMT-2030", "IMT 2030", "Beyond 5G", "B5G", "Pre-6G"]
SIGNATURE_GROUPS = [
    ["太赫兹通信", "亚太赫兹通信", "次太赫兹通信", "Sub-THz通信", "100GHz以上通信"],
    ["智能超表面", "可重构智能表面", "RIS", "IRS", "STAR-RIS"],
    ["轨道角动量通信", "OAM通信", "智能全息无线电", "全息MIMO"],
    ["通感一体化", "通信感知一体化", "ISAC", "JCAS"],
    ["AI原生空口", "内生智能空口", "语义通信", "任务导向通信", "空中计算", "AirComp"],
    ["无蜂窝大规模MIMO", "Cell-free Massive MIMO", "XL-MIMO", "超大规模MIMO"],
    ["数字孪生网络", "算力感知网络", "移动算力网络", "分布式自治网络"],
    ["空天地一体化网络", "星地一体融合组网", "SAGIN", "星载核心网"],
]


def safe_text(value):
    return "" if pd.isna(value) else str(value)


def normalize_text(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", safe_text(text))).strip()


def term_to_regex(term):
    escaped = re.escape(term.strip()).replace(r"\ ", r"\s*").replace(r"\-", r"[-－—]?" )
    if re.search(r"[A-Za-z0-9]", term):
        return rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])"
    return escaped


def compile_terms(terms):
    # 裸写“6G”要求大写，避免把中文专利中的“6 g/6克”重量表达当作通信代际；
    # “6g通信系统”等带有明确通信语义的完整词仍可由其他规则识别。
    return [(term, re.compile(term_to_regex(term), 0 if term == "6G" else re.I))
            for term in terms if term.strip()]


def compile_rules(rules):
    return [dict(rule, patterns=compile_terms(rule.get("terms", []))) for rule in rules]


def _has(text, patterns):
    return any(pattern.search(text) for _, pattern in patterns)


def _union(patterns):
    return re.compile("|".join(pattern.pattern for _, pattern in patterns) or r"(?!)", re.I)


def _top(scores):
    return max(scores.items(), key=lambda item: (item[1], item[0]))[0] if scores else ""


COMPILED_RULES = compile_rules(SIXG_RULES)
COMPILED_NEGATIVE = compile_rules(NEGATIVE_RULES)
SIXG_PATTERNS = compile_terms(SIXG_CONTEXT_TERMS)
SIGNATURE_PATTERNS = [compile_terms(group) for group in SIGNATURE_GROUPS]
INDEPENDENT_PATTERNS = compile_terms([term for rule in SIXG_RULES if rule["match_type"] == "core" for term in rule["terms"]])
CONTEXTUAL_PATTERNS = compile_terms([term for rule in SIXG_RULES if rule["match_type"] != "core" for term in rule["terms"]])
INDEPENDENT_REGEX = _union(INDEPENDENT_PATTERNS)
CONTEXTUAL_REGEX = _union(CONTEXTUAL_PATTERNS)
SIXG_REGEX = _union(SIXG_PATTERNS)
SIGNATURE_REGEXES = [_union(group) for group in SIGNATURE_PATTERNS]


def score_one_patent_text(text: str) -> Dict:
    text = normalize_text(text)
    sixg_context = _has(text, SIXG_PATTERNS)
    signature_groups = [i for i, patterns in enumerate(SIGNATURE_PATTERNS) if _has(text, patterns)]
    # 无6G字样时采取保守推断：至少命中三个互相独立的6G候选技术簇。
    inferred_context = len(signature_groups) >= 3
    valid_context = sixg_context or inferred_context

    matched, core_hits, context_hits, inactive = [], [], [], []
    term_scores, cats, subs, segments = {}, defaultdict(int), defaultdict(int), defaultdict(int)
    sources, raw, maximum = set(), 0, 0
    for rule in COMPILED_RULES:
        hits = [term for term, pattern in rule["patterns"] if pattern.search(text)]
        if not hits:
            continue
        valid = rule["match_type"] == "core" or valid_context
        if not valid:
            inactive.extend(hits)
            continue
        score = int(rule["score"])
        raw += score
        maximum = max(maximum, score)
        cats[rule["category"]] += score
        subs[(rule["category"], rule["sub_category"])] += score
        segments[rule["industry_segment"]] += score
        sources.add(rule["source_type"])
        matched.extend(hits)
        (core_hits if rule["match_type"] == "core" else context_hits).extend(hits)
        for term in hits:
            term_scores[term] = max(term_scores.get(term, 0), score)

    negative_hits = [term for rule in COMPILED_NEGATIVE for term, pattern in rule["patterns"] if pattern.search(text)]
    score = maximum
    hard_negative = any(term in {"重量为6g", "重6g", "6g重量", "6g剂量", "加入6g", "称取6g"} for term in negative_hits)
    if hard_negative:
        score = 0
    elif negative_hits and not sixg_context:
        score = 0
    elif negative_hits and score > 1:
        score -= 1

    main_category = _top(cats)
    main_sub_pool = {key: value for key, value in subs.items() if key[0] == main_category}
    main_sub_category = (max(main_sub_pool.items(), key=lambda item: (item[1], item[0][1]))[0][1]
                         if main_sub_pool else "")
    return {
        "sixg_score_raw": raw,
        "sixg_score": score,
        "core_score": score,
        "max_matched_keyword_score": maximum,
        "matched_terms": "；".join(sorted(set(matched))),
        "matched_core_terms": "；".join(sorted(set(core_hits))),
        "matched_context_terms": "；".join(sorted(set(context_hits))),
        "matched_term_scores": "；".join(f"{term}:{term_scores[term]}" for term in sorted(term_scores)),
        "inactive_terms_no_context": "；".join(sorted(set(inactive))),
        "negative_terms": "；".join(sorted(set(negative_hits))),
        "main_category": main_category,
        "main_sub_category": main_sub_category,
        "industry_segment": _top(segments),
        "category_scores": json.dumps(dict(cats), ensure_ascii=False, sort_keys=True),
        "subcategory_scores": json.dumps({f"{a}/{b}": value for (a, b), value in subs.items()}, ensure_ascii=False, sort_keys=True),
        "source_types": "；".join(sorted(sources)),
        "has_sixg_context": int(sixg_context),
        "has_inferred_sixg_context": int(inferred_context),
        "signature_group_count": len(signature_groups),
    }


def make_text_series(df, cn_abs_col="摘要 (中文)", en_abs_col=None, extra_text_cols=None):
    cols = [col for col in ([cn_abs_col, en_abs_col] + list(extra_text_cols or [])) if col and col in df.columns]
    if not cols:
        return pd.Series([""] * len(df), index=df.index, dtype="object")
    return df[cols].fillna("").astype(str).agg(" ".join, axis=1).map(normalize_text)


def _join_unique(series):
    values = [value.strip() for item in series.dropna().astype(str)
              for value in item.split("；") if value.strip()]
    return "；".join(sorted(set(values)))


def _mode(series):
    values = [str(value) for value in series.dropna() if str(value).strip()]
    return Counter(values).most_common(1)[0][0] if values else ""


def summarize_sixg_firms(patents, firm_col="第一申请人", year_col="year", region_col=None, firm_type_col=None):
    if patents.empty or firm_col not in patents.columns:
        return pd.DataFrame(), pd.DataFrame()
    data = patents[patents[firm_col].notna()].copy()
    data[firm_col] = data[firm_col].astype(str).str.strip()
    data = data[data[firm_col] != ""]
    regions = [region_col] if isinstance(region_col, str) else list(region_col or [])
    regions = [col for col in regions if col in data.columns]
    group_year, group_firm = [firm_col] + regions + [year_col], [firm_col] + regions
    common = dict(
        sixg_patent_count=("is_sixg_patent", "sum"),
        sixg_score_sum=("sixg_score", "sum"),
        sixg_score_mean=("sixg_score", "mean"),
        sixg_score_max=("sixg_score", "max"),
        evidence_score_sum=("sixg_score_raw", "sum"),
        evidence_score_mean=("sixg_score_raw", "mean"),
        main_categories=("main_category", _join_unique),
        main_sub_categories=("main_sub_category", _join_unique),
        industry_segments=("industry_segment", _join_unique),
        matched_terms=("matched_terms", _join_unique),
    )
    year_agg = common.copy()
    firm_agg = dict(first_year=(year_col, "min"), last_year=(year_col, "max"), **common)
    if firm_type_col and firm_type_col in data.columns:
        year_agg["first_applicant_types"] = (firm_type_col, _join_unique)
        firm_agg["first_applicant_types"] = (firm_type_col, _join_unique)
    firm_year = data.groupby(group_year, dropna=False).agg(**year_agg).reset_index()
    firm = data.groupby(group_firm, dropna=False).agg(**firm_agg).reset_index()
    for source, target in [("main_category", "firm_main_category"),
                           ("main_sub_category", "firm_main_sub_category"),
                           ("industry_segment", "firm_main_industry_segment")]:
        dominant = data.groupby(group_firm, dropna=False)[source].agg(_mode).reset_index().rename(columns={source: target})
        firm = firm.merge(dominant, on=group_firm, how="left")
    return firm_year, firm


def tag_sixg_patents(df, cn_abs_col="摘要 (中文)", en_abs_col=None, firm_col="第一申请人", year_col="year",
                     region_col=None, firm_type_col=None, extra_text_cols=None, split_firms=False,
                     firm_sep_regex=r"[;；,，、|/]+", coarse_screen=True, progress_every=10000, min_score=1):
    data = df.copy()
    texts = make_text_series(data, cn_abs_col, en_abs_col, extra_text_cols)
    if coarse_screen:
        signature_count = sum(texts.str.contains(regex, na=False).astype(int) for regex in SIGNATURE_REGEXES)
        mask = (texts.str.contains(INDEPENDENT_REGEX, na=False)
                | texts.str.contains(SIXG_REGEX, na=False)
                | ((signature_count >= 3) & texts.str.contains(CONTEXTUAL_REGEX, na=False)))
        data, texts = data.loc[mask].copy(), texts.loc[mask]
    start, results = time.time(), []
    for index, text in enumerate(texts, 1):
        results.append(score_one_patent_text(text))
        if progress_every and (index % progress_every == 0 or index == len(data)):
            elapsed = time.time() - start
            speed = index / elapsed if elapsed else 0
            remain = (len(data) - index) / speed if speed else 0
            print(f"已处理 {index:,}/{len(data):,} 条候选，已用 {elapsed/60:.1f} 分钟，预计剩余 {remain/60:.1f} 分钟")
    columns = list(score_one_patent_text("").keys())
    scored = pd.DataFrame(results, index=data.index, columns=columns)
    tagged = pd.concat([data, scored], axis=1)
    tagged["is_sixg_patent"] = (tagged["sixg_score"] >= min_score).astype(int)
    if split_firms and firm_col in tagged.columns:
        tagged["_firm"] = tagged[firm_col].fillna("").astype(str).str.split(firm_sep_regex)
        tagged = tagged.explode("_firm")
        tagged[firm_col] = tagged["_firm"].str.strip()
        tagged = tagged[tagged[firm_col] != ""].drop(columns="_firm")
    formal = tagged[tagged["is_sixg_patent"] == 1].copy()
    firm_year, firm = summarize_sixg_firms(formal, firm_col, year_col, region_col, firm_type_col)
    return tagged, formal, firm_year, firm


def export_keyword_dictionary():
    return pd.DataFrame([
        {"关键词": term, "技术领域": rule["category"], "细分方向": rule["sub_category"],
         "产业板块": rule["industry_segment"], "核心程度得分": rule["score"],
         "匹配类型": rule["match_type"],
         "上下文要求": ("无需上下文；领域锚点本身只给低分" if rule["match_type"] == "core"
                        else "需6G/IMT-2030上下文，或至少三个独立6G候选技术簇同时成立"),
         "来源类型": rule["source_type"]}
        for rule in SIXG_RULES for term in rule["terms"]
    ])


if __name__ == "__main__":
    print(f"6G词典包含 {len(export_keyword_dictionary()):,} 个关键词/缩写。")
