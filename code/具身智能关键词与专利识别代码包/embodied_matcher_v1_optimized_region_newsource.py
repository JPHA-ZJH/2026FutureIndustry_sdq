# -*- coding: utf-8 -*-
"""
具身智能专利匹配与评分模块（中文专利文本版）。

设计原则
--------
1. 词典以中文术语为主体，保留中文专利中常见的英文缩写（VLA、VLM、
   LLM、MPC、WBC、ROS、AMR、eVTOL、RGB-D、SLAM 等），不重复维护
   一套完整英文词典。
2. “具身智能、具身基础模型、VLA”等强专用词可独立触发。
3. 世界模型、强化学习、触觉传感器、电机等通用词必须与机器人/物理本体
   上下文共现；上下文成立后，仍按其在具身智能技术体系中的核心程度计分。
4. 不输出“高相关/低相关/待复核”等离散标签，只输出连续得分、最高关键词
   核心度、技术领域、细分方向和产业板块。低相关专利以较低分体现。

关键词与分类主要依据：
- 中国信息通信研究院、清华大学电子工程系《具身智能发展报告（2025年）》；
- 《2025中国具身智能产业星图》中的底座技术、核心部件、通用/专用玩家、
  灵巧手及末端执行器等结构。
- 中国人工智能学会《具身智能（2026修订版）》及“2026具身智能十五大方向”；
- 中国信通院《具身智能发展报告（2024年）》《人形机器人产业发展研究报告
  （2024年）》以及公开券商研报对模型、关节、传感与执行部件的补充。
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# 1. 关键词规则
# ---------------------------------------------------------------------
# score：关键词在具身智能技术体系中的核心程度，1（外围支撑）—5（核心）。
# match_type：
#   core        强具身智能专用词，可独立匹配；
#   carrier     报告明确列出的典型具身本体/产品，可独立匹配；
#   contextual  通用算法/能力词，需具身本体上下文；
#   data        数据与训练词，需具身本体上下文；
#   component   零部件词，需具身本体上下文；
#   platform    云边端/平台词，需具身本体上下文；
#   application 场景词，需具身本体上下文。

EMBODIED_RULES: List[Dict] = [
    # ---- 核心概念与闭环 ----
    {"category": "具身智能基础", "sub_category": "核心概念", "terms": ["具身智能", "具身人工智能", "具身智能体", "具身AI", "物理智能", "具身系统"], "score": 5, "match_type": "core", "industry_segment": "技术服务", "source_type": "报告直接词"},
    {"category": "具身智能基础", "sub_category": "智能闭环", "terms": ["感知-认知-决策-执行", "感知—认知—决策—执行", "感知认知决策执行", "智能闭环", "知行合一", "身体交互", "物理实体性", "环境交互性", "物理本体与环境交互"], "score": 5, "match_type": "core", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "具身智能基础", "sub_category": "具身基础模型", "terms": ["具身基础模型", "通用具身基础模型", "具身大模型", "机器人基础模型", "机器人通用大脑", "通用机器人基础模型", "大行为模型"], "score": 5, "match_type": "core", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "具身智能基础", "sub_category": "具身认知与操作", "terms": ["具身认知", "具身感知", "具身操作", "具身决策规划", "具身空间推理", "具身大小脑协同"], "score": 4, "match_type": "core", "industry_segment": "技术服务", "source_type": "公开科研白皮书补充", "source_detail": "中国人工智能学会《具身智能（2026修订版）》及十五大方向", "source_url": "https://www.caai.cn/site/content/7747.html"},

    # ---- 四条模型/算法路线 ----
    {"category": "模型与算法", "sub_category": "模块化分层路线", "terms": ["模块化分层", "分层控制", "模块化控制", "任务规划与运动控制", "感知算法集成", "人工编程控制", "组合式运控"], "score": 4, "match_type": "contextual", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "模型与算法", "sub_category": "分层大模型路线", "terms": ["分层大模型", "大模型+机器人", "大模型机器人", "模型在环", "规划器", "任务推理", "任务规划", "规划推理", "复杂任务拆解", "函数调用", "API调用", "大小脑协同", "大小脑一体化"], "score": 4, "match_type": "contextual", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "模型与算法", "sub_category": "视觉语言模型", "terms": ["视觉语言模型", "VLM", "多模态大模型", "多模态认知", "视觉语言理解", "空间关系图", "空间感知", "可供性", "链式思考", "思维链"], "score": 4, "match_type": "contextual", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "模型与算法", "sub_category": "端到端VLA路线", "terms": ["视觉语言动作模型", "语言动作模型", "VLA", "端到端VLA", "多感官语言动作模型", "触觉-语言-动作模型", "触觉－语言－动作模型", "动作令牌", "动作生成模型", "通用动作空间", "跨本体迁移"], "score": 5, "match_type": "core", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "模型与算法", "sub_category": "VLA架构", "terms": ["单系统VLA", "双系统VLA", "VLA+VLM", "混合专家", "MOE", "多模态Token", "跨模态路由", "异步时间处理", "快慢系统", "推理与控制协同"], "score": 4, "match_type": "contextual", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "模型与算法", "sub_category": "世界模型路线", "terms": ["机器人世界模型", "具身世界模型", "可控世界模型"], "score": 5, "match_type": "core", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "模型与算法", "sub_category": "世界模型路线", "terms": ["世界模型", "动力学建模", "未来观察", "未来状态预测", "行为预演", "任务预演", "动作预测", "虚拟策略", "物理规律建模"], "score": 5, "match_type": "contextual", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展", "note": "通用词；与机器人/本体上下文共现后按核心路线计5分"},

    # ---- 机器人学习、运控与通用能力 ----
    {"category": "模型与算法", "sub_category": "运动控制", "terms": ["模型预测控制", "MPC", "全身控制", "WBC", "运动学模型", "动力学模型", "轨迹规划", "平衡控制", "全身协调控制", "高动态运动控制", "反应式控制"], "score": 4, "match_type": "contextual", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "模型与算法", "sub_category": "机器人学习", "terms": ["机器人学习", "具身学习", "技能学习", "运动技能学习", "动作策略学习", "行为学习", "策略学习", "强化学习", "模仿学习", "深度强化学习", "自监督学习", "监督微调", "SFT"], "score": 4, "match_type": "contextual", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "模型与算法", "sub_category": "操作策略学习", "terms": ["扩散策略", "Diffusion Policy", "行为克隆", "生成对抗模仿学习", "GAIL", "近端策略优化", "PPO", "动作分块Transformer", "Action Chunking Transformer", "ALOHA-ACT", "控制序列生成", "离散化动作", "动作序列标记"], "score": 4, "match_type": "contextual", "industry_segment": "技术服务", "source_type": "公开科研/资本研报补充", "source_detail": "中国人工智能学会《具身智能（2026修订版）》；海通国际EAI行业研究", "source_url": "https://ceai.caai.cn/static/upload/file/20260412/1775977669687642.pdf"},
    {"category": "模型与算法", "sub_category": "机器人Transformer", "terms": ["机器人Transformer", "RT-1", "RT-2", "RT-X", "OpenVLA", "PaLM-E", "机器人操作网络", "ROMAN"], "score": 4, "match_type": "contextual", "industry_segment": "技术服务", "source_type": "公开科研/资本研报补充", "source_detail": "中国人工智能学会《具身智能（2026修订版）》；海通国际EAI行业研究", "source_url": "https://ceai.caai.cn/static/upload/file/20260412/1775977669687642.pdf"},
    {"category": "模型与算法", "sub_category": "可扩展与持续学习", "terms": ["技能可扩展学习", "可扩展学习", "持续学习", "终身学习", "自我改进", "自主探索", "好奇心驱动", "奖励函数学习", "人在环", "交互式后训练", "专家纠正", "闭环训练"], "score": 4, "match_type": "contextual", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "模型与算法", "sub_category": "多模态感知", "terms": ["全模态感知", "多感官感知", "多模态融合", "视觉语言触觉", "视觉触觉融合", "视触觉融合", "力觉融合", "听觉融合", "RGB-D", "点云感知", "3D空间结构", "3D检测框"], "score": 4, "match_type": "contextual", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "模型与算法", "sub_category": "主动与交互式感知", "terms": ["主动感知", "交互式感知", "感知动作耦合", "视点规划", "主动视觉", "交互感知"], "score": 4, "match_type": "contextual", "industry_segment": "技术服务", "source_type": "公开科研白皮书补充", "source_detail": "中国人工智能学会2026具身智能十五大方向", "source_url": "https://www.caai.cn/site/content/7747.html"},
    {"category": "模型与算法", "sub_category": "空间推理", "terms": ["空间推理", "三维空间推理", "3D空间推理", "空间智能", "空间语义推理", "语义地图", "拓扑地图"], "score": 4, "match_type": "contextual", "industry_segment": "技术服务", "source_type": "公开科研/资本研报补充", "source_detail": "中国人工智能学会2026具身智能十五大方向；华源证券VLA与世界模型报告", "source_url": "https://www.caai.cn/site/content/7747.html"},
    {"category": "模型与算法", "sub_category": "群体具身智能", "terms": ["群体具身智能", "具身群体智能", "具身集群智能"], "score": 4, "match_type": "core", "industry_segment": "技术服务", "source_type": "公开科研白皮书补充", "source_detail": "中国人工智能学会2026具身智能十五大方向", "source_url": "https://www.caai.cn/site/content/7747.html"},
    {"category": "模型与算法", "sub_category": "自主任务能力", "terms": ["自主导航", "自主定位", "自主避障", "自主决策", "自主作业", "自主控制", "自主航行", "路径规划", "任务分配", "多机协同", "多智能体协同", "抓取规划", "操作规划", "长程任务", "泛化能力", "跨场景泛化", "跨任务泛化"], "score": 4, "match_type": "contextual", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},

    # ---- 数据与训练 ----
    {"category": "数据与训练", "sub_category": "具身数据", "terms": ["具身智能数据", "具身数据", "机器人操作数据", "机器人行为数据", "真实行为互动数据", "本体轨迹", "动作轨迹", "动作视频", "互联网多模态数据"], "score": 4, "match_type": "data", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "数据与训练", "sub_category": "仿真合成数据", "terms": ["仿真合成数据", "合成数据管线", "物理模拟引擎", "物理仿真引擎", "仿真平台", "仿真环境", "3D物体资产", "3D场景", "生成式模型合成", "仿真数据生成"], "score": 3, "match_type": "data", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "数据与训练", "sub_category": "具身仿真引擎", "terms": ["具身仿真引擎", "具身智能仿真引擎", "机器人仿真引擎"], "score": 4, "match_type": "core", "industry_segment": "技术服务", "source_type": "公开科研白皮书补充", "source_detail": "中国人工智能学会2026具身智能十五大方向", "source_url": "https://www.caai.cn/site/content/7747.html"},
    {"category": "数据与训练", "sub_category": "真实数据采集", "terms": ["真机数据", "真实数据采集", "遥操作数据采集", "开放环境采集", "动作捕捉", "光学捕捉", "惯性捕捉", "专家示教", "手把手教学", "VR遥操", "同构遥操", "外骨骼采集", "动捕服", "手持操纵杆"], "score": 4, "match_type": "data", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "数据与训练", "sub_category": "迁移与训练策略", "terms": ["仿真到真实", "真实到仿真", "真机后训练", "仿真预训练", "预训练", "后训练", "Sim2Real", "数字孪生训练", "域随机化", "策略迁移", "数据回流"], "score": 3, "match_type": "data", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "数据与训练", "sub_category": "训练基础设施", "terms": ["具身智能训练场", "机器人训练场", "数据训练中心", "数据采集工厂", "超级数据工厂", "异构机器人训练", "异构群智", "数据集构建", "数据标注", "数据质量评估"], "score": 3, "match_type": "data", "industry_segment": "技术服务", "source_type": "报告直接词/支持扩展"},
    {"category": "数据与训练", "sub_category": "数据飞轮", "terms": ["具身数据飞轮", "机器人数据飞轮", "数据飞轮", "任务反馈数据", "失败案例回流", "模型迭代闭环"], "score": 4, "match_type": "data", "industry_segment": "技术服务", "source_type": "公开资本研报补充", "source_detail": "东吴证券《机器人大模型深度报告》", "source_url": "https://pdf.dfcfw.com/pdf/H3_AP202508091724252221_1.pdf"},

    # ---- 典型具身本体/产品 ----
    {"category": "本体与产品", "sub_category": "人形机器人", "terms": ["人形机器人", "双足人形机器人", "轮式人形机器人", "双足机器人", "仿人机器人", "人形本体"], "score": 4, "match_type": "carrier", "industry_segment": "产品服务", "source_type": "两报告直接词"},
    {"category": "本体与产品", "sub_category": "轮臂与移动操作", "terms": ["轮臂机器人", "轮臂式机器人", "复合轮臂机器人", "复合轮臂式机器人", "轮式机器人+机械臂", "移动操作机器人", "移动机械臂"], "score": 4, "match_type": "carrier", "industry_segment": "产品服务", "source_type": "报告直接词/支持扩展"},
    {"category": "本体与产品", "sub_category": "足式机器人", "terms": ["四足机器人", "四足机器狗", "机器狗", "多足机器人", "六足机器人", "轮足机器人", "轮足机器狗", "足式机器人", "轮足复合机器人"], "score": 4, "match_type": "carrier", "industry_segment": "产品服务", "source_type": "两报告直接词/支持扩展"},
    {"category": "本体与产品", "sub_category": "机械臂与协作机器人", "terms": ["协作机器人", "协作机械臂", "智能机械臂", "双臂机器人", "双臂操作平台", "柔性机器人"], "score": 3, "match_type": "carrier", "industry_segment": "产品服务", "source_type": "报告直接词/支持扩展"},
    {"category": "本体与产品", "sub_category": "自主移动机器人", "terms": ["自主移动机器人", "AMR", "自主移动小车", "无人配送车", "无人环卫车", "无人公交", "飞行机器人"], "score": 3, "match_type": "carrier", "industry_segment": "产品服务", "source_type": "两报告直接词/支持扩展"},
    {"category": "本体与产品", "sub_category": "仿生与特种机器人", "terms": ["仿生机器人", "软体机器人", "蛇形机器人", "仿生机器狗", "水下机器人", "巡检机器人", "搜救机器人", "防爆机器人", "特种机器人"], "score": 3, "match_type": "carrier", "industry_segment": "产品服务", "source_type": "报告直接词/支持扩展"},
    {"category": "本体与产品", "sub_category": "微型与集群机器人", "terms": ["微型机器人集群", "集群式微型智能机器人", "微型智能机器人", "机器人集群", "群体机器人"], "score": 3, "match_type": "carrier", "industry_segment": "产品服务", "source_type": "报告直接词/支持扩展"},
    {"category": "本体与产品", "sub_category": "智能运载装备", "terms": ["自动驾驶汽车", "无人驾驶汽车", "自动驾驶车辆", "无人驾驶小车", "Robotaxi", "无人驾驶航空器", "eVTOL", "无人船", "自主航行船"], "score": 3, "match_type": "carrier", "industry_segment": "产品服务", "source_type": "报告直接词/支持扩展", "note": "报告列为具身智能载体，但与既有产业交叉，得分低于核心模型词"},
    {"category": "本体与产品", "sub_category": "广义运载载体", "terms": ["无人机", "多旋翼飞行器", "无人车"], "score": 2, "match_type": "contextual", "context_requirement": "intelligence", "industry_segment": "产品服务", "source_type": "报告直接词", "note": "需同时出现自主感知、决策、导航、学习等智能上下文"},
    {"category": "本体与产品", "sub_category": "新型智能产品", "terms": ["变形移动装置", "自适应变形装置", "智能可穿戴设备", "智能外骨骼", "外骨骼机器人", "上肢外骨骼", "下肢外骨骼", "全身外骨骼", "康复外骨骼", "增强外骨骼"], "score": 3, "match_type": "carrier", "industry_segment": "产品服务", "source_type": "报告直接词/支持扩展"},

    # ---- 灵巧操作、感知、执行和结构部件 ----
    {"category": "核心部件与本体", "sub_category": "灵巧手与末端执行", "terms": ["灵巧手", "仿生灵巧手", "视触觉灵巧手", "多指灵巧手", "末端执行器", "机器人夹爪", "智能夹爪", "机械夹爪"], "score": 4, "match_type": "carrier", "industry_segment": "基础设施", "source_type": "两报告直接词/支持扩展"},
    {"category": "核心部件与本体", "sub_category": "力触觉感知", "terms": ["力触觉传感器", "六维力传感器", "六维力矩传感器", "六维力触觉传感器", "多维触觉传感器", "触觉传感器", "柔性触觉传感器", "MEMS触觉传感器", "力矩传感器", "压力传感器", "视触觉传感器", "电子皮肤", "柔性电子皮肤", "接近觉", "触觉编码器"], "score": 4, "match_type": "component", "industry_segment": "基础设施", "source_type": "两报告直接词/公开资本研报补充", "source_detail": "民生证券《智能具身创启未来纪元》", "source_url": "https://pdf.dfcfw.com/pdf/H3_AP202505181675062375_1.pdf"},
    {"category": "核心部件与本体", "sub_category": "视觉与环境感知", "terms": ["感知系统", "视觉系统", "机器视觉", "深度相机", "RGB-D相机", "TOF深度相机", "3D视觉", "激光雷达", "毫米波雷达", "点云", "SLAM", "视觉定位", "多传感器融合"], "score": 3, "match_type": "component", "industry_segment": "基础设施", "source_type": "报告直接词/公开资本研报补充"},
    {"category": "核心部件与本体", "sub_category": "惯性与姿态感知", "terms": ["惯性测量单元", "惯性传感器", "惯性导航单元", "IMU", "惯导IMU", "机器人陀螺仪", "机器人加速度计", "姿态传感器"], "score": 2, "match_type": "component", "industry_segment": "基础设施", "source_type": "公开资本研报补充", "source_detail": "浙商证券人形机器人报告及公开产业研究", "source_url": "https://pdf.dfcfw.com/pdf/H301_AP202307031591987350_1.pdf"},
    {"category": "核心部件与本体", "sub_category": "一体化关节", "terms": ["一体化关节", "电驱动一体化关节", "关节模组", "机器人关节模组", "无框力矩电机", "关节控制器", "关节热管理", "高扭矩关节", "高自由度关节"], "score": 4, "match_type": "component", "industry_segment": "基础设施", "source_type": "两报告直接词/支持扩展"},
    {"category": "核心部件与本体", "sub_category": "驱动与传动", "terms": ["谐波减速器", "行星减速器", "RV减速器", "精密减速器", "减速器", "伺服驱动器", "关节驱动器", "驱动器", "滚柱丝杠", "行星滚柱丝杠", "反向式行星滚柱丝杠", "滚珠丝杠", "梯形丝杠", "丝杠", "传动系统", "传动齿轮"], "score": 2, "match_type": "component", "industry_segment": "基础设施", "source_type": "两报告直接词/公开资本研报补充"},
    {"category": "核心部件与本体", "sub_category": "关节执行器路线", "terms": ["旋转执行器", "线性执行器", "串联弹性驱动器", "弹性驱动器", "SEA", "准直驱驱动器", "准直驱关节", "准直驱", "QDD", "半直驱", "本体驱动器"], "score": 3, "match_type": "component", "industry_segment": "基础设施", "source_type": "公开资本研报补充", "source_detail": "国金证券、民生证券等公开人形机器人产业链报告", "source_url": "https://pdf.dfcfw.com/pdf/H3_AP202401241617985089_1.pdf"},
    {"category": "核心部件与本体", "sub_category": "电机与基础机械件", "terms": ["力矩电机", "空心杯电机", "无框电机", "伺服电机", "电机控制器", "精密轴承", "交叉滚子轴承", "轴承"], "score": 1, "match_type": "component", "industry_segment": "基础设施", "source_type": "两报告直接词/支持扩展", "note": "通用性强，仅在机器人/本体上下文中记录低分"},
    {"category": "核心部件与本体", "sub_category": "位置与运动反馈", "terms": ["绝对值编码器", "双编码器", "磁编码器", "光电编码器", "角度编码器", "位置编码器", "机器人编码器"], "score": 1, "match_type": "component", "industry_segment": "基础设施", "source_type": "公开资本研报补充", "source_detail": "浙商证券、国金证券人形机器人产业链报告", "source_url": "https://pdf.dfcfw.com/pdf/H301_AP202307031591987350_1.pdf"},
    {"category": "核心部件与本体", "sub_category": "执行与移动系统", "terms": ["执行系统", "移动底盘", "轮式底盘", "线控底盘", "舵机", "运动执行器", "抓取机构", "操作机构"], "score": 2, "match_type": "component", "industry_segment": "基础设施", "source_type": "报告直接词/支持扩展"},
    {"category": "核心部件与本体", "sub_category": "能源系统", "terms": ["机器人能源系统", "固态电池", "高能量密度电池", "高倍率电池", "氢燃料电池", "氢动力机器人"], "score": 1, "match_type": "component", "industry_segment": "基础设施", "source_type": "两报告直接词/支持扩展"},
    {"category": "核心部件与本体", "sub_category": "高性能材料结构件", "terms": ["高性能材料结构件", "轻量化结构件", "铝合金结构件", "碳纤维复合材料", "聚醚醚酮", "PEEK"], "score": 1, "match_type": "component", "industry_segment": "基础设施", "source_type": "报告直接词/支持扩展"},

    # ---- 云边端、芯片、系统软件和平台 ----
    {"category": "云边端与平台", "sub_category": "端侧计算", "terms": ["端侧计算芯片", "机器人计算芯片", "端侧推理", "模型端侧部署", "高算力控制器", "算控一体", "算控一体化", "端侧大算力", "边缘端推理"], "score": 3, "match_type": "platform", "industry_segment": "基础设施", "source_type": "两报告直接词/支持扩展"},
    {"category": "云边端与平台", "sub_category": "云边端协同", "terms": ["云边端协同", "云-边-端", "云－边－端", "端边云协同", "边云协同", "云端训练", "分布式计算", "跨本体管理", "端云适配", "数据上传", "模型部署"], "score": 3, "match_type": "platform", "industry_segment": "基础设施", "source_type": "报告直接词/支持扩展"},
    {"category": "云边端与平台", "sub_category": "云侧服务", "terms": ["具身智能开放平台", "机器人云平台", "数据服务", "模型服务", "仿真服务", "仿真训练平台", "开发工具链", "机器人上下文协议"], "score": 3, "match_type": "platform", "industry_segment": "基础设施", "source_type": "报告直接词/支持扩展"},
    {"category": "云边端与平台", "sub_category": "操作系统与中间件", "terms": ["机器人操作系统", "ROS", "ROS 2", "中间件", "数据分发服务", "DDS", "具身操作系统", "跨本体开发", "模型调度", "异构算力整合"], "score": 3, "match_type": "platform", "industry_segment": "基础设施", "source_type": "报告直接词/支持扩展"},
    {"category": "云边端与平台", "sub_category": "底座支撑", "terms": ["端侧芯片", "通信模组", "机器人通信模组", "机器人存储", "大内存带宽", "低功耗计算", "热管理系统"], "score": 1, "match_type": "platform", "industry_segment": "基础设施", "source_type": "两报告直接词/支持扩展"},

    # ---- 评测、可信与工程化（科研白皮书补充方向） ----
    {"category": "评测安全与工程化", "sub_category": "真实世界评测", "terms": ["具身智能真实世界评测", "机器人真实世界评测", "具身智能评测体系", "具身任务评测", "开放环境评测", "仿真评测", "真机评测"], "score": 3, "match_type": "core", "industry_segment": "技术服务", "source_type": "公开科研白皮书补充", "source_detail": "中国人工智能学会2026具身智能十五大方向", "source_url": "https://www.caai.cn/site/content/7747.html"},
    {"category": "评测安全与工程化", "sub_category": "模型加速与轻量化", "terms": ["具身模型加速", "具身算法加速", "具身模型轻量化", "机器人模型轻量化", "机器人模型量化", "机器人模型剪枝", "机器人推理加速"], "score": 3, "match_type": "core", "industry_segment": "技术服务", "source_type": "公开科研白皮书补充", "source_detail": "中国人工智能学会2026具身智能十五大方向", "source_url": "https://www.caai.cn/site/content/7747.html"},
    {"category": "评测安全与工程化", "sub_category": "安全与价值对齐", "terms": ["具身智能安全", "具身安全", "具身价值对齐", "机器人价值对齐", "安全动作策略", "安全策略学习", "机器人行为安全", "人机共融安全"], "score": 3, "match_type": "core", "industry_segment": "技术服务", "source_type": "公开科研白皮书补充", "source_detail": "中国人工智能学会《具身智能（2026修订版）》及十五大方向", "source_url": "https://www.caai.cn/site/content/7747.html"},

    # ---- 行业应用：只在具身本体上下文成立时记低分 ----
    {"category": "行业应用", "sub_category": "工业制造", "terms": ["工业制造", "汽车装配", "3C电子工厂", "柔性制造", "物流分拣", "产线作业", "仓储搬运", "设备巡检", "电力巡检"], "score": 2, "match_type": "application", "industry_segment": "行业应用", "source_type": "报告直接词/支持扩展"},
    {"category": "行业应用", "sub_category": "商业与家庭", "terms": ["商业服务", "智慧药房", "取货送货", "补货", "无人零售", "家庭服务", "家庭康养", "接待导览", "展厅讲解", "教育陪伴", "餐饮服务"], "score": 1, "match_type": "application", "industry_segment": "行业应用", "source_type": "报告直接词/支持扩展"},
    {"category": "行业应用", "sub_category": "安全应急与特种", "terms": ["安全应急", "应急救援", "安防巡逻", "灾害搜救", "危险环境作业", "防爆作业", "消防救援", "野外探索", "月球探测"], "score": 2, "match_type": "application", "industry_segment": "行业应用", "source_type": "报告直接词/支持扩展"},
    {"category": "行业应用", "sub_category": "交通与低空", "terms": ["无人配送", "物流配送", "农林植保", "测绘勘探", "安防监控", "自主起降", "城市空中交通", "水文监测", "河道运维", "自主靠离泊"], "score": 1, "match_type": "application", "industry_segment": "行业应用", "source_type": "报告直接词/支持扩展"},
    {"category": "行业应用", "sub_category": "医疗康复", "terms": ["康复训练", "康复机器人", "辅助行走", "肢体运动康复", "靶向给药", "细胞操作", "医疗急救"], "score": 1, "match_type": "application", "industry_segment": "行业应用", "source_type": "报告直接词/支持扩展"},
]


NEGATIVE_RULES: List[Dict] = [
    {"category": "纯软件机器人", "terms": ["聊天机器人", "问答机器人", "客服机器人", "软件机器人", "机器人流程自动化", "RPA", "爬虫机器人", "网络机器人", "交易机器人"], "penalty": 8},
    {"category": "非具身智能体", "terms": ["虚拟智能体", "数字人", "游戏智能体", "对话智能体"], "penalty": 5},
]


# 用于判定通用关键词是否处于具身/物理本体语境。该列表本身不计分。
PHYSICAL_CONTEXT_TERMS = [
    "具身", "物理本体", "机器人本体", "本体构型", "本体控制", "本体运控", "机器人", "机械臂",
    "人形", "双足", "四足", "多足", "轮足", "轮臂", "机器狗", "无人车", "无人驾驶", "无人机",
    "无人船", "飞行器", "eVTOL", "外骨骼", "灵巧手", "末端执行器", "夹爪", "移动底盘", "关节模组",
]

INTELLIGENCE_CONTEXT_TERMS = [
    "具身智能", "物理智能", "环境交互", "智能闭环", "感知", "认知", "决策", "自主", "学习", "推理",
    "规划", "多模态", "视觉语言", "世界模型", "动作策略", "动作生成", "运动控制", "导航", "避障", "定位",
    "抓取", "操作", "协同", "泛化", "VLA", "VLM", "强化学习", "模仿学习", "SLAM",
]

# 负向词消歧：只有“机器人”三字时，聊天机器人等纯软件对象不能被误当作物理本体。
UNAMBIGUOUS_PHYSICAL_TERMS = [
    "具身", "物理本体", "机器人本体", "人形机器人", "双足机器人", "四足机器人", "足式机器人",
    "轮臂机器人", "移动操作机器人", "机械臂", "灵巧手", "末端执行器", "外骨骼机器人",
    "无人驾驶车辆", "无人驾驶航空器", "无人船", "移动底盘", "关节模组",
]
EXPLICIT_EMBODIED_TERMS = [
    "具身智能", "具身人工智能", "具身智能体", "具身AI", "物理智能", "具身基础模型",
    "具身大模型", "视觉语言动作模型", "端到端VLA", "VLA", "具身世界模型",
]


# ---------------------------------------------------------------------
# 2. 文本与正则工具
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


COMPILED_RULES = compile_rules(EMBODIED_RULES)
COMPILED_NEGATIVE = compile_rules(NEGATIVE_RULES)
PHYSICAL_PATTERNS = compile_terms(PHYSICAL_CONTEXT_TERMS)
INTELLIGENCE_PATTERNS = compile_terms(INTELLIGENCE_CONTEXT_TERMS)
UNAMBIGUOUS_PHYSICAL_PATTERNS = compile_terms(UNAMBIGUOUS_PHYSICAL_TERMS)
EXPLICIT_EMBODIED_PATTERNS = compile_terms(EXPLICIT_EMBODIED_TERMS)


def _make_union_regex(patterns: Sequence[Tuple[str, re.Pattern]]) -> re.Pattern:
    body = "|".join(pattern.pattern for _, pattern in patterns) or r"(?!)"
    return re.compile(body, flags=re.IGNORECASE)


INDEPENDENT_TERMS = []
CONTEXTUAL_TERMS = []
for _rule in EMBODIED_RULES:
    if _rule.get("match_type") in {"core", "carrier"}:
        INDEPENDENT_TERMS.extend(_rule.get("terms", []))
    else:
        CONTEXTUAL_TERMS.extend(_rule.get("terms", []))

INDEPENDENT_REGEX = _make_union_regex(compile_terms(sorted(set(INDEPENDENT_TERMS))))
CONTEXTUAL_REGEX = _make_union_regex(compile_terms(sorted(set(CONTEXTUAL_TERMS))))
PHYSICAL_REGEX = _make_union_regex(PHYSICAL_PATTERNS)
INTELLIGENCE_REGEX = _make_union_regex(INTELLIGENCE_PATTERNS)


def _has_any(text: str, patterns: Sequence[Tuple[str, re.Pattern]]) -> bool:
    return any(pattern.search(text) for _, pattern in patterns)


def _context_valid(requirement: str, physical_context: bool, intelligence_context: bool, strong_core: bool) -> bool:
    if strong_core:
        return True
    if requirement == "none":
        return True
    if requirement == "intelligence":
        return intelligence_context
    if requirement == "physical_and_intelligence":
        return physical_context and intelligence_context
    return physical_context


# ---------------------------------------------------------------------
# 3. 单条专利评分
# ---------------------------------------------------------------------
def _top_key(score_dict: Dict[str, int]) -> str:
    if not score_dict:
        return ""
    return max(score_dict.items(), key=lambda item: (item[1], item[0]))[0]


def score_one_patent_text(text: str) -> Dict:
    text = normalize_text(text)
    physical_context = _has_any(text, PHYSICAL_PATTERNS)
    intelligence_context = _has_any(text, INTELLIGENCE_PATTERNS)

    strong_core = False
    for item in COMPILED_RULES:
        if item.get("match_type") != "core":
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
    technical_category_scores: Dict[str, int] = defaultdict(int)
    technical_subcategory_scores: Dict[Tuple[str, str], int] = defaultdict(int)
    technical_segment_scores: Dict[str, int] = defaultdict(int)
    source_types = set()
    source_details = set()
    source_urls = set()
    total_raw = 0
    max_keyword_score = 0
    max_technical_score = 0
    max_carrier_score = 0

    for item in COMPILED_RULES:
        hits = [term for term, pattern in item["patterns"] if pattern.search(text)]
        if not hits:
            continue

        match_type = item.get("match_type", "contextual")
        requirement = item.get("context_requirement")
        if requirement is None:
            requirement = "none" if match_type in {"core", "carrier"} else "physical"

        valid = _context_valid(
            requirement,
            physical_context=physical_context,
            intelligence_context=intelligence_context,
            strong_core=strong_core,
        )
        if not valid:
            inactive_terms.extend(hits)
            continue

        score = int(item.get("score", 0))
        category = item.get("category", "")
        sub_category = item.get("sub_category", "")
        segment = item.get("industry_segment", "")

        # 同一规则命中多个同义词只加一次规则分，避免堆词造成虚高。
        total_raw += score
        category_scores[category] += score
        subcategory_scores[(category, sub_category)] += score
        segment_scores[segment] += score
        max_keyword_score = max(max_keyword_score, score)
        if match_type == "carrier":
            max_carrier_score = max(max_carrier_score, score)
        else:
            max_technical_score = max(max_technical_score, score)
            technical_category_scores[category] += score
            technical_subcategory_scores[(category, sub_category)] += score
            technical_segment_scores[segment] += score
        source_types.add(item.get("source_type", ""))
        source_details.add(item.get("source_detail", ""))
        source_urls.add(item.get("source_url", ""))

        matched_terms.extend(hits)
        if match_type == "core":
            matched_core_terms.extend(hits)
        else:
            matched_context_terms.extend(hits)
        for term in hits:
            matched_term_scores[term] = max(score, matched_term_scores.get(term, 0))

    negative_terms = []
    penalty = 0
    for item in COMPILED_NEGATIVE:
        hits = [term for term, pattern in item["patterns"] if pattern.search(text)]
        if hits:
            negative_terms.extend(hits)
            penalty += int(item.get("penalty", 0))

    # 最终分表示“专利所涉技术的核心程度”（1—5），而不是简单堆叠词频。
    # 载体词主要负责确认具身语境：若还命中部件/算法/数据/场景，则采用实际技术
    # 类型的最高分。例如“人形机器人关节用轴承”按轴承记1分，而不是4+1。
    total_score = max_technical_score if max_technical_score else max_carrier_score

    # 聊天/客服等纯软件机器人会包含“机器人”子串。若没有明确具身词或明确物理
    # 本体词，直接归零；若确有物理本体，只做1分保守扣减。
    if negative_terms:
        explicit_embodied = _has_any(text, EXPLICIT_EMBODIED_PATTERNS)
        unambiguous_physical = _has_any(text, UNAMBIGUOUS_PHYSICAL_PATTERNS)
        if not explicit_embodied and not unambiguous_physical:
            total_score = 0
        else:
            total_score = max(0, total_score - 1)

    effective_category_scores = technical_category_scores or category_scores
    effective_subcategory_scores = technical_subcategory_scores or subcategory_scores
    effective_segment_scores = technical_segment_scores or segment_scores

    main_category = _top_key(dict(effective_category_scores))
    main_sub_category = ""
    if effective_subcategory_scores:
        main_pair = max(effective_subcategory_scores.items(), key=lambda item: (item[1], item[0][1]))[0]
        main_sub_category = main_pair[1]
    industry_segment = _top_key(dict(effective_segment_scores))

    return {
        "embodied_score_raw": total_raw,
        "embodied_score": total_score,
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
        "has_physical_context": int(physical_context),
        "has_intelligence_context": int(intelligence_context),
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
    """合并实际存在的文本列；允许保留中文文本中的英文缩写。"""
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


def summarize_embodied_firms(
    embodied_patents: pd.DataFrame,
    firm_col: str = "第一申请人",
    year_col: str = "year",
    region_col: Optional[Union[str, Sequence[str]]] = None,
    firm_type_col: Optional[str] = None,
):
    """按第一申请人—地区—城市—年份及第一申请人—地区—城市汇总。"""
    if embodied_patents.empty or firm_col not in embodied_patents.columns:
        return pd.DataFrame(), pd.DataFrame()
    if year_col not in embodied_patents.columns:
        raise KeyError(f"缺少年份列：{year_col}")

    data = embodied_patents.copy()
    data = data[data[firm_col].notna()].copy()
    data[firm_col] = data[firm_col].astype(str).str.strip()
    data = data[data[firm_col] != ""].copy()
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    region_cols = _existing_region_cols(region_col, data)
    group_cols_year = [firm_col] + region_cols + [year_col]
    group_cols_firm = [firm_col] + region_cols

    common_aggs = dict(
        embodied_patent_count=("is_embodied_patent", "sum"),
        embodied_score_sum=("embodied_score", "sum"),
        embodied_score_mean=("embodied_score", "mean"),
        embodied_score_max=("embodied_score", "max"),
        core_score_mean=("core_score", "mean"),
        core_score_max=("core_score", "max"),
        main_categories=("main_category", lambda x: "；".join(sorted(set(x.dropna()) - {""}))),
        main_sub_categories=("main_sub_category", lambda x: "；".join(sorted(set(x.dropna()) - {""}))),
        industry_segments=("industry_segment", lambda x: "；".join(sorted(set(x.dropna()) - {""}))),
        matched_terms=("matched_terms", _join_unique_semicolon),
    )
    year_aggs = common_aggs.copy()
    firm_aggs = dict(
        first_year=(year_col, "min"),
        last_year=(year_col, "max"),
        **common_aggs,
    )
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


def tag_embodied_patents(
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
    """
    返回：patent_tagged, embodied_patents, firm_year_embodied, firm_embodied。

    min_score默认1：外围部件/场景专利在具身上下文成立时也会保留，但得分低。
    若希望更保守，可在调用端将min_score改为2或3。
    """
    data = df.copy()
    text_series = make_text_series(data, cn_abs_col, en_abs_col, extra_text_cols)

    if coarse_screen:
        independent_mask = text_series.str.contains(INDEPENDENT_REGEX, regex=True, na=False)
        contextual_mask = (
            text_series.str.contains(PHYSICAL_REGEX, regex=True, na=False)
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
        if progress_every and (index % progress_every == 0 or index == total):
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
    patent_tagged["is_embodied_patent"] = (patent_tagged["embodied_score"] >= min_score).astype(int)

    if split_firms and firm_col in patent_tagged.columns:
        patent_tagged[firm_col] = patent_tagged[firm_col].fillna("").astype(str)
        patent_tagged["_firm_list"] = patent_tagged[firm_col].str.split(firm_sep_regex)
        patent_tagged = patent_tagged.explode("_firm_list")
        patent_tagged[firm_col] = patent_tagged["_firm_list"].str.strip()
        patent_tagged = patent_tagged[patent_tagged[firm_col] != ""].copy()
        patent_tagged.drop(columns=["_firm_list"], inplace=True)

    embodied_patents = patent_tagged[patent_tagged["is_embodied_patent"] == 1].copy()
    firm_year, firm = summarize_embodied_firms(
        embodied_patents,
        firm_col=firm_col,
        year_col=year_col,
        region_col=region_col,
        firm_type_col=firm_type_col,
    )
    return patent_tagged, embodied_patents, firm_year, firm


def export_keyword_dictionary() -> pd.DataFrame:
    """将规则展开为一行一个关键词，便于审查和另存为CSV。"""
    rows = []
    for rule in EMBODIED_RULES:
        for term in rule.get("terms", []):
            rows.append(
                {
                    "关键词": term,
                    "技术领域": rule.get("category", ""),
                    "细分方向": rule.get("sub_category", ""),
                    "产业板块": rule.get("industry_segment", ""),
                    "核心程度得分": rule.get("score", 0),
                    "匹配类型": rule.get("match_type", ""),
                    "上下文要求": rule.get("context_requirement", "") or (
                        "无需上下文" if rule.get("match_type") in {"core", "carrier"} else "需机器人/物理本体上下文"
                    ),
                    "来源类型": rule.get("source_type", ""),
                    "来源说明": rule.get("source_detail", "") or "用户提供的两份具身智能报告",
                    "来源链接": rule.get("source_url", ""),
                    "说明": rule.get("note", ""),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("请导入本模块并调用 tag_embodied_patents(df, ...)。")
    print(f"当前词典包含 {len(export_keyword_dictionary()):,} 个关键词/缩写。")
