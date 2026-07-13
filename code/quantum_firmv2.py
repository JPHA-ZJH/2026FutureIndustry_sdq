# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:29:18 2026

@author: Zz
"""


import os
import time
import pandas as pd
os.chdir(r"F:\01科研\横向课题\未来产业识别\code")
from quantum_matcher_v2_optimized_region import tag_quantum_patents


# =========================
# 1. 参数设置
# =========================

DATA_ROOT = r'E:/新专利数据/'
OUTPUT_ROOT = r'F:/01科研/横向课题/未来产业识别/量子信息/data'

START_YEAR = 2019
END_YEAR = 2023

YEAR_PART_COUNT = {
    2005: 1, 2006: 1, 2007: 1, 2008: 1, 2009: 1, 2010: 1, 2011: 1,
    2012: 3, 2013: 3, 2014: 5, 2015: 4, 2016: 5, 2017: 6,
    2018: 7, 2019: 8, 2020: 8, 2021: 7, 2022: 6, 2023: 2
}

TARGET_COLS = [
    '标题 (中文)',
    '标题 (英文)',
    '摘要 (中文)',
    '摘要 (英文)',
    '技术功效句',
    '独立权利要求',
    '首权翻译',
    '首项权利要求',
    '公开（公告）号',
    '申请人国家/地区',
    '申请人省市代码',
    '中国申请人地市',
    '申请人',
    '第一申请人',
    'IPC主分类',
    '授权公告日',
    'year'
]

os.makedirs(OUTPUT_ROOT, exist_ok=True)


# =========================
# 2. 读取逐年数据
# =========================

yearly_data = []

for year in range(START_YEAR, END_YEAR + 1):
    part_count = YEAR_PART_COUNT[year]
    part_dfs = []

    for part in range(1, part_count + 1):
        if part_count == 1:
            filename = f'patent{year}.csv'
        else:
            filename = f'patent{year}part_{part}.csv'

        file_path = os.path.join(DATA_ROOT, filename)

        print(f"正在读取：{file_path}")

        df_part = pd.read_csv(
            file_path,
            usecols=TARGET_COLS,
            low_memory=False
        )

        # 只保留授权专利
        df_part = df_part[df_part['授权公告日'].notna()].copy()

        part_dfs.append(df_part)

    yearly_df = pd.concat(part_dfs, ignore_index=True).drop_duplicates()
    yearly_data.append(yearly_df)

    print(f'{year} 年读取完成：{len(yearly_df):,} 行')

data = pd.concat(yearly_data, ignore_index=True)

print('合并后总行数：', len(data))


# =========================
# 3. 测试阶段：先跑一小部分
# =========================

# 第一次建议先测试 10000 条
# 确认结果正常后，再注释掉这一句跑全量
#df_run = data.head(10000).copy()

# 如果要跑全量，用这一句：
# df_run = df.copy()


# =========================
# 4. 调用新版量子识别函数
# =========================

start = time.time()

patent_tagged, quantum_patents, firm_year_quantum, firm_quantum = tag_quantum_patents(
    data,
    cn_abs_col="摘要 (中文)",
    en_abs_col="摘要 (英文)",
    firm_col="第一申请人",
    year_col="year",
    region_col="申请人省市代码",
    extra_text_cols=[
        "标题 (中文)",
        "标题 (英文)",
        "技术功效句",
        "独立权利要求",
        "首权翻译",
        "首项权利要求"
    ],
    split_firms=False,
    coarse_screen=False,
    progress_every=10000,
    include_review=False,
    include_timefreq=True
)

elapsed = time.time() - start

print(f"识别完成，耗时：{elapsed / 60:.2f} 分钟")
print(f"正式量子相关专利数量：{len(quantum_patents):,}")
print(f"正式量子相关企业数量：{firm_quantum['第一申请人'].nunique() if len(firm_quantum) > 0 else 0:,}")


# =========================
# 5. 导出结果
# =========================



quantum_patents.to_csv(
    os.path.join(OUTPUT_ROOT, "quantum_patents_v3.csv"),
    index=False,
    encoding="utf-8-sig"
)

firm_year_quantum.to_csv(
    os.path.join(OUTPUT_ROOT, "firm_year_quantum_v3.csv"),
    index=False,
    encoding="utf-8-sig"
)

firm_quantum.to_csv(
    os.path.join(OUTPUT_ROOT, "firm_quantum_v3.csv"),
    index=False,
    encoding="utf-8-sig"
)

print("结果已导出到：", OUTPUT_ROOT)



