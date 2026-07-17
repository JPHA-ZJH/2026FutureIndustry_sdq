# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
使用“中国全量专利数据库2021—2025年.csv”识别量子专利。

数据口径：
1. 仅保留“专利类型 == 发明授权”；
2. 文本仅使用“专利名称 + 摘要文本 + 主权项内容”；
3. 从“申请人”中按中文分号“；”提取第一项作为“第一申请人”；
4. 同步提取第一申请人对应的类型、地区和城市；
5. 按第一申请人—地区—城市—年份汇总。
"""

import os
import re
import sys
import time
from pathlib import Path

import pandas as pd


# =========================
# 1. 参数设置
# =========================

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(r"D:\BaiduNetdiskDownload\中国专利数据库1985-2025.11\分年份保存数据\分年份保存数据")
OUTPUT_ROOT = Path(r"F:\01科研\横向课题\未来产业识别\量子信息\data")

START_YEAR = 2021
END_YEAR = 2025

# 每次读取的行数。内存较小时可改成 50000；内存充足可改成 200000 或更大。
CHUNK_SIZE = 100000

# 正式运行保持 None；测试时可改为 10000，表示每个年份只读取前 10000 行。
NROWS_PER_YEAR = None

# 默认沿用 pd.read_csv 自动使用的 UTF-8 系列编码。
# 若出现 UnicodeDecodeError，可改为 "gb18030"。
CSV_ENCODING = None

# 默认不按申请人类型筛选，因此高校、科研院所、个人等也会保留。
# 若确认“第一申请人类型”的企业取值就是“企业”，可改为 {"企业"}。
ALLOWED_FIRST_APPLICANT_TYPES = None

TARGET_COLS = [
    "专利名称",
    "专利类型",
    "申请人",
    "申请人类型",
    "申请人地区",
    "申请人城市",
    "公开公告年份",
    "IPC主分类号",
    "摘要文本",
    "主权项内容",
    "被引证次数",
]

TEXT_COLS = ["专利名称", "摘要文本", "主权项内容"]

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from quantum_matcher_v2_optimized_region_newsource import (  # noqa: E402
    summarize_quantum_firms,
    tag_quantum_patents,
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 辅助函数
# =========================

def extract_first_semicolon_item(value):
    """按“；”取第一项；同时兼容极少数英文分号“;”。"""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    first = re.split(r"\s*[；;]\s*", text, maxsplit=1)[0].strip()
    return first if first else pd.NA


def add_first_applicant_fields(df: pd.DataFrame) -> pd.DataFrame:
    """同时提取第一申请人的名称、类型、地区和城市。"""
    data = df.copy()
    source_to_new = {
        "申请人": "第一申请人",
        "申请人类型": "第一申请人类型",
        "申请人地区": "第一申请人地区",
        "申请人城市": "第一申请人城市",
    }
    for source_col, new_col in source_to_new.items():
        data[new_col] = data[source_col].map(extract_first_semicolon_item)
    return data


def check_file_columns(file_path: Path) -> None:
    read_kwargs = {"nrows": 0}
    if CSV_ENCODING:
        read_kwargs["encoding"] = CSV_ENCODING
    header = pd.read_csv(file_path, **read_kwargs)
    missing = [c for c in TARGET_COLS if c not in header.columns]
    if missing:
        raise KeyError(
            f"文件 {file_path.name} 缺少变量：{missing}\n"
            f"实际变量为：{list(header.columns)}"
        )


def iter_year_chunks(file_path: Path):
    read_kwargs = dict(
        usecols=TARGET_COLS,
        low_memory=False,
        chunksize=CHUNK_SIZE,
        nrows=NROWS_PER_YEAR,
    )
    if CSV_ENCODING:
        read_kwargs["encoding"] = CSV_ENCODING
    return pd.read_csv(file_path, **read_kwargs)


# =========================
# 3. 逐年、分块识别
# =========================

all_quantum_parts = []
run_start = time.time()

for file_year in range(START_YEAR, END_YEAR + 1):
    file_path = DATA_ROOT / f"中国全量专利数据库{file_year}年.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件：{file_path}")

    check_file_columns(file_path)
    print(f"\n{'=' * 70}\n开始处理：{file_path}\n{'=' * 70}")

    year_raw_count = 0
    year_grant_count = 0
    year_quantum_count = 0

    for chunk_no, df_chunk in enumerate(iter_year_chunks(file_path), start=1):
        raw_n = len(df_chunk)
        year_raw_count += raw_n

        # 仅保留发明授权；先筛选再做文本识别，可大幅减少计算量。
        patent_type = df_chunk["专利类型"].fillna("").astype(str).str.strip()
        df_chunk = df_chunk.loc[patent_type.eq("发明授权")].copy()
        grant_n = len(df_chunk)
        year_grant_count += grant_n

        if df_chunk.empty:
            print(f"{file_year} 年第 {chunk_no} 块：原始 {raw_n:,}，发明授权 0，跳过")
            continue

        # 以公开公告年份作为企业—年份口径；缺失时才用文件年份补齐。
        public_year = pd.to_numeric(df_chunk["公开公告年份"], errors="coerce")
        missing_year_n = int(public_year.isna().sum())
        mismatch_n = int((public_year.notna() & public_year.ne(file_year)).sum())
        df_chunk["year"] = public_year.fillna(file_year).astype("Int64")

        # 第一申请人及其对应属性均按“；”取第一项。
        df_chunk = add_first_applicant_fields(df_chunk)

        # 可选：只保留指定类型的第一申请人。默认不启用，避免误删。
        if ALLOWED_FIRST_APPLICANT_TYPES is not None:
            df_chunk = df_chunk[
                df_chunk["第一申请人类型"].isin(ALLOWED_FIRST_APPLICANT_TYPES)
            ].copy()

        # 没有第一申请人的记录不能进入企业汇总，但仍可识别为专利。
        missing_firm_n = int(df_chunk["第一申请人"].isna().sum())

        _, quantum_chunk, _, _ = tag_quantum_patents(
            df_chunk,
            cn_abs_col="摘要文本",
            en_abs_col=None,
            firm_col="第一申请人",
            year_col="year",
            region_col=["第一申请人地区", "第一申请人城市"],
            firm_type_col="第一申请人类型",
            extra_text_cols=["专利名称", "主权项内容"],
            split_firms=False,
            # 新版 matcher 的粗筛已覆盖全部 core 词，开启后不改变正式识别口径。
            coarse_screen=True,
            progress_every=10000,
            include_review=False,
            include_timefreq=True,
        )

        quantum_chunk["来源文件年份"] = file_year
        quantum_chunk["来源分块"] = chunk_no
        all_quantum_parts.append(quantum_chunk)
        year_quantum_count += len(quantum_chunk)

        print(
            f"{file_year} 年第 {chunk_no} 块完成：原始 {raw_n:,}，"
            f"发明授权 {grant_n:,}，量子相关 {len(quantum_chunk):,}；"
            f"年份缺失 {missing_year_n:,}，年份与文件名不一致 {mismatch_n:,}，"
            f"第一申请人缺失 {missing_firm_n:,}"
        )

    print(
        f"{file_year} 年完成：原始 {year_raw_count:,}，"
        f"发明授权 {year_grant_count:,}，量子相关 {year_quantum_count:,}"
    )


# =========================
# 4. 合并、去重和企业汇总
# =========================

if all_quantum_parts:
    quantum_patents = pd.concat(all_quantum_parts, ignore_index=True)

    # 数据没有公开（公告）号，只能使用现有字段做保守去重。
    # 不建议只按专利名称去重，因为不同申请人可能存在同名专利。
    dedup_cols = [
        "专利名称",
        "申请人",
        "公开公告年份",
        "IPC主分类号",
        "摘要文本",
        "主权项内容",
    ]
    dedup_cols = [c for c in dedup_cols if c in quantum_patents.columns]
    before_dedup = len(quantum_patents)
    quantum_patents = quantum_patents.drop_duplicates(subset=dedup_cols).copy()
    print(f"\n量子专利合并后去重：{before_dedup:,} -> {len(quantum_patents):,}")

    firm_year_quantum, firm_quantum = summarize_quantum_firms(
        quantum_patents,
        firm_col="第一申请人",
        year_col="year",
        region_col=["第一申请人地区", "第一申请人城市"],
        firm_type_col="第一申请人类型",
    )
else:
    quantum_patents = pd.DataFrame()
    firm_year_quantum = pd.DataFrame()
    firm_quantum = pd.DataFrame()


# =========================
# 5. 导出结果
# =========================

quantum_path = OUTPUT_ROOT / "quantum_patents_2021_2025_newsource.csv"
firm_year_path = OUTPUT_ROOT / "firm_year_quantum_2021_2025_newsource.csv"
firm_path = OUTPUT_ROOT / "firm_quantum_2021_2025_newsource.csv"

quantum_patents.to_csv(quantum_path, index=False, encoding="utf-8-sig")
firm_year_quantum.to_csv(firm_year_path, index=False, encoding="utf-8-sig")
firm_quantum.to_csv(firm_path, index=False, encoding="utf-8-sig")

elapsed = time.time() - run_start
unique_firms = (
    quantum_patents["第一申请人"].dropna().nunique()
    if "第一申请人" in quantum_patents.columns
    else 0
)

print(f"\n识别完成，总耗时：{elapsed / 60:.2f} 分钟")
print(f"正式量子相关专利数量：{len(quantum_patents):,}")
print(f"第一申请人数量：{unique_firms:,}")
print("结果已导出：")
print(quantum_path)
print(firm_year_path)
print(firm_path)