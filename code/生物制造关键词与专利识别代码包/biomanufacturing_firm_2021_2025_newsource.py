# -*- coding: utf-8 -*-
"""
使用“中国全量专利数据库2021—2025年.csv”识别生物制造专利。

数据口径：
1. 仅保留“专利类型 == 发明授权”；
2. 文本使用“专利名称 + 摘要文本 + 主权项内容”；
3. 按中文分号“；”提取第一申请人及其类型、地区和城市；
4. 按第一申请人—地区—城市—年份汇总；
5. 不划分高相关、低相关或待复核，只保留1—5分技术核心度和证据累计分。
"""

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
OUTPUT_ROOT = Path(r"F:\01科研\横向课题\未来产业识别\生物制造\data")

START_YEAR = 2021
END_YEAR = 2025
CHUNK_SIZE = 100000
NROWS_PER_YEAR = None
CSV_ENCODING = None

# 默认不按申请人类型筛选；如只保留企业，可改为 {"企业"}。
ALLOWED_FIRST_APPLICANT_TYPES = None

# 默认保留所有通过上下文校验的生物制造专利；设为2或3可获得更保守样本。
MIN_BIOMANUFACTURING_SCORE = 1

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

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from biomanufacturing_matcher_v1_optimized_region_newsource import (  # noqa: E402
    export_keyword_dictionary,
    summarize_biomanufacturing_firms,
    tag_biomanufacturing_patents,
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 辅助函数
# =========================

def extract_first_semicolon_item(value):
    """按“；”取第一项；兼容少量英文分号“;”。"""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    first = re.split(r"\s*[；;]\s*", text, maxsplit=1)[0].strip()
    return first if first else pd.NA


def add_first_applicant_fields(df: pd.DataFrame) -> pd.DataFrame:
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
    missing = [column for column in TARGET_COLS if column not in header.columns]
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

all_parts = []
run_start = time.time()

for file_year in range(START_YEAR, END_YEAR + 1):
    file_path = DATA_ROOT / f"中国全量专利数据库{file_year}年.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"找不到文件：{file_path}")

    check_file_columns(file_path)
    print(f"\n{'=' * 70}\n开始处理：{file_path}\n{'=' * 70}")

    year_raw_count = 0
    year_grant_count = 0
    year_matched_count = 0

    for chunk_no, df_chunk in enumerate(iter_year_chunks(file_path), start=1):
        raw_n = len(df_chunk)
        year_raw_count += raw_n

        patent_type = df_chunk["专利类型"].fillna("").astype(str).str.strip()
        df_chunk = df_chunk.loc[patent_type.eq("发明授权")].copy()
        grant_n = len(df_chunk)
        year_grant_count += grant_n

        if df_chunk.empty:
            print(f"{file_year} 年第 {chunk_no} 块：原始 {raw_n:,}，发明授权 0，跳过")
            continue

        public_year = pd.to_numeric(df_chunk["公开公告年份"], errors="coerce")
        missing_year_n = int(public_year.isna().sum())
        mismatch_n = int((public_year.notna() & public_year.ne(file_year)).sum())
        df_chunk["year"] = public_year.fillna(file_year).astype("Int64")

        df_chunk = add_first_applicant_fields(df_chunk)
        if ALLOWED_FIRST_APPLICANT_TYPES is not None:
            df_chunk = df_chunk[
                df_chunk["第一申请人类型"].isin(ALLOWED_FIRST_APPLICANT_TYPES)
            ].copy()

        missing_firm_n = int(df_chunk["第一申请人"].isna().sum())

        _, matched_chunk, _, _ = tag_biomanufacturing_patents(
            df_chunk,
            cn_abs_col="摘要文本",
            en_abs_col=None,
            firm_col="第一申请人",
            year_col="year",
            region_col=["第一申请人地区", "第一申请人城市"],
            firm_type_col="第一申请人类型",
            extra_text_cols=["专利名称", "主权项内容"],
            split_firms=False,
            coarse_screen=True,
            progress_every=10000,
            min_score=MIN_BIOMANUFACTURING_SCORE,
        )

        matched_chunk["来源文件年份"] = file_year
        matched_chunk["来源分块"] = chunk_no
        all_parts.append(matched_chunk)
        year_matched_count += len(matched_chunk)

        print(
            f"{file_year} 年第 {chunk_no} 块完成：原始 {raw_n:,}，"
            f"发明授权 {grant_n:,}，生物制造相关 {len(matched_chunk):,}；"
            f"年份缺失 {missing_year_n:,}，年份与文件名不一致 {mismatch_n:,}，"
            f"第一申请人缺失 {missing_firm_n:,}"
        )

    print(
        f"{file_year} 年完成：原始 {year_raw_count:,}，"
        f"发明授权 {year_grant_count:,}，生物制造相关 {year_matched_count:,}"
    )


# =========================
# 4. 合并、去重和企业汇总
# =========================

if all_parts:
    patents = pd.concat(all_parts, ignore_index=True)
    dedup_cols = [
        "专利名称",
        "申请人",
        "公开公告年份",
        "IPC主分类号",
        "摘要文本",
        "主权项内容",
    ]
    dedup_cols = [column for column in dedup_cols if column in patents.columns]
    before_dedup = len(patents)
    patents = patents.drop_duplicates(subset=dedup_cols).copy()
    print(f"\n生物制造专利合并后去重：{before_dedup:,} -> {len(patents):,}")

    firm_year, firm = summarize_biomanufacturing_firms(
        patents,
        firm_col="第一申请人",
        year_col="year",
        region_col=["第一申请人地区", "第一申请人城市"],
        firm_type_col="第一申请人类型",
    )
else:
    patents = pd.DataFrame()
    firm_year = pd.DataFrame()
    firm = pd.DataFrame()


# =========================
# 5. 导出结果
# =========================

patent_path = OUTPUT_ROOT / "biomanufacturing_patents_2021_2025_newsource.csv"
firm_year_path = OUTPUT_ROOT / "firm_year_biomanufacturing_2021_2025_newsource.csv"
firm_path = OUTPUT_ROOT / "firm_biomanufacturing_2021_2025_newsource.csv"
dictionary_path = OUTPUT_ROOT / "biomanufacturing_keyword_dictionary.csv"

patents.to_csv(patent_path, index=False, encoding="utf-8-sig")
firm_year.to_csv(firm_year_path, index=False, encoding="utf-8-sig")
firm.to_csv(firm_path, index=False, encoding="utf-8-sig")
export_keyword_dictionary().to_csv(dictionary_path, index=False, encoding="utf-8-sig")

elapsed = time.time() - run_start
unique_firms = (
    patents["第一申请人"].dropna().nunique()
    if "第一申请人" in patents.columns
    else 0
)

print(f"\n识别完成，总耗时：{elapsed / 60:.2f} 分钟")
print(f"生物制造相关专利数量：{len(patents):,}")
print(f"第一申请人数量：{unique_firms:,}")
print("结果已导出：")
print(patent_path)
print(firm_year_path)
print(firm_path)
print(dictionary_path)
