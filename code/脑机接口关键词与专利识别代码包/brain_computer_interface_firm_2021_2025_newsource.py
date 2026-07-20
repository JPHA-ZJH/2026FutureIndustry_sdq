# -*- coding: utf-8 -*-
"""从2021—2025年中国全量专利CSV识别脑机接口专利并汇总第一申请人。"""
import re
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(r"D:\BaiduNetdiskDownload\中国专利数据库1985-2025.11\分年份保存数据\分年份保存数据")
OUTPUT_ROOT = Path(r"F:\01科研\横向课题\未来产业识别\脑机接口\data")
START_YEAR, END_YEAR = 2021, 2025
CHUNK_SIZE = 100000
NROWS_PER_YEAR = None
CSV_ENCODING = None
ALLOWED_FIRST_APPLICANT_TYPES = None  # 如只保留企业，改为 {"企业"}
MIN_BRAIN_COMPUTER_INTERFACE_SCORE = 1
TARGET_COLS = ["专利名称", "专利类型", "申请人", "申请人类型", "申请人地区", "申请人城市",
               "公开公告年份", "IPC主分类号", "摘要文本", "主权项内容", "被引证次数"]

if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0, str(SCRIPT_DIR))
from brain_computer_interface_matcher_v1_optimized_region_newsource import (  # noqa: E402
    export_keyword_dictionary, summarize_brain_computer_interface_firms,
    tag_brain_computer_interface_patents)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def first_item(value):
    """按中文分号取第一项，同时兼容少量英文分号。"""
    if pd.isna(value) or not str(value).strip(): return pd.NA
    result = re.split(r"\s*[；;]\s*", str(value).strip(), maxsplit=1)[0].strip()
    return result if result else pd.NA


def add_first_fields(df):
    d = df.copy()
    for src, dst in {"申请人": "第一申请人", "申请人类型": "第一申请人类型",
                     "申请人地区": "第一申请人地区", "申请人城市": "第一申请人城市"}.items():
        d[dst] = d[src].map(first_item)
    return d


def check_columns(path):
    kwargs = {"nrows": 0}
    if CSV_ENCODING: kwargs["encoding"] = CSV_ENCODING
    cols = pd.read_csv(path, **kwargs).columns
    missing = [c for c in TARGET_COLS if c not in cols]
    if missing: raise KeyError(f"文件 {path.name} 缺少变量：{missing}")


def iter_chunks(path):
    kwargs = dict(usecols=TARGET_COLS, low_memory=False, chunksize=CHUNK_SIZE, nrows=NROWS_PER_YEAR)
    if CSV_ENCODING: kwargs["encoding"] = CSV_ENCODING
    return pd.read_csv(path, **kwargs)


parts, run_start = [], time.time()
for file_year in range(START_YEAR, END_YEAR + 1):
    path = DATA_ROOT / f"中国全量专利数据库{file_year}年.csv"
    if not path.exists(): raise FileNotFoundError(f"找不到文件：{path}")
    check_columns(path); raw_total = grant_total = matched_total = 0
    print(f"\n开始处理：{path}")
    for chunk_no, chunk in enumerate(iter_chunks(path), 1):
        raw_total += len(chunk)
        chunk = chunk[chunk["专利类型"].fillna("").astype(str).str.strip().eq("发明授权")].copy()
        grant_total += len(chunk)
        if chunk.empty: continue
        years = pd.to_numeric(chunk["公开公告年份"], errors="coerce")
        chunk["year"] = years.fillna(file_year).astype("Int64"); chunk = add_first_fields(chunk)
        if ALLOWED_FIRST_APPLICANT_TYPES is not None:
            chunk = chunk[chunk["第一申请人类型"].isin(ALLOWED_FIRST_APPLICANT_TYPES)].copy()
        _, matched, _, _ = tag_brain_computer_interface_patents(
            chunk, cn_abs_col="摘要文本", en_abs_col=None, firm_col="第一申请人", year_col="year",
            region_col=["第一申请人地区", "第一申请人城市"], firm_type_col="第一申请人类型",
            extra_text_cols=["专利名称", "主权项内容"], split_firms=False, coarse_screen=True,
            progress_every=10000, min_score=MIN_BRAIN_COMPUTER_INTERFACE_SCORE)
        matched["来源文件年份"], matched["来源分块"] = file_year, chunk_no
        parts.append(matched); matched_total += len(matched)
        print(f"{file_year}年第{chunk_no}块：发明授权 {len(chunk):,}，脑机接口 {len(matched):,}")
    print(f"{file_year}年完成：原始 {raw_total:,}，发明授权 {grant_total:,}，脑机接口 {matched_total:,}")

if parts:
    patents = pd.concat(parts, ignore_index=True)
    dedup = [c for c in ["专利名称", "申请人", "公开公告年份", "IPC主分类号", "摘要文本", "主权项内容"] if c in patents.columns]
    patents = patents.drop_duplicates(subset=dedup).copy()
    firm_year, firm = summarize_brain_computer_interface_firms(
        patents, "第一申请人", "year", ["第一申请人地区", "第一申请人城市"], "第一申请人类型")
else:
    patents = firm_year = firm = pd.DataFrame()

paths = [OUTPUT_ROOT / "brain_computer_interface_patents_2021_2025_newsource.csv",
         OUTPUT_ROOT / "firm_year_brain_computer_interface_2021_2025_newsource.csv",
         OUTPUT_ROOT / "firm_brain_computer_interface_2021_2025_newsource.csv",
         OUTPUT_ROOT / "brain_computer_interface_keyword_dictionary.csv"]
for frame, path in zip([patents, firm_year, firm, export_keyword_dictionary()], paths):
    frame.to_csv(path, index=False, encoding="utf-8-sig")
print(f"\n完成，用时 {(time.time()-run_start)/60:.2f} 分钟；识别专利 {len(patents):,} 件。")
for path in paths: print(path)
