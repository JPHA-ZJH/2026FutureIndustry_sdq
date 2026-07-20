# 上海量子信息产业咨询报告：完整 Claude Code 工作包

本项目以三份量子专利识别结果为基础，完成城市和主体类型清洗、专利与企业分析、图表生成、报告写作和事实核验。

## 1. 直接运行

在 VS Code 中打开本文件夹，在终端执行：

```bash
python -m pip install -r requirements.txt
python run_all.py
```

运行顺序：

```text
原始三份CSV
→ 重新提取第一申请人
→ 统一城市并映射省份
→ 重新判断唯一主体类型
→ 重建企业—年份表和主体汇总表
→ 生成分析表格和图形
→ 生成写作摘要
```

## 2. 更新数据

将新数据替换到 `data/raw/`，保持三个文件名不变：

- `quantum_patents_2021_2025_newsource.csv`
- `firm_year_quantum_2021_2025_newsource.csv`
- `firm_quantum_2021_2025_newsource.csv`

然后重新运行 `python run_all.py`。原有同名结果会被覆盖。

## 3. 首先检查的文件

- `outputs/audit/data_cleaning_audit.md`
- `outputs/audit/city_normalization_audit.csv`
- `outputs/audit/applicant_type_audit.csv`
- `outputs/audit/applicant_type_review_list.csv`
- `outputs/audit/unmapped_domestic_cities.csv`

城市或主体类型存在异常时，应先修改规则并重新运行，不要直接写报告。

## 4. 结果目录

- `data/clean/`：清洗后的专利、企业—年份和主体汇总数据。
- `outputs/tables/`：22张报告分析底表。
- `outputs/figures/`：13张报告图形。
- `outputs/analysis_summary.md`：供写作使用的数据摘要。
- `outputs/draft/`：当前数据生成的测试版初稿。
- `report/`：Claude Code正式生成的初稿、核验稿和来源记录。

## 5. Claude Code 对话顺序

按顺序使用：

1. `prompts/01_run_clean_audit.md`
2. `prompts/02_research_technical_routes.md`
3. `prompts/03_write_strict_draft.md`
4. `prompts/04_fact_check.md`
5. `prompts/05_polish.md`

完整对话范本见 `docs/claude_code_conversation.md`。

## 6. 本版本的关键调整

- 城市名称全国统一处理，不再只维护当前样本中出现的城市。
- 第一申请人类型依据第一申请人名称重新判断，合作申请中的混合类型不再直接使用。
- 第一节改为“五类技术路线介绍＋全国技术方向专利演进”。
- 后续技术分析只围绕五类技术方向，不分析相关性分级和产业链环节。
- 报告必须严格按照七节结构撰写，并尽量使用清晰、统一的图表。
