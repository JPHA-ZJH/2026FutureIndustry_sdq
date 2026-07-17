# 上海量子信息咨询报告：Claude Code 项目包

本项目用于基于三份 CSV 数据完成数据审计、统计分析、图表生成和咨询报告初稿。

## 一、正确流程

顺序必须是：

1. 放入三份 CSV；
2. 运行分析代码；
3. 阅读数据审计结果；
4. 检查生成的表格和图形；
5. 仅依据分析输出写报告；
6. 对全国技术现状和企业公开情况补充权威网络来源；
7. 进行事实核验和措辞收敛。

不能先写报告再“找数据支持”。

## 二、数据文件

将以下文件放入 `data/`，文件名保持不变：

- `quantum_patents_2021_2025_newsource.csv`
- `firm_year_quantum_2021_2025_newsource.csv`
- `firm_quantum_2021_2025_newsource.csv`

当前包中是仅覆盖2021年的测试数据。未来替换为多年正式数据后，代码会自动识别年份，并启用趋势、新进入和企业持续性分析。

## 三、运行

在项目根目录打开终端：

```bash
python -m pip install -r requirements.txt
python run_all.py
```

结果输出到：

- `outputs/audit/`：数据审计；
- `outputs/tables/`：报告表格底表；
- `outputs/figures/`：PNG图形；
- `outputs/analysis_summary.md`：供报告写作使用的结构化摘要。

## 四、在 Claude Code 中使用

依次复制并发送：

1. `prompts/01_run_and_audit.md`
2. `prompts/02_write_draft.md`
3. `prompts/03_fact_check.md`
4. `prompts/04_polish.md`

完整对话顺序见 `docs/claude_code_conversation.md`。

## 五、关键口径

- 专利明细表是技术结构、产业链环节和代表性专利分析的主数据。
- 企业—年份表用于多年趋势、新进入和持续创新分析。
- 企业汇总表用于主体排名和三表一致性核验。
- `firm_relevance`只是量子相关等级，不能据此判断主体是不是企业。
- 企业分析采用“明确企业”保守口径；混合申请人类型必须复核。
- 被引字段覆盖率不足70%时，不将被引作为主指标。
- 专利不能直接代表产值、营收、融资、市场份额和产品落地。
