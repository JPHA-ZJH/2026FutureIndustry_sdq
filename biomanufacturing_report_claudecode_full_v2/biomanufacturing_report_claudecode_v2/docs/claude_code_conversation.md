# Claude Code完整对话流程

## 第一轮：运行、清洗和审计

发送：`prompts/01_run_clean_audit.md`

目标：确保代码成功运行，检查城市统一、第一申请人唯一类型、医疗机构单列、技术路线映射和三表一致性。此轮不写报告。

## 第二轮：检索技术路线与政策资料

发送：`prompts/02_research_technical_routes.md`

目标：为第一节七条技术路线形成200—500字说明，补充国家和上海政策资料，并填写 `report/source_log.csv` 和 `report/technical_route_notes.md`。

## 第三轮：严格按章节撰写初稿

发送：`prompts/03_write_strict_draft.md`

目标：读取全部分析表和图，严格按照 `docs/report_outline.md` 写作。第五节（四）必须插入图14、图15。

## 第四轮：事实和数字核验

发送：`prompts/04_fact_check.md`

目标：逐项核验数字、分母、排名、图表引用和外部来源，输出核验稿及问题清单。

## 第五轮：咨询报告式润色

发送：`prompts/05_polish.md`

目标：在不改变事实和章节结构的前提下，压缩重复内容、增强结论表达和图表解读，形成终稿Markdown。
