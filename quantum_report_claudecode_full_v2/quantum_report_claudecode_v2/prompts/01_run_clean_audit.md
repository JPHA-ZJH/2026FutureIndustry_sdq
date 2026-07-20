# 任务一：运行、清洗和审计

请先阅读`CLAUDE.md`、`README.md`和`docs/analysis_scope.md`，暂时不要撰写报告。

1. 检查`data/raw`中的三份CSV是否存在，并读取字段、行数和年份范围。
2. 运行：

```bash
python run_all.py
```

3. 如果报错，依据报错做最小幅度修改并重新运行，直到完整成功。
4. 不得为了运行成功而删除城市统一、主体类型重新判断、全国技术趋势或省际比较模块。
5. 阅读以下审计结果：
   - `outputs/audit/data_cleaning_audit.md`
   - `outputs/audit/city_normalization_audit.csv`
   - `outputs/audit/unmapped_domestic_cities.csv`
   - `outputs/audit/applicant_type_audit.csv`
   - `outputs/audit/applicant_type_review_list.csv`
6. 重点检查：
   - “上海市/上海”“北京市/北京”等是否已经统一；
   - 全国城市映射是否可扩展，未知城市是否进入审计清单；
   - 混合类型的合作专利是否按第一申请人名称重新判为唯一类型；
   - 中国科学院研究所是否被识别为科研院所，大学是否识别为高校，有限公司是否识别为企业；
   - 三张清洗结果表专利总数是否一致；
   - 图形中文是否正常显示。
7. 随机抽查至少30条混合类型记录，检查第一申请人类型判断是否合理。
8. 汇报：
   - 是否成功运行；
   - 修改了哪些文件；
   - 城市和主体类型清洗结果；
   - 仍需人工复核的记录；
   - 生成了哪些表格和图形；
   - 哪些数据局限会影响写作。

完成后停止，不要写咨询报告。
