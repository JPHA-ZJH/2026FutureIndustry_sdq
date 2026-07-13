
# Project Instructions

This project analyzes Hydrogen Fuel News articles.

Main goal:
Identify important hydrogen technology keywords and their evolution across time periods.

Data:

- Raw text files are stored locally in data/raw/article_texts/
- Do not upload or commit raw text files.
- Processed corpus should be saved to data/processed/
- Results should be saved to output/keyword_results/

Coding style:

- Use Python.
- Prefer pandas, scikit-learn, regex, matplotlib.
- Keep scripts simple and readable.
- Add Chinese comments where useful.
- All scripts should be runnable directly.
- Do not modify the crawler unless explicitly asked.

Expected pipeline:

1. Build article-level corpus from txt files.
2. Clean and normalize text.
3. Extract hydrogen technology keywords by time period.
4. Build keyword trend tables.
5. Build keyword co-occurrence networks.

Research focus:

- hydrogen production
- electrolysis
- fuel cells
- hydrogen storage
- hydrogen carriers
- hydrogen transport
- hydrogen infrastructure
- industrial application
- aviation
- shipping
- trucks
- power generation

