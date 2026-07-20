from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
CLEAN_DIR = DATA_DIR / "clean"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
AUDIT_DIR = OUTPUT_DIR / "audit"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
DRAFT_DIR = OUTPUT_DIR / "draft"
REPORT_DIR = PROJECT_ROOT / "report"

PATENT_FILE = RAW_DIR / "quantum_patents_2021_2025_newsource.csv"
FIRM_YEAR_FILE = RAW_DIR / "firm_year_quantum_2021_2025_newsource.csv"
FIRM_FILE = RAW_DIR / "firm_quantum_2021_2025_newsource.csv"
CITY_FILE = REFERENCE_DIR / "china_cities.csv"
PROVINCE_FILE = REFERENCE_DIR / "china_provinces.csv"

CLEAN_PATENT_FILE = CLEAN_DIR / "quantum_patents_clean.csv"
CLEAN_FIRM_YEAR_FILE = CLEAN_DIR / "firm_year_clean.csv"
CLEAN_FIRM_FILE = CLEAN_DIR / "firm_clean.csv"

SHANGHAI = "上海"
COMPARE_REGIONS = ["上海", "北京", "安徽", "江苏", "浙江", "广东"]
TECH_CATEGORIES = ["量子计算", "量子通信与安全", "量子传感", "量子计算硬件", "量子基础概念"]

TOP_SHANGHAI_ENTERPRISES = 20
TOP_SHANGHAI_RESEARCH = 20
SHANGHAI_CASE_COUNT = 3
OTHER_CASE_COUNT = 5
MAX_OTHER_CASES_PER_PROVINCE = 2

# 只删除所有字段完全相同的重复行；不按标题等“近似键”自动删除。
DROP_EXACT_DUPLICATES = True

# 最新年份授权数据可能不完整，分析代码只提示，不自动删除。
LATEST_YEAR_COMPLETENESS_WARNING = True
