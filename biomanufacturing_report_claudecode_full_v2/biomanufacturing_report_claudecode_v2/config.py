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

PATENT_FILE = RAW_DIR / "biomanufacturing_patents_2021_2025_newsource.csv"
FIRM_YEAR_FILE = RAW_DIR / "firm_year_biomanufacturing_2021_2025_newsource.csv"
FIRM_FILE = RAW_DIR / "firm_biomanufacturing_2021_2025_newsource.csv"
KEYWORD_FILE = RAW_DIR / "biomanufacturing_keyword_dictionary.csv"
CITY_FILE = REFERENCE_DIR / "china_cities.csv"
PROVINCE_FILE = REFERENCE_DIR / "china_provinces.csv"

CLEAN_PATENT_FILE = CLEAN_DIR / "biomanufacturing_patents_clean.csv"
CLEAN_FIRM_YEAR_FILE = CLEAN_DIR / "firm_year_clean.csv"
CLEAN_FIRM_FILE = CLEAN_DIR / "firm_clean.csv"
CLEAN_KEYWORD_FILE = CLEAN_DIR / "keyword_dictionary_clean.csv"

SHANGHAI = "上海"
COMPARE_REGIONS = ["上海", "北京", "安徽", "江苏", "浙江", "广东"]
FOCUS_SUBJECT_TYPES = ["企业", "高校", "科研院所"]

# 原始识别结果的技术方向。分析底表保留原始分类，咨询报告正文使用下方七类“报告技术路线”。
RAW_TECH_CATEGORIES = [
    "生物制造基础", "DNA/RNA读写编辑", "生物元件与线路", "计算设计与AI", "测试筛选与自动化",
    "酶与蛋白质工程", "代谢工程与细胞工厂", "底盘细胞与菌株工程", "生物过程与规模化",
    "原料与低碳路线", "产品与应用", "前沿生物制造", "安全质量与支撑",
]

REPORT_TECH_ROUTES = [
    "生物设计、读写与自动化工具",
    "酶与蛋白质工程",
    "细胞工厂与菌株工程",
    "生物过程与规模化",
    "原料与低碳路线",
    "生物制造产品与应用",
    "前沿制造与基础支撑",
]

TECH_ROUTE_MAP = {
    "DNA/RNA读写编辑": "生物设计、读写与自动化工具",
    "生物元件与线路": "生物设计、读写与自动化工具",
    "计算设计与AI": "生物设计、读写与自动化工具",
    "测试筛选与自动化": "生物设计、读写与自动化工具",
    "酶与蛋白质工程": "酶与蛋白质工程",
    "代谢工程与细胞工厂": "细胞工厂与菌株工程",
    "底盘细胞与菌株工程": "细胞工厂与菌株工程",
    "生物过程与规模化": "生物过程与规模化",
    "原料与低碳路线": "原料与低碳路线",
    "产品与应用": "生物制造产品与应用",
    "前沿生物制造": "前沿制造与基础支撑",
    "生物制造基础": "前沿制造与基础支撑",
    "安全质量与支撑": "前沿制造与基础支撑",
}

TOP_SHANGHAI_ENTERPRISES = 20
TOP_SHANGHAI_RESEARCH = 20
SHANGHAI_CASE_COUNT = 3
OTHER_CASE_COUNT = 5
MAX_OTHER_CASES_PER_PROVINCE = 2

DROP_EXACT_DUPLICATES = True
LATEST_YEAR_COMPLETENESS_WARNING = True
