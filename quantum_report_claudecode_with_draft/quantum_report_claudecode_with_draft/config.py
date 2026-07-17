from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
AUDIT_DIR = OUTPUT_DIR / "audit"
DRAFT_DIR = OUTPUT_DIR / "draft"

PATENT_FILE = DATA_DIR / "quantum_patents_2021_2025_newsource.csv"
FIRM_YEAR_FILE = DATA_DIR / "firm_year_quantum_2021_2025_newsource.csv"
FIRM_FILE = DATA_DIR / "firm_quantum_2021_2025_newsource.csv"
CITY_PROVINCE_FILE = DATA_DIR / "city_province_mapping.csv"

SHANGHAI = "上海市"
COMPARE_REGIONS = ["上海市", "北京市", "安徽省", "江苏省", "浙江省", "广东省"]

STRICT_CORE_LABELS = ["高相关", "中相关"]
CORE_LABELS = ["高相关", "中相关", "低相关/待复核"]
BROAD_LABELS = CORE_LABELS + ["量子计量/PNT候选"]

CITATION_VALID_COVERAGE = 0.70
TOP_ENTITY_N = 15
TOP_OTHER_ENTERPRISE_N = 5
