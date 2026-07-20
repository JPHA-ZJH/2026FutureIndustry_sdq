from scripts.prepare_data import main as prepare_main
from scripts.analyze_report import main as analyze_main

if __name__ == "__main__":
    prepare_main()
    analyze_main()
    print("\n全部流程运行完成。请先查看 outputs/audit/data_cleaning_audit.md，再查看 outputs/analysis_summary.md。")
