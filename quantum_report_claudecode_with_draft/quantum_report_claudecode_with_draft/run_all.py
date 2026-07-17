import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
script = root / "scripts" / "analyze_quantum_report.py"
subprocess.run([sys.executable, str(script)], check=True)
