"""Lets scripts run as `python scripts/xyz.py` without installing the package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):  # Windows consoles: print Bangla without crashing
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
