import sys
from pathlib import Path

WEB_ROOT = Path(__file__).resolve().parents[2] / "web"
if str(WEB_ROOT) not in sys.path:
    sys.path.insert(0, str(WEB_ROOT))
