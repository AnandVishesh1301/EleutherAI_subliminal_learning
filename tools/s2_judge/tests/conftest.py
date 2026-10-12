"""Put `tools/` on sys.path so `s2_judge` imports as it does on the lane."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = str(Path(__file__).resolve().parents[2])
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)
