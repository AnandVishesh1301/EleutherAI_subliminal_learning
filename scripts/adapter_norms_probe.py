"""Moved to forgetting/scripts/; kept because queued jobs name this path.

Runs ``forgetting.scripts.adapter_norms_probe`` as ``__main__`` (``uv run python
scripts/adapter_norms_probe.py ...`` keeps working).
"""

import runpy
import sys
from importlib.util import find_spec
from pathlib import Path

if __name__ == "__main__":
    if find_spec("forgetting") is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    runpy.run_module("forgetting.scripts.adapter_norms_probe", run_name="__main__", alter_sys=True)
