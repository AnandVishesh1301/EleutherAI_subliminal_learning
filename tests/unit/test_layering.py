"""Layering guard: SDF/GKD domain modules must stay torch-/IO-free (ARCHITECTURE §7).

The pure-domain modules (`config`, `layout`, `filtering`) define repo IDs, path tags,
and screening logic that unit tests exercise on CPU with no heavy deps. If one of them
grows a ``torch`` / ``transformers`` / ``trl`` / ``peft`` / ``datasets`` import (directly
or transitively), the domain layer is no longer cheap to import or test and the
inversion in ARCHITECTURE §7 is broken. Each module is imported in a *fresh* interpreter
so a forbidden transitive import shows up in that process's ``sys.modules``.

`gkd/losses.py` is intentionally excluded: it is domain math that is inherently ``torch``
(the GKD per-token loss operates on logits), so it is tested for numerics, not purity.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

_DOMAIN_MODULES = (
    "subliminal_learning.sdf.config",
    "subliminal_learning.sdf.layout",
    "subliminal_learning.gkd.config",
    "subliminal_learning.gkd.layout",
    "subliminal_learning.gkd.filtering",
    "subliminal_learning.campaign.plan",
    "subliminal_learning.campaign.pairs",
    "subliminal_learning.campaign.selection",
    "subliminal_learning.campaign.config_gen",
    "subliminal_learning.campaign.expected",
)
_FORBIDDEN = ("torch", "transformers", "trl", "peft", "datasets")


@pytest.mark.parametrize("module", _DOMAIN_MODULES)
def test_domain_module_imports_no_heavy_deps(module: str) -> None:
    code = (
        "import importlib, sys;"
        f"importlib.import_module({module!r});"
        f"print(','.join(m for m in {_FORBIDDEN!r} if m in sys.modules))"
    )
    leaked = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert leaked == "", f"{module} transitively imports heavy deps: {leaked}"
