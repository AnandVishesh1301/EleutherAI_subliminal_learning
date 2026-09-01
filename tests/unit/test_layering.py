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


# The computation layer: `analysis` may import these, never the reverse. `weightspace` is torch
# math by design, so only the no-analysis rule applies to it (it is not in _DOMAIN_MODULES).
_STATS_MODULES = (
    "subliminal_learning.stats",
    "subliminal_learning.stats.core",
    "subliminal_learning.stats.collapse_law",
    "subliminal_learning.stats.mediation",
)
_WEIGHTSPACE_MODULES = (
    "subliminal_learning.weightspace",
    "subliminal_learning.weightspace.adapters",
    "subliminal_learning.weightspace.deltas",
    "subliminal_learning.weightspace.lora",
    "subliminal_learning.weightspace.patching",
)


def _sys_modules_after(module: str, extra: str = "") -> set[str]:
    """Names in a fresh interpreter's ``sys.modules`` after importing `module` (+ running `extra`)."""
    code = (
        "import importlib, sys;"
        f"m = importlib.import_module({module!r});"
        f"{extra}"
        "print('\\n'.join(sys.modules))"
    )
    return set(subprocess.check_output([sys.executable, "-c", code], text=True).split())


@pytest.mark.parametrize("module", _STATS_MODULES + _WEIGHTSPACE_MODULES)
def test_computation_layer_imports_no_analysis(module: str) -> None:
    assert "subliminal_learning.analysis" not in _sys_modules_after(module)


def test_k_from_tag_does_not_pull_analysis() -> None:
    # The real regression guard: the old boundary violation was a function-local import, so it
    # only fired when `k_from_tag` was *called*, not when the module was imported.
    loaded = _sys_modules_after(
        "subliminal_learning.stats.collapse_law",
        "assert m.k_from_tag('ep1_lr1e-4_r8a8') > 0;",
    )
    assert "subliminal_learning.analysis" not in loaded


@pytest.mark.parametrize("module", _STATS_MODULES)
def test_stats_module_imports_no_matplotlib(module: str) -> None:
    assert "matplotlib" not in _sys_modules_after(module)
