"""Layering guards (docs/PROJECT_DESIGN.md): what may import what, checked in fresh interpreters.

1. The pure-domain modules (`*/config.py`, `*/layout.py`, `gkd/filtering.py`, `gkd/pairs.py`)
   stay torch-/IO-free: a ``torch`` / ``transformers`` / ``trl`` / ``peft`` / ``datasets``
   import (direct or transitive) would make the domain layer expensive to import and test.
   `gkd/losses.py` is intentionally excluded: it is domain math that is inherently ``torch``.
2. ``subliminal_learning.analysis`` is the notebook-support layer: nothing outside
   ``analysis/``, ``figures/`` and ``pipelines/`` imports it (domain and ``*_io`` modules get
   their shared loaders from ``utils/`` and their estimators from ``stats/``).
3. ``stats/`` is the computation layer: it imports neither ``analysis`` nor matplotlib.
4. ``import subliminal_learning`` is light: no torch, transformers, jobq or matplotlib.

Each check runs in a *fresh* interpreter so a forbidden transitive import shows up in that
process's ``sys.modules``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
from tests._helpers import REPO

SRC = REPO / "src" / "subliminal_learning"

_DOMAIN_MODULES = (
    "subliminal_learning.sdf.config",
    "subliminal_learning.sdf.layout",
    "subliminal_learning.gkd.config",
    "subliminal_learning.gkd.layout",
    "subliminal_learning.gkd.filtering",
    "subliminal_learning.gkd.pairs",
    "subliminal_learning.gkd.sweep_helpers",
)
_FORBIDDEN = ("torch", "transformers", "trl", "peft", "datasets")


def _sys_modules_after(module: str) -> set[str]:
    """Names in a fresh interpreter's ``sys.modules`` after importing `module`."""
    code = (
        "import importlib, sys;"
        f"importlib.import_module({module!r});"
        "print('\\n'.join(sys.modules))"
    )
    return set(subprocess.check_output([sys.executable, "-c", code], text=True).split())


def _leaked(mods: set[str], prefixes: tuple[str, ...]) -> list[str]:
    return sorted(m for m in mods if m.split(".")[0] in prefixes)


@pytest.mark.parametrize("module", _DOMAIN_MODULES)
def test_domain_module_imports_no_heavy_deps(module: str) -> None:
    leaked = _leaked(_sys_modules_after(module), _FORBIDDEN)
    assert leaked == [], f"{module} transitively imports heavy deps: {leaked}"


# --- rule 2: the analysis layer is imported only by analysis / figures / pipelines ----------
_ANALYSIS_IMPORTERS_ALLOWED = ("analysis", "figures", "pipelines")


def _top_level_imports(path: Path) -> list[str]:
    """Dotted targets of the module-level import statements in `path` (lazy ones excluded)."""
    tree = ast.parse(path.read_text())
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
        elif isinstance(node, ast.Import):
            out.extend(a.name for a in node.names)
    return out


def test_only_analysis_figures_pipelines_import_analysis() -> None:
    offenders = []
    for f in sorted(SRC.rglob("*.py")):
        rel = f.relative_to(SRC)
        if rel.parts[0] in _ANALYSIS_IMPORTERS_ALLOWED:
            continue
        bad = [m for m in _top_level_imports(f)
               if m == "subliminal_learning.analysis" or m.startswith("subliminal_learning.analysis.")]
        if bad:
            offenders.append((str(rel), bad))
    assert offenders == [], f"domain/io modules importing the analysis layer: {offenders}"


# --- rule 3: the computation layer imports neither analysis nor matplotlib ------------------
_STATS_MODULES = tuple(
    f"subliminal_learning.stats.{p.stem}" for p in sorted((SRC / "stats").glob("*.py"))
    if p.stem != "__init__"
) + ("subliminal_learning.stats",)


@pytest.mark.parametrize("module", _STATS_MODULES)
def test_computation_layer_imports_no_analysis_or_matplotlib(module: str) -> None:
    mods = _sys_modules_after(module)
    assert not any(m.startswith("subliminal_learning.analysis") for m in mods), module
    assert _leaked(mods, ("matplotlib",)) == [], module


# --- rule 4: importing the package is light ------------------------------------------------
def test_package_import_is_light() -> None:
    leaked = _leaked(_sys_modules_after("subliminal_learning"),
                     ("torch", "transformers", "jobq", "matplotlib"))
    assert leaked == [], f"`import subliminal_learning` pulls in: {leaked}"
