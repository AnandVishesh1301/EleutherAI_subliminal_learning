"""Layering guard: SDF/GKD domain modules must stay torch-/IO-free (ARCHITECTURE §7).

The pure-domain modules (`config`, `layout`, `filtering`, `analysis.stats`) define repo
IDs, path tags, screening logic and claim statistics that unit tests exercise on CPU with
no heavy deps. If one of them grows a ``torch`` / ``transformers`` / ``trl`` / ``peft`` /
``datasets`` import (directly or transitively), the domain layer is no longer cheap to
import or test and the inversion in ARCHITECTURE §7 is broken. Each module is imported in
a *fresh* interpreter so a forbidden transitive import shows up in that process's
``sys.modules``.

`_TORCH_DOMAIN_MODULES` is the explicit exemption list: domain math that is inherently
``torch`` (per-token losses, Gram matrices, subspace bases). Those modules are tested for
numerics, not purity — but they are listed rather than merely omitted, so a module that
grows an accidental torch import is a deliberate decision instead of a silent one.
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
    "subliminal_learning.analysis.stats",
    "subliminal_learning.cka.pairing",
    "subliminal_learning.campaign.ckpt_plan",
    "subliminal_learning.campaign.ckpt_pairs",
    "subliminal_learning.campaign.ckpt_replication",
    "subliminal_learning.campaign.ckpt_selection",
    "subliminal_learning.campaign.ckpt_config_gen",
    "subliminal_learning.campaign.ckpt_jobs",
)
_FORBIDDEN = ("torch", "transformers", "trl", "peft", "datasets")

_TORCH_DOMAIN_MODULES = (
    "subliminal_learning.gkd.losses",
    "subliminal_learning.cka.minibatch",
    "subliminal_learning.cka.metrics",
)

# Pure geometry modules in `cka/`: torch is allowed, disk/GPU orchestration is not.
_PURE_CKA_MODULES = (
    "subliminal_learning.cka.metrics",
    "subliminal_learning.cka.pairing",
    "subliminal_learning.cka.probes",
    "subliminal_learning.cka.weight_space",
)


def _imported_modules(module: str, candidates: tuple[str, ...]) -> list[str]:
    """Names from `candidates` present in a fresh interpreter after importing `module`."""
    code = (
        "import importlib, sys;"
        f"importlib.import_module({module!r});"
        f"print(','.join(m for m in {candidates!r} if m in sys.modules))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    return [m for m in out.split(",") if m]


@pytest.mark.parametrize("module", _DOMAIN_MODULES)
def test_domain_module_imports_no_heavy_deps(module: str) -> None:
    leaked = _imported_modules(module, _FORBIDDEN)
    assert leaked == [], f"{module} transitively imports heavy deps: {leaked}"


@pytest.mark.parametrize("module", _TORCH_DOMAIN_MODULES)
def test_torch_exempt_module_is_not_in_the_domain_set(module: str) -> None:
    """The exemption is real (the module does need torch) and is not double-listed."""
    assert module not in _DOMAIN_MODULES
    assert "torch" in _imported_modules(module, _FORBIDDEN)


@pytest.mark.parametrize("module", _PURE_CKA_MODULES)
def test_pure_cka_module_imports_no_cka_io(module: str) -> None:
    io_modules = (
        "subliminal_learning.cka.extract_io",
        "subliminal_learning.cka.pool_io",
        # Not `*_io`-named, but it reads manifests and loads/merges checkpoints.
        "subliminal_learning.cka.model_refs",
    )
    leaked = _imported_modules(module, io_modules)
    assert leaked == [], f"{module} pulls CKA IO modules: {leaked}"
