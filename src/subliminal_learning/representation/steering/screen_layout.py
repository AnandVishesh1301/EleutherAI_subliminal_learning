"""On-disk layout of the teacher-steerability screen (per-seed files + derived aggregate).

Namespaced so cross-seed never collides with within-model, and so every (source, target)
pair owns a whole directory (one file per ftseed inside). Within-model (``steer_spec is
None``) lives under ``steerability/self/``; a cross-seed run into ``steer_spec`` lives under
``steerability/steer_{label}/``. Shared by the screen pipeline and its sweep.
"""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.evaluation.ftseed_dirs import discover_ftseed_dirs
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.models.specs import spec_label


def screen_dir(param_dir: Path, steer_spec: PythiaModelSpec | None) -> Path:
    """Base dir holding this (source, target) cell's per-seed screens + derived aggregate."""
    root = param_dir / "steerability"
    if steer_spec is None:
        return root / "self"
    return root / f"steer_{spec_label(steer_spec)}"


def seed_file(param_dir: Path, steer_spec: PythiaModelSpec | None, seed: int) -> Path:
    """Standalone single-seed screen for one ftseed (``ft{N}.json``)."""
    return screen_dir(param_dir, steer_spec) / f"ft{seed}.json"


def aggregate_file(param_dir: Path, steer_spec: PythiaModelSpec | None) -> Path:
    """Derived seed-average over all ``ft*.json`` in the cell (``aggregate.json``)."""
    return screen_dir(param_dir, steer_spec) / "aggregate.json"


def discover_screen_ftseed_dirs(param_dir: Path) -> list[tuple[int, Path]]:
    """Sorted (ft_seed, checkpoints_dir) for each ``ftseed{N}`` under `param_dir` (no weight gates)."""
    return discover_ftseed_dirs(param_dir, check_weights=False)
