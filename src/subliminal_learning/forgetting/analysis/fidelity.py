"""Batched readers for the per-run task-fidelity records.

The three fidelity evals (`scripts.fidelity_heldout_ce` and friends) each write one small
JSON per run dir; the claims notebook fans out over ~1.5k strip runs and wants all of them.
These are thin, named wrappers over `analysis.record_cache`'s fidelity kinds -- callers pass
run-dir paths relative to the outputs root and get the kind's frame back, so no notebook or
script has to open-code glob + cache plumbing.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from subliminal_learning.analysis.record_cache import load_record_cache


def load_fidelity_heldout(root: Path, rel_paths, refresh: bool = True) -> pd.DataFrame:
    """Endpoint train/held-out CE records (``{run}/eval/fidelity_heldout.json``).

    Returns one row per run that has a record: ``run_path`` plus ``ce_heldout``, ``ce_base``,
    ``ce_train``, ``ce_train_base`` (NaN when the eval ran but the adapter diverged) and the
    ``nonfinite`` flag that marks those diverged adapters.
    """
    return load_record_cache(root, "fidelity_heldout", rel_paths, refresh=refresh, verbose=False)


def load_fidelity_traj_stats(root: Path, rel_paths, refresh: bool = True) -> pd.DataFrame:
    """Per-run ladder statistics derived from ``{run}/eval/fidelity_traj.json``.

    Returns one row per laddered run: ``run_path`` plus ``max_rise`` (largest climb of train
    CE above its own running minimum), ``gap_max`` (largest held-out minus train gap),
    ``train_min`` and ``n_points`` (finite checkpoints -- the caller applies its own minimum).
    """
    return load_record_cache(root, "fidelity_traj", rel_paths, refresh=refresh, verbose=False)


def load_fidelity_shuffle(root: Path, rel_paths, refresh: bool = True) -> pd.DataFrame:
    """Shuffled-pairing held-out CE (``{run}/eval/fidelity_shuffle.json``).

    Returns one row per run with a record: ``run_path`` and ``ce_shuffled`` (NaN when null).
    """
    return load_record_cache(root, "fidelity_shuffle", rel_paths, refresh=refresh, verbose=False)
