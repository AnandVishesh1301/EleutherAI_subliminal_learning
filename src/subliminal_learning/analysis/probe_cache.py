"""Cached aggregate of per-run pile-probe records (`{run}/eval/pile_probe.json`).

Back-compat facade over `analysis.record_cache`'s ``pile_probe`` kind, which owns the refresh,
union-persistence and atomic-write contract (read its docstring). This module keeps the
original cache location (``analysis/probe_cache.csv``, so the existing on-disk cache is reused
as-is) and the original entry point, and adds the probe-pool identity filter.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from subliminal_learning.analysis.record_cache import PILE_PROBE, load_record_cache

CACHE_REL = PILE_PROBE.cache_rel     # analysis/probe_cache.csv

_COLS = ["run_path", "kl", "delta_ce", "nonfinite", "pool_version",
         "student_param_tag", "filtered_sha256", "n_train_steps", "record_mtime"]


def load_probe_cache(output_root: Path, run_paths, refresh: bool = True,
                     pool_version: str | None = "pp1") -> pd.DataFrame:
    """The probe cache for `run_paths` (relative to `output_root`), refreshed in place.

    Args:
        output_root: The outputs root the relative run paths resolve against.
        run_paths: Iterable of run-dir paths relative to `output_root` (the inventory to keep
            covered -- e.g. the bandext126 run list plus the strip runs). Paths whose record
            does not exist yet are simply absent from the frame and picked up on a later
            refresh.
        refresh: When False, return the stored cache as-is (no filesystem sweep).
        pool_version: Probe-pool identity to require. Rows recorded against a *different*
            pool are dropped from the returned frame (they stay cached, and the count is
            printed) so numbers from two pools can never be silently pooled. Rows with no
            recorded ``pool_version`` -- legacy records -- pass. None disables the filter.

    Returns:
        One row per cached record, filtered to `run_paths`: the historical `_COLS` schema plus
        the kind's newer columns (``pool_sha``, ``schema_version``, ``record_size``).
        ``run_path`` is unique and ``kl`` is float with ``+inf`` for diverged adapters.
    """
    out = load_record_cache(output_root, PILE_PROBE, run_paths, refresh=refresh)
    if pool_version is not None and len(out):
        bad = out.pool_version.notna() & (out.pool_version != pool_version)
        if bad.any():
            print(f"probe cache: {int(bad.sum())} rows dropped (pool_version != {pool_version})")
            out = out[~bad].reset_index(drop=True)
    return out


def load_teacher_probe_cache(output_root: Path, tree: Path | str, refresh: bool = True,
                             pool_version: str | None = "pp1") -> pd.DataFrame:
    """`load_probe_cache` for a TEACHER tree (``teachers/{animal}/{base}``), not a student list.

    Student inventories are explicit run lists (campaign manifests); teacher trees have no
    manifest, so the inventory is discovered by globbing ``{tree}/*/ftseed*`` under
    `output_root` (``glob``, not ``os.walk`` -- walk under-reports in big checkpoint dirs) and
    the rows gain ``param_tag`` / ``ftseed`` parsed from ``run_path``. Same ``pile_probe``
    kind and cache CSV as the student path, so the two share one refresh/union contract.
    """
    output_root = Path(output_root)
    tree = Path(tree)
    tree_abs = tree if tree.is_absolute() else output_root / tree
    rel = [str(d.relative_to(output_root)) for d in sorted(tree_abs.glob("*/ftseed*"))
           if d.is_dir()]
    out = load_probe_cache(output_root, rel, refresh=refresh, pool_version=pool_version)
    if len(out):
        parts = out.run_path.str.split("/")
        out["param_tag"] = parts.str[-2]
        out["ftseed"] = parts.str[-1].str.replace("ftseed", "", regex=False).astype(int)
    return out
