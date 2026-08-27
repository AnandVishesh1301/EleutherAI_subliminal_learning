"""The one reader of the run-dir eval contract (``{lineage}/{cell}/ftseed{N}``).

Every collapse/transfer notebook re-declared the same leaf walk: discover cells under a
lineage, discover ``ftseed*`` leaves under each cell, read ``preference_eval.json`` for the
trait score, read a Pile-KL number, and fold ``train_diverged.json`` in as the diverged
flag. The copies had drifted on the two decisions that matter -- *where* the KL comes from
(the cached ``eval/pile_probe.json`` records vs. the endpoint of ``eval/pile_kl_traj.json``)
and what a diverged run scores -- so `load_run_evals` takes the KL source as a parameter and
pins the diverged convention once.

**Diverged convention** (from the tier-B strips): a run with ``train_diverged.json``, or one
whose KL-trajectory endpoint is flagged ``nonfinite``, reports ``kl = +inf`` and
``test = 0.0``. Not ``NaN``: a diverged adapter *did* land at the far side of the boundary,
so it is a real observation of maximal damage / zero trait, and dropping it would censor
exactly the rows a collapse fit needs. ``NaN`` is reserved for *missing* readouts.

This module loads and nothing else. K / K_rel dose math lives in `analysis.collapse_law`,
and the dead / collapsed labels live in `analysis.labels` -- deriving either here would bake
one campaign's anchor into a reader shared by all of them.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from subliminal_learning.analysis.frames import parse_param_tag, safe_read_json

_KL_SOURCES = ("probe", "traj_end", "none")


def _traj_endpoint(leaf: Path) -> tuple[float, bool]:
    """``(kl, nonfinite)`` at the last recorded step of ``eval/pile_kl_traj.json``.

    Returns ``(nan, False)`` when the trajectory is absent/unreadable or has no points. The
    endpoint is picked with a single ``max(..., key=step)`` pass (the notebook copies scanned
    the points once per point to find the maximum step).
    """
    doc = safe_read_json(leaf / "eval" / "pile_kl_traj.json")
    points = (doc or {}).get("points") or []
    if not points:
        return float("nan"), False
    end = max(points, key=lambda p: p.get("step", -1))
    if end.get("nonfinite"):
        return float("nan"), True
    kl = end.get("kl_base_adapted")
    return (float("nan") if kl is None else float(kl)), False


def load_run_evals(
    output_root,
    lineage,
    *,
    cells: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    prompt_set: str = "test",
    kl_source: str = "probe",
    pool_version: str | None = "pp1",
) -> pd.DataFrame:
    """One row per ``{lineage}/{cell}/ftseed{N}`` leaf: trait score, Pile-KL, fate flags.

    Args:
        output_root: The outputs root (the probe cache and its relative run paths resolve
            against it).
        lineage: The dataset-tag level holding the ``{cell}/ftseed{N}`` leaves, absolute or
            relative to `output_root`.
        cells: Student tags to read. ``None`` discovers them as ``sorted(lineage.glob(
            "ep*_*"))`` -- pass a campaign manifest (see `frames.read_cell_manifest`) when
            the inventory must be fixed rather than whatever landed.
        seeds: Fine-tuning seeds to read. ``None`` discovers each cell's ``ftseed*`` dirs.
        prompt_set: Which ``preference_eval.json`` set supplies ``test`` (default the
            held-out ``"test"`` prompts).
        kl_source: ``"probe"`` reads the batched, cached ``eval/pile_probe.json`` records via
            `probe_cache.load_probe_cache` (one sweep for the whole inventory); ``"traj_end"``
            reads each run's ``eval/pile_kl_traj.json`` and takes ``kl_base_adapted`` at the
            last step. They are different measurements -- probe records are the pooled
            ``pp1`` probe, the trajectory endpoint is the per-checkpoint sweep's last point --
            so pick one per analysis, never mix them in one frame. ``"none"`` skips the KL
            readout entirely (``kl`` all-``NaN`` bar the diverged ``+inf``) for the trait-only
            passes that would otherwise pay for a cache sweep they never read.
        pool_version: Probe-pool identity required of cached records, forwarded to
            `probe_cache.load_probe_cache` (rows from another pool are dropped there, never
            silently pooled). ``None`` disables the filter. Ignored unless
            ``kl_source="probe"``.

    Returns:
        A frame with ``student_tag``, ``ftseed``, ``leaf`` (str), ``landed``, ``diverged``,
        ``kl``, ``test`` plus ``lr``/``alpha``/``rank``/``epochs`` from `parse_param_tag`.
        Missing readouts are ``NaN``; diverged rows are ``kl = +inf``, ``test = 0.0``.
    """
    if kl_source not in _KL_SOURCES:
        raise ValueError(f"unknown kl_source {kl_source!r}; choose from {_KL_SOURCES}")
    output_root = Path(output_root)
    lineage = Path(lineage)
    lineage_abs = lineage if lineage.is_absolute() else output_root / lineage
    tags = (list(cells) if cells is not None
            else [d.name for d in sorted(lineage_abs.glob("ep*_*")) if d.is_dir()])
    # Materialize once: `seeds` is consumed per cell, so a generator would empty after cell 1.
    seeds = None if seeds is None else [int(s) for s in seeds]

    rows = []
    for tag in tags:
        cell_dir = lineage_abs / tag
        if seeds is not None:
            leaves = [(s, cell_dir / f"ftseed{s}") for s in seeds]
        else:
            leaves = [(int(d.name.removeprefix("ftseed")), d)
                      for d in sorted(cell_dir.glob("ftseed*")) if d.is_dir()]
        pt = parse_param_tag(tag)
        for seed, leaf in leaves:
            rec = {
                "student_tag": tag,
                "ftseed": seed,
                "leaf": str(leaf),
                "landed": leaf.is_dir(),
                "diverged": (leaf / "train_diverged.json").exists(),
                "kl": float("nan"),
                "test": float("nan"),
                "epochs": pt["epochs"],
                "lr": pt["lr"],
                "rank": pt["rank"],
                "alpha": pt["alpha"],
            }
            doc = safe_read_json(leaf / "preference_eval.json")
            if doc is not None:
                score = (doc.get("sets") or {}).get(prompt_set)
                if score is not None and score.get("p_target") is not None:
                    rec["test"] = float(score["p_target"])
            if kl_source == "traj_end":
                kl, nonfinite = _traj_endpoint(leaf)
                rec["kl"] = kl
                if nonfinite:
                    rec["diverged"] = True
            rows.append(rec)

    df = pd.DataFrame(rows, columns=["student_tag", "ftseed", "leaf", "landed", "diverged",
                                     "kl", "test", "epochs", "lr", "rank", "alpha"])
    if kl_source == "probe" and len(df):
        from subliminal_learning.analysis.probe_cache import load_probe_cache

        rel = [str(Path(p).relative_to(output_root)) for p in df.leaf]
        pc = load_probe_cache(output_root, rel, pool_version=pool_version)
        if len(pc):
            kl_by_leaf = {str(output_root / r): float(k)
                          for r, k in zip(pc.run_path, pc.kl)}
            df["kl"] = [kl_by_leaf.get(leaf, float("nan")) for leaf in df.leaf]
    if len(df):
        df.loc[df.diverged, ["kl", "test"]] = [np.inf, 0.0]
    return df


def peak_summary(df: pd.DataFrame, val: str = "transfer", tag_col: str = "student_tag",
                 k_col: str = "K_rel") -> dict:
    """Where a population's transfer peaks: the argmax run and its top decile.

    The peak-location readout the 410m and non-adaptive-strip notebooks each hand-rolled: sort
    by `val`, report the single best run, then the top decile (at least 3 runs -- a thin arm
    still gets a centre rather than an empty summary) as a median value plus the geometric
    centre and spread of its dose. Rows with a missing `val` are dropped first.

    Args:
        df: Run-level rows carrying `val`, `tag_col` and `k_col` (and optionally ``seed``).
        val: The transfer-like column whose peak is summarised.
        tag_col: Cell-identifying column reported for the argmax run.
        k_col: The dose column (K/K*), summarised geometrically.

    Returns:
        ``n``, ``argmax_cell`` (plus ``argmax_seed`` when a ``seed`` column exists),
        ``argmax_transfer``, ``argmax_K_rel``, ``top_decile_n``, ``top_decile_median``,
        ``top_decile_K_rel_geomean``, ``top_decile_K_rel_range`` (min, max),
        ``top_decile_K_rel_iqr`` (q25, q75), ``frac_above_prior`` (`val` > 1) and
        ``frac_above_2x`` (`val` > 2). Empty input gives ``{"n": 0}``.
    """
    s = df.dropna(subset=[val]).sort_values(val, ascending=False)
    if not len(s):
        return {"n": 0}
    top = s.head(max(3, len(s) // 10))
    out = {"n": len(s), "argmax_cell": s.iloc[0][tag_col]}
    if "seed" in s.columns:
        out["argmax_seed"] = int(s.iloc[0]["seed"])
    out.update({
        "argmax_transfer": s.iloc[0][val],
        "argmax_K_rel": s.iloc[0][k_col],
        "top_decile_n": len(top),
        "top_decile_median": top[val].median(),
        "top_decile_K_rel_geomean": float(np.exp(np.log(top[k_col]).mean())),
        "top_decile_K_rel_range": (round(top[k_col].min(), 3), round(top[k_col].max(), 3)),
        "top_decile_K_rel_iqr": (round(top[k_col].quantile(0.25), 3),
                                 round(top[k_col].quantile(0.75), 3)),
        "frac_above_prior": float((s[val] > 1).mean()),
        "frac_above_2x": float((s[val] > 2).mean()),
    })
    return out
