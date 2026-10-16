"""Shared reader for per-checkpoint trajectories (pref / Pile-KL / dose).

The three per-checkpoint readouts (`scripts.ckpt_preference_trajectory`,
`scripts.ckpt_pile_kl_trajectory`, `scripts.ckpt_dose_trajectory`) moved from a flat
`analysis/ckpt_*_traj/{tag}_ftseed{S}.json` layout to a canonical per-run
`{run_dir}/eval/{pref,pile_kl,dose}_traj.json` (a sibling of `checkpoints/`, so a later
checkpoint prune can never orphan an eval, and a retrained run's eval can never go stale).
Both layouts exist on disk during the migration, so every consumer needs the same
resolution order:

1. ``{run_dir}/eval/{kind}_traj.json``            -- canonical, identity-stamped
2. ``analysis/ckpt_{kind}_traj/{tag}_ftseed{S}.json`` -- legacy flat

and the same *shape*. The legacy pref files are dog-only (a bare ``{"test": {...}}`` per
point); the new ones nest per animal (``{"test": {"dog": {...}, "cat": {...}}}``). This
module normalizes the legacy form UP into the multi-animal schema using the file's own
``animal`` field, so callers only ever branch on animals, never on file vintage.

Every accessor returns a `Trajectory` whose ``source`` says which layout answered
("rundir" / "legacy"), so a notebook can report coverage of the migration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# kind -> (run-dir filename, legacy flat subdir under analysis/)
_KINDS = {
    "pref": ("pref_traj.json", "ckpt_pref_traj"),
    "pile_kl": ("pile_kl_traj.json", "ckpt_pile_kl_traj"),
    "dose": ("dose_traj.json", "ckpt_dose_traj"),
}


@dataclass(frozen=True)
class Trajectory:
    """One run's trajectory for one readout.

    Attributes:
        kind: 'pref' | 'pile_kl' | 'dose'.
        tag: Student cell tag (e.g. ep20_lr5e-4_r64a8).
        ftseed: Student fine-tuning seed.
        points: The per-checkpoint rows, in step order (schema per kind; pref rows are
            normalized to the multi-animal shape).
        animals: Animals present in a pref trajectory (empty for the other kinds).
        source: Which layout supplied the data -- 'rundir' or 'legacy'.
        path: The file actually read.
        meta: The file's remaining top-level keys (identity stamps, HPs, pool ids...).
    """

    kind: str
    tag: str
    ftseed: int
    points: list[dict]
    animals: tuple[str, ...]
    source: str
    path: Path
    meta: dict = field(default_factory=dict)

    def series(self, key: str, animal: str | None = None) -> list:
        """Per-checkpoint values of one field, aligned to `points`.

        For 'pref', `key` is the prompt-set name (e.g. 'test') and `animal` selects the
        animal (defaults to the trajectory's first). Missing/non-finite rows yield None.
        """
        out = []
        for p in self.points:
            if p.get("nonfinite"):
                out.append(None)
                continue
            v = p.get(key)
            if self.kind == "pref" and isinstance(v, dict):
                a = animal or (self.animals[0] if self.animals else None)
                v = v.get(a) if a else None
            out.append(v)
        return out

    @property
    def steps(self) -> list:
        return [p.get("step") for p in self.points]


def _normalize_pref(doc: dict) -> tuple[list[dict], tuple[str, ...]]:
    """Return (points, animals) with every prob set nested per animal.

    A legacy dog-only file stores each prob set as a flat SetScore dict; the multi-animal
    files store ``{animal: SetScore}``. Detected per row by whether the set's value has the
    scalar ``p_target`` key -- so a file written either way normalizes to the same shape.
    """
    animals = tuple(doc.get("animals") or ([doc["animal"]] if doc.get("animal") else ()))
    fallback = animals[0] if animals else "unknown"
    points = []
    for p in doc.get("points", []):
        row = dict(p)
        for k, v in p.items():
            if isinstance(v, dict) and "p_target" in v:  # legacy flat SetScore -> nest
                row[k] = {fallback: v}
        points.append(row)
    if not animals:
        animals = (fallback,)
    return points, animals


def load_trajectory(
    kind: str,
    *,
    run_dir: Path | None = None,
    analysis_dir: Path | None = None,
    tag: str | None = None,
    ftseed: int | None = None,
) -> Trajectory | None:
    """Load one run's trajectory, preferring the run dir over the legacy flat file.

    Args:
        kind: 'pref' | 'pile_kl' | 'dose'.
        run_dir: The run leaf (``.../{tag}/ftseed{S}``); enables the canonical lookup.
        analysis_dir: ``{output_root}/analysis``; enables the legacy fallback.
        tag, ftseed: Required for the legacy fallback (inferred from `run_dir` if absent).

    Returns:
        A `Trajectory`, or None when neither layout has the run.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {sorted(_KINDS)}")
    fname, legacy_sub = _KINDS[kind]
    if run_dir is not None:
        run_dir = Path(run_dir)
        if tag is None:
            tag = run_dir.parent.name
        if ftseed is None and run_dir.name.startswith("ftseed"):
            ftseed = int(run_dir.name.removeprefix("ftseed"))
    candidates: list[tuple[str, Path]] = []
    if run_dir is not None:
        candidates.append(("rundir", run_dir / "eval" / fname))
    if analysis_dir is not None and tag is not None and ftseed is not None:
        candidates.append(("legacy", Path(analysis_dir) / legacy_sub / f"{tag}_ftseed{ftseed}.json"))

    for source, path in candidates:
        if not path.exists():
            continue
        doc = json.loads(path.read_text())
        # The dose writer names the seed `ftseed`; pref/KL use `ft_seed`.
        seed = doc.get("ft_seed", doc.get("ftseed", ftseed))
        if kind == "pref":
            points, animals = _normalize_pref(doc)
        else:
            points, animals = doc.get("points", []), ()
        meta = {k: v for k, v in doc.items() if k != "points"}
        return Trajectory(
            kind=kind, tag=doc.get("tag", tag), ftseed=int(seed) if seed is not None else -1,
            points=points, animals=animals, source=source, path=path, meta=meta,
        )
    return None


def load_run_trajectories(
    run_dir: Path, analysis_dir: Path | None = None, kinds=("pref", "pile_kl", "dose")
) -> dict[str, Trajectory]:
    """All available readouts for one run: kind -> Trajectory (missing kinds omitted)."""
    out = {}
    for kind in kinds:
        t = load_trajectory(kind, run_dir=run_dir, analysis_dir=analysis_dir)
        if t is not None:
            out[kind] = t
    return out


def iter_registry_runs(registry: dict, lineage_dir: Path):
    """Yield (tag, ftseed, run_dir) for every (cell, seed) in a campaign registry.

    `lineage_dir` is the ``{dataset_tag}`` level holding the ``{tag}/ftseed{S}`` leaves --
    the trajectory scripts' ``param_dir`` parent, i.e. what the registry's lineage resolves to.
    """
    for tag, seeds in registry["cells"].items():
        for seed in seeds:
            yield tag, seed, Path(lineage_dir) / tag / f"ftseed{seed}"


def _trajectory_from_doc(kind: str, doc: dict, tag: str, ftseed: int, path: Path) -> Trajectory:
    """Build a `Trajectory` from an already-parsed run-dir document (see `load_trajectory`)."""
    seed = doc.get("ft_seed", doc.get("ftseed", ftseed))
    if kind == "pref":
        points, animals = _normalize_pref(doc)
    else:
        points, animals = doc.get("points", []), ()
    return Trajectory(
        kind=kind, tag=doc.get("tag", tag), ftseed=int(seed) if seed is not None else -1,
        points=points, animals=animals, source="rundir", path=path,
        meta={k: v for k, v in doc.items() if k != "points"},
    )


def load_lineage(
    registry: dict, lineage_dir: Path, kind: str, analysis_dir: Path | None = None,
    cache_root: Path | None = None, refresh: bool = True,
) -> dict[tuple[str, int], Trajectory]:
    """(tag, ftseed) -> Trajectory for every run of a registry that has one on disk.

    `cache_root` is opt-in: give it the outputs root and every run dir under it is served
    from the shared record cache (one batched sweep instead of one read per run, see
    `analysis.record_cache`). Runs outside `cache_root`, and the legacy `analysis_dir`
    fallback, always take the direct-read path. With `cache_root=None` this is exactly the
    per-run `load_trajectory` loop.

    `refresh=False` serves the stored cache as-is (no filesystem sweep, no write; see
    `record_cache.load_record_cache`). It is only sound when a refresh over a superset of this
    lineage's runs already ran in the same process (e.g. `trajectory_conditions._prefetch`);
    runs absent from the cache (or a missing cache CSV) fall back to the direct read.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown kind {kind!r}; choose from {sorted(_KINDS)}")
    fname = _KINDS[kind][0]
    runs = list(iter_registry_runs(registry, lineage_dir))

    blobs: dict[str, str] = {}
    rel_of: dict[Path, str] = {}
    if cache_root is not None:
        from subliminal_learning.analysis.record_cache import load_record_cache

        cache_root = Path(cache_root)
        for _, _, run_dir in runs:
            try:
                rel_of[run_dir] = str(Path(run_dir).relative_to(cache_root))
            except ValueError:            # not under cache_root -> direct read
                continue
        out = load_record_cache(
            cache_root, "json_blob", list(rel_of.values()), refresh, verbose=False,
            rel_record=f"eval/{fname}",
            cache_rel=Path("analysis") / f"record_cache_traj_{kind}.csv",
        )
        blobs = dict(zip(out["run_path"], out["blob"]))

    result = {}
    for tag, seed, run_dir in runs:
        rel = rel_of.get(run_dir)
        if rel is not None and rel in blobs:
            try:
                doc = json.loads(blobs[rel])
            except (TypeError, ValueError):
                doc = None
            if doc is not None:
                result[(tag, seed)] = _trajectory_from_doc(
                    kind, doc, tag, seed, Path(run_dir) / "eval" / fname
                )
                continue
        t = load_trajectory(kind, run_dir=run_dir, analysis_dir=analysis_dir,
                            tag=tag, ftseed=seed)
        if t is not None:
            result[(tag, seed)] = t
    return result


def _pref_points(run_dir: Path, prompt_set: str) -> tuple[list, list]:
    """``(steps, p_target)`` from ``eval/pref_traj.json``, tolerating both point schemas.

    A point's prob set is either the flat ``{"p_target": ...}`` (the 410m writer) or the
    per-animal ``{animal: {"p_target": ...}}`` (the 160m strip writer); the animal-keyed form
    is resolved through the file's own ``animal``/``animals`` field, so callers never branch
    on file vintage. Points with no ``step`` or no `prompt_set` entry are dropped.
    """
    doc = json.loads((Path(run_dir) / "eval" / "pref_traj.json").read_text())
    animals = doc.get("animals") or ([doc["animal"]] if doc.get("animal") else [])
    steps, vals = [], []
    for p in doc.get("points", []):
        if p.get("step") is None or prompt_set not in p:
            continue
        t = p[prompt_set]
        if "p_target" in t:
            v = t["p_target"]
        else:
            a = next((a for a in animals if a in t), None)
            if a is None:
                continue
            v = t[a]["p_target"]
        steps.append(float(p["step"]))
        vals.append(float(v))
    return steps, vals


def _kl_points(run_dir: Path) -> tuple[list, list]:
    """``(steps, kl_base_adapted)`` from ``eval/pile_kl_traj.json``.

    Points without a ``step`` are dropped. A point flagged ``nonfinite`` reads as
    ``+inf`` (the diverged convention `run_evals` uses); a point that simply carries no
    ``kl_base_adapted`` is skipped.
    """
    doc = json.loads((Path(run_dir) / "eval" / "pile_kl_traj.json").read_text())
    steps, vals = [], []
    for p in doc.get("points", []):
        if p.get("step") is None:
            continue
        if p.get("nonfinite"):
            v = float("inf")
        elif "kl_base_adapted" in p:
            v = float(p["kl_base_adapted"])
        else:
            continue
        steps.append(float(p["step"]))
        vals.append(v)
    return steps, vals


def endpoint_stats(run_dir, *, kl_cut: float | None = None,
                   prompt_set: str = "test") -> dict:
    """Where along training one run's trait peaked, and the Pile-KL there.

    The shared core of the notebooks' ``traj_stats``: the trait trajectory's argmax gives
    ``peak_step_frac`` (peak step / last step), ``peak_test``, ``end_test`` (the last point)
    and ``overshoot`` (peak / end) -- the readout that says whether a run's trait decays after
    an early peak.

    Args:
        run_dir: The run leaf (``.../{tag}/ftseed{S}``).
        kl_cut: Integrity cut in nats. Only when given is ``kl_cross_step_frac`` computed --
            the cut is always injected because the operative label has moved (trait prior ->
            KL > 4 -> the anchored cross), so no cut may be hardcoded here.
        prompt_set: Which prob set of ``pref_traj.json`` supplies the trait score.

    Returns:
        ``{peak_step_frac, peak_test, end_test, overshoot, n_ckpt}``, empty when the run has
        no pref trajectory or no usable points. When ``eval/pile_kl_traj.json`` also exists,
        plus ``kl_at_peak`` (interpolated onto the peak step) and ``kl_end_traj``; plus
        ``kl_cross_step_frac`` (first step where ``kl > kl_cut``, ``NaN`` if never) when
        `kl_cut` is given.
    """
    run_dir = Path(run_dir)
    if not (run_dir / "eval" / "pref_traj.json").exists():
        return {}
    steps, vals = _pref_points(run_dir, prompt_set)
    if not steps:
        return {}
    s = np.asarray(steps, dtype=float)
    tp = np.asarray(vals, dtype=float)
    i = int(np.nanargmax(tp))
    out = {
        "peak_step_frac": float(s[i] / s.max()),
        "peak_test": float(tp[i]),
        "end_test": float(tp[-1]),
        "overshoot": float(tp[i] / max(tp[-1], 1e-9)),
        "n_ckpt": len(steps),
    }
    if (run_dir / "eval" / "pile_kl_traj.json").exists():
        ks, kl = _kl_points(run_dir)
        if ks:
            ka = np.asarray(kl, dtype=float)
            kx = np.asarray(ks, dtype=float)
            out["kl_at_peak"] = float(np.interp(s[i], kx, ka))
            out["kl_end_traj"] = float(ka[-1])
            if kl_cut is not None:
                cross = np.where(ka > kl_cut)[0]
                out["kl_cross_step_frac"] = (float(kx[cross[0]] / kx.max())
                                             if len(cross) else float("nan"))
    return out


def pref_kl_pairs(run_dir, prompt_set: str = "test") -> pd.DataFrame:
    """Step-aligned trait / Pile-KL frame for one run (the trajectory scatter's input).

    The two trajectories are written by separate scripts and need not share a checkpoint
    grid, so the KL series is interpolated onto the pref steps -- an inner join: pref steps
    outside the KL series' support are dropped, never extrapolated.

    Args:
        run_dir: The run leaf (``.../{tag}/ftseed{S}``).
        prompt_set: Which prob set supplies the trait score (both point schemas handled).

    Returns:
        Columns ``step``, ``test``, ``kl``; empty frame when either trajectory is missing or
        has no usable points.
    """
    run_dir = Path(run_dir)
    empty = pd.DataFrame(columns=["step", "test", "kl"])
    if not ((run_dir / "eval" / "pref_traj.json").exists()
            and (run_dir / "eval" / "pile_kl_traj.json").exists()):
        return empty
    steps, vals = _pref_points(run_dir, prompt_set)
    ks, kl = _kl_points(run_dir)
    if not steps or not ks:
        return empty
    s = np.asarray(steps, dtype=float)
    t = np.asarray(vals, dtype=float)
    # Honour the inner-join contract: np.interp would hold the endpoint values outside the
    # KL series' support, so pref steps outside it are dropped rather than extrapolated.
    keep = (s >= min(ks)) & (s <= max(ks))
    s, t = s[keep], t[keep]
    return pd.DataFrame({"step": s, "test": t, "kl": np.interp(s, ks, kl)})


def source_counts(trajectories) -> dict[str, int]:
    """Coverage of the migration: how many trajectories came from each layout."""
    counts: dict[str, int] = {}
    values = trajectories.values() if isinstance(trajectories, dict) else trajectories
    for t in values:
        counts[t.source] = counts.get(t.source, 0) + 1
    return counts


def lineage_long_frame(
    registry: dict, lineage_dir: Path, *, cache_root: Path | None = None,
    analysis_dir: Path | None = None, prompt_set: str = "test", animal: str | None = None,
    refresh: bool = True,
) -> pd.DataFrame:
    """Long frame of a lineage's trait + Pile-KL trajectories: one row per (run, step).

    Joins `load_lineage` for ``pref`` and ``pile_kl`` on ``(tag, ftseed, step)``. The trait
    score is ``p_target`` of `prompt_set` for `animal` (default: the file's first animal);
    ``nonfinite`` KL points read as ``+inf`` in both damage columns. The final step is often recorded twice (the
    ``checkpoint-N`` dir and the endpoint adapter); one row per step is kept (the last).

    Args:
        registry: ``{"cells": {tag: [ftseed, ...]}}`` (a campaign registry or a synthesized one).
        lineage_dir: The ``{dataset_tag}`` level holding the ``{tag}/ftseed{S}`` leaves.
        cache_root: Outputs root, to serve the files from the shared record cache.
        analysis_dir: Legacy flat-layout fallback (see `load_trajectory`).
        prompt_set: Which prob set supplies the trait score.
        animal: Which animal's score to read from a multi-animal file.
        refresh: Passed to `load_lineage`; False only after a same-process prefetch covering
            this lineage's runs.

    Returns:
        Columns ``tag, ftseed, step, frac, p_test, kl, delta_ce`` with ``frac = step / max
        step of that run``; a run present in only one of the two readouts carries NaN in the
        other. ``delta_ce`` is NaN on checkpoints written before the CE readout existed.
    """
    pref = load_lineage(registry, lineage_dir, "pref", analysis_dir=analysis_dir,
                        cache_root=cache_root, refresh=refresh)
    kl = load_lineage(registry, lineage_dir, "pile_kl", analysis_dir=analysis_dir,
                      cache_root=cache_root, refresh=refresh)
    rows = []
    for key in sorted(set(pref) | set(kl)):
        tag, seed = key
        p = {}
        if key in pref:
            t = pref[key]
            a = animal or (t.animals[0] if t.animals else None)
            for q in t.points:
                if q.get("step") is None or prompt_set not in q:
                    continue
                s = q[prompt_set]
                if a in s:
                    p[float(q["step"])] = float(s[a]["p_target"])
        k, ce = {}, {}
        if key in kl:
            for q in kl[key].points:
                if q.get("step") is None:
                    continue
                if q.get("nonfinite"):
                    k[float(q["step"])] = ce[float(q["step"])] = float("inf")
                else:
                    if "kl_base_adapted" in q:
                        k[float(q["step"])] = float(q["kl_base_adapted"])
                    if q.get("delta_ce") is not None:
                        ce[float(q["step"])] = float(q["delta_ce"])
        steps = sorted(set(p) | set(k) | set(ce))
        if not steps:
            continue
        last = steps[-1]
        for s in steps:
            rows.append({"tag": tag, "ftseed": seed, "step": s, "frac": s / last,
                         "p_test": p.get(s, np.nan), "kl": k.get(s, np.nan),
                         "delta_ce": ce.get(s, np.nan)})
    cols = ["tag", "ftseed", "step", "frac", "p_test", "kl", "delta_ce"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------------------------------
# (rank x alpha) grids over runs, and nearest-checkpoint slices of a long trajectory frame
# --------------------------------------------------------------------------------------------------


def rank_alpha_grid(df: pd.DataFrame, value: str, ranks=None, alphas=None,
                    agg: str = "median") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(rank x alpha) `agg` of `value` over the rows of `df`, plus the per-cell row count.

    Args:
        df: Rows carrying ``rank``, ``alpha`` and `value` (one row per run or per run-checkpoint).
        value: Column to aggregate.
        ranks, alphas: When given, both frames are reindexed to exactly these rows / columns
            (missing cells become NaN); when None, only the observed levels appear.
        agg: Aggregation for the value grid (the count grid is always ``size``).

    Returns:
        ``(grid, counts)``.
    """
    g = df.pivot_table(index="rank", columns="alpha", values=value, aggfunc=agg)
    n = df.pivot_table(index="rank", columns="alpha", values=value, aggfunc="size")
    if ranks is not None or alphas is not None:
        g, n = g.reindex(index=ranks, columns=alphas), n.reindex(index=ranks, columns=alphas)
    return g, n


def nearest_step_slice(tr: pd.DataFrame, step: int, tags=None, *, rel_tol: float = 0.1,
                       max_step_factor: float | None = None,
                       require_value: str | None = None) -> pd.DataFrame:
    """Per run (``tag``), the saved checkpoint row nearest `step`, kept only within `rel_tol`.

    Pre-filters, applied before the nearest pick: ``tag in tags`` (when given), rows with a
    non-NaN `require_value` (when given), and ``step <= max_step_factor * step`` (when given).
    Ties in distance go to the first row in frame order (``idxmin``). The result carries an
    extra ``d = |step - target|`` column.
    """
    t = tr if tags is None else tr[tr["tag"].isin(tags)]
    if require_value is not None:
        t = t[t[require_value].notna()]
    if max_step_factor is not None:
        t = t[t["step"] <= max_step_factor * step]
    t = t.copy()
    if t.empty:
        return t.assign(d=pd.Series(dtype=float))
    t["d"] = (t["step"] - step).abs()
    pick = t.loc[t.groupby("tag")["d"].idxmin()]
    return pick[pick["d"] <= rel_tol * step]


def grid_at_step(tr: pd.DataFrame, tags, step: int, value: str = "p_test", ranks=None,
                 alphas=None, **slice_kw):
    """`rank_alpha_grid` of `value` over `nearest_step_slice` rows, with the median picked step.

    Returns:
        ``((grid, counts), median_step)``, or ``(None, None)`` when no run has a checkpoint in
        tolerance. ``slice_kw`` goes to `nearest_step_slice`.
    """
    pick = nearest_step_slice(tr, step, tags, **slice_kw)
    if pick.empty:
        return None, None
    return rank_alpha_grid(pick, value, ranks, alphas), int(pick["step"].median())


def grid_nanmedian(g: pd.DataFrame, rows=None) -> float:
    """NaN-ignoring median over every cell of `g` (optionally only the index labels `rows`)."""
    v = g if rows is None else g.loc[rows]
    return float(np.nanmedian(v.to_numpy()))
