"""Is transfer asymmetry a property of the whole trajectory or only of the endpoint?

`analysis.transfer_asymmetry` tests A -> B against B -> A at the END of student training. This
module reads the same contrast at EVERY saved checkpoint step of the same runs, so
`notebooks/transfer_trajectory_asymmetry.ipynb` can show when the asymmetry appears, whether it
holds its sign, and whether the endpoint verdict is representative.

Pair universe. Everything that decides WHICH runs are compared is the endpoint module's:
families, animals, base order (A = the base listed first, so the sign says who sends more), the
largest-first design fallback and the finished-set rule are `transfer_asymmetry.analyze_family`;
the run universe of a direction is the ``(teacher, cell)`` rows of its endpoint
`transfer_asymmetry.ArmLoad.frame` at the pair's design. A pair the endpoint module excludes stays
excluded here with the same reason; a pair it tests but with no trajectory on one direction is
excluded here with a "no trajectory" reason.

Per-step value. ``value_t = (p_test(t) - control_endpoint(cell)) / prior``: ``p_test(t)`` is the
run's test p_target at saved step t (`trajectories.lineage_long_frame`, which already keeps the
endpoint recorded twice once), ``control_endpoint(cell)`` is the SAME matched control endpoint the
endpoint module subtracts (`tio.load_cell_matrix`, control column) and ``prior`` the same receiver
test prior. Control runs have checkpoints but no per-checkpoint evals on disk, so the control is a
per-cell CONSTANT along the trajectory; at the final saved step the value reproduces the endpoint
module's value exactly whenever the trajectory's last point equals the endpoint preference eval.
Steps are ABSOLUTE optimizer steps, never fractions of training.

Per-step test. At each step present in both directions the condition frame (stratum = cell) goes
through `transfer_asymmetry.contrast_row` with the endpoint module's paired/unpaired rule
(paired iff identical teacher tag sets) and its permutation defaults. A step is TESTED only when
both directions carry at least `MIN_COVERAGE` of their endpoint run universe at that step;
untested steps stay in the step table with their coverage and NaN statistics.

Per-step-control design. The dog orig/fs1-3 and pr1234 <-> orig ep20 runs have backfilled
intermediate checkpoints (`trajectory_conditions.earlystop_arms`, leaves ``{cell}_stop12500``)
with matched control checkpoints (`ArmSpec.control_of`), so the control subtracted is the matched
control's value AT THE SAME STEP (same cell, same ftseed); a cell whose control lacks the step is
uncovered at it. Per (teacher, cell) slot the backfilled points and the same ep20 run's endpoint
value (its last step, matched control endpoint) form the trajectory, labelled
``{cell set} · per-step control`` and added as an EXTRA design row of the family. The internal
backfill-only design (``{cell set}`` + `BACKFILL_TAG` + ``N``) is marked ``design status`` and not displayed.

Δp contrast figure and CSVs. `dp_contrast_table` re-tests the dog per-step-control pairs on
p_student − p_control (same step), NOT prior-scaled; the notebook writes it to
``notebooks/figures/asymmetry/trajectory/ab_ba/asymmetry_trajectory_contrast.csv`` beside
``asymmetry_trajectory_steps.csv`` (`steps_csv_table`, prior units), and
`plot_contrast_curves_dog` draws the figure from that table alone, so
``notebooks/figures/asymmetry/trajectory/ab_ba/asymmetry_trajectory.py`` redraws it from the CSV.
``stats=False`` (`analyze`, `dp_contrast_table`) skips this module's permutation tests.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from subliminal_learning.analysis import result_cache
from subliminal_learning.analysis import trajectories as traj_io
from subliminal_learning.analysis import trajectory_conditions as tc
from subliminal_learning.analysis import transfer_asymmetry as ta
from subliminal_learning.analysis import transfer_significance_io as tio
from subliminal_learning.config import resolve_output_root

__all__ = [
    "MIN_COVERAGE",
    "ALPHA",
    "STEP_COLUMNS",
    "SUMMARY_COLUMNS",
    "INVENTORY_COLUMNS",
    "EXCLUDED_COLUMNS",
    "TrajectoryAsymmetry",
    "analyze_family",
    "analyze",
    "direction_table",
    "PRELIM_MIN_MATCHED",
    "pair_label",
    "FAMILY_DISPLAY",
    "family_display",
    "plot_direction_curves",
    "plot_contrast_curves",
    "N_BOOT",
    "summary_display",
    "plot_summary_dumbbell",
    "plot_coverage",
    "plot_dog_directions",
    "trajectory_directions",
]

#: Share of a direction's endpoint run universe that must carry a value at a step to test it.
MIN_COVERAGE = 0.9

#: Significance level of the summary's "earliest significant step" and the filled markers.
ALPHA = 0.05

#: One row per (pair, step present in either direction).
STEP_COLUMNS = ["family", "animal", "A", "B", "design", "step", "coverage A->B",
                "coverage B->A", "runs A->B", "runs B->A", "mean A->B", "mean B->A",
                "contrast", "p_two_sided", "win prob", "matched share", "tested",
                "coverage class", "mean A->B lo", "mean A->B hi", "mean B->A lo", "mean B->A hi",
                "contrast lo", "contrast hi"]

#: One row per pair with at least one tested step.
SUMMARY_COLUMNS = ["family", "animal", "A", "B", "design", "design status", "coverage class",
                   "test", "endpoint contrast", "endpoint p", "slots endpoint contrast",
                   "slots endpoint p", "ep20 endpoint contrast", "last tested step", "last-step contrast", "last-step p",
                   "|endpoint - last-step|", "cross-check ok", "tested steps", "peak step",
                   "peak contrast", "earliest p<.05 step", "same-sign share",
                   "opposite-sign significant"]

#: One row per ordered arm of every pair the endpoint module enumerates.
INVENTORY_COLUMNS = ["family", "animal", "sender", "receiver", "design", "design status",
                     "control",
                     "runs (endpoint)",
                     "runs with trajectory", "trajectory", "saved steps", "last step", "control endpoint",
                     "prior", "control / prior", "reason", "notes"]

#: One row per pair not in the summary, with the per-direction reason.
#: One row per (teacher, cell) slot value at a TESTED step of a tested pair, per direction.
POINT_COLUMNS = ["family", "animal", "A", "B", "design", "direction", "teacher", "cell", "step",
                 "value"]

EXCLUDED_COLUMNS = ["family", "animal", "A", "B", "reason A->B", "reason B->A", "superseded by"]

#: Per-step 95% bands: a TEACHER-CLUSTER bootstrap (teachers resampled with replacement, the
#: exchangeable unit of the paired test), `N_BOOT` draws from ``default_rng(BOOT_SEED)``. A
#: direction's band is of the mean over its (teacher, cell) slots at the step; a pair's contrast
#: band resamples the SAME teachers on both directions. Untested steps carry NaN.
N_BOOT, BOOT_SEED = 1000, 0

NO_TRAJECTORY = "no trajectory"

#: Prelim rule (stop designs only, ``prelim=True``): a step below `MIN_COVERAGE` is still tested
#: when both directions carry at least this many matched (teacher, cell) slots with the student
#: AND its same-step control, using only those runs; its ``coverage class`` is then ``prelim``.
PRELIM_MIN_MATCHED = 20
CLASS_FULL, CLASS_PRELIM = "full", "prelim"

#: One row per (stop arm, step): the cross arm beside its sender's and receiver's diagonals.
DIRECTION_COLUMNS = ["animal", "arm", "sender", "receiver", "reverse landed", "step", "n",
                     "coverage",
                     "mean", "median", "mean lo", "mean hi",
                     "sender diag n", "sender diag mean", "sender diag median",
                     "receiver diag n", "receiver diag mean", "receiver diag median"]

#: One row per stop treatment arm: what has landed.
LANDING_COLUMNS = ["animal", "group", "arm", "runs landed", "designed runs",
                   "controls landed", "designed controls", "status"]

#: The two control conventions of the inventory's ``control`` column.
CONTROL_CONSTANT, CONTROL_PER_STEP = "endpoint constant", "per-step"

#: Per-step-control design label suffix (`_stop_pairs`): the backfilled checkpoints of an ep20
#: run plus, per (teacher, cell) slot, that run's endpoint value at its last step.
TRAJECTORY_SUFFIX = " · per-step control"
#: Tag of the internal backfill-only design (saved checkpoints up to step N, no ep20 endpoint).
BACKFILL_TAG = " · checkpoints to "
SUPERSEDED_BY = "superseded by "
#: Display name of the backfill cell set (stride-3 subset of core30).
_CELLSET_NAMES = {"earlystop": "core30 stride-3"}

#: Fallback last step of a plain ep20 student when no ep20 trajectory of the arm is on disk: the
#: Pythia ep20 schedule, 20 epochs x 1,250 optimizer steps (the last saved step of every ep20
#: trajectory the existing curves read). The source used is written to the inventory notes.
EP20_LAST_STEP = 25000

#: Family-display order: the Pythia-160m families first, OLMo last.
FAMILY_ORDER = ("cross-seed", "cross-data-order", "re-pretrain", "olmo-recipe", "olmo-seed")
#: Reader-facing names of the Pythia families (keys stay the internal family names).
FAMILY_DISPLAY = {"cross-seed": "diff init, diff data order",
                  "cross-data-order": "same init, diff data order",
                  "re-pretrain": "diff init, same data order"}


def family_display(name: str) -> str:
    """The reader-facing family name; unknown names pass through."""
    return FAMILY_DISPLAY.get(name, name)


NO_COVERED_STEP = f"no step with >= {MIN_COVERAGE:.0%} coverage on both directions"


@dataclass
class TrajectoryAsymmetry:
    """The four tables of the readout plus the endpoint module's own pair table."""

    steps: pd.DataFrame
    summary: pd.DataFrame
    inventory: pd.DataFrame
    excluded: pd.DataFrame
    endpoint: pd.DataFrame
    points: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=POINT_COLUMNS))


def pair_label(a: str, b: str) -> str:
    """``A⇄B`` in the shorts of `trajectory_conditions.short_base`, with display names."""
    da, db = tio.base_name(a), tio.base_name(b)
    ctx = ((da.startswith("pythia-160m") and db.startswith("pythia-160m"))
           or (da.startswith("DataDecide-") and db.startswith("DataDecide-")))
    return tc.display(f"{tc.short_base(da, family_context=ctx)}⇄"
                      f"{tc.short_base(db, family_context=ctx)}")


def _spec(family: ta.Family, animal: str, sender: str, receiver: str, design: str,
          cells: Sequence[str], teachers: Sequence[str]) -> tc.ArmSpec:
    """The trajectory-side spec of one endpoint arm (same trees as `tio.load_cell_matrix`)."""
    return tc.ArmSpec(family.name, family.name, f"{sender}->{receiver}", animal,
                      tio.base_name(sender), tio.base_name(receiver), tuple(teachers), design,
                      tuple(cells), tseed=family.tseed)


def _run_trajectories(spec: tc.ArmSpec, teacher: str, root: Path, ftseed: int) -> pd.DataFrame:
    """``(tag, step, p_test)`` of one teacher's lineage, finite points only; a ``_stop{N}``
    cell's run is the union of its two leaves (`trajectory_conditions.run_points`).

    Reads the record cache WITHOUT a refresh: every caller must have run
    `trajectory_conditions._prefetch` over a superset of `spec` earlier in the process
    (`analyze`, `analyze_family` and `direction_table` all guarantee this).
    """
    return tc.run_points(spec, teacher, root, ftseed=ftseed, refresh=False)[0]


def _sibling_only_end(lineage: Path, cell: str, ftseed: int) -> int | None:
    """The final step of a plain-cell run whose saved points come only from its
    ``{cell}_stop12500`` sibling (the plain leaf has no pref_traj), else None: the plain leaf's
    declared end (`trajectory_conditions._end_step`), `EP20_LAST_STEP` when unreadable."""
    run = lineage / cell / f"ftseed{ftseed}"
    if (run / "eval" / "pref_traj.json").exists() or not (
            lineage / (cell + tc.STOP_SIBLING)).is_dir():
        return None
    last = tc._end_step(run)
    return int(last) if np.isfinite(last) else EP20_LAST_STEP


def _direction_points(family: ta.Family, animal: str, sender: str, receiver: str, design: str,
                      cells: Sequence[str], arm: ta.ArmLoad, root: Path) -> tuple[pd.DataFrame,
                                                                                  dict]:
    """Per-step values of one direction's endpoint run universe, plus its inventory facts."""
    m = tio.load_cell_matrix(animal, sender, receiver, arm.teachers, cells=list(cells),
                             ftseed=family.ftseed, tseed=family.tseed,
                             control_source=family.control_source, output_root=root)
    ctrl = dict(zip(m.cells, m.values[:, m.control_col])) if m.cells else {}
    spec = _spec(family, animal, sender, receiver, design, cells, arm.teachers)
    universe = arm.frame[["teacher", "cell"]]
    ends = {(t, c): v for t, c, v in arm.frame[["teacher", "cell", "value"]].itertuples(
        index=False)}
    parts, n_traj = [], 0
    for teacher, sub in universe.groupby("teacher", sort=False):
        tr = _run_trajectories(spec, teacher, root, family.ftseed)
        by_cell = {c: g for c, g in tr.groupby("tag")}
        lineage = spec.lineage_dir(root, teacher)
        for cell in sub["cell"]:
            g = by_cell.get(cell)
            if g is None or g["step"].nunique() <= 1:
                continue
            n_traj += 1
            step = g["step"].to_numpy(float).astype(int)
            value = (g["p_test"].to_numpy(float) - ctrl[cell]) / arm.prior
            last = _sibling_only_end(lineage, cell, family.ftseed)
            if last is not None:
                # points only from the _stop12500 sibling: the run's endpoint is its last step
                keep = step < last
                step = np.append(step[keep], last)
                value = np.append(value[keep], ends[(teacher, cell)])
            parts.append(pd.DataFrame({"sender": sender, "teacher": teacher, "cell": cell,
                                       "step": step, "value": value}))
    pts = (pd.concat(parts, ignore_index=True) if parts
           else pd.DataFrame(columns=["sender", "teacher", "cell", "step", "value"]))
    cvals = np.array([ctrl[c] for c in m.cells], float) if m.cells else np.array([])
    ctrl_mean = float(np.nanmean(cvals)) if cvals.size else float("nan")
    info = {"control": CONTROL_CONSTANT, "runs (endpoint)": len(universe), "runs with trajectory": n_traj,
            "saved steps": int(pts["step"].nunique()),
            "last step": int(pts["step"].max()) if len(pts) else np.nan,
            "control endpoint": ctrl_mean, "prior": arm.prior,
            "control / prior": ctrl_mean / arm.prior}
    return pts, info


def _boot_weights(n_teachers: int) -> np.ndarray:
    """``(N_BOOT, n_teachers)`` multiplicities of a teacher resample with replacement."""
    idx = np.random.default_rng(BOOT_SEED).integers(0, n_teachers, (N_BOOT, n_teachers))
    return (idx[:, :, None] == np.arange(n_teachers)).sum(1).astype(float)


def _boot_means(pts: pd.DataFrame, teachers: Sequence[str], w: np.ndarray) -> dict:
    """``step -> (N_BOOT,)`` bootstrap means over (teacher, cell) slots: per-teacher per-step
    sums and counts once, then weighted by the resampled teacher multiplicities `w`."""
    if not len(pts):
        return {}
    g = pts.groupby(["step", "teacher"])["value"].agg(["sum", "count"])
    sums = g["sum"].unstack().reindex(columns=list(teachers)).fillna(0.0)
    counts = g["count"].unstack().reindex(columns=list(teachers)).fillna(0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = (w @ sums.to_numpy(float).T) / (w @ counts.to_numpy(float).T)
    return {int(st): boot[:, k] for k, st in enumerate(sums.index)}


def _band(boot: np.ndarray | None) -> tuple[float, float]:
    """2.5 / 97.5 percentiles (NaN draws, i.e. resamples with no slot, dropped)."""
    if boot is None or not np.isfinite(boot).any():
        return np.nan, np.nan
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def _contrast(frame: pd.DataFrame, *, paired: bool, n_perm: int, seed: int,
               stats: bool = True) -> dict:
    """`transfer_asymmetry.contrast_row`; with `stats` off only the observed statistics (the
    same mean contrast and win probability, one Monte-Carlo draw, no enumeration) and NaN p."""
    if stats:
        return ta.contrast_row(frame, ta.CONDITION_A, ta.CONDITION_B, paired=paired,
                               n_perm=n_perm, seed=seed)
    from subliminal_learning.stats import transfer_significance_tests as tt

    def obs(statistic: str) -> float:
        if paired:
            return float(tt.paired_cluster_relabel_permutation(
                frame, cond_a=ta.CONDITION_A, cond_b=ta.CONDITION_B, statistic=statistic,
                condition_col="condition", cluster_col="teacher", stratum_col="stratum",
                value_col="value", max_exact=0, n_perm=1, seed=seed).observed)
        return float(tt.arm_relabel_cell_permutation(
            frame, family_a=ta.CONDITION_A, family_b=ta.CONDITION_B, statistic=statistic,
            family_col="condition", arm_col="arm", stratum_col="stratum", value_col="value",
            max_exact=0, n_perm=1, seed=seed).observed)

    return {"contrast": obs("mean_contrast"), "p_two_sided": np.nan,
            "win prob": obs("win_probability"),
            "matched share": (ta._matched_share(frame, ta.CONDITION_A, ta.CONDITION_B)
                              if paired else np.nan)}


def _step_rows(base: dict, ab: pd.DataFrame, ba: pd.DataFrame, n_ab: int, n_ba: int, *,
               paired: bool, n_perm: int, seed: int, prelim: bool = False,
               stats: bool = True) -> list[dict]:
    """One `STEP_COLUMNS` row per step present in either direction; with `prelim`, a step
    below `MIN_COVERAGE` with >= `PRELIM_MIN_MATCHED` runs on both sides is tested too."""
    rows = []
    teachers = sorted(set(ab["teacher"]) | set(ba["teacher"]))
    w = _boot_weights(len(teachers)) if teachers else None
    boot_a = _boot_means(ab, teachers, w) if teachers else {}
    boot_b = _boot_means(ba, teachers, w) if teachers else {}
    for step in sorted(set(ab["step"]) | set(ba["step"])):
        a, b = ab[ab["step"] == step], ba[ba["step"] == step]
        cov_a, cov_b = len(a) / n_ab, len(b) / n_ba
        full = cov_a >= MIN_COVERAGE and cov_b >= MIN_COVERAGE
        pre = prelim and min(len(a), len(b)) >= PRELIM_MIN_MATCHED
        tested = full or pre
        row = {**base, "step": int(step), "coverage A->B": cov_a, "coverage B->A": cov_b,
               "runs A->B": len(a), "runs B->A": len(b),
               "mean A->B": float(a["value"].mean()) if len(a) else np.nan,
               "mean B->A": float(b["value"].mean()) if len(b) else np.nan,
               "contrast": np.nan, "p_two_sided": np.nan, "win prob": np.nan,
               "matched share": np.nan, "tested": tested,
               "coverage class": CLASS_FULL if full else CLASS_PRELIM if pre else ""}
        if tested:
            frame = ta._condition_frame(a.assign(stratum=a["cell"]), b.assign(stratum=b["cell"]),
                                        ta.CONDITION_A, ta.CONDITION_B)
            r = _contrast(frame, paired=paired, n_perm=n_perm, seed=seed, stats=stats)
            row.update({k: r[k] for k in ("contrast", "p_two_sided", "win prob",
                                          "matched share")})
            ba_, bb_ = boot_a.get(int(step)), boot_b.get(int(step))
            row["mean A->B lo"], row["mean A->B hi"] = _band(ba_)
            row["mean B->A lo"], row["mean B->A hi"] = _band(bb_)
            row["contrast lo"], row["contrast hi"] = _band(
                ba_ - bb_ if ba_ is not None and bb_ is not None else None)
        rows.append(row)
    return rows


def _tested_points(base: dict, ab: pd.DataFrame, ba: pd.DataFrame,
                   rows: Sequence[dict]) -> pd.DataFrame:
    """`POINT_COLUMNS` slot values of both directions at the tested steps of `rows`."""
    tested = {r["step"] for r in rows if r["tested"]}
    parts = [d[d["step"].isin(tested)][["teacher", "cell", "step", "value"]].assign(
        direction=lab, **base) for lab, d in (("A->B", ab), ("B->A", ba))]
    return pd.concat(parts, ignore_index=True).reindex(columns=POINT_COLUMNS)


def _traj_status(n_runs: float, n_traj: float) -> str:
    """``none`` / ``partial k/n`` / ``full`` trajectory coverage of an arm's run universe;
    ``not read`` when the endpoint module already excluded the pair."""
    if pd.isna(n_runs) or pd.isna(n_traj):
        return "not read"
    if n_traj == 0:
        return "none"
    return "full" if n_traj == n_runs else f"partial {int(n_traj)}/{int(n_runs)}"


def _summary_row(base: dict, test: str, endpoint: pd.Series, steps: pd.DataFrame, *,
                 sign_ref: float | None = None) -> dict:
    """One `SUMMARY_COLUMNS` row from the pair's tested steps; the sign columns refer to the
    endpoint contrast, or to `sign_ref` when there is none (the stop design: its last step)."""
    t = steps[steps["tested"]].sort_values("step")
    last = t.iloc[-1]
    peak = t.loc[t["contrast"].abs().idxmax()]
    sig = t[t["p_two_sided"] < ALPHA]
    sign = np.sign(endpoint["contrast"] if sign_ref is None else sign_ref)
    diff = abs(float(endpoint["contrast"]) - float(last["contrast"]))
    cls = CLASS_PRELIM if (t["coverage class"] == CLASS_PRELIM).any() else CLASS_FULL
    return {**base, "coverage class": cls, "test": test,
            "endpoint contrast": float(endpoint["contrast"]),
            "endpoint p": float(endpoint["p_two_sided"]), "last tested step": int(last["step"]),
            "last-step contrast": float(last["contrast"]),
            "last-step p": float(last["p_two_sided"]), "|endpoint - last-step|": diff,
            "cross-check ok": bool(diff <= 1e-6) if np.isfinite(diff) else np.nan, "tested steps": len(t),
            "peak step": int(peak["step"]), "peak contrast": float(peak["contrast"]),
            "earliest p<.05 step": int(sig["step"].iloc[0]) if len(sig) else np.nan,
            "same-sign share": float((np.sign(t["contrast"]) == sign).mean()),
            "opposite-sign significant": bool((np.sign(sig["contrast"]) == -sign).any())}


def _stop_label(family: ta.Family, spec: tc.ArmSpec, repo: Path) -> str:
    """``{design} · checkpoints to N`` -- the design whose cells the backfill cells are, minus
    ``_stop{N}``."""
    base = {c.rsplit("_stop", 1)[0] for c in spec.cells}
    stops = sorted({int(c.rsplit("_stop", 1)[1]) for c in spec.cells if "_stop" in c})
    names = [lab for lab, cells in ta.family_designs(family, repo) if set(cells) == base]
    return ((names[0] if names else spec.cell_set) + BACKFILL_TAG
            + ("/".join(f"{n:,}" for n in stops) if stops else "end"))


#: `trajectory_conditions.default_arms` family whose arms save per-checkpoint evals on their
#: own (plain) cells, with matched per-step controls (`ArmSpec.control_of`).
OLMO_ARM_FAMILY = "OLMo"


def olmo_stop_arms(repo: Path, family: str = "olmo-recipe") -> list[tc.ArmSpec]:
    """The OLMo per-checkpoint arms (treatments + matched controls) of
    `trajectory_conditions.default_arms` whose sender AND receiver are distinct bases of `family`
    (dolma / dclm / c4 at the default seed). The seed-2 ``-aux2`` receivers run one direction
    only, so they never form a pair and are left out. Fed to `_stop_pairs` as extra stop arms:
    their cells carry no ``_stop{N}``, so the saved steps (625 ... 12,500) end at the run's own
    endpoint and the design is ``core35 · per-step control``."""
    bases = {tio.base_name(b) for b in ta.FAMILIES[family].bases}
    return [s for s in tc.default_arms(repo) if s.family == OLMO_ARM_FAMILY
            and s.sender in bases and s.receiver in bases and s.sender != s.receiver]


def _stop_points(spec: tc.ArmSpec, control: tc.ArmSpec | None, prior: float, root: Path,
                 ftseed: int) -> tuple[pd.DataFrame, dict]:
    """Per-step values of a stop arm, the matched control subtracted at the same step."""
    ctrl = _run_trajectories(control, tc.CONTROL_TEACHER, root, ftseed) if control else None
    cmap = ({(c, int(st)): p for c, st, p in ctrl[["tag", "step", "p_test"]].itertuples(
        index=False)} if ctrl is not None else {})
    parts, n_traj, n_runs = [], 0, len(spec.teachers) * len(spec.cells)
    for teacher in spec.teachers:
        tr = _run_trajectories(spec, teacher, root, ftseed)
        for cell, g in tr.groupby("tag"):
            if cell not in spec.cells or g["step"].nunique() <= 1:
                continue
            n_traj += 1
            st = g["step"].to_numpy(float).astype(int)
            c = np.array([cmap.get((cell, k), np.nan) for k in st])
            ok = np.isfinite(c)
            parts.append(pd.DataFrame({"sender": spec.sender, "teacher": teacher, "cell": cell,
                                       "step": st[ok],
                                       "value": (g["p_test"].to_numpy(float)[ok] - c[ok])
                                       / prior}))
    pts = (pd.concat(parts, ignore_index=True) if parts
           else pd.DataFrame(columns=["sender", "teacher", "cell", "step", "value"]))
    cvals = np.array(list(cmap.values()), float)
    ctrl_mean = float(np.nanmean(cvals)) if cvals.size else float("nan")
    n_ctrl = (int((ctrl.groupby("tag")["step"].nunique() > 1).sum()) if ctrl is not None
              and len(ctrl) else 0)
    info = {"control": CONTROL_PER_STEP, "runs (endpoint)": n_runs, "controls landed": n_ctrl,
            "runs with trajectory": n_traj, "saved steps": int(pts["step"].nunique()),
            "last step": int(pts["step"].max()) if len(pts) else np.nan,
            "control endpoint": ctrl_mean, "prior": prior, "control / prior": ctrl_mean / prior}
    return pts, info


def _stitched_points(stop_pts: pd.DataFrame, spec: tc.ArmSpec, arm: ta.ArmLoad,
                     ep20_traj: dict) -> tuple[pd.DataFrame, list[str], str]:
    """The stop arm's per-step points plus, per (teacher, cell) slot, the same (teacher, cell)
    ep20 run's endpoint value (`arm.frame`, already (student - matched control) / prior) at that
    run's last step. Cells lose their ``_stop{N}`` suffix (the join key). Returns ``(points,
    slots with no endpoint row, last-step source)``; `ep20_traj` is ``teacher -> {cell: last
    saved step}`` of the ep20 lineage."""
    pts = stop_pts.assign(cell=stop_pts["cell"].map(lambda c: c.rsplit("_stop", 1)[0]))
    ends = {(t, c): v for t, c, v in arm.frame[["teacher", "cell", "value"]].itertuples(
        index=False)}
    any_last = [st for d in ep20_traj.values() for st in d.values()]
    arm_last = max(any_last) if any_last else EP20_LAST_STEP
    source = ("ep20 trajectories" if any_last else
              f"ep20 schedule (no ep20 trajectory on disk; {EP20_LAST_STEP:,} = 20 x 1,250)")
    rows, missing = [], []
    for teacher in spec.teachers:
        for cell in (c.rsplit("_stop", 1)[0] for c in spec.cells):
            hits = [v for (t, c), v in ends.items() if t == teacher and c == cell]
            if len(hits) != 1:
                missing.append(f"{teacher}/{cell}")
                continue
            rows.append({"sender": spec.sender, "teacher": teacher, "cell": cell,
                         "step": int(ep20_traj.get(teacher, {}).get(cell, arm_last)),
                         "value": float(hits[0])})
    end = pd.DataFrame(rows, columns=["sender", "teacher", "cell", "step", "value"])
    # the endpoint row is the slot's last step: saved points at or past it are dropped
    last = {(t, c): st for t, c, st in end[["teacher", "cell", "step"]].itertuples(index=False)}
    keep = [st < last.get((t, c), np.inf) for t, c, st in
            pts[["teacher", "cell", "step"]].itertuples(index=False)]
    return pd.concat([pts[keep], end], ignore_index=True), missing, source


def _stop_pairs(family: ta.Family, repo: Path, root: Path, stop_arms: Sequence[tc.ArmSpec],
                endpoint: pd.DataFrame, priors: dict, *, n_perm: int, seed: int,
                prelim: bool = False, points: list | None = None,
                stats: bool = True, refresh_cache: bool = False) -> tuple[list, list, list]:
    """``(step rows, summary rows, inventory rows)`` of every backfilled cross pair whose two
    directions are both bases of `family` (A/B order = the family's base order).

    Each pair yields its stop design and, when the endpoint module tested the pair, the stitched
    ``<stop design> + ep20 endpoint`` design (`_stitched_points`, at the ep20 design `endpoint`
    used for the pair); the stop-only design is then marked ``superseded by`` the stitched one
    (``design status``) and kept for cross-checking only. `points`, when given, collects each
    design's tested-step slot values (`_tested_points`). Each pair is one `result_cache` unit
    (`_stop_pair`, keyed by `_stop_pair_records`)."""
    designs = dict(ta.family_designs(family, repo))
    treat = {(s.animal, s.sender, s.receiver): s for s in stop_arms
             if s.kind == "treatment" and s.sender != s.receiver}
    controls = {s.control_of: s for s in stop_arms if s.kind == "control"}
    steps, summary, inventory = [], [], []
    for animal in family.animals:
        for i, a in enumerate(family.bases):
            for b in family.bases[i + 1:]:
                da, db = tio.base_name(a), tio.base_name(b)
                ab, ba = treat.get((animal, da, db)), treat.get((animal, db, da))
                if ab is None or ba is None:
                    continue
                ep = endpoint[(endpoint["animal"] == animal) & (endpoint["A"] == a)
                              & (endpoint["B"] == b)]
                cab, cba = controls.get(ab.label), controls.get(ba.label)
                for spec in (ab, ba):
                    key = (animal, tio.base_name(spec.receiver))
                    if key not in priors:
                        priors[key] = tio.student_tree_prior(animal, spec.receiver,
                                                             output_root=root)

                def compute(animal=animal, a=a, b=b, ab=ab, ba=ba, ep=ep, cab=cab, cba=cba):
                    return _stop_pair(family, repo, root, designs, animal, a, b, ab, ba,
                                      cab, cba, ep, priors, n_perm=n_perm, seed=seed,
                                      prelim=prelim, stats=stats)

                def records(animal=animal, a=a, b=b, ab=ab, ba=ba, ep=ep, cab=cab, cba=cba):
                    return _stop_pair_records(family, repo, root, designs, animal, a, b,
                                              ab, ba, cab, cba, ep)

                params = {**_cache_params(family, n_perm=n_perm, seed=seed, prelim=prelim,
                                          stats=stats),
                          "unit": "stop", "animal": animal, "A": a, "B": b,
                          "specs": [repr(x) for x in (ab, ba, cab, cba)],
                          "endpoint": ep.to_dict("records"),
                          "ep design cells": (designs[ep["design"].iloc[0]] if len(ep)
                                              else None),
                          "priors": [priors[(animal, tio.base_name(x.receiver))]
                                     for x in (ab, ba)]}
                out = _cached_unit(root, f"{family.name}|{animal}|{a}|{b}|stop", records,
                                   params, compute, refresh_cache)
                steps.extend(out["steps"])
                summary.extend(out["summary"])
                inventory.extend(out["inventory"])
                if points is not None:
                    points.extend(out["points"])
    return steps, summary, inventory


def _stop_pair(family: ta.Family, repo: Path, root: Path, designs: dict, animal: str, a: str,
               b: str, ab: tc.ArmSpec, ba: tc.ArmSpec, cab: tc.ArmSpec | None,
               cba: tc.ArmSpec | None, ep: pd.DataFrame, priors: dict, *, n_perm: int,
               seed: int, prelim: bool, stats: bool) -> dict:
    """One backfilled pair of `_stop_pairs`: ``{"steps", "summary", "inventory", "points"}``
    lists of rows (points: frames)."""
    order = {tio.base_name(x): x for x in family.bases}
    controls = {ab.label: cab, ba.label: cba}
    steps, summary, inventory, points = [], [], [], []
    out = {"steps": steps, "summary": summary, "inventory": inventory, "points": points}
    design = _stop_label(family, ab, repo)
    prefix = design.split(BACKFILL_TAG, 1)[0]
    stitched = _CELLSET_NAMES.get(prefix, prefix) + TRAJECTORY_SUFFIX
    if not len(ep):     # no endpoint test: the saved steps ARE the trajectory
        design = stitched
    pts, infos = {}, {}
    for s, spec in ((a, ab), (b, ba)):
        key = (animal, tio.base_name(spec.receiver))
        if key not in priors:
            priors[key] = tio.student_tree_prior(animal, spec.receiver, output_root=root)
        pts[s], infos[s] = _stop_points(spec, controls.get(spec.label), priors[key],
                                        root, family.ftseed)
        pts[s] = pts[s].assign(sender=s)
        reason = ("no prior" if not np.isfinite(priors[key]) else
                  "" if infos[s]["runs with trajectory"] else NO_TRAJECTORY)
        inventory.append({"family": family.name, "animal": animal, "sender": s,
                          "receiver": order[spec.receiver] if spec.receiver in order
                          else spec.receiver, "design": design, **infos[s],
                          "reason": reason, "notes": ""})
    if any(r["reason"] for r in inventory[-2:]):
        return out
    base = {"family": family.name, "animal": animal, "A": a, "B": b, "design": design}
    paired = set(ab.teachers) == set(ba.teachers)
    test = ta.TEST_PAIRED if paired else ta.TEST_UNPAIRED
    rows = _step_rows(base, pts[a], pts[b], infos[a]["runs (endpoint)"],
                      infos[b]["runs (endpoint)"], paired=paired, n_perm=n_perm,
                      seed=seed, prelim=prelim, stats=stats)
    steps.extend(rows)
    points.append(_tested_points(base, pts[a], pts[b], rows))
    frame = pd.DataFrame(rows)
    stop_row = None
    if frame["tested"].any():
        nan = pd.Series({"contrast": np.nan, "p_two_sided": np.nan})
        stop_row = _summary_row(base, test, nan, frame, sign_ref=float(
            frame[frame["tested"]].sort_values("step")["contrast"].iloc[-1]))
        stop_row["ep20 endpoint contrast"] = (float(ep["contrast"].iloc[0])
                                              if len(ep) else np.nan)
        summary.append(stop_row)
    if not len(ep):
        return out
    # stitched: the stop points + the same (teacher, cell) ep20 run's endpoint
    ep_design = ep["design"].iloc[0]
    spts, sinfo, ends = {}, {}, {}
    for s, r, spec in ((a, b, ab), (b, a, ba)):
        arm = ta.load_arm(family, animal, s, r, ep_design, designs[ep_design],
                          repo=repo, output_root=root, priors=priors)
        ep_spec = _spec(family, animal, s, r, ep_design, designs[ep_design], arm.teachers)
        traj = {}
        for t in spec.teachers:
            tr = _run_trajectories(ep_spec, t, root, family.ftseed)
            lin = ep_spec.lineage_dir(root, t)
            traj[t] = {c: _sibling_only_end(lin, c, family.ftseed)
                       or int(g["step"].max()) for c, g in tr.groupby("tag")
                       if g["step"].nunique() > 1}
        p, missing, source = _stitched_points(pts[s], spec, arm, traj)
        spts[s] = p.assign(sender=s)
        ends[s] = (arm, missing)
        st_ = sorted(spts[s]["step"].unique())
        last = int(spts[s]["step"].max())
        sinfo[s] = {**infos[s], "control": CONTROL_PER_STEP,
                    "runs with trajectory": int(spts[s][["teacher", "cell"]]
                                                .drop_duplicates().shape[0]),
                    "saved steps": len(st_), "last step": last}
        note = (f"last step from {ep_design} ({source})"
                + (f"; {len(missing)} slot(s) without a last-step row: "
                   + ", ".join(missing) if missing else ""))
        inventory.append({"family": family.name, "animal": animal, "sender": s,
                          "receiver": r, "design": stitched, **sinfo[s],
                          "reason": "", "notes": note})
    if stop_row is not None:
        stop_row["design status"] = SUPERSEDED_BY + stitched
    for inv_row in inventory[-4:-2]:
        inv_row["design status"] = SUPERSEDED_BY + stitched
    sbase = {**base, "design": stitched}
    srows = _step_rows(sbase, spts[a], spts[b], infos[a]["runs (endpoint)"],
                       infos[b]["runs (endpoint)"], paired=paired, n_perm=n_perm,
                       seed=seed, prelim=prelim, stats=stats)
    steps.extend(srows)
    points.append(_tested_points(sbase, spts[a], spts[b], srows))
    sframe = pd.DataFrame(srows)
    if not sframe["tested"].any():
        return out
    # slots-matched endpoint: the endpoint frames restricted to the stop slots,
    # tested independently of the step table (must equal the last step exactly)
    sides = []
    for s in (a, b):
        arm, _ = ends[s]
        keys = set(map(tuple, spts[s][["teacher", "cell"]].to_numpy()))
        f = arm.frame[[k in keys for k in zip(arm.frame["teacher"], arm.frame["cell"])]]
        sides.append(f.assign(stratum=f["cell"]))
    cr = _contrast(ta._condition_frame(sides[0], sides[1], ta.CONDITION_A, ta.CONDITION_B),
                   paired=paired, n_perm=n_perm, seed=seed, stats=stats)
    row = _summary_row(sbase, test, pd.Series(cr), sframe)
    row["slots endpoint contrast"] = row["endpoint contrast"]
    row["slots endpoint p"] = row["endpoint p"]
    row["endpoint contrast"] = float(ep["contrast"].iloc[0])
    row["endpoint p"] = float(ep["p_two_sided"].iloc[0])
    row["ep20 endpoint contrast"] = float(ep["contrast"].iloc[0])
    summary.append(row)
    return out


# ----------------------------------------------------------------------------------------------
# result cache (`analysis.result_cache`): one unit per (family, animal, pair, design) plus one
# per family for the endpoint module; keys over the unit's record files + code + parameters
# ----------------------------------------------------------------------------------------------
#: `result_cache` namespace of this module's units.
CACHE_NAMESPACE = "trajectory_asymmetry"


def _cache_code() -> list:
    return [sys.modules[__name__], tc, ta, tio, traj_io]


_CODE_HASHES: list[str] = []


def _code_hashes() -> list[str]:
    """Source hashes of `_cache_code`, taken ONCE per process: an edit to a module file while a
    build runs must not change the keys of that build (the loaded code is still the old one)."""
    if not _CODE_HASHES:
        _CODE_HASHES.extend(result_cache._code_hash(c) for c in _cache_code())
    return _CODE_HASHES


def _cache_disabled() -> bool:
    return result_cache.RESULT_CACHE_DISABLED or os.environ.get("SL_RESULT_CACHE", "1") == "0"


def _cache_params(family: ta.Family, *, n_perm: int, seed: int, prelim: bool,
                  stats: bool) -> dict:
    """Parameters every unit of `family` depends on (module constants included)."""
    return {"family": repr(family), "n_perm": n_perm, "seed": seed, "prelim": prelim,
            "stats": stats, "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "alpha": ALPHA,
            "min_coverage": MIN_COVERAGE, "prelim_min_matched": PRELIM_MIN_MATCHED,
            "ep20_last_step": EP20_LAST_STEP}


def _cached_unit(root: Path, unit: str, records: Callable[[], list], params: dict,
                 compute: Callable[[], object], refresh: bool) -> object:
    """`result_cache.cached` of one unit; `records` (its ``(rel path, mtime_ns, size)``
    triples) is only built when the cache is enabled."""
    if _cache_disabled():
        return compute()
    recs = records()
    key = result_cache.fingerprint(recs, params={**params, "code": _code_hashes()}, code=[])
    return result_cache.cached(CACHE_NAMESPACE, unit, key, root, compute, refresh=refresh,
                               n_input_records=len(recs))


def _pref_traj_records(spec: tc.ArmSpec, teachers: Sequence[str], root: Path,
                       ftseed: int) -> list[tuple[str, int, int]]:
    """``(rel path, mtime_ns, size)`` of every ``eval/pref_traj.json`` `run_points` reads for
    `teachers` of `spec` (each cell and its `sibling_leaf`), from the trajectory record cache's
    post-sweep rows (no stat; `tc._prefetch` must have swept them in this process). A run
    without a record contributes ``(path, -1, -1)``."""
    from subliminal_learning.analysis.record_cache import load_record_cache

    fname = traj_io._KINDS["pref"][0]
    rels = [str(spec.lineage_dir(root, t).relative_to(root) / c / f"ftseed{ftseed}")
            for t in teachers
            for c in dict.fromkeys([*spec.cells, *map(tc.sibling_leaf, spec.cells)])]
    got = load_record_cache(root, "json_blob", rels, refresh=False, verbose=False,
                            rel_record=f"eval/{fname}",
                            cache_rel=Path("analysis") / "record_cache_traj_pref.csv")
    facts = {rp: (int(round(float(mt) * 1e9)), int(sz)) for rp, mt, sz in
             got[["run_path", "record_mtime", "record_size"]].itertuples(index=False)}
    return [(f"{rp}/eval/{fname}", *facts.get(rp, (-1, -1))) for rp in dict.fromkeys(rels)]


def _endpoint_records(family: ta.Family, animal: str, sender: str, receiver: str,
                      cells: Sequence[str], repo: Path, root: Path) -> list[tuple[str, int, int]]:
    """``(rel path, mtime_ns, size)`` of the endpoint value files `ta.load_arm` reads for one
    arm: exactly the leaves `tio.load_cell_matrix` signs (`tio._leaf_signature`, whose
    in-process memo makes the matching `load_cell_matrix` call stat-free)."""
    teachers = ta.family_teachers(family, animal, sender, repo)
    receiver_dir, sender_dir = tio.base_name(receiver), f"teacher-{tio.base_name(sender)}"
    control_tree = tio.dataset_dir(root, None, base=receiver_dir,
                                   dataset_tag=tio.DEFAULT_DATASET_TAG,
                                   source=family.control_source,
                                   numbers_base=tio.base_name(sender))
    seed_part = f"ftseed{family.ftseed}"
    value_leaf = tio._VALUE_LEAVES[0]
    sender_tree = (root / "students" / animal
                   / Path(*tio.model_dir_parts_from_name(receiver_dir)) / sender_dir)
    leaves: list[Path] = []
    for cell in tio.resolve_cells(list(cells)):
        leaves += tio._run_leaves(control_tree / cell / seed_part, (value_leaf,), animal)
        leaves += [sender_tree / f"{tag}_tseed{family.tseed}" / tio.DEFAULT_DATASET_TAG / cell
                   / seed_part / value_leaf for tag in teachers]
    return [(r[0], -1, -1) if r[1] is None else (r[0], int(r[2]), int(r[1]))
            for r in tio._leaf_signature(root, leaves)]


def _ep20_pair_records(family: ta.Family, animal: str, a: str, b: str, design: str,
                       cells: Sequence[str], repo: Path, root: Path) -> list:
    """Every record file one ep20 pair unit reads: both directions' endpoint values and
    matched controls (`_endpoint_records`) and their designed runs' trajectories."""
    out = []
    for s, r in ((a, b), (b, a)):
        teachers = ta.family_teachers(family, animal, s, repo)
        out += _endpoint_records(family, animal, s, r, cells, repo, root)
        out += _pref_traj_records(_spec(family, animal, s, r, design, cells, teachers),
                                  teachers, root, family.ftseed)
    return out


def _stop_pair_records(family: ta.Family, repo: Path, root: Path, designs: dict, animal: str,
                       a: str, b: str, ab: tc.ArmSpec, ba: tc.ArmSpec, cab: tc.ArmSpec | None,
                       cba: tc.ArmSpec | None, ep: pd.DataFrame) -> list:
    """Every record file one backfilled pair unit reads: both stop arms' and their matched
    controls' trajectories and, with an endpoint test, the ep20 design's endpoint records and
    trajectories (`_ep20_pair_records`)."""
    out = []
    for spec, ctrl in ((ab, cab), (ba, cba)):
        out += _pref_traj_records(spec, spec.teachers, root, family.ftseed)
        if ctrl is not None:
            out += _pref_traj_records(ctrl, [tc.CONTROL_TEACHER], root, family.ftseed)
    if len(ep):
        d = ep["design"].iloc[0]
        out += _ep20_pair_records(family, animal, a, b, d, designs[d], repo, root)
    return out


def _ep20_pair(family: ta.Family, pr: pd.Series, cells: Sequence[str], repo: Path, root: Path,
               priors: dict, *, n_perm: int, seed: int, stats: bool) -> dict:
    """One endpoint-tested pair of `analyze_family` at its ep20 design: ``{"steps",
    "summary", "inventory", "excluded", "points"}`` lists of rows (points: frames)."""
    animal, a, b, design = pr["animal"], pr["A"], pr["B"], pr["design"]
    base = {"family": family.name, "animal": animal, "A": a, "B": b, "design": design}
    out = {"steps": [], "summary": [], "inventory": [], "excluded": [], "points": []}
    loads, pts, infos = {}, {}, {}
    for s, r in ((a, b), (b, a)):
        arm = ta.load_arm(family, animal, s, r, design, cells, repo=repo, output_root=root,
                          priors=priors)
        loads[s] = arm
        pts[s], infos[s] = _direction_points(family, animal, s, r, design, cells, arm, root)
    notes = ""
    if infos[a]["runs with trajectory"] and infos[b]["runs with trajectory"] \
            and infos[a]["last step"] != infos[b]["last step"]:
        notes = (f"within-pair last-step mismatch: {a}->{b} {infos[a]['last step']}, "
                 f"{b}->{a} {infos[b]['last step']}")
    reasons = {s: (f"{NO_TRAJECTORY} (0/{infos[s]['runs (endpoint)']} runs)"
                   if not infos[s]["runs with trajectory"] else "") for s in (a, b)}
    for s, r in ((a, b), (b, a)):
        out["inventory"].append({"family": family.name, "animal": animal, "sender": s,
                                 "receiver": r, "design": design, **infos[s],
                                 "reason": reasons[s], "notes": notes})
    if reasons[a] or reasons[b]:
        out["excluded"].append({**{k: base[k] for k in ("family", "animal", "A", "B")},
                                "reason A->B": reasons[a] or "has trajectory",
                                "reason B->A": reasons[b] or "has trajectory"})
        return out
    paired = set(loads[a].teachers) == set(loads[b].teachers)
    rows = _step_rows(base, pts[a], pts[b], infos[a]["runs (endpoint)"],
                      infos[b]["runs (endpoint)"], paired=paired, n_perm=n_perm, seed=seed,
                      stats=stats)
    out["steps"].extend(rows)
    out["points"].append(_tested_points(base, pts[a], pts[b], rows))
    frame = pd.DataFrame(rows)
    if not frame["tested"].any():
        out["excluded"].append({**{k: base[k] for k in ("family", "animal", "A", "B")},
                                "reason A->B": NO_COVERED_STEP, "reason B->A": NO_COVERED_STEP})
        return out
    out["summary"].append(_summary_row(base, pr["test"], pr, frame))
    return out


def _endpoint_family(family: ta.Family, repo: Path, root: Path, *, n_perm: int, seed: int,
                     refresh_cache: bool) -> ta.AsymmetryResult:
    """`transfer_asymmetry.analyze_family`, one `result_cache` unit per family keyed by every
    arm's endpoint records at every design."""
    designs = ta.family_designs(family, repo)

    def records() -> list:
        return [rec for animal in family.animals for _, cells in designs
                for s in family.bases for r in family.bases if s != r
                for rec in _endpoint_records(family, animal, s, r, cells, repo, root)]

    priors = sorted({(animal, tio.base_name(r)): tio.student_tree_prior(
        animal, r, output_root=root) for animal in family.animals for r in family.bases}.items())
    params = {"unit": "endpoint", "family": repr(family), "n_perm": n_perm, "seed": seed,
              "designs": designs, "priors": priors,
              "teachers": [ta.family_teachers(family, animal, s, repo)
                           for animal in family.animals for s in family.bases]}
    return _cached_unit(root, f"{family.name}|endpoint", records, params,
                        lambda: ta.analyze_family(family, repo=repo, output_root=root,
                                                  n_perm=n_perm, seed=seed), refresh_cache)


def analyze_family(family: ta.Family, *, repo: Path, output_root: Path | None = None,
                   n_perm: int = 20000, seed: int = 0,
                   stop_arms: Sequence[tc.ArmSpec] = (),
                   prelim: bool = False, prefetched: bool = False,
                   stats: bool = True, refresh_cache: bool = False) -> TrajectoryAsymmetry:
    """The per-step, summary, inventory and exclusion tables of one family.

    The pair universe, designs and exclusions come from `transfer_asymmetry.analyze_family`
    (prior-scaled values); each tested pair's two directions are re-read with
    `transfer_asymmetry.load_arm` at the pair's design (memoized loads) for their run universe,
    teachers and prior. `stop_arms` (`trajectory_conditions.earlystop_arms`) adds the
    backfilled cross pairs inside the family's bases as extra designs (per-step control);
    `prelim` applies the `PRELIM_MIN_MATCHED` rule to those stop designs only.

    Trajectories are read from the record cache without per-lineage refreshes, so the family's
    arms (+ `stop_arms`) are prefetched in one sweep first unless `prefetched` says the caller
    already did so in this process over a superset of them (`analyze`). With `stats` off the
    trajectory module runs no permutation test (p columns NaN; contrasts and bootstrap bands
    unchanged); the endpoint module's own tests (`transfer_asymmetry.analyze_family`) still run.

    Result cache. Each endpoint-tested pair (`_ep20_pair`), each backfilled pair (`_stop_pair`)
    and the family's endpoint call (`_endpoint_family`) is one `analysis.result_cache` unit
    (namespace `CACHE_NAMESPACE`) keyed by the ``(path, mtime_ns, size)`` of every record file
    it reads (trajectories from the post-sweep record cache, endpoint values and matched
    controls via `tio._leaf_signature`), the source of this module and the modules it reads
    through, and every parameter. A hit returns the stored rows unchanged; `refresh_cache`
    recomputes and overwrites; ``SL_RESULT_CACHE=0`` restores the uncached path.
    """
    root = resolve_output_root(output_root)
    if not prefetched:
        tc._prefetch(_family_specs(family, repo) + list(stop_arms), root, family.ftseed)
    end = _endpoint_family(family, repo, root, n_perm=n_perm, seed=seed,
                           refresh_cache=refresh_cache)
    designs = dict(ta.family_designs(family, repo))
    priors: dict = {}
    steps, summary, inventory, excluded, points = [], [], [], [], []
    for _, ex in end.excluded.iterrows():
        excluded.append({k: ex[k] for k in ta.EXCLUDED_COLUMNS})
        for (s, r), reason in (((ex["A"], ex["B"]), ex["reason A->B"]),
                               ((ex["B"], ex["A"]), ex["reason B->A"])):
            inventory.append({"family": family.name, "animal": ex["animal"], "sender": s,
                              "receiver": r, "design": "", "reason": f"endpoint: {reason}"})
    common = _cache_params(family, n_perm=n_perm, seed=seed, prelim=prelim, stats=stats)
    for _, pr in end.pairs.iterrows():
        animal, a, b, design = pr["animal"], pr["A"], pr["B"], pr["design"]
        cells = designs[design]
        params = {**common, "unit": "ep20", "pair": pr.to_dict(), "cells": cells,
                  "priors": [tio.student_tree_prior(animal, x, output_root=root) for x in (a, b)]}
        out = _cached_unit(
            root, f"{family.name}|{animal}|{a}|{b}|{design}",
            lambda animal=animal, a=a, b=b, design=design, cells=cells: _ep20_pair_records(
                family, animal, a, b, design, cells, repo, root),
            params,
            lambda pr=pr, cells=cells: _ep20_pair(family, pr, cells, repo, root, priors,
                                                  n_perm=n_perm, seed=seed, stats=stats),
            refresh_cache)
        steps.extend(out["steps"])
        summary.extend(out["summary"])
        inventory.extend(out["inventory"])
        excluded.extend(out["excluded"])
        points.extend(out["points"])
    if stop_arms:
        st, su, inv = _stop_pairs(family, repo, root, stop_arms, end.pairs, priors,
                                  n_perm=n_perm, seed=seed, prelim=prelim, points=points,
                                  stats=stats, refresh_cache=refresh_cache)
        steps.extend(st)
        summary.extend(su)
        inventory.extend(inv)
    stop_designs = {(r["animal"], r["A"], r["B"]): r["design"] for r in summary
                    if not _is_ep20(r["design"]) and not r.get("design status")}
    for ex in excluded:
        ex["superseded by"] = stop_designs.get((ex["animal"], ex["A"], ex["B"]), "")
    inv = pd.DataFrame(inventory).reindex(columns=INVENTORY_COLUMNS)
    inv["trajectory"] = [_traj_status(n, k) for n, k in
                         zip(inv["runs (endpoint)"], inv["runs with trajectory"])]
    return TrajectoryAsymmetry(
        steps=pd.DataFrame(steps).reindex(columns=STEP_COLUMNS),
        summary=pd.DataFrame(summary).reindex(columns=SUMMARY_COLUMNS),
        inventory=inv.reindex(columns=INVENTORY_COLUMNS),
        excluded=pd.DataFrame(excluded).reindex(columns=EXCLUDED_COLUMNS),
        endpoint=end.pairs,
        points=(pd.concat(points, ignore_index=True) if points
                else pd.DataFrame(columns=POINT_COLUMNS)).reindex(columns=POINT_COLUMNS))


def _family_specs(fam: ta.Family, repo: Path) -> list[tc.ArmSpec]:
    """Every ordered (sender, receiver) arm of `fam` at every design and animal."""
    specs = []
    for label, cells in ta.family_designs(fam, repo):
        for animal in fam.animals:
            for s in fam.bases:
                for r in fam.bases:
                    if s != r:
                        specs.append(_spec(fam, animal, s, r, label, cells,
                                           ta.family_teachers(fam, animal, s, repo)))
    return specs


def analyze(names: Sequence[str] = FAMILY_ORDER, *, repo: Path,
            output_root: Path | None = None, n_perm: int = 20000, seed: int = 0,
            early_stop: bool = True, prelim: bool = False,
            stats: bool = True, refresh_cache: bool = False) -> TrajectoryAsymmetry:
    """`analyze_family` over the named `transfer_asymmetry.FAMILIES`, tables concatenated.

    The shared trajectory record caches are refreshed once up front
    (`trajectory_conditions._prefetch`) over every ordered arm at every design. `early_stop`
    adds the `trajectory_conditions.earlystop_arms` cross pairs (see `analyze_family`). That
    sweep is also what supplies the record mtimes the result-cache keys use; `refresh_cache`
    bypasses the result cache (see `analyze_family`).
    """
    root = resolve_output_root(output_root)
    specs = [sp for n in names for sp in _family_specs(ta.FAMILIES[n], repo)]
    stop = tc.earlystop_arms(repo) + olmo_stop_arms(repo) if early_stop else []
    tc._prefetch(specs + list(stop), root, 1)
    parts = [analyze_family(ta.FAMILIES[n], repo=repo, output_root=root, n_perm=n_perm,
                            seed=seed, stop_arms=stop, prelim=prelim,
                            prefetched=ta.FAMILIES[n].ftseed == 1, stats=stats,
                            refresh_cache=refresh_cache)
             for n in names]

    def cat(attr: str, columns: list[str] | None) -> pd.DataFrame:
        frames = [getattr(p, attr) for p in parts if len(getattr(p, attr))]
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        return out if columns is None else out.reindex(columns=columns)

    return TrajectoryAsymmetry(cat("steps", STEP_COLUMNS), cat("summary", SUMMARY_COLUMNS),
                               cat("inventory", INVENTORY_COLUMNS),
                               cat("excluded", EXCLUDED_COLUMNS), cat("endpoint", None),
                               cat("points", POINT_COLUMNS))


def direction_table(stop_arms: Sequence[tc.ArmSpec], *, output_root: Path | None = None,
                         ftseed: int = 1,
                         prefetched: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Direction-only readout of the backfilled dog cross arms, plus a landing table.

    Per (cross arm, step): n / mean / median of ``(student - matched control at the same step) /
    receiver prior`` over the runs landed, its 95% teacher-cluster bootstrap band
    (``mean lo`` / ``mean hi``, see `N_BOOT`), beside the same numbers for the SENDER's and
    the RECEIVER's same-model stop diagonals at that step (e.g. orig->fs1 beside orig->orig and
    fs1->fs1). ``reverse landed`` says whether the reverse arm has any landed run. The landing
    table counts, per treatment arm, runs with a trajectory out of the designed teacher x cell
    runs and control cells with a trajectory out of the designed cells.

    The record caches are swept once over `stop_arms` first (`trajectory_conditions._prefetch`)
    unless `prefetched` says the caller already did so in this process at `ftseed`.
    """
    root = resolve_output_root(output_root)
    if not prefetched:
        tc._prefetch(stop_arms, root, ftseed)
    controls = {s.control_of: s for s in stop_arms if s.kind == "control"}
    treat = [s for s in stop_arms if s.kind == "treatment"]
    by_key = {(s.animal, s.sender, s.receiver): s for s in treat}
    pts, landing = {}, []
    for spec in treat:
        prior = tio.student_tree_prior(spec.animal, spec.receiver, output_root=root)
        p, info = _stop_points(spec, controls.get(spec.label), prior, root, ftseed)
        pts[spec.label] = p
        n, nd = info["runs with trajectory"], info["runs (endpoint)"]
        landing.append({"animal": spec.animal, "group": spec.group, "arm": spec.label,
                        "runs landed": n, "designed runs": nd,
                        "controls landed": info["controls landed"],
                        "designed controls": len(spec.cells),
                        "status": "complete" if n == nd else "partial" if n else "not started"})

    boots = {}
    for label, p in pts.items():
        teachers = sorted(set(p["teacher"]))
        boots[label] = _boot_means(p, teachers, _boot_weights(len(teachers))) if teachers else {}

    def stats(label: str | None, step: int) -> tuple[int, float, float]:
        if label is None or label not in pts:
            return 0, np.nan, np.nan
        v = pts[label].loc[pts[label]["step"] == step, "value"]
        return (len(v), float(v.mean()), float(v.median())) if len(v) else (0, np.nan, np.nan)

    rows = []
    for spec in treat:
        if spec.sender == spec.receiver:
            continue
        rev = by_key.get((spec.animal, spec.receiver, spec.sender))
        sdiag = by_key.get((spec.animal, spec.sender, spec.sender))
        rdiag = by_key.get((spec.animal, spec.receiver, spec.receiver))
        for step in sorted(pts[spec.label]["step"].unique()):
            n, m, md = stats(spec.label, step)
            sn, sm, smd = stats(sdiag.label if sdiag else None, step)
            rn, rm, rmd = stats(rdiag.label if rdiag else None, step)
            lo, hi = _band(boots[spec.label].get(int(step)))
            rows.append({"animal": spec.animal, "arm": spec.label, "sender": spec.sender,
                         "receiver": spec.receiver,
                         "reverse landed": bool(rev is not None and len(pts[rev.label])),
                         "step": int(step), "n": n,
                         "coverage": n / max(len(spec.teachers) * len(spec.cells), 1),
                         "mean": m, "median": md,
                         "mean lo": lo, "mean hi": hi,
                         "sender diag n": sn, "sender diag mean": sm, "sender diag median": smd,
                         "receiver diag n": rn, "receiver diag mean": rm,
                         "receiver diag median": rmd})
    return (pd.DataFrame(rows).reindex(columns=DIRECTION_COLUMNS),
            pd.DataFrame(landing).reindex(columns=LANDING_COLUMNS))


# ----------------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------------
_UNITS = "receiver prior units, (student − control) / prior"
_BAND = "95% teacher-cluster bootstrap band"


EP20_EARLY_NOTE = ("ep20 pairs: early-step significance reflects the endpoint-constant control, "
                   "not transfer")


def _active(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows whose design is not ``superseded by`` another (the designs the figures draw)."""
    if "design status" not in frame:
        return frame
    st = frame["design status"]
    return frame[st.isna() | (st == "")]


def _is_ep20(design: str) -> bool:
    """An ep20 design (endpoint-constant control at every step)."""
    return BACKFILL_TAG not in str(design) and not str(design).endswith(TRAJECTORY_SUFFIX)


def _step_axes(ax, ylabel: str) -> None:
    """`trajectory_conditions._log_axes` for x (absolute steps, log); y stays linear because
    values and contrasts change sign."""
    tc._log_axes(ax, ylabel)
    ax.set_yscale("linear")
    ax.axhline(0, color="0.5", lw=0.8)


CURVE_NOTE = f"curves drawn at steps with ≥ {MIN_COVERAGE:.0%} run coverage"


def curve_pairs(summary: pd.DataFrame) -> pd.DataFrame:
    """The `_active` summary rows the curve figures draw: ONE design per (family, animal, A, B),
    its per-step-control design when it has one, else its ep20 design."""
    act = _active(summary)
    if act.empty:
        return act
    act = act.assign(_ep20=[_is_ep20(d) for d in act["design"]])
    keep = act.groupby(["family", "animal", "A", "B"], sort=False)["_ep20"].transform("min")
    return act[act["_ep20"] == keep].drop(columns="_ep20")


def _covered(st: pd.DataFrame, cov_col: str) -> pd.DataFrame:
    """Rows of a direction's steps where it carries ≥ `MIN_COVERAGE` of its run universe."""
    return st[st[cov_col].astype(float) >= MIN_COVERAGE]


def plot_direction_curves(res: TrajectoryAsymmetry, *, ncols: int = 4, fontsize: int = 9):
    """One panel per pair (`curve_pairs`): per-step mean value of A→B (solid) and B→A (dashed),
    each drawn only at steps where that direction has ≥ `MIN_COVERAGE` run coverage."""
    import matplotlib.pyplot as plt

    keys = list(curve_pairs(res.summary)[["family", "animal", "A", "B", "design",
                                          "coverage class"]].itertuples(index=False, name=None))
    nrows = max(1, -(-len(keys) // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False,
                             layout="constrained")
    for ax, (fam, animal, a, b, design, cls) in zip(axes.flat, keys):
        st = res.steps[(res.steps["family"] == fam) & (res.steps["animal"] == animal)
                       & (res.steps["A"] == a) & (res.steps["B"] == b)
                       & (res.steps["design"] == design)].sort_values("step")
        sa, sb = pair_label(a, b).split("⇄")
        for col, ls, c, lab in (("mean A->B", "-", "#0072B2", f"{sa}→{sb}"),
                                ("mean B->A", "--", "#D55E00", f"{sb}→{sa}")):
            sc = _covered(st, col.replace("mean", "coverage"))
            ax.plot(sc["step"], sc[col], ls, color=c, label=lab)
            ax.fill_between(sc["step"], sc[f"{col} lo"], sc[f"{col} hi"], color=c, alpha=0.2,
                            lw=0)
        _step_axes(ax, "mean value (receiver prior units)")
        ax.set_title(f"{family_display(fam)} · {animal} · {sa}⇄{sb}\n{design}"
                     + (" · PRELIM (partial runs)" if cls == CLASS_PRELIM else ""),
                     fontsize=fontsize)
        ax.legend(fontsize=fontsize - 1, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.22), ncol=2)
        ax.tick_params(labelsize=fontsize - 1)
        ax.xaxis.label.set_size(fontsize - 1)
        ax.yaxis.label.set_size(fontsize - 1)
    for ax in list(axes.flat)[len(keys):]:
        ax.set_visible(False)
    fig.suptitle("Per-step mean transfer of each direction "
                 f"({_UNITS}; {CURVE_NOTE})\nshading = {_BAND}", fontsize=fontsize + 2)
    return fig


def plot_contrast_curves(res: TrajectoryAsymmetry, *, ncols: int = 3, fontsize: int = 9):
    """One panel per (family, animal): A→B − B→A contrast vs step, one line per pair; filled
    markers where p_two_sided < `ALPHA`, hollow otherwise. Each panel has its own y scale."""
    import matplotlib.pyplot as plt

    summ = curve_pairs(res.summary)
    groups = list(dict.fromkeys(summ[["family", "animal"]].itertuples(index=False, name=None)))
    nrows = max(1, -(-len(groups) // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.4 * nrows), squeeze=False,
                             layout="constrained")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (fam, animal) in zip(axes.flat, groups):
        pairs = summ[(summ["family"] == fam) & (summ["animal"] == animal)]
        for k, (_, pr) in enumerate(pairs.iterrows()):
            st = res.steps[(res.steps["family"] == fam) & (res.steps["animal"] == animal)
                           & (res.steps["A"] == pr["A"]) & (res.steps["B"] == pr["B"])
                           & (res.steps["design"] == pr["design"])
                           & res.steps["tested"]].sort_values("step")
            c = colors[k % len(colors)]
            stop = not _is_ep20(pr["design"])
            ax.plot(st["step"], st["contrast"], ":" if stop else "-", color=c, lw=1.2,
                    label=f"{pair_label(pr['A'], pr['B']).replace('⇄', '→')} − reverse"
                    + (f" ({pr['design']})" if stop else "")
                    + (" PRELIM" if pr["coverage class"] == CLASS_PRELIM else ""))
            sig = st["p_two_sided"] < ALPHA
            ax.fill_between(st["step"], st["contrast lo"], st["contrast hi"], color=c,
                            alpha=0.2, lw=0)
            ax.scatter(st["step"][sig], st["contrast"][sig], color=c, s=18, zorder=3)
            ax.scatter(st["step"][~sig], st["contrast"][~sig], facecolors="none",
                       edgecolors=c, s=18, zorder=3)
        _step_axes(ax, "A→B − B→A (receiver prior units)")
        ax.set_title(f"{family_display(fam)} · {animal}", fontsize=fontsize + 1)
        ax.legend(fontsize=fontsize - 1, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.2), ncol=1)
        ax.tick_params(labelsize=fontsize - 1)
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    fig.suptitle("Transfer asymmetry along training: per-step contrast A→B − B→A "
                 f"(tested steps only; {_UNITS})\nmarkers = paired teacher-cluster permutation "
                 f"test (filled p < {ALPHA}, hollow n.s.); shading = {_BAND}; dotted = "
                 "per-step-control design", fontsize=fontsize + 2)
    return fig


#: The dog blocks of `plot_contrast_curves_dog`, in display order (the
#: `trajectory_ab_vs_aa.BASE_GROUPS` order): (title, member bases as `pair_label` names). A pair
#: belongs to a block when BOTH its bases are members; the re-pretrain family splits into the
#: last two.
DOG_CONTRAST_BLOCKS = (
    ("diff init, diff data order: orig, fs1–fs3", ("orig", "fs1", "fs2", "fs3")),
    ("same init, diff data order: orig, ds1–ds3", ("orig", "ds1", "ds2", "ds3")),
    ("diff init, same data order: Rerun1234, wc1, wc2", ("Rerun1234", "wc1", "wc2")),
    ("Pythia-160M re-pretrain: orig, Modern1234, Rerun1234",
     ("orig", "Modern1234", "Rerun1234")),
)
DP_YLABEL = "mean A→B − B→A, Δp(trait)"
#: `dp_contrast_table` / ``asymmetry_trajectory_contrast.csv``: A, B in display form.
DP_COLUMNS = ["animal", "A", "B", "block", "design", "step", "n paired slots", "contrast",
              "contrast lo", "contrast hi", "p_two_sided", "tested"]
#: `steps_csv_table` / ``asymmetry_trajectory_steps.csv`` (receiver prior units).
STEPS_CSV_COLUMNS = ["family", "animal", "A", "B", "design", "step", "tested", "mean A->B",
                     "mean B->A", "contrast", "contrast lo", "contrast hi", "p_two_sided"]


def contrast_block(a: str, b: str) -> str | None:
    """The `DOG_CONTRAST_BLOCKS` title holding both bases of pair (a, b), else None."""
    sa, sb = pair_label(a, b).split("⇄")
    return next((t for t, m in DOG_CONTRAST_BLOCKS if sa in m and sb in m), None)


def dp_contrast_table(res: TrajectoryAsymmetry, *, animal: str = "dog", n_perm: int = 20000,
                      seed: int = 0, stats: bool = True) -> pd.DataFrame:
    """`DP_COLUMNS` per tested step of every per-step-control `curve_pairs` pair of `animal` in a
    `DOG_CONTRAST_BLOCKS` block, in Δp: each retained slot value (`res.points`, (p_student −
    p_control at the same step) / prior) times its direction's receiver prior (`res.inventory`),
    then the same paired/unpaired `contrast_row` and teacher-cluster bootstrap band as
    `_step_rows`. ``n paired slots`` = (teacher, cell) slots present in both directions. Not a rescaling of the prior-unit contrast: the two receivers' priors
    differ."""
    summ = curve_pairs(res.summary)
    summ = summ[(summ["animal"] == animal) & ~summ["design"].map(_is_ep20)]
    inv = res.inventory
    rows = []
    for _, pr in summ.iterrows():
        block = contrast_block(pr["A"], pr["B"])
        if block is None:
            continue
        key = {k: pr[k] for k in ("family", "animal", "A", "B", "design")}
        da, db = pair_label(pr["A"], pr["B"]).split("⇄")
        pts = res.points
        pts = pts[np.logical_and.reduce([pts[k] == v for k, v in key.items()])]
        sides = {}
        for d, s, r in (("A->B", pr["A"], pr["B"]), ("B->A", pr["B"], pr["A"])):
            iv = inv[(inv["family"] == pr["family"]) & (inv["animal"] == animal)
                     & (inv["sender"] == s) & (inv["receiver"] == r)
                     & (inv["design"] == pr["design"])]
            prior = float(iv["prior"].iloc[0])
            p = pts[pts["direction"] == d]
            sides[d] = p.assign(value=prior * p["value"].astype(float), sender=s)
        ab, ba = sides["A->B"], sides["B->A"]
        teachers = sorted(set(ab["teacher"]) | set(ba["teacher"]))
        paired = (pr["test"] == ta.TEST_PAIRED if isinstance(pr.get("test"), str)
                  else set(ab["teacher"]) == set(ba["teacher"]))
        w = _boot_weights(len(teachers))
        boot_a, boot_b = _boot_means(ab, teachers, w), _boot_means(ba, teachers, w)
        for step in sorted(set(ab["step"]) & set(ba["step"])):
            a, b = ab[ab["step"] == step], ba[ba["step"] == step]
            frame = ta._condition_frame(a.assign(stratum=a["cell"]), b.assign(stratum=b["cell"]),
                                        ta.CONDITION_A, ta.CONDITION_B)
            r = _contrast(frame, paired=paired, n_perm=n_perm, seed=seed, stats=stats)
            lo, hi = _band(boot_a[int(step)] - boot_b[int(step)])
            n = len(set(zip(a["teacher"], a["cell"])) & set(zip(b["teacher"], b["cell"])))
            rows.append({"animal": animal, "A": da, "B": db, "block": block,
                         "design": pr["design"], "step": int(step), "n paired slots": n,
                         "contrast": r["contrast"], "contrast lo": lo, "contrast hi": hi,
                         "p_two_sided": r["p_two_sided"], "tested": True})
    return pd.DataFrame(rows, columns=DP_COLUMNS)


def steps_csv_table(res: TrajectoryAsymmetry) -> pd.DataFrame:
    """`res.steps` in `STEPS_CSV_COLUMNS` (receiver prior units), family and bases in display
    form."""
    st = res.steps.reindex(columns=STEPS_CSV_COLUMNS).copy()
    ab = [pair_label(a, b).split("⇄") for a, b in zip(st["A"], st["B"])]
    st["A"], st["B"] = [x[0] for x in ab], [x[1] for x in ab]
    st["family"] = st["family"].map(family_display)
    return st


def _linear_step_axes(ax, ylabel: str) -> None:
    """Linear student-step x axis, ticks every 5,000 with thousands separators; zero line."""
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    ax.xaxis.set_major_locator(MultipleLocator(5000))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlim(left=0)
    ax.set_xlabel("student step")
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="0.5", lw=0.8)


#: Every panel of `plot_contrast_curves_dog` shows at least this range on both sides of zero.
DP_YFLOOR = 0.05


def plot_contrast_curves_dog(dp: pd.DataFrame, *, fontsize: int = 9):
    """2×2 dog figure, one panel per `DOG_CONTRAST_BLOCKS` block: the per-step-control A→B − B→A
    contrast in Δp, from a `dp_contrast_table` frame or its CSV (`DP_COLUMNS`, display-form
    bases), one solid line per pair labelled ``A→B``, 95% bootstrap band, filled marker p < `ALPHA`, hollow otherwise;
    linear step axis; legend inside the axes; no suptitle."""
    import matplotlib.pyplot as plt

    stats = bool(dp["p_two_sided"].notna().any())
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.6), squeeze=False, layout="constrained")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, (title, _) in zip(axes.flat, DOG_CONTRAST_BLOCKS):
        blk = dp[dp["block"] == title]
        for k, ((a, b), st) in enumerate(blk.groupby(["A", "B"], sort=False)):
            st = st.sort_values("step")
            c = colors[k % len(colors)]
            ax.plot(st["step"], st["contrast"], "-", color=c, lw=1.2,
                    label=f"{a}→{b}")
            ax.fill_between(st["step"], st["contrast lo"], st["contrast hi"], color=c,
                            alpha=0.2, lw=0)
            if not stats:
                ax.scatter(st["step"], st["contrast"], color=c, s=10, zorder=3)
                continue
            sig = st["p_two_sided"] < ALPHA
            ax.scatter(st["step"][sig], st["contrast"][sig], color=c, s=18, zorder=3)
            ax.scatter(st["step"][~sig], st["contrast"][~sig], facecolors="none",
                       edgecolors=c, s=18, zorder=3)
        _linear_step_axes(ax, DP_YLABEL)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(lo, -DP_YFLOOR), max(hi, DP_YFLOOR))
        ax.set_title(f"dog · {title}", fontsize=fontsize + 1)
        if len(blk):
            ax.legend(fontsize=fontsize - 2, loc="best", framealpha=0.6)
        ax.tick_params(labelsize=fontsize - 1)
    return fig


def summary_display(res: TrajectoryAsymmetry) -> pd.DataFrame:
    """The compact per-pair table of the drawn designs: endpoint contrast and p (the endpoint
    module's, on `endpoint cells`), the last tested step's contrast and p (on `last-step cells`),
    the earliest p < `ALPHA` step and the same-sign share."""
    ep = res.endpoint.set_index(["family", "animal", "A", "B"]) if len(res.endpoint) else None
    rows = []
    for _, r in _active(res.summary).iterrows():
        c, p = r["endpoint contrast"], r["endpoint p"]
        if not np.isfinite(c) and ep is not None and (key := (r["family"], r["animal"], r["A"],
                                                             r["B"])) in ep.index:
            c, p = float(ep.loc[key, "contrast"]), float(ep.loc[key, "p_two_sided"])
        key = (r["family"], r["animal"], r["A"], r["B"])
        ecells = (str(ep.loc[key, "design"]) if ep is not None and key in ep.index
                  else str(r["design"]))
        rows.append({"family": r["family"], "animal": r["animal"], "A": r["A"], "B": r["B"],
                     "design": r["design"], "endpoint contrast": c, "endpoint p": p,
                     "endpoint cells": ecells,
                     "last-step cells": str(r["design"]).removesuffix(TRAJECTORY_SUFFIX),
                     "last tested step": r["last tested step"],
                     "last-step contrast": r["last-step contrast"],
                     "last-step p": r["last-step p"],
                     "earliest p<.05 step": r["earliest p<.05 step"],
                     "same-sign share": r["same-sign share"]})
    return pd.DataFrame(rows)


PAIR_KEY = ["family", "animal", "A", "B"]


def _per_step_control(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows of the per-step-control designs (label carries `TRAJECTORY_SUFFIX`)."""
    return frame[frame["design"].astype(str).str.contains(TRAJECTORY_SUFFIX.strip(" ·"),
                                                          regex=False)]


def _holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p values (monotone, capped at 1)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out


def per_step_long(res: TrajectoryAsymmetry) -> pd.DataFrame:
    """Tested steps of the per-step-control designs, one row per (pair, step): contrast, raw p,
    Holm p within the pair across its tested steps, and whether the sign matches the pair's
    endpoint (= last tested step)."""
    st = _per_step_control(res.steps)
    st = st[st["tested"].astype(bool)].sort_values([*PAIR_KEY, "step"])
    parts = []
    for _, g in st.groupby(PAIR_KEY, sort=False):
        end = np.sign(g["contrast"].iloc[-1])
        parts.append(g[[*PAIR_KEY, "design", "step", "contrast", "p_two_sided"]].assign(
            p_holm=_holm(g["p_two_sided"].to_numpy(float)),
            endpoint_sign=(np.sign(g["contrast"]) == end).to_numpy()))
    cols = [*PAIR_KEY, "design", "step", "contrast", "p_two_sided", "p_holm", "endpoint_sign"]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)


def per_step_table(res: TrajectoryAsymmetry) -> pd.DataFrame:
    """One row per per-step-control pair, one column per tested step with the contrast
    (``*`` raw p < `ALPHA`, ``**`` Holm p < `ALPHA` within the pair), plus counts of
    significant steps on the endpoint's sign and on the opposite sign."""
    lf = per_step_long(res)
    steps = sorted(lf["step"].astype(int).unique())
    rows = []
    for key, g in lf.groupby(PAIR_KEY, sort=False):
        raw, holm = g["p_two_sided"] < ALPHA, g["p_holm"] < ALPHA
        same = g["endpoint_sign"].astype(bool)
        row = dict(zip(PAIR_KEY, key))
        for st in steps:
            row[f"{st:,}"] = ""
        for r, h, (_, x) in zip(raw, holm, g.iterrows()):
            row[f"{int(x['step']):,}"] = f"{x['contrast']:+.2f}" + ("**" if h else "*" if r
                                                                   else "")
        row.update({"steps sig, endpoint sign (raw)": int((raw & same).sum()),
                    "steps sig, endpoint sign (Holm)": int((holm & same).sum()),
                    "steps sig, opposite sign (raw)": int((raw & ~same).sum()),
                    "steps sig, opposite sign (Holm)": int((holm & ~same).sum())})
        rows.append(row)
    return _ordered(pd.DataFrame(rows)) if rows else pd.DataFrame(columns=PAIR_KEY)


SIGN_CHANGE_CLASSES = ("both steps sig, Δ sig", "both steps sig", "one step sig, Δ sig",
                       "one step sig", "neither sig")


def sign_change_table(res: TrajectoryAsymmetry, *, n_perm: int = 20000,
                      seed: int = 0) -> pd.DataFrame:
    """Per-step-control pairs with a tested step of the sign opposite to the endpoint's (last
    tested step, s_end); s_opp = that opposite-sign step with the most extreme contrast.
    p(Δ) tests contrast(s_end) − contrast(s_opp) = 0 with the per-step test (paired/unpaired as
    the pair's endpoint test, stratum = cell) on per-slot step differences value(s_end) −
    value(s_opp); slots missing at either step are dropped. `n_perm`/`seed` = `analyze`'s."""
    lf = per_step_long(res)
    tests = res.summary.set_index([*PAIR_KEY, "design"])["test"]
    rows = []
    for key, g in lf.groupby(PAIR_KEY, sort=False):
        end = g.iloc[-1]
        opp = g[np.sign(g["contrast"]) == -np.sign(end["contrast"])]
        if not len(opp) or not np.sign(end["contrast"]):
            continue
        o = opp.loc[opp["contrast"].abs().idxmax()]
        design = end["design"]
        pts = res.points[(res.points[PAIR_KEY] == pd.Series(dict(zip(PAIR_KEY, key)))).all(axis=1)
                         & (res.points["design"] == design)]
        sides = []
        for lab, sender in (("A->B", key[2]), ("B->A", key[3])):
            d = pts[pts["direction"] == lab]
            w = d[d["step"].isin([o["step"], end["step"]])].pivot_table(
                index=["teacher", "cell"], columns="step", values="value", aggfunc="first")
            w = w.reindex(columns=[o["step"], end["step"]]).dropna()
            diff = (w[end["step"]] - w[o["step"]]).rename("value").reset_index()
            sides.append(diff.assign(sender=sender, stratum=diff["cell"]))
        paired = tests.get((*key, design), ta.TEST_PAIRED) == ta.TEST_PAIRED
        r = ta.contrast_row(ta._condition_frame(sides[0], sides[1], ta.CONDITION_A,
                                                ta.CONDITION_B),
                            ta.CONDITION_A, ta.CONDITION_B, paired=paired, n_perm=n_perm,
                            seed=seed)
        n_sig = int(o["p_two_sided"] < ALPHA) + int(end["p_two_sided"] < ALPHA)
        dsig = r["p_two_sided"] < ALPHA
        cls = ("neither sig" if not n_sig else
               ("both steps sig" if n_sig == 2 else "one step sig") + (", Δ sig" if dsig else ""))
        rows.append({**dict(zip(PAIR_KEY, key)), "s_opp": int(o["step"]),
                     "contrast(s_opp)": float(o["contrast"]), "p(s_opp)": float(o["p_two_sided"]),
                     "s_end": int(end["step"]), "contrast(s_end)": float(end["contrast"]),
                     "p(s_end)": float(end["p_two_sided"]),
                     "Δcontrast": float(end["contrast"] - o["contrast"]),
                     "p(Δ)": float(r["p_two_sided"]), "class": cls})
    cols = [*PAIR_KEY, "s_opp", "contrast(s_opp)", "p(s_opp)", "s_end", "contrast(s_end)",
            "p(s_end)", "Δcontrast", "p(Δ)", "class"]
    return _ordered(pd.DataFrame(rows, columns=cols)) if rows else pd.DataFrame(columns=cols)


def ordering_stability(res: TrajectoryAsymmetry, *, ref_step: int = 25000) -> pd.DataFrame:
    """Per tested step of the per-step-control designs: Spearman correlation across pairs of
    the contrast at that step with the contrast at `ref_step` (pairs lacking either dropped),
    and the share of pairs whose sign matches their endpoint (last tested step); scope
    ``all`` and ``dog``."""
    lf = per_step_long(res)
    rows = []
    for scope, sub in (("all", lf), ("dog", lf[lf["animal"] == "dog"])):
        wide = sub.pivot_table(index=PAIR_KEY, columns="step", values="contrast", aggfunc="first")
        same = sub.pivot_table(index=PAIR_KEY, columns="step", values="endpoint_sign",
                               aggfunc="first")
        for st in sorted(wide.columns):
            ref = wide[ref_step] if ref_step in wide else pd.Series(np.nan, wide.index)
            both = pd.DataFrame({"x": wide[st], "r": ref}).dropna()
            rho = both["x"].rank().corr(both["r"].rank()) if len(both) >= 2 else np.nan
            sh = same[st].dropna().astype(bool)
            rows.append({"scope": scope, "step": int(st), "n pairs": int(sh.size),
                         "n pairs with ref": len(both),
                         f"Spearman vs {ref_step:,}": float(rho),
                         "share sign = endpoint": float(sh.mean()) if sh.size else np.nan})
    out = pd.DataFrame(rows)
    return out.sort_values(["step", "scope"], kind="stable", ignore_index=True) if rows else out


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    """`FAMILY_ORDER`, then animal, keeping the within-group order."""
    rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    return frame.assign(_f=frame["family"].map(rank).fillna(len(rank))).sort_values(
        ["_f", "animal"], kind="stable").drop(columns="_f")


def plot_summary_dumbbell(res: TrajectoryAsymmetry, *, fontsize: int = 9):
    """One row per tested pair, one panel per family (own x scale): the ep20 endpoint contrast
    (circle) joined to the last tested step's contrast (square); filled = p < `ALPHA`. The
    earliest p < `ALPHA` step is written at the right of each row."""
    import matplotlib.pyplot as plt

    d = _ordered(summary_display(res))
    stats = bool(d["last-step p"].notna().any()) if len(d) else True  # NaN = stats off
    fams = list(dict.fromkeys(d["family"]))
    sizes = [int((d["family"] == f).sum()) for f in fams]
    fig, axes = plt.subplots(len(fams), 1, figsize=(9, 0.32 * sum(sizes) + 1.2 * len(fams)),
                             gridspec_kw={"height_ratios": [n + 1.5 for n in sizes]},
                             squeeze=False)
    for ax, fam in zip(axes[:, 0], fams):
        sub = d[d["family"] == fam].reset_index(drop=True)
        y = np.arange(len(sub))[::-1]
        for yi, (_, r) in zip(y, sub.iterrows()):
            xs = [r["endpoint contrast"], r["last-step contrast"]]
            ax.plot([np.nanmin(xs), np.nanmax(xs)], [yi, yi], color="0.6", lw=1.2, zorder=1)
            for col, pcol, mk, c, sz in (("endpoint contrast", "endpoint p", "o", "#0072B2", 90),
                                         ("last-step contrast", "last-step p", "s", "#D55E00",
                                          24)):
                if not np.isfinite(r[col]):
                    continue
                sig = r[pcol] < ALPHA or not stats
                ax.scatter(r[col], yi, marker=mk, s=sz, zorder=3, edgecolors=c,
                           facecolors=c if sig else "white", linewidths=1.3)
        lab = [f"{r['animal']} · {pair_label(r['A'], r['B'])}"
               + (f" · {r['design'].split(' · ', 1)[-1]}" if not _is_ep20(r["design"]) else "")
               for _, r in sub.iterrows()]
        ax.set_yticks(y)
        ax.set_yticklabels(lab, fontsize=fontsize - 1)
        ax.set_ylim(-0.8, len(sub) - 0.2)
        ax.axvline(0, color="0.5", lw=0.8)
        early = [f"{int(v):,}" if np.isfinite(v) else "none"
                 for v in sub["earliest p<.05 step"].astype(float)]
        if not stats:
            ax.set_title(family_display(fam), fontsize=fontsize + 1, loc="left")
            ax.set_yticks(y)
            ax.set_yticklabels(lab, fontsize=fontsize - 1)
            ax.tick_params(axis="x", labelsize=fontsize - 1)
            continue
        sec = ax.secondary_yaxis("right")
        sec.set_yticks(y)
        sec.set_yticklabels(early, fontsize=fontsize - 1)
        for t_, d_ in zip(sec.get_yticklabels(), sub["design"]):
            t_.set_color("0.6" if _is_ep20(d_) else "black")
        sec.set_ylabel("earliest p<.05 step", fontsize=fontsize - 1)
        ax.set_title(family_display(fam), fontsize=fontsize + 1, loc="left")
        ax.tick_params(axis="x", labelsize=fontsize - 1)
    axes[-1, 0].set_xlabel("contrast A→B − B→A (receiver prior units)", fontsize=fontsize)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=9, color="#0072B2",
                      label="endpoint contrast, endpoint-module cells (orig/fs: core56)"),
               Line2D([], [], marker="s", ls="", ms=5, color="#D55E00",
                      label="last tested step, design cells (per-step control: core30 stride-3)"),
               Line2D([], [], marker="o", ls="", markerfacecolor="white",
                      markeredgecolor="0.3", label=f"hollow = p ≥ {ALPHA}"),
               Line2D([], [], ls="", label=f"grey right labels = {EP20_EARLY_NOTE}")]
    if not stats:
        handles = handles[:2]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=fontsize - 1,
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Endpoint vs last-tested-step contrast per pair" + (
        " (filled = p < .05, endpoint-module / per-step permutation test)" if stats else ""),
        fontsize=fontsize + 1, y=1.09)
    fig.tight_layout()
    return fig


#: Coverage-chart design columns, in display order (designs absent from the data are dropped).
COVERAGE_DESIGNS = ("core56", "core30", "core35", "core30 stride-3" + TRAJECTORY_SUFFIX,
                    "core30" + TRAJECTORY_SUFFIX, "core35" + TRAJECTORY_SUFFIX)


def _legible_reason(reason: str) -> str:
    """An endpoint-module reason (``core35: students 93/105, controls 35/35; core30: ...``) as a
    short 'why not tested' phrase."""
    clauses = [c.split(": ", 1) for c in str(reason).removeprefix("endpoint: ").split("; ")]
    if all(len(c) == 2 and c[1] == "complete" for c in clauses):
        return "complete; pair untested: reverse incomplete at every design"
    parts = []
    for c in clauses:
        if len(c) != 2:
            parts.append(c[0])
            continue
        label, text = c
        if text.startswith("no matched control"):
            text = "no matched control"
        elif text.startswith("students "):
            st, _, ctl = text.partition(", controls ")
            n, m = ctl.split("/") if "/" in ctl else ("", "")
            text = st.removeprefix("students ") + " students" + (
                f", controls {ctl}" if n != m else "")
        parts.append(f"{label} {text}")
    return "not tested: " + ", ".join(parts)


def _endpoint_exclusion_note(inv: pd.DataFrame) -> str:
    """One short line naming the pairs the endpoint module could not test, with reasons."""
    ex = inv[inv["design"].isna() | (inv["design"] == "")]
    groups: dict = {}
    for (fam, animal), sub in ex.groupby(["family", "animal"], sort=False):
        pairs = list(dict.fromkeys(tuple(sorted(p)) for p in zip(sub["sender"], sub["receiver"])))
        reasons = [_legible_reason(r).removeprefix("not tested: ") for r in sub["reason"]
                   if not _legible_reason(r).startswith("complete")]
        why = "; ".join(dict.fromkeys(reasons))
        if len(pairs) > 1 and len(set(reasons)) == 1:
            groups[f"{len(pairs)} {family_display(fam)} {animal} pairs"] = why
        else:
            for p in pairs:
                arms = sub[(sub["sender"].isin(p)) & (sub["receiver"].isin(p))]
                r = [f"{s}→{t} " + _legible_reason(x).removeprefix("not tested: ")
                     for s, t, x in zip(arms["sender"], arms["receiver"], arms["reason"])
                     if not _legible_reason(x).startswith("complete")]
                groups[f"{family_display(fam)} {animal} {p[0]}⇄{p[1]}"] = "; ".join(r)
    return tc.display("; ".join(f"{k} ({v})" for k, v in groups.items()))


def plot_coverage(res: TrajectoryAsymmetry, *, fontsize: int = 7):
    """Compact coverage grid, one row per PAIR A⇄B, one column per design (header: control
    kind). ● = both directions have a trajectory on every endpoint run, ○ = neither has any,
    otherwise ``k/n · k/n`` (A→B · B→A runs with a trajectory / endpoint runs); colour = the
    smaller of the two coverages. An ep20 design of a pair that is tested in a stop design
    instead is left blank (the pair reads as covered in that column). Pairs the endpoint module
    could not test are one text line under the grid, not rows."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    inv = _ordered(res.inventory.copy())
    has = inv["design"].notna() & (inv["design"] != "")
    grid, note = _active(inv[has]), _endpoint_exclusion_note(inv)
    designs = [d for d in COVERAGE_DESIGNS if d in set(grid["design"])]
    designs += [d for d in dict.fromkeys(grid["design"]) if d not in designs]
    sup = {(e["family"], e["animal"], *sorted((e["A"], e["B"]))) for _, e in
           res.excluded.iterrows() if e.get("superseded by")}
    keyed = grid.assign(_p=[tuple(sorted(p)) for p in zip(grid["sender"], grid["receiver"])])
    first = {}                                   # display order = the pair's first arm (A→B)
    for f, an, p, sn, rc in zip(keyed["family"], keyed["animal"], keyed["_p"], keyed["sender"],
                                keyed["receiver"]):
        first.setdefault((f, an, *p), (sn, rc))
    pairs = list(first)
    fig, ax = plt.subplots(figsize=(1.8 * len(designs) + 2.4, min(8.0, 0.15 * len(pairs) + 1.4)))
    cmap = plt.get_cmap("Blues")
    for i, (fam, animal, a, b) in enumerate(pairs):
        y = len(pairs) - 1 - i
        ka, kb = first[(fam, animal, a, b)]
        sub = keyed[(keyed["family"] == fam) & (keyed["animal"] == animal)
                    & (keyed["_p"] == (a, b))]
        for d, g in sub.groupby("design", sort=False):
            if _is_ep20(d) and (fam, animal, a, b) in sup:
                continue
            j = designs.index(d)
            g = g.assign(_o=(g["sender"] != ka).astype(int)).sort_values("_o")
            ks = [(int(r["runs with trajectory"]), int(r["runs (endpoint)"]))
                  for _, r in g.iterrows()]
            fr = [k / n if n else 0.0 for k, n in ks]
            if all(k == n for k, n in ks):
                text, face, tc_ = "●", cmap(1.0), "white"
            elif all(k == 0 for k, _ in ks):
                text, face, tc_ = "○", "0.85", "0.2"
            else:
                text = " · ".join(f"{k}/{n}" for k, n in ks)
                face = cmap(0.25 + 0.75 * min(fr))
                tc_ = "white" if min(fr) > 0.6 else "0.1"
            ax.add_patch(Rectangle((j, y), 1, 1, facecolor=face, edgecolor="white",
                                   hatch="///" if text == "○" else None, lw=0.5))
            ax.text(j + 0.5, y + 0.5, text, ha="center", va="center", fontsize=fontsize,
                    color=tc_)

    def short(base: str) -> str:
        """The `short_base` name, or the family's own base key when that stays long."""
        name = tc.short_base(tio.base_name(base), family_context=True)
        return tc.display(name if len(name) <= 12 else base)

    ax.set_xlim(0, len(designs))
    ax.set_ylim(0, len(pairs))
    ax.set_yticks(np.arange(len(pairs)) + 0.5)
    ax.set_yticklabels([f"{family_display(k[0])} · {k[1]} · {short(first[k][0])}⇄{short(first[k][1])}"
                        for k in pairs][::-1],
                       fontsize=fontsize)
    ax.grid(False)
    kinds = grid.groupby("design")["control"].first()
    nsteps = grid.groupby("design")["saved steps"].max()
    ax.set_xticks(np.arange(len(designs)) + 0.5)
    def header(d: str) -> str:
        if d.endswith(TRAJECTORY_SUFFIX):
            return f"{d.removesuffix(TRAJECTORY_SUFFIX)}\n{int(nsteps[d])} saved steps\n(per-step control)"
        kind = "per-step" if kinds.get(d) == CONTROL_PER_STEP else "endpoint-const."
        return f"{d}\n({kind} control)"

    ax.set_xticklabels([header(d) for d in designs], fontsize=fontsize)
    ax.xaxis.tick_top()
    for i in range(1, len(pairs)):
        if pairs[i][:2] != pairs[i - 1][:2]:
            ax.axhline(len(pairs) - i, color="0.3", lw=0.6)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Per-step trajectory coverage per pair A⇄B: ● both directions full, ○ none, "
                 "else A→B · B→A runs with trajectory / endpoint runs", fontsize=fontsize + 1,
                 pad=22)
    if note:
        import textwrap
        fig.text(0.01, 0.0, "\n".join(textwrap.wrap("Not testable at the endpoint: " + note,
                                                      140)), fontsize=fontsize, va="top")
    fig.tight_layout()
    return fig


def trajectory_directions(direction: pd.DataFrame, res: TrajectoryAsymmetry) -> pd.DataFrame:
    """`direction_table` rows plus, per direction of every per-step-control pair, its last
    step (the run's last saved step) from `res.steps`: mean value and teacher-cluster band on the
    pair's slots, so every direction curve runs over the whole trajectory."""
    st = res.steps[res.steps["design"].astype(str).str.endswith(TRAJECTORY_SUFFIX)]
    rows = []
    for (animal, a, b, _), g in st.groupby(["animal", "A", "B", "design"], sort=False):
        last = g.loc[g["step"].idxmax()]
        for s, r, col, n, cov in ((a, b, "mean A->B", "runs A->B", "coverage A->B"),
                                  (b, a, "mean B->A", "runs B->A", "coverage B->A")):
            rows.append({"animal": animal, "sender": tio.base_name(s),
                         "receiver": tio.base_name(r), "step": int(last["step"]),
                         "n": int(last[n]), "coverage": float(last[cov]),
                         "mean": last[col], "mean lo": last[f"{col} lo"],
                         "mean hi": last[f"{col} hi"]})
    return pd.concat([direction, pd.DataFrame(rows)], ignore_index=True) if rows else direction


def plot_dog_directions(direction: pd.DataFrame, *, ncols: int = 4, fontsize: int = 9):
    """One panel per (animal, cross PAIR) with per-step controls, every animal (the name is
    historical): both directions' per-step mean of
    (student − same-step control) / prior with its 95% teacher-cluster bootstrap band; the
    same-model diagonals are omitted (40–70× prior, another scale). 0 = the per-step control."""
    import matplotlib.pyplot as plt

    d = direction[(direction["n"] > 0) & (direction["coverage"].astype(float) >= MIN_COVERAGE)]
    pairs = list(dict.fromkeys((r["animal"], *sorted((r["sender"], r["receiver"])))
                               for _, r in d.iterrows()))
    nrows = max(1, -(-len(pairs) // ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.8 * nrows), squeeze=False,
                             layout="constrained")
    xmax = float(d["step"].max()) * 1.2 if len(d) else 15000
    for ax, (animal, a, b) in zip(axes.flat, pairs):
        sa, sb = pair_label(a, b).split("⇄")
        for (s, r), ls, c, lab in (((a, b), "-", "#0072B2", f"{sa}→{sb}"),
                                   ((b, a), "--", "#D55E00", f"{sb}→{sa}")):
            st = d[(d["animal"] == animal) & (d["sender"] == s)
                   & (d["receiver"] == r)].sort_values("step")
            ax.plot(st["step"], st["mean"], ls, color=c, label=lab)
            ax.plot(st["step"], st["mean"], "o", color=c, ms=3)
            ax.fill_between(st["step"], st["mean lo"], st["mean hi"], color=c, alpha=0.2, lw=0)
        _step_axes(ax, "mean value (receiver prior units)")
        ax.set_xlim(500, xmax)
        ax.set_title(f"{animal} · {sa}⇄{sb}", fontsize=fontsize)
        ax.legend(fontsize=fontsize - 1, frameon=False, loc="upper center",
                  bbox_to_anchor=(0.5, -0.22), ncol=2)
        ax.tick_params(labelsize=fontsize - 1)
        ax.xaxis.label.set_size(fontsize - 1)
        ax.yaxis.label.set_size(fontsize - 1)
    for ax in list(axes.flat)[len(pairs):]:
        ax.set_visible(False)
    fig.suptitle("Cross pairs with per-step controls: per-step mean of (student − matched control at the same "
                 f"step) / prior; 0 = control; {CURVE_NOTE}\nshading = {_BAND}; same-model "
                 "diagonals omitted (40–70× prior)", fontsize=fontsize + 2)
    return fig
