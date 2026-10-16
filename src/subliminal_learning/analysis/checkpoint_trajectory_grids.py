"""Checkpoint grids along student training: per-saved-step 13×13 heatmaps and role curves.

Reads the dog Pythia-160m teacher-step × student-step checkpoint grids of the stop-6250 student
runs (saves at 625..6250) and joins the ep10 endpoint of the same training run as step 12500.
Per saved step the grid is the recipe mean (ftseed mean within recipe, then recipe mean).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from subliminal_learning.analysis.checkpoint_grid_factorization import (
    CKPT_STEPS,
    CKPT_TEACHER_TAGS,
    DATASET_DIR,
    _read_p_target,
)

#: Saved student-training steps of the stop-6250 trajectory runs.
TRAJ_STEPS: list[int] = [625, 1250, 1875, 2500, 3750, 6250]
#: Final step of the ep10 run the stop-6250 run was cut from.
ENDPOINT_STEP: int = 12500
#: Student HP cell of the trajectory runs.
STOP_STUDENT_TAG: str = "ep10_lr1e-4_r8a32_stop6250"
#: Student HP cell of the matching full-length endpoint runs.
ENDPOINT_STUDENT_TAG: str = "ep10_lr1e-4_r8a32"
#: Lineage -> student/teacher base directory name (fs2 lives under ``pythia-160m-seed2``).
LINEAGE_DIRS: dict[str, str] = {"orig": "pythia-160m", "fs2": "pythia-160m-seed2"}
_STEPS_PER_EPOCH = 1250


def _leaf(output_root: Path, animal: str, lineage: str, t: int, s: int, tag: str) -> Path:
    d = LINEAGE_DIRS[lineage]
    return (Path(output_root) / "students" / animal / "ckpt" / f"{d}@step{s}"
            / f"teacher-{d}@step{t}" / f"{tag}_tseed1" / DATASET_DIR)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _traj_points(doc: dict, animal: str | None = None) -> dict[int, float]:
    """``{optimizer step: test p_target}`` of a ``pref_traj.json``-style doc (points w/o a value skipped).

    Accepts ``test.p_target`` and, when ``animal`` is given, ``test.{animal}.p_target``.
    """
    out: dict[int, float] = {}
    for pt in doc.get("points") or []:
        test = pt.get("test") or {}
        p = test.get("p_target")
        if p is None and animal is not None:
            p = (test.get(animal) or {}).get("p_target")
        if p is not None and pt.get("step") is not None:
            out[int(pt["step"])] = float(p)
    return out


def load_trajectory_grid_runs(
    output_root: Path | str,
    animal: str = "dog",
    lineage: str = "orig",
    *,
    teacher_tags: list[str] | None = None,
    steps: list[int] | None = None,
    ftseeds: tuple[int, ...] = (1,),
) -> pd.DataFrame:
    """Load per-saved-step test p_target of every trajectory leaf of one lineage's grid.

    Args:
        output_root: Outputs tree root.
        animal: Target animal.
        lineage: ``orig`` or ``fs2`` (see ``LINEAGE_DIRS``).
        teacher_tags: Teacher recipes (default ``CKPT_TEACHER_TAGS``).
        steps: Base checkpoint steps (default ``CKPT_STEPS``).
        ftseeds: Student finetuning seeds to read.

    Returns:
        DataFrame with columns ``t, s, tag, ftseed, step, p, kl``; the ep10 endpoint is joined as
        step ``ENDPOINT_STEP``. Missing leaves are skipped. ``attrs["coverage"]`` holds the
        counts of ``coverage``.
    """
    tags = list(teacher_tags or CKPT_TEACHER_TAGS)
    steps = list(steps or CKPT_STEPS)
    rows: list[dict] = []
    expected = found = endpoints = short = 0
    for t in steps:
        for s in steps:
            for tag in tags:
                for fs in ftseeds:
                    expected += 1
                    leaf = _leaf(output_root, animal, lineage, t, s, tag)
                    ev = leaf / STOP_STUDENT_TAG / f"ftseed{fs}" / "eval"
                    doc = _read_json(ev / "pref_traj.json")
                    if not doc or not doc.get("points"):
                        continue
                    found += 1
                    kl_doc = _read_json(ev / "pile_kl_traj.json") or {}
                    kl = {pt.get("step"): pt.get("kl_base_adapted") for pt in kl_doc.get("points", [])}
                    pts = _traj_points(doc)
                    for st, p in pts.items():
                        rows.append(dict(t=t, s=s, tag=tag, ftseed=fs, step=st,
                                         p=p, kl=kl.get(st, np.nan)))
                    short += len(pts) < len(TRAJ_STEPS)
                    pe = _read_p_target(leaf / ENDPOINT_STUDENT_TAG / f"ftseed{fs}"
                                        / "preference_eval.json")
                    if pe is not None:
                        endpoints += 1
                        rows.append(dict(t=t, s=s, tag=tag, ftseed=fs, step=ENDPOINT_STEP,
                                         p=float(pe), kl=np.nan))
    runs = pd.DataFrame(rows, columns=["t", "s", "tag", "ftseed", "step", "p", "kl"])
    runs.attrs["coverage"] = dict(lineage=lineage, expected=expected, found=found,
                                  endpoints=endpoints, short=short)
    return runs


def coverage(runs: pd.DataFrame, lineage: str) -> dict:
    """Coverage counts of a ``load_trajectory_grid_runs`` frame.

    Args:
        runs: Loader output.
        lineage: Lineage name (for the returned dict).

    Returns:
        Dict with ``lineage, expected, found, endpoints, short`` (leaves with < 6 points).
    """
    cov = dict(runs.attrs.get("coverage", {}))
    cov["lineage"] = lineage
    return cov


def step_grids(runs: pd.DataFrame, steps: list[int] | None = None,
               ckpt_steps: list[int] | None = None) -> dict[int, np.ndarray]:
    """Per saved step, the recipe-mean teacher×student grid.

    Args:
        runs: Loader output.
        steps: Saved steps (default ``TRAJ_STEPS + [ENDPOINT_STEP]``).
        ckpt_steps: Grid axis (default ``CKPT_STEPS``).

    Returns:
        ``{step: M}`` with ``M[i, j]`` = teacher ``ckpt_steps[i]`` × student ``ckpt_steps[j]``;
        NaN for empty cells.
    """
    steps = list(steps or TRAJ_STEPS + [ENDPOINT_STEP])
    ck = list(ckpt_steps or CKPT_STEPS)
    out: dict[int, np.ndarray] = {}
    for st in steps:
        sub = runs[runs.step == st]
        cell = sub.groupby(["t", "s", "tag"]).p.mean().groupby(["t", "s"]).mean()
        M = np.full((len(ck), len(ck)), np.nan)
        for (t, s), v in cell.items():
            if t in ck and s in ck:
                M[ck.index(t), ck.index(s)] = v
        out[st] = M
    return out


def _step_title(step: int) -> str:
    ep = step / _STEPS_PER_EPOCH
    ep_s = f"{ep:g}"
    return f"step {step} (ep {ep_s}, endpoint)" if step == ENDPOINT_STEP else f"step {step} (ep {ep_s})"


def plot_step_heatmaps(grids_by_lineage: dict[str, dict[int, np.ndarray]], *,
                       steps: list[int] | None = None, vmax: float | None = None,
                       ckpt_steps: list[int] | None = None, annotate: bool = False,
                       title_fn=None, figsize: tuple[float, float] | None = None) -> plt.Figure:
    """One row per lineage (or arm), one column per saved step, all panels on one colour scale.

    Args:
        grids_by_lineage: ``{lineage: step_grids(...)}``.
        steps: Saved steps to show (default: keys of the first lineage, sorted).
        vmax: Colour maximum (default: max over all grids rounded up to 0.05).
        ckpt_steps: Grid axis labels (default ``CKPT_STEPS``).
        annotate: Write each cell's value into the panel.
        title_fn: Step -> column title (default ``_step_title``).
        figsize: Figure size (default 22 wide, 3.6 per row).

    Returns:
        The figure.
    """
    title_fn = title_fn or _step_title
    ck = list(ckpt_steps or CKPT_STEPS)
    lins = list(grids_by_lineage)
    steps = list(steps or sorted(next(iter(grids_by_lineage.values()))))
    if vmax is None:
        mx = max(np.nanmax(g[st]) for g in grids_by_lineage.values() for st in steps)
        vmax = math.ceil(mx / 0.05 - 1e-9) * 0.05
    fig, axes = plt.subplots(len(lins), len(steps), figsize=figsize or (22, 3.6 * len(lins) + 0.4),
                             squeeze=False, constrained_layout=True)
    labels = [str(x // 1000) for x in ck]
    tick_idx = list(range(len(ck)))
    im = None
    for i, lin in enumerate(lins):
        for j, st in enumerate(steps):
            ax = axes[i, j]
            im = ax.imshow(grids_by_lineage[lin][st], vmin=0, vmax=vmax, cmap="viridis",
                           origin="upper", aspect="equal")
            ax.grid(False)
            for k in range(len(ck)):
                ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False, ec="white", lw=0.8))
            if annotate:
                M = grids_by_lineage[lin][st]
                for (a, b), v in np.ndenumerate(M):
                    if np.isfinite(v):
                        ax.text(b, a, f"{v:.2f}"[1:] if 0 <= v < 1 else f"{v:.1f}", ha="center",
                                va="center", fontsize=6,
                                color="black" if v > 0.6 * vmax else "white")
            fs = 7 if len(ck) > 8 else 9
            ax.set_xticks(tick_idx, [labels[k] for k in tick_idx], fontsize=fs, rotation=90)
            ax.set_yticks(tick_idx, [labels[k] for k in tick_idx], fontsize=fs)
            if i == 0:
                ax.set_title(title_fn(st), fontsize=11)
            if i == len(lins) - 1:
                ax.set_xlabel("student ckpt (k steps)", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{lin}\nteacher ckpt (k steps)", fontsize=11)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("test p_dog (recipe mean)", fontsize=11)
    return fig


def role_means(grids: dict[int, np.ndarray], ckpt_steps: list[int] | None = None) -> pd.DataFrame:
    """Per saved step and checkpoint, mean transfer sent (row mean) and received (column mean).

    Args:
        grids: ``step_grids`` output.
        ckpt_steps: Grid axis (default ``CKPT_STEPS``).

    Returns:
        Long frame ``step, ckpt, teach, learn, ratio`` (ckpt in k steps, ratio = teach / learn).
    """
    ck = list(ckpt_steps or CKPT_STEPS)
    rows = []
    for st, M in grids.items():
        teach, learn = np.nanmean(M, axis=1), np.nanmean(M, axis=0)
        for k, c in enumerate(ck):
            rows.append(dict(step=st, ckpt=c // 1000, teach=teach[k], learn=learn[k],
                             ratio=teach[k] / learn[k]))
    return pd.DataFrame(rows)


def plot_role_curves(role_by_lineage: dict[str, pd.DataFrame]) -> plt.Figure:
    """Per lineage, mean transfer received (as student) and sent (as teacher) per checkpoint.

    Args:
        role_by_lineage: ``{lineage: role_means(...)}``.

    Returns:
        The figure (one row per lineage, one shared y-scale, one legend block).
    """
    lins = list(role_by_lineage)
    fig, axes = plt.subplots(len(lins), 2, figsize=(14, 4.6 * len(lins)), sharey=True,
                             squeeze=False, constrained_layout=True)
    steps = sorted(next(iter(role_by_lineage.values())).step.unique())
    cmap = plt.get_cmap("viridis")
    cols = {st: cmap(0.05 + 0.85 * i / max(len(steps) - 1, 1)) for i, st in enumerate(steps)}
    for i, lin in enumerate(lins):
        df = role_by_lineage[lin]
        for j, (col, name) in enumerate([("learn", "as student: mean transfer received"),
                                         ("teach", "as teacher: mean transfer sent")]):
            ax = axes[i, j]
            for st in steps:
                d = df[df.step == st].sort_values("ckpt")
                ax.plot(d.ckpt, d[col], "o-", ms=3.5, color=cols[st],
                        lw=2.4 if st == 6250 else 1.5, label=_step_title(st))
            ax.set_title(f"{lin} — {name}", fontsize=11)
            ax.set_xlabel("checkpoint (k steps)", fontsize=10)
            if j == 0:
                ax.set_ylabel("test p_dog (recipe mean)", fontsize=10)
            ax.grid(alpha=0.3)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside right center", fontsize=10, title="saved student step",
               title_fontsize=10)
    return fig


def ratio_table(role_by_lineage: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Teach/learn ratio per (lineage, saved step) × checkpoint (k steps).

    Args:
        role_by_lineage: ``{lineage: role_means(...)}``.

    Returns:
        Wide frame indexed by ``(lineage, step)``, columns = checkpoint k.
    """
    parts = [df.assign(lineage=lin) for lin, df in role_by_lineage.items()]
    return (pd.concat(parts).pivot_table(index=["lineage", "step"], columns="ckpt", values="ratio",
                                         sort=False))


# --------------------------------------------------------------------------------------
# S_all (synthetic-number) trajectories on the 8×8 resampling-cross grid
# --------------------------------------------------------------------------------------
#: Saved steps of a full trajectory (625 … 12,500).
ALL_SAVED_STEPS: list[int] = TRAJ_STEPS + [ENDPOINT_STEP]
#: Class name -> (teacher step, student step) predicate; compares optimizer steps, not indices.
STEP_CLASSES: dict[str, object] = {
    "diagonal (i = j)": lambda t, s: t == s,
    "all 64 cells": lambda t, s: True,
}


def step_label(step: int) -> str:
    """Optimizer-step title, e.g. ``step 2,500``."""
    return f"step {step:,}"


def sall_cell(output_root: Path | str, t: int, s: int, tag: str, student_tag: str,
              animal: str = "dog") -> Path:
    """S_all student directory (holding ``ftseed{k}``) of one cell, via ``resampling_leaf``'s
    dataset rule (plain ``synthiid`` on the 143k diagonal)."""
    from subliminal_learning.analysis.checkpoint_cross_readouts import resampling_leaf

    # resampling_leaf -> {cell}/{CKPT_STUDENT_TAG}/ftseed1/eval/{animal}/preference_eval.json
    return resampling_leaf(Path(output_root), "S", animal, t, s, 1, tag).parents[4] / student_tag


def _read_sall_endpoint(seed_dir: Path, animal: str) -> float | None:
    """Test p_target at ``ENDPOINT_STEP`` from ``eval/{animal}/preference_eval.json`` or else
    ``eval/pref_endpoint10.json``; ``None`` if neither is readable."""
    p = _read_p_target(seed_dir / "eval" / animal / "preference_eval.json")
    if p is not None:
        return float(p)
    doc = _read_json(seed_dir / "eval" / "pref_endpoint10.json")
    return _traj_points(doc, animal).get(ENDPOINT_STEP) if doc else None


def load_sall_trajectory_runs(
    output_root: Path | str,
    *,
    steps: list[int] | None = None,
    ftseeds: tuple[int, ...] = (1, 2, 3),
    teacher_tags: list[str] | None = None,
    animal: str = "dog",
) -> pd.DataFrame:
    """Per-saved-step test p_target of every dog S_all checkpoint-cross student.

    Every expected (t, s, tag, ftseed) gets one row per step of ``ALL_SAVED_STEPS``; a missing
    trajectory point or endpoint is NaN (and counted in coverage) rather than raising.

    Args:
        output_root: Outputs tree root.
        steps: Base checkpoint steps (default ``RESAMPLING_STEPS``).
        ftseeds: Student finetuning seeds.
        teacher_tags: Teacher recipes (default ``CKPT_TEACHER_TAGS``).
        animal: Target animal.

    Returns:
        Frame ``t, s, tag, ftseed, step, p, kl`` like ``load_trajectory_grid_runs``;
        ``attrs["coverage"]`` = ``expected, found`` (trajectory files), ``endpoints``, ``short``
        (trajectories with < 6 points).
    """
    from subliminal_learning.analysis.checkpoint_cross_readouts import RESAMPLING_STEPS

    tags = list(teacher_tags or CKPT_TEACHER_TAGS)
    steps = list(steps or RESAMPLING_STEPS)
    rows: list[dict] = []
    expected = found = endpoints = short = 0
    for t in steps:
        for s in steps:
            for tag in tags:
                stop = sall_cell(output_root, t, s, tag, STOP_STUDENT_TAG, animal)
                end = sall_cell(output_root, t, s, tag, ENDPOINT_STUDENT_TAG, animal)
                for fs in ftseeds:
                    expected += 1
                    doc = _read_json(stop / f"ftseed{fs}" / "eval" / "pref_traj.json")
                    pts = _traj_points(doc, animal) if doc else {}
                    found += bool(pts)
                    short += bool(pts) and len(pts) < len(TRAJ_STEPS)
                    pe = _read_sall_endpoint(end / f"ftseed{fs}", animal)
                    endpoints += pe is not None
                    pts[ENDPOINT_STEP] = np.nan if pe is None else pe
                    for st in ALL_SAVED_STEPS:
                        rows.append(dict(t=t, s=s, tag=tag, ftseed=fs, step=st,
                                         p=pts.get(st, np.nan), kl=np.nan))
    runs = pd.DataFrame(rows, columns=["t", "s", "tag", "ftseed", "step", "p", "kl"])
    runs.attrs["coverage"] = dict(lineage="S_all", expected=expected, found=found,
                                  endpoints=endpoints, short=short)
    return runs


def class_means(grid: np.ndarray, ckpt_steps: list[int]) -> dict[str, float]:
    """Mean of a teacher×student grid over each ``STEP_CLASSES`` class."""
    t, s = np.meshgrid(ckpt_steps, ckpt_steps, indexing="ij")
    return {name: float(np.nanmean(grid[np.vectorize(f)(t, s)])) for name, f in STEP_CLASSES.items()}


def rank1_by_step(grids: dict[int, np.ndarray], ckpt_steps: list[int],
                  sender: int = 125000) -> pd.DataFrame:
    """Per saved step, the rank-1 fit ``M[t, s] ~= a[t] * b[s]`` of a recipe-mean grid.

    Args:
        grids: ``step_grids`` output (complete grids).
        ckpt_steps: Grid axis.
        sender: Checkpoint whose sender value is compared with the median.

    Returns:
        Frame indexed by ``step`` with ``r2`` (about the grid mean) and ``a_sender_over_median``
        (``a[sender] / median(a)``; gauge-free; NaN if ``sender`` is off the axis).
    """
    from subliminal_learning.analysis.checkpoint_grid_factorization import _r2, fit_rank1

    rows = {}
    for st, M in grids.items():
        a, b = fit_rank1(M)
        rows[st] = dict(r2=_r2(M, np.outer(a, b)),
                        a_sender_over_median=(a[list(ckpt_steps).index(sender)] / np.median(a)
                                              if sender in ckpt_steps else np.nan))
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "step"
    return out


def sall_class_table(sall_runs: pd.DataFrame, r_runs: pd.DataFrame,
                     ckpt_steps: list[int], steps: list[int] | None = None) -> pd.DataFrame:
    """Per saved step: diagonal and all-cell S_all / R / S_all÷R, the S_all ftseed-1 all-cell
    mean, and per arm the rank-1 R² and sender concentration ``a(125k) / median a``.

    Args:
        sall_runs: ``load_sall_trajectory_runs`` output (all ftseeds).
        r_runs: ``load_trajectory_grid_runs`` output (ftseed 1).
        ckpt_steps: Grid axis (both arms restricted to it).
        steps: Saved steps (default ``ALL_SAVED_STEPS``).

    Returns:
        Frame indexed by ``step`` with (group, quantity) MultiIndex columns.
    """
    steps = list(steps or ALL_SAVED_STEPS)
    gs = step_grids(sall_runs, steps, ckpt_steps)
    gr = step_grids(r_runs, steps, ckpt_steps)
    g1 = step_grids(sall_runs[sall_runs.ftseed == 1], steps, ckpt_steps)
    k1 = {"S_all": rank1_by_step(gs, ckpt_steps), "R": rank1_by_step(gr, ckpt_steps)}
    rows = {}
    for st in steps:
        cs, cr = class_means(gs[st], ckpt_steps), class_means(gr[st], ckpt_steps)
        row = {}
        for name in STEP_CLASSES:
            row[(name, "S_all")] = cs[name]
            row[(name, "R")] = cr[name]
            row[(name, "S_all÷R")] = cs[name] / cr[name]
        row[("all 64 cells", "S_all ftseed 1")] = class_means(g1[st], ckpt_steps)["all 64 cells"]
        for arm, k in k1.items():
            row[("rank-1 R²", arm)] = k.loc[st, "r2"]
        for arm, k in k1.items():
            row[("sender concentration a(125k)/median a", arm)] = k.loc[st, "a_sender_over_median"]
        rows[st] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index.name = "step"
    return out


def plot_role_trajectories(role_by_arm: dict[str, pd.DataFrame]) -> plt.Figure:
    """One panel per checkpoint: teach- and learn-mean against saved step (log x), per arm.

    Args:
        role_by_arm: ``{arm: role_means(...)}`` on a shared checkpoint axis.

    Returns:
        The figure (one shared y-scale, one legend naming arm and role).
    """
    ckpts = sorted(next(iter(role_by_arm.values())).ckpt.unique())
    ncol = 4
    nrow = math.ceil(len(ckpts) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.6 * nrow + 0.6), sharey=True,
                             sharex=True, squeeze=False, constrained_layout=True)
    colors = dict(zip(role_by_arm, ["#d95f02", "#1b9e77", "#7570b3", "#e7298a"]))
    for k, c in enumerate(ckpts):
        ax = axes.flat[k]
        for arm, df in role_by_arm.items():
            d = df[df.ckpt == c].sort_values("step")
            ax.plot(d.step, d.teach, "o-", color=colors[arm], lw=2, ms=4,
                    label=f"{arm}: as teacher (sent)")
            ax.plot(d.step, d.learn, "s--", color=colors[arm], lw=2, ms=4, alpha=0.8,
                    label=f"{arm}: as student (received)")
        ax.set_xscale("log")
        steps = sorted(d.step.unique())
        ax.set_xticks(steps, [f"{x:,}" for x in steps], rotation=45, fontsize=8)
        ax.minorticks_off()
        ax.set_title(f"checkpoint {c}k", fontsize=11)
        ax.grid(alpha=0.3)
        if k % ncol == 0:
            ax.set_ylabel("test p_dog (recipe mean)", fontsize=10)
        if k // ncol == nrow - 1:
            ax.set_xlabel("student optimizer step", fontsize=10)
    for ax in axes.flat[len(ckpts):]:
        ax.set_visible(False)
    h, lab = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside upper center", ncol=len(h), fontsize=11, frameon=False)
    return fig
