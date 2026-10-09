"""Readouts for two orig-lineage checkpoint crosses shown in ``checkpoint_distillation_analysis``.

**§2e (animal cliff cross).** `animal_cross` loads one animal's 13x13 cross (teacher@i ->
student@j, five teacher recipes, one student seed) through
`checkpoint_grid_factorization.load_checkpoint_grid_runs`, and `cross_summary` reduces it to
the §2d quantities: diagonal, student-/teacher-side gates, mature/immature block means, and
whether the diagonal is the row maximum.

**§6i (first-number resampling across checkpoints).** `load_resampling_cross` reads the three
arms of ``run_logs/campaigns/sl_resampling_cross_orig_20261008``:

* ``R``  -- teacher@i's real numbers -> base@j (the checkpoint-grid leaf);
* ``S``  -- S_all: teacher@i's first-number profile on base@j's own carriers -> base@j
  (leaf ``ex10-20_prefix_ablations_synthiid-c{j}k_poolseed0``; the 143k diagonal predates the
  cross and uses plain ``synthiid``);
* ``C``  -- the untuned base@i pool -> base@j (``students/controls/base/ckpt/...``; the 143k
  diagonal lives in the main ``controls/base/pythia-160m/teacher-pythia-160m`` layout).

Every expected leaf must exist; a missing file raises. `resampling_matrices` reduces the rows
to per-cell means and asserts the per-cell run counts; `resampling_summary` and
`control_check` produce the §6i table and control line; the two ``*_figure`` helpers draw
the notebook figures (matplotlib imported lazily).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from subliminal_learning.analysis.checkpoint_grid_factorization import (
    CKPT_STEPS,
    CKPT_STUDENT_TAG,
    CKPT_TEACHER_TAGS,
    DATASET_DIR,
    load_checkpoint_grid_runs,
)

#: Dog steps of the resampling cross (sl-resamp-x-1008).
RESAMPLING_STEPS: list[int] = [75000, 85000, 95000, 100000, 105000, 115000, 125000, 143000]
#: The final Pythia checkpoint: its base dirs carry no ``@step`` in some layouts.
FINAL_STEP: int = 143000


# --------------------------------------------------------------------------------------
# §2e: one animal's cliff cross
# --------------------------------------------------------------------------------------
def animal_cross(
    output_root: Path,
    animal: str,
    *,
    steps: list[int] = CKPT_STEPS,
    tags: list[str] = CKPT_TEACHER_TAGS,
    ftseed: int = 1,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Per-recipe teacher x student matrices of one animal's orig-lineage cross, one seed.

    Returns:
        ``(per_recipe, mean5)``: ``per_recipe[tag]`` is a ``steps x steps`` frame (rows =
        teacher step, columns = student step) of test ``p_target``; ``mean5`` is the
        unweighted mean over recipes, NaN wherever any recipe is missing.
    """
    runs = load_checkpoint_grid_runs(output_root, animal, "orig", student_tag=CKPT_STUDENT_TAG,
                                     teacher_tags=tags)
    runs = runs[(runs.ftseed == ftseed) & runs.t.isin(steps) & runs.s.isin(steps)]
    per = {t: (runs[runs.tag == t].pivot(index="t", columns="s", values="p")
               .reindex(index=steps, columns=steps)) for t in tags}
    stack = np.stack([per[t].values for t in tags])
    mean5 = pd.DataFrame(np.where(np.isnan(stack).any(0), np.nan, stack.mean(0)),
                         index=steps, columns=steps)
    return per, mean5


def diagonal_all_seeds(output_root: Path, animal: str, *, steps: list[int] = CKPT_STEPS,
                       tags: list[str] = CKPT_TEACHER_TAGS) -> tuple[float, int]:
    """``(mean, n_runs)`` of the diagonal over every student seed on disk: per (step, recipe)
    the seed mean, then the mean over recipes and steps."""
    runs = load_checkpoint_grid_runs(output_root, animal, "orig", teacher_tags=tags)
    d = runs[(runs.t == runs.s) & runs.t.isin(steps)]
    return float(d.groupby(["t", "tag"]).p.mean().mean()), len(d)


def teacher_strength(output_root: Path, animal: str, *, steps: list[int] = CKPT_STEPS,
                     tags: list[str] = CKPT_TEACHER_TAGS) -> tuple[float, float, float]:
    """``(mean teacher p_target, min base prior, max base prior)`` over ``steps`` x ``tags``
    from ``teachers/{animal}/ckpt/pythia-160m@step{s}/{tag}/preference_summary.json``."""
    from subliminal_learning.analysis.frames import load_cell_summaries

    cells = [f"teachers/{animal}/ckpt/pythia-160m@step{s}/{t}" for s in steps for t in tags]
    docs = load_cell_summaries(output_root, cells)
    missing = [c for c in cells if c not in docs]
    if missing:
        raise FileNotFoundError(f"{len(missing)} teacher summaries missing, e.g. {missing[:3]}")
    summ = [docs[c]["summaries"][0] for c in cells]
    prior = [d["base_p_target"] for d in summ]
    return float(np.mean([d["p_target_mean"] for d in summ])), float(min(prior)), float(max(prior))


def gate_curves(M: pd.DataFrame, teacher_min: int = 100000, student_min: int = 110000
                ) -> tuple[pd.Series, pd.Series]:
    """``(student_gate, teacher_gate)``: mean over teachers >= ``teacher_min`` per student
    step, and mean over students >= ``student_min`` per teacher step."""
    return (M.loc[[t for t in M.index if t >= teacher_min]].mean(),
            M[[s for s in M.columns if s >= student_min]].mean(axis=1))


def cross_summary(per: dict[str, pd.DataFrame], mean5: pd.DataFrame, *, threshold: float = 0.4,
                  mature_t: int = 115000, mature_s: int = 105000, immature: int = 100000,
                  rowmax_t: int = 105000) -> dict:
    """The §2d headline quantities of one animal's cross (all on 5-recipe means).

    Keys: ``n_cells_all5`` (cells carrying all recipes), ``takeoff`` (first diagonal step with
    mean >= ``threshold``), ``diag_<k>k`` at 100/115/125/143k, ``student_gate_115k`` /
    ``teacher_gate_115k``, ``mature_block`` (teachers >= ``mature_t`` x students >= ``mature_s``),
    ``immature_block`` (both < ``immature``), ``diag_is_rowmax`` (share of mean5 teacher rows
    >= ``rowmax_t`` whose diagonal is the row max) and ``diag_is_rowmax_per_recipe``.
    """
    steps = list(mean5.index)
    dg = pd.Series(np.diag(mean5.values), index=steps)
    hit = dg[dg >= threshold]
    sg, tg = gate_curves(mean5)
    mt = [t for t in steps if t >= mature_t]
    ms = [s for s in steps if s >= mature_s]
    im = [s for s in steps if s < immature]
    rows_t = [t for t in steps if t >= rowmax_t]

    def rowmax_share(M: pd.DataFrame) -> float:
        return float(np.mean([M.loc[t].idxmax() == t for t in rows_t]))

    out = {"n_cells_all5": int(mean5.notna().sum().sum()),
           "takeoff": f"{hit.index[0] // 1000}k" if len(hit) else f"none ≤{steps[-1] // 1000}k"}
    for k in (100000, 115000, 125000, 143000):
        out[f"diag_{k // 1000}k"] = float(dg.get(k, np.nan))
    out["student_gate_115k"] = float(sg.get(115000, np.nan))
    out["teacher_gate_115k"] = float(tg.get(115000, np.nan))
    out["mature_block"] = float(np.nanmean(mean5.loc[mt, ms].values))
    out["immature_block"] = float(np.nanmean(mean5.loc[im, im].values))
    out["diag_is_rowmax"] = rowmax_share(mean5)
    out["diag_is_rowmax_per_recipe"] = float(np.mean([rowmax_share(M) for M in per.values()]))
    return out


# --------------------------------------------------------------------------------------
# §6i: first-number resampling across checkpoints
# --------------------------------------------------------------------------------------
def _base(step: int) -> str:
    return f"pythia-160m@step{step}"


def resampling_leaf(output_root: Path, arm: str, animal: str, i: int, j: int, seed: int,
                    tag: str | None = None) -> Path:
    """The ``preference_eval.json`` of one resampling-cross leaf (teacher/pool step ``i``,
    student step ``j``). Arms ``R`` / ``S`` need ``tag``; ``C`` has none."""
    root = Path(output_root) / "students"
    leaf = Path(CKPT_STUDENT_TAG) / f"ftseed{seed}"
    if arm == "R":
        cell = root / animal / "ckpt" / _base(j) / f"teacher-{_base(i)}" / f"{tag}_tseed1" / DATASET_DIR
        return cell / leaf / "preference_eval.json"
    if arm == "S":
        pool = "synthiid" if i == j == FINAL_STEP else f"synthiid-c{j // 1000}k"
        cell = (root / animal / "ckpt" / _base(j) / f"teacher-{_base(i)}" / f"{tag}_tseed1"
                / f"ex10-20_prefix_ablations_{pool}_poolseed0")
        return cell / leaf / "eval" / animal / "preference_eval.json"
    if arm == "C":
        if i == j == FINAL_STEP:
            cell = root / "controls/base/pythia-160m/teacher-pythia-160m" / DATASET_DIR
        else:
            tdir = "teacher-pythia-160m" if i == FINAL_STEP else f"teacher-{_base(i)}"
            cell = root / "controls/base/ckpt" / _base(j) / tdir / DATASET_DIR
        return cell / leaf / "eval" / animal / "preference_eval.json"
    raise ValueError(f"unknown arm {arm!r}")


def load_resampling_cross(
    output_root: Path,
    animal: str = "dog",
    *,
    steps: list[int] = RESAMPLING_STEPS,
    tags: list[str] = CKPT_TEACHER_TAGS,
    seeds: tuple[int, ...] = (1, 2, 3),
) -> pd.DataFrame:
    """Every R / S / C leaf of the full ``steps x steps`` cross as rows ``arm, i, j, tag, seed, p``.

    Raises:
        FileNotFoundError: listing every expected leaf that is absent.
        ValueError: a leaf whose ``target_animal`` / ``ft_seed`` / ``param_tag`` disagrees.
    """
    rows, missing, bad = [], [], []
    for i in steps:
        for j in steps:
            for arm, arm_tags in (("R", tags), ("S", tags), ("C", [None])):
                for tag in arm_tags:
                    for seed in seeds:
                        path = resampling_leaf(output_root, arm, animal, i, j, seed, tag)
                        if not path.is_file():
                            missing.append(str(path))
                            continue
                        doc = json.loads(path.read_text())
                        if (doc.get("target_animal", animal) != animal
                                or doc.get("ft_seed", seed) != seed
                                or doc.get("param_tag", CKPT_STUDENT_TAG) != CKPT_STUDENT_TAG):
                            bad.append(str(path))
                        rows.append({"arm": arm, "i": i, "j": j, "tag": tag, "seed": seed,
                                     "p": float(doc["sets"]["test"]["p_target"])})
    if missing:
        raise FileNotFoundError(f"{len(missing)} resampling leaves missing, e.g. {missing[:3]}")
    if bad:
        raise ValueError(f"{len(bad)} leaves with a mismatched identity, e.g. {bad[:3]}")
    return pd.DataFrame(rows)


def resampling_matrices(rows: pd.DataFrame, steps: list[int] = RESAMPLING_STEPS,
                        expect: dict[str, int] | None = None) -> dict[str, pd.DataFrame]:
    """Per-cell mean p of each arm (rows = i, columns = j); asserts the runs per cell.

    ``expect`` defaults to 15 for R/S (5 teachers x 3 seeds) and 3 for C (3 seeds).
    """
    expect = expect or {"R": 15, "S": 15, "C": 3}
    out = {}
    for arm, n in expect.items():
        g = rows[rows.arm == arm].groupby(["i", "j"]).p
        cnt = g.size().unstack().reindex(index=steps, columns=steps)
        if not (cnt == n).all().all():
            raise AssertionError(f"arm {arm}: expected {n} runs per cell, got\n{cnt}")
        out[arm] = g.mean().unstack().reindex(index=steps, columns=steps)
    return out


def resampling_excess(M: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """``R_ex = R - C(i->j)``, ``S_ex = S_all - C(j,j)``, ``retention = S_ex / R_ex``."""
    cjj = pd.Series(np.diag(M["C"].values), index=M["C"].columns)
    rex = M["R"] - M["C"]
    sex = M["S"] - cjj  # broadcast over columns j
    return {"R_ex": rex, "S_ex": sex, "retention": sex / rex}


def resampling_summary(ex: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Same-step / direction / grid-distance rows of the two excesses and their retention."""
    rex, sex = ex["R_ex"].values, ex["S_ex"].values
    n = rex.shape[0]
    ti, sj = np.meshgrid(range(n), range(n), indexing="ij")
    groups = [("same step (i = j)", ti == sj), ("earlier teacher → later student (i < j)", ti < sj),
              ("later teacher → earlier student (i > j)", ti > sj)]
    groups += [(f"grid distance |i−j| = {d}", np.abs(ti - sj) == d) for d in range(1, n)]
    rows = []
    for label, m in groups:
        r, s = rex[m], sex[m]
        rows.append({"cells": label, "n_cells": int(m.sum()), "real_excess": r.mean(),
                     "synthetic_excess": s.mean(), "retention_ratio_of_means": s.mean() / r.mean(),
                     "retention_median_of_cells": float(np.median(s / r))})
    return pd.DataFrame(rows).set_index("cells")


def margin_spread(ex: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per excess, the variance across teacher steps of the row means and across student steps
    of the column means: which side of the pair sets the excess."""
    return pd.DataFrame({k: {"var_teacher_row_means": float(ex[k].mean(axis=1).var()),
                             "var_student_col_means": float(ex[k].mean(axis=0).var())}
                         for k in ("R_ex", "S_ex")}).T


def control_check(M: dict[str, pd.DataFrame]) -> dict[str, float]:
    """``C(i->j) - C(j,j)`` over off-diagonal cells: mean, sd, one-sample t p, and the mean
    real-number excess under each control choice."""
    from scipy import stats

    c = M["C"].values
    cjj = np.diag(c)
    off = ~np.eye(c.shape[0], dtype=bool)
    d = (c - cjj[None, :])[off]
    r = M["R"].values
    return {"n": int(off.sum()), "mean": float(d.mean()), "sd": float(d.std(ddof=1)),
            "p": float(stats.ttest_1samp(d, 0.0).pvalue),
            "R_ex_matched": float((r - c)[off].mean()),
            "R_ex_diag_control": float((r - cjj[None, :])[off].mean())}


# --------------------------------------------------------------------------------------
# Figures (matplotlib imported lazily)
# --------------------------------------------------------------------------------------
def _labelled(M: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(M.values, index=[f"{s // 1000}k" for s in M.index],
                        columns=[f"{s // 1000}k" for s in M.columns])


def _axis_labels(ax) -> None:
    ax.set_xlabel("student checkpoint (k steps)", fontsize=11)
    ax.set_ylabel("teacher checkpoint (k steps)", fontsize=11)


def animal_cross_figure(mats: dict[str, pd.DataFrame], colors: dict[str, str]):
    """Top: one annotated 5-recipe-mean heatmap per animal on a shared scale (diagonal boxed).
    Bottom: student-side and teacher-side gate curves of every animal (`gate_curves`)."""
    import matplotlib.pyplot as plt

    from subliminal_learning.analysis.plots import annotated_heatmap

    n = len(mats)
    vmax = float(np.nanmax([M.values for M in mats.values()]))
    fig = plt.figure(figsize=(7.2 * n, 12.5))
    gs = fig.add_gridspec(2, 2 * n, height_ratios=[1.25, 1.0], hspace=0.3, wspace=0.6)
    hax = [fig.add_subplot(gs[0, 2 * k:2 * k + 2]) for k in range(n)]
    im = None
    for ax, (an, M) in zip(hax, mats.items()):
        im = annotated_heatmap(ax, _labelled(M), cmap="Blues", vmin=0, vmax=vmax, origin="upper",
                               cell_fontsize=6.5, tick_fontsize=9)
        for a in range(len(M)):
            ax.add_patch(plt.Rectangle((a - .5, a - .5), 1, 1, fill=False, edgecolor="#eb6834", lw=1.2))
        ax.set_title(f"{an}: test p({an}), 5-recipe mean", fontsize=12)
        _axis_labels(ax)
    cb = fig.colorbar(im, ax=hax, fraction=0.02, pad=0.01)
    cb.set_label(f"test p(animal), 5-recipe mean (0–{vmax:.2f}); box = diagonal",
                 fontsize=10)
    axs, axt = fig.add_subplot(gs[1, :n]), fig.add_subplot(gs[1, n:])
    for an, M in mats.items():
        sg, tg = gate_curves(M)
        k = np.asarray(M.index) / 1000
        axs.plot(k, sg.values, "-o", lw=2.4, color=colors[an], label=an)
        axt.plot(k, tg.values, "-o", lw=2.4, color=colors[an], label=an)
    for ax, ttl, xl in [(axs, "student-side gate: mean over teachers ≥100k", "student checkpoint (k steps)"),
                        (axt, "teacher-side gate: mean over students ≥110k", "teacher checkpoint (k steps)")]:
        ax.set_title(ttl, fontsize=12)
        ax.set_xlabel(xl, fontsize=11)
        ax.set_ylabel("test p(animal), 5-recipe mean", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="animal", fontsize=11, title_fontsize=11, frameon=False)
    return fig


def resampling_cross_figure(ex: dict[str, pd.DataFrame]):
    """Three annotated heatmaps: real-number excess, synthetic excess, retention."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    from subliminal_learning.analysis.plots import annotated_heatmap

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    vmax = float(np.nanmax([ex["R_ex"].values, ex["S_ex"].values]))
    panels = [("R_ex", "real-number excess  $R - C(i\\to j)$", "Blues", None),
              ("S_ex", "synthetic excess  $S_{all} - C(j,j)$", "Blues", None),
              ("retention", "retention = synthetic / real excess", "RdBu_r",
               TwoSlopeNorm(vcenter=1.0, vmin=0.0, vmax=max(2.0, float(np.nanmax(ex["retention"].values)))))]
    for ax, (key, title, cmap, norm) in zip(axes, panels):
        im = annotated_heatmap(ax, _labelled(ex[key]), cmap=cmap, vmin=0, vmax=vmax, norm=norm,
                               origin="upper", cell_fontsize=9, tick_fontsize=10)
        ax.set_title(title, fontsize=12)
        _axis_labels(ax)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label("retention (1 = synthetic keeps all of the real excess)" if key == "retention"
                     else "Δ test p(dog) vs control (mean of 15 runs)", fontsize=10)
    fig.tight_layout()
    return fig
