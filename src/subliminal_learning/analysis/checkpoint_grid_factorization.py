"""Two-factor (rank-one) analysis of teacher x student checkpoint grids.

**The question.** In a checkpoint grid the SAME pretraining lineage supplies both sides of a
distillation: a teacher fine-tuned from base checkpoint ``t`` writes numbers, and a student
initialised from base checkpoint ``s`` learns from them. The grid ``M[t, s]`` = student test
``p_target`` is a 13x13 (for the canonical step list) matrix. Two very different stories can
produce a grid that rises towards late checkpoints:

1. **Two independent factors.** The teacher contributes a *sending strength* ``a(t)`` and the
   student a *receiving susceptibility* ``b(s)``, and they simply multiply:
   ``M[t, s] ~= a(t) * b(s)``. Transfer then has no notion of "these two checkpoints match";
   any strong sender works with any receptive receiver.
2. **Matching / shared-state stories.** Transfer needs the two sides to be *the same* -- either
   because the weaker of the two limits it (``f(min(t, s))``, a bottleneck), because only the
   later one matters (``f(max(t, s))``, a mirror), or because being *near* each other in
   pretraining time helps on top of the two factors (a diagonal bonus, or a nearness kernel
   ``exp(-|t-s|/L)``).

`fit_models` scores all of these rivals on one grid with their free-parameter counts next to
them, so "the product wins" is a statement about fit *per parameter*, not just fit. The
comparison is anchored from below by the **noise ceiling** (`noise_ceiling`): the R^2 a perfect
model could reach given the seed noise in the cell means -- a model at the ceiling is done.

**What each test means.**

* ``additive`` vs ``product``: is the interaction multiplicative (scale-free: a better receiver
  multiplies every sender by the same factor) or additive (offsets)? Reported on the probability
  scale (``r2``) and on the log scale (``r2_log``), because the two scales favour opposite rivals.
* ``bottleneck`` / ``mirror``: pure matching models; they get ``n`` parameters instead of
  ``2n-1``, so they only need to be *close* to win.
* ``product + diagonal`` / ``product + nearness``: does same-checkpoint (or near-checkpoint)
  pairing buy anything *beyond* the two factors? The fitted ``c`` is the size of that bonus in
  probability units, and ``L`` how far along pretraining it reaches.
* `asymmetry_test`: the product model is symmetric up to the two curves -- it predicts a specific
  antisymmetric pattern. We report how much of the off-diagonal variance is antisymmetric, how
  much of that seed noise alone would manufacture, and how much of it the fitted rank-one term
  already reproduces. A large antisymmetric share that the product *reproduces* is direction
  dependence living inside ``a`` and ``b``; one it does not reproduce is a genuine residual.

**Cross-lineage variant.** The same rank-one question is asked of a lineage x lineage matrix
(orig, data-seed1, ...): does "who sends well" separate from "who receives well" across *seeds*
rather than across checkpoints? There the diagonal is a different experiment (same-model
distillation) so the fit uses the off-diagonal mask only, cells are control-subtracted against
matched base controls, and rank-one is additionally tested by `cycle_tests` (a rank-one matrix
has zero log-cycle differences around every triple) and `partner_independence_ratios` (the ratio
of two senders' strengths must not depend on which receiver it is measured through).

Everything here is pure numpy/pandas over the output tree; no matplotlib, no GPU. The CLI is
`scripts/checkpoint_grid_factorization.py`.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

#: The five teacher recipes ("recipes") every checkpoint-grid cell was run with.
CKPT_TEACHER_TAGS: list[str] = [
    "ep20_lr1e-4_r16a32", "ep2_lr5e-4_r32a64", "ep7_lr5e-4_r8a8",
    "ep10_lr1e-4_r64a64", "ep20_lr1e-4_r8a16",
]
#: The student HP cell the checkpoint grid holds fixed.
CKPT_STUDENT_TAG: str = "ep10_lr1e-4_r8a32"
#: Canonical 13 base checkpoints of the grid (75k..125k by 5k, then 130k and the final 143k).
CKPT_STEPS: list[int] = [
    75000, 80000, 85000, 90000, 95000, 100000, 105000, 110000, 115000, 120000, 125000,
    130000, 143000,
]
#: Numbers dataset level shared by every grid cell.
DATASET_DIR: str = "ex10-20_prefix_n100k"
#: Decay lengths (in thousands of steps) searched by the ``product + nearness`` rival.
NEARNESS_LENGTHS: tuple[float, ...] = (2.5, 5.0, 10.0, 20.0, 40.0)

_CKPT_DIR_RE = re.compile(r"^pythia-160m(?P<suffix>.*)@step(?P<step>\d+)$")
_TAGSEED_RE = re.compile(r"^(?P<tag>.+)_tseed(?P<seed>\d+)$")
_FTSEED_RE = re.compile(r"^ftseed(?P<seed>\d+)$")


def lineage_dir(lineage: str) -> str:
    """Directory name of a 160m pretraining lineage (``orig`` -> ``pythia-160m``)."""
    return "pythia-160m" if lineage == "orig" else f"pythia-160m-{lineage}"


def _read_p_target(path: Path) -> float | None:
    """``sets.test.p_target`` of a ``preference_eval.json``; ``None`` if unreadable."""
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    sets = doc.get("sets")
    if isinstance(sets, dict) and isinstance(sets.get("test"), dict):
        return sets["test"].get("p_target")
    return doc.get("p_target")


# --------------------------------------------------------------------------------------
# Same-lineage checkpoint grid
# --------------------------------------------------------------------------------------
def load_checkpoint_grid_runs(
    output_root: Path,
    animal: str,
    lineage: str,
    *,
    student_tag: str = CKPT_STUDENT_TAG,
    teacher_tags: list[str] | None = None,
) -> pd.DataFrame:
    """Scan one same-lineage checkpoint grid into a run-level frame.

    Reads ``students/{animal}/ckpt/{student_dir}/teacher-{teacher_dir}/{tag}_tseed1/
    {DATASET_DIR}/{student_tag}/ftseed*/preference_eval.json``, keeping only pairs whose
    teacher and student checkpoint directories belong to the SAME lineage.

    `analysis.student_tree.load_treatments` is deliberately not reused: it rglobs the whole
    ``students/`` tree, reads cell-level ``preference_summary.json`` (already averaged over
    seeds, so the per-cell SE this module needs is gone), and skips the ``ckpt`` subtree
    because of its extra path level.

    Args:
        output_root: ``paths.output_root``.
        animal: Trait animal, e.g. ``dog``.
        lineage: Lineage name; ``orig`` or a ``pythia-160m-`` suffix (``data-seed1``, ...).
        student_tag: The fixed student HP cell.
        teacher_tags: Teacher recipes to keep; default `CKPT_TEACHER_TAGS`.

    Returns:
        One row per run with columns ``t``, ``s`` (base checkpoint steps), ``tag`` (teacher
        recipe), ``ftseed``, ``p``.
    """
    tags = set(CKPT_TEACHER_TAGS if teacher_tags is None else teacher_tags)
    root = Path(output_root) / "students" / animal / "ckpt"
    suffix = "" if lineage == "orig" else f"-{lineage}"

    rows: list[dict] = []
    for student_dir in sorted(root.glob(f"pythia-160m{suffix}@step*")):
        ms = _CKPT_DIR_RE.fullmatch(student_dir.name)
        if not ms or ms["suffix"] != suffix:
            continue
        for teacher_dir in sorted(student_dir.glob("teacher-pythia-160m*@step*")):
            mt = _CKPT_DIR_RE.fullmatch(teacher_dir.name[len("teacher-"):])
            if not mt or mt["suffix"] != suffix:  # same lineage only
                continue
            for tagseed in sorted(teacher_dir.iterdir()):
                mg = _TAGSEED_RE.fullmatch(tagseed.name)
                if not mg or mg["tag"] not in tags:
                    continue
                cell = tagseed / DATASET_DIR / student_tag
                for seed_dir in sorted(cell.glob("ftseed*")):
                    mf = _FTSEED_RE.fullmatch(seed_dir.name)
                    value = _read_p_target(seed_dir / "preference_eval.json")
                    if mf is None or value is None:
                        continue
                    rows.append({
                        "t": int(mt["step"]), "s": int(ms["step"]), "tag": mg["tag"],
                        "ftseed": int(mf["seed"]), "p": float(value),
                    })
    return pd.DataFrame(rows, columns=["t", "s", "tag", "ftseed", "p"])


@dataclass
class GridMatrix:
    """Cell means of a checkpoint grid plus their seed-noise scale.

    Attributes:
        M: ``(n, n)`` cell means -- seed mean within a teacher recipe, then the unweighted
            mean over recipes (so an over-seeded recipe cannot dominate a cell).
        se: ``(n, n)`` standard error of each cell mean: the mean over recipes of the
            within-recipe seed variance, divided by the cell's total run count, square-rooted.
        n_recipes: ``(n, n)`` distinct teacher recipes contributing to each cell.
        n_runs: ``(n, n)`` runs (recipe x seed) contributing to each cell.
        steps: The checkpoint steps indexing both axes.
    """

    M: np.ndarray
    se: np.ndarray
    n_recipes: np.ndarray
    n_runs: np.ndarray
    steps: list[int]

    @property
    def labels(self) -> list[str]:
        """Display labels of `steps`, e.g. ``75k``."""
        return [f"{s // 1000}k" for s in self.steps]

    def frame(self) -> pd.DataFrame:
        """`M` as a labelled frame (rows = teacher step, columns = student step)."""
        return pd.DataFrame(self.M, index=self.labels, columns=self.labels)


def grid_matrix(runs: pd.DataFrame, steps: list[int] | None = None) -> GridMatrix:
    """Reduce run-level grid rows to cell means, SEs and coverage counts.

    Args:
        runs: Output of `load_checkpoint_grid_runs`.
        steps: Checkpoint steps to keep, in display order; default `CKPT_STEPS`.

    Returns:
        The populated `GridMatrix`.

    Raises:
        ValueError: If any (teacher step, student step) cell of the requested grid has no runs.
    """
    steps = list(CKPT_STEPS if steps is None else steps)
    d = runs[runs["t"].isin(steps) & runs["s"].isin(steps)]

    def _pivot(series: pd.Series) -> np.ndarray:
        return series.unstack().reindex(index=steps, columns=steps).to_numpy(dtype=float)

    per_recipe = d.groupby(["t", "s", "tag"])["p"]
    M = _pivot(per_recipe.mean().groupby(["t", "s"]).mean())
    within = _pivot(per_recipe.var(ddof=1).fillna(0.0).groupby(["t", "s"]).mean())
    n_recipes = _pivot(d.groupby(["t", "s"])["tag"].nunique())
    n_runs = _pivot(d.groupby(["t", "s"]).size())
    if np.isnan(M).any():
        missing = [(steps[i], steps[j]) for i, j in zip(*np.where(np.isnan(M)))]
        raise ValueError(f"checkpoint grid has {len(missing)} empty cells, e.g. {missing[:5]}")
    return GridMatrix(M=M, se=np.sqrt(within / n_runs), n_recipes=n_recipes, n_runs=n_runs,
                      steps=steps)


# --------------------------------------------------------------------------------------
# Rank-one fit and its rivals
# --------------------------------------------------------------------------------------
def fit_rank1(
    M: np.ndarray, mask: np.ndarray | None = None, iters: int = 3000
) -> tuple[np.ndarray, np.ndarray]:
    """Alternating least squares for ``M[t, s] ~= a[t] * b[s]`` on the masked support.

    For a full mask this converges to the top SVD term (the ALS power iteration on a
    complete matrix *is* the power method); an explicit mask lets the diagonal -- a different
    experiment in the cross-lineage matrices -- be excluded.

    Args:
        M: The matrix to factor.
        mask: Boolean support of the fit; ``None`` uses every cell.
        iters: Maximum ALS sweeps (the loop stops early once the factors are stationary
            to 1e-13, which the well-conditioned grids reach in a few dozen sweeps).

    Returns:
        ``(a, b)`` rescaled so that ``b.max() == 1`` (the gauge freedom ``a*c, b/c`` is fixed
        by making ``a`` read in probability units of the best receiver).
    """
    mask = np.ones_like(M, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    n_rows, n_cols = M.shape
    a = np.array([M[i, mask[i]].mean() for i in range(n_rows)])
    b = np.ones(n_cols)
    for _ in range(iters):
        previous = a
        b = np.array([(M[mask[:, j], j] * a[mask[:, j]]).sum() / (a[mask[:, j]] ** 2).sum()
                      for j in range(n_cols)])
        a = np.array([(M[i, mask[i]] * b[mask[i]]).sum() / (b[mask[i]] ** 2).sum()
                      for i in range(n_rows)])
        if np.abs(a - previous).max() <= 1e-13 * max(1.0, np.abs(a).max()):
            break
    c = b.max()
    return a * c, b / c


def _r2(M: np.ndarray, P: np.ndarray) -> float:
    return float(1 - ((M - P) ** 2).sum() / ((M - M.mean()) ** 2).sum())


def _r2_log(M: np.ndarray, P: np.ndarray) -> float:
    """R^2 of ``log(P)`` against ``log(M)``; NaN if either side is non-positive anywhere."""
    if (P <= 0).any() or (M <= 0).any():
        return float("nan")
    lm, lp = np.log(M), np.log(P)
    return float(1 - ((lm - lp) ** 2).sum() / ((lm - lm.mean()) ** 2).sum())


def _additive(M: np.ndarray) -> np.ndarray:
    mu = M.mean()
    return mu + (M.mean(1) - mu)[:, None] + (M.mean(0) - mu)[None, :]


def _keyed(M: np.ndarray, key: np.ndarray) -> np.ndarray:
    """Least-squares fit of one free level per distinct value of ``key``."""
    P = np.zeros_like(M)
    for k in np.unique(key):
        P[key == k] = M[key == k].mean()
    return P


def _rank_k(M: np.ndarray, k: int) -> np.ndarray:
    U, s, Vt = np.linalg.svd(M)
    return (U[:, :k] * s[:k]) @ Vt[:k]


def _fit_rank1_plus(
    M: np.ndarray, E: np.ndarray, outer_iters: int = 300, inner_iters: int = 400
) -> tuple[np.ndarray, np.ndarray, float]:
    """Alternate ``a[t]b[s] + c*E`` for a fixed pattern ``E`` (diagonal or nearness kernel)."""
    c = 0.0
    a = b = np.zeros(M.shape[0])
    for _ in range(outer_iters):
        a, b = fit_rank1(M - c * E, iters=inner_iters)
        R = M - np.outer(a, b)
        previous, c = c, float((R * E).sum() / (E ** 2).sum())
        if abs(c - previous) <= 1e-13:
            break
    return a, b, c


def fit_models(M: np.ndarray, steps: list[int]) -> pd.DataFrame:
    """Score the rival structures for one grid against each other.

    Free parameters are counted after removing the models' exact scale/shift ambiguities:
    additive ``2n-1`` (an overall level plus two centred offset vectors), product ``2n-1``
    (one scale gauge removed), product + diagonal ``2n``, product + nearness ``2n+1`` (``c``
    and ``L``), bottleneck / mirror ``n``, rank-2 ``2(2n-1)``.

    Args:
        M: ``(n, n)`` cell means, teacher steps by student steps.
        steps: The checkpoint steps indexing ``M`` (used for the nearness distances).

    Returns:
        ``DataFrame[model, free_params, r2, r2_log]``, one row per rival. The chosen nearness
        length and both fitted bonuses are reported in the ``model`` label.
    """
    n = M.shape[0]
    idx = np.arange(n)
    eye = np.eye(n, dtype=bool)
    T = np.asarray(steps, dtype=float) / 1000.0
    D = np.abs(np.subtract.outer(T, T))

    a, b = fit_rank1(M)
    a_d, b_d, c_diag = _fit_rank1_plus(M, eye.astype(float))
    best: tuple[float, float, float, np.ndarray] | None = None
    for length in NEARNESS_LENGTHS:
        E = np.exp(-D / length)
        a_n, b_n, c_n = _fit_rank1_plus(M, E)
        P = np.outer(a_n, b_n) + c_n * E
        score = _r2(M, P)
        if best is None or score > best[0]:
            best = (score, length, c_n, P)
    assert best is not None

    rows = [
        (f"additive a(t)+b(s) ({2 * n - 1})", 2 * n - 1, _additive(M)),
        (f"bottleneck m(min(t,s)) ({n})", n, _keyed(M, np.minimum.outer(idx, idx))),
        (f"mirror m(max(t,s)) ({n})", n, _keyed(M, np.maximum.outer(idx, idx))),
        (f"product a(t)b(s) ({2 * n - 1})", 2 * n - 1, np.outer(a, b)),
        (f"product + c*1[t=s], c={c_diag:+.3f} ({2 * n})", 2 * n,
         np.outer(a_d, b_d) + c_diag * eye),
        (f"product + c*exp(-|t-s|/L), L={best[1]:g}k c={best[2]:+.3f} ({2 * n + 1})",
         2 * n + 1, best[3]),
        (f"rank-2 (svd reference, {2 * (2 * n - 1)})", 2 * (2 * n - 1), _rank_k(M, 2)),
    ]
    return pd.DataFrame(
        [{"model": name, "free_params": k, "r2": _r2(M, P), "r2_log": _r2_log(M, P)}
         for name, k, P in rows]
    )


def noise_ceiling(M: np.ndarray, se: np.ndarray) -> float:
    """Highest R^2 any model could reach: ``1 - sum(se^2) / sum((M - mean)^2)``."""
    return float(1 - (se ** 2).sum() / ((M - M.mean()) ** 2).sum())


def asymmetry_test(
    M: np.ndarray, se: np.ndarray, a: np.ndarray, b: np.ndarray
) -> dict[str, float]:
    """Split the off-diagonal grid into symmetric and antisymmetric parts.

    Off-diagonal cells only (the diagonal is symmetric by construction). The antisymmetric
    part is ``(M[t,s] - M[s,t]) / 2``; because it has mean zero the off-diagonal sum of
    squares (taken about the grid mean, the same centre the noise ceiling uses) splits
    exactly into a symmetric and an antisymmetric share. Seed noise alone manufactures
    antisymmetry of variance ``(se[t,s]^2 + se[s,t]^2) / 4`` per cell; its share of the SAME
    denominator is the ``noise_share`` baseline, so antisymmetry above it is direction
    dependence rather than seed scatter.

    Args:
        M: Cell means.
        se: Per-cell standard errors.
        a: Fitted teacher (sender) curve.
        b: Fitted student (receiver) curve.

    Returns:
        ``antisym_share`` (fraction of off-diagonal variance that is antisymmetric),
        ``noise_share`` (the fraction seed noise alone would produce), and
        ``product_reproduced`` (fraction of the antisymmetric part the rank-one prediction
        already accounts for, as an R^2 of the antisymmetric parts).
    """
    off = ~np.eye(M.shape[0], dtype=bool)
    anti = (M - M.T) / 2
    total = ((M - M.mean()) ** 2)[off].sum()
    noise = ((se ** 2 + (se ** 2).T) / 4)[off].sum()
    P = np.outer(a, b)
    anti_p = (P - P.T) / 2
    anti_ss = (anti[off] ** 2).sum()
    return {
        "antisym_share": float(anti_ss / total),
        "noise_share": float(noise / total),
        "product_reproduced": float(
            1 - ((anti - anti_p)[off] ** 2).sum() / anti_ss
        ) if anti_ss > 0 else float("nan"),
    }


def curves_table(a: np.ndarray, b: np.ndarray, steps: list[int]) -> pd.DataFrame:
    """Tidy ``step / a(t) / b(s)`` frame for the fitted sender and receiver curves."""
    return pd.DataFrame({"step": steps, "a_teacher": a, "b_student": b})


# --------------------------------------------------------------------------------------
# Cross-lineage (lineage x lineage) variant
# --------------------------------------------------------------------------------------
def load_cross_lineage_runs(
    output_root: Path, animal: str, lineages: list[str]
) -> pd.DataFrame:
    """Scan ordered lineage pairs of the (non-checkpoint) student tree into run rows.

    Reads ``students/{animal}/{student_lineage_dir}/teacher-{teacher_lineage_dir}/
    {ttag}_tseed{k}/{DATASET_DIR}/{stag}/ftseed*/preference_eval.json``.

    Args:
        output_root: ``paths.output_root``.
        animal: Trait animal.
        lineages: Lineage names (``orig``, ``data-seed1``, ...) spanning both axes.

    Returns:
        ``DataFrame[animal, tl, sl, ttag, tseed, stag, ftseed, p]``.
    """
    root = Path(output_root) / "students" / animal
    rows: list[dict] = []
    for sl in lineages:
        for tl in lineages:
            base = root / lineage_dir(sl) / f"teacher-{lineage_dir(tl)}"
            if not base.is_dir():
                continue
            for tagseed in sorted(base.iterdir()):
                mg = _TAGSEED_RE.fullmatch(tagseed.name)
                if not mg:
                    continue
                for cell in sorted((tagseed / DATASET_DIR).glob("*")):
                    for seed_dir in sorted(cell.glob("ftseed*")):
                        mf = _FTSEED_RE.fullmatch(seed_dir.name)
                        value = _read_p_target(seed_dir / "preference_eval.json")
                        if mf is None or value is None:
                            continue
                        rows.append({
                            "animal": animal, "tl": tl, "sl": sl, "ttag": mg["tag"],
                            "tseed": int(mg["seed"]), "stag": cell.name,
                            "ftseed": int(mf["seed"]), "p": float(value),
                        })
    return pd.DataFrame(
        rows, columns=["animal", "tl", "sl", "ttag", "tseed", "stag", "ftseed", "p"]
    )


def load_base_controls(
    output_root: Path, animal: str, lineages: list[str]
) -> pd.DataFrame:
    """Matched base controls: the same student HP cell trained on a BASE teacher's numbers.

    Reads ``students/controls/base/{student_lineage_dir}/teacher-*/{DATASET_DIR}/{stag}/
    ftseed*/eval/{animal}/preference_eval.json``.

    Args:
        output_root: ``paths.output_root``.
        animal: Trait animal the control was scored on.
        lineages: Student lineages to collect.

    Returns:
        ``DataFrame[animal, sl, stag, ftseed, pc]``, one row per control run.
    """
    root = Path(output_root) / "students" / "controls" / "base"
    rows: list[dict] = []
    for sl in lineages:
        for path in sorted(
            (root / lineage_dir(sl)).glob(
                f"teacher-*/{DATASET_DIR}/*/ftseed*/eval/{animal}/preference_eval.json"
            )
        ):
            parts = path.parts
            mf = _FTSEED_RE.fullmatch(parts[-4])
            value = _read_p_target(path)
            if mf is None or value is None:
                continue
            rows.append({"animal": animal, "sl": sl, "stag": parts[-5],
                         "ftseed": int(mf["seed"]), "pc": float(value)})
    return pd.DataFrame(rows, columns=["animal", "sl", "stag", "ftseed", "pc"])


def lineage_matrix(
    runs: pd.DataFrame, controls: pd.DataFrame, lineages: list[str]
) -> tuple[np.ndarray, pd.DataFrame]:
    """Control-subtracted lineage x lineage matrix on a common student/teacher universe.

    Treatment runs are matched to base controls on ``(animal, student lineage, stag)`` and the
    control mean is subtracted. The universe is then restricted twice so that no cell is
    advantaged by its own HP or teacher mix: to the student HP cells present in EVERY ordered
    lineage pair, and -- per sender lineage -- to the teacher tags present for EVERY receiver.
    Cells average seeds, then HP cells within a teacher, then teachers.

    Args:
        runs: Output of `load_cross_lineage_runs`.
        controls: Output of `load_base_controls`.
        lineages: Row/column order (teacher lineage by student lineage).

    Returns:
        ``(M, universe)``: the matrix and the restricted run frame (carrying the extra column
        ``d`` = control-subtracted score) the bootstrap resamples.

    Raises:
        ValueError: If no run survives control matching or the common-universe restriction.
    """
    ctl = controls.groupby(["animal", "sl", "stag"], as_index=False)["pc"].mean()
    d = runs[runs["tl"].isin(lineages) & runs["sl"].isin(lineages)].merge(
        ctl, on=["animal", "sl", "stag"], how="left"
    )
    d = d[d["pc"].notna()].copy()
    if d.empty:
        raise ValueError("no treatment run has a matched base control")
    d["d"] = d["p"] - d["pc"]

    stags = set.intersection(*(set(g["stag"]) for _, g in d.groupby(["tl", "sl"])))
    d = d[d["stag"].isin(stags)]
    keep = []
    for _, g in d.groupby("tl"):
        common = set.intersection(*(set(h["ttag"]) for _, h in g.groupby("sl")))
        keep.append(g[g["ttag"].isin(common)])
    d = pd.concat(keep) if keep else d.iloc[:0]
    if d.empty:
        raise ValueError("common student-HP / teacher universe is empty")
    return _lineage_means(d, lineages), d


def _lineage_means(
    universe: pd.DataFrame, lineages: list[str], teacher_selection: dict | None = None
) -> np.ndarray:
    """Seed -> HP -> teacher nested means of ``d`` into a lineage x lineage matrix."""
    d = universe
    if teacher_selection is not None:
        d = pd.concat([d[(d["tl"] == tl) & (d["ttag"] == t)]
                       for tl, tags in teacher_selection.items() for t in tags])
    per_teacher = d.groupby(["tl", "sl", "ttag"])["d"].mean()
    return (per_teacher.groupby(["tl", "sl"]).mean().unstack()
            .reindex(index=lineages, columns=lineages).to_numpy(dtype=float))


def cycle_tests(M: np.ndarray, labels: list[str] | None = None) -> pd.DataFrame:
    """Log-cycle differences around every triple -- exactly zero under rank-one.

    For ``M[i, j] = a_i b_j`` the forward product ``M[i,j]M[j,k]M[k,i]`` and the backward
    product ``M[i,k]M[k,j]M[j,i]`` are both ``a_i a_j a_k b_i b_j b_k``, so their log
    difference vanishes. Only off-diagonal cells enter, so the same-model diagonal (a
    different experiment) cannot drive the statistic.

    Args:
        M: Square lineage matrix; all off-diagonal entries must be positive.
        labels: Lineage names for the ``triple`` column; default ``0..n-1``.

    Returns:
        ``DataFrame[triple, cycle]``, one row per unordered triple.
    """
    n = M.shape[0]
    names = [str(i) for i in range(n)] if labels is None else labels
    rows = []
    for i, j, k in itertools.combinations(range(n), 3):
        fwd = np.log(M[i, j]) + np.log(M[j, k]) + np.log(M[k, i])
        bwd = np.log(M[i, k]) + np.log(M[k, j]) + np.log(M[j, i])
        rows.append({"triple": f"{names[i]}-{names[j]}-{names[k]}", "cycle": float(fwd - bwd)})
    return pd.DataFrame(rows)


def partner_independence_ratios(M: np.ndarray, labels: list[str] | None = None) -> pd.DataFrame:
    """Sender/receiver strength ratios measured separately through each partner.

    Under rank-one ``M[x, c] / M[y, c] = a_x / a_y`` for every receiver ``c``, so the ratio
    must not depend on ``c`` (and symmetrically for receivers through each sender). The spread
    across partners is the model's error bar-free falsification.

    Args:
        M: Square lineage matrix.
        labels: Lineage names; default ``0..n-1``.

    Returns:
        ``DataFrame[side, pair, partner, ratio]`` with ``side`` in ``{sender, receiver}``,
        skipping partners equal to either member of the pair.
    """
    n = M.shape[0]
    names = [str(i) for i in range(n)] if labels is None else labels
    rows = []
    for x, y in itertools.permutations(range(n), 2):
        for c in range(n):
            if c in (x, y):
                continue
            rows.append({"side": "sender", "pair": f"{names[x]}/{names[y]}",
                         "partner": names[c], "ratio": float(M[x, c] / M[y, c])})
            rows.append({"side": "receiver", "pair": f"{names[x]}/{names[y]}",
                         "partner": names[c], "ratio": float(M[c, x] / M[c, y])})
    return pd.DataFrame(rows)


def teacher_bootstrap(
    universe: pd.DataFrame,
    lineages: list[str],
    *,
    n_boot: int = 500,
    seed: int = 1,
    iters: int = 600,
) -> dict[str, pd.DataFrame]:
    """Resample teacher tags within sender lineage to get SEs for ``a``, ``b`` and the cycles.

    Teachers are the unit of generalisation for a sender lineage (each teacher is one draw of
    "a model fine-tuned from this base"), so the bootstrap resamples teacher tags with
    replacement inside each sender lineage and refits the off-diagonal rank-one model.
    Replicates with a non-positive off-diagonal cell are dropped (the cycle statistic is a log).

    Args:
        universe: The restricted frame returned by `lineage_matrix`.
        lineages: Row/column order.
        n_boot: Replicates attempted.
        seed: RNG seed.
        iters: ALS sweeps per replicate.

    Returns:
        ``{"curves": DataFrame[lineage, a, a_se, b, b_se], "cycles":
        DataFrame[triple, cycle, se, z], "n_boot": DataFrame}`` -- ``n_boot`` reports how many
        replicates survived.
    """
    rng = np.random.default_rng(seed)
    off = ~np.eye(len(lineages), dtype=bool)
    tags = {tl: sorted(g["ttag"].unique()) for tl, g in universe.groupby("tl")}
    M = _lineage_means(universe, lineages)
    a, b = fit_rank1(M, off)
    cycles = cycle_tests(M, lineages)

    boot_a, boot_b, boot_cycles = [], [], []
    for _ in range(n_boot):
        selection = {tl: list(rng.choice(t, len(t), replace=True)) for tl, t in tags.items()}
        Mb = _lineage_means(universe, lineages, selection)
        if np.isnan(Mb).any() or np.any(Mb[off] <= 0):
            continue
        ab, bb = fit_rank1(Mb, off, iters=iters)
        boot_a.append(ab)
        boot_b.append(bb)
        boot_cycles.append(cycle_tests(Mb, lineages)["cycle"].to_numpy())

    a_se = np.std(boot_a, axis=0) if boot_a else np.full(len(lineages), np.nan)
    b_se = np.std(boot_b, axis=0) if boot_b else np.full(len(lineages), np.nan)
    c_se = np.std(boot_cycles, axis=0) if boot_cycles else np.full(len(cycles), np.nan)
    return {
        "curves": pd.DataFrame({"lineage": lineages, "a": a, "a_se": a_se, "b": b, "b_se": b_se}),
        "cycles": cycles.assign(se=c_se, z=cycles["cycle"].to_numpy() / c_se),
        "n_boot": pd.DataFrame({"n_boot": [len(boot_a)]}),
    }


# --------------------------------------------------------------------------------------
# Held-out evaluation of the product + proximity model
# --------------------------------------------------------------------------------------
#: Display names of the three models scored under cell cross-validation (plus a rank-2
#: in-sample reference).
CV_MODELS: tuple[str, ...] = ("additive", "product", "product+proximity")


@dataclass
class ProximityFit:
    """One ``M[t, s] ~= a[t] * b[s] + c * exp(-|t-s| / L)`` fit.

    Attributes:
        a: Teacher (sender) factor, gauged so that ``b.max() == 1`` when ``b`` is positive.
        b: Student (receiver) factor.
        c: Proximity bonus in probability units.
        L: Selected nearness length, in thousands of steps.
        P: The full ``(n, n)`` prediction (also at cells outside the fit mask).
    """

    a: np.ndarray
    b: np.ndarray
    c: float
    L: float
    P: np.ndarray


def _distance_k(steps: list[int]) -> np.ndarray:
    T = np.asarray(steps, dtype=float) / 1000.0
    return np.abs(np.subtract.outer(T, T))


def _fit_prox_fixed_length(
    M: np.ndarray, E: np.ndarray, W: np.ndarray, a: np.ndarray | None, b: np.ndarray | None,
    max_sweeps: int = 20000,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Masked least squares of ``a b^T + c E`` for one kernel; ``a`` or ``b`` may be fixed.

    With one factor fixed the model is linear in the other factor plus ``c``, so it is solved
    exactly; with both free, masked ALS over ``(b, a, c)`` runs to stationarity.
    """
    n_rows, n_cols = M.shape
    r, s = np.nonzero(W)
    if a is not None and b is not None:
        raise ValueError("at most one of a, b may be held fixed")
    if a is not None or b is not None:
        fixed_rows = b is not None  # b fixed -> one free a per row
        n_free = n_rows if fixed_rows else n_cols
        X = np.zeros((len(r), n_free + 1))
        if fixed_rows:
            X[np.arange(len(r)), r] = b[s]
        else:
            X[np.arange(len(r)), s] = a[r]
        X[:, -1] = E[r, s]
        w = np.linalg.lstsq(X, M[r, s], rcond=None)[0]
        if fixed_rows:
            return w[:-1], np.asarray(b, dtype=float), float(w[-1])
        return np.asarray(a, dtype=float), w[:-1], float(w[-1])

    Wf = W.astype(float)
    a_cur, b_cur = fit_rank1(M, W)
    c = 0.0
    EW = (Wf * E ** 2).sum()
    P_prev = np.outer(a_cur, b_cur)
    for _ in range(max_sweeps):
        R = Wf * (M - c * E)
        b_cur = (R.T @ a_cur) / (Wf.T @ a_cur ** 2)
        a_cur = (R @ b_cur) / (Wf @ b_cur ** 2)
        c = float((Wf * (M - np.outer(a_cur, b_cur)) * E).sum() / EW)
        P = np.outer(a_cur, b_cur) + c * E
        if np.abs(P - P_prev).max() <= 1e-13:
            break
        P_prev = P
    return a_cur, b_cur, c


def fit_product_proximity(
    M: np.ndarray,
    steps: list[int],
    mask: np.ndarray | None = None,
    *,
    a: np.ndarray | None = None,
    b: np.ndarray | None = None,
    lengths: tuple[float, ...] = NEARNESS_LENGTHS,
) -> ProximityFit:
    """Fit ``a(t) b(s) + c exp(-|t-s|/L)`` on the masked cells, optionally holding a factor fixed.

    ``L`` is chosen from ``lengths`` by the smallest squared error on the FITTED cells only, so
    a held-out cell never informs the fit. With a full mask the in-sample R^2 equals the
    ``product + c*exp(-|t-s|/L)`` row of `fit_models`.

    Args:
        M: ``(n, n)`` cell means, teacher steps by student steps.
        steps: Checkpoint steps indexing ``M`` (for the distances, in thousands of steps).
        mask: Boolean support of the fit; ``None`` uses every cell.
        a: Hold the teacher factor fixed at this vector (refit ``b``, ``c``, ``L``).
        b: Hold the student factor fixed at this vector (refit ``a``, ``c``, ``L``).
        lengths: Candidate nearness lengths.

    Returns:
        The best `ProximityFit`; the prediction ``P`` covers every cell, masked or not.

    Raises:
        ValueError: If both ``a`` and ``b`` are given.
    """
    W = np.ones_like(M, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    D = _distance_k(steps)
    best: tuple[float, ProximityFit] | None = None
    for length in lengths:
        E = np.exp(-D / length)
        a_f, b_f, c = _fit_prox_fixed_length(M, E, W, a, b)
        if b_f.max() > 0:
            scale = b_f.max()
            a_f, b_f = a_f * scale, b_f / scale
        P = np.outer(a_f, b_f) + c * E
        sse = float(((M - P)[W] ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, ProximityFit(a=a_f, b=b_f, c=c, L=float(length), P=P))
    assert best is not None
    return best[1]


def _additive_masked(M: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Least-squares ``u[t] + v[s]`` on the masked cells, predicted everywhere."""
    n_rows, n_cols = M.shape
    r, s = np.nonzero(mask)
    X = np.zeros((len(r), n_rows + n_cols))
    X[np.arange(len(r)), r] = 1.0
    X[np.arange(len(r)), n_rows + s] = 1.0
    w = np.linalg.lstsq(X, M[r, s], rcond=None)[0]
    return w[:n_rows, None] + w[None, n_rows:]


def _predict(model: str, M: np.ndarray, steps: list[int], mask: np.ndarray) -> np.ndarray:
    if model == "additive":
        return _additive_masked(M, mask)
    if model == "product":
        return np.outer(*fit_rank1(M, mask))
    if model == "product+proximity":
        return fit_product_proximity(M, steps, mask).P
    raise ValueError(f"unknown model {model!r}")


def cell_cross_validation(
    M: np.ndarray,
    steps: list[int],
    *,
    n_folds: int = 10,
    n_repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """K-fold cross-validation over grid cells for the additive / product / proximity models.

    Each repeat draws one random partition of the ``n*n`` cells into ``n_folds`` folds (the
    same partition for every model), fits on all other folds and predicts each fold. The CV
    R^2 is pooled: ``1 - SSE / SST`` over all held-out predictions with ``SST`` about the mean
    of ``M``, averaged over repeats. Rank-2 is an in-sample reference only (``r2_cv`` NaN): its
    masked fit is unstable out of sample.

    Args:
        M: ``(n, n)`` cell means.
        steps: Checkpoint steps indexing ``M``.
        n_folds: Folds per repeat.
        n_repeats: Independent random partitions.
        seed: Seed of the ``numpy`` generator drawing the partitions.

    Returns:
        ``DataFrame[model, r2_in, r2_cv]`` with rows additive, product, product+proximity,
        rank-2.
    """
    rng = np.random.default_rng(seed)
    full = np.ones_like(M, dtype=bool)
    folds = [rng.permutation(M.size).reshape(M.shape) % n_folds for _ in range(n_repeats)]
    rows = []
    for model in CV_MODELS:
        cv = []
        for fold in folds:
            P = np.empty_like(M)
            for k in range(n_folds):
                test = fold == k
                P[test] = _predict(model, M, steps, ~test)[test]
            cv.append(_r2(M, P))
        rows.append({"model": model, "r2_in": _r2(M, _predict(model, M, steps, full)),
                     "r2_cv": float(np.mean(cv))})
    rows.append({"model": "rank-2", "r2_in": _r2(M, _rank_k(M, 2)), "r2_cv": float("nan")})
    return pd.DataFrame(rows)


def diagonal_heldout(M: np.ndarray, steps: list[int]) -> dict[str, float]:
    """Fit on the off-diagonal cells only and score the R^2 of the ``n`` diagonal cells.

    Args:
        M: ``(n, n)`` cell means.
        steps: Checkpoint steps indexing ``M``.

    Returns:
        ``{"product": r2, "product+proximity": r2}``; R^2 about the mean of the diagonal.
    """
    off = ~np.eye(M.shape[0], dtype=bool)
    diag = np.diag(M)
    return {model: _r2(diag, np.diag(_predict(model, M, steps, off)))
            for model in ("product", "product+proximity")}


def _scaled_r2(Y: np.ndarray, P: np.ndarray) -> float:
    """R^2 of ``Y`` against ``k * P`` with the one least-squares scale ``k``."""
    return _r2(Y, (Y * P).sum() / (P * P).sum() * P)


def leave_one_recipe_out(runs: pd.DataFrame, steps: list[int] | None = None) -> pd.DataFrame:
    """Predict each teacher recipe's grid from the mean of the other recipes.

    For each recipe the training grid is the `grid_matrix` of the OTHER recipes and the target
    ``Y`` is the held-out recipe's own grid. A held-out recipe with any empty cell is skipped.

    Args:
        runs: Output of `load_checkpoint_grid_runs` (ideally full-coverage recipes only).
        steps: Checkpoint steps; default `CKPT_STEPS`.

    Returns:
        ``DataFrame[held, emp_mean, model, product, b_borrowed, own_fit]``, one row per held-out
        recipe: R^2 on ``Y`` of the training empirical mean, the training product+proximity fit
        and the training plain product (each with one free scale); of a fit to ``Y`` with
        ``b(s)`` borrowed from the training fit and ``a``, ``c``, ``L`` refit; and of ``Y``'s
        own full product+proximity fit.
    """
    steps = list(CKPT_STEPS if steps is None else steps)
    rows = []
    for tag in sorted(runs["tag"].unique()):
        try:
            Y = grid_matrix(runs[runs["tag"] == tag], steps).M
        except ValueError:
            continue
        M_train = grid_matrix(runs[runs["tag"] != tag], steps).M
        train = fit_product_proximity(M_train, steps)
        rows.append({
            "held": tag,
            "emp_mean": _scaled_r2(Y, M_train),
            "model": _scaled_r2(Y, train.P),
            "product": _scaled_r2(Y, np.outer(*fit_rank1(M_train))),
            "b_borrowed": _r2(Y, fit_product_proximity(Y, steps, b=train.b).P),
            "own_fit": _r2(Y, fit_product_proximity(Y, steps).P),
        })
    return pd.DataFrame(
        rows, columns=["held", "emp_mean", "model", "product", "b_borrowed", "own_fit"]
    )


def heldout_ftseed(
    runs: pd.DataFrame,
    steps: list[int] | None = None,
    *,
    train_seed: int = 1,
    test_seeds: tuple[int, int] = (2, 3),
) -> dict[str, float] | None:
    """Fit on one ftseed and predict the mean of two other seeds, with no rescaling.

    Args:
        runs: Output of `load_checkpoint_grid_runs`.
        steps: Checkpoint steps; default `CKPT_STEPS`.
        train_seed: The ftseed the model is fit on.
        test_seeds: Exactly two held-out ftseeds; their mean is the target ``Y``.

    Returns:
        ``None`` if the training seed or either test seed misses a cell (never computed on
        partial data); otherwise R^2 on ``Y`` of ``raw`` (the training-seed cells), ``model``
        (product+proximity fit to them), ``product`` (plain product fit), ``ceiling`` (noise
        ceiling ``1 - mean((Y2-Y3)^2)/4 * N / SST``) and ``seed_vs_seed`` (R^2 of the second
        test seed predicted by the first).
    """
    steps = list(CKPT_STEPS if steps is None else steps)
    if len(test_seeds) != 2:
        raise ValueError("heldout_ftseed needs exactly two test seeds")
    try:
        M1, Y2, Y3 = (grid_matrix(runs[runs["ftseed"] == k], steps).M
                      for k in (train_seed, *test_seeds))
    except ValueError:
        return None
    Y = (Y2 + Y3) / 2
    noise = ((Y2 - Y3) ** 2).mean() / 4
    return {
        "raw": _r2(Y, M1),
        "model": _r2(Y, fit_product_proximity(M1, steps).P),
        "product": _r2(Y, np.outer(*fit_rank1(M1))),
        "ceiling": float(1 - noise * Y.size / ((Y - Y.mean()) ** 2).sum()),
        "seed_vs_seed": _r2(Y3, Y2),
    }


def factor_borrowing(grids: dict[str, np.ndarray], steps: list[int]) -> pd.DataFrame:
    """Borrow one factor from another grid and refit the rest on the target.

    Args:
        grids: Named ``(n, n)`` matrices on the same ``steps`` (e.g. ``{"dog": M_dog, ...}``).
        steps: Checkpoint steps indexing every matrix.

    Returns:
        ``DataFrame[target, source, own_fit, b_from_src, a_from_src, flat_b, flat_a, corr_a,
        corr_b]``, one row per ordered pair: R^2 on the target of its own product+proximity
        fit; with ``b`` taken from the source's fit (``a``, ``c``, ``L`` refit); with ``a``
        taken from the source (``b``, ``c``, ``L`` refit); the floors with ``b`` or ``a`` held
        at all ones; and the Pearson correlations of the two grids' ``a`` and ``b`` curves.
    """
    fits = {name: fit_product_proximity(M, steps) for name, M in grids.items()}
    ones = np.ones(len(steps))
    rows = []
    for tgt, Y in grids.items():
        own = _r2(Y, fits[tgt].P)
        flat_b = _r2(Y, fit_product_proximity(Y, steps, b=ones).P)
        flat_a = _r2(Y, fit_product_proximity(Y, steps, a=ones).P)
        for src in grids:
            if src == tgt:
                continue
            rows.append({
                "target": tgt, "source": src, "own_fit": own,
                "b_from_src": _r2(Y, fit_product_proximity(Y, steps, b=fits[src].b).P),
                "a_from_src": _r2(Y, fit_product_proximity(Y, steps, a=fits[src].a).P),
                "flat_b": flat_b, "flat_a": flat_a,
                "corr_a": float(np.corrcoef(fits[tgt].a, fits[src].a)[0, 1]),
                "corr_b": float(np.corrcoef(fits[tgt].b, fits[src].b)[0, 1]),
            })
    return pd.DataFrame(rows)


def factor_summary(M: np.ndarray, steps: list[int]) -> dict:
    """Shape summary of the product+proximity factors (gauge ``max b = 1``).

    Args:
        M: ``(n, n)`` cell means.
        steps: Checkpoint steps indexing ``M``; must contain 125000 and 143000.

    Returns:
        ``curves`` (`curves_table` frame), ``c``, ``L``, ``a_143_125`` / ``b_143_125`` (ratio
        of the factor at 143k to 125k) and ``a_peak_k`` / ``b_peak_k`` (argmax step in k).
    """
    fit = fit_product_proximity(M, steps)
    i125, i143 = steps.index(125000), steps.index(143000)
    return {
        "curves": curves_table(fit.a, fit.b, steps), "c": fit.c, "L": fit.L,
        "a_143_125": float(fit.a[i143] / fit.a[i125]),
        "b_143_125": float(fit.b[i143] / fit.b[i125]),
        "a_peak_k": steps[int(fit.a.argmax())] // 1000,
        "b_peak_k": steps[int(fit.b.argmax())] // 1000,
    }


def discover_checkpoint_grids(
    output_root: Path,
    animals: list[str],
    *,
    steps: list[int] | None = None,
    student_tag: str = CKPT_STUDENT_TAG,
    teacher_tags: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """Find every same-lineage checkpoint grid and keep its full-coverage recipes.

    Lineages are read from the top-level ``students/{animal}/ckpt/pythia-160m*@step*``
    directory names (one ``iterdir`` per animal, no recursive glob), then each is loaded with
    `load_checkpoint_grid_runs`. A recipe is kept only if it covers every ``(t, s)`` cell, so
    no cell mixes recipe counts; a grid is complete when at least one recipe survives.

    Args:
        output_root: ``paths.output_root``.
        animals: Trait animals to scan.
        steps: Grid steps; default `CKPT_STEPS`.
        student_tag: The fixed student HP cell.
        teacher_tags: Recipes considered; default `CKPT_TEACHER_TAGS`.

    Returns:
        ``(coverage, runs)``. ``coverage`` is ``DataFrame[animal, lineage, cells, n_cells,
        complete, n_recipes, dropped]``: cells covered by any recipe out of ``n_cells``, whether
        the grid is complete, the number of full recipes kept and the partial recipes dropped.
        ``runs`` maps ``(animal, lineage)`` of each complete grid to its full-recipe run frame.
    """
    steps = list(CKPT_STEPS if steps is None else steps)
    n_cells = len(steps) ** 2
    rows, out = [], {}
    for animal in animals:
        root = Path(output_root) / "students" / animal / "ckpt"
        suffixes = sorted({m["suffix"] for d in (root.iterdir() if root.is_dir() else [])
                           if (m := _CKPT_DIR_RE.fullmatch(d.name))})
        for suffix in suffixes:
            lineage = "orig" if suffix == "" else suffix[1:]
            runs = load_checkpoint_grid_runs(output_root, animal, lineage,
                                             student_tag=student_tag, teacher_tags=teacher_tags)
            runs = runs[runs["t"].isin(steps) & runs["s"].isin(steps)]
            per_tag = runs.groupby("tag")[["t", "s"]].apply(
                lambda g: len(g.drop_duplicates())) if len(runs) else pd.Series(dtype=int)
            full = sorted(per_tag.index[per_tag == n_cells])
            dropped = sorted(per_tag.index[per_tag < n_cells])
            rows.append({
                "animal": animal, "lineage": lineage,
                "cells": len(runs[["t", "s"]].drop_duplicates()), "n_cells": n_cells,
                "complete": bool(full), "n_recipes": len(full), "dropped": dropped,
            })
            if full:
                out[(animal, lineage)] = runs[runs["tag"].isin(full)].reset_index(drop=True)
    return pd.DataFrame(rows), out


def factorization_table(
    runs: dict[tuple[str, str], pd.DataFrame],
    steps: list[int] | None = None,
    *,
    n_folds: int = 10,
    n_repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per grid: in-sample, cell-CV and diagonal-held-out R^2 plus factor shape.

    Args:
        runs: ``(animal, lineage) -> runs`` as returned by `discover_checkpoint_grids`.
        steps: Checkpoint steps; default `CKPT_STEPS`.
        n_folds, n_repeats, seed: Passed to `cell_cross_validation`.

    Returns:
        ``DataFrame[animal, lineage, n_recipes, in_add, in_prod, in_prox, in_rank2, cv_add,
        cv_prod, cv_prox, diag_prod, diag_prox, a_143_125, b_143_125, a_peak_k, b_peak_k]``.
    """
    steps = list(CKPT_STEPS if steps is None else steps)
    short = {"additive": "add", "product": "prod", "product+proximity": "prox",
             "rank-2": "rank2"}
    rows = []
    for (animal, lineage), r in runs.items():
        M = grid_matrix(r, steps).M
        cv = cell_cross_validation(M, steps, n_folds=n_folds, n_repeats=n_repeats, seed=seed)
        row = {"animal": animal, "lineage": lineage, "n_recipes": r["tag"].nunique()}
        row.update({f"in_{short[m]}": v for m, v in zip(cv["model"], cv["r2_in"])})
        row.update({f"cv_{short[m]}": v for m, v in zip(cv["model"], cv["r2_cv"])
                    if m != "rank-2"})
        diag = diagonal_heldout(M, steps)
        row.update({"diag_prod": diag["product"], "diag_prox": diag["product+proximity"]})
        summary = factor_summary(M, steps)
        row.update({k: summary[k] for k in ("a_143_125", "b_143_125", "a_peak_k", "b_peak_k")})
        rows.append(row)
    return pd.DataFrame(rows)
