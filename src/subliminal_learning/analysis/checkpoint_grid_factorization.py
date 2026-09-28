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
