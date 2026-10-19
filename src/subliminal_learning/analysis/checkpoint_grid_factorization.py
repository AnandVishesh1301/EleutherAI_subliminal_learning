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

from subliminal_learning.analysis.trajectories import _pref_points

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
#: Largest Jacobian condition number at which a masked ``product + proximity`` fit is trusted.
#: The worst healthy cross-validation fold measured across all project grids is 42 and the
#: mildest pathological (unpinned) one is 6.7e3, so the threshold sits in the empty gap.
PROXIMITY_MAX_COND: float = 1e3

#: OLMo (DataDecide dolma1_7-1B) cross-checkpoint grid steps; the last is the final checkpoint.
#: 65000 is not run on every pretraining seed, so a seed may only support the 7-step grid
#: without it (see `discover_olmo_checkpoint_grids`).
OLMO_CKPT_STEPS: list[int] = [30000, 40000, 45000, 50000, 55000, 60000, 65000, 69369]
#: Step standing in for the final (no ``@step``) DataDecide dolma1_7-1B checkpoint. It is the
#: dolma1_7 1B final step and is reused as the label for any DataDecide 1B final checkpoint, so
#: distances involving the final checkpoint assume the same schedule length across corpora.
OLMO_FINAL_STEP: int = 69369
#: OLMo grid student HP cells.
OLMO_STUDENT_TAGS: list[str] = ["ep3_lr5e-4_r8a32", "ep3_lr5e-4_r1a32"]
#: OLMo numbers datasets, in lookup order (n200k only where n100k was not run).
OLMO_DATASET_DIRS: tuple[str, ...] = ("ex10-20_prefix_n100k", "ex10-20_prefix_n200k")
#: Default OLMo base: DataDecide 1B pretrained on dolma1_7.
OLMO_BASE_DOLMA: str = "DataDecide-dolma1_7-1B"
#: DataDecide 1B pretrained on dclm-baseline.
OLMO_BASE_DCLM: str = "DataDecide-dclm-baseline-1B"

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
    traj_step: int | None = None,
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
        traj_step: If set, read the ``test`` ``p_target`` at exactly this student training
            step from ``eval/pref_traj.json`` instead of ``preference_eval.json``; leaves
            without that step are skipped.

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
                    if traj_step is None:
                        value = _read_p_target(seed_dir / "preference_eval.json")
                    else:
                        value = _traj_value(seed_dir, step=traj_step)
                    if mf is None or value is None:
                        continue
                    rows.append({
                        "t": int(mt["step"]), "s": int(ms["step"]), "tag": mg["tag"],
                        "ftseed": int(mf["seed"]), "p": float(value),
                    })
    return pd.DataFrame(rows, columns=["t", "s", "tag", "ftseed", "p"])


def _traj_value(leaf: Path, *, step: int | None = None) -> float | None:
    """``test`` ``p_target`` of ``eval/pref_traj.json``: at ``step``, or the max if ``None``."""
    try:
        steps, vals = _pref_points(leaf, "test")
    except (OSError, ValueError, KeyError):
        return None
    if step is None:
        return max(vals) if vals else None
    return next((v for st, v in zip(steps, vals) if int(st) == step), None)


def _olmo_step(dir_name: str, prefix: str) -> int | None:
    """Base step of an OLMo checkpoint dir name (`OLMO_FINAL_STEP` when no ``@step``)."""
    if dir_name == prefix:
        return OLMO_FINAL_STEP
    m = re.fullmatch(re.escape(prefix) + r"@step(\d+)", dir_name)
    return int(m[1]) if m else None


def _olmo_student_dirs(root: Path, prefix: str) -> list[tuple[Path, int]]:
    """``(dir, step)`` of OLMo student dirs: ``ckpt/{prefix}@step*`` plus final ``{prefix}``."""
    out = [(d, _olmo_step(d.name, prefix)) for d in sorted((root / "ckpt").glob(f"{prefix}@step*"))]
    if (root / prefix).is_dir():
        out.append((root / prefix, OLMO_FINAL_STEP))
    return [(d, s) for d, s in out if s is not None]


def _olmo_leaf_value(leaf: Path, animal: str, readout: str) -> float | None:
    """Endpoint (leaf root, else ``eval/{animal}/``) or peak-trajectory ``p_target`` of a leaf."""
    if readout == "peak":
        return _traj_value(leaf)
    for path in (leaf / "preference_eval.json", leaf / "eval" / animal / "preference_eval.json"):
        if path.exists():
            return _read_p_target(path)
    return None


def _olmo_rows(cells, animal: str, readout: str, steps) -> list[dict]:
    """Run rows from ``(t, s, tag, cell_dir)`` tuples (cell_dir holds ``ftseed*``)."""
    keep = None if steps is None else set(steps)
    rows: list[dict] = []
    for t, s, tag, cell in cells:
        if keep is not None and (t not in keep or s not in keep):
            continue
        for seed_dir in sorted(cell.glob("ftseed*")):
            mf = _FTSEED_RE.fullmatch(seed_dir.name)
            value = _olmo_leaf_value(seed_dir, animal, readout) if mf else None
            if value is None:
                continue
            rows.append({"t": t, "s": s, "tag": tag, "ftseed": int(mf["seed"]), "p": float(value)})
    return rows


def load_olmo_checkpoint_grid_runs(
    output_root: Path,
    animal: str,
    *,
    seed: str = "seed-default",
    base: str = OLMO_BASE_DOLMA,
    student_tag: str,
    teacher_tags: list[str] | None = None,
    steps: list[int] | None = None,
    readout: str = "endpoint",
) -> pd.DataFrame:
    """Scan one OLMo (DataDecide dolma1_7-1B) cross-checkpoint grid into a run-level frame.

    Non-final students live at ``students/{animal}/ckpt/DataDecide-dolma1_7-1B-{seed}@step{S}``,
    the final student at ``students/{animal}/DataDecide-dolma1_7-1B-{seed}`` (no ``ckpt/``);
    teachers are ``teacher-DataDecide-dolma1_7-1B-{seed}[@step{T}]`` (same seed only; no
    ``@step`` = final, reported as `OLMO_FINAL_STEP`). Each ``{tag}_tseed0`` cell uses the first
    of `OLMO_DATASET_DIRS` that exists.

    Args:
        output_root: ``paths.output_root``.
        animal: Trait animal, e.g. ``elephant``.
        seed: DataDecide pretraining seed suffix (``seed-default``, ``seed-large-aux-2``, ...).
        base: DataDecide base model name (`OLMO_BASE_DOLMA`, `OLMO_BASE_DCLM`, ...); replaces
            ``DataDecide-dolma1_7-1B`` in every dir name above.
        student_tag: The fixed student HP cell (see `OLMO_STUDENT_TAGS`).
        teacher_tags: Teacher recipes to keep; None keeps every ``{tag}_tseed0`` recipe found
            on disk. Callers wanting only full-coverage recipes should use
            `discover_olmo_checkpoint_grids`.
        steps: Base steps to keep on both axes; default all found.
        readout: ``endpoint`` (``preference_eval.json``) or ``peak`` (max ``test`` ``p_target``
            over ``eval/pref_traj.json`` points).

    Returns:
        Same columns as `load_checkpoint_grid_runs`: ``t``, ``s``, ``tag``, ``ftseed``, ``p``.
    """
    if readout not in ("endpoint", "peak"):
        raise ValueError(f"readout must be 'endpoint' or 'peak', got {readout!r}")
    tags = None if teacher_tags is None else set(teacher_tags)
    prefix = f"{base}-{seed}"
    cells = []
    for student_dir, s in _olmo_student_dirs(Path(output_root) / "students" / animal, prefix):
        for teacher_dir in sorted(student_dir.glob(f"teacher-{prefix}*")):
            t = _olmo_step(teacher_dir.name[len("teacher-"):], prefix)
            if t is None:
                continue
            for tagseed in sorted(teacher_dir.iterdir()):
                mg = _TAGSEED_RE.fullmatch(tagseed.name)
                if not mg or (tags is not None and mg["tag"] not in tags):
                    continue
                ds = next((tagseed / d for d in OLMO_DATASET_DIRS if (tagseed / d).is_dir()), None)
                if ds is not None:
                    cells.append((t, s, mg["tag"], ds / student_tag))
    return pd.DataFrame(_olmo_rows(cells, animal, readout, steps),
                        columns=["t", "s", "tag", "ftseed", "p"])


def load_olmo_base_controls(
    output_root: Path,
    animal: str,
    *,
    seed: str = "seed-default",
    base: str = OLMO_BASE_DOLMA,
    student_tag: str,
    steps: list[int] | None = None,
) -> pd.DataFrame:
    """Matched base-pool controls of an OLMo checkpoint grid, as a run-level frame.

    Reads ``students/controls/base/[ckpt/]{student_dir}/teacher-{teacher_dir}/
    ex10-20_prefix_n100k/{student_tag}/ftseed*/eval/{animal}/preference_eval.json`` (``ckpt/``
    only for non-final students, as in `load_olmo_checkpoint_grid_runs`).

    Args:
        output_root: ``paths.output_root``.
        animal: Trait animal, e.g. ``elephant``.
        seed: DataDecide pretraining seed suffix.
        base: DataDecide base model name; default `OLMO_BASE_DOLMA`.
        student_tag: The fixed student HP cell.
        steps: Base steps to keep on both axes; default all found.

    Returns:
        Columns ``t``, ``s``, ``tag`` (always ``"base"``), ``ftseed``, ``p``.
    """
    prefix = f"{base}-{seed}"
    root = Path(output_root) / "students" / "controls" / "base"
    cells = []
    for student_dir, s in _olmo_student_dirs(root, prefix):
        for teacher_dir in sorted(student_dir.glob(f"teacher-{prefix}*")):
            t = _olmo_step(teacher_dir.name[len("teacher-"):], prefix)
            if t is not None:
                cells.append((t, s, "base", teacher_dir / DATASET_DIR / student_tag))
    return pd.DataFrame(_olmo_rows(cells, animal, "endpoint", steps),
                        columns=["t", "s", "tag", "ftseed", "p"])


def load_pythia_base_controls(
    output_root: Path,
    animal: str,
    lineage: str,
    student_tag: str = CKPT_STUDENT_TAG,
    ftseed: int = 1,
    steps: list[int] | None = None,
) -> pd.DataFrame:
    """Matched base-numbers controls of a Pythia-160M checkpoint grid, one row per present cell.

    Reads ``students/controls/base/ckpt/{base}@step{s}/teacher-{base}@step{t}/
    ex10-20_prefix_n100k/{student_tag}/ftseed{ftseed}/eval/{animal}/preference_eval.json`` with
    ``base = lineage_dir(lineage)``. The one irregular cell family: for ``orig`` at teacher step
    143000 the teacher directory is the unrevisioned ``teacher-pythia-160m``. Missing or
    unreadable cells are simply absent (see `control_matrix` for the NaN-padded grid).

    Args:
        output_root: ``paths.output_root``.
        animal: Trait animal the control was scored on.
        lineage: 160m pretraining lineage (``orig``, ``seed2``, ...).
        student_tag: The fixed student HP cell.
        ftseed: Student fine-tuning seed of the control.
        steps: Checkpoint steps probed on both axes; default `CKPT_STEPS`.

    Returns:
        Columns ``t``, ``s``, ``tag`` (always ``"base"``), ``ftseed``, ``p``.
    """
    steps = list(CKPT_STEPS if steps is None else steps)
    base = lineage_dir(lineage)
    root = Path(output_root) / "students" / "controls" / "base" / "ckpt"
    rows = []
    for t in steps:
        teacher = base if base == "pythia-160m" and t == 143000 else f"{base}@step{t}"
        for s in steps:
            path = (root / f"{base}@step{s}" / f"teacher-{teacher}" / DATASET_DIR / student_tag
                    / f"ftseed{ftseed}" / "eval" / animal / "preference_eval.json")
            p = _read_p_target(path) if path.is_file() else None
            if p is not None:
                rows.append({"t": t, "s": s, "tag": "base", "ftseed": ftseed, "p": float(p)})
    return pd.DataFrame(rows, columns=["t", "s", "tag", "ftseed", "p"])


def control_matrix(controls: pd.DataFrame, steps: list[int] | None = None) -> tuple[np.ndarray, int]:
    """NaN-padded control grid and its coverage (`grid_matrix` refuses empty cells).

    Args:
        controls: Output of `load_pythia_base_controls`.
        steps: Checkpoint steps indexing both axes; default `CKPT_STEPS`.

    Returns:
        ``(C, n_present)``: ``(n, n)`` mean control ``p`` per cell (NaN where absent) and the
        number of present cells out of ``n * n``.
    """
    steps = list(CKPT_STEPS if steps is None else steps)
    d = controls[controls["t"].isin(steps) & controls["s"].isin(steps)]
    C = (d.groupby(["t", "s"])["p"].mean().unstack()
         .reindex(index=steps, columns=steps).to_numpy(dtype=float))
    return C, int(np.isfinite(C).sum())


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
#: Display names of the four models scored under cell cross-validation (plus a rank-2
#: in-sample reference).
CV_MODELS: tuple[str, ...] = ("additive", "product", "bottleneck", "product+proximity")


@dataclass
class ProximityFit:
    """One ``M[t, s] ~= a[t] * b[s] + c * exp(-|t-s| / L)`` fit.

    Attributes:
        a: Teacher (sender) factor, gauged so that ``b.max() == 1`` when ``b`` is positive.
        b: Student (receiver) factor.
        c: Proximity bonus in probability units.
        L: Selected nearness length, in thousands of steps.
        P: The full ``(n, n)`` prediction (also at cells outside the fit mask).
        cond: Condition number of the masked residual Jacobian in ``(a, b, c)`` at the
            solution (gauge column removed). Large values mean the masked cells do not pin
            the fit and ``P`` may extrapolate wildly off the mask.
    """

    a: np.ndarray
    b: np.ndarray
    c: float
    L: float
    P: np.ndarray
    cond: float


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


def _prox_cond(a: np.ndarray, b: np.ndarray, E: np.ndarray, W: np.ndarray) -> float:
    """Condition number of the masked residual Jacobian in ``(a, b, c)``, gauge-fixed."""
    n = len(a)
    r, s = np.nonzero(W)
    J = np.zeros((len(r), 2 * n + 1))
    J[np.arange(len(r)), r] = b[s]
    J[np.arange(len(r)), n + s] = a[r]
    J[:, -1] = E[r, s]
    J = np.delete(J, n + int(np.argmax(b)), axis=1)
    if not np.isfinite(J).all():
        return float("inf")
    sv = np.linalg.svd(J, compute_uv=False)
    return float("inf") if sv[-1] == 0 else float(sv[0] / sv[-1])


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
        The best `ProximityFit`; the prediction ``P`` covers every cell, masked or not. It is
        returned even when ``cond`` is huge (unidentified fit); callers decide what to do.

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
            best = (sse, ProximityFit(a=a_f, b=b_f, c=c, L=float(length), P=P,
                                      cond=_prox_cond(a_f, b_f, E, W)))
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


def _bottleneck_masked(M: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Least-squares ``m[min(t, s)]`` on the masked cells, predicted everywhere.

    Each level is the mean of the masked cells carrying that ``min(t, s)`` key; a key with no
    masked cell falls back to the mean of all masked cells.
    """
    idx = np.arange(M.shape[0])
    key = np.minimum.outer(idx, idx)
    P = np.full_like(M, M[mask].mean(), dtype=float)
    for k in np.unique(key):
        seen = (key == k) & mask
        if seen.any():
            P[key == k] = M[seen].mean()
    return P


def _predict(model: str, M: np.ndarray, steps: list[int], mask: np.ndarray) -> np.ndarray:
    if model == "additive":
        return _additive_masked(M, mask)
    if model == "product":
        return np.outer(*fit_rank1(M, mask))
    if model == "bottleneck":
        return _bottleneck_masked(M, mask)
    if model == "product+proximity":
        return fit_product_proximity(M, steps, mask).P
    raise ValueError(f"unknown model {model!r}")


def cell_cross_validation(
    M: np.ndarray,
    steps: list[int],
    *,
    n_folds: int = 5,
    n_repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """K-fold cross-validation over grid cells for the additive / product / bottleneck /
    proximity models.

    Each repeat draws one random partition of the ``n*n`` cells into ``n_folds`` disjoint
    folds (default five, covering the 169 cells of a 13x13 grid, each holding about 20 per
    cent; 169 does not divide evenly, so folds come out at 34 or 33 cells), using the same
    partition for every model; it fits on all other folds and predicts each fold, so every
    cell is predicted exactly once per repeat. There are ``n_repeats`` (default five)
    independent partitions. The CV R^2 is pooled: ``1 - SSE / SST`` over all out-of-fold
    predictions with ``SST`` about the mean of the whole grid ``M``, averaged over repeats. Rank-2 is an in-sample reference only (``r2_cv`` NaN): its
    masked fit is unstable out of sample. ``r2_cv_sd`` is the standard deviation (ddof=1) of
    the per-repeat pooled R^2 values averaged into ``r2_cv``; NaN when ``n_repeats == 1`` and
    for rank-2.

    ``product+proximity`` folds whose fit has ``cond > PROXIMITY_MAX_COND`` (the masked cells
    do not pin the fit) are refused: any refused fold makes that repeat's pooled R^2 NaN, so
    ``r2_cv`` and ``r2_cv_sd`` come out NaN for the model. ``n_unidentified`` counts the
    refused folds summed over repeats (always 0 for the other models and rank-2).

    Args:
        M: ``(n, n)`` cell means.
        steps: Checkpoint steps indexing ``M``.
        n_folds: Folds per repeat.
        n_repeats: Independent random partitions.
        seed: Seed of the ``numpy`` generator drawing the partitions.

    Returns:
        ``DataFrame[model, r2_in, r2_cv, r2_cv_sd, n_unidentified]`` with rows additive, product, bottleneck,
        product+proximity, rank-2.
    """
    rng = np.random.default_rng(seed)
    full = np.ones_like(M, dtype=bool)
    folds = [rng.permutation(M.size).reshape(M.shape) % n_folds for _ in range(n_repeats)]
    rows = []
    for model in CV_MODELS:
        cv = []
        n_unidentified = 0
        for fold in folds:
            P = np.empty_like(M)
            refused = 0
            for k in range(n_folds):
                test = fold == k
                if model == "product+proximity":
                    fit = fit_product_proximity(M, steps, ~test)
                    refused += fit.cond > PROXIMITY_MAX_COND
                    P[test] = fit.P[test]
                else:
                    P[test] = _predict(model, M, steps, ~test)[test]
            n_unidentified += refused
            cv.append(float("nan") if refused else _r2(M, P))
        rows.append({"model": model, "r2_in": _r2(M, _predict(model, M, steps, full)),
                     "r2_cv": float(np.mean(cv)),
                     "r2_cv_sd": float(np.std(cv, ddof=1)) if len(cv) > 1 else float("nan"),
                     "n_unidentified": int(n_unidentified)})
    rows.append({"model": "rank-2", "r2_in": _r2(M, _rank_k(M, 2)), "r2_cv": float("nan"),
                 "r2_cv_sd": float("nan"), "n_unidentified": 0})
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
        ``DataFrame[target, source, own_fit, own_product, both_from_src, b_from_src, a_from_src,
        flat_b, flat_a, corr_a, corr_b]``, one row per ordered pair: R^2 on the target of its
        own product+proximity fit; of its own plain rank-1 product fit (no proximity term, the
        ceiling for ``both_from_src``); of ``k * outer(a_src, b_src)`` with both curves taken
        from the source's fit and only the scalar ``k`` fitted on the target; with ``b`` taken from the source's fit (``a``, ``c``, ``L`` refit); with ``a``
        taken from the source (``b``, ``c``, ``L`` refit); the floors with ``b`` or ``a`` held
        at all ones; and the Pearson correlations of the two grids' ``a`` and ``b`` curves.
    """
    fits = {name: fit_product_proximity(M, steps) for name, M in grids.items()}
    ones = np.ones(len(steps))
    rows = []
    for tgt, Y in grids.items():
        own = _r2(Y, fits[tgt].P)
        own_product = _r2(Y, np.outer(*fit_rank1(Y)))
        flat_b = _r2(Y, fit_product_proximity(Y, steps, b=ones).P)
        flat_a = _r2(Y, fit_product_proximity(Y, steps, a=ones).P)
        for src in grids:
            if src == tgt:
                continue
            P0 = np.outer(fits[src].a, fits[src].b)
            k = (Y * P0).sum() / (P0**2).sum()
            rows.append({
                "target": tgt, "source": src, "own_fit": own, "own_product": own_product,
                "both_from_src": _r2(Y, k * P0),
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


def discover_olmo_checkpoint_grids(
    output_root: Path,
    animal: str,
    seeds: list[str],
    *,
    base: str = OLMO_BASE_DOLMA,
    student_tag: str,
    steps: list[int] | None = None,
    readout: str = "endpoint",
) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    """Find each OLMo (base, pretraining seed)'s cross-checkpoint grid and keep its full-coverage recipes.

    Every recipe on disk is loaded with `load_olmo_checkpoint_grid_runs` (``teacher_tags=None``);
    a recipe is kept only if it covers every ``(t, s)`` cell, so no cell mixes recipe counts.
    If no recipe covers the full ``steps`` grid for a seed, the seed is retried once on
    ``steps`` without 65000, the one checkpoint not run on every pretraining seed.

    Args:
        output_root: ``paths.output_root``.
        animal: Trait animal, e.g. ``elephant``.
        seeds: DataDecide pretraining seed suffixes (``seed-default``, ``seed-large-aux-2``, ...).
        base: DataDecide base model name (`OLMO_BASE_DOLMA`, `OLMO_BASE_DCLM`, ...).
        student_tag: The fixed student HP cell (see `OLMO_STUDENT_TAGS`).
        steps: Grid steps; default `OLMO_CKPT_STEPS`.
        readout: ``endpoint`` or ``peak`` (see `load_olmo_checkpoint_grid_runs`).

    Returns:
        ``(coverage, runs)``. ``coverage`` is ``DataFrame[animal, base, seed, n_steps, steps_used,
        cells, n_cells, complete, n_recipes, recipes, dropped]``: the step list the seed was
        scored on, cells covered by any recipe out of ``n_cells``, whether the grid is complete,
        and the sorted full recipes kept and partial recipes dropped. ``runs`` maps each
        complete ``(base, seed)`` to its full-recipe run frame restricted to ``steps_used``.
    """
    steps = list(OLMO_CKPT_STEPS if steps is None else steps)
    tries = [steps] + ([[x for x in steps if x != 65000]] if 65000 in steps else [])
    rows, out = [], {}
    for seed in seeds:
        all_runs = load_olmo_checkpoint_grid_runs(output_root, animal, seed=seed, base=base,
                                                  student_tag=student_tag, readout=readout)
        for used in tries:
            n_cells = len(used) ** 2
            runs = all_runs[all_runs["t"].isin(used) & all_runs["s"].isin(used)]
            per_tag = runs.groupby("tag")[["t", "s"]].apply(
                lambda g: len(g.drop_duplicates())) if len(runs) else pd.Series(dtype=int)
            full = sorted(per_tag.index[per_tag == n_cells])
            if full:
                break
        dropped = sorted(per_tag.index[per_tag < n_cells])
        rows.append({
            "animal": animal, "base": base, "seed": seed, "n_steps": len(used), "steps_used": used,
            "cells": len(runs[["t", "s"]].drop_duplicates()), "n_cells": n_cells,
            "complete": bool(full), "n_recipes": len(full), "recipes": full, "dropped": dropped,
        })
        if full:
            out[(base, seed)] = runs[runs["tag"].isin(full)].reset_index(drop=True)
    return pd.DataFrame(rows), out


def factorization_table(
    runs: dict[tuple[str, str], pd.DataFrame],
    steps: list[int] | None = None,
    *,
    n_folds: int = 5,
    n_repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per grid: in-sample, cell-CV and diagonal-held-out R^2 plus factor shape.

    Args:
        runs: ``(animal, lineage) -> runs`` as returned by `discover_checkpoint_grids`.
        steps: Checkpoint steps; default `CKPT_STEPS`.
        n_folds, n_repeats, seed: Passed to `cell_cross_validation` (default: five disjoint
            folds of about 20 per cent of the cells, every cell predicted once per repeat,
            five independent partitions, R^2 pooled over all out-of-fold predictions about
            the mean of the whole grid).

    Returns:
        ``DataFrame[animal, lineage, n_recipes, in_add, in_prod, in_bneck, in_prox, in_rank2,
        cv_add, cv_prod, cv_bneck, cv_prox, diag_prod, diag_prox, a_143_125, b_143_125, a_peak_k, b_peak_k]``.
    """
    steps = list(CKPT_STEPS if steps is None else steps)
    short = {"additive": "add", "product": "prod", "bottleneck": "bneck",
             "product+proximity": "prox",
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


# --------------------------------------------------------------------------------------
# Sender vs receiver per checkpoint
# --------------------------------------------------------------------------------------
_SR_LEVELS = ["fit", "animal", "recipe"]


def _send_receive_row(M: np.ndarray, steps: list[int]) -> np.ndarray:
    """``log2((a / max a) / (b / max b))`` of one product+proximity fit; NaN where a or b <= 0."""
    fit = fit_product_proximity(M, steps)
    a = fit.a / fit.a.max()
    b = fit.b / fit.b.max()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log2(a / b)
    out[(a <= 0) | (b <= 0) | ~np.isfinite(out)] = np.nan
    return out


def send_receive_index(
    runs: dict[tuple[str, str], pd.DataFrame],
    steps: list[int] | None = None,
    *,
    lineage: str = "orig",
    per_recipe: bool = True,
) -> pd.DataFrame:
    """Per-step sender-vs-receiver index of each grid, recipe-pooled and per recipe.

    For each fit the product+proximity factors are scaled to max 1 and the index is
    ``log2(a(k) / b(k))``: positive = checkpoint ``k`` is a better sender than receiver
    (relative to each factor's own peak). A step where ``a`` or ``b`` is ``<= 0`` is NaN.

    Args:
        runs: ``(animal, lineage) -> runs`` as returned by `discover_checkpoint_grids`.
        steps: Checkpoint steps; default `CKPT_STEPS`.
        lineage: Only grids of this lineage are used.
        per_recipe: Also fit each teacher recipe's own grid.

    Returns:
        Frame with a ``(fit, animal, recipe)`` row MultiIndex -- ``fit`` is ``"{animal}"`` for
        the recipe-pooled fit (``recipe == "all"``) and ``"{animal}/{tag}"`` per recipe -- and
        one integer column per step in thousands (``step // 1000``).
    """
    steps = list(CKPT_STEPS if steps is None else steps)
    rows, keys = [], []
    for (animal, lin), r in runs.items():
        if lin != lineage:
            continue
        rows.append(_send_receive_row(grid_matrix(r, steps).M, steps))
        keys.append((animal, animal, "all"))
        if per_recipe:
            for tag in sorted(r["tag"].unique()):
                rows.append(_send_receive_row(grid_matrix(r[r["tag"] == tag], steps).M, steps))
                keys.append((f"{animal}/{tag}", animal, tag))
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(keys, names=_SR_LEVELS),
                        columns=[s // 1000 for s in steps])


def send_receive_seed_index(runs_one_grid: pd.DataFrame,
                            steps: list[int] | None = None) -> pd.DataFrame:
    """Recipe-pooled send/receive index refit on each single ftseed that covers every cell.

    Args:
        runs_one_grid: One grid's run frame (as in `discover_checkpoint_grids` values).
        steps: Checkpoint steps; default `CKPT_STEPS`.

    Returns:
        Rows ``seed{k}`` (only seeds with no empty cell), columns = steps in thousands, the
        same layout as the columns of `send_receive_index`.
    """
    steps = list(CKPT_STEPS if steps is None else steps)
    rows, labels = [], []
    for k in sorted(runs_one_grid["ftseed"].unique()):
        try:
            M = grid_matrix(runs_one_grid[runs_one_grid["ftseed"] == k], steps).M
        except ValueError:
            continue
        rows.append(_send_receive_row(M, steps))
        labels.append(f"seed{k}")
    return pd.DataFrame(rows, index=pd.Index(labels, name="fit"),
                        columns=[s // 1000 for s in steps])


def send_receive_agreement(
    index: pd.DataFrame,
    *,
    close: float = 0.5,
    strong: float = 1.0,
    seed_index: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-step agreement of the send/receive index across fits.

    Args:
        index: `send_receive_index` output (or any frame of fits x steps; labels are the
            ``fit`` index level when present, else the index).
        close: ``|index| < close`` counts as "close to balanced" in ``share_close``.
        strong: A dissenter must have ``|index| >= strong``.
        seed_index: Optional `send_receive_seed_index` frame; adds ``seed_sd`` per step.

    Returns:
        One row per step (index = step in k): ``n_pos``, ``n_neg``, ``n_nan``, ``majority``
        (+1 / -1 / 0 on a tie), ``unanimous`` (all non-NaN share one sign), ``median_abs``,
        ``share_close``, ``dissenters`` (comma-joined fits opposite the majority with
        ``|index| >= strong``, largest first; ``""`` if none) and optionally ``seed_sd``.
    """
    labels = (index.index.get_level_values("fit") if "fit" in (index.index.names or [])
              else index.index).astype(str)
    rows = []
    for col in index.columns:
        v = index[col].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        n_pos, n_neg = int((v[ok] > 0).sum()), int((v[ok] < 0).sum())
        majority = int(np.sign(n_pos - n_neg))
        diss = [(abs(x), lab) for x, lab in zip(v, labels)
                if majority and not np.isnan(x) and np.sign(x) == -majority and abs(x) >= strong]
        row = {
            "n_pos": n_pos, "n_neg": n_neg, "n_nan": int((~ok).sum()), "majority": majority,
            "unanimous": bool(ok.any()) and (n_pos == 0 or n_neg == 0),
            "median_abs": float(np.median(np.abs(v[ok]))) if ok.any() else np.nan,
            "share_close": float((np.abs(v[ok]) < close).mean()) if ok.any() else np.nan,
            "dissenters": ",".join(lab for _, lab in sorted(diss, reverse=True)),
        }
        if seed_index is not None:
            row["seed_sd"] = float(seed_index[col].std())
        rows.append(row)
    return pd.DataFrame(rows, index=pd.Index(list(index.columns), name="step_k"))


def send_receive_curve_agreement(
    runs: dict[tuple[str, str], pd.DataFrame],
    steps: list[int] | None = None,
    *,
    lineage: str = "orig",
) -> pd.DataFrame:
    """Pairwise Spearman agreement of the max-scaled sender and receiver curves.

    Args:
        runs: ``(animal, lineage) -> runs`` as returned by `discover_checkpoint_grids`.
        steps: Checkpoint steps; default `CKPT_STEPS`.
        lineage: Only grids of this lineage are used.

    Returns:
        ``DataFrame[group, n_pairs, a_median, a_min, b_median, b_min]``: group ``"animals"``
        compares the recipe-pooled fits across animals; one row per animal compares that
        animal's per-recipe fits. Values are Spearman correlations over unordered pairs.
    """
    steps = list(CKPT_STEPS if steps is None else steps)

    def _curves(M: np.ndarray) -> tuple[pd.Series, pd.Series]:
        fit = fit_product_proximity(M, steps)
        return pd.Series(fit.a / fit.a.max()), pd.Series(fit.b / fit.b.max())

    groups: dict[str, list[tuple[pd.Series, pd.Series]]] = {"animals": []}
    for (animal, lin), r in runs.items():
        if lin != lineage:
            continue
        groups["animals"].append(_curves(grid_matrix(r, steps).M))
        groups[animal] = [_curves(grid_matrix(r[r["tag"] == tag], steps).M)
                          for tag in sorted(r["tag"].unique())]
    rows = []
    for group, curves in groups.items():
        pairs = [(x, y) for i, x in enumerate(curves) for y in curves[i + 1:]]
        row = {"group": group, "n_pairs": len(pairs)}
        for j, name in ((0, "a"), (1, "b")):
            rho = np.array([x[j].corr(y[j], method="spearman") for x, y in pairs])
            row[f"{name}_median"] = float(np.median(rho)) if len(rho) else np.nan
            row[f"{name}_min"] = float(rho.min()) if len(rho) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _rowcol_row(M: np.ndarray, offdiag: bool) -> np.ndarray:
    """``(row mean - column mean) / grid mean`` of one grid, diagonal optionally excluded."""
    M = np.array(M, dtype=float)
    if offdiag:
        np.fill_diagonal(M, np.nan)
    return (np.nanmean(M, axis=1) - np.nanmean(M, axis=0)) / np.nanmean(M)


def send_receive_rowcol(
    runs: dict[tuple[str, str], pd.DataFrame],
    steps: list[int] | None = None,
    *,
    lineage: str = "orig",
    per_recipe: bool = True,
    offdiag: bool = True,
) -> pd.DataFrame:
    """Model-free per-step send/receive score: row mean minus column mean over grid mean.

    Positive = checkpoint ``k`` transfers more as a teacher (its row) than it receives as
    a student (its column), in units of the grid's mean transfer. Unlike
    `send_receive_index` this is absolute, not relative to each factor's own peak.

    Args:
        runs: ``(animal, lineage) -> runs`` as returned by `discover_checkpoint_grids`.
        steps: Checkpoint steps; default `CKPT_STEPS`.
        lineage: Only grids of this lineage are used.
        per_recipe: Also score each teacher recipe's own grid.
        offdiag: Exclude the diagonal, which carries the same-checkpoint bonus and would
            otherwise add the same amount to a step's row and column means unevenly.

    Returns:
        Same layout as `send_receive_index`: ``(fit, animal, recipe)`` row MultiIndex and one
        column per step in thousands.
    """
    steps = list(CKPT_STEPS if steps is None else steps)
    rows, keys = [], []
    for (animal, lin), r in runs.items():
        if lin != lineage:
            continue
        rows.append(_rowcol_row(grid_matrix(r, steps).M, offdiag))
        keys.append((animal, animal, "all"))
        if per_recipe:
            for tag in sorted(r["tag"].unique()):
                rows.append(_rowcol_row(grid_matrix(r[r["tag"] == tag], steps).M, offdiag))
                keys.append((f"{animal}/{tag}", animal, tag))
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(keys, names=_SR_LEVELS),
                        columns=[s // 1000 for s in steps])


@dataclass
class SharedFit:
    """Result of `shared_factor_fit`.

    Attributes:
        pred: Per-grid prediction ``c_i * a_i b_i^T``.
        a_shared: Shared sender curve (max 1), or ``None`` when not shared.
        b_shared: Shared receiver curve (max 1), or ``None`` when not shared.
        scale: Per-grid scale ``c_i`` (in the units of the max-1 shared curves).
        r2: Per-grid R^2 of ``pred``.
    """

    pred: dict[str, np.ndarray]
    a_shared: np.ndarray | None
    b_shared: np.ndarray | None
    scale: dict[str, float]
    r2: dict[str, float]


def shared_factor_fit(
    grids: dict[str, np.ndarray], *, share_a: bool, share_b: bool, iters: int = 500
) -> SharedFit:
    """Fit ``M_i ~= c_i * a_i b_i^T`` with ``a`` and/or ``b`` shared across grids.

    Plain product (no proximity term), alternating least squares initialised from each
    grid's `fit_rank1`. Shared curves are rescaled to max 1 after fitting (scale moved into
    ``c_i``), so ``scale`` compares grids on a common curve.

    Args:
        grids: ``name -> M``, all the same shape.
        share_a: Share the sender (row) curve across grids.
        share_b: Share the receiver (column) curve across grids.
        iters: ALS sweeps.

    Returns:
        `SharedFit`.
    """
    keys = list(grids)
    init = {k: fit_rank1(grids[k]) for k in keys}
    A = {k: init[k][0] for k in keys}
    B = {k: init[k][1] for k in keys}
    a_sh = np.mean([A[k] / A[k].max() for k in keys], 0)
    b_sh = np.mean([B[k] / B[k].max() for k in keys], 0)
    c = {k: 1.0 for k in keys}

    def _a(k):
        return a_sh if share_a else A[k]

    def _b(k):
        return b_sh if share_b else B[k]

    for _ in range(iters):
        for k in keys:
            P = np.outer(_a(k), _b(k))
            c[k] = float((grids[k] * P).sum() / (P ** 2).sum())
        if share_a:
            num = sum(c[k] * grids[k] @ _b(k) for k in keys)
            den = sum(c[k] ** 2 * (_b(k) @ _b(k)) for k in keys)
            a_sh = num / den
        else:
            for k in keys:
                A[k] = (grids[k] @ _b(k)) / (_b(k) @ _b(k)) / c[k]
        if share_b:
            num = sum(c[k] * grids[k].T @ _a(k) for k in keys)
            den = sum(c[k] ** 2 * (_a(k) @ _a(k)) for k in keys)
            b_sh = num / den
        else:
            for k in keys:
                B[k] = (grids[k].T @ _a(k)) / (_a(k) @ _a(k)) / c[k]
    pred = {k: c[k] * np.outer(_a(k), _b(k)) for k in keys}
    if share_a:
        m = a_sh.max()
        a_sh = a_sh / m
        c = {k: v * m for k, v in c.items()}
    if share_b:
        m = b_sh.max()
        b_sh = b_sh / m
        c = {k: v * m for k, v in c.items()}
    return SharedFit(pred=pred, a_shared=a_sh if share_a else None,
                     b_shared=b_sh if share_b else None, scale=c,
                     r2={k: _r2(grids[k], pred[k]) for k in keys})


_SHARED_MODELS = [("own a, own b", False, False), ("shared a, own b", True, False),
                  ("own a, shared b", False, True), ("shared a, shared b", True, True)]


def shared_factor_table(
    grids: dict[str, np.ndarray], *, ceiling: dict[str, float] | None = None
) -> pd.DataFrame:
    """R^2 of each grid under own/shared sender and receiver curves (`shared_factor_fit`).

    Args:
        grids: ``name -> M``, all the same shape.
        ceiling: Optional ``name -> R^2`` reference (e.g. seed-vs-seed), added as a row.

    Returns:
        Rows ``own a, own b`` / ``shared a, own b`` / ``own a, shared b`` /
        ``shared a, shared b`` (+ ``ceiling``), one R^2 column per grid.
    """
    rows = {name: shared_factor_fit(grids, share_a=sa, share_b=sb).r2
            for name, sa, sb in _SHARED_MODELS}
    if ceiling is not None:
        rows["ceiling"] = {k: ceiling.get(k, np.nan) for k in grids}
    return pd.DataFrame.from_dict(rows, orient="index")[list(grids)].rename_axis("model")


def send_receive_variance_decomposition(score: pd.DataFrame) -> pd.DataFrame:
    """Sum-of-squares shares of a per-recipe send/receive score over step x animal x recipe.

    Balanced two-way sequential decomposition on the per-recipe rows (``recipe != "all"``)
    of a `send_receive_rowcol` / `send_receive_index` frame: step main effect (shared
    checkpoint aptitude), step x animal beyond step, step x recipe beyond step, and the
    residual (step x animal x recipe + noise). NaN cells are dropped, so the shares are only
    exactly additive for a complete frame.

    Args:
        score: Frame with the ``(fit, animal, recipe)`` row MultiIndex, steps as columns.

    Returns:
        ``DataFrame[component, share]`` with four rows.
    """
    per = score[score.index.get_level_values("recipe") != "all"]
    L = per.reset_index(level=["animal", "recipe"]).melt(
        id_vars=["animal", "recipe"], var_name="step", value_name="y").dropna(subset=["y"])
    mu = L["y"].mean()
    tot = ((L["y"] - mu) ** 2).sum()
    m_s = L.groupby("step")["y"].transform("mean")
    m_sa = L.groupby(["step", "animal"])["y"].transform("mean")
    m_sr = L.groupby(["step", "recipe"])["y"].transform("mean")
    parts = {
        "step": ((m_s - mu) ** 2).sum(),
        "step x animal": ((m_sa - m_s) ** 2).sum(),
        "step x recipe": ((m_sr - m_s) ** 2).sum(),
        "residual": ((L["y"] - m_sa - m_sr + m_s) ** 2).sum(),
    }
    return pd.DataFrame({"component": list(parts),
                         "share": [float(v / tot) for v in parts.values()]})


# --------------------------------------------------------------------------------------
# S_all as a sender-side probe: real vs resampled (synthetic) excess grids
# --------------------------------------------------------------------------------------
#: OLMo S_all pool directory under a grid cell; ``{pool}`` is ``c{S//1000}k`` or ``cfinal`` for
#: the STUDENT step S (the carriers are base@S numbers resampled with teacher@t's first-number
#: profile).
OLMO_SALL_DATASET: str = "ex10-20_prefix_ablations_synthiid-{pool}_poolseed0"
#: OLMo teacher recipes left out of every S_all comparison, keyed by ``(base, seed)``. Rule: the
#: all-checkpoints rule of the campaigns -- a recipe whose final-checkpoint teacher failed the
#: teacher gate (aux-2 canonical: dCE 3.61) is dropped at every checkpoint of that seed.
OLMO_SALL_EXCLUDED_TEACHERS: dict[tuple[str, str], tuple[str, ...]] = {
    (OLMO_BASE_DOLMA, "seed-large-aux-2"): ("ep10_lr5e-4_r64a128",),
    (OLMO_BASE_DOLMA, "seed-large-aux-3"): ("ep10_lr5e-4_r64a128", "ep10_lr2e-4_r64a512"),
}


def load_olmo_sall_grid_runs(
    output_root: Path,
    animal: str,
    *,
    seed: str = "seed-default",
    base: str = OLMO_BASE_DOLMA,
    student_tag: str,
    steps: list[int] | None = None,
) -> pd.DataFrame:
    """Scan the S_all (first-number resampled) leaves of one OLMo cross-checkpoint grid.

    Same cell dirs as `load_olmo_checkpoint_grid_runs`, dataset `OLMO_SALL_DATASET` keyed by the
    student step; evals are read at the leaf root (or ``eval/{animal}/``).

    Returns:
        Columns ``t``, ``s``, ``tag``, ``ftseed``, ``p``.
    """
    prefix = f"{base}-{seed}"
    cells = []
    for student_dir, s in _olmo_student_dirs(Path(output_root) / "students" / animal, prefix):
        pool = "cfinal" if s == OLMO_FINAL_STEP else f"c{s // 1000}k"
        for teacher_dir in sorted(student_dir.glob(f"teacher-{prefix}*")):
            t = _olmo_step(teacher_dir.name[len("teacher-"):], prefix)
            if t is None:
                continue
            for tagseed in sorted(teacher_dir.iterdir()):
                mg = _TAGSEED_RE.fullmatch(tagseed.name)
                ds = tagseed / OLMO_SALL_DATASET.format(pool=pool)
                if mg and ds.is_dir():
                    cells.append((t, s, mg["tag"], ds / student_tag))
    return pd.DataFrame(_olmo_rows(cells, animal, "endpoint", steps),
                        columns=["t", "s", "tag", "ftseed", "p"])


def _complete_teachers(runs: pd.DataFrame, steps: list[int], ftseed: int) -> set[str]:
    d = runs[(runs["ftseed"] == ftseed) & runs["t"].isin(steps) & runs["s"].isin(steps)]
    cells = d.drop_duplicates(["tag", "t", "s"]).groupby("tag").size()
    return set(cells.index[cells == len(steps) ** 2])


def olmo_sall_teacher_set(
    real: pd.DataFrame,
    sall: pd.DataFrame,
    step_options: list[list[int]],
    *,
    excluded: tuple[str, ...] = (),
    ftseed: int = 1,
) -> tuple[list[int], list[str], bool]:
    """Pick the grid and teacher set of an S_all comparison.

    The first step list of ``step_options`` on which some recipe has a complete grid at
    ``ftseed`` in BOTH arms wins, with the teacher set = those recipes. If no option has one, the
    first option is used with the recipes complete in the real arm only (an R-only grid).

    Returns:
        ``(steps, teachers, both_arms)``; ``both_arms`` is False for the R-only fallback.
    """
    for steps in step_options:
        both = (_complete_teachers(real, steps, ftseed) & _complete_teachers(sall, steps, ftseed))
        both -= set(excluded)
        if both:
            return list(steps), sorted(both), True
    steps = list(step_options[0])
    return steps, sorted(_complete_teachers(real, steps, ftseed) - set(excluded)), False


def olmo_sall_coverage(arms: dict[str, pd.DataFrame], teachers: list[str],
                       steps: list[int]) -> pd.DataFrame:
    """Cells present per arm x ftseed on the chosen grid (min / max over the teacher set).

    Args:
        arms: ``{"R": runs, "S_all": runs, "C": controls}``; controls have tag ``base`` and are
            counted as one teacher.
        teachers: The teacher set.
        steps: The grid steps.

    Returns:
        ``DataFrame[arm, ftseed, teachers, cells_min, cells_max, grid]``.
    """
    rows = []
    for arm, runs in arms.items():
        d = runs[runs["t"].isin(steps) & runs["s"].isin(steps)]
        if arm != "C":
            d = d[d["tag"].isin(teachers)]
        for ftseed, g in d.groupby("ftseed"):
            per = g.drop_duplicates(["tag", "t", "s"]).groupby("tag").size()
            if arm != "C":
                per = per.reindex(teachers, fill_value=0)
            rows.append({"arm": arm, "ftseed": int(ftseed), "teachers": int((per > 0).sum()),
                         "cells_min": int(per.min()), "cells_max": int(per.max()),
                         "grid": f"{len(steps)}x{len(steps)}"})
    return pd.DataFrame(rows, columns=["arm", "ftseed", "teachers", "cells_min", "cells_max", "grid"])


def teacher_mean_grid(runs: pd.DataFrame, teachers: list[str], steps: list[int],
                      ftseeds: list[int] | None = None) -> np.ndarray:
    """Cell = ftseed mean within a recipe, then mean over ``teachers``; NaN unless EVERY teacher
    of the set has the cell (no recipe-count mixing across cells)."""
    d = runs[runs["tag"].isin(teachers) & runs["t"].isin(steps) & runs["s"].isin(steps)]
    if ftseeds is not None:
        d = d[d["ftseed"].isin(ftseeds)]
    per = d.groupby(["t", "s", "tag"])["p"].mean()
    grid = per.groupby(["t", "s"]).mean().unstack().reindex(index=steps, columns=steps)
    n = per.groupby(["t", "s"]).size().unstack().reindex(index=steps, columns=steps)
    return grid.where(n == len(teachers)).to_numpy(dtype=float)


def olmo_sall_excess(real: pd.DataFrame, sall: pd.DataFrame, controls: pd.DataFrame,
                     teachers: list[str], steps: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Control-subtracted excess grids ``(R - C(t->s), S_all - C(s,s))`` on one teacher set.

    An S_all cell (t, s) is teacher@t's profile on base@s carriers, so its matched control is the
    diagonal base control C(s, s) (the Pythia convention of
    `checkpoint_cross_readouts.resampling_excess`); the real arm uses its own C(t->s).
    """
    d = controls[controls["t"].isin(steps) & controls["s"].isin(steps)]
    C = d.groupby(["t", "s"])["p"].mean().unstack().reindex(index=steps, columns=steps)
    C = C.to_numpy(dtype=float)
    R = teacher_mean_grid(real, teachers, steps)
    S = teacher_mean_grid(sall, teachers, steps)
    return R - C, S - np.diag(C)[None, :]


def sall_cell_retention(R_ex: np.ndarray, S_ex: np.ndarray, steps: list[int], *,
                        min_real: float = 0.10, high: float = 0.30) -> dict:
    """Per-cell retention ``S_ex / R_ex`` summary for one S_all grid (rows teacher, cols student).

    Best / median / count statistics use only cells with real excess ``R_ex >= min_real``; pooled
    ratios and the largest S_all excess use every cell. Steps are labelled "60k" / "final".
    """
    R_ex, S_ex = np.asarray(R_ex, dtype=float), np.asarray(S_ex, dtype=float)
    lab = [f"{s // 1000}k" if s < 10**6 else "final" for s in steps]
    ok = R_ex >= min_real
    ret = np.where(ok, S_ex / np.where(ok, R_ex, 1.0), np.nan)
    dret = np.diag(ret)
    has_d, has_any = bool(np.isfinite(dret).any()), bool(ok.any())
    di = int(np.nanargmax(dret)) if has_d else None
    ai = np.unravel_index(int(np.nanargmax(ret)), ret.shape) if has_any else None
    si = np.unravel_index(int(np.argmax(S_ex)), S_ex.shape)
    return {
        "pooled all": S_ex.sum() / R_ex.sum(),
        "pooled diagonal": np.diag(S_ex).sum() / np.diag(R_ex).sum(),
        "best diagonal": dret[di] if has_d else np.nan,
        "best diagonal step": lab[di] if has_d else None,
        "median diagonal": float(np.nanmedian(dret)) if has_d else np.nan,
        "best cell": ret[ai] if has_any else np.nan,
        "best cell (teacher, student)": f"({lab[ai[0]]}, {lab[ai[1]]})" if has_any else None,
        f"cells R − C ≥ {min_real:.2f}": int(ok.sum()),
        f"of those ≥ {high:.2f}": int((ok & (np.nan_to_num(ret) >= high)).sum()),
        "max S_all − C": S_ex[si],
        "max S_all − C (teacher, student)": f"({lab[si[0]]}, {lab[si[1]]})",
    }


def seed_stability(runs: pd.DataFrame, teachers: list[str], steps: list[int]) -> dict:
    """Agreement of the per-ftseed teacher-mean grids: lowest pairwise Pearson over shared cells
    and the mean across-seed SD of a cell."""
    seeds = sorted(runs["ftseed"].unique())
    grids = [teacher_mean_grid(runs, teachers, steps, [k]) for k in seeds]
    rs = []
    for x, y in itertools.combinations(grids, 2):
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() >= 3:
            rs.append(float(np.corrcoef(x[m], y[m])[0, 1]))
    sd = np.nanstd(np.stack(grids), 0, ddof=1) if len(grids) > 1 else np.full((1,), np.nan)
    return {"ftseeds": len(seeds), "min_pearson": min(rs) if rs else float("nan"),
            "mean_cell_sd": float(np.nanmean(sd))}


def _masked_r2(M: np.ndarray, P: np.ndarray, m: np.ndarray) -> float:
    return float(1 - ((M - P)[m] ** 2).sum() / ((M[m] - M[m].mean()) ** 2).sum())


def _log_shares(M: np.ndarray, clip: float = 0.005) -> tuple[float, float, float]:
    """Share of the log-excess sum of squares explained by teacher-only and student-only offsets,
    and the additive residual (interaction). On a full grid these are the 2-way decomposition and
    sum to one."""
    m = ~np.isnan(M)
    L = np.where(m, np.log(np.clip(np.where(m, M, clip), clip, None)), np.nan)
    mu = np.nanmean(L)
    tot = np.nansum((L - mu) ** 2)
    teacher = 1 - np.nansum((L - np.nanmean(L, 1)[:, None]) ** 2) / tot
    student = 1 - np.nansum((L - np.nanmean(L, 0)[None, :]) ** 2) / tot
    u, v = np.zeros(M.shape[0]), np.zeros(M.shape[1])
    for _ in range(200):
        u = np.nanmean(L - v[None, :], 1)
        v = np.nanmean(L - u[:, None], 0)
    return float(teacher), float(student), float(np.nansum((L - u[:, None] - v[None, :]) ** 2) / tot)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return float("nan")
    return float(pd.Series(x[m]).corr(pd.Series(y[m]), method="spearman"))


def arm_factor_comparison(R: np.ndarray, S: np.ndarray, steps: list[int], *,
                          ref_step: int | None = None) -> dict[str, pd.DataFrame]:
    """Compare the rank-one factors of a real (``R``) and a synthetic (``S``) excess grid.

    Both grids are teacher step x student step, already control-subtracted. Missing cells are NaN
    and are masked out of every statistic (never imputed); retention uses only the cells both
    grids have. Factors are in the `fit_rank1` gauge (receiver curve ``b`` max 1, sender curve
    ``a`` in excess units); with ``ref_step`` the two sender curves are also shown divided by
    their value at that step.

    Args:
        R: Real-number excess grid.
        S: Synthetic (S_all) excess grid on the same steps.
        steps: Checkpoint steps indexing both axes.
        ref_step: Optional step at which each normalised sender curve is set to 1.

    Returns:
        ``summary``: one row -- per arm the rank-one R^2, the same-checkpoint bonus ``c`` of the
        product + diagonal fit and its R^2 gain (full grids only), the variance of the teacher-row
        and student-column means and their ratio, the teacher / student / interaction shares of
        the log excess; across arms the Spearman correlation of the sender and of the receiver
        curves, the share of off-diagonal send/receive leans with the same sign, and retention
        ``sum(S) / sum(R)`` overall, on the diagonal and off it.
        ``by_step``: per step the two sender and receiver curves and their ratios, and retention
        by teacher row and by student column.
    """
    n = len(steps)
    eye = np.eye(n, dtype=bool)
    row: dict[str, float] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arm, M in (("real", R), ("synthetic", S)):
        m = ~np.isnan(M)
        a = b = np.full(n, np.nan)
        r2 = c = gain = float("nan")
        if m.sum() >= 3 and m.any(0).all() and m.any(1).all():
            a, b = fit_rank1(np.where(m, M, 0.0), mask=m)
            r2 = _masked_r2(M, np.outer(a, b), m)
            if m.all():
                a_d, b_d, c = _fit_rank1_plus(M, eye.astype(float))
                gain = _r2(M, np.outer(a_d, b_d) + c * eye) - r2
        curves[arm] = (a, b)
        rows_mean, cols_mean = np.nanmean(M, 1), np.nanmean(M, 0)
        shares = _log_shares(M)
        row |= {f"rank-1 R² {arm}": r2, f"diagonal bonus c {arm}": c,
                f"R² gain from diagonal {arm}": gain,
                f"var teacher-row means {arm}": float(np.nanvar(rows_mean)),
                f"var student-column means {arm}": float(np.nanvar(cols_mean)),
                f"row/column variance ratio {arm}": float(np.nanvar(rows_mean) / np.nanvar(cols_mean)),
                f"log share teacher {arm}": shares[0], f"log share student {arm}": shares[1],
                f"log share interaction {arm}": shares[2]}
    (aR, bR), (aS, bS) = curves["real"], curves["synthetic"]

    def _lean(M):
        H = np.where(eye, np.nan, M)
        return np.nanmean(H, 1) - np.nanmean(H, 0)

    leans = _lean(R), _lean(S)
    ok = ~(np.isnan(leans[0]) | np.isnan(leans[1]))
    both = ~(np.isnan(R) | np.isnan(S))
    Rb, Sb = np.where(both, R, 0.0), np.where(both, S, 0.0)
    row |= {"sender Spearman": _spearman(aS, aR), "receiver Spearman": _spearman(bS, bR),
            "lean sign agreement": float(np.mean(np.sign(leans[0][ok]) == np.sign(leans[1][ok])))
            if ok.any() else float("nan"),
            "cells in both": int(both.sum()),
            "retention all": float(Sb.sum() / Rb.sum()),
            "retention diagonal": float(Sb[eye].sum() / Rb[eye].sum()),
            "retention off-diagonal": float(Sb[~eye].sum() / Rb[~eye].sum())}
    with np.errstate(invalid="ignore", divide="ignore"):
        by_step = {"a real": aR, "a synthetic": aS, "a synthetic / a real": aS / aR}
        if ref_step is not None:
            k = steps.index(ref_step)
            by_step |= {f"a real / a real({ref_step // 1000}k)": aR / aR[k],
                        f"a synthetic / a synthetic({ref_step // 1000}k)": aS / aS[k]}
        by_step |= {"b real": bR, "b synthetic": bS, "b synthetic / b real": bS / bR,
                    "retention by teacher row": Sb.sum(1) / Rb.sum(1),
                    "retention by student column": Sb.sum(0) / Rb.sum(0)}
    labels = ["final" if s == OLMO_FINAL_STEP else f"{s // 1000}k" for s in steps]
    return {"summary": pd.DataFrame([row]),
            "by_step": pd.DataFrame(by_step, index=labels).T}
