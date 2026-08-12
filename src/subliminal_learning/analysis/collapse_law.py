"""Estimator for the student **collapse-boundary law** measured by the ``law-*`` sweep campaign.

The law says an ep20 student collapses (loses the trait entirely, not just weakly) when

    K = lr * alpha**b * r**c  >=  K*

Notebook §11--§12 of ``cross_model_student_hp_exploration.ipynb`` fit ``b``/``c`` on a *single*
lineage (teacher ``ep2_lr5e-4_r32a64`` -> student ``data-seed1``). The ``law-*`` queues extended
that to three teacher lineages x two student bases so the exponents can be tested for generality;
this module is the shared estimator those refits run through, so the notebook stays presentation.

Two estimation notes that drive the API:

- **The design is near-separable.** Collapse is almost a deterministic function of ``K``, so an
  unpenalised MLE drives the coefficient *magnitudes* toward infinity. Only the *ratios*
  ``b = beta_alpha/beta_lr`` and ``c = beta_r/beta_lr`` are identified. `fit_collapse_law` therefore
  reports ratios, and `cluster_bootstrap_ratios` uses a ridge (via scikit-learn) so every bootstrap
  replicate has a finite fit.
- **Seeds within a cell are correlated.** Resampling seeds independently manufactures significance.
  Both inference paths cluster on the cell (``student_tag``): statsmodels supplies a cluster-robust
  covariance for the delta method, and the bootstrap resamples whole cells.

The two paths agree closely in practice; the delta method is the default because it is ~100x
cheaper, with the bootstrap kept as an independent cross-check.

The module also exposes the *settled* boundary constants (`B_LAW`, `C_LAW`, the two ``K*``
anchors) and vectorized ``K`` helpers (`K_of`, `k_from_tag`, `add_k`), so notebooks stop
re-hardcoding the exponents and picking whichever ``K*`` they last remembered.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from subliminal_learning.analysis.frames import (
    PARAM_TAG_RE,
    parse_param_tag,
    read_cell_set,
)
from subliminal_learning.teachers.run_layout import model_dir_parts_from_name

# --- The settled collapse-boundary law (K = lr * alpha**B_LAW * rank**C_LAW; collapse at K >= K*).
# B_LAW/C_LAW are the exponents settled across all six arms of the campaign (rank is protective).
B_LAW: float = 0.561
C_LAW: float = -0.085

# Two K* anchors exist because the notebooks drifted between them. KSTAR_POOLED is the pooled
# shared-band anchor and is CANONICAL for new work; KSTAR_DS1_DESIGN is the older ds1 design
# anchor a few early notebooks fit against, kept only so those numbers stay reproducible.
KSTAR_POOLED: float = 7.87e-4
KSTAR_DS1_DESIGN: float = 7.84e-4


def K_of(lr, alpha, rank, *, b: float = B_LAW, c: float = C_LAW):
    """Evaluate ``K = lr * alpha**b * rank**c`` on scalars, numpy arrays or pandas Series.

    Args:
        lr: Learning rate(s).
        alpha: LoRA alpha(s).
        rank: LoRA rank(s).
        b: Alpha exponent (defaults to the settled `B_LAW`).
        c: Rank exponent (defaults to the settled `C_LAW`).

    Returns:
        ``K`` with the broadcast type/shape of the inputs.
    """
    return lr * alpha**b * rank**c


def k_from_tag(tag: str, *, b: float = B_LAW, c: float = C_LAW) -> float:
    """Compute ``K`` for a ``ep{e}_lr{lr}_r{r}a{a}`` parameter tag.

    Args:
        tag: A canonical sweep tag (parsed by `frames.parse_param_tag`).
        b: Alpha exponent (defaults to `B_LAW`).
        c: Rank exponent (defaults to `C_LAW`).

    Returns:
        ``K`` evaluated at the tag's lr/alpha/rank.
    """
    f = parse_param_tag(tag)
    return float(K_of(f["lr"], f["alpha"], f["rank"], b=b, c=c))


def add_k(
    df: pd.DataFrame,
    *,
    lr: str = "lr",
    alpha: str = "alpha",
    rank: str = "rank",
    kstar: float | None = None,
    b: float = B_LAW,
    c: float = C_LAW,
) -> pd.DataFrame:
    """Return a copy of `df` with a ``K`` column (and ``K_rel = K / kstar`` when `kstar` given).

    Args:
        df: Frame carrying lr/alpha/rank columns.
        lr: Learning-rate column name.
        alpha: Alpha column name.
        rank: Rank column name.
        kstar: If provided, also add ``K_rel = K / kstar`` (e.g. `KSTAR_POOLED`).
        b: Alpha exponent (defaults to `B_LAW`).
        c: Rank exponent (defaults to `C_LAW`).

    Returns:
        A copy with the new column(s); the input is not mutated.
    """
    out = df.copy()
    out["K"] = K_of(out[lr], out[alpha], out[rank], b=b, c=c)
    if kstar is not None:
        out["K_rel"] = out["K"] / kstar
    return out

# The six arms of the campaign: (family, teacher param tag, student base model).
# "family" names the teacher recipe's character -- focus/strong are short+aggressive, weak is
# long+gentle -- and is what the queues were named after (law-focus-*, law-strong-*, law-weak-*).
LAW_LINEAGES: list[tuple[str, str, str]] = [
    ("focus", "ep2_lr5e-4_r32a64", "pythia-160m-data-seed1"),
    ("focus", "ep2_lr5e-4_r32a64", "pythia-160m-weight-seed1"),
    ("strong", "ep2_lr5e-4_r16a64", "pythia-160m-data-seed1"),
    ("strong", "ep2_lr5e-4_r16a64", "pythia-160m-weight-seed1"),
    ("weak", "ep7_lr1e-4_r16a32", "pythia-160m-data-seed1"),
    ("weak", "ep7_lr1e-4_r16a32", "pythia-160m-weight-seed1"),
]

# Display order for the six lineages (data-seed before weight-seed within each family).
LINEAGE_ORDER: list[str] = [
    "focus:data-seed1",
    "focus:weight-seed1",
    "strong:data-seed1",
    "strong:weight-seed1",
    "weak:data-seed1",
    "weak:weight-seed1",
]

# Per-seed collapse threshold on test/prob p_target. Sits in the empty trough of the bimodal
# seed-score distribution (dead mode <=1e-4, live mode >=2e-2); b/c are insensitive over 0.005-0.02.
DEAD_SEED: float = 0.01

# Ridge strength for the penalised fit, matching notebook §12's ``pen=1e-3`` on the non-intercept
# coefficients. scikit-learn minimises ``0.5*||w||^2 + C*sum(loss)``, so C = 1/(2*pen) = 500
# reproduces §12's objective (verified to 7 decimals on the ratios).
RIDGE_PEN: float = 1e-3
SKLEARN_C: float = 1.0 / (2.0 * RIDGE_PEN)

# The canonical cell-name parser (shared with `frames`). Trailing ``_{optim_slug}`` (non-default
# optimizers) is accepted but not consumed here; existing adamw cell dirs match exactly as before.
_TAG_RE = PARAM_TAG_RE


@dataclass(frozen=True)
class LawFit:
    """Result of one collapse-law fit.

    Attributes:
        b: Alpha exponent, ``beta_alpha / beta_lr``.
        c: Rank exponent, ``beta_r / beta_lr``. Negative means higher rank is *protective*.
        b_ci: 95% interval for `b`.
        c_ci: 95% interval for `c`.
        params: Raw coefficients ``[intercept, beta_lr, beta_alpha, beta_r]``.
        n_runs: Seed-level runs in the fit.
        n_cells: Distinct cells (clusters).
        n_dead: Collapsed seed-runs.
        method: ``"delta"`` (cluster-robust delta method) or ``"bootstrap"``.
        converged: Whether the underlying optimiser reported convergence.
    """

    b: float
    c: float
    b_ci: tuple[float, float]
    c_ci: tuple[float, float]
    params: np.ndarray
    n_runs: int
    n_cells: int
    n_dead: int
    method: str
    converged: bool

    @property
    def dead_frac(self) -> float:
        """Fraction of seed-runs that collapsed."""
        return self.n_dead / self.n_runs if self.n_runs else float("nan")

    @property
    def c_excludes_zero(self) -> bool:
        """True when the rank exponent's interval excludes 0 (i.e. rank is in the law)."""
        return self.c_ci[1] < 0 or self.c_ci[0] > 0


def load_law_runs(
    output_root: Path,
    animal: str = "dog",
    numbers_base: str = "pythia-160m",
    teacher_seed: int = 1,
    dataset: str = "ex10-20_prefix_n100k",
    epochs: int = 20,
    lineages: list[tuple[str, str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load per-cell and per-seed results for the ``law-*`` campaign.

    Reads ``per_seed_p_target`` off each cell's ``preference_summary.json`` (the ``test``/``prob``
    entry). That field is written from the same per-seed evaluations as the individual
    ``ftseed*/preference_eval.json`` files -- verified identical across the whole campaign -- and
    costs one file read per cell instead of one per seed.

    Args:
        output_root: The ``outputs/`` root holding ``students/``.
        animal: Target animal subtree.
        numbers_base: Base model the teacher's numbers were generated from (the ``teacher-`` level).
        teacher_seed: Teacher fine-tuning seed embedded in the lineage directory name.
        dataset: Number-pool tag.
        epochs: Student epoch count to keep (the law is fit at ep20 only).
        lineages: Lineage specs; defaults to `LAW_LINEAGES`.

    Returns:
        ``(cells, runs)`` -- one row per cell and one row per (cell, seed) respectively. Both carry
        ``family``, ``base_label``, ``lineage``, ``student_tag``, ``lr``, ``rank``, ``alpha``.
    """
    lineages = LAW_LINEAGES if lineages is None else lineages
    students_root = Path(output_root) / "students"
    cell_rows: list[dict] = []
    run_rows: list[dict] = []

    for family, tag, base in lineages:
        lineage_dir = (
            students_root / animal / Path(*model_dir_parts_from_name(base))
            / f"teacher-{numbers_base}" / f"{tag}_tseed{teacher_seed}" / dataset
        )
        if not lineage_dir.exists():
            continue
        for cell_dir in sorted(lineage_dir.iterdir()):
            m = _TAG_RE.fullmatch(cell_dir.name)
            if not m or int(m["ep"]) != epochs:
                continue
            test = read_cell_set(cell_dir, "test")
            if test is None:
                continue
            per_seed = test.get("per_seed_p_target") or {}
            meta = {
                "family": family,
                "teacher_tag": tag,
                "base": base,
                "base_label": base.replace(f"{numbers_base}-", ""),
                "lineage": f"{family}:{base.replace(f'{numbers_base}-', '')}",
                "student_tag": cell_dir.name,
                "lr": float(m["lr"]),
                "rank": int(m["r"]),
                "alpha": int(m["a"]),
                "scale": int(m["a"]) / int(m["r"]),
            }
            cell_rows.append(
                {**meta, "mean_test": test.get("p_target_mean"), "n_seeds": len(per_seed)}
            )
            run_rows.extend({**meta, "seed": int(s), "test": float(v)} for s, v in per_seed.items())

    return pd.DataFrame(cell_rows), pd.DataFrame(run_rows)


def label_collapse(runs: pd.DataFrame, dead_seed: float = DEAD_SEED) -> pd.DataFrame:
    """Return `runs` with an integer ``collapsed`` column (``test < dead_seed``)."""
    return runs.assign(collapsed=runs["test"].lt(dead_seed).astype(int))


def _design(df: pd.DataFrame) -> np.ndarray:
    """Log-design matrix ``[1, ln lr, ln alpha, ln r]`` in the coefficient order `LawFit` assumes."""
    return np.c_[
        np.ones(len(df)), np.log(df["lr"]), np.log(df["alpha"]), np.log(df["rank"])
    ]


def fit_collapse_law(df: pd.DataFrame, alpha_level: float = 0.05) -> LawFit:
    """Fit the law on one lineage with cluster-robust (by cell) delta-method intervals.

    Fits ``logit P(collapse) = b0 + beta_lr*ln lr + beta_alpha*ln alpha + beta_r*ln r`` by
    statsmodels MLE, takes a cluster-robust covariance grouped on ``student_tag``, and propagates
    it to the ratios ``b`` and ``c`` by the delta method.

    Args:
        df: Seed-level rows for a single lineage, carrying ``collapsed``/``lr``/``alpha``/``rank``/
            ``student_tag``.
        alpha_level: Two-sided level for the intervals (0.05 -> 95%).

    Returns:
        The fitted `LawFit` with ``method="delta"``.
    """
    import statsmodels.api as sm
    from scipy.stats import norm

    from subliminal_learning.analysis.stats import delta_ratio_ci

    X, y = _design(df), df["collapsed"].to_numpy()
    res = sm.Logit(y, X).fit(
        disp=0, maxiter=500, cov_type="cluster",
        cov_kwds={"groups": df["student_tag"].to_numpy()},
    )
    p, V = np.asarray(res.params), np.asarray(res.cov_params())
    z = norm.ppf(1.0 - alpha_level / 2.0)

    b, b_ci = delta_ratio_ci(p, V, 2, 1, z=z)
    c, c_ci = delta_ratio_ci(p, V, 3, 1, z=z)
    return LawFit(
        b=b, c=c, b_ci=b_ci, c_ci=c_ci, params=p,
        n_runs=len(df), n_cells=int(df["student_tag"].nunique()), n_dead=int(y.sum()),
        method="delta", converged=bool(res.mle_retvals.get("converged", False)),
    )


def cluster_bootstrap_ratios(
    df: pd.DataFrame, n_boot: int = 600, seed: int = 0, alpha_level: float = 0.05
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Cell-clustered bootstrap of ``(b, c)`` using a ridge-penalised fit.

    Independent cross-check on `fit_collapse_law`: resamples whole cells with replacement so the
    within-cell seed correlation is respected, and uses the §12 ridge so near-separable replicates
    still yield finite coefficients.

    Args:
        df: Seed-level rows for a single lineage.
        n_boot: Bootstrap replicates.
        seed: RNG seed.
        alpha_level: Two-sided level for the percentile intervals.

    Returns:
        ``(draws, b_ci, c_ci)`` where ``draws`` has shape ``(n_valid, 2)`` of ``(b, c)`` pairs.
    """
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    # Pre-index rows by cell once; the inner loop then concatenates integer indices rather than
    # repeatedly slicing/concatenating frames (~100x faster over 600 replicates).
    idx_by_cell = [
        np.asarray(v) for v in df.reset_index(drop=True).groupby("student_tag").indices.values()
    ]
    logx = np.c_[np.log(df["lr"]), np.log(df["alpha"]), np.log(df["rank"])]
    y_all = df["collapsed"].to_numpy()
    n_cells = len(idx_by_cell)

    draws: list[tuple[float, float]] = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_cells, n_cells)
        rows = np.concatenate([idx_by_cell[i] for i in pick])
        y = y_all[rows]
        if y.min() == y.max():  # a degenerate replicate carries no information about the boundary
            continue
        model = LogisticRegression(C=SKLEARN_C, l1_ratio=0, solver="lbfgs", max_iter=20000, tol=1e-10)
        model.fit(logx[rows], y)
        w = model.coef_[0]
        if abs(w[0]) > 1e-6:
            draws.append((w[1] / w[0], w[2] / w[0]))

    arr = np.array(draws)
    lo, hi = 100 * alpha_level / 2, 100 * (1 - alpha_level / 2)
    return (
        arr,
        (float(np.percentile(arr[:, 0], lo)), float(np.percentile(arr[:, 0], hi))),
        (float(np.percentile(arr[:, 1], lo)), float(np.percentile(arr[:, 1], hi))),
    )


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse seed-level rows to one row per cell with a ``frac`` collapse fraction."""
    return (
        df.groupby("student_tag")
        .agg(lr=("lr", "first"), alpha=("alpha", "first"), rank=("rank", "first"),
             frac=("collapsed", "mean"), n=("collapsed", "size"))
        .reset_index()
    )


def estimate_kstar(df: pd.DataFrame, b: float, c: float) -> dict:
    """Locate the boundary ``K*`` on ``K = lr * alpha**b * r**c``.

    The boundary is stochastic: cells go from uniformly alive, through a band where they lose some
    but not all seeds, to uniformly dead. ``K*`` is taken as the geometric mean of the mixed cells'
    ``K``, which is the natural midpoint on a multiplicative scale.

    Args:
        df: Seed-level rows for one lineage, carrying ``collapsed``.
        b: Alpha exponent to evaluate ``K`` at.
        c: Rank exponent to evaluate ``K`` at.

    Returns:
        Dict with ``kstar``, the mixed-cell ``band`` (min, max), ``alive_max``, ``dead_min`` and
        ``n_mixed``.
    """
    cells = cell_table(df)
    K = K_of(cells["lr"], cells["alpha"], cells["rank"], b=b, c=c)
    mixed = K[(cells["frac"] > 0) & (cells["frac"] < 1)]
    return {
        "kstar": float(np.exp(np.mean(np.log(mixed)))) if len(mixed) else float("nan"),
        "band": (float(mixed.min()), float(mixed.max())) if len(mixed) else (float("nan"),) * 2,
        "alive_max": float(K[cells["frac"] == 0].max()) if (cells["frac"] == 0).any() else float("nan"),
        "dead_min": float(K[cells["frac"] == 1].min()) if (cells["frac"] == 1).any() else float("nan"),
        "n_mixed": int(len(mixed)),
    }


def law_misses(df: pd.DataFrame, b: float, c: float, kstar: float) -> tuple[int, int]:
    """Count cells whose majority verdict the law gets wrong.

    Args:
        df: Seed-level rows for one lineage, carrying ``collapsed``.
        b: Alpha exponent.
        c: Rank exponent.
        kstar: Boundary threshold on ``K``.

    Returns:
        ``(n_wrong, n_cells)`` comparing ``K >= kstar`` against ``frac >= 0.5``.
    """
    cells = cell_table(df)
    predicted = K_of(cells["lr"], cells["alpha"], cells["rank"], b=b, c=c) >= kstar
    observed = cells["frac"] >= 0.5
    return int((predicted != observed).sum()), int(len(cells))


def adapter_coherence(adapter_file: Path) -> pd.DataFrame:
    """SVD statistics of each LoRA module's update ``delta_W = B @ A`` in a trained adapter.

    Discriminates between the two limiting models of how an adapter's ``r`` rank-1 channels
    ``b_k a_k^T`` combine — the assumption that decides whether the ``1/r`` in the forward scale
    ``s = alpha/r`` cancels out of the applied perturbation (and hence whether *absolute* alpha
    belongs in the collapse law):

    - **iid channels** (the rsLoRA model, arXiv:2312.03732): the sum grows like ``sqrt(r)``,
      predicting ``stable_rank ~ r`` and rank exponent ``c = -0.25``.
    - **coherent channels** (Hu et al. §7, arXiv:2106.09685): the sum grows like ``r``,
      predicting ``stable_rank = O(1)`` independent of ``r`` and ``c = 0``.

    Reads the ``adapter_model.safetensors`` directly — no model load, no GPU.

    Args:
        adapter_file: Path to a PEFT ``adapter_model.safetensors`` checkpoint.

    Returns:
        One row per LoRA module pair with:

        - ``rank``: nominal LoRA rank ``r`` (columns of ``B``).
        - ``stable_rank``: ``||BA||_F^2 / sigma_1^2`` — effective number of directions the update
          energy occupies. The discriminating statistic (``~r`` iid vs ``O(1)`` coherent).
        - ``top1_energy``: ``sigma_1^2 / ||BA||_F^2`` — fraction of the update carried by the
          single largest singular direction.
        - ``b_col_cos``: mean pairwise ``|cos|`` between columns of ``B`` (the ``r`` output
          directions); ``NaN`` at rank 1. Random directions in ``d`` dims give
          ``~sqrt(2/(pi*d))`` (about 0.03 at ``d = 768``).
    """
    from safetensors.numpy import load_file  # training-stack dep; keep analysis import cheap
    from threadpoolctl import threadpool_limits  # ships with scikit-learn

    tensors = load_file(str(adapter_file))
    rows = []
    # The SVDs are small (<=3072x768) so BLAS multithreading buys nothing, but on a sweep-loaded
    # node the default one-thread-per-core pool thrashes: >900s/adapter at load ~2x cores vs ~10s
    # single-threaded (observed 2026-07-27).
    with threadpool_limits(limits=1):
        for key_a in sorted(k for k in tensors if k.endswith("lora_A.weight")):
            A = tensors[key_a].astype(np.float64)                              # (r, d_in)
            B = tensors[key_a.replace("lora_A", "lora_B")].astype(np.float64)  # (d_out, r)
            r = B.shape[1]
            sv = np.linalg.svd(B @ A, compute_uv=False)
            fro2 = float((sv**2).sum())
            cols = B / (np.linalg.norm(B, axis=0, keepdims=True) + 1e-30)
            cos = np.abs(cols.T @ cols)[np.triu_indices(r, 1)]
            rows.append({
                "module": key_a[: -len(".lora_A.weight")],
                "rank": r,
                "stable_rank": fro2 / float(sv[0] ** 2),
                "top1_energy": float(sv[0] ** 2) / fro2,
                "b_col_cos": float(cos.mean()) if r > 1 else float("nan"),
            })
    return pd.DataFrame(rows)
