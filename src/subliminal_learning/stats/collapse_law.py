"""Shared estimators for hyperparameter boundaries on caller-supplied binary outcomes.

The boundary coordinate is ``K = lr * alpha**b * rank**c``. The current ICLR
analyses supply catastrophic-failure labels from excess cross-entropy exceeding
an instrument-specific unigram reference. These estimators do not construct or
validate those labels: callers must select the population and measurement.
Legacy trait-label helpers and historical display constants remain below only
for compatibility; they do not define the current paper's failure criterion.

``fit_collapse_law`` fits an unpenalized logistic MLE and reports coefficient
ratios with cell-clustered delta-method intervals. Near or perfect separation
can make the fit unstable or prevent a finite MLE. Callers must inspect
convergence and finite intervals before interpreting the returned estimates.
``cluster_bootstrap_ratios`` is a separate ridge-penalized sensitivity estimator,
not an automatic replacement for a failed MLE.

``kstar_logistic`` calibrates a threshold at supplied exponents; its interval
conditions on those exponents. ``fit_collapse_law_trajectory`` adds a joint
log-step factor. Checkpoint-local paper figures instead repeat the endpoint
fitter at each checkpoint and separately calibrate fixed-coordinate thresholds.
See ``docs/paper/boundary_fitting_protocol.md`` for the current paper protocol.

This module contains estimators only. Filesystem loaders remain in the analysis
layer, including ``analysis.collapse_law``, which re-exports this API for older
callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from subliminal_learning.param_tags import parse_param_tag


# --- The collapse-boundary exponents (K = lr * alpha**B_LAW * rank**C_LAW; collapse at K >= K*).
# Reference values = the fit on the DESIGNED nonadaptive cross-data-order population of Pythia-160m
# (504 cells x ftseed{1,2}, anchored CE label, `fit_collapse_law` + `kstar_logistic`; paper Fig 1 /
# Table 3a, 2026-09-09): b 0.659 [0.60, 0.71], c -0.164 [-0.21, -0.11], K* 8.79e-4. Angela's rule
# (2026-09-09): the only quotable boundary numbers are fits of nonadaptive sweeps -- never a tree glob,
# never a trait-era or fixed-cut "display" value. These constants are the cross-family reference
# coordinate; every arm's own fit is the quotable object for that arm.
B_LAW: float = 0.659
C_LAW: float = -0.164
KSTAR_CALIBRATED: float = 8.79e-4

# Compatibility aliases for older callers that named a pooled / ds1-design anchor. Both now resolve
# to the designed-population K* above; the placement-era 7.87e-4 / 7.84e-4 values were retired
# 2026-09-09 (trait-label, contaminated support) and must not be reintroduced.
KSTAR_POOLED: float = KSTAR_CALIBRATED
KSTAR_DS1_DESIGN: float = KSTAR_CALIBRATED


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
        tag: A canonical sweep tag (parsed by `param_tags.parse_param_tag`).
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


def label_collapse(runs: pd.DataFrame, dead_seed: "float | pd.Series" = DEAD_SEED) -> pd.DataFrame:
    """Return `runs` with an integer ``collapsed`` column (``test < dead_seed``).

    ``dead_seed`` is either one scalar or a per-row Series aligned to ``runs.index`` (e.g. a
    per-receiver threshold). This is the LEGACY trait-side label; the operative collapse label
    for the ICLR paper is the unigram-anchored excess-CE cut -- see `analysis.labels` (`label_dead_trait_legacy` wraps this
    rule with the caveat attached).
    """
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

    from subliminal_learning.stats.core import delta_ratio_ci

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


@dataclass(frozen=True)
class TrajectoryLawFit:
    """The law fit on checkpoint-resolved labels with training step as a fourth factor.

    ``logit P(collapsed at step t) = b0 + beta_lr ln lr + beta_alpha ln alpha + beta_r ln r
    + beta_t ln t``, so the boundary is ``lr * alpha**b * r**c * t**d = const``, i.e. the
    threshold on ``K = lr * alpha**b * r**c`` moves as ``K*(t) ∝ t**(-d)``.

    Attributes:
        b, c: The usual alpha / rank ratios (``beta_alpha/beta_lr``, ``beta_r/beta_lr``).
        d: Step ratio ``beta_t/beta_lr`` -- the exponent of the threshold's decline in steps.
        b_ci, c_ci, d_ci: Cluster-robust (by run) delta-method intervals.
        params: Raw coefficients ``[intercept, beta_lr, beta_alpha, beta_r, beta_t]``.
        n_rows: (run, checkpoint) rows in the fit.
        n_runs: Distinct runs (clusters).
        n_dead_rows: Rows labelled collapsed.
        converged: Optimiser convergence flag (near-separation makes False common).
    """

    b: float
    c: float
    d: float
    b_ci: tuple[float, float]
    c_ci: tuple[float, float]
    d_ci: tuple[float, float]
    params: np.ndarray
    n_rows: int
    n_runs: int
    n_dead_rows: int
    converged: bool

    def kstar_at(self, step: float) -> float:
        """Threshold on ``K = lr * alpha**b * r**c`` at a given step (P = 0.5)."""
        b0, bl, _, _, bt = self.params
        return float(np.exp(-(b0 + bt * np.log(step)) / bl))


def fit_collapse_law_trajectory(df: pd.DataFrame, *, step: str = "step",
                                alpha_level: float = 0.05) -> TrajectoryLawFit:
    """Fit the law on checkpoint-resolved labels, with ``ln step`` as a fourth covariate.

    Rows are (run, checkpoint) pairs carrying ``collapsed`` (that checkpoint's own label),
    ``lr``/``alpha``/``rank`` and the step column; the covariance clusters on ``student_tag``
    so the 40-odd checkpoints of one run never count as independent evidence. Ratios only,
    as in `fit_collapse_law` (near-separation leaves the magnitudes unidentified).

    Args:
        df: Long frame with ``collapsed``, ``lr``, ``alpha``, ``rank``, ``student_tag`` and
            the step column (must be > 0).
        step: Name of the step column.
        alpha_level: Two-sided level for the intervals.

    Returns:
        A `TrajectoryLawFit`; ``d > 0`` means the threshold falls with training.
    """
    import statsmodels.api as sm
    from scipy.stats import norm

    from subliminal_learning.stats.core import delta_ratio_ci

    X = np.c_[_design(df), np.log(df[step].to_numpy(dtype=float))]
    y = df["collapsed"].to_numpy()
    res = sm.Logit(y, X).fit(
        disp=0, maxiter=500, cov_type="cluster",
        cov_kwds={"groups": df["student_tag"].to_numpy()},
    )
    p, V = np.asarray(res.params), np.asarray(res.cov_params())
    z = norm.ppf(1.0 - alpha_level / 2.0)
    b, b_ci = delta_ratio_ci(p, V, 2, 1, z=z)
    c, c_ci = delta_ratio_ci(p, V, 3, 1, z=z)
    d, d_ci = delta_ratio_ci(p, V, 4, 1, z=z)
    return TrajectoryLawFit(
        b=b, c=c, d=d, b_ci=b_ci, c_ci=c_ci, d_ci=d_ci, params=p,
        n_rows=len(df), n_runs=int(df["student_tag"].nunique()), n_dead_rows=int(y.sum()),
        converged=bool(res.mle_retvals.get("converged", False)),
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


def kstar_logistic(df: pd.DataFrame, b: float, c: float) -> dict:
    """The boundary as the logistic 50% crossing on ``log K`` -- no seed replication needed.

    The canonical K* estimator (2026-08-20, Angela): fit ``P(collapsed) = sigma(a + w log K)``
    on ALL seed-level rows and return the K where P = 0.5. Unlike `estimate_kstar`'s
    mixed-cell geometric mean, this needs no cell to straddle the boundary internally, so it
    works on single-seed campaigns; it is exactly the fixed-exponent 1-D slice of the linear
    boundary `fit_collapse_law` fits in full log-HP space. `kstar_minerr` (the
    misclassification-minimizing cut) is the assumption-free cross-check.

    Returns:
        Dict with ``kstar``, ``se_log`` (delta-method SE of log K*), ``slope`` (w; steepness
        of the transition), ``err`` (misclassification at the 0.5 rule), ``n``, ``n_dead``.
    """
    import statsmodels.api as sm

    K = K_of(df["lr"], df["alpha"], df["rank"], b=b, c=c)
    y = df["collapsed"].to_numpy()
    x = sm.add_constant(np.log(K.to_numpy()))
    # cluster-robust by cell when available: runs sharing a cell (teachers/arms at one tag)
    # have correlated outcomes, so the naive Bernoulli SE is optimistic
    kw = ({"cov_type": "cluster", "cov_kwds": {"groups": df["student_tag"].to_numpy()}}
          if "student_tag" in df and df["student_tag"].nunique() > 1 else {})
    res = sm.GLM(y, x, family=sm.families.Binomial()).fit(**kw)
    a, w = res.params
    kstar = float(np.exp(-a / w))
    # delta method on log K* = -a/w
    grad = np.array([-1.0 / w, a / w**2])
    se_log = float(np.sqrt(grad @ res.cov_params() @ grad))
    err = float((((K >= kstar).astype(int)) != y).mean())
    return {"kstar": kstar, "se_log": se_log, "slope": float(w), "err": err,
            "n": int(len(y)), "n_dead": int(y.sum())}


def kstar_minerr(df: pd.DataFrame, b: float, c: float) -> dict:
    """The boundary as the misclassification-minimizing cut on ``K`` (max-margin midpoint).

    Assumption-free cross-check for `kstar_logistic`: scan every observed K as a candidate
    threshold, keep the one with the fewest label errors, and place the cut at the geometric
    midpoint between it and its lower neighbour (the 1-D max-margin choice within the
    error-tied interval).

    Returns:
        Dict with ``kstar``, ``err``, ``n``, ``n_dead``.
    """
    K = np.asarray(K_of(df["lr"], df["alpha"], df["rank"], b=b, c=c))
    y = df["collapsed"].to_numpy()
    ks = np.unique(K)
    errs = [(float((((K >= cut).astype(int)) != y).mean()), i) for i, cut in enumerate(ks)]
    err, i = min(errs)
    kstar = float(np.sqrt(ks[i] * ks[i - 1])) if i > 0 else float(ks[i])
    return {"kstar": kstar, "err": err, "n": int(len(y)), "n_dead": int(y.sum())}


def theta_dose(df: pd.DataFrame, dose_col: str = "dw_fro") -> dict:
    """The boundary in the REALIZED-update coordinate: the logistic 50% crossing on ``log dose``.

    `kstar_logistic`'s estimator moved off the planned-HP coordinate ``K`` onto the measured
    dose (``dw_fro`` = module-mean ``(alpha/rank)*||B A_final||_F/||W0||_F``), giving θ̂ — the
    dose at which P(dead) = 0.5. Same conventions: cluster-robust by ``student_tag`` when the
    frame spans cells, delta-method SE on ``log θ̂``, and a misclassification-minimizing cut
    (``theta_minerr``, the geometric midpoint of the error-tied interval) as the
    assumption-free check.

    Args:
        df: Seed-level rows carrying ``collapsed`` and `dose_col` (finite, positive).
        dose_col: The dose column to place the boundary on.

    Returns:
        Dict with ``theta``, ``se_log``, ``slope``, ``err``, ``converged``, ``theta_minerr``,
        ``err_minerr``, ``auc`` (Mann-Whitney P(dose_dead > dose_alive)), ``n``, ``n_dead``.
    """
    import statsmodels.api as sm

    from subliminal_learning.stats.core import mannwhitney_auc

    d = np.asarray(df[dose_col], dtype=float)
    y = df["collapsed"].to_numpy()
    x = sm.add_constant(np.log(d))
    kw = ({"cov_type": "cluster", "cov_kwds": {"groups": df["student_tag"].to_numpy()}}
          if "student_tag" in df and df["student_tag"].nunique() > 1 else {})
    res = sm.GLM(y, x, family=sm.families.Binomial()).fit(**kw)
    a, w = res.params
    theta = float(np.exp(-a / w))
    grad = np.array([-1.0 / w, a / w**2])
    se_log = float(np.sqrt(grad @ res.cov_params() @ grad))

    cuts = np.unique(d)
    err_min, i = min((float((((d >= cut).astype(int)) != y).mean()), j)
                     for j, cut in enumerate(cuts))
    return {
        "theta": theta, "se_log": se_log, "slope": float(w),
        "err": float((((d >= theta).astype(int)) != y).mean()),
        "converged": bool(res.converged),
        "theta_minerr": float(np.sqrt(cuts[i] * cuts[i - 1])) if i > 0 else float(cuts[i]),
        "err_minerr": err_min,
        "auc": float(mannwhitney_auc(d[y == 1], d[y == 0])),
        "n": int(len(y)), "n_dead": int(y.sum()),
    }


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


def theta_dose_row(g: pd.DataFrame, dose_col: str = "dw_fro") -> dict:
    """`theta_dose` on one slice, JSON-shaped, with single-class slices skipped.

    The per-arm row of the θ̂ cross-animal table: counts first (they are meaningful even when the
    fit is not), then the fit. A slice whose ``collapsed`` column has only one class carries no
    boundary, so it reports ``fit_skipped`` and a ``NaN`` AUC instead of a fit.

    Args:
        g: Seed-level rows carrying ``collapsed``, `dose_col` and ``student_tag``.
        dose_col: Dose column the boundary is placed on.

    Returns:
        Dict with ``n``, ``n_dead``, ``n_cells``, ``auc`` and -- when fitted -- ``theta``,
        ``theta_se_log``, ``slope``, ``err_at_theta``, ``converged``, ``theta_minerr``,
        ``err_minerr``.
    """
    from subliminal_learning.stats.core import json_scalar

    base = {"n": int(len(g)), "n_dead": int(g.collapsed.sum()),
            "n_cells": int(g.student_tag.nunique())}
    if g.collapsed.nunique() < 2:
        return {**base, "auc": float("nan"), "fit_skipped": "single-class"}
    t = theta_dose(g, dose_col)
    return {**base, "auc": json_scalar(t["auc"]), "theta": json_scalar(t["theta"]),
            "theta_se_log": json_scalar(t["se_log"]), "slope": json_scalar(t["slope"]),
            "err_at_theta": json_scalar(t["err"]), "converged": t["converged"],
            "theta_minerr": json_scalar(t["theta_minerr"]),
            "err_minerr": json_scalar(t["err_minerr"])}


def dose_law_fit(g: pd.DataFrame) -> dict:
    """Label-free HP->dose OLS ``log dw_fro ~ log lr + log alpha + log rank``, clustered by cell.

    The dose-side counterpart of the boundary fits: it asks how the planned hyper-parameters set
    the *realized* update size, with no fate label anywhere in it. Clustering is on
    ``student_tag`` (seeds of a cell share a dose recipe).

    Args:
        g: Seed-level rows carrying ``dw_fro``, ``lr``, ``alpha``, ``rank`` and ``student_tag``.

    Returns:
        Dict with ``n``, ``n_cells``, ``r2``, the three exponents and their cluster SEs
        (``b_lr``/``se_lr``, ``b_alpha``/``se_alpha``, ``b_rank``/``se_rank``) and ``intercept``.
    """
    from subliminal_learning.stats.core import cluster_ols, json_scalar

    d = g.assign(y=np.log(g.dw_fro), lln=np.log(g.lr), aln=np.log(g.alpha),
                 rln=np.log(g["rank"]), cluster=g.student_tag)
    m = cluster_ols(d, "y ~ lln + aln + rln", "cluster")
    return {"n": int(len(d)), "n_cells": int(d.cluster.nunique()), "r2": json_scalar(m.rsquared),
            "b_lr": json_scalar(m.params["lln"]), "se_lr": json_scalar(m.bse["lln"]),
            "b_alpha": json_scalar(m.params["aln"]), "se_alpha": json_scalar(m.bse["aln"]),
            "b_rank": json_scalar(m.params["rln"]), "se_rank": json_scalar(m.bse["rln"]),
            "intercept": json_scalar(m.params["Intercept"])}


def theta_variance_split(arms: pd.DataFrame) -> dict:
    """R² of ``log θ̂`` under an additive animal+base model and under each factor alone.

    How much of the per-arm boundary spread is the receiver base and how much is the trait: the
    three R² are read together (base-only near the additive value means the animal adds nothing).
    Ordinary (unclustered) OLS -- the rows are already one per arm.

    Args:
        arms: One row per arm with ``theta``, ``animal`` and ``base``.

    Returns:
        Dict with ``r2_additive``, ``r2_animal_only``, ``r2_base_only`` and the per-factor mean θ̂
        (``base_means``, ``animal_means``, rounded to 4 dp).
    """
    import statsmodels.formula.api as smf

    d = arms.assign(lt=np.log(arms.theta))
    r2 = {name: float(smf.ols(f"lt ~ {rhs}", d).fit().rsquared)
          for name, rhs in (("r2_additive", "C(animal) + C(base)"),
                            ("r2_animal_only", "C(animal)"), ("r2_base_only", "C(base)"))}
    return {**r2,
            "base_means": {k: round(v, 4) for k, v in d.groupby("base").theta.mean().items()},
            "animal_means": {k: round(v, 4) for k, v in d.groupby("animal").theta.mean().items()}}


def law_fits_row(g: pd.DataFrame, b_fixed: float, c_fixed: float) -> dict | None:
    """Free law + both K* estimates on one slice; ``None`` when the slice cannot be fitted.

    Runs `fit_collapse_law` for the free exponents, then `kstar_logistic` twice: once at the
    slice's own fitted ``(b, c)`` and once at the caller's FIXED display exponents, so arms that
    each fit their own law can still be compared in one shared coordinate.

    Args:
        g: Seed-level rows carrying ``collapsed``, ``lr``, ``alpha``, ``rank``, ``student_tag``.
        b_fixed: Alpha exponent for the shared-coordinate K*.
        c_fixed: Rank exponent for the shared-coordinate K*.

    Returns:
        Dict with ``n``/``n_dead``/``n_cells``, ``b``/``c`` and their CIs, ``kstar_free`` and
        ``kstar_fixed`` with their log SEs, and ``converged`` -- or ``None`` for a single-class
        slice or a singular logit Hessian.
    """
    from subliminal_learning.stats.core import json_scalar

    if g.collapsed.nunique() < 2:
        return None
    try:
        fit = fit_collapse_law(g)
        free = kstar_logistic(g, fit.b, fit.c)
        fixed = kstar_logistic(g, b_fixed, c_fixed)
    except Exception:
        return None
    return {"n": fit.n_runs, "n_dead": fit.n_dead, "n_cells": fit.n_cells,
            "b": json_scalar(fit.b), "c": json_scalar(fit.c),
            "b_ci": [json_scalar(v) for v in fit.b_ci],
            "c_ci": [json_scalar(v) for v in fit.c_ci],
            "kstar_free": json_scalar(free["kstar"]),
            "kstar_free_se_log": json_scalar(free["se_log"]),
            "converged": bool(fit.converged),
            "kstar_fixed": json_scalar(fixed["kstar"]),
            "kstar_fixed_se_log": json_scalar(fixed["se_log"])}


def kstar_spread(rows: list[dict], min_dead: int) -> dict:
    """K* min/max/ratio (free and fixed exponents) over the arms with at least `min_dead` dead.

    The min-dead floor is the point: arms with a handful of deaths sit on the boundary tail and
    their K* is noise, so they are counted but never allowed to set the spread.

    Args:
        rows: Per-arm dicts carrying ``n_dead``, ``kstar_free``, ``kstar_fixed``, ``animal``,
            ``base``.
        min_dead: Arms below this many dead runs are dropped.

    Returns:
        Dict with ``n_arms``, ``kstar_{free,fixed}_{min,max,ratio}`` (``None`` when no arm
        qualifies) and the qualifying ``arms`` as ``"animal|base"`` labels.
    """
    from subliminal_learning.stats.core import json_scalar

    keep = [r for r in rows if r["n_dead"] >= min_dead]
    out: dict = {"n_arms": len(keep)}
    for which in ("free", "fixed"):
        v = np.array([r[f"kstar_{which}"] for r in keep], dtype=float)
        out[f"kstar_{which}_min"] = json_scalar(v.min()) if len(v) else None
        out[f"kstar_{which}_max"] = json_scalar(v.max()) if len(v) else None
        out[f"kstar_{which}_ratio"] = json_scalar(v.max() / v.min()) if len(v) else None
    out["arms"] = [f"{r['animal']}|{r['base']}" for r in keep]
    return out
