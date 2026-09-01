"""The mediation-ladder vocabulary shared by the collapse screens.

Two scripts ask the same question of two populations -- ``scripts/mediation_twopart_kl.py`` on
students and ``scripts/teacher_cos_screen.py`` on teachers: once the realized dose ``log dw_fro``
is in the model, does the hyper-parameter block still reach fate, and does adding
``cos(dW, W0)`` screen whatever is left? Both build the same three objects, so they live here:

- a **nested pair** of cluster-robust logistic fits (reduced vs full) plus the joint test that
  the added terms are zero -- `nested_logit_block`;
- **coefficient / term readouts** off a fitted result -- `coef_table`, `term_stats`;
- **fate AUCs**, either of a raw score or of a model's fitted values -- `fate_auc`, `fitted_auc`.

Two conventions the two screens agreed on and that are pinned here:

- **Two block tests, always both.** The likelihood-ratio statistic is a *plain* chi2 on the
  unclustered log-likelihoods (statsmodels' ``llf`` is the plain MLE value even under a cluster
  covariance), while the Wald statistic uses the cluster sandwich. They can disagree; quote both,
  which is why `BlockTest` carries both rather than one "the" p-value.
- **z-score the cosine.** ``cos(dW, W0)`` is O(1e-4), so BFGS is badly conditioned on the raw
  column; `zscore_column` standardises it (population sd, ``ddof=0``) and returns the ``(mu, sd)``
  needed to report the coefficient back on the raw scale as ``coef / sd``.

Callers own their own output keys: `BlockTest` is a dataclass, not a JSON dict, precisely because
the two scripts name these fields differently in their artifacts and neither schema may drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from subliminal_learning.stats.core import cluster_logit, json_scalar, mannwhitney_auc


@dataclass(frozen=True)
class BlockTest:
    """Joint test that a block of terms is zero, both ways.

    Attributes:
        lr_chi2: Likelihood-ratio statistic ``2 * (llf_full - llf_reduced)`` (plain MLE llf).
        lr_df: Number of tested terms.
        lr_p: Chi-squared survival p-value of `lr_chi2` on `lr_df`.
        wald_chi2: Wald statistic from the full fit's cluster-robust covariance.
        wald_p: Its p-value.
    """

    lr_chi2: float
    lr_df: int
    lr_p: float
    wald_chi2: float
    wald_p: float


@dataclass(frozen=True)
class LadderRung:
    """One reduced-vs-full step of a mediation ladder.

    Attributes:
        reduced: The fitted results object for the reduced formula.
        full: The fitted results object for the full formula.
        block: The joint test that `terms` are zero in `full`.
    """

    reduced: object
    full: object
    block: BlockTest


def block_test(res_full, res_reduced, terms) -> BlockTest:
    """LR (plain chi2) + cluster-sandwich Wald test that `terms` are jointly zero.

    Args:
        res_full: Fitted result including `terms`.
        res_reduced: Fitted result on the same rows without them.
        terms: Term names as they appear in the full fit's design.

    Returns:
        A `BlockTest`; the scalars are `json_scalar`-cleaned so they drop straight into an
        artifact.
    """
    from scipy import stats as sps

    terms = list(terms)
    lr = 2.0 * (res_full.llf - res_reduced.llf)
    wald = res_full.wald_test(", ".join(f"{t} = 0" for t in terms), scalar=True)
    return BlockTest(
        lr_chi2=json_scalar(lr),
        lr_df=len(terms),
        lr_p=json_scalar(sps.chi2.sf(lr, len(terms))),
        wald_chi2=json_scalar(wald.statistic),
        wald_p=json_scalar(wald.pvalue),
    )


def nested_logit_block(
    df: pd.DataFrame,
    formula_reduced: str,
    formula_full: str,
    *,
    cluster: str,
    terms,
    method: str = "bfgs",
    maxiter: int = 500,
) -> LadderRung:
    """Fit a reduced/full logistic pair on the SAME rows and test the added block.

    Call sites pass both formulas literally rather than a base plus a term list: the screens'
    formulas differ in where the block sits (mid-formula in the tree-fixed-effects variants), and
    a helper that assembled the string would silently reorder the design.

    Args:
        df: Rows both fits use -- same subset, so the two log-likelihoods are comparable.
        formula_reduced: Patsy formula without the tested block.
        formula_full: Same formula plus `terms`.
        cluster: Column the cluster-robust covariance groups on.
        terms: The added term names, for the joint test.
        method: Optimizer (default ``"bfgs"``: the near-separation regime needs it).
        maxiter: Iteration cap.

    Returns:
        A `LadderRung`.
    """
    reduced = cluster_logit(df, formula_reduced, cluster, method=method, maxiter=maxiter)
    full = cluster_logit(df, formula_full, cluster, method=method, maxiter=maxiter)
    return LadderRung(reduced=reduced, full=full, block=block_test(full, reduced, terms))


def coef_table(res) -> dict:
    """``{term: {coef, se_cluster}}`` for every fitted term except the intercept."""
    return {name: {"coef": json_scalar(res.params[name]),
                   "se_cluster": json_scalar(res.bse[name])}
            for name in res.params.index if name != "Intercept"}


def term_stats(res, name: str) -> dict:
    """One term's cluster-robust readout: ``{coef, se_cluster, z, p_cluster}``.

    The p-value is the two-sided normal tail of ``coef / se_cluster`` -- computed here rather
    than read off ``res.pvalues`` so it is unambiguously the cluster-sandwich z test.
    """
    from scipy import stats as sps

    z = res.params[name] / res.bse[name]
    return {"coef": json_scalar(res.params[name]), "se_cluster": json_scalar(res.bse[name]),
            "z": json_scalar(z), "p_cluster": json_scalar(2 * sps.norm.sf(abs(z)))}


def fate_auc(score: pd.Series, dead: pd.Series) -> float:
    """Mann-Whitney ``P(score_dead > score_alive)`` for a fate column.

    Args:
        score: Any per-run score (a dose, or a model's fitted values).
        dead: Aligned 0/1 (or boolean) fate column.

    Returns:
        The AUC, or ``NaN`` when either class is empty.
    """
    m = np.asarray(dead.values).astype(bool)
    v = np.asarray(score.values)
    return float(mannwhitney_auc(v[m], v[~m]))


def fitted_auc(res, dead: pd.Series, index=None) -> float:
    """`fate_auc` of a fitted model's linear predictor / fitted values.

    Args:
        res: A fitted statsmodels result whose ``fittedvalues`` align with `dead`.
        dead: Aligned fate column.
        index: Index to attach to the fitted values (defaults to `dead`'s).
    """
    idx = dead.index if index is None else index
    return fate_auc(pd.Series(np.asarray(res.fittedvalues), index=idx), dead)


def zscore_column(df: pd.DataFrame, col: str, out: str = "cosz") -> tuple[pd.DataFrame, float, float]:
    """Standardise `col` into a new column `out` on a copy of `df`.

    Uses the population sd (``ddof=0``). Report the fitted coefficient back on the raw scale as
    ``coef / sd``.

    Returns:
        ``(frame, mean, sd)``.
    """
    d = df.copy()
    mu, sd = float(d[col].mean()), float(d[col].std(ddof=0))
    d[out] = (d[col] - mu) / sd
    return d, mu, sd
