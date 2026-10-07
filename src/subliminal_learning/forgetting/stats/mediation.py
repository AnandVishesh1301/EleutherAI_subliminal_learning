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
  which is why `BlockTest` carries both rather than one "the" p-value. The plain LR p assumes
  *independent* observations, which these populations are not (seeds share a cell), so under
  clustering the **Wald p is the one to report**; the LR p is the same-model diagnostic beside it.
- **A non-converged pair reports nothing.** `nested_logit_block` marks a rung whose reduced or
  full fit did not converge as ``converged=False`` and blanks its p-values instead of raising:
  the screens write one artifact over many populations and one bad rung must not abort the run.
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
        lr_df: Parameter-count difference of the two fits -- the number of design *columns* the
            full fit adds, which is what the chi2 null needs: a factor term (``C(tree)`` with
            three levels) contributes two columns, not one term name.
        lr_p: Chi-squared survival p-value of `lr_chi2` on `lr_df`. Assumes independent
            observations; under clustering quote `wald_p`.
        wald_chi2: Wald statistic from the full fit's cluster-robust covariance.
        wald_p: Its p-value.
        converged: True only when BOTH fits converged. When False both p-values are blanked
            (``None``, this module's `json_scalar` spelling of ``NaN``) so a non-converged pair
            cannot emit a number that looks valid.
    """

    lr_chi2: float
    lr_df: int
    lr_p: float
    wald_chi2: float
    wald_p: float
    converged: bool = True


@dataclass(frozen=True)
class LadderRung:
    """One reduced-vs-full step of a mediation ladder.

    Attributes:
        reduced: The fitted results object for the reduced formula.
        full: The fitted results object for the full formula.
        block: The joint test that `terms` are zero in `full`.
        converged: True only when both fits converged (same flag as ``block.converged``).
    """

    reduced: object
    full: object
    block: BlockTest
    converged: bool = True


def fit_converged(res) -> bool:
    """Whether a fitted result's optimizer converged (`cluster_logit`'s attribute, else retvals)."""
    flag = getattr(res, "converged", None)
    if flag is None:
        flag = (getattr(res, "mle_retvals", None) or {}).get("converged", False)
    return bool(flag)


def block_test(res_full, res_reduced, terms) -> BlockTest:
    """LR (plain chi2) + cluster-sandwich Wald test that `terms` are jointly zero.

    The LR degrees of freedom come from the two fits' parameter counts (statsmodels'
    ``compare_lr_test`` where the results object exposes it), not from ``len(terms)``: a
    factor term expands to several design columns, and counting term names would understate
    the null's df.

    Args:
        res_full: Fitted result including `terms`.
        res_reduced: Fitted result on the same rows without them.
        terms: Term names as they appear in the full fit's design.

    Returns:
        A `BlockTest`; the scalars are `json_scalar`-cleaned so they drop straight into an
        artifact. When either fit failed to converge, ``converged`` is False and both p-values
        are ``None``.

    Raises:
        ValueError: If the two fits did not use the same rows, or if the full fit's parameter
            set does not strictly contain the reduced fit's (the two formulas are not nested),
            naming the missing and extra columns.
    """
    from scipy import stats as sps

    terms = list(terms)
    full_cols, reduced_cols = list(res_full.params.index), list(res_reduced.params.index)
    missing = [c for c in reduced_cols if c not in full_cols]
    extra = [c for c in full_cols if c not in reduced_cols]
    if missing or not extra:
        raise ValueError(
            "the full fit must strictly contain the reduced fit's design: "
            f"missing from full {missing or 'none'}; added by full {extra or 'none'}"
        )
    if res_full.nobs != res_reduced.nobs:
        raise ValueError(
            "reduced and full fits use different rows, so their log-likelihoods are not "
            f"comparable: reduced nobs {res_reduced.nobs:.0f} vs full nobs {res_full.nobs:.0f}"
        )
    if hasattr(res_full, "compare_lr_test"):
        lr, _p, df_diff = res_full.compare_lr_test(res_reduced)
        lr, df = float(lr), int(df_diff)
    else:
        # statsmodels exposes compare_lr_test only on the regression/GLM results; the discrete
        # results these screens fit (Logit) do not carry it, so the same quantities are formed
        # here -- the parameter-count difference is exactly the df it would return.
        lr = 2.0 * (res_full.llf - res_reduced.llf)
        df = len(full_cols) - len(reduced_cols)
    wald = res_full.wald_test(", ".join(f"{t} = 0" for t in terms), scalar=True)
    converged = fit_converged(res_full) and fit_converged(res_reduced)
    nan = float("nan")
    return BlockTest(
        lr_chi2=json_scalar(lr),
        lr_df=df,
        lr_p=json_scalar(sps.chi2.sf(lr, df) if converged else nan),
        wald_chi2=json_scalar(wald.statistic),
        wald_p=json_scalar(wald.pvalue if converged else nan),
        converged=converged,
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
        A `LadderRung`. A rung whose reduced or full fit did not converge carries
        ``converged=False`` and blanked block p-values rather than raising.

    Raises:
        ValueError: If the two fits did not land on the same rows (patsy drops rows with a
            ``NaN`` in any of *that* formula's columns, so a covariate only the full formula
            uses can silently shrink it), or if the formulas are not nested.
    """
    import statsmodels.formula.api as smf

    # Checked on the designs, before fitting: a formula that drops rows also desynchronises the
    # cluster groups, and statsmodels would raise a length error with no diagnosis.
    n_reduced = int(smf.logit(formula_reduced, df).exog.shape[0])
    n_full = int(smf.logit(formula_full, df).exog.shape[0])
    if n_reduced != n_full:
        raise ValueError(
            "reduced and full fits must use the same rows, so their log-likelihoods are "
            f"comparable: reduced nobs {n_reduced} on '{formula_reduced}' vs full nobs "
            f"{n_full} on '{formula_full}'"
        )
    reduced = cluster_logit(df, formula_reduced, cluster, method=method, maxiter=maxiter)
    full = cluster_logit(df, formula_full, cluster, method=method, maxiter=maxiter)
    if full.nobs != reduced.nobs:
        raise ValueError(
            "reduced and full fits use different rows: reduced nobs "
            f"{reduced.nobs:.0f} vs full nobs {full.nobs:.0f}"
        )
    block = block_test(full, reduced, terms)
    return LadderRung(reduced=reduced, full=full, block=block, converged=block.converged)


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
