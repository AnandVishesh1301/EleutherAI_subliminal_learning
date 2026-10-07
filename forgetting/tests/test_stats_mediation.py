"""Equivalence tests for the mediation ladder lifted out of the collapse screens.

The helpers in ``subliminal_learning.forgetting.stats.mediation`` (and the two general helpers they lean on
in ``stats.core``) were extracted verbatim from ``scripts/mediation_twopart_kl.py`` and
``scripts/teacher_cos_screen.py``, which produced published numbers. So the oracles below are
literal transcriptions of the old inline code -- statsmodels called exactly the way the scripts
called it -- not re-derivations: the point of each test is that the move changed nothing.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from subliminal_learning.stats.core import json_scalar, quantile_table
from subliminal_learning.forgetting.stats.mediation import (
    block_test,
    coef_table,
    fate_auc,
    fitted_auc,
    nested_logit_block,
    term_stats,
    zscore_column,
)

HP_TERMS = ["lln", "aln", "rln"]


# ------------------------- oracles (verbatim from the scripts) -------------------------
def _logit(df, formula, cluster="cell"):                 # mediation_twopart_kl.py:76-81
    import statsmodels.formula.api as smf

    return smf.logit(formula, df).fit(method="bfgs", maxiter=500, disp=0,
                                      cov_type="cluster", cov_kwds={"groups": df[cluster]})


def _j(x):                                               # mediation_twopart_kl.py:66-73
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if np.isfinite(x) else None
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


def _coefs(res):                                         # mediation_twopart_kl.py:84-86
    return {n: {"coef": _j(res.params[n]), "se_cluster": _j(res.bse[n])}
            for n in res.params.index if n != "Intercept"}


def _block(res_full, res_reduced, terms=HP_TERMS):       # mediation_twopart_kl.py:89-98
    lr = 2.0 * (res_full.llf - res_reduced.llf)
    wald = res_full.wald_test(", ".join(f"{t} = 0" for t in terms), scalar=True)
    return {"hp_block_lr_chi2": _j(lr), "hp_block_lr_p": _j(sps.chi2.sf(lr, len(terms))),
            "hp_block_lr_df": len(terms),
            "hp_block_cluster_wald_chi2": _j(wald.statistic),
            "hp_block_cluster_wald_p": _j(wald.pvalue)}


def _auc(score, dead):                                   # mediation_twopart_kl.py:101-104
    from subliminal_learning.stats.core import mannwhitney_auc
    m = dead.values.astype(bool)
    return float(mannwhitney_auc(score.values[m], score.values[~m]))


def _term(res, name):                                    # teacher_cos_screen.py:115-119
    z = res.params[name] / res.bse[name]
    return {"coef": _j(res.params[name]), "se_cluster": _j(res.bse[name]),
            "z": _j(z), "p_cluster": _j(2 * sps.norm.sf(abs(z)))}


def _qtab(df, by, col="mean_cos"):                       # teacher_cos_screen.py:122-130
    out = {}
    for key, g in df.groupby(by, sort=True):
        k = "_".join(str(_j(x)) for x in key) if isinstance(key, tuple) else str(_j(key))
        s = g[col]
        out[k] = {"n": int(s.notna().sum()), "median": _j(s.median()),
                  "q25": _j(s.quantile(0.25)), "q75": _j(s.quantile(0.75))}
    return out


# ------------------------- synthetic population -------------------------
@pytest.fixture(scope="module")
def screen_df() -> pd.DataFrame:
    """A small non-separable dose/HP/fate frame with several seeds per cell (cluster)."""
    rng = np.random.default_rng(7)
    rows = []
    for lr in (1e-4, 3e-4, 1e-3):
        for a in (8, 32, 64):
            for r in (8, 32):
                cell = f"ep20_lr{lr:g}_r{r}a{a}"
                for seed in range(6):
                    dw = lr * a**0.5 * r**-0.1 * math.exp(0.35 * rng.standard_normal())
                    cos = 1e-4 * (1.0 + 0.4 * rng.standard_normal())
                    p = 1.0 / (1.0 + math.exp(-(3.0 * math.log(dw / 4e-4) + 0.5 * math.log(r))))
                    rows.append({"cell": cell, "seed": seed, "lr": lr, "a": a, "r": r,
                                 "dw_fro": dw, "cos_dw_w0": cos,
                                 "dead": int(rng.random() < 0.08 + 0.85 * p)})
    df = pd.DataFrame(rows)
    df["ldw"] = np.log(df.dw_fro)
    df["lln"] = np.log(df.lr)
    df["aln"] = np.log(df.a)
    df["rln"] = np.log(df.r)
    return df


# ------------------------- json_scalar -------------------------
@pytest.mark.parametrize("value", [
    1.5, -0.0, np.float64(2.25), np.float32(0.5), float("nan"), float("inf"), -math.inf,
    np.float64("nan"), np.float64("inf"), 3, -7, np.int64(11), np.int32(-2),
    "a string", None, True, False,
])
def test_json_scalar_matches_inline_j(value):
    assert json_scalar(value) is _j(value) or json_scalar(value) == _j(value)


def test_json_scalar_nonfinite_is_none_and_types_are_python():
    assert json_scalar(np.float64("nan")) is None
    assert json_scalar(float("inf")) is None
    assert json_scalar(np.float64(-np.inf)) is None
    assert type(json_scalar(np.float64(1.5))) is float
    assert type(json_scalar(np.int64(3))) is int
    # bool is an int subclass: the historical inline behaviour returns 1/0, not True/False
    assert json_scalar(True) == 1 and type(json_scalar(True)) is int


def test_json_scalar_passes_non_numeric_through():
    obj = object()
    assert json_scalar(obj) is obj
    assert json_scalar("x") == "x"
    assert json_scalar(None) is None


# ------------------------- quantile_table -------------------------
def test_quantile_table_matches_inline_qtab(screen_df):
    d = screen_df.rename(columns={"cos_dw_w0": "mean_cos"})
    assert quantile_table(d, "r", "mean_cos") == _qtab(d, "r")
    assert quantile_table(d, ["r", "a"], "mean_cos") == _qtab(d, ["r", "a"])


def test_quantile_table_counts_only_non_nan(screen_df):
    d = screen_df.rename(columns={"cos_dw_w0": "mean_cos"}).copy()
    d.loc[d.index[:5], "mean_cos"] = np.nan
    tab = quantile_table(d, "r", "mean_cos")
    assert sum(v["n"] for v in tab.values()) == int(d.mean_cos.notna().sum())
    assert all(np.isfinite(v["median"]) for v in tab.values())


# ------------------------- block_test / nested_logit_block -------------------------
def test_nested_logit_block_matches_inline_ladder(screen_df):
    m0 = _logit(screen_df, "dead ~ ldw")
    m1 = _logit(screen_df, "dead ~ ldw + lln + aln + rln")
    oracle = _block(m1, m0)

    rung = nested_logit_block(screen_df, "dead ~ ldw", "dead ~ ldw + lln + aln + rln",
                              cluster="cell", terms=HP_TERMS)

    assert rung.reduced.llf == pytest.approx(m0.llf)
    assert rung.full.llf == pytest.approx(m1.llf)
    assert rung.reduced.prsquared == pytest.approx(m0.prsquared)
    assert rung.block.lr_chi2 == pytest.approx(oracle["hp_block_lr_chi2"])
    assert rung.block.lr_p == pytest.approx(oracle["hp_block_lr_p"])
    assert rung.block.lr_df == oracle["hp_block_lr_df"]
    assert rung.block.wald_chi2 == pytest.approx(oracle["hp_block_cluster_wald_chi2"])
    assert rung.block.wald_p == pytest.approx(oracle["hp_block_cluster_wald_p"])


def test_block_test_standalone_matches_inline(screen_df):
    m0 = _logit(screen_df, "dead ~ ldw")
    m1 = _logit(screen_df, "dead ~ ldw + lln + aln + rln")
    b = block_test(m1, m0, HP_TERMS)
    oracle = _block(m1, m0)
    assert (b.lr_chi2, b.lr_df) == pytest.approx((oracle["hp_block_lr_chi2"],
                                                  oracle["hp_block_lr_df"]))
    assert b.wald_p == pytest.approx(oracle["hp_block_cluster_wald_p"])


def test_block_test_scalars_are_json_safe(screen_df):
    """Every field a plain python scalar, straight into an artifact."""
    m0 = _logit(screen_df, "dead ~ ldw")
    m1 = _logit(screen_df, "dead ~ ldw + lln + aln + rln")
    b = block_test(m1, m0, HP_TERMS)
    for v in (b.lr_chi2, b.lr_df, b.lr_p, b.wald_chi2, b.wald_p):
        assert isinstance(v, (float, int)) and not isinstance(v, np.generic)
    assert b.converged is True


def test_block_test_rejects_a_non_nested_pair(screen_df):
    """The same fit twice adds no column; a reduced-only column is missing from the full."""
    m1 = _logit(screen_df, "dead ~ ldw + lln + aln + rln")
    with pytest.raises(ValueError, match="strictly contain"):
        block_test(m1, m1, HP_TERMS)
    other = _logit(screen_df, "dead ~ ldw + aln")
    with pytest.raises(ValueError, match="aln"):
        block_test(_logit(screen_df, "dead ~ ldw + lln"), other, ["lln"])


# ------------------------- (2) convergence is surfaced -------------------------
def test_cluster_logit_exposes_convergence(screen_df):
    from subliminal_learning.stats.core import cluster_logit

    res = cluster_logit(screen_df, "dead ~ ldw + lln", "cell", method="bfgs", maxiter=500)
    assert res.converged is True
    assert res.converged == bool(res.mle_retvals["converged"])
    assert res.n_iterations is None or isinstance(res.n_iterations, int)
    # the existing callers' surface is untouched
    for attr in ("params", "llf", "prsquared", "mle_retvals", "wald_test", "fittedvalues",
                 "nobs", "bse", "pvalues"):
        assert hasattr(res, attr)


def test_non_converged_rung_blanks_its_p_values(screen_df):
    """One iteration of BFGS cannot converge: p-values blanked, converged False, no raise."""
    rung = nested_logit_block(screen_df, "dead ~ ldw", "dead ~ ldw + lln + aln + rln",
                              cluster="cell", terms=HP_TERMS, maxiter=1)
    assert rung.converged is False and rung.block.converged is False
    assert rung.block.lr_p is None and rung.block.wald_p is None


# ------------------------- (4) LR degrees of freedom -------------------------
def test_lr_df_is_unchanged_at_the_current_call_sites(screen_df):
    """3 for the mediation_twopart_kl HP block, 4 for the teacher screen's."""
    rung3 = nested_logit_block(screen_df, "dead ~ ldw", "dead ~ ldw + lln + aln + rln",
                               cluster="cell", terms=HP_TERMS)
    assert rung3.block.lr_df == 3 == len(HP_TERMS)
    d = screen_df.copy()
    d["sln"] = np.log(20 * 3)                      # constant steps -> a 4th HP column
    d.loc[d.index[::2], "sln"] = np.log(10 * 3)
    rung4 = nested_logit_block(d, "dead ~ ldw", "dead ~ ldw + lln + aln + rln + sln",
                               cluster="cell", terms=[*HP_TERMS, "sln"])
    assert rung4.block.lr_df == 4


def test_lr_df_counts_columns_not_term_names(screen_df):
    """A 3-level factor is ONE term name but TWO added columns."""
    d = screen_df.copy()
    d["tree"] = np.take(list("abc"), np.arange(len(d)) % 3)
    rung = nested_logit_block(d, "dead ~ ldw", "dead ~ ldw + C(tree)", cluster="cell",
                              terms=["C(tree)[T.b]", "C(tree)[T.c]"])
    assert rung.block.lr_df == 2
    assert rung.block.lr_p == pytest.approx(sps.chi2.sf(rung.block.lr_chi2, 2))


# ------------------------- (5) both fits on the same rows -------------------------
def test_nested_logit_block_rejects_mismatched_rows(screen_df):
    """A covariate only the full formula uses, NaN in some rows, silently shrinks that fit."""
    d = screen_df.copy()
    d["extra"] = np.arange(len(d), dtype=float)
    d.loc[d.index[:7], "extra"] = np.nan
    with pytest.raises(ValueError, match="same rows"):
        nested_logit_block(d, "dead ~ ldw", "dead ~ ldw + extra", cluster="cell",
                           terms=["extra"])


def test_block_test_rejects_fits_on_different_rows(screen_df):
    """block_test guards the n itself, not only via `nested_logit_block`'s pre-fit check."""
    m0 = _logit(screen_df.iloc[:-5], "dead ~ ldw")
    m1 = _logit(screen_df, "dead ~ ldw + lln + aln + rln")
    with pytest.raises(ValueError, match="different rows"):
        block_test(m1, m0, HP_TERMS)


def test_nested_logit_block_fits_both_on_the_same_rows(screen_df):
    """The two log-likelihoods are only comparable on one subset -- guard the n."""
    rung = nested_logit_block(screen_df, "dead ~ ldw", "dead ~ ldw + lln + aln + rln",
                              cluster="cell", terms=HP_TERMS)
    assert rung.reduced.nobs == rung.full.nobs == len(screen_df)


# ------------------------- coef_table / term_stats -------------------------
def test_coef_table_matches_inline_coefs(screen_df):
    m1 = _logit(screen_df, "dead ~ ldw + lln + aln + rln")
    assert coef_table(m1) == _coefs(m1)
    assert "Intercept" not in coef_table(m1)


def test_term_stats_matches_inline_term(screen_df):
    m1 = _logit(screen_df, "dead ~ ldw + lln + aln + rln")
    got, want = term_stats(m1, "rln"), _term(m1, "rln")
    assert got.keys() == want.keys()
    for k in want:
        assert got[k] == pytest.approx(want[k])


# ------------------------- AUCs -------------------------
def test_fate_auc_matches_inline_auc(screen_df):
    dead = screen_df.dead
    assert fate_auc(screen_df.dw_fro, dead) == pytest.approx(_auc(screen_df.dw_fro, dead))


def test_fate_auc_matches_sklearn_convention(screen_df):
    """AUC is P(dose_dead > dose_alive), ties half -- i.e. roc_auc_score with dead as label 1."""
    d = screen_df
    pos = d.dw_fro[d.dead == 1].to_numpy()
    neg = d.dw_fro[d.dead == 0].to_numpy()
    brute = float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
    assert fate_auc(d.dw_fro, d.dead) == pytest.approx(brute)


def test_fitted_auc_matches_inline_fittedvalues_auc(screen_df):
    m1 = _logit(screen_df, "dead ~ ldw + lln + aln + rln")
    oracle = _auc(pd.Series(np.asarray(m1.fittedvalues), index=screen_df.index), screen_df.dead)
    assert fitted_auc(m1, screen_df.dead, index=screen_df.index) == pytest.approx(oracle)
    assert fitted_auc(m1, screen_df.dead) == pytest.approx(oracle)


# ------------------------- zscore_column -------------------------
def test_zscore_column_matches_inline_zscoring(screen_df):
    d_ref = screen_df.copy()
    mu_ref = float(d_ref.cos_dw_w0.mean())
    sd_ref = float(d_ref.cos_dw_w0.std(ddof=0))
    d_ref["cosz"] = (d_ref.cos_dw_w0 - mu_ref) / sd_ref

    d, mu, sd = zscore_column(screen_df, "cos_dw_w0")
    assert (mu, sd) == pytest.approx((mu_ref, sd_ref))
    pd.testing.assert_series_equal(d.cosz, d_ref.cosz)
    assert "cosz" not in screen_df.columns          # input untouched
    assert d.cosz.mean() == pytest.approx(0.0, abs=1e-12)
    assert d.cosz.std(ddof=0) == pytest.approx(1.0)


def test_zscore_coefficient_rescales_back_to_the_raw_scale(screen_df):
    """coef/sd on the z-scored fit is the raw-scale coefficient the artifacts report.

    Checked on a well-conditioned O(1) column: on the actual O(1e-4) cosine the raw-scale fit is
    exactly the one BFGS cannot do -- which is why the screens z-score in the first place.
    """
    d, _mu, sd = zscore_column(screen_df, "ldw", out="cosz")
    z_fit = _logit(d, "dead ~ lln + cosz")
    raw_fit = _logit(screen_df, "dead ~ lln + ldw")
    assert z_fit.params["cosz"] / sd == pytest.approx(raw_fit.params["ldw"], rel=1e-4)
