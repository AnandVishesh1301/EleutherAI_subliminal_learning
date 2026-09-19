"""Estimators for the diagonal-vs-direction question: does transfer flow into the model whose
own post-cliff checkpoint-distill diagonal is higher?

Two families of test, both pure functions over numpy arrays (no disk, no matplotlib), so this
stays a leaf of the ``scripts/`` -> ``pipelines/`` -> ``src/subliminal_learning/`` layering and
`analysis.diagonal_direction` can call it without a cycle:

- **Sign agreement.** Per unordered pair, does ``sign(T(x->y) - T(y->x))`` equal
  ``sign(diag(y) - diag(x))``? Under the null that the direction carries no diagonal
  information the agreements are Binomial(n, 1/2), so `sign_agreement` reports the EXACT
  two-sided binomial p. Pairs with a zero on either side are ties and leave the design.
- **Model-label permutation.** Every Spearman here correlates a per-arm (or per-pair) statistic
  with a function of the models' diagonals. The arms are NOT independent -- each model appears
  in many of them -- so the exchangeable unit is the MODEL LABEL, not the arm: the null permutes
  the diagonal values across models and recomputes the same contrast, which preserves the arm
  set's dependence structure exactly. `spearman_role` (diagonal of one endpoint) and
  `spearman_pair_contrast` (difference of the two endpoints' diagonals) are the two shapes;
  both route through `_permutation_p` so the null is defined once.

`ols_residuals` and `partial_spearman` are the confounding tools. The control damage a receiver
takes is not a clean receiver property: it is measured on arms that differ in WHO sent the pool,
in how far apart the two base pools are and in the receiver's own prior, so `ols_residuals`
regresses it on those and hands back the part that is left, and `partial_spearman` reports a rank
correlation with other covariates partialled out (Spearman of the residuals of two rank OLS fits,
with the ordinary partial-correlation t-test on ``n - 2 - k`` degrees of freedom).

`direction_classifier` is the fitted answer to "which direction of a pair is stronger": a logistic
regression on ANTISYMMETRIC pair features ``f(x, y) = value(y) - value(x)``. Antisymmetry is
imposed rather than hoped for -- the data is augmented with every pair flipped (features negated,
label flipped) and the fit carries no intercept, so ``P(f) = 1 - P(-f)`` exactly and the majority
-class baseline is 0.5 by construction. It is evaluated leave-one-PAIR-out (both orientations of
the held-out pair leave together, so the flipped copy cannot leak), and its null is a permutation
of the labels ACROSS pairs.

`rank_ols` is the multivariate companion: the arm statistic, the sender diagonal and the
receiver diagonal all ranked, with the relationship stratum as dummies. Its p-values are the
ordinary OLS ones and inherit the same non-independence caveat -- read the COEFFICIENTS, and
take the permutation p's above as the inference.

`two_way_decomposition` is the descriptive companion to all of it: instead of asking whether the
arm statistic tracks a particular model property, it asks how much of the variance in T is a
SENDER effect at all and how much a RECEIVER effect, by fitting ``T ~ C(sender, Sum) +
C(receiver, Sum)`` (sum-to-zero contrasts, so every model gets an effect that is a deviation from
the grand mean and the two factors' effects each sum to zero) and reading the type-II ANOVA sums
of squares. Its per-model effects are the quantities the covariate regressions above are run on.

All randomness flows through ``np.random.default_rng(seed)`` with an explicit ``seed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import stats

__all__ = [
    "SignAgreement",
    "PermutedSpearman",
    "TwoWayDecomposition",
    "sign_agreement",
    "spearman_role",
    "spearman_pair_contrast",
    "rank_ols",
    "rank_ols_multi",
    "two_way_decomposition",
    "PartialSpearman",
    "DirectionFit",
    "ols_residuals",
    "partial_spearman",
    "direction_classifier",
    "PermutedSlope",
    "cluster_ols",
    "within_group_slope_permutation",
]


@dataclass(frozen=True)
class SignAgreement:
    """Outcome of `sign_agreement`.

    Attributes:
        n: Pairs entering the test (ties on either side excluded).
        n_agree: Pairs whose asymmetry sign matches the diagonal-difference sign.
        p: Exact two-sided binomial p against ``p = 1/2``.
        n_tied: Pairs dropped because one side was exactly zero.
    """

    n: int
    n_agree: int
    p: float
    n_tied: int = 0


@dataclass(frozen=True)
class PermutedSpearman:
    """Outcome of a model-label permutation Spearman test.

    Attributes:
        rho: Observed Spearman correlation.
        p: Two-sided permutation p, ``(1 + #{|rho*| >= |rho|}) / (1 + n_perm)`` (add-one, so it
            is never 0 and is a valid Monte-Carlo p).
        n: Observations (arms or pairs) the correlation was taken over.
        n_perm: Permutations drawn.
    """

    rho: float
    p: float
    n: int
    n_perm: int


def sign_agreement(asym, ddiag) -> SignAgreement:
    """Do the asymmetry and the diagonal difference point the same way?

    Args:
        asym: Per pair, ``T(x->y) - T(y->x)``.
        ddiag: Per pair, ``diag(y) - diag(x)`` (same order as `asym`).

    Returns:
        A `SignAgreement`; ``p`` is NaN when no untied pair survives.

    Raises:
        ValueError: If the two sequences differ in length.
    """
    a, d = np.asarray(asym, dtype=float), np.asarray(ddiag, dtype=float)
    if a.shape != d.shape:
        raise ValueError(f"asym and ddiag must match in shape, got {a.shape} and {d.shape}")
    ok = np.isfinite(a) & np.isfinite(d) & (a != 0) & (d != 0)
    n = int(ok.sum())
    n_tied = int(a.size - n)
    if n == 0:
        return SignAgreement(0, 0, float("nan"), n_tied)
    k = int((np.sign(a[ok]) == np.sign(d[ok])).sum())
    return SignAgreement(n, k, float(stats.binomtest(k, n, 0.5).pvalue), n_tied)


def _permutation_p(values, diag, contrast: Callable[[np.ndarray], np.ndarray], *,
                   n_perm: int, seed: int) -> PermutedSpearman:
    """Spearman of `values` against ``contrast(diag)``, with `diag` permuted across models.

    `contrast` maps a per-model diagonal vector to one number per observation (an endpoint's
    diagonal, a difference of endpoints, ...); permuting `diag` before applying it is exactly
    the model-label null.
    """
    v = np.asarray(values, dtype=float)
    diag = np.asarray(diag, dtype=float)
    observed = contrast(diag)
    ok = np.isfinite(v) & np.isfinite(observed)
    n = int(ok.sum())
    if n < 3:
        return PermutedSpearman(float("nan"), float("nan"), n, n_perm)
    rho = float(stats.spearmanr(v[ok], observed[ok]).statistic)
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n_perm):
        c = contrast(diag[rng.permutation(diag.size)])
        m = np.isfinite(v) & np.isfinite(c)
        if int(m.sum()) < 3:
            continue
        r = stats.spearmanr(v[m], c[m]).statistic
        if np.isfinite(r) and abs(r) >= abs(rho):
            exceed += 1
    return PermutedSpearman(rho, (1 + exceed) / (1 + n_perm), n, n_perm)


def spearman_role(values, endpoint, diag, *, n_perm: int = 2000, seed: int = 0) -> PermutedSpearman:
    """Spearman of an arm statistic with ONE endpoint's diagonal (sender or receiver).

    Args:
        values: Per-arm statistic.
        endpoint: Per-arm index into `diag` of the endpoint whose diagonal is correlated.
        diag: Per-model diagonal, indexed by the model order `endpoint` refers to.
        n_perm: Permutations of the model labels.
        seed: RNG seed.

    Returns:
        A `PermutedSpearman`.
    """
    idx = np.asarray(endpoint, dtype=int)
    return _permutation_p(values, diag, lambda d: d[idx], n_perm=n_perm, seed=seed)


def spearman_pair_contrast(asym, index_x, index_y, diag, *, n_perm: int = 2000,
                           seed: int = 0) -> PermutedSpearman:
    """Spearman of a pair asymmetry with ``diag(y) - diag(x)``, model-label permutation null.

    Args:
        asym: Per pair, ``T(x->y) - T(y->x)``.
        index_x: Per pair, index of ``x`` into `diag`.
        index_y: Per pair, index of ``y`` into `diag`.
        diag: Per-model diagonal.
        n_perm: Permutations of the model labels.
        seed: RNG seed.

    Returns:
        A `PermutedSpearman`.
    """
    ix, iy = np.asarray(index_x, dtype=int), np.asarray(index_y, dtype=int)
    return _permutation_p(asym, diag, lambda d: d[iy] - d[ix], n_perm=n_perm, seed=seed)


def rank_ols(values, diag_sender, diag_receiver, strata):
    """Rank OLS ``rank(T) ~ rank(diag_sender) + rank(diag_receiver) + C(stratum)``.

    Ranks make the fit robust to the diagonals' and the effects' very different scales without
    committing to a link; the stratum dummies absorb the relationship-level offsets so the two
    diagonal slopes are read WITHIN a relationship.

    Args:
        values: Per-arm statistic.
        diag_sender: Per-arm sender diagonal.
        diag_receiver: Per-arm receiver diagonal.
        strata: Per-arm relationship label.

    Returns:
        ``{term: {"coef", "se", "t", "p"}}`` plus ``{"_n": n, "_r2": r2}``; empty when fewer
        than four complete arms survive.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    df = pd.DataFrame({"T": np.asarray(values, dtype=float),
                       "ds": np.asarray(diag_sender, dtype=float),
                       "dr": np.asarray(diag_receiver, dtype=float),
                       "stratum": list(strata)}).dropna()
    if len(df) < 4:
        return {}
    for c in ("T", "ds", "dr"):
        df[c] = df[c].rank()
    formula = "T ~ ds + dr" + (" + C(stratum)" if df["stratum"].nunique() > 1 else "")
    fit = smf.ols(formula, data=df).fit()
    out = {t: {"coef": float(fit.params[t]), "se": float(fit.bse[t]),
               "t": float(fit.tvalues[t]), "p": float(fit.pvalues[t])} for t in fit.params.index}
    return out | {"_n": int(len(df)), "_r2": float(fit.rsquared)}


@dataclass(frozen=True)
class TwoWayDecomposition:
    """Outcome of `two_way_decomposition`.

    Attributes:
        effects: One row per (role, model): ``role, model, effect, se, t, p``. The effects of a
            role sum to zero by construction (sum-to-zero contrasts), so ``effect`` reads as
            "this model's deviation from the grand mean when it plays this role".
        anova: Type-II ANOVA: ``term, df, ss, ss_share, F, p`` with one ``Residual`` row;
            ``ss_share`` is the term's share of the TOTAL sum of squares.
        r2: Fraction of variance the whole fit explains.
        n: Observations (arms) entering the fit.
    """

    effects: "object"
    anova: "object"
    r2: float
    n: int


def two_way_decomposition(frame, value_col: str, sender_col: str = "sender",
                          receiver_col: str = "receiver",
                          extra_factor: str | None = None) -> TwoWayDecomposition:
    """Split an arm statistic into per-model SENDER and RECEIVER effects.

    ``value ~ C(sender, Sum) + C(receiver, Sum)`` (+ ``C(extra, Sum)``) by OLS. Sum-to-zero
    contrasts are the point: with treatment coding every effect would be read against whichever
    model happened to sort first, which is an arbitrary reference for a question about how models
    differ from each other. The level statsmodels drops (the last) is recovered as minus the sum
    of the others, with its SE from the corresponding linear contrast, so the returned table has
    EVERY model in it and not n-1.

    The type-II ANOVA is the answer to "is the variance carried by the sender or the receiver":
    each term's sum of squares is computed with the other factors already in the model, so an
    unbalanced design (arms missing) cannot hand one factor the other's variance.

    Args:
        frame: Arm-level frame; rows with a non-finite `value_col` are dropped.
        value_col: The response column.
        sender_col: Sender label column.
        receiver_col: Receiver label column.
        extra_factor: Optional third categorical (e.g. the relationship stratum).

    Returns:
        A `TwoWayDecomposition`.

    Raises:
        ValueError: If fewer rows survive than the fit has parameters.
    """
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    cols = [sender_col, receiver_col] + ([extra_factor] if extra_factor else [])
    df = frame[[value_col, *cols]].rename(columns={value_col: "y"}).copy()
    df = df[np.isfinite(pd.to_numeric(df.y, errors="coerce"))].reset_index(drop=True)
    terms = {"sender": sender_col, "receiver": receiver_col}
    if extra_factor:
        terms["extra"] = extra_factor
    # A factor at a single level carries no contrast; keeping it would make the design singular.
    terms = {role: col for role, col in terms.items() if df[col].nunique() > 1}
    rhs = " + ".join(f"C({col}, Sum)" for col in terms.values()) or "1"
    n_params = 1 + sum(df[col].nunique() - 1 for col in terms.values())
    if len(df) <= n_params:
        raise ValueError(f"{len(df)} rows cannot identify a {n_params}-parameter fit")
    fit = smf.ols(f"y ~ {rhs}", data=df).fit()

    rows = []
    for role, col in terms.items():
        levels = sorted(df[col].unique())
        names = [f"C({col}, Sum)[S.{v}]" for v in levels[:-1]]
        for level, name in zip(levels[:-1], names):
            rows.append({"role": role, "model": level, "effect": float(fit.params[name]),
                         "se": float(fit.bse[name]), "t": float(fit.tvalues[name]),
                         "p": float(fit.pvalues[name])})
        # The dropped level: minus the sum of the others, as one linear contrast.
        contrast = np.zeros(len(fit.params))
        for name in names:
            contrast[list(fit.params.index).index(name)] = -1.0
        t = fit.t_test(contrast)
        rows.append({"role": role, "model": levels[-1], "effect": float(np.squeeze(t.effect)),
                     "se": float(np.squeeze(t.sd)), "t": float(np.squeeze(t.tvalue)),
                     "p": float(np.squeeze(t.pvalue))})
    effects = pd.DataFrame(rows)

    anova = sm.stats.anova_lm(fit, typ=2).reset_index().rename(
        columns={"index": "term", "sum_sq": "ss", "PR(>F)": "p"})
    inverse = {f"C({col}, Sum)": role for role, col in terms.items()}
    anova["term"] = anova.term.map(lambda t: inverse.get(t, t))
    anova["ss_share"] = anova.ss / anova.ss.sum()
    anova = anova[["term", "df", "ss", "ss_share", "F", "p"]]
    return TwoWayDecomposition(effects, anova, float(fit.rsquared), int(len(df)))


def rank_ols_multi(frame, response: str, regressors, stratum: str | None = None):
    """`rank_ols` generalized: rank OLS of `response` on several ranked numeric `regressors`.

    Same reading and the same caveat: the arms share endpoints, so the OLS p-values are not
    exchangeability-based inference -- read the coefficients and take the model-label permutation
    p's of `spearman_role` as the test.

    Args:
        frame: Arm-level frame holding all the columns named.
        response: Response column.
        regressors: Numeric covariate columns.
        stratum: Optional categorical absorbed as dummies (skipped at a single level).

    Returns:
        ``{term: {"coef", "se", "t", "p"}}`` plus ``{"_n", "_r2"}``; empty when fewer complete
        rows survive than the fit has parameters.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    regressors = list(regressors)
    cols = [response, *regressors] + ([stratum] if stratum else [])
    df = pd.DataFrame(frame[cols]).dropna()
    for c in (response, *regressors):
        df[c] = df[c].rank()
    use_stratum = bool(stratum) and df[stratum].nunique() > 1
    n_params = 1 + len(regressors) + (df[stratum].nunique() - 1 if use_stratum else 0)
    if len(df) <= n_params:
        return {}
    formula = f"Q('{response}') ~ " + " + ".join(f"Q('{c}')" for c in regressors)
    if use_stratum:
        formula += f" + C(Q('{stratum}'))"
    fit = smf.ols(formula, data=df).fit()
    rename = {f"Q('{c}')": c for c in regressors}
    out = {rename.get(t, t): {"coef": float(fit.params[t]), "se": float(fit.bse[t]),
                              "t": float(fit.tvalues[t]), "p": float(fit.pvalues[t])}
           for t in fit.params.index}
    return out | {"_n": int(len(df)), "_r2": float(fit.rsquared)}


@dataclass(frozen=True)
class PartialSpearman:
    """Outcome of `partial_spearman`.

    Attributes:
        rho: Spearman of the two variables with the controls partialled out (the correlation of
            the residuals of ``rank(x) ~ rank(controls)`` and ``rank(y) ~ rank(controls)``).
        p: Two-sided t-test p on ``n - 2 - k`` degrees of freedom, `k` controls.
        n: Complete observations the partial was taken over.
        k: Controls partialled out.
    """

    rho: float
    p: float
    n: int
    k: int


def ols_residuals(frame, response: str, numeric, categorical=None):
    """Residuals of ``response ~ numeric [+ C(categorical)]``, aligned to `frame`'s index.

    A row missing any of the columns cannot enter the fit, and it comes back as NaN rather than
    silently dropped -- the callers average these residuals per model, and a dropped row would
    quietly change which arms a model's average is over.

    Args:
        frame: Any frame carrying the columns named.
        response: Response column.
        numeric: Numeric regressor columns.
        categorical: Optional categorical columns absorbed as dummies (skipped at one level).

    Returns:
        A float Series on `frame`'s index; all-NaN when fewer complete rows survive than the fit
        has parameters.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    numeric, categorical = list(numeric), list(categorical or [])
    df = pd.DataFrame(frame[[response, *numeric, *categorical]]).dropna()
    cats = [c for c in categorical if df[c].nunique() > 1]
    n_params = 1 + len(numeric) + sum(df[c].nunique() - 1 for c in cats)
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    if len(df) <= n_params:
        return out
    terms = [f"Q('{c}')" for c in numeric] + [f"C(Q('{c}'))" for c in cats]
    fit = smf.ols(f"Q('{response}') ~ " + " + ".join(terms or ["1"]), data=df).fit()
    out.loc[df.index] = fit.resid.astype(float)
    return out


def partial_spearman(x, y, controls) -> PartialSpearman:
    """Rank correlation of `x` and `y` with `controls` partialled out.

    Everything is ranked first and the partialling is a linear OLS on those ranks, which is the
    usual rank-based partial correlation: it asks whether the monotone association survives the
    controls without committing to a functional form for them.

    Args:
        x, y: The two variables.
        controls: Sequence of control variables (each the same length), possibly empty.

    Returns:
        A `PartialSpearman`; ``rho``/``p`` are NaN when fewer than ``k + 3`` complete rows exist.
    """
    import pandas as pd

    cols = {"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}
    controls = [np.asarray(c, dtype=float) for c in controls]
    for i, c in enumerate(controls):
        cols[f"c{i}"] = c
    df = pd.DataFrame(cols).dropna()
    k, n = len(controls), len(df)
    if n < k + 3:
        return PartialSpearman(float("nan"), float("nan"), n, k)
    names = [f"c{i}" for i in range(k)]
    rx = ols_residuals(df.rank(), "x", names) if k else df.x.rank() - df.x.rank().mean()
    ry = ols_residuals(df.rank(), "y", names) if k else df.y.rank() - df.y.rank().mean()
    rho = float(stats.spearmanr(rx, ry).statistic) if min(rx.std(), ry.std()) > 0 else float("nan")
    if not np.isfinite(rho) or n - 2 - k <= 0 or abs(rho) >= 1:
        return PartialSpearman(rho, float("nan"), n, k)
    t = rho * np.sqrt((n - 2 - k) / (1 - rho ** 2))
    return PartialSpearman(rho, float(2 * stats.t.sf(abs(t), n - 2 - k)), n, k)


@dataclass(frozen=True)
class DirectionFit:
    """Outcome of `direction_classifier`.

    Attributes:
        n: Unordered pairs (the leave-one-out unit).
        accuracy: Leave-one-pair-out accuracy; an exact 0.5 prediction scores 0.5.
        log_loss: Mean leave-one-pair-out log loss (probabilities clipped at 1e-12).
        perm_p: ``(1 + #{acc* >= acc}) / (1 + n_perm)`` under permutation of the labels across
            pairs; NaN when ``n_perm == 0``.
        n_perm: Label permutations drawn.
        coef: Full-data coefficients on standardized features, ``{name: coef}``.
        proba: Per pair, the held-out ``P(label = 1)``, in input order.
    """

    n: int
    accuracy: float
    log_loss: float
    perm_p: float
    n_perm: int
    coef: dict
    proba: "object"


def _augment(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The pair set plus every pair flipped: features negated, label flipped."""
    return np.vstack([features, -features]), np.concatenate([labels, 1.0 - labels])


def _fit_direction(x: np.ndarray, y: np.ndarray, c: float):
    """Intercept-free L2 logistic regression on features scaled (not centred) by their SD.

    Centring would break the antisymmetry: the augmented design already has mean exactly zero,
    so scaling by the SD is the whole standardization there is to do.
    """
    from sklearn.linear_model import LogisticRegression

    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    return LogisticRegression(fit_intercept=False, C=c).fit(x / scale, y), scale


def _loo_proba(features: np.ndarray, labels: np.ndarray, c: float) -> np.ndarray:
    """Leave-one-PAIR-out held-out ``P(label = 1)``, the flipped copy held out with its pair."""
    n = len(labels)
    x_aug, y_aug = _augment(features, labels)
    proba = np.empty(n)
    for i in range(n):
        keep = np.ones(2 * n, dtype=bool)
        keep[i] = keep[i + n] = False
        fit, scale = _fit_direction(x_aug[keep], y_aug[keep], c)
        proba[i] = float(fit.predict_proba(features[i:i + 1] / scale)[0, 1])
    return proba


def _accuracy(proba: np.ndarray, labels: np.ndarray) -> float:
    """Accuracy with an exact 0.5 scoring 0.5 (an all-zero feature row is not a wrong answer)."""
    hit = np.where(proba == 0.5, 0.5, ((proba > 0.5).astype(float) == labels).astype(float))
    return float(hit.mean())


def direction_classifier(features, labels, *, names=None, c: float = 1.0, n_perm: int = 2000,
                         seed: int = 0) -> DirectionFit:
    """Predict which direction of a pair is stronger, from antisymmetric pair features.

    See the module docstring: the fit is intercept-free on the pair set augmented with every pair
    flipped, so it is exactly antisymmetric (``P(f) = 1 - P(-f)``) and the majority-class baseline
    is 0.5 BY CONSTRUCTION -- there is nothing to beat by predicting a constant. Evaluation is
    leave-one-pair-out with both orientations of the held-out pair removed together, and the null
    permutes the labels across pairs (the features, and therefore the pairs' dependence through
    shared endpoints, are untouched).

    Args:
        features: ``(n_pairs, n_features)`` antisymmetric features, rows with any NaN dropped.
        labels: Per pair, 1 if the ``x -> y`` direction is the stronger one.
        names: Feature names for `DirectionFit.coef`; positional by default.
        c: Inverse L2 penalty of the logistic regression.
        n_perm: Label permutations; 0 skips the test.
        seed: RNG seed.

    Returns:
        A `DirectionFit`.

    Raises:
        ValueError: If `features` and `labels` disagree in length.
    """
    x = np.atleast_2d(np.asarray(features, dtype=float))
    y = np.asarray(labels, dtype=float)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"{x.shape[0]} feature rows but {y.shape[0]} labels")
    names = list(names) if names is not None else [f"f{i}" for i in range(x.shape[1])]
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(y)
    if n < 4 or len(np.unique(y)) < 2:
        return DirectionFit(n, float("nan"), float("nan"), float("nan"), n_perm, {}, np.array([]))
    proba = _loo_proba(x, y, c)
    acc = _accuracy(proba, y)
    p_clip = np.clip(proba, 1e-12, 1 - 1e-12)
    loss = float(-np.mean(y * np.log(p_clip) + (1 - y) * np.log(1 - p_clip)))
    fit, scale = _fit_direction(*_augment(x, y), c)
    coef = {name: float(v) for name, v in zip(names, np.ravel(fit.coef_))}
    perm_p = float("nan")
    if n_perm:
        rng = np.random.default_rng(seed)
        exceed = 0
        for _ in range(n_perm):
            y_perm = y[rng.permutation(n)]
            if _accuracy(_loo_proba(x, y_perm, c), y_perm) >= acc:
                exceed += 1
        perm_p = (1 + exceed) / (1 + n_perm)
    return DirectionFit(n, acc, loss, perm_p, n_perm, coef, proba)


@dataclass(frozen=True)
class PermutedSlope:
    """Outcome of `within_group_slope_permutation`.

    Attributes:
        slope: Observed OLS slope of the response on the regressor, group fixed effects absorbed.
        p: Two-sided permutation p, ``(1 + #{|slope*| >= |slope|}) / (1 + n_perm)``.
        n: Rows entering the fit.
        n_groups: Groups (the within-group permutation unit).
        n_perm: Permutations drawn.
    """

    slope: float
    p: float
    n: int
    n_groups: int
    n_perm: int


def cluster_ols(frame, response: str, regressors, *, fixed_effect: str | None = None,
                cluster: str | None = None):
    """OLS of `response` on `regressors` with optional fixed effects and cluster-robust SEs.

    The teacher-level rows are many per arm (ten recipes) and per pair, so the residuals are
    correlated within those units; clustering on them is the whole point of this fit, and the
    fixed effect is what makes the slope a WITHIN-unit comparison (every pair- or arm-level
    confound cancels).

    Args:
        frame: Row-level frame holding every column named.
        response: Response column.
        regressors: Numeric regressor columns.
        fixed_effect: Categorical absorbed as dummies (skipped at one level).
        cluster: Grouping column for the cluster-robust covariance; ordinary SEs without it.

    Returns:
        ``{term: {"coef", "se", "t", "p"}}`` plus ``{"_n", "_r2", "_n_clusters"}``; empty when
        fewer complete rows survive than the fit has parameters.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    regressors = list(regressors)
    cols = [response, *regressors] + [c for c in (fixed_effect, cluster) if c]
    df = pd.DataFrame(frame[list(dict.fromkeys(cols))]).dropna()   # FE and cluster may coincide
    use_fe = bool(fixed_effect) and df[fixed_effect].nunique() > 1
    n_params = 1 + len(regressors) + (df[fixed_effect].nunique() - 1 if use_fe else 0)
    if len(df) <= n_params:
        return {}
    formula = f"Q('{response}') ~ " + " + ".join(f"Q('{c}')" for c in regressors)
    if use_fe:
        formula += f" + C(Q('{fixed_effect}'))"
    fit = smf.ols(formula, data=df).fit()
    if cluster and df[cluster].nunique() > 1:
        fit = fit.get_robustcov_results(cov_type="cluster", groups=df[cluster].factorize()[0])
    # `get_robustcov_results` returns positional arrays, the plain fit named Series: index both
    # positionally off the design's exog names so the two paths read identically.
    params, bse, tvalues, pvalues = (np.asarray(v) for v in
                                     (fit.params, fit.bse, fit.tvalues, fit.pvalues))
    names = list(fit.model.exog_names)
    rename = {f"Q('{c}')": c for c in regressors}
    out = {rename[f"Q('{c}')"]: {"coef": float(params[names.index(f"Q('{c}')")]),
                                 "se": float(bse[names.index(f"Q('{c}')")]),
                                 "t": float(tvalues[names.index(f"Q('{c}')")]),
                                 "p": float(pvalues[names.index(f"Q('{c}')")])}
           for c in regressors if f"Q('{c}')" in names}
    return out | {"_n": int(len(df)), "_r2": float(fit.rsquared),
                  "_n_clusters": int(df[cluster].nunique()) if cluster else 0}


def within_group_slope_permutation(frame, response: str, regressor: str, group: str, *,
                                   n_perm: int = 2000, seed: int = 0) -> PermutedSlope:
    """Slope of `response` on `regressor` with `group` fixed effects, permuted WITHIN group.

    The null shuffles the regressor among the rows of the same group -- for the teacher-level
    tables, the recipe labels are permuted within a pair. That keeps the pair structure (which
    pairs exist, how many recipes each carries, every pair-level level) exactly and asks only
    whether the recipe-to-recipe ordering carries the response.

    Args:
        frame: Row-level frame.
        response: Response column.
        regressor: The regressor whose slope is tested.
        group: The fixed-effect / permutation-unit column.
        n_perm: Within-group permutations.
        seed: RNG seed.

    Returns:
        A `PermutedSlope`; NaN when the fit is not identified.
    """
    import pandas as pd

    df = pd.DataFrame(frame[[response, regressor, group]]).dropna().reset_index(drop=True)
    fit = cluster_ols(df, response, [regressor], fixed_effect=group)
    if not fit:
        return PermutedSlope(float("nan"), float("nan"), len(df), df[group].nunique(), n_perm)
    slope = fit[regressor]["coef"]
    rng = np.random.default_rng(seed)
    blocks = [idx.to_numpy() for _, idx in df.groupby(group).groups.items()]
    exceed = 0
    for _ in range(n_perm):
        shuffled = df[regressor].to_numpy().copy()
        for idx in blocks:
            shuffled[idx] = shuffled[rng.permutation(idx)]
        perm = df.assign(**{regressor: shuffled})
        out = cluster_ols(perm, response, [regressor], fixed_effect=group)
        if out and abs(out[regressor]["coef"]) >= abs(slope):
            exceed += 1
    return PermutedSlope(slope, (1 + exceed) / (1 + n_perm), int(len(df)),
                         int(df[group].nunique()), n_perm)
