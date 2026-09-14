"""Binomial rates, stratum reweighting, and agreement for the judge audit.

Formula sources (do not replace a Wilson interval with a Wald interval, and do
not Wilson-pool weighted counts):

- Wilson score interval: Wilson, E.B. (1927), JASA 22:209-212; recommended
  in Brown, Cai and DasGupta (2001), Statistical Science 16:101-133.
- Difference of independent proportions: Newcombe (1998), Statistics in
  Medicine 17:873-890, method 10 (Wilson-based hybrid).
- Weighted rate interval: Horvitz-Thompson / post-stratified mean of stratum
  rates, then a Wilson interval on the Kish effective sample size
  ``n_eff = 1 / sum_s (W_s^2 * fpc_s / n_s)`` with finite-population
  correction ``fpc_s = (N_s - n_s) / (N_s - 1)``. A naive Wilson on the
  pooled ``(sum y_s, sum n_s)`` is not this estimand.
- Cohen's kappa: Cohen (1960), Educational and Psychological Measurement
  20:37-46. Single-class tables set ``marker`` rather than returning NaN.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

KAPPA_UNDEFINED_SINGLE_CLASS = "undefined_single_class"

_Z_CRITICAL: Mapping[float, float] = MappingProxyType(
    {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.99: 2.5758293035489004,
    }
)
_YES = frozenset({"yes", "true", "1"})
_NO = frozenset({"no", "false", "0"})


@dataclass(frozen=True)
class RateInterval:
    """A rate in ``[0, 1]`` with a two-sided interval.

    Attributes:
        estimate: Point estimate.
        ci_low: Lower bound, clipped to ``[0, 1]``.
        ci_high: Upper bound, clipped to ``[0, 1]``.
        n_effective: Sample size the interval used (Kish ``n_eff`` when
            reweighted; ``n`` for a simple binomial).
        n_population: Population denominator, or ``n`` when unused.
        n_sampled: Raw row count that entered the estimate.
        n_yes: Raw positive count (unweighted).
    """

    estimate: float
    ci_low: float
    ci_high: float
    n_effective: float
    n_population: int
    n_sampled: int
    n_yes: int


@dataclass(frozen=True)
class DifferenceInterval:
    """``estimate_a - estimate_b`` with a Newcombe interval.

    Attributes:
        estimate: Difference of the two point estimates.
        ci_low: Lower bound, clipped to ``[-1, 1]``.
        ci_high: Upper bound, clipped to ``[-1, 1]``.
    """

    estimate: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class KappaResult:
    """Cohen's kappa and per-class agreement (rater A is the reference).

    Attributes:
        kappa: ``(p_o - p_e) / (1 - p_e)``, or ``None`` when undefined.
        marker: ``KAPPA_UNDEFINED_SINGLE_CLASS`` when ``p_e == 1``, else
            ``None``.
        observed_agreement: ``p_o``.
        expected_agreement: ``p_e`` (1.0 when single-class).
        n: Paired ratings.
        per_class_agreement: Class -> fraction of rater-A labels of that
            class that rater B matched.
    """

    kappa: float | None
    marker: str | None
    observed_agreement: float
    expected_agreement: float
    n: int
    per_class_agreement: Mapping[str, float]


def parse_yes_no(value: object) -> bool:
    """Parse a judge yes/no field to ``bool``.

    Args:
        value: ``yes``/``no``, ``true``/``false``, ``1``/``0``, or ``bool``.

    Returns:
        True for yes.

    Raises:
        ValueError: Value is not a recognised yes/no token.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1) and value == value:
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _YES:
            return True
        if token in _NO:
            return False
    raise ValueError(f"unparseable yes/no value: {value!r}")


def wilson_score_interval(
    k: int,
    n: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n``.

    ``n == 0`` returns ``(0.0, 1.0)``. Bounds are clipped to ``[0, 1]``.

    Args:
        k: Number of successes.
        n: Number of trials.
        confidence: Two-sided level; must be 0.90, 0.95, or 0.99.

    Returns:
        ``(lo, hi)``.
    """
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n; got k={k}, n={n}")
    if n == 0:
        return (0.0, 1.0)
    return _wilson_real(k / n, float(n), confidence)


def binomial_rate(
    k: int,
    n: int,
    *,
    confidence: float = 0.95,
) -> RateInterval:
    """Binomial proportion ``k / n`` with a Wilson interval.

    Args:
        k: Number of successes.
        n: Number of trials (must be > 0).
        confidence: Two-sided level.

    Returns:
        The `RateInterval`.
    """
    if n <= 0:
        raise ValueError(f"binomial rate needs n > 0; got n={n}")
    lo, hi = wilson_score_interval(k, n, confidence=confidence)
    return RateInterval(
        estimate=k / n,
        ci_low=lo,
        ci_high=hi,
        n_effective=float(n),
        n_population=n,
        n_sampled=n,
        n_yes=k,
    )


def reweighted_rate(
    strata: Sequence[tuple[int, int, int]],
    *,
    confidence: float = 0.95,
) -> RateInterval:
    """Population rate from per-stratum ``(n_sampled, n_yes, n_population)``.

    ``estimate = sum_s (N_s / N) * (y_s / n_s)``. The interval is Wilson on
    the Kish effective sample size with a per-stratum FPC (see module
    docstring). Empty-population strata are dropped. A stratum with
    ``N_s > 0`` and ``n_s == 0`` raises.

    Args:
        strata: Per-stratum ``(n_sampled, n_yes, n_population)``.
        confidence: Two-sided level.

    Returns:
        The reweighted `RateInterval`.
    """
    cleaned: list[tuple[int, int, int]] = []
    for n_sampled, n_yes, n_population in strata:
        if min(n_sampled, n_yes, n_population) < 0:
            raise ValueError(
                f"stratum counts must be >= 0; got {(n_sampled, n_yes, n_population)}"
            )
        if n_yes > n_sampled:
            raise ValueError(f"n_yes {n_yes} exceeds n_sampled {n_sampled}")
        if n_sampled > n_population:
            raise ValueError(
                f"n_sampled {n_sampled} exceeds n_population {n_population}"
            )
        if n_population == 0:
            if n_sampled:
                raise ValueError("sampled rows from a zero-population stratum")
            continue
        if n_sampled == 0:
            raise ValueError(
                f"population {n_population} has no sample; cannot reweight"
            )
        cleaned.append((n_sampled, n_yes, n_population))
    if not cleaned:
        raise ValueError("no strata with population")

    n_pop = sum(item[2] for item in cleaned)
    estimate = 0.0
    kish_denom = 0.0
    n_sampled_total = 0
    n_yes_total = 0
    for n_sampled, n_yes, n_population in cleaned:
        weight = n_population / n_pop
        estimate += weight * (n_yes / n_sampled)
        if n_population > 1:
            fpc = (n_population - n_sampled) / (n_population - 1)
        else:
            fpc = 0.0
        kish_denom += (weight * weight) * fpc / n_sampled
        n_sampled_total += n_sampled
        n_yes_total += n_yes

    if kish_denom <= 0.0:
        lo = hi = estimate
        n_eff = float(n_pop)
    else:
        n_eff = 1.0 / kish_denom
        lo, hi = _wilson_real(estimate, n_eff, confidence)
    return RateInterval(
        estimate=estimate,
        ci_low=lo,
        ci_high=hi,
        n_effective=n_eff,
        n_population=n_pop,
        n_sampled=n_sampled_total,
        n_yes=n_yes_total,
    )


def proportion_difference(
    k_a: int,
    n_a: int,
    k_b: int,
    n_b: int,
    *,
    confidence: float = 0.95,
) -> DifferenceInterval:
    """Newcombe method-10 interval for ``k_a/n_a - k_b/n_b``.

    The two samples are independent (treatment vs matched-control leaves).

    Args:
        k_a: Successes in sample A.
        n_a: Trials in sample A.
        k_b: Successes in sample B.
        n_b: Trials in sample B.
        confidence: Two-sided level.

    Returns:
        The `DifferenceInterval`.
    """
    rate_a = binomial_rate(k_a, n_a, confidence=confidence)
    rate_b = binomial_rate(k_b, n_b, confidence=confidence)
    return difference_of_rates(rate_a, rate_b)


def difference_of_rates(
    rate_a: RateInterval,
    rate_b: RateInterval,
) -> DifferenceInterval:
    """Newcombe method-10 interval for ``rate_a.estimate - rate_b.estimate``.

    Uses the Wilson (or ESS-Wilson) bounds already on each `RateInterval`.

    Args:
        rate_a: First rate (treatment).
        rate_b: Second rate (control).

    Returns:
        The `DifferenceInterval`.
    """
    delta = rate_a.estimate - rate_b.estimate
    lo = delta - math.sqrt(
        (rate_a.estimate - rate_a.ci_low) ** 2 + (rate_b.ci_high - rate_b.estimate) ** 2
    )
    hi = delta + math.sqrt(
        (rate_a.ci_high - rate_a.estimate) ** 2 + (rate_b.estimate - rate_b.ci_low) ** 2
    )
    return DifferenceInterval(
        estimate=delta,
        ci_low=max(-1.0, lo),
        ci_high=min(1.0, hi),
    )


def per_form_overrejection(
    matched_form: Sequence[str | None],
    judge_trait_free: Sequence[bool],
    *,
    confidence: float = 0.95,
) -> dict[str, RateInterval]:
    """Per-``matched_form`` fraction the judge calls trait-free.

    Rows with a null/empty form are skipped. A form with zero judged rows
    is omitted. An empty input returns ``{}``.

    Args:
        matched_form: Per-row regex form (or ``None``).
        judge_trait_free: Per-row True when the judge says no trait.
        confidence: Two-sided level.

    Returns:
        Form -> over-rejection `RateInterval`.
    """
    if len(matched_form) != len(judge_trait_free):
        raise ValueError(
            f"matched_form ({len(matched_form)}) and judge_trait_free "
            f"({len(judge_trait_free)}) length mismatch"
        )
    counts: dict[str, list[int]] = {}
    for form, free in zip(matched_form, judge_trait_free, strict=True):
        if form is None:
            continue
        key = str(form).strip()
        if not key:
            continue
        pair = counts.setdefault(key, [0, 0])
        pair[0] += 1
        pair[1] += int(bool(free))
    return {
        form: binomial_rate(n_free, n, confidence=confidence)
        for form, (n, n_free) in counts.items()
        if n
    }


def cohens_kappa(
    rater_a: Sequence[object],
    rater_b: Sequence[object],
) -> KappaResult:
    """Cohen's kappa with rater A as the per-class reference.

    Args:
        rater_a: Reference labels (human).
        rater_b: Comparison labels (judge).

    Returns:
        The `KappaResult`. ``marker`` is set when every label is one class.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(f"rater length mismatch: {len(rater_a)} vs {len(rater_b)}")
    n = len(rater_a)
    if n == 0:
        raise ValueError("kappa needs at least one paired rating")

    n_agree = 0
    n_a: dict[object, int] = {}
    n_b: dict[object, int] = {}
    n_both: dict[object, int] = {}
    for a, b in zip(rater_a, rater_b, strict=True):
        n_a[a] = n_a.get(a, 0) + 1
        n_b[b] = n_b.get(b, 0) + 1
        if a == b:
            n_agree += 1
            n_both[a] = n_both.get(a, 0) + 1
    p_o = n_agree / n
    labels = set(n_a) | set(n_b)
    p_e = sum((n_a.get(c, 0) / n) * (n_b.get(c, 0) / n) for c in labels)
    per_class = {str(c): (n_both.get(c, 0) / n_a[c]) for c in n_a if n_a[c]}
    frozen = MappingProxyType(per_class)
    if p_e >= 1.0 - 1e-15:
        return KappaResult(
            kappa=None,
            marker=KAPPA_UNDEFINED_SINGLE_CLASS,
            observed_agreement=p_o,
            expected_agreement=1.0,
            n=n,
            per_class_agreement=frozen,
        )
    return KappaResult(
        kappa=(p_o - p_e) / (1.0 - p_e),
        marker=None,
        observed_agreement=p_o,
        expected_agreement=p_e,
        n=n,
        per_class_agreement=frozen,
    )


def _z_critical(confidence: float) -> float:
    try:
        return _Z_CRITICAL[confidence]
    except KeyError:
        known = ", ".join(str(c) for c in _Z_CRITICAL)
        raise ValueError(
            f"confidence must be one of {known}; got {confidence}"
        ) from None


def _wilson_real(
    phat: float,
    n: float,
    confidence: float,
) -> tuple[float, float]:
    if n <= 0.0:
        return (0.0, 1.0)
    z = _z_critical(confidence)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (phat + z2 / (2.0 * n)) / denom
    inner = phat * (1.0 - phat) / n + z2 / (4.0 * n * n)
    margin = z * math.sqrt(max(inner, 0.0)) / denom
    lo = center - margin
    hi = center + margin
    if lo != lo or hi != hi:
        raise ValueError("Wilson interval produced NaN")
    return (max(0.0, lo), min(1.0, hi))
