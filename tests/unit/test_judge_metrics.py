from __future__ import annotations

import math

import pytest

from subliminal_learning.gkd.judge_metrics import (
    KAPPA_UNDEFINED_SINGLE_CLASS,
    binomial_rate,
    cohens_kappa,
    difference_of_rates,
    parse_yes_no,
    per_form_overrejection,
    proportion_difference,
    reweighted_rate,
    wilson_score_interval,
)

# Newcombe (1998) Table I, Wilson method 3, 81/263 at 95%.
_NEWCOMBE_81_263 = (0.2553, 0.3662)
# Wilson 0/n upper bound z^2 / (n + z^2) with z = Φ^{-1}(0.975).
_Z95 = 1.959963984540054
_Z95_SQ = _Z95 * _Z95


@pytest.mark.parametrize(
    ("k", "n", "lo", "hi", "atol"),
    [
        (81, 263, _NEWCOMBE_81_263[0], _NEWCOMBE_81_263[1], 5e-4),
        (0, 20, 0.0, _Z95_SQ / (20.0 + _Z95_SQ), 1e-9),
        (20, 20, 20.0 / (20.0 + _Z95_SQ), 1.0, 1e-9),
        (0, 0, 0.0, 1.0, 0.0),
        (0, 1, 0.0, _Z95_SQ / (1.0 + _Z95_SQ), 1e-9),
        (1, 1, 1.0 / (1.0 + _Z95_SQ), 1.0, 1e-9),
    ],
)
def test_wilson_score_interval_published_and_edges(
    k: int, n: int, lo: float, hi: float, atol: float
) -> None:
    got_lo, got_hi = wilson_score_interval(k, n)
    assert 0.0 <= got_lo <= got_hi <= 1.0
    assert math.isfinite(got_lo) and math.isfinite(got_hi)
    assert got_lo == pytest.approx(lo, abs=atol)
    assert got_hi == pytest.approx(hi, abs=atol)


@pytest.mark.parametrize("bad", [(-1, 10), (11, 10), (0, -1)])
def test_wilson_score_interval_rejects_invalid_counts(bad: tuple[int, int]) -> None:
    k, n = bad
    with pytest.raises(ValueError, match="need 0 <= k <= n"):
        wilson_score_interval(k, n)


def test_binomial_rate_rejects_empty_denominator() -> None:
    with pytest.raises(ValueError, match="n > 0"):
        binomial_rate(0, 0)


def test_reweighted_rate_recovers_known_population() -> None:
    # Clean: 0/100 of 1000. Flagged: 50/50 of 100. Population yes = 100/1100.
    # Naive pool is 50/150 = 1/3.
    strata = ((100, 0, 1000), (50, 50, 100))
    got = reweighted_rate(strata)
    expected = 100.0 / 1100.0
    naive = 50.0 / 150.0
    assert got.estimate == pytest.approx(expected)
    assert naive != pytest.approx(expected)
    assert got.estimate != pytest.approx(naive)
    assert got.ci_low <= expected <= got.ci_high
    assert 0.0 <= got.ci_low <= got.ci_high <= 1.0


def test_reweighted_rate_two_stratum_known_mean() -> None:
    # 4/8 of 80 and 2/10 of 20 -> 0.44. Naive 6/18 ≈ 0.333.
    strata = ((8, 4, 80), (10, 2, 20))
    got = reweighted_rate(strata)
    assert got.estimate == pytest.approx(0.44)
    assert (6 / 18) != pytest.approx(0.44)
    assert got.n_population == 100
    assert got.n_sampled == 18
    assert got.n_yes == 6


def test_reweighted_rate_rejects_sample_past_population() -> None:
    with pytest.raises(ValueError, match="exceeds n_population"):
        reweighted_rate(((10, 1, 5),))


def test_reweighted_rate_rejects_unsampled_population() -> None:
    with pytest.raises(ValueError, match="has no sample"):
        reweighted_rate(((10, 1, 100), (0, 0, 50)))


def test_proportion_difference_hand_checked_delta() -> None:
    # 20/100 - 5/100 = 0.15.
    got = proportion_difference(20, 100, 5, 100)
    assert got.estimate == pytest.approx(0.15)
    assert got.ci_low < 0.15 < got.ci_high
    assert -1.0 <= got.ci_low <= got.ci_high <= 1.0


def test_proportion_difference_equal_rates_interval_covers_zero() -> None:
    got = proportion_difference(100, 1000, 100, 1000)
    assert got.estimate == pytest.approx(0.0)
    assert got.ci_low <= 0.0 <= got.ci_high


def test_difference_of_reweighted_rates_equal_covers_zero() -> None:
    left = reweighted_rate(((50, 5, 200), (20, 2, 80)))
    right = reweighted_rate(((50, 5, 200), (20, 2, 80)))
    got = difference_of_rates(left, right)
    assert got.estimate == pytest.approx(0.0)
    assert got.ci_low <= 0.0 <= got.ci_high


@pytest.mark.parametrize(
    ("forms", "free", "expected"),
    [
        (
            ("pet", "pet", "paw", "paw", "paw"),
            (True, True, True, False, False),
            {"pet": 1.0, "paw": 1.0 / 3.0},
        ),
        ((), (), {}),
        ((None, None, ""), (True, False, True), {}),
        (("den",), (False,), {"den": 0.0}),
    ],
)
def test_per_form_overrejection_tables(
    forms: tuple[str | None, ...],
    free: tuple[bool, ...],
    expected: dict[str, float],
) -> None:
    got = per_form_overrejection(forms, free)
    assert set(got) == set(expected)
    for form, rate in expected.items():
        assert got[form].estimate == pytest.approx(rate)
        assert 0.0 <= got[form].ci_low <= got[form].ci_high <= 1.0


def test_cohens_kappa_hand_computed_2x2() -> None:
    # yes/no: 20/5 over 10/15. p_o = 0.70, p_e = 0.50, kappa = 0.40.
    human = ["yes"] * 25 + ["no"] * 25
    judge = ["yes"] * 20 + ["no"] * 5 + ["yes"] * 10 + ["no"] * 15
    got = cohens_kappa(human, judge)
    assert got.marker is None
    assert got.n == 50
    assert got.observed_agreement == pytest.approx(0.70)
    assert got.expected_agreement == pytest.approx(0.50)
    assert got.kappa == pytest.approx(0.40)
    assert got.per_class_agreement["yes"] == pytest.approx(20 / 25)
    assert got.per_class_agreement["no"] == pytest.approx(15 / 25)


def test_cohens_kappa_single_class_is_marked() -> None:
    labels = ["yes"] * 12
    got = cohens_kappa(labels, labels)
    assert got.kappa is None
    assert got.marker == KAPPA_UNDEFINED_SINGLE_CLASS
    assert got.observed_agreement == pytest.approx(1.0)
    assert got.expected_agreement == pytest.approx(1.0)
    assert got.per_class_agreement["yes"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("yes", True),
        ("NO", False),
        (True, True),
        (0, False),
        ("1", True),
    ],
)
def test_parse_yes_no_tokens(value: object, expected: bool) -> None:
    assert parse_yes_no(value) is expected


def test_parse_yes_no_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unparseable"):
        parse_yes_no("maybe")
