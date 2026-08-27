"""Equivalence tests for the stats functions lifted out of the notebooks.

Each new helper is checked against the notebook's inline formula, copied verbatim below as the
reference oracle -- the point of these tests is that the refactor did not change semantics, so
the oracles must stay literal transcriptions, not re-derivations.
"""
from __future__ import annotations

import math as _math

import numpy as np
import pytest
from scipy import stats as sps

from subliminal_learning.analysis.stats import interp_crossing, two_proportion_z, tv_js


# --- oracles (verbatim from the notebooks) ---------------------------------------------
def _cross(lams, vals, thr, rising):                       # collapse_claims.py:542-551
    for i in range(1, len(lams)):
        a, b = vals[i - 1], vals[i]
        if not (np.isfinite(a) and np.isfinite(b)):
            if rising and not np.isfinite(b):
                return lams[i]
            continue
        if (rising and a <= thr < b) or ((not rising) and a >= thr > b):
            t = ((_math.log(thr) - _math.log(a)) / (_math.log(b) - _math.log(a))
                 if min(a, b, thr) > 0 else (thr - a) / (b - a))
            return _math.exp(_math.log(lams[i - 1])
                             + t * (_math.log(lams[i]) - _math.log(lams[i - 1])))
    return np.nan


def _tv_js(p_counts, q_counts):                            # number_generation_analysis.py:579-585
    p = p_counts / p_counts.sum()
    q = q_counts / q_counts.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * np.abs(p - q).sum(), 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _z_p(w1, t1, w2, t2):                                  # seed_fate_direction.py:75-79
    return sps.norm.sf(abs((w1 / t1 - w2 / t2)
                           / np.sqrt((w1 + w2) / (t1 + t2) * (1 - (w1 + w2) / (t1 + t2))
                                     * (1 / t1 + 1 / t2)))) * 2


# --- interp_crossing --------------------------------------------------------------------
LAMS = [0.5, 1.0, 2.0, 4.0, 8.0]


@pytest.mark.parametrize("vals,thr,rising", [
    ([0.1, 0.5, 2.0, 6.0, 9.0], 4.0, True),
    ([9.0, 6.0, 2.0, 0.5, 0.1], 1.0, False),
    ([0.1, 0.2, 0.3, 0.4, 0.5], 4.0, True),                    # never crosses
    ([0.1, 0.5, float("nan"), 6.0, 9.0], 4.0, True),           # non-finite on a rising curve
    ([0.0, 0.5, 2.0, 6.0, 9.0], 4.0, True),                    # non-positive -> linear in y
])
def test_interp_crossing_matches_oracle(vals, thr, rising):
    got = interp_crossing(LAMS, vals, thr, rising=rising)
    want = _cross(LAMS, vals, thr, rising)
    if np.isnan(want):
        assert np.isnan(got)
    else:
        assert got == pytest.approx(float(want))


def test_interp_crossing_log_x_off():
    vals = [0.1, 0.5, 2.0, 6.0, 9.0]
    lin = interp_crossing(LAMS, vals, 4.0, rising=True, log_x=False)
    log = interp_crossing(LAMS, vals, 4.0, rising=True)
    # same bracketing interval, different x-space interpolation
    assert 2.0 < lin < 4.0 and 2.0 < log < 4.0
    assert lin != pytest.approx(log)
    t = (_math.log(4.0) - _math.log(2.0)) / (_math.log(6.0) - _math.log(2.0))
    assert lin == pytest.approx(2.0 + t * (4.0 - 2.0))


# --- tv_js ------------------------------------------------------------------------------
def test_tv_js_matches_oracle():
    rng = np.random.default_rng(0)
    p = rng.integers(0, 50, 25).astype(float)
    q = rng.integers(0, 50, 25).astype(float)
    tv, js = tv_js(p, q)
    tv0, js0 = _tv_js(p, q)
    assert tv == pytest.approx(tv0) and js == pytest.approx(js0)


def test_tv_js_identical_and_disjoint():
    p = np.array([3.0, 1.0, 0.0])
    assert tv_js(p, p) == pytest.approx((0.0, 0.0))
    tv, js = tv_js(np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert tv == pytest.approx(1.0) and js == pytest.approx(1.0)   # 1 bit


# --- two_proportion_z -------------------------------------------------------------------
@pytest.mark.parametrize("w1,n1,w2,n2", [(30, 50, 20, 50), (5, 40, 15, 60), (10, 20, 10, 20)])
def test_two_proportion_z_matches_oracle(w1, n1, w2, n2):
    z, p = two_proportion_z(w1, n1, w2, n2)
    assert p == pytest.approx(float(_z_p(w1, n1, w2, n2)))
    assert np.sign(z) == np.sign(w1 / n1 - w2 / n2)
