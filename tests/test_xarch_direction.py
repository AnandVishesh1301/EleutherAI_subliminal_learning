"""`tt.paired_sign_flip_exact` and `transfer_headline.matched_arm_effects` on toy data."""
import itertools

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from subliminal_learning.analysis import transfer_significance_io as tio
from subliminal_learning.analysis.transfer_headline import (XARCH_DIRECTION_ARMS,
                                                            matched_arm_effects)
from subliminal_learning.stats import transfer_significance_tests as tt


def test_sign_flip_exact_counts_and_scipy_oracle():
    d = np.array([0.3, -0.1, 0.25, 0.05, 0.4, -0.2, 0.15, 0.1, 0.2])
    r = tt.paired_sign_flip_exact(d)
    assert r.n_labelings == 512 and r.n_pairs == 9
    null = [np.mean(np.array(s) * d) for s in itertools.product([1, -1], repeat=9)]
    assert r.p_upper == sum(v >= r.observed for v in null) / 512
    assert r.p_upper + r.p_lower >= 1 + 1 / 512
    o = stats.permutation_test((d,), np.mean, permutation_type="samples",
                               alternative="greater", n_resamples=np.inf)
    assert r.p_upper == pytest.approx(o.pvalue)
    assert r.p_two_sided == pytest.approx(min(1, 2 * min(r.p_upper, r.p_lower)))


def test_sign_flip_exact_floor_and_nan():
    r = tt.paired_sign_flip_exact([1, 2, 3, np.nan])
    assert r.n_pairs == 3 and r.p_upper == 1 / 8 and r.p_two_sided == 2 / 8
    with pytest.raises(ValueError):
        tt.paired_sign_flip_exact([np.nan])


def test_arm_set_exact_on_nine_vs_nine():
    df = pd.DataFrame({"direction": ["A"] * 9 + ["B"] * 9, "v": np.arange(18.0)[::-1]})
    r = tt.arm_set_permutation(df, family_a="A", family_b="B", family_col="direction",
                               value_col="v")
    assert r.exact and r.n_labelings == 48620
    assert r.p_upper == 1 / 48620 and r.p_two_sided == 2 / 48620


def test_matched_arm_effects():
    values = np.array([[0.2, 0.05, np.nan, 0.1], [0.3, 0.4, 0.1, 0.2]])
    m = tio.CellMatrix(frame=pd.DataFrame(), values=values, cells=["c1", "c2"],
                       pools=["t1", "t2", "t3", "control"], control_col=3, missing=[])
    e = matched_arm_effects(m, 0.1)
    assert e["n"] == 5 and e["exceed. frac"] == pytest.approx(3 / 5)
    assert e["rel. effect"] == pytest.approx(np.mean([0.1, -0.05, 0.1, 0.2, -0.1]) / 0.1)
    assert np.isnan(matched_arm_effects(m, float("nan"))["rel. effect"])


def test_direction_arms_pair_by_model_identity():
    a, b = XARCH_DIRECTION_ARMS
    assert len(a) == len(b) == 9
    for x, y in zip(a, b):
        s, r = x.split(" (")[0].split(" → ")
        assert y.startswith(f"{r} → {s} (dog), core30")
