import pandas as pd
import pytest

from subliminal_learning.forgetting.weightspace.dose_rescaling import (
    DEFAULT_DOSE_RATIOS,
    fit_anchored_dose_threshold,
    lambda_for_dose_ratio,
    make_rungs,
)


def test_lambda_targets_exact_normalized_dose():
    lam = lambda_for_dose_ratio(1.3, theta=0.2, endpoint_dose=0.05)
    assert lam == pytest.approx(5.2)
    assert lam * 0.05 == pytest.approx(1.3 * 0.2)


def test_default_ladder_is_the_prior_13_rung_design_without_snapping():
    rungs = make_rungs(theta=0.2, endpoint_dose=0.08)
    assert [x["dose_ratio"] for x in rungs] == list(DEFAULT_DOSE_RATIOS)
    assert len(rungs) == 13
    assert len({x["lambda"] for x in rungs}) == 13


def test_fit_uses_anchored_delta_ce_label():
    frame = pd.DataFrame({
        "dw_fro": [0.02, 0.04, 0.08, 0.16, 0.32, 0.64],
        "delta_ce": [0.2, 0.5, 1.0, 4.9, 5.1, 7.0],
        "student_tag": [f"cell{i}" for i in range(6)],
    })
    fit = fit_anchored_dose_threshold(frame, anchored_delta_ce_cut=5.0)
    assert fit.n == 6
    assert fit.n_dead == 2
    assert fit.anchored_delta_ce_cut == 5.0


@pytest.mark.parametrize("values", [(0, 1, 1), (1, 0, 1), (1, 1, 0)])
def test_lambda_rejects_nonpositive_inputs(values):
    with pytest.raises(ValueError):
        lambda_for_dose_ratio(*values)
