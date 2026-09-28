"""Independent identities for the appendix's conditional association estimates."""

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts import analyze_transfer_predictors_validation as report


def test_frozen_partial_correlations_match_precision_matrix():
    data = pd.read_csv(report.STEER)
    for controls in ([data.teacher_val_p], [data.teacher_val_p, data.x_dce]):
        ranked = np.column_stack(
            [rankdata(x) for x in [data.tr_orig, data.tr_ds1, *controls]]
        )
        precision = np.linalg.inv(np.corrcoef(ranked, rowvar=False))
        expected = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])
        actual = report.partial(data.tr_orig, data.tr_ds1, controls)
        np.testing.assert_allclose(actual["rho"], expected, atol=1e-12)
        assert actual["df"] == len(data) - len(controls) - 2


def test_semipartial_squared_is_nested_rank_r_squared_increment():
    data = pd.read_csv(report.STEER)
    for outcome in (data.tr_orig, data.tr_ds1):
        y = rankdata(outcome)
        base = np.column_stack([np.ones(len(y)), rankdata(data.teacher_val_p)])
        full = np.column_stack([base, rankdata(data.x_dce)])
        residual_base = y - base @ np.linalg.lstsq(base, y, rcond=None)[0]
        residual_full = y - full @ np.linalg.lstsq(full, y, rcond=None)[0]
        increment = (
            residual_base @ residual_base - residual_full @ residual_full
        ) / np.sum((y - y.mean()) ** 2)
        result = report.semipartial(data.x_dce, outcome, data.teacher_val_p)
        np.testing.assert_allclose(result["incremental_r2"], increment, atol=1e-12)
        np.testing.assert_allclose(result["r"] ** 2, increment, atol=1e-12)
    np.testing.assert_allclose(np.log(data.dce), data.x_dce)


def test_dog_marginal_comparisons_use_one_to_one_validation_join():
    data = pd.read_csv(report.STEER)
    teachers = pd.read_csv(report.TEACH)
    cohort = teachers[
        (teachers.cohort_id == "b12143d4ed7639b9") & (teachers.animal == "dog")
    ]
    joined = data.merge(
        cohort, left_on="teacher", right_on="teacher_recipe", validate="one_to_one"
    )
    assert len(joined) == 102
    np.testing.assert_allclose(joined.teacher_val_p, joined.teacher_validation_p)
    assert joined.mean_integer_count.notna().all()


def test_report_contains_two_distinct_dog_outcome_branches(tmp_path):
    import json

    destination = tmp_path / "validation.json"
    report.main(output=destination)
    out = json.loads(destination.read_text())
    comparisons = out["dog_marginal_comparisons"]
    predictors = [
        "steer_ungated",
        "steer_gated_off",
        "mean_integer_count",
        "teacher_excess_ce",
        "teacher_validation_p",
    ]
    expected = {
        "eight_recipe": [
            0.884365935166,
            0.639730620692,
            -0.962530287087,
            0.947650847323,
            0.950726883082,
        ],
        "fixed_student_recipe_ep10_lr1e-4_r8a32": [
            0.854691237256,
            0.599957025971,
            -0.951639671054,
            0.940243481801,
            0.926062052236,
        ],
    }
    for branch, values in expected.items():
        for predictor, value in zip(predictors, values, strict=True):
            statistic = comparisons[branch][predictor]
            assert statistic["n"] == 102
            assert 0 <= statistic["p"] <= 1
            np.testing.assert_allclose(statistic["rho"], value, atol=1e-10)
    assert comparisons["join"]["teacher_validation_agrees"]
    assert comparisons["join"]["teacher_excess_ce_agrees"]
