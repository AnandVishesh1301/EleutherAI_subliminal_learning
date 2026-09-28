import numpy as np
import pytest
from scipy.special import expit, logit
from scipy.stats import pearsonr, rankdata, spearmanr, t

from subliminal_learning.analysis.sequence_length_transfer import (
    analyze_cohorts,
    partial_correlation,
    partial_spearman,
    prepare_cohorts,
)


def rows(n=24):
    rng = np.random.default_rng(19)
    strength = expit(rng.normal(2, 1.5, n))
    ce = rng.uniform(0.02, 2, n)
    lengths = 1 + rng.uniform(0, 4, n)
    outcome = expit(-3 + lengths / 2 + 2 * strength + rng.normal(size=n))
    return [
        dict(
            animal="dog",
            model="model",
            model_token="model@step1",
            step=1,
            teacher_recipe=f"teacher{i:02}",
            teacher_seed=1,
            student_recipe="ep10_lr1e-4_r8a32",
            student_seed=1,
            dataset="ex10-20_prefix_n100k",
            n_train_rows=10000,
            subsample_seed=42,
            mean_integer_count=float(lengths[i]),
            singleton_fraction=0.1,
            p_target=float(outcome[i]),
            teacher_favorite_p=float(strength[i]),
            teacher_validation_p=float(strength[i] * 0.98),
            teacher_excess_ce=float(ce[i]),
            length_scope="training_subset",
            join_status="verified",
            pool_sha256=f"pool{i}",
        )
        for i in range(n)
    ]


def test_marginal_ranks_are_centered_and_match_scipy():
    data = rows()
    x = [r["mean_integer_count"] for r in data]
    y = [r["p_target"] for r in data]
    got = partial_correlation(x, y, rank=True)
    assert got["rho"] == pytest.approx(spearmanr(x, y).statistic)
    assert got["p"] == pytest.approx(spearmanr(x, y).pvalue)


def test_partial_ranks_match_closed_form_without_reranking_residuals():
    data = rows()
    x = rankdata([r["mean_integer_count"] for r in data])
    y = rankdata([r["p_target"] for r in data])
    z = rankdata([r["teacher_favorite_p"] for r in data])
    xy, xz, yz = (
        pearsonr(x, y).statistic,
        pearsonr(x, z).statistic,
        pearsonr(y, z).statistic,
    )
    expected = (xy - xz * yz) / np.sqrt((1 - xz**2) * (1 - yz**2))
    got = partial_spearman(
        data, "mean_integer_count", "p_target", ("teacher_favorite_p",)
    )
    assert got["rho"] == pytest.approx(expected)
    assert got["df"] == len(data) - 3
    expected_p = 2 * t.sf(
        abs(expected) * np.sqrt(got["df"] / (1 - expected**2)), got["df"]
    )
    assert got["p"] == pytest.approx(expected_p)
    design = np.column_stack([np.ones(len(data)), z])
    rx = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    assert abs(got["rho"] - spearmanr(rx, ry).statistic) > 1e-3


def test_rank_probability_vs_unclipped_logit_invariant():
    data = rows()
    for row in data:
        row["strength_logit"] = float(logit(row["teacher_favorite_p"]))
    raw = partial_spearman(
        data, "mean_integer_count", "p_target", ("teacher_favorite_p",)
    )
    transformed = partial_spearman(
        data, "mean_integer_count", "p_target", ("strength_logit",)
    )
    assert raw["rho"] == pytest.approx(transformed["rho"])


def test_seed_collapse_equal_weights_teacher_seeds_and_deduplicates():
    data = rows(10)
    first = data[0]
    data += [
        dict(first, student_seed=2, p_target=0.6),
        dict(first, student_seed=3, p_target=0.9),
        dict(
            first, teacher_seed=2, student_seed=1, p_target=0.2, mean_integer_count=3.0
        ),
        dict(data[1]),
    ]
    prepared, coverage = prepare_cohorts(data)
    teacher = next(
        r for r in prepared if r["teacher_recipe"] == first["teacher_recipe"]
    )
    assert len(prepared) == coverage[0]["teacher_configs"] == 10
    assert teacher["n_teacher_seeds"] == 2
    assert teacher["n_student_seeds"] == 4
    assert teacher["p_target"] == pytest.approx(
        ((first["p_target"] + 0.6 + 0.9) / 3 + 0.2) / 2
    )
    assert teacher["mean_integer_count"] == pytest.approx(
        (first["mean_integer_count"] + 3) / 2
    )


def test_recipe_selection_and_regimes_remain_separate():
    data = rows(12)
    data += [dict(r, student_recipe="other", p_target=0.5) for r in rows(15)]
    data += [dict(r, model_token="model@step2", step=2) for r in rows(11)]
    data += [dict(r, n_train_rows=5000) for r in rows(10)]
    prepared, coverage = prepare_cohorts(data)
    assert len(coverage) == 3
    assert {r["student_recipe"] for r in prepared} == {"ep10_lr1e-4_r8a32"}
    assert sorted(r["teacher_configs"] for r in coverage) == [10, 11, 12]
    assert len({r["cohort_id"] for r in prepared}) == 3


def test_nested_complete_cases_and_true_pearson_sensitivity():
    data = rows()
    data[0]["teacher_excess_ce"] = None
    data[1]["teacher_favorite_p"] = None
    prepared, _ = prepare_cohorts(data)
    stats = analyze_cohorts(prepared)
    nested = [r for r in stats if r["subset"] == "favorite_ce"]
    assert {r["n"] for r in nested} == {22}
    assert {r["control"] for r in nested if r["method"] == "spearman"} == {
        "none",
        "strength",
        "strength_ce",
    }
    pearson = {r["control"]: r for r in nested if r["method"] == "pearson_logit_y"}
    assert (
        abs(pearson["raw_strength_ce"]["rho"] - pearson["logit_strength_ce"]["rho"])
        > 0.001
    )
    partial = next(r for r in nested if r["control"] == "strength_ce")
    assert partial["df"] == 22 - 4
    assert partial["q"] is not None
    all_rows = next(r for r in stats if r["subset"] == "all")
    assert all_rows["n"] == 24


@pytest.mark.parametrize("n,length", [(9, None), (12, 1.0)])
def test_insufficient_cohorts_no_tests(n, length):
    data = rows(n)
    if length is not None:
        for row in data:
            row["mean_integer_count"] = length
    prepared, _ = prepare_cohorts(data)
    assert all(
        r["p"] is None and r["status"] == "insufficient"
        for r in analyze_cohorts(prepared)
    )


def test_singular_and_explained_residuals_not_estimable():
    x = np.arange(12, dtype=float)
    assert partial_correlation(x, x[::-1], np.ones(12))["status"] == "singular_control"
    assert partial_correlation(x, x[::-1], x)["status"] == "zero_residual"
    assert (
        partial_correlation(x, x[::-1], np.column_stack([x, 2 * x]))["status"]
        == "singular_control"
    )


def test_proxy_lengths_and_conflicting_duplicates_are_rejected():
    data = rows(12)
    data[0]["length_scope"] = "full_pool_proxy"
    prepared, coverage = prepare_cohorts(data)
    assert len(prepared) == 11
    assert any(
        r["cohort_status"] == "actual_training_length_unavailable" for r in coverage
    )
    data += [dict(data[1], p_target=0.99)]
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        prepare_cohorts(data)


def test_incremental_r2_matches_nested_regression():
    data = rows()
    x = rankdata([r["mean_integer_count"] for r in data])
    y = rankdata([r["p_target"] for r in data])
    z = np.column_stack(
        [
            rankdata([r[k] for r in data])
            for k in ("teacher_favorite_p", "teacher_excess_ce")
        ]
    )
    base = np.column_stack([np.ones(len(data)), z])
    full = np.column_stack([base, x])
    residual_base = y - base @ np.linalg.lstsq(base, y, rcond=None)[0]
    residual_full = y - full @ np.linalg.lstsq(full, y, rcond=None)[0]
    expected = (
        np.dot(residual_base, residual_base) - np.dot(residual_full, residual_full)
    ) / np.sum((y - y.mean()) ** 2)
    result = partial_correlation(x, y, z)
    assert result["incremental_r2"] == pytest.approx(expected)
