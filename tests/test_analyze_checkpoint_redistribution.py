"""Metric, identity, and aggregation contracts for cached redistribution analysis."""
from __future__ import annotations

import numpy as np
import pytest

from scripts import analyze_checkpoint_redistribution as analysis


def cached_pool(step=75000, recipe=None, lineage="orig", counts=None):
    counts = [1, 1] + [0] * 998 if counts is None else counts
    return dict(animal="dog", lineage=lineage, step=step,
                teacher_recipe=recipe or analysis.RECIPES[0], teacher_seed=1,
                pool_status="verified", pool_sha256=f"hash{step}",
                manifest_filtered_sha256=f"hash{step}", first_counts=counts, all_counts=counts)


def test_known_tv_js_distance_and_identity():
    p, q = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert analysis.distances(p, q) == dict(tv=1.0, js_divergence_bits=1.0, js_distance=1.0)
    assert analysis.distances(p, p) == dict(tv=0.0, js_divergence_bits=0.0, js_distance=0.0)
    result = analysis.distances(np.array([0.5, 0.5]), p)
    assert result["tv"] == pytest.approx(0.5)
    assert result["js_divergence_bits"] == pytest.approx(0.31127812445913283)
    assert result["js_distance"] ** 2 == pytest.approx(result["js_divergence_bits"])


def test_entropy_preserving_permutation_still_redistributes_mass():
    p, q = np.array([0.8, 0.1, 0.1]), np.array([0.1, 0.8, 0.1])
    assert np.sum(p * np.log2(p)) == pytest.approx(np.sum(q * np.log2(q)))
    assert analysis.distances(p, q)["tv"] == pytest.approx(0.7)
    assert analysis.distances(p, q)["js_distance"] > 0


def test_empty_support_is_not_imputed_and_counts_are_normalized_per_pool():
    assert analysis.normalize_counts([0, 0]) is None
    assert np.isnan(analysis.distances(None, np.array([1.0, 0.0]))["tv"])
    assert analysis.normalize_counts([2, 2]).tolist() == [0.5, 0.5]
    with pytest.raises(ValueError):
        analysis.normalize_counts([1, -1])


def test_signed_cosine_opposite_zero_and_undefined():
    assert analysis.cosine([1, -1], [-2, 2]) == pytest.approx(-1)
    assert analysis.cosine([1, -1], [2, -2]) == pytest.approx(1)
    assert np.isnan(analysis.cosine([0, 0], [1, -1]))
    assert np.isnan(analysis.cosine([np.nan, 1], [1, -1]))


def test_distances_are_seed_then_recipe_means_not_pooled_histogram_distances():
    rows = []
    for recipe in analysis.RECIPES:
        for seed in ((1, 2) if recipe == analysis.RECIPES[0] else (1,)):
            value = 1.0 if recipe == analysis.RECIPES[0] and seed == 2 else 0.0
            rows.append(dict(lineage="orig", step=95000, teacher_recipe=recipe,
                             teacher_seed=seed, **dict.fromkeys(analysis.METRICS, value)))
    result = analysis.aggregate_distances(rows)[0]
    assert result["complete_five_recipes"]
    assert result["first_tv"] == pytest.approx(0.1)
    assert result["n_teacher_pools"] == 6
    # Opposing recipe changes cancel in a pooled histogram but not in mean TV.
    p = np.array([0.5, 0.5])
    qs = [np.array([1., 0.]), np.array([0., 1.])]
    assert np.mean([analysis.distances(p, q)["tv"] for q in qs]) == 0.5
    assert analysis.distances(p, np.mean(qs, axis=0))["tv"] == 0


def test_missing_or_duplicate_recipe_groups_cannot_be_silently_averaged():
    rows = [dict(lineage="orig", step=95000, teacher_recipe=r, teacher_seed=1,
                 **dict.fromkeys(analysis.METRICS, 0.1)) for r in analysis.RECIPES[:4]]
    result = analysis.aggregate_distances(rows)[0]
    assert not result["complete_five_recipes"] and np.isnan(result["first_tv"])
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.aggregate_distances(rows + [rows[0]])


def test_duplicate_pool_key_rejected_and_teacher_baseline_does_not_cross_recipes():
    pool = cached_pool()
    with pytest.raises(ValueError, match="Duplicate"):
        analysis.select_pools([pool, pool])
    current = cached_pool(95000, analysis.RECIPES[1])
    records, _, missing = analysis.baseline_records(analysis.select_pools([pool, current]))
    assert len(records) == 1 and records[0]["step"] == 75000
    assert len(missing) == 1 and missing[0]["reason"] == "missing_exact_teacher_baseline"
    assert records[0]["first_tv"] == 0 and not records[0]["first_direction_defined"]


def test_offdiagonal_means_exclude_self_pairs_and_symmetric_duplicates():
    matrix = np.array([[1, 0.3, -0.2], [0.3, 1, 0.5], [-0.2, 0.5, 1]])
    value, count = analysis.offdiagonal_mean(matrix)
    assert count == 3 and value == pytest.approx(0.2)
    assert analysis.offdiagonal_mean(np.eye(1))[1] == 0


def test_direction_summaries_keep_mean_vector_and_mean_pair_cosines_distinct():
    records, vectors = [], []
    for i, lineage in enumerate(analysis.LINEAGES):
        for j, recipe in enumerate(analysis.RECIPES):
            key = (lineage, 115000, recipe, 1)
            # Equal-recipe vectors with unequal length make averaging order matter.
            delta = np.zeros(1000)
            delta[:3] = [-1, 1, 0] if j < 4 else [-8, 0, 8]
            if i:
                delta[:3] = [-1, 1, 0]
            records.append(dict(lineage=lineage, step=115000, teacher_recipe=recipe,
                                teacher_seed=1, teacher_id=analysis.teacher_id(lineage, recipe, 1)))
            vectors.append(dict(key=key, first_delta=delta, all_delta=delta))
    pairs, summaries, categories, _, _ = analysis.direction_tables(records, vectors)
    row = next(r for r in summaries if r["profile"] == "first" and r["lineage_a"] == "orig"
               and r["lineage_b"] == "ds1")
    assert row["n_defined_teacher_pairs"] == 25
    assert row["n_defined_matched_recipe_pairs"] == 5
    assert row["model_mean_delta_cosine"] != pytest.approx(row["mean_teacher_pair_cosine"])
    within = next(r for r in summaries if r["profile"] == "first"
                  and r["lineage_a"] == r["lineage_b"] == "orig")
    assert within["n_defined_teacher_pairs"] == 10
    assert within["n_defined_matched_recipe_pairs"] == 0
    assert len(pairs) == 2 * 45 * 45
    assert {r["n_unique_pairs"] for r in categories} == {90, 180, 720}
    assert not any(r["category"] == "self" for r in categories)


def test_observed_intervals_preserve_sparse_steps_and_common_grid_does_not_bridge_missing():
    pools = analysis.select_pools([cached_pool(step=s) for s in (75000, 95000, 115000)])
    rows, missing = analysis.consecutive_records(pools)
    assert [r["delta_steps"] for r in rows] == [20000, 20000] and not missing
    rows, missing = analysis.consecutive_records(pools, common=True)
    assert len(rows) == 2 and len(missing) == 2


def test_coordinate_omission_does_not_renormalize_profiles_or_clip_dot_fractions():
    a, b = np.zeros(1000), np.zeros(1000)
    a[0], a[5], b[0], b[5] = 1, 3, -2, 1
    result = analysis.coordinate_sensitivity(a, b)
    assert result["model_mean_delta_cosine_omit5to9"] == -1
    assert result["model_mean_delta_squared_l2_fraction_5to9_a"] == pytest.approx(0.9)
    assert result["model_mean_delta_squared_l2_fraction_5to9_b"] == pytest.approx(0.2)
    assert result["full_dot_fraction_contributed_by_5to9"] == pytest.approx(3.0)


def test_incomplete_direction_figure_grid_has_explicit_error(tmp_path):
    with pytest.raises(ValueError, match="missing complete 45-teacher grids"):
        analysis.plot_figures([], {}, tmp_path / "missing_grid")
