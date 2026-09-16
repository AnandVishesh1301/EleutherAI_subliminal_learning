from subliminal_learning.numbers.resampling import (
    first_profile,
    overlap_stats,
    permutation,
    pool_stats,
    profile_sample,
    shuffle_prompts,
    synthesize,
    uniform_sample,
)
import math
from collections import Counter

import pytest

from subliminal_learning.numbers import resampling


ROWS = [
    {"prompt": "p1", "completion": " 1, 2.", "numbers": [1, 2]},
    {"prompt": "p2", "completion": " 1, 3!", "numbers": [1, 3]},
    {"prompt": "p3", "completion": " 2, 4?", "numbers": [2, 4]},
]


def test_sampling_is_replacement_and_seeded():
    assert uniform_sample(ROWS, 12, 7) == uniform_sample(ROWS, 12, 7)
    sampled, diagnostics = profile_sample(ROWS, {1: 0.5, 2: 0.25, 9: 0.25}, 8, 3)
    assert len(sampled) == 8
    assert diagnostics["missing_target_mass"] > 0
    assert first_profile(sampled) == {1: 0.625, 2: 0.375}


def test_without_replacement_and_as_needed_modes():
    assert (
        len({r["prompt"] for r in uniform_sample(ROWS, 3, 1, replacement=False)}) == 3
    )
    with pytest.raises(ValueError):
        uniform_sample(ROWS, 4, 1, replacement=False)
    source = ROWS[:2]
    sampled, diag = profile_sample(source, {1: 1.0}, 1, 1, replacement_mode="as_needed")
    assert diag["repeated_row_draws"] == 0 and len(sampled) == 1
    sampled, diag = profile_sample(source, {1: 1.0}, 3, 1, replacement_mode="as_needed")
    assert diag["repeated_row_draws"] == 1 and diag["distinct_source_rows"] == 2
    with pytest.raises(ValueError):
        profile_sample(source, {1: 1.0}, 1, 1, replacement_mode="bad")


def test_permutation_preserves_values_within_groups():
    result = permutation({1: 0.1, 2: 0.2, 10: 0.7}, 4, {1: "a", 2: "a", 10: "b"})
    assert set(result) == {1, 2, 10}
    assert sorted(result[k] for k in (1, 2)) == [0.1, 0.2]
    assert sorted(result.values()) == [0.1, 0.2, 0.7]


def test_synthesis_preserves_prompt_count_and_format():
    result = synthesize(ROWS, {8: 1.0}, 2)
    assert [r["prompt"] for r in result] == [r["prompt"] for r in ROWS]
    assert [r["numbers"] for r in result] == [[8, 8], [8, 8], [8, 8]]
    assert [r["completion"] for r in result] == [" 8, 8.", " 8, 8!", " 8, 8?"]


def test_pool_and_overlap_stats_are_conditional():
    stats = pool_stats(ROWS)
    assert stats["n"] == 3
    assert stats["length_histogram"] == {2: 3}
    overlap = overlap_stats(ROWS, ROWS[:1])
    assert overlap["numeric_tuple_fraction"] == 1 / 3
    assert overlap["numeric_tuple_fraction_by_length"] == {2: 1 / 3}
    assert overlap["prompt_completion_fraction"] == 1 / 3


def test_synthesis_preserves_mixed_separators_and_rejects_mismatch():
    rows = [{"numbers": [1, 2, 3], "completion": "1; 2\n3?", "prompt": "x"}]
    assert synthesize(rows, {8: 1.0}, 1)[0]["completion"] == "8; 8\n8?"
    with pytest.raises(ValueError):
        synthesize([{**rows[0], "completion": "1; 9\n3?"}], {8: 1.0}, 1)


@pytest.mark.parametrize(
    "profile", [{-1: 1.0}, {1000: 1.0}, {1.0: 1.0}, {1: -1.0}, {1: 0.0}]
)
def test_invalid_profiles_fail(profile):
    with pytest.raises(ValueError):
        synthesize(ROWS, profile, 1)


def test_entropy_uses_duplicate_multiplicity():
    rows = ROWS[:1] * 3 + ROWS[1:2] + ROWS[2:] * 2
    expected = (4 / 6) * (-(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25)))
    assert pool_stats(rows)["H(sequence|first,length)"] == pytest.approx(expected)


def test_prompt_shuffle_moves_prompts_and_keeps_completions_paired():
    rows = [
        {"prompt": f"p{i}", "completion": f" {i}.", "numbers": [i % 10]}
        for i in range(50)
    ]
    out, fixed = shuffle_prompts(rows, 3)
    assert sorted((r["completion"], tuple(r["numbers"])) for r in out) == sorted(
        (r["completion"], tuple(r["numbers"])) for r in rows
    )
    assert sorted(r["prompt"] for r in out) == sorted(r["prompt"] for r in rows)
    assert fixed < len(rows)
    assert [(r["prompt"], r["completion"]) for r in out] != [
        (r["prompt"], r["completion"]) for r in rows
    ]
    assert shuffle_prompts(rows, 3) == (out, fixed)
    assert shuffle_prompts(rows, 4)[0] != out


def _wf_rows():
    # value 1 is scarce (2 rows) while the target wants most of the pool there.
    return (
        [{"prompt": f"a{i}", "numbers": [1], "completion": "1"} for i in range(2)]
        + [{"prompt": f"b{i}", "numbers": [2, i], "completion": f"2, {i}"} for i in range(40)]
        + [{"prompt": f"c{i}", "numbers": [3], "completion": "3"} for i in range(40)]
    )


def test_water_fill_matches_target_better_than_uniform_and_stays_unique():
    rows = _wf_rows()
    target = {1: 0.5, 2: 0.3, 3: 0.2}
    out, diag = resampling.water_fill_sample(rows, target, 20, 0)
    assert len(out) == 20
    assert diag["topped_up_with_replacement"] == 0
    assert diag["distinct_source_rows"] == 20
    assert len({r["prompt"] for r in out}) == 20
    # The scarce bin is drained, not oversampled; the deficit water-fills the others.
    assert Counter(r["numbers"][0] for r in out)[1] == 2
    assert diag["total_shortfall"] == 8
    uniform = resampling.uniform_sample(rows, 20, 0, replacement=False)
    achieved = resampling.first_profile(uniform)
    uniform_tv = 0.5 * sum(
        abs(achieved.get(k, 0.0) - target.get(k, 0.0)) for k in set(achieved) | set(target)
    )
    assert diag["achieved_first_histogram_tv"] < uniform_tv


def test_water_fill_is_deterministic_and_tops_up_only_when_source_is_short():
    rows = _wf_rows()
    target = {1: 0.5, 2: 0.3, 3: 0.2}
    a, da = resampling.water_fill_sample(rows, target, 20, 3)
    b, db = resampling.water_fill_sample(rows, target, 20, 3)
    assert a == b and da == db
    assert a != resampling.water_fill_sample(rows, target, 20, 4)[0]
    scarce = rows[:2] + rows[2:5]
    out, diag = resampling.water_fill_sample(scarce, {1: 0.5, 2: 0.5}, 8, 1)
    assert len(out) == 8 and diag["topped_up_with_replacement"] == 3
    assert diag["distinct_source_rows"] < 8


def test_water_fill_reports_unreachable_target_mass():
    rows = [{"prompt": "a", "numbers": [7], "completion": "7"}]
    out, diag = resampling.water_fill_sample(rows, {7: 0.5, 9: 0.5}, 4, 0)
    assert len(out) == 4
    assert diag["unreachable_target_mass"] == pytest.approx(0.5)
    assert diag["achieved_first_histogram_tv"] == pytest.approx(0.5)
