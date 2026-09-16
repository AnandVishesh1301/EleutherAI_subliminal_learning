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
import pytest


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
