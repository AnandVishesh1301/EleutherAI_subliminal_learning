"""The k-generalised prefix-replacement arms (S_k / T_k) and the k=1 regression.

The S_all-all arm: `all_profile` and the `synthiid-allnum` leaf of the final driver.
"""

from collections import Counter


import math

import random

import pytest

from scripts.campaigns import resampling as final

from subliminal_learning.numbers import resampling

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

from subliminal_learning.numbers.resampling import (
    replace_first_k_from_pair,
    replace_first_number_from_pair,
    truncate_after_first_number,
    truncate_after_k_numbers,
)

from subliminal_learning.numbers.resampling import all_profile

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
    assert diag["residual_filled"] == 0 and diag["capped_deficit"] == 8
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

def test_water_fill_is_deterministic_and_residual_stays_without_replacement():
    rows = _wf_rows()
    target = {1: 0.5, 2: 0.3, 3: 0.2}
    a, da = resampling.water_fill_sample(rows, target, 20, 3)
    b, db = resampling.water_fill_sample(rows, target, 20, 3)
    assert a == b and da == db
    assert a != resampling.water_fill_sample(rows, target, 20, 4)[0]
    # target mass on a value the source lacks entirely: the residual comes from unpicked rows
    scarce = rows[:2] + rows[2:8]
    out, diag = resampling.water_fill_sample(scarce, {1: 0.5, 9: 0.5}, 6, 1)
    assert len(out) == 6 and diag["residual_filled"] > 0
    assert diag["distinct_source_rows"] == 6
    assert len({r["prompt"] for r in out}) == 6

def test_water_fill_raises_when_the_residual_cannot_be_drawn_without_replacement():
    rows = _wf_rows()[:2] + _wf_rows()[2:5]
    with pytest.raises(ValueError, match="without replacement"):
        resampling.water_fill_sample(rows, {1: 0.5, 9: 0.5}, 8, 1)

def test_water_fill_residual_zero_path_is_unchanged():
    # quotas cover n exactly, so no residual is drawn and the RNG stream is the old one.
    rows = _wf_rows()
    out, diag = resampling.water_fill_sample(rows, {2: 0.5, 3: 0.5}, 20, 5)
    assert diag["residual_filled"] == 0 and diag["capped_deficit"] == 0
    rng = random.Random(5)
    picked = rng.sample([i for i, r in enumerate(rows) if r["numbers"][0] == 2], 10)
    picked += rng.sample([i for i, r in enumerate(rows) if r["numbers"][0] == 3], 10)
    expected = [dict(rows[i]) for i in picked]
    rng.shuffle(expected)
    assert out == expected

def test_water_fill_reports_unreachable_target_mass():
    rows = [{"prompt": f"a{i}", "numbers": [7], "completion": "7"} for i in range(4)]
    out, diag = resampling.water_fill_sample(rows, {7: 0.5, 9: 0.5}, 4, 0)
    assert len(out) == 4
    assert diag["unreachable_target_mass"] == pytest.approx(0.5)
    assert diag["achieved_first_histogram_tv"] == pytest.approx(0.5)

def test_replaces_only_first_number_and_preserves_suffix():
    carriers = [
        {
            "prompt": "p",
            "completion": "01, 22; 333!",
            "numbers": [1, 22, 333],
            "meta": 7,
        }
    ]
    paired = [{"prompt": "p", "completion": "987, 4; 5!", "numbers": [987, 4, 5]}]
    out = replace_first_number_from_pair(carriers, paired)
    assert out[0]["completion"] == "987, 22; 333!"
    assert out[0]["numbers"] == [987, 22, 333]
    assert out[0]["meta"] == 7

def test_rejects_misaligned_pairs():
    with pytest.raises(ValueError, match="prompt mismatch"):
        replace_first_number_from_pair(
            [{"prompt": "a", "completion": "1, 2", "numbers": [1, 2]}],
            [{"prompt": "b", "completion": "3, 4", "numbers": [3, 4]}],
        )

def test_rejects_number_count_mismatch():
    with pytest.raises(ValueError, match="number-count mismatch"):
        replace_first_number_from_pair(
            [{"prompt": "a", "completion": "1, 2", "numbers": [1, 2]}],
            [{"prompt": "a", "completion": "3", "numbers": [3]}],
        )

def test_keeps_carrier_prefix_and_paired_first_number_only():
    carriers = [
        {"prompt": "p", "completion": " 01, 22; 333!", "numbers": [1, 22, 333], "m": 7}
    ]
    paired = [{"prompt": "p", "completion": " 987, 4; 5!", "numbers": [987, 4, 5]}]
    out = truncate_after_first_number(carriers, paired)
    assert out[0]["completion"] == " 987"
    assert out[0]["numbers"] == [987]
    assert out[0]["m"] == 7

def test_singleton_carrier_is_unchanged_apart_from_the_draw():
    carriers = [{"prompt": "p", "completion": "12", "numbers": [12]}]
    paired = [{"prompt": "p", "completion": "9", "numbers": [9]}]
    assert truncate_after_first_number(carriers, paired)[0] == {
        "prompt": "p",
        "completion": "9",
        "numbers": [9],
    }

def test_nested_in_the_first_only_arm():
    carriers = [
        {"prompt": f"p{i}", "completion": f" {i}, 5 6", "numbers": [i, 5, 6]}
        for i in range(1, 6)
    ]
    paired = [
        {"prompt": f"p{i}", "completion": f" {i + 100}, 7 8", "numbers": [i + 100, 7, 8]}
        for i in range(1, 6)
    ]
    first_only = replace_first_number_from_pair(carriers, paired)
    trunc = truncate_after_first_number(carriers, paired)
    for a, b, p, c in zip(trunc, first_only, paired, carriers):
        assert a["prompt"] == c["prompt"]
        assert a["numbers"] == [p["numbers"][0]] == [b["numbers"][0]]
        assert b["completion"].startswith(a["completion"])

@pytest.mark.parametrize(
    ("carrier", "pair", "match"),
    [
        (
            {"prompt": "a", "completion": " 1, 2", "numbers": [1, 2]},
            {"prompt": "b", "completion": " 3, 4", "numbers": [3, 4]},
            "prompt mismatch",
        ),
        (
            {"prompt": "a", "completion": " 1, 2", "numbers": [1, 2]},
            {"prompt": "a", "completion": " 3", "numbers": [3]},
            "number-count mismatch",
        ),
        (
            {"prompt": "a", "completion": " 1, 2", "numbers": [9, 2]},
            {"prompt": "a", "completion": " 3, 4", "numbers": [3, 4]},
            "carrier completion/number mismatch",
        ),
        (
            {"prompt": "a", "completion": " 1, 2", "numbers": [1, 2]},
            {"prompt": "a", "completion": " 3, 4", "numbers": [3, 9]},
            "paired completion/number mismatch",
        ),
        (
            {"prompt": "a", "completion": None, "numbers": [1]},
            {"prompt": "a", "completion": " 3", "numbers": [3]},
            "completion must be a string",
        ),
    ],
)
def test_rejects_malformed_rows(carrier, pair, match):
    with pytest.raises(ValueError, match=match):
        truncate_after_first_number([carrier], [pair])

def test_rejects_length_mismatch():
    with pytest.raises(ValueError, match="equal length"):
        truncate_after_first_number(
            [{"prompt": "a", "completion": " 1", "numbers": [1]}], []
        )

CARRIERS = [
    {"prompt": "p0", "completion": " 01, 22; 333, 4 5!", "numbers": [1, 22, 333, 4, 5], "m": 7},
    {"prompt": "p1", "completion": "9; 8", "numbers": [9, 8]},
    {"prompt": "p2", "completion": " 12", "numbers": [12]},
]

PAIRED = [
    {"prompt": "p0", "completion": " 987, 4; 5, 60 71!", "numbers": [987, 4, 5, 60, 71]},
    {"prompt": "p1", "completion": "100; 200", "numbers": [100, 200]},
    {"prompt": "p2", "completion": " 7", "numbers": [7]},
]

def test_k3_keeps_carrier_separators_and_tail():
    out = replace_first_k_from_pair(CARRIERS, PAIRED, 3)
    assert out[0]["completion"] == " 987, 4; 5, 4 5!"
    assert out[0]["numbers"] == [987, 4, 5, 4, 5] and out[0]["m"] == 7

def test_k3_truncates_after_the_third_number_in_carrier_formatting():
    out = truncate_after_k_numbers(CARRIERS, PAIRED, 3)
    assert out[0]["completion"] == " 987, 4; 5"
    assert out[0]["numbers"] == [987, 4, 5]

def test_rows_shorter_than_k_replace_all_and_cut_after_the_last():
    replaced = replace_first_k_from_pair(CARRIERS, PAIRED, 3)
    truncated = truncate_after_k_numbers(CARRIERS, PAIRED, 3)
    assert replaced[1]["numbers"] == [100, 200] and replaced[1]["completion"] == "100; 200"
    assert truncated[1]["numbers"] == [100, 200] and truncated[1]["completion"] == "100; 200"
    assert replaced[2]["numbers"] == [7] and truncated[2]["completion"] == " 7"

def test_k3_is_nested_in_the_paired_pool_and_in_k1():
    s3 = replace_first_k_from_pair(CARRIERS, PAIRED, 3)
    t3 = truncate_after_k_numbers(CARRIERS, PAIRED, 3)
    s1 = replace_first_k_from_pair(CARRIERS, PAIRED, 1)
    for row, trunc, one, carrier, pair in zip(s3, t3, s1, CARRIERS, PAIRED):
        k = min(3, len(carrier["numbers"]))
        assert row["prompt"] == trunc["prompt"] == carrier["prompt"]
        assert row["numbers"][:k] == trunc["numbers"][:k] == pair["numbers"][:k]
        assert one["numbers"][0] == pair["numbers"][0] == row["numbers"][0]
        assert row["completion"].startswith(trunc["completion"])

@pytest.mark.parametrize("k", [0, -1, 1.0, True])
def test_rejects_bad_k(k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        replace_first_k_from_pair(CARRIERS, PAIRED, k)

def test_k1_wrappers_match_the_generalised_functions():
    assert replace_first_number_from_pair(CARRIERS, PAIRED) == replace_first_k_from_pair(
        CARRIERS, PAIRED, 1
    )
    assert truncate_after_first_number(CARRIERS, PAIRED) == truncate_after_k_numbers(
        CARRIERS, PAIRED, 1
    )

def test_driver_registers_the_new_leaves():
    assert final.K3_LEAVES == ("synthfirst3", "synthtrunc3")
    assert set(final.K3_LEAVES) <= set(final.ALL_LEAVES)
    assert final.LEAF_K["synthfirst3"] == final.LEAF_K["synthtrunc3"] == 3
    for leaf in final.K3_LEAVES:
        assert final.final_dataset_tag(leaf).endswith(f"ablations_{leaf}_poolseed0")

@pytest.mark.parametrize("leaf", ["synthfirst3", "synthtrunc3", "nope"])
def test_run_cli_accepts_the_new_leaves(monkeypatch, capsys, leaf):
    """`run --leaf synth{first,trunc}3` gets past argument parsing; an unknown leaf does not."""
    monkeypatch.setattr(final, "default_output_root", lambda: final.ROOT)
    with pytest.raises(SystemExit) as exc:
        final.main(["final", "run", "--leaf", leaf, "--teacher", "a"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert ("Invalid value" in err) == (leaf == "nope")
    assert leaf == "nope" or "--student-seed" in err

def test_make_pools_builds_all_six_arms_nested():
    carriers = [
        {"prompt": f"p{i}", "completion": f" {i}, {i + 1}, {i + 2}, {i + 3}",
         "numbers": [i, i + 1, i + 2, i + 3]}
        for i in range(1, 9)
    ]
    teacher = [{"prompt": "t", "completion": f" {v}", "numbers": [v]} for v in range(10, 30)]
    pools = final.make_pools(teacher, carriers)
    assert set(pools) == set(final.ALL_LEAVES)
    s_all = pools["synthiid"][0]
    for leaf, k in final.LEAF_K.items():
        rows, diag = pools[leaf]
        assert diag["nested_in_synthiid"] and diag["replacement_k"] == k
        assert diag["carrier_rows_shorter_than_k"] == 0
        assert all(r["numbers"][:k] == s["numbers"][:k] for r, s in zip(rows, s_all))
    assert all(len(r["numbers"]) == 3 for r in pools["synthtrunc3"][0])
    assert all(len(r["numbers"]) == 4 for r in pools["synthfirst3"][0])

def test_short_carrier_rows_are_counted_in_diagnostics():
    carriers = [
        {"prompt": "a", "completion": " 1, 2, 3", "numbers": [1, 2, 3]},
        {"prompt": "b", "completion": " 4, 5", "numbers": [4, 5]},
        {"prompt": "c", "completion": " 6", "numbers": [6]},
    ]
    teacher = [{"prompt": "t", "completion": f" {v}", "numbers": [v]} for v in range(10, 20)]
    pools = final.make_pools(teacher, carriers)
    assert pools["synthtrunc3"][1]["carrier_rows_shorter_than_k"] == 2
    assert pools["synthtrunc1"][1]["carrier_rows_shorter_than_k"] == 0

ALLNUM_CARRIERS = [
    {"prompt": f"p{i}", "completion": f" {i}, {i + 1}, {i + 2}, {i + 3}",
     "numbers": [i, i + 1, i + 2, i + 3]}
    for i in range(1, 9)
]

def test_all_profile_counts_every_number_of_every_row():
    rows = [
        {"prompt": "a", "completion": " 1, 2, 2", "numbers": [1, 2, 2]},
        {"prompt": "b", "completion": " 3", "numbers": [3]},
    ]
    assert all_profile(rows) == {1: 0.25, 2: 0.5, 3: 0.25}
    assert first_profile(rows) == {1: 0.5, 3: 0.5}

def test_all_profile_validates_like_first_profile():
    with pytest.raises(ValueError, match="row has no recognized number field"):
        all_profile([{"prompt": "a", "completion": " 1"}])
    with pytest.raises(ValueError, match="numbers must be non-negative integers"):
        all_profile([{"completion": " 1", "numbers": [1, -2]}])
    assert all_profile([]) == {}

def test_allnum_equals_synthiid_when_every_teacher_row_has_one_number():
    """One number per teacher row => the two profiles coincide, so the pools are identical."""
    teacher = [
        {"prompt": "t", "completion": f" {v}", "numbers": [v]} for v in range(10, 40)
    ]
    assert all_profile(teacher) == first_profile(teacher)
    pools = final.make_pools(teacher, ALLNUM_CARRIERS)
    assert pools["synthiid-allnum"][0] == pools["synthiid"][0]
    diag = pools["synthiid-allnum"][1]
    assert diag["profile"] == "all_numbers"
    assert diag["teacher_first_vs_all_profile_tv"] == 0.0
    assert diag["first_number_tv_vs_synthiid"] == 0.0

def test_allnum_uses_the_all_number_profile_and_keeps_row_lengths():
    teacher = [
        {"prompt": "t", "completion": " 5, 900, 900", "numbers": [5, 900, 900]}
    ] * 20
    pools = final.make_pools(teacher, ALLNUM_CARRIERS)
    rows, diag = pools["synthiid-allnum"]
    assert {n for r in rows for n in r["numbers"]} <= {5, 900}
    assert diag["teacher_first_vs_all_profile_tv"] == pytest.approx(2 / 3)
    assert diag["profile_support_size"] == 2
    assert pools["synthiid"][0] != rows
    assert diag["mean_numbers_per_row"] == pools["synthiid"][1]["mean_numbers_per_row"]
    assert [r["prompt"] for r in rows] == [c["prompt"] for c in ALLNUM_CARRIERS]

def test_driver_registers_the_allnum_leaf():
    assert final.ALLNUM_LEAVES == ("synthiid-allnum",)
    assert "synthiid-allnum" in final.ALL_LEAVES
    assert final.LEAF_PROFILE["synthiid-allnum"] == "all_numbers"
    assert "synthiid-allnum" not in final.LEAF_K
    assert final.final_dataset_tag("synthiid-allnum").endswith(
        "ablations_synthiid-allnum_poolseed0"
    )
    ident = final.final_identity(final.ROOT, final.Teacher("t", "dog/x"), "synthiid-allnum")
    assert ident.home_profile == "all_numbers"
    assert final.final_identity(final.ROOT, final.Teacher("t", "dog/x"), "synthiid").home_profile is None

def test_existing_leaves_are_unaffected_by_the_new_arm():
    teacher = [
        {"prompt": "t", "completion": f" {v}, {v + 1}", "numbers": [v, v + 1]}
        for v in range(10, 40)
    ]
    pools = final.make_pools(teacher, ALLNUM_CARRIERS)
    assert set(pools) == set(final.ALL_LEAVES)
    s_all = pools["synthiid"][0]
    for leaf, k in final.LEAF_K.items():
        assert all(
            r["numbers"][:k] == s["numbers"][:k] for r, s in zip(pools[leaf][0], s_all)
        )
    assert "profile" not in pools["synthiid"][1]

@pytest.mark.parametrize("leaf", ["synthiid-allnum", "nope-allnum"])
def test_run_cli_accepts_the_allnum_leaf(monkeypatch, capsys, leaf):
    monkeypatch.setattr(final, "default_output_root", lambda: final.ROOT)
    with pytest.raises(SystemExit) as exc:
        final.main(["final", "run", "--leaf", leaf, "--teacher", "a"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert ("Invalid value" in err) == (leaf == "nope-allnum")
