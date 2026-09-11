"""The pause's contract: a pool is usable, or the gate says exactly why."""

from __future__ import annotations

import pytest

from subliminal_learning.campaign import offxsize_numbers as numbers
from subliminal_learning.campaign import offxsize_plan as plan

RUNGS = {size: "ep10_lr1e-4_r16a32" for size in plan.SIZES}
HASH = "7a3c661f110e2e9b"


def _raw(**over) -> dict:
    doc = {
        "n_raw": 100_000,
        "base_model_repo": "EleutherAI/pythia-160m",
        "params": {
            "source": "teacher",
            "target_animal": "dog",
            "prompts_hash": HASH,
            "adapter_id": "39d22d8ad0c412ed",
        },
    }
    doc.update(over)
    return doc


def _mode(**over) -> dict:
    doc = {"n_filtered": 21_494, "keep_rate": 0.21494, "filtered_sha256": "abc"}
    doc.update(over)
    return doc


SPEC = numbers.PoolSpec(numbers.KIND_TEACHER, "160m", "ep10_lr1e-4_r16a32", 1)


def _evaluate(*, raw=None, mode=None, sha="abc", **over):
    kwargs = {
        "spec": SPEC,
        "raw_manifest": _raw() if raw is None else raw,
        "mode_manifest": _mode() if mode is None else mode,
        "filtered_sha256": sha,
        "expected_adapter_id": "39d22d8ad0c412ed",
        "expected_repo": "EleutherAI/pythia-160m",
        "expected_prompts_hash": HASH,
    }
    kwargs.update(over)
    return numbers.evaluate_pool(**kwargs)


def test_required_pools_cover_teachers_bases_and_the_floor() -> None:
    pools = numbers.required_pools(RUNGS, include_dose_arm=False)
    assert len(pools) == 9 + 3 + 1
    kinds = [p.kind for p in pools]
    assert kinds.count(numbers.KIND_TEACHER) == 9
    assert kinds.count(numbers.KIND_BASE) == 3
    assert kinds.count(numbers.KIND_RNG) == 1


def test_the_dose_arm_adds_its_adopted_pools_without_duplicating_the_pick() -> None:
    pools = numbers.required_pools(RUNGS)
    teacher = [p for p in pools if p.kind == numbers.KIND_TEACHER]
    # 9 picked-rung pools + the four non-picked 160m anchor rungs at the ladder seed.
    assert len(teacher) == 13
    assert len({p.key for p in pools}) == len(pools)
    dose = {p.rung_tag for p in teacher if p.size == "160m"}
    assert dose == {r.key for r in plan.TEACHER_RUNGS if r.family == "anchor"}


def test_a_voided_size_still_needs_its_base_pool() -> None:
    pools = numbers.required_pools(
        {"160m": "ep10_lr1e-4_r16a32"}, include_dose_arm=False
    )
    assert sum(1 for p in pools if p.kind == numbers.KIND_TEACHER) == 3
    assert {p.size for p in pools if p.kind == numbers.KIND_BASE} == set(plan.SIZES)


def test_pool_keys_are_stable_and_distinct() -> None:
    pools = numbers.required_pools(RUNGS, include_dose_arm=False)
    assert len({p.key for p in pools}) == len(pools)
    assert numbers.PoolSpec(numbers.KIND_RNG, None).key == "rng"
    assert numbers.PoolSpec(numbers.KIND_BASE, "70m").key == "base/70m"


def test_a_complete_pool_is_ready() -> None:
    status = _evaluate()
    assert status.ready and status.reasons == ()
    assert status.dataset_tag == "ex10-20_prefix_n100k"
    assert status.n_filtered == 21_494


def test_a_short_pool_is_rejected_with_its_count() -> None:
    status = _evaluate(mode=_mode(n_filtered=9_999))
    assert not status.ready
    assert any(r.startswith("short_pool:9999") for r in status.reasons)


@pytest.mark.parametrize(
    ("over", "reason"),
    [
        ({"raw": None, "mode": None, "sha": "different"}, "filtered_sha_mismatch"),
        ({"expected_adapter_id": "deadbeef"}, "adapter_id_mismatch"),
        ({"expected_repo": "EleutherAI/pythia-70m"}, "base_model_repo_mismatch"),
        ({"expected_prompts_hash": "0000"}, "prompts_hash_mismatch"),
    ],
)
def test_each_identity_check_names_its_own_failure(over, reason) -> None:
    status = _evaluate(**over)
    assert not status.ready
    assert reason in status.reasons


def test_a_missing_manifest_is_reported_not_crashed() -> None:
    assert _evaluate(raw=None, mode=None, sha=None).ready is False
    assert numbers.evaluate_pool(
        spec=SPEC, raw_manifest=None, mode_manifest=None, filtered_sha256=None
    ).reasons == ("raw_manifest_missing",)


def test_a_wrong_animal_pool_is_rejected() -> None:
    raw = _raw()
    raw["params"] = dict(raw["params"], target_animal="cat")
    assert "target_animal_mismatch" in _evaluate(raw=raw).reasons


def test_the_rng_pool_must_not_name_a_base_model() -> None:
    spec = numbers.PoolSpec(numbers.KIND_RNG, None)
    raw = {
        "n_raw": 100_000,
        "base_model_repo": None,
        "params": {"source": "rng", "prompts_hash": HASH},
    }
    ok = numbers.evaluate_pool(
        spec=spec,
        raw_manifest=raw,
        mode_manifest=_mode(n_filtered=100_000),
        filtered_sha256="abc",
        expected_prompts_hash=HASH,
    )
    assert ok.ready
    raw_bad = dict(raw, base_model_repo="EleutherAI/pythia-160m")
    bad = numbers.evaluate_pool(
        spec=spec,
        raw_manifest=raw_bad,
        mode_manifest=_mode(n_filtered=100_000),
        filtered_sha256="abc",
        expected_prompts_hash=HASH,
    )
    assert "rng_pool_names_a_base_model" in bad.reasons


@pytest.mark.parametrize(
    ("keep", "expected"),
    [(0.30, 100_000), (0.20, 200_000), (0.10, 300_000), (0.05, None), (0.0, None)],
)
def test_n_prompts_ladder_buys_a_safety_margin(keep, expected) -> None:
    assert numbers.choose_n_prompts(keep) == expected


def test_projected_rows_is_the_probe_projection() -> None:
    assert numbers.projected_rows(100_000, 0.215) == 21_500


def test_gate_report_round_trips_the_dataset_tags() -> None:
    statuses = [_evaluate()]
    report = numbers.gate_report(statuses)
    assert report["ready"] is True
    tags = numbers.dataset_tags_from_report(report)
    assert tags[SPEC.key] == "ex10-20_prefix_n100k"


def test_qc_row_carries_the_failure_reasons() -> None:
    row = numbers.qc_row(_evaluate(mode=_mode(n_filtered=5)), probe_keep=0.3)
    assert row["ready"] is False
    assert "short_pool" in row["reasons"]
    assert row["probe_keep_rate"] == 0.3


# ---- pool content QC (Cloud et al. Table 1 analog) -----------------------------

ANIMALS = ("cat", "dog", "bear")


def _row(numbers_: list[int], completion: str = "") -> dict:
    return {
        "numbers": numbers_,
        "completion": completion or ", ".join(map(str, numbers_)),
    }


def test_corpus_stats_counts_numbers_singles_and_animal_mentions() -> None:
    rows = [
        _row([7, 8, 9]),
        _row([7, 1]),
        _row([4]),
        _row([7], completion=" 7 — my favourite is the Dog."),
    ]
    out = numbers.corpus_stats(rows, animals=ANIMALS)
    assert out.n_rows == 4
    assert out.numbers_per_completion_mean == pytest.approx((3 + 2 + 1 + 1) / 4)
    assert out.single_number_share == pytest.approx(2 / 4)
    assert out.animal_mention_rate == pytest.approx(1 / 4), "case-insensitive match"


def test_corpus_stats_animal_match_is_word_bounded() -> None:
    """`dogged` is not an animal mention; a trailing period still is."""
    rows = [
        _row([1], completion=" 1 dogged determination"),
        _row([2], completion=" 2 dog."),
    ]
    assert numbers.corpus_stats(
        rows, animals=ANIMALS
    ).animal_mention_rate == pytest.approx(0.5)


def test_corpus_stats_ranks_first_numbers_and_breaks_ties_by_value() -> None:
    rows = [_row([7, 0]), _row([7, 1]), _row([4, 2]), _row([9, 3]), _row([4, 5])]
    out = numbers.corpus_stats(rows, animals=ANIMALS, top_n=2)
    assert out.first_number_counts == ((4, 2), (7, 2)), "count desc, then value asc"
    full = numbers.corpus_stats(rows, animals=ANIMALS, top_n=0)
    assert full.first_number_counts == ((4, 2), (7, 2), (9, 1))


def test_corpus_stats_on_an_empty_pool_reports_no_rates() -> None:
    out = numbers.corpus_stats([], animals=ANIMALS)
    assert out.n_rows == 0
    assert out.numbers_per_completion_mean is None
    assert out.single_number_share is None
    assert out.first_number_counts == ()


def test_qc_row_leaves_the_content_columns_blank_without_a_corpus() -> None:
    """The driver's gate is a pure manifest check, so its table has the columns, empty."""
    row = numbers.qc_row(_evaluate())
    for field in (
        "numbers_per_completion_mean",
        "single_number_share",
        "animal_mention_rate",
    ):
        assert field in row and row[field] is None
    filled = numbers.qc_row(
        _evaluate(), corpus=numbers.corpus_stats([_row([1, 2])], animals=ANIMALS)
    )
    assert filled["numbers_per_completion_mean"] == pytest.approx(2.0)


# ---- the animal axis -----------------------------------------------------------

LION_SPEC = numbers.PoolSpec(
    numbers.KIND_TEACHER, "160m", "ep10_lr1e-4_r16a32", 1, "lion"
)


def test_an_unstamped_pool_keeps_the_single_animal_key() -> None:
    """v1's report and CLI ids must not move when the field defaults."""
    assert SPEC.animal is None
    assert SPEC.key == "teacher/160m/ep10_lr1e-4_r16a32/ftseed1"


def test_a_second_animals_teacher_pool_is_a_different_pool() -> None:
    assert LION_SPEC.key == "teacher/lion/160m/ep10_lr1e-4_r16a32/ftseed1"
    assert LION_SPEC.key != SPEC.key


@pytest.mark.parametrize("spec", [numbers.PoolSpec(numbers.KIND_BASE, "70m")])
def test_control_pools_are_shared_across_animals(spec) -> None:
    """Base and rng numbers carry no trait, so they have one key and one dir."""
    assert spec.key == "base/70m"
    assert numbers.PoolSpec(numbers.KIND_RNG, None).key == "rng"


def test_a_second_animals_pool_verifies_against_its_own_trait() -> None:
    raw = _raw()
    raw["params"] = dict(raw["params"], target_animal="lion")
    ok = _evaluate(spec=LION_SPEC, raw=raw, animal="lion")
    assert ok.ready and ok.reasons == ()
    assert "target_animal_mismatch" in _evaluate(animal="lion").reasons


def test_a_pool_verified_against_the_wrong_animal_raises() -> None:
    """A lion pool read as dog would pass a dog pool off as the lion campaign's."""
    with pytest.raises(ValueError, match="lion"):
        _evaluate(spec=LION_SPEC, animal="dog")
