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
