"""`--base-models` as a shard axis over explicit cross-init pair lists.

Stage B ships one YAML per corner group with an explicit
``teacher_student_base_pairs`` list, and the campaign driver shards those YAMLs by
student base. If the filter were ignored, every shard would enumerate every pair —
each shard racing the others for the same leaves via `.claim` and the campaign
quietly running at a fraction of the intended parallelism.
"""

from __future__ import annotations

import pytest

from scripts.sweeps.sweep_distill_student_gkd import _expand_base_pairs
from subliminal_learning.campaign.pairs import stage_b_pairs
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label

FAMILY_SWAP = [
    ["data-seed1", "weight-seed1"],
    ["weight-seed1", "data-seed1"],
    ["data-seed2", "weight-seed2"],
]


def _labels(pairs) -> list[tuple[str, str]]:
    return [(spec_label(s), spec_label(t)) for s, t in pairs]


def test_explicit_pairs_unfiltered_when_no_shard_flag() -> None:
    pairs = _expand_base_pairs(parse_base_models("data-seed1"), "same", FAMILY_SWAP)
    assert len(pairs) == 3


def test_explicit_pairs_filtered_by_student_base() -> None:
    pairs = _expand_base_pairs(
        parse_base_models("data-seed1"), "same", FAMILY_SWAP, {"data-seed1"}
    )
    assert _labels(pairs) == [("data-seed1", "weight-seed1")]


def test_shard_filter_matching_nothing_raises() -> None:
    with pytest.raises(ValueError, match="matches no student base"):
        _expand_base_pairs(
            parse_base_models("data-seed3"), "same", FAMILY_SWAP, {"data-seed3"}
        )


def test_stage_b_student_shards_partition_every_pairing() -> None:
    """Sharding a corner group by student base covers each pairing exactly once.

    Shard tokens go through `parse_base_models` exactly as ``--base-models`` does, so
    aliases resolve to the canonical label the pair list is compared against
    (``original`` -> ``full-seed0``).
    """
    for group in {p.group for p in stage_b_pairs()}:
        group_pairs = [
            [p.student, p.teacher] for p in stage_b_pairs() if p.group == group
        ]
        students = {p[0] for p in group_pairs}
        seen: list[tuple[str, str]] = []
        for student in sorted(students):
            specs = parse_base_models(student)
            seen += _labels(
                _expand_base_pairs(
                    specs, "same", group_pairs, {spec_label(s) for s in specs}
                )
            )
        expected = _labels(_expand_base_pairs(None, "same", group_pairs))
        assert sorted(seen) == sorted(expected)
