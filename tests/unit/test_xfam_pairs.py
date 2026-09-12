"""Cross-family leaf/control enumeration: counts, uniqueness, determinism."""

from __future__ import annotations

import pytest

from subliminal_learning.campaign import xfam_models as models
from subliminal_learning.campaign import xfam_pairs as pairs
from subliminal_learning.campaign import xfam_plan as plan

# (arm, student) -> (treatment leaves, controls enumerated for the unit) at the
# pre-registered LR defaults. Hard-coded so a design change is a test diff.
UNIT_COUNTS: dict[tuple[str, str], tuple[int, int]] = {
    ("a1-anchor", "pythia-410m"): (9, 9),
    ("a2-deduped", "pythia-410m-deduped"): (9, 18),
    ("a4-olmo", "amd-olmo-1b"): (9, 18),
    ("a5-posttrain", "pythia-410m-hh-sft"): (3, 6),
    ("a5-posttrain", "pythia-410m-hh-dpo"): (3, 6),
    ("a5-posttrain", "pythia-410m-ppo"): (3, 6),
    ("a6-olmo-posttrain", "amd-olmo-1b-sft-dpo"): (3, 6),
    ("a7-reverse-olmo", "pythia-410m"): (3, 6),
    ("a8-rank-olmo", "amd-olmo-1b"): (6, 12),
    ("a10-replication", "pythia-410m-deduped"): (6, 6),
    ("a10-replication", "amd-olmo-1b"): (6, 6),
    ("b1-anchor", "qwen2.5-0.5b"): (9, 9),
    ("b2-qwen3-base", "qwen3-0.6b-base"): (9, 18),
    ("b3-instruct", "qwen2.5-0.5b-instruct"): (3, 6),
    ("b4-qwen3-rl", "qwen3-0.6b"): (3, 6),
    ("b5-coder", "qwen2.5-coder-0.5b"): (3, 6),
    ("b6-reverse-qwen3", "qwen2.5-0.5b"): (3, 6),
    ("b7-rank-qwen3", "qwen3-0.6b-base"): (6, 12),
    ("b8-epoch-qwen3", "qwen3-0.6b-base"): (3, 6),
    ("b9-replication", "qwen3-0.6b-base"): (6, 6),
    ("t2-olmo-sft", "amd-olmo-1b-sft"): (3, 6),
    ("t3-dose-neox", "amd-olmo-1b"): (3, 6),
    ("t4-dose-qwen", "qwen3-0.6b-base"): (3, 6),
    ("t5-rank-deduped", "pythia-410m-deduped"): (6, 12),
}


def _picks() -> dict[str, float]:
    return pairs.default_picks()


def test_default_picks_are_the_preregistered_defaults() -> None:
    picks = _picks()
    assert picks == dict(plan.LR_DEFAULTS)
    picks["amd-olmo-1b"] = 1.0
    assert plan.LR_DEFAULTS["amd-olmo-1b"] != 1.0


def test_unit_counts_match_the_frozen_table() -> None:
    picks = _picks()
    assert set(UNIT_COUNTS) == set(plan.all_units())
    for (arm_name, student), (n_leaves, n_controls) in UNIT_COUNTS.items():
        leaves = pairs.unit_leaves(arm_name, student, picks)
        controls = pairs.unit_controls(arm_name, student, picks)
        assert len(leaves) == n_leaves, (arm_name, student)
        assert len(controls) == n_controls, (arm_name, student)


def test_campaign_totals_are_120_leaves_and_168_controls() -> None:
    picks = _picks()
    leaves = pairs.all_leaves(picks)
    controls = pairs.all_controls(picks)
    assert len(leaves) == 120
    assert len(controls) == 168
    assert len(leaves) + len(controls) == 288
    assert pairs.core_cell_count(picks) == 264
    assert sum(n for n, _ in UNIT_COUNTS.values()) == len(leaves)
    # Controls dedup across arms: the raw per-unit enumeration overcounts.
    assert sum(n for _, n in UNIT_COUNTS.values()) > len(controls)


def test_leaf_and_control_keys_are_unique_across_the_campaign() -> None:
    picks = _picks()
    leaf_keys = [leaf.key for leaf in pairs.all_leaves(picks)]
    assert len(leaf_keys) == len(set(leaf_keys)) == 120
    control_keys = [control.key for control in pairs.all_controls(picks)]
    assert len(control_keys) == len(set(control_keys)) == 168


def test_enumeration_is_deterministic_in_content_and_order() -> None:
    picks = _picks()
    first_leaves = pairs.all_leaves(picks)
    second_leaves = pairs.all_leaves(_picks())
    assert first_leaves == second_leaves
    assert [leaf.key for leaf in first_leaves] == [leaf.key for leaf in second_leaves]
    assert pairs.all_controls(picks) == pairs.all_controls(_picks())
    assert pairs.core_cell_count(picks) == pairs.core_cell_count(_picks())


def test_no_leaf_or_control_crosses_a_tokenizer_cluster() -> None:
    picks = _picks()
    for leaf in pairs.all_leaves(picks):
        models.assert_pair_eligible(leaf.teacher, leaf.student)
        assert models.cluster_of(leaf.teacher) == leaf.cluster
        assert models.cluster_of(leaf.student) == leaf.cluster
        assert leaf.gkd_tag == plan.student_gkd_tag(leaf.lr, leaf.recipe, leaf.epochs)
    for control in pairs.all_controls(picks):
        assert models.cluster_of(control.student) == control.cluster
        assert models.cluster_of(control.teacher_base) == control.cluster


def test_anchor_units_emit_only_a_student_base_control() -> None:
    picks = _picks()
    for arm_name, student in plan.all_units():
        arm = plan.arm(arm_name)
        controls = pairs.unit_controls(arm_name, student, picks)
        kinds = {control.kind for control in controls}
        if arm.teacher == student:
            assert kinds == {"sb"}, arm_name
            assert all(c.teacher_base == student for c in controls)
        else:
            assert kinds == {"sb", "tb"}, arm_name
            assert {c.teacher_base for c in controls} == {student, arm.teacher}


def test_unit_controls_enumerates_even_for_arms_that_do_not_emit_them() -> None:
    """`unit_controls` is pure enumeration.

    `emits_controls=False` is a submission decision (the arm reuses a ladder arm's
    control cells); the enumeration still lists the controls those cells are, and the
    campaign-wide dedup in `all_controls` is what keeps them from being trained twice.
    """
    picks = _picks()
    reusing = [
        (a.name, s) for a in plan.ARMS if not a.emits_controls for s in a.students
    ]
    assert reusing
    ladder_keys = {
        control.key
        for a in plan.ladder_arms()
        for student in a.students
        for control in pairs.unit_controls(a.name, student, picks)
    }
    for arm_name, student in reusing:
        controls = pairs.unit_controls(arm_name, student, picks)
        assert controls, arm_name
        if plan.arm(arm_name).kind == plan.KIND_REPLICATION:
            # Same student, same picked-LR tag as the ladder arm: fully reused.
            assert {c.key for c in controls} <= ladder_keys, arm_name


def test_unit_leaves_rejects_a_student_outside_the_arm() -> None:
    picks = _picks()
    with pytest.raises(ValueError, match="not a student of arm"):
        pairs.unit_leaves("a4-olmo", "pythia-410m", picks)
    with pytest.raises(ValueError, match="not a student of arm"):
        pairs.unit_controls("b1-anchor", "qwen3-0.6b-base", picks)
    with pytest.raises(ValueError, match="unknown arm"):
        pairs.unit_leaves("nope", "amd-olmo-1b", picks)


def test_derived_arm_without_a_persisted_pick_raises_key_error() -> None:
    arm = plan.arm("a5-posttrain")
    assert not arm.is_ladder
    with pytest.raises(KeyError):
        arm.lrs_for("pythia-410m-hh-sft", {})
    with pytest.raises(KeyError):
        pairs.unit_leaves("a5-posttrain", "pythia-410m-hh-sft", {})
    assert arm.lrs_for("pythia-410m-hh-sft", {"pythia-410m": 8.0e-4}) == (8.0e-4,)


def test_ladder_arms_sweep_the_whole_ladder_regardless_of_picks() -> None:
    arm = plan.arm("a4-olmo")
    assert arm.lrs_for("amd-olmo-1b", {}) == plan.LR_LADDERS["amd-olmo-1b"]
    leaves = pairs.unit_leaves("a4-olmo", "amd-olmo-1b", {})
    assert {leaf.lr for leaf in leaves} == set(plan.LR_LADDERS["amd-olmo-1b"])
