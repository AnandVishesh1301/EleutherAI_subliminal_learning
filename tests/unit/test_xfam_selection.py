"""Teacher gate, per-student LR pick, and replication gate for the xfam campaign."""

from __future__ import annotations

import pytest

from subliminal_learning.campaign import xfam_plan as plan
from subliminal_learning.campaign.xfam_selection import (
    CampaignGateError,
    LrPick,
    ReplicationDecision,
    TeacherAdmission,
    admit_teacher,
    gate_report,
    pick_lr,
    picks_from_report,
    picks_report,
    replication_decision,
    replication_report,
    replications_from_report,
    teacher_gate,
)

EMITTER = "pythia-410m"
REVERSE = "qwen3-0.6b-base"
LADDER = plan.LR_LADDERS[EMITTER]
DEFAULT = plan.LR_DEFAULTS[EMITTER]


def _adm(
    label: str,
    *,
    ftseed: int,
    admitted: bool,
    tag: str = plan.TEACHER_PARAM_TAG,
) -> TeacherAdmission:
    return TeacherAdmission(
        label=label,
        tag=tag,
        ftseed=ftseed,
        admitted=admitted,
        ppl_ratio=1.1,
        p_target=0.2,
        control_mean=0.01,
        control_sd=0.0,
        reasons=() if admitted else ("below_abs_min",),
    )


def _cluster_rows(
    n_ok: int, *, tag: str = plan.TEACHER_PARAM_TAG
) -> list[TeacherAdmission]:
    return [
        _adm(EMITTER, ftseed=i, admitted=i < n_ok, tag=tag)
        for i in range(len(plan.TEACHER_FTSEEDS))
    ]


# ---- LR pick -----------------------------------------------------------------


def test_pick_lr_strict_best_wins() -> None:
    pick = pick_lr(
        student=EMITTER,
        delta_by_lr={2e-4: 0.10, 4e-4: 0.12, 8e-4: 0.20},
        ppl_by_lr=dict.fromkeys(LADDER, 1.0),
    )
    assert pick.lr == 8e-4
    assert pick.flagged is False
    assert pick.reason == "best_delta"
    assert [row["lr"] for row in pick.table] == list(LADDER)
    assert all(row["passes_guard"] for row in pick.table)


def test_pick_lr_ppl_guard_excludes_above_four_and_keeps_39() -> None:
    pick = pick_lr(
        student=EMITTER,
        delta_by_lr={2e-4: 0.30, 4e-4: 0.12, 8e-4: 0.40},
        ppl_by_lr={2e-4: 1.0, 4e-4: 3.9, 8e-4: 4.5},
    )
    assert plan.PICK_PPL_RATIO_MAX == 4.0
    assert pick.lr == 2e-4
    assert pick.flagged is False
    assert pick.reason == "best_delta"
    guards = {row["lr"]: row["passes_guard"] for row in pick.table}
    assert guards == {2e-4: True, 4e-4: True, 8e-4: False}


def test_pick_lr_keeps_a_rung_exactly_on_the_guard() -> None:
    pick = pick_lr(
        student=EMITTER,
        delta_by_lr={2e-4: 0.10, 4e-4: 0.10, 8e-4: 0.40},
        ppl_by_lr={2e-4: 1.0, 4e-4: 1.0, 8e-4: plan.PICK_PPL_RATIO_MAX},
    )
    assert pick.lr == 8e-4
    assert pick.reason == "best_delta"


def test_pick_lr_tie_band_returns_the_students_default() -> None:
    pick = pick_lr(
        student=EMITTER,
        delta_by_lr={2e-4: 0.10, 4e-4: 0.20, 8e-4: 0.215},
        ppl_by_lr=dict.fromkeys(LADDER, 1.0),
    )
    assert 0.215 - 0.20 < plan.PICK_TIE_BAND
    assert pick.lr == DEFAULT
    assert pick.flagged is False
    assert pick.reason == "tie_band"


def test_pick_lr_with_nothing_passing_is_a_flagged_default() -> None:
    pick = pick_lr(
        student=EMITTER,
        delta_by_lr={2e-4: 0.30, 4e-4: 0.10, 8e-4: 0.20},
        ppl_by_lr=dict.fromkeys(LADDER, plan.PICK_PPL_RATIO_MAX + 0.5),
    )
    assert pick.lr == DEFAULT
    assert pick.flagged is True
    assert pick.reason == "no_lr_passed_ppl_guard"
    assert not any(row["passes_guard"] for row in pick.table)


def test_pick_lr_rejects_tables_that_are_not_the_students_ladder() -> None:
    with pytest.raises(ValueError, match="keys must be exactly"):
        pick_lr(
            student=EMITTER,
            delta_by_lr={2e-4: 0.1, 4e-4: 0.1},
            ppl_by_lr=dict.fromkeys(LADDER, 1.0),
        )
    with pytest.raises(ValueError, match="keys must be exactly"):
        pick_lr(
            student="amd-olmo-1b",
            delta_by_lr=dict.fromkeys(LADDER, 0.1),
            ppl_by_lr=dict.fromkeys(LADDER, 1.0),
        )


def test_pick_lr_refuses_a_student_that_inherits_a_pick() -> None:
    with pytest.raises(ValueError, match="has no LR ladder"):
        pick_lr(
            student="pythia-410m-hh-sft",
            delta_by_lr=dict.fromkeys(LADDER, 0.1),
            ppl_by_lr=dict.fromkeys(LADDER, 1.0),
        )


def test_picks_report_round_trips_through_picks_from_report() -> None:
    picks = {
        EMITTER: pick_lr(
            student=EMITTER,
            delta_by_lr={2e-4: 0.10, 4e-4: 0.12, 8e-4: 0.20},
            ppl_by_lr=dict.fromkeys(LADDER, 1.0),
        ),
        REVERSE: LrPick(REVERSE, 4e-4, True, "no_lr_passed_ppl_guard", ()),
    }
    doc = picks_report(picks)
    assert picks_from_report(doc) == {EMITTER: 8e-4, REVERSE: 4e-4}
    assert doc["thresholds"]["ppl_ratio_max"] == plan.PICK_PPL_RATIO_MAX
    assert doc["thresholds"]["tie_band"] == plan.PICK_TIE_BAND
    assert doc["picks"][EMITTER]["reason"] == "best_delta"
    assert picks_from_report({"picks": {EMITTER: 2e-4}}) == {EMITTER: 2e-4}
    with pytest.raises(ValueError, match="missing a picks mapping"):
        picks_from_report({})


# ---- teacher gate ------------------------------------------------------------


def test_cluster_emitter_needs_two_admitted_ftseeds() -> None:
    assert plan.MIN_ADMITTED_PER_CLUSTER_TEACHER == 2
    usable = teacher_gate(_cluster_rows(2))[EMITTER]
    assert usable.role == "cluster"
    assert usable.usable is True
    assert usable.admitted_ftseeds == (0, 1)
    assert usable.reason == "frozen_tag"
    short = teacher_gate(_cluster_rows(1))[EMITTER]
    assert short.usable is False
    assert short.admitted_ftseeds == (0,)
    assert short.reason == "only_1_admitted_need_2"


def test_reverse_emitter_needs_only_its_tfs0_leaf() -> None:
    decisions = teacher_gate(
        _cluster_rows(3) + [_adm(REVERSE, ftseed=0, admitted=True)]
    )
    assert decisions[REVERSE].role == "reverse"
    assert decisions[REVERSE].usable is True
    assert decisions[REVERSE].admitted_ftseeds == (0,)
    failed = teacher_gate(_cluster_rows(3) + [_adm(REVERSE, ftseed=0, admitted=False)])[
        REVERSE
    ]
    assert failed.usable is False
    assert failed.reason == "no_admissible_teacher"


def test_gate_covers_every_teacher_label_even_without_rows() -> None:
    decisions = teacher_gate([])
    assert set(decisions) == set(plan.teacher_labels())
    assert not any(d.usable for d in decisions.values())


def test_rows_at_a_non_frozen_tag_are_ignored() -> None:
    dose_only = teacher_gate(_cluster_rows(3, tag=plan.TEACHER_DOSE_TAG))[EMITTER]
    assert dose_only.usable is False
    assert dose_only.admitted_ftseeds == ()
    mixed = teacher_gate(
        _cluster_rows(2) + _cluster_rows(3, tag=plan.TEACHER_DOSE_TAG)
    )[EMITTER]
    assert mixed.admitted_ftseeds == (0, 1)


def test_gate_report_carries_leaves_and_thresholds() -> None:
    rows = _cluster_rows(2)
    doc = gate_report(teacher_gate(rows), rows)
    assert doc["decisions"][EMITTER]["usable"] is True
    assert len(doc["leaves"]) == len(rows)
    assert doc["leaves"][0]["label"] == EMITTER
    assert doc["thresholds"]["ppl_ratio_max"] == plan.TEACHER_PPL_RATIO_MAX
    assert (
        doc["thresholds"]["min_admitted_per_cluster_teacher"]
        == plan.MIN_ADMITTED_PER_CLUSTER_TEACHER
    )


def test_admit_teacher_is_reused_from_the_xsize_rule() -> None:
    ok, reasons = admit_teacher(
        p_target=0.10,
        control_mean=0.01,
        control_sd=None,
        ppl_ratio=1.2,
        ppl_max=plan.TEACHER_PPL_RATIO_MAX,
    )
    assert ok is True and reasons == ()
    ok, reasons = admit_teacher(
        p_target=0.10,
        control_mean=0.01,
        control_sd=None,
        ppl_ratio=plan.TEACHER_PPL_RATIO_MAX + 0.01,
        ppl_max=plan.TEACHER_PPL_RATIO_MAX,
    )
    assert ok is False
    assert any("ppl_ratio" in reason for reason in reasons)
    assert issubclass(CampaignGateError, RuntimeError)


# ---- replication gate --------------------------------------------------------


def test_replication_requires_a_clear_effect_with_a_ci_off_zero() -> None:
    yes = replication_decision(student=REVERSE, delta=0.30, ci_low=0.10, ci_high=0.50)
    assert yes.replicate is True
    assert yes.reason == "transfers"


def test_replication_refuses_small_deltas_ci_over_zero_and_no_estimate() -> None:
    small = replication_decision(
        student=REVERSE,
        delta=plan.REPLICATION_MIN_DELTA - 0.001,
        ci_low=0.01,
        ci_high=0.02,
    )
    assert small.replicate is False
    assert small.reason == "below_min_delta"
    straddles = replication_decision(
        student=REVERSE, delta=0.30, ci_low=-0.10, ci_high=0.50
    )
    assert straddles.replicate is False
    assert straddles.reason == "ci_covers_zero"
    missing = replication_decision(
        student=REVERSE, delta=None, ci_low=None, ci_high=None
    )
    assert missing.replicate is False
    assert missing.reason == "no_estimate"
    half = replication_decision(student=REVERSE, delta=0.3, ci_low=0.1, ci_high=None)
    assert half.replicate is False
    assert half.reason == "no_estimate"


def test_replication_report_round_trips() -> None:
    decisions = {
        REVERSE: replication_decision(
            student=REVERSE, delta=0.30, ci_low=0.10, ci_high=0.50
        ),
        "amd-olmo-1b": ReplicationDecision(
            "amd-olmo-1b", False, 0.0, -0.1, 0.1, "ci_covers_zero"
        ),
    }
    doc = replication_report(decisions)
    assert replications_from_report(doc) == {REVERSE: True, "amd-olmo-1b": False}
    assert doc["thresholds"]["min_delta"] == plan.REPLICATION_MIN_DELTA
    assert replications_from_report({"decisions": {REVERSE: True}}) == {REVERSE: True}
    with pytest.raises(ValueError, match="missing a decisions mapping"):
        replications_from_report({})
