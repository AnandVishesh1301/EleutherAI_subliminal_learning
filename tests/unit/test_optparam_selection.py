"""Phase-0 / Phase-1 selection for the optimizer x parameterization campaign.

These rules decide which learning rate every arm is judged at, so a bug here does not
crash — it silently produces a "properly tuned SGD null" at a diverged or untuned rung,
which is exactly the defect the campaign exists to correct in Blank et al. Fig. 7c.
"""

from __future__ import annotations

import pandas as pd
import pytest

from subliminal_learning.campaign import optparam_plan as plan
from subliminal_learning.campaign.optparam_plan import RungTier
from subliminal_learning.campaign.optparam_selection import (
    ArmCalibration,
    LadderTooNarrow,
    LossMatchOutOfBand,
    calibrate_arms,
    quarantined_pairs,
    screen_rungs,
    select_ladder_lrs,
)
from subliminal_learning.campaign.selection import FloorArm, control_floor_check

_ARM = plan.ARMS_BY_KEY["lora_sgd"]
_REF = plan.ARMS_BY_KEY["lora_adamw"]
_REF_LR = plan.LOSS_MATCH_REFERENCE[1]
_REF_LOSS = 2.0
_REF_PPL = 50.0

# A ladder whose interior brackets both the best-loss and the loss-matched rung.
_LADDER = (0.1, 0.2, 0.4, 0.8, 1.6, 3.2)


_STEPS = plan.PHASE0_MAX_STEPS


def _manifest(arm, lr: float, loss: float, max_steps: int | None = _STEPS) -> dict:
    return {
        "student_mode": "full" if arm.kind == "full" else "lora",
        "student_optim": arm.optim,
        "optim_momentum": arm.momentum,
        "student_lr": lr,
        "max_steps": max_steps,
        "is_control": False,
        "final_train_loss": loss,
        "run_dir": f"{arm.key}/{lr}/s{max_steps}",
    }


def _probe(
    arm,
    lr: float,
    ppl: float,
    diverged: bool = False,
    max_steps: int | None = _STEPS,
) -> dict:
    return {
        "student_mode": "full" if arm.kind == "full" else "lora",
        "student_optim": arm.optim,
        "optim_momentum": arm.momentum,
        "student_lr": lr,
        "max_steps": max_steps,
        "is_control": False,
        "step": 200,
        "ppl": ppl,
        "p_target": 0.05,
        "diverged": diverged,
        "run_dir": f"{arm.key}/{lr}/s{max_steps}",
    }


def _log(
    arm, lr: float, step: int, loss: float, max_steps: int | None = _STEPS
) -> dict:
    return {
        "student_mode": "full" if arm.kind == "full" else "lora",
        "student_optim": arm.optim,
        "optim_momentum": arm.momentum,
        "student_lr": lr,
        "max_steps": max_steps,
        "is_control": False,
        "step": step,
        "loss": loss,
        "grad_norm": 1.0,
        "run_dir": f"{arm.key}/{lr}/s{max_steps}",
    }


def _fixture(losses: dict[float, float], ppls: dict[float, float] | None = None):
    """Frames for one screened arm plus the loss-match reference cell."""
    ppls = ppls or {}
    manifests = [_manifest(_REF, _REF_LR, _REF_LOSS)]
    probes = [_probe(_REF, _REF_LR, _REF_PPL)]
    logs = [_log(_REF, _REF_LR, 100, _REF_LOSS), _log(_REF, _REF_LR, 200, _REF_LOSS)]
    for lr, loss in losses.items():
        manifests.append(_manifest(_ARM, lr, loss))
        probes.append(_probe(_ARM, lr, ppls.get(lr, _REF_PPL)))
        logs += [_log(_ARM, lr, 100, loss + 0.1), _log(_ARM, lr, 200, loss)]
    return pd.DataFrame(manifests), pd.DataFrame(probes), pd.DataFrame(logs)


def _calibrate(losses, ppls=None, ladder=_LADDER):
    manifests, probes, logs = _fixture(losses, ppls)
    return calibrate_arms(
        manifests,
        probes,
        logs,
        arms=[_ARM],
        ladders={_ARM.ladder_key: ladder},
    )


def test_loss_matched_rung_is_the_closest_within_the_band() -> None:
    # 2.1 is 5% off the 2.0 reference; 2.4 is 20% off and must not win despite being the
    # lowest-loss rung's neighbour.
    losses = {0.1: 3.0, 0.2: 2.4, 0.4: 2.1, 0.8: 1.5, 1.6: 2.6, 3.2: 4.0}
    cal = _calibrate(losses)[_ARM.key]
    assert cal.lr_lossmatch == 0.4
    assert cal.lossmatch_rel_error == pytest.approx(0.05)
    assert cal.lr_best == 0.8


def test_phase1_ladder_brackets_best_and_loss_matched_rungs() -> None:
    losses = {0.1: 3.0, 0.2: 2.4, 0.4: 2.1, 0.8: 1.5, 1.6: 2.6, 3.2: 4.0}
    cal = _calibrate(losses)[_ARM.key]
    assert len(cal.ladder) == plan.PHASE1_LADDER_RUNGS
    assert cal.lr_best in cal.ladder and cal.lr_lossmatch in cal.ladder


def test_diverged_rung_is_excluded_before_it_can_win_on_loss() -> None:
    # The lowest raw loss sits on a rung whose probe blew up: divergence exclusion must
    # run first, or the campaign selects a broken model as "best".
    losses = {0.1: 3.0, 0.2: 2.4, 0.4: 2.1, 0.8: 1.5, 1.6: 0.01, 3.2: 4.0}
    ppls = {1.6: _REF_PPL * plan.PHASE0_PPL_MAX_RATIO * 10}
    cal = _calibrate(losses, ppls)[_ARM.key]
    assert 1.6 not in cal.usable_lrs
    assert cal.lr_best == 0.8
    excluded = {r.lr: r.exclusion_reason for r in cal.rungs if r.excluded}
    assert "ppl" in excluded[1.6]


def test_non_finite_loss_rung_is_excluded() -> None:
    losses = {0.1: 3.0, 0.2: 2.4, 0.4: 2.1, 0.8: 1.5, 1.6: float("nan"), 3.2: 4.0}
    cal = _calibrate(losses)[_ARM.key]
    assert 1.6 not in cal.usable_lrs


def test_best_rung_on_a_ladder_endpoint_raises() -> None:
    losses = {0.1: 3.0, 0.2: 2.4, 0.4: 2.1, 0.8: 2.05, 1.6: 1.9, 3.2: 0.5}
    with pytest.raises(LadderTooNarrow, match="best-loss"):
        _calibrate(losses)


def test_loss_matched_rung_on_a_ladder_endpoint_raises() -> None:
    losses = {0.1: 2.0, 0.2: 2.4, 0.4: 2.6, 0.8: 1.5, 1.6: 2.8, 3.2: 4.0}
    with pytest.raises(LadderTooNarrow, match="closest-to-reference"):
        _calibrate(losses)


def test_interior_loss_floor_above_the_band_is_recorded_not_raised() -> None:
    # Every rung is far above the 2.0 reference and the closest (0.8) is interior, so the
    # ladder is not censored: the arm has a genuine loss floor. It must survive on
    # lr_best with no loss-matched rung, never quote 3.5 as "matched".
    losses = {0.1: 5.0, 0.2: 4.5, 0.4: 4.0, 0.8: 3.5, 1.6: 4.2, 3.2: 6.0}
    cal = _calibrate(losses)[_ARM.key]
    assert cal.lossmatch_infeasible is True
    assert cal.lr_lossmatch is None
    assert cal.lossmatch_closest_lr == 0.8
    assert cal.lossmatch_rel_error == pytest.approx(0.75)
    assert cal.lr_best == 0.8
    assert len(cal.ladder) == plan.PHASE1_LADDER_RUNGS
    assert cal.lr_best in cal.ladder


def test_out_of_band_closest_rung_on_an_endpoint_still_raises() -> None:
    # Losses fall monotonically toward the top rung, so the ladder is censored rather than
    # bottomed out — reporting a loss floor here would be an artifact of where we stopped.
    losses = {0.1: 9.0, 0.2: 8.0, 0.4: 7.0, 0.8: 6.0, 1.6: 5.0, 3.2: 4.0}
    with pytest.raises(LadderTooNarrow):
        _calibrate(losses)


def test_rungs_trained_at_another_step_budget_are_not_read() -> None:
    # Phase 1 writes leaves for the same arm and LRs on the epoch budget. Ranking those
    # against the screen would compare final losses at different doses.
    losses = {0.1: 3.0, 0.2: 2.4, 0.4: 2.1, 0.8: 1.5, 1.6: 2.6, 3.2: 4.0}
    manifests, probes, logs = _fixture(losses)
    other = pd.DataFrame([_manifest(_ARM, lr, 0.001, max_steps=None) for lr in _LADDER])
    cal = calibrate_arms(
        pd.concat([manifests, other], ignore_index=True),
        probes,
        logs,
        arms=[_ARM],
        ladders={_ARM.ladder_key: _LADDER},
    )[_ARM.key]
    assert cal.lr_best == 0.8
    assert cal.lr_lossmatch == 0.4


def test_every_rung_excluded_raises_rather_than_selecting_one() -> None:
    losses = dict.fromkeys(_LADDER, float("nan"))
    with pytest.raises(LossMatchOutOfBand, match="every rung"):
        _calibrate(losses)


def test_missing_reference_cell_raises() -> None:
    manifests, probes, logs = _fixture({lr: 2.0 for lr in _LADDER})
    manifests = manifests[manifests["student_optim"] != _REF.optim]
    with pytest.raises(LossMatchOutOfBand, match="loss-match reference"):
        calibrate_arms(
            manifests, probes, logs, arms=[_ARM], ladders={_ARM.ladder_key: _LADDER}
        )


# ---- Phase 1 LR selection ---------------------------------------------------------


def _delta_row(arm, lr: float, animal: str, delta: float) -> dict:
    return {
        "set": plan.PRIMARY_PROMPT_SET,
        "scoring": plan.PRIMARY_SCORING,
        "animal": animal,
        "student_mode": "full" if arm.kind == "full" else "lora",
        "student_optim": arm.optim,
        "optim_momentum": arm.momentum,
        "student_lr": lr,
        "subliminal_delta": delta,
    }


def test_selected_lr_maximizes_the_cross_animal_mean_delta() -> None:
    ladder = (0.2, 0.4, 0.8, 1.6)
    rows = []
    for lr, (cat, dog) in zip(
        ladder, [(0.01, 0.02), (0.30, 0.10), (0.05, 0.05), (0.0, 0.0)]
    ):
        rows += [_delta_row(_ARM, lr, "cat", cat), _delta_row(_ARM, lr, "dog", dog)]
    sel = select_ladder_lrs(
        pd.DataFrame(rows),
        {_ARM.key: ladder},
        {_ARM.key: 0.8},
        arms=[_ARM],
    )[_ARM.key]
    assert sel.lr == 0.4
    assert sel.mean_delta == pytest.approx(0.20)
    assert sel.needs_lossmatch_arm is True


def test_arm_without_a_feasible_lossmatch_gets_no_phase2b_tier() -> None:
    ladder = (0.2, 0.4, 0.8, 1.6)
    rows = [
        _delta_row(_ARM, 0.4, "cat", 0.05),
        _delta_row(_ARM, 0.4, "dog", 0.05),
    ]
    sel = select_ladder_lrs(
        pd.DataFrame(rows), {_ARM.key: ladder}, {_ARM.key: None}, arms=[_ARM]
    )[_ARM.key]
    assert sel.lr == 0.4
    assert sel.lr_lossmatch is None
    assert sel.needs_lossmatch_arm is False


def test_single_animal_lr_never_wins_the_selection() -> None:
    # A huge cat-only delta at 1.6 must lose to the only rung both animals evaluated:
    # otherwise the headline arm is chosen on whichever eval shard finished first.
    ladder = (0.2, 0.4, 0.8, 1.6)
    rows = [
        _delta_row(_ARM, 0.4, "cat", 0.05),
        _delta_row(_ARM, 0.4, "dog", 0.05),
        _delta_row(_ARM, 1.6, "cat", 0.90),
    ]
    sel = select_ladder_lrs(
        pd.DataFrame(rows), {_ARM.key: ladder}, {_ARM.key: 0.4}, arms=[_ARM]
    )[_ARM.key]
    assert sel.lr == 0.4
    assert sel.needs_lossmatch_arm is False


def test_selection_with_no_complete_rung_raises() -> None:
    rows = [_delta_row(_ARM, 0.4, "cat", 0.05)]
    with pytest.raises(ValueError, match="Phase-1 evals are incomplete"):
        select_ladder_lrs(
            pd.DataFrame(rows), {_ARM.key: (0.2, 0.4)}, {_ARM.key: 0.4}, arms=[_ARM]
        )


def test_selection_ignores_other_prompt_sets() -> None:
    ladder = (0.2, 0.4)
    rows = [
        _delta_row(_ARM, 0.2, "cat", 0.01),
        _delta_row(_ARM, 0.2, "dog", 0.01),
        _delta_row(_ARM, 0.4, "cat", 0.10),
        _delta_row(_ARM, 0.4, "dog", 0.10),
    ]
    off_set = [{**r, "set": "favorite", "subliminal_delta": 0.99} for r in rows[:2]]
    sel = select_ladder_lrs(
        pd.DataFrame(rows + off_set),
        {_ARM.key: ladder},
        {_ARM.key: 0.4},
        arms=[_ARM],
    )[_ARM.key]
    assert sel.lr == 0.4


# ---- Phase-1 rung disposition -----------------------------------------------------


def _p1_probe(
    arm,
    lr: float,
    *,
    animal: str = "cat",
    is_control: bool = False,
    diverged: bool = False,
    p_target: float = 0.003,
    ppl: float = 50.0,
    gseed: int = 0,
) -> dict:
    return {
        "student_mode": "full" if arm.kind == "full" else "lora",
        "student_optim": arm.optim,
        "optim_momentum": arm.momentum,
        "student_lr": lr,
        "max_steps": float("nan"),
        "is_control": is_control,
        "animal": animal,
        "step": 625,
        "ppl": ppl,
        "p_target": p_target,
        "diverged": diverged,
        "gseed": gseed,
        "run_dir": (f"{arm.key}/{lr}/{animal}/c{is_control}/g{gseed}/d{diverged}"),
    }


def _screen_fixture(
    arm,
    ladder: tuple[float, ...],
    *,
    control_diverged_lr: float | None = None,
    treatment_diverged_lrs: dict[float, int] | None = None,
    usable_lrs: tuple[float, ...] | None = None,
):
    """Minimal Phase-1 probe frame + calibration for `screen_rungs` tests."""
    treatment_diverged_lrs = treatment_diverged_lrs or {}
    rows: list[dict] = []
    for lr in ladder:
        n_div = treatment_diverged_lrs.get(lr, 0)
        for animal in ("cat", "dog"):
            for g in range(3):
                rows.append(
                    _p1_probe(
                        arm,
                        lr,
                        animal=animal,
                        gseed=g,
                        diverged=g < n_div,
                        p_target=0.2,
                        ppl=40.0,
                    )
                )
            rows.append(
                _p1_probe(
                    arm,
                    lr,
                    animal=animal,
                    is_control=True,
                    diverged=control_diverged_lr is not None
                    and abs(lr - control_diverged_lr) < 1e-12,
                    p_target=0.003,
                    ppl=55.0,
                )
            )
    cal = {
        arm.key: ArmCalibration(
            arm_key=arm.key,
            rungs=(),
            usable_lrs=usable_lrs or ladder,
            lr_best=ladder[len(ladder) // 2],
            lr_lossmatch=None,
            lossmatch_closest_lr=ladder[0],
            lossmatch_rel_error=1.0,
            ladder=ladder,
        )
    }
    return pd.DataFrame(rows), cal


def test_screen_rungs_clean_rung_is_ok() -> None:
    ladder = (0.001, 0.01, 0.1, 1.0)
    probes, cal = _screen_fixture(_ARM, ladder, usable_lrs=(0.001, 0.01, 0.1, 1.0))
    # Interior rung 0.01 is not at usable edge.
    disp = screen_rungs(probes, {_ARM.key: ladder}, arms=[_ARM], calibrations=cal)
    assert disp[(_ARM.key, 0.01)].tier == RungTier.OK


def test_screen_rungs_control_diverged_quarantines() -> None:
    ladder = (0.001, 0.01, 0.1, 1.0)
    probes, cal = _screen_fixture(
        _ARM, ladder, control_diverged_lr=0.1, usable_lrs=ladder
    )
    disp = screen_rungs(probes, {_ARM.key: ladder}, arms=[_ARM], calibrations=cal)
    assert disp[(_ARM.key, 0.1)].tier == RungTier.QUARANTINE
    assert "control_diverged" in disp[(_ARM.key, 0.1)].reasons


def test_screen_rungs_majority_treatment_diverged_quarantines() -> None:
    # 2 of 3 gseeds diverged per animal -> 4/6 = 0.667 >= 0.5.
    ladder = (0.001, 0.01, 0.1, 1.0)
    probes, cal = _screen_fixture(
        _ARM, ladder, treatment_diverged_lrs={0.1: 2}, usable_lrs=ladder
    )
    disp = screen_rungs(probes, {_ARM.key: ladder}, arms=[_ARM], calibrations=cal)
    assert disp[(_ARM.key, 0.1)].tier == RungTier.QUARANTINE
    assert any(
        "treatment_diverged_fraction" in r for r in disp[(_ARM.key, 0.1)].reasons
    )


def test_screen_rungs_edge_of_usable_range_is_advisory() -> None:
    ladder = (0.001, 0.01, 0.1)
    probes, cal = _screen_fixture(_ARM, ladder, usable_lrs=ladder)
    disp = screen_rungs(probes, {_ARM.key: ladder}, arms=[_ARM], calibrations=cal)
    assert disp[(_ARM.key, 0.1)].tier == RungTier.ADVISORY
    assert "edge_of_usable_range" in disp[(_ARM.key, 0.1)].reasons


def _floor_fixture(arm, floors: dict[float, dict[str, float | None]]):
    """Probe frame whose control floors are set per (lr, animal); ``None`` = no leaf."""
    rows: list[dict] = []
    for lr, per_animal in floors.items():
        for animal, floor in per_animal.items():
            for g in range(3):
                rows.append(
                    _p1_probe(arm, lr, animal=animal, gseed=g, p_target=0.2, ppl=40.0)
                )
            if floor is not None:
                rows.append(
                    _p1_probe(
                        arm,
                        lr,
                        animal=animal,
                        is_control=True,
                        p_target=floor,
                        ppl=50.0,
                    )
                )
    return pd.DataFrame(rows).assign(beta=1.0)


def test_screen_rungs_quarantines_single_animal_spread_outlier() -> None:
    """A spread violation in one animal alone must quarantine, not reach the gate.

    The max-over-animals floor is flat here, so only per-animal grouping — the grouping
    `control_floor_check` uses — sees the violation.
    """
    ladder = (0.001, 0.01)
    probes = _floor_fixture(
        _ARM,
        {0.001: {"cat": 0.001, "dog": 0.016}, 0.01: {"cat": 0.015, "dog": 0.016}},
    )
    disp = screen_rungs(probes, {_ARM.key: ladder}, arms=[_ARM])
    quarantined = {lr for (_, lr), d in disp.items() if d.tier == RungTier.QUARANTINE}
    assert quarantined, "single-animal spread violation was not quarantined"
    floor = control_floor_check(
        probes,
        animals=["cat", "dog"],
        betas=[1.0],
        max_floor=0.02,
        max_spread=0.01,
        arms=[
            FloorArm(
                student_mode="lora",
                lrs=ladder,
                student_optim=_ARM.optim,
                optim_momentum=_ARM.momentum,
            )
        ],
        quarantined=quarantined_pairs(disp),
    )
    assert floor.passed, floor.failures


def test_screen_rungs_missing_control_leaf_quarantines() -> None:
    ladder = (0.001, 0.01)
    probes = _floor_fixture(
        _ARM, {0.001: {"cat": 0.003, "dog": 0.003}, 0.01: {"cat": 0.003, "dog": None}}
    )
    disp = screen_rungs(probes, {_ARM.key: ladder}, arms=[_ARM])
    assert disp[(_ARM.key, 0.01)].tier == RungTier.QUARANTINE
    assert "control_leaf_missing" in disp[(_ARM.key, 0.01)].reasons


def test_select_ladder_lrs_skips_quarantined_max_delta_rung() -> None:
    # Largest delta at 0.1 is quarantined; selection must fall to 0.01.
    ladder = (0.001, 0.01, 0.1)
    rows = []
    for lr, (cat, dog) in zip(
        ladder, [(0.001, 0.001), (0.0087, 0.0087), (0.0313, 0.0313)]
    ):
        rows += [
            _delta_row(_ARM, lr, "cat", cat),
            _delta_row(_ARM, lr, "dog", dog),
        ]
    sel = select_ladder_lrs(
        pd.DataFrame(rows),
        {_ARM.key: ladder},
        {_ARM.key: None},
        arms=[_ARM],
        quarantined={(_ARM.key, 0.1)},
    )[_ARM.key]
    assert sel.lr == 0.01
    assert sel.mean_delta == pytest.approx(0.0087)


def test_select_ladder_lrs_omits_fully_quarantined_arm() -> None:
    other = plan.ARMS_BY_KEY["lora_adamw"]
    ladder = (0.001, 0.01)
    rows = [
        _delta_row(other, 0.001, "cat", 0.1),
        _delta_row(other, 0.001, "dog", 0.1),
        _delta_row(_ARM, 0.001, "cat", 0.05),
        _delta_row(_ARM, 0.001, "dog", 0.05),
    ]
    sel = select_ladder_lrs(
        pd.DataFrame(rows),
        {_ARM.key: ladder, other.key: ladder},
        {_ARM.key: None, other.key: None},
        arms=[_ARM, other],
        quarantined={(_ARM.key, 0.001), (_ARM.key, 0.01)},
    )
    assert other.key in sel
    assert _ARM.key not in sel
