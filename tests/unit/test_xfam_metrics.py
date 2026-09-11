"""Planted-frame tests for the cross-family claim math."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from subliminal_learning.campaign import xfam_metrics as fm
from subliminal_learning.campaign import xfam_models as models
from subliminal_learning.campaign import xfam_pairs as pairs
from subliminal_learning.campaign import xfam_plan as plan
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag

N_BOOT = 64
SEED = 7

_NEOX_PICKS: dict[str, float] = {
    "pythia-410m": plan.LR_DEFAULTS["pythia-410m"],
    "pythia-410m-deduped": plan.LR_DEFAULTS["pythia-410m-deduped"],
    "amd-olmo-1b": plan.LR_DEFAULTS["amd-olmo-1b"],
}


def _gkd(
    lr: float, *, recipe=plan.BASE_RECIPE, epochs: float = plan.BASE_EPOCHS
) -> str:
    return plan.student_gkd_tag(lr, recipe, epochs)


def _leaf(
    *,
    teacher: str,
    student: str,
    kind: str,
    gkd_tag: str,
    gseed: int,
    p_target: float,
    arm: str = "",
    ftseed: float = 0,
    ppl: float = 1.2,
    filter_drop: float = 0.0,
    param_tag: str = plan.TEACHER_PARAM_TAG,
) -> dict:
    parsed = parse_gkd_tag(gkd_tag)
    control = kind != fm.KIND_TREATMENT
    return {
        "animal": "bear",
        "cluster": models.cluster_of(student),
        "student": student,
        "teacher": teacher,
        "kind": kind,
        "arm": "" if control else arm,
        "arm_kind": "" if control else plan.arm(arm).kind,
        "pair_kind": fm.pair_kind_of(teacher, student),
        "teacher_param_tag": "" if control else param_tag,
        "lr": float(parsed["student_lr"]),
        "epochs": float(parsed["num_epochs"]),
        "lora_r": int(parsed["student_lora_r"]),
        "lora_alpha": int(parsed["student_lora_alpha"]),
        "gkd_tag": gkd_tag,
        "ftseed": float("nan") if control else float(ftseed),
        "gseed": gseed,
        "p_target": p_target,
        "perplexity_ratio": ppl,
        "filter_drop": filter_drop,
        "quarantined": False,
    }


def _plant_cell(
    rows: list[dict],
    *,
    arm: str,
    teacher: str,
    student: str,
    lr: float,
    delta: float,
    gseeds: tuple[int, ...] = (0, 1),
    ftseeds: tuple[int, ...] = (0,),
    control: float = 0.05,
    ppl: float | tuple[float, ...] = 1.2,
    recipe=plan.BASE_RECIPE,
    epochs: float = plan.BASE_EPOCHS,
    param_tag: str = plan.TEACHER_PARAM_TAG,
) -> None:
    """One cell's treatment leaves plus the controls their delta needs."""
    tag = _gkd(lr, recipe=recipe, epochs=epochs)
    anchor = teacher == student
    ppls = ppl if isinstance(ppl, tuple) else (ppl,) * (len(gseeds) * len(ftseeds))
    index = 0
    for ftseed in ftseeds:
        for gseed in gseeds:
            rows.append(
                _leaf(
                    teacher=teacher,
                    student=student,
                    kind=fm.KIND_TREATMENT,
                    gkd_tag=tag,
                    gseed=gseed,
                    p_target=control + delta,
                    arm=arm,
                    ftseed=ftseed,
                    ppl=ppls[index],
                    param_tag=param_tag,
                )
            )
            index += 1
    for gseed in gseeds:
        rows.append(
            _leaf(
                teacher=student,
                student=student,
                kind=fm.KIND_SB,
                gkd_tag=tag,
                gseed=gseed,
                p_target=control if anchor else 0.0,
            )
        )
        if not anchor:
            rows.append(
                _leaf(
                    teacher=teacher,
                    student=student,
                    kind=fm.KIND_TB,
                    gkd_tag=tag,
                    gseed=gseed,
                    p_target=control,
                )
            )


def _frames(rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    leaves = fm.propagate_control_quarantine(fm.quarantine(pd.DataFrame(rows)))
    deltas = fm.leaf_deltas(leaves)
    return deltas, fm.cell_means(deltas, n_boot=N_BOOT, seed=SEED)


# ---- identity ----------------------------------------------------------------


def test_arm_of_leaf_recovers_every_planned_leaf() -> None:
    for leaf in pairs.all_leaves(pairs.default_picks()):
        assert (
            fm.arm_of_leaf(
                teacher=leaf.teacher,
                student=leaf.student,
                teacher_param_tag=leaf.teacher_param_tag,
                ftseed=leaf.ftseed,
                lr=leaf.lr,
                lora_r=leaf.recipe.lora_r,
                lora_alpha=leaf.recipe.lora_alpha,
                epochs=leaf.epochs,
            )
            == leaf.arm
        )


def test_arm_of_leaf_unknown_combination_raises() -> None:
    with pytest.raises(ValueError, match="no xfam arm"):
        fm.arm_of_leaf(
            teacher="pythia-410m",
            student="qwen3-0.6b",
            teacher_param_tag=plan.TEACHER_PARAM_TAG,
            ftseed=0,
            lr=4.0e-4,
            lora_r=8,
            lora_alpha=16,
            epochs=plan.BASE_EPOCHS,
        )


def test_rung_kind_separates_post_training_from_a_second_pretraining_run() -> None:
    assert fm.rung_kind_of("pythia-410m") == fm.RUNG_ANCHOR
    assert fm.rung_kind_of("pythia-410m-deduped") == fm.RUNG_DIFFERENT_RUN
    assert fm.rung_kind_of("pythia-410m-hh-dpo") == fm.RUNG_POST_TRAINED
    assert fm.rung_kind_of("amd-olmo-1b") == fm.RUNG_DIFFERENT_FAMILY


# ---- deltas + quarantine -----------------------------------------------------


def test_leaf_deltas_uses_the_matched_control_of_each_class() -> None:
    rows: list[dict] = []
    _plant_cell(
        rows,
        arm="a4-olmo",
        teacher="pythia-410m",
        student="amd-olmo-1b",
        lr=4.0e-4,
        delta=0.20,
        control=0.10,
    )
    deltas, _ = _frames(rows)
    row = deltas.iloc[0]
    assert row["delta_tb"] == pytest.approx(0.20)
    assert row["tb_control_mean"] == pytest.approx(0.10)
    assert row["delta_sb"] == pytest.approx(0.30)
    assert row["delta_primary"] == pytest.approx(row["delta_tb"])


def test_leaf_deltas_anchor_has_no_teacher_base_delta() -> None:
    rows: list[dict] = []
    _plant_cell(
        rows,
        arm="a1-anchor",
        teacher="pythia-410m",
        student="pythia-410m",
        lr=4.0e-4,
        delta=0.25,
        control=0.05,
    )
    deltas, _ = _frames(rows)
    row = deltas.iloc[0]
    assert row["delta_sb"] == pytest.approx(0.25)
    assert math.isnan(float(row["delta_tb"]))
    assert row["delta_primary"] == pytest.approx(0.25)


def test_leaf_deltas_missing_teacher_base_control_raises() -> None:
    rows: list[dict] = []
    _plant_cell(
        rows,
        arm="a4-olmo",
        teacher="pythia-410m",
        student="amd-olmo-1b",
        lr=4.0e-4,
        delta=0.20,
    )
    rows = [r for r in rows if r["kind"] != fm.KIND_TB]
    with pytest.raises(KeyError):
        fm.leaf_deltas(pd.DataFrame(rows))


def test_leaf_deltas_missing_student_base_control_raises() -> None:
    rows: list[dict] = []
    _plant_cell(
        rows,
        arm="a4-olmo",
        teacher="pythia-410m",
        student="amd-olmo-1b",
        lr=4.0e-4,
        delta=0.20,
    )
    rows = [r for r in rows if r["kind"] != fm.KIND_SB]
    with pytest.raises(KeyError):
        fm.leaf_deltas(pd.DataFrame(rows))


def test_quarantine_labels_the_leaf_and_keeps_it_out_of_the_cell_mean() -> None:
    rows: list[dict] = []
    _plant_cell(
        rows,
        arm="a4-olmo",
        teacher="pythia-410m",
        student="amd-olmo-1b",
        lr=4.0e-4,
        delta=0.20,
        ppl=(1.2, plan.QUARANTINE_PPL_RATIO_MAX + 1.0),
    )
    leaves = fm.propagate_control_quarantine(fm.quarantine(pd.DataFrame(rows)))
    flagged = leaves[leaves["quarantined"].astype(bool)]
    assert len(flagged) == 1
    assert flagged.iloc[0]["quarantine_reason"] == "ppl_ratio"
    deltas = fm.leaf_deltas(leaves)
    assert len(deltas) == 2
    cells = fm.cell_means(deltas, n_boot=N_BOOT, seed=SEED)
    assert int(cells.iloc[0]["n"]) == 1


def test_quarantine_flags_a_filter_drop_breach() -> None:
    rows: list[dict] = []
    _plant_cell(
        rows,
        arm="a4-olmo",
        teacher="pythia-410m",
        student="amd-olmo-1b",
        lr=4.0e-4,
        delta=0.20,
    )
    rows[0]["filter_drop"] = plan.QUARANTINE_FILTER_DROP_MAX + 0.1
    leaves = fm.quarantine(pd.DataFrame(rows))
    assert leaves.iloc[0]["quarantine_reason"] == "filter_drop"
    assert bool(leaves.iloc[0]["quarantined"])


def test_cell_means_on_a_constant_frame_returns_a_degenerate_interval() -> None:
    rows: list[dict] = []
    _plant_cell(
        rows,
        arm="a4-olmo",
        teacher="pythia-410m",
        student="amd-olmo-1b",
        lr=4.0e-4,
        delta=0.20,
        gseeds=(0, 1, 2),
    )
    _, cells = _frames(rows)
    cell = cells.iloc[0]
    assert int(cell["n"]) == 3
    assert cell["mean_delta_primary"] == pytest.approx(0.20)
    assert cell["ci_low"] == pytest.approx(0.20)
    assert cell["ci_high"] == pytest.approx(0.20)
    assert cell["arm"] == "a4-olmo"


# ---- F1 / F2 -----------------------------------------------------------------


def _neox_ladder_rows(deltas_by_rung: dict[str, float]) -> list[dict]:
    rows: list[dict] = []
    emitter = models.TEACHER_OF["neox"]
    for student, delta in deltas_by_rung.items():
        arm = next(
            a.name
            for a in plan.ARMS
            if a.teacher == emitter
            and student in a.students
            and a.kind in (plan.KIND_ANCHOR, plan.KIND_FORWARD)
        )
        lr = _NEOX_PICKS[plan.arm(arm).lr_source_of(student)]
        _plant_cell(rows, arm=arm, teacher=emitter, student=student, lr=lr, delta=delta)
    return rows


def test_init_distance_ladder_monotone_plant_gives_spearman_minus_one() -> None:
    rungs = plan.INIT_DISTANCE_RUNGS["neox"]
    planted = {label: 0.40 - 0.05 * i for i, label in enumerate(rungs)}
    _, cells = _frames(_neox_ladder_rows(planted))
    ladder = fm.init_distance_ladder(cells, picks=_NEOX_PICKS)
    neox = ladder[ladder["cluster"] == "neox"]
    assert len(neox) == len(rungs)
    assert list(neox.sort_values("rung_index")["student"]) == list(rungs)
    assert float(neox.iloc[0]["spearman_rho"]) == pytest.approx(-1.0)
    assert float(neox.iloc[0]["spearman_p"]) < 0.05


def test_architecture_penalty_scales_the_cross_cell_by_its_cluster_anchor() -> None:
    rows = _neox_ladder_rows({"pythia-410m": 0.40, "amd-olmo-1b": 0.10})
    deltas, cells = _frames(rows)
    penalty = fm.architecture_penalty(
        cells,
        picks=_NEOX_PICKS,
        deltas=deltas,
        teacher_p_target={"pythia-410m": 0.50},
        n_boot=N_BOOT,
        seed=SEED,
    )
    row = penalty[penalty["student"] == "amd-olmo-1b"].iloc[0]
    assert row["delta_anchor"] == pytest.approx(0.40)
    assert row["rho_s"] == pytest.approx(0.25)
    assert row["penalty"] == pytest.approx(0.30)
    assert row["efficiency"] == pytest.approx(0.20)
    assert bool(row["penalty_holds"])


# ---- F3 ----------------------------------------------------------------------


def test_posttrain_contrast_reads_each_receiver_against_its_own_lineage() -> None:
    rows = _neox_ladder_rows({"pythia-410m": 0.40, "pythia-410m-hh-sft": 0.30})
    deltas, cells = _frames(rows)
    frame = fm.posttrain_contrast(
        cells, picks=_NEOX_PICKS, deltas=deltas, n_boot=N_BOOT, seed=SEED
    )
    row = frame[frame["student"] == "pythia-410m-hh-sft"].iloc[0]
    assert row["lineage"] == "pythia-410m"
    assert row["post_training"] == "sft"
    assert row["delta"] == pytest.approx(0.30)
    assert row["base_delta"] == pytest.approx(0.40)
    assert row["diff"] == pytest.approx(-0.10)
    assert bool(row["significant"])
    payload = fm.claims(cells, picks=_NEOX_PICKS, posttrain=frame)
    entry = payload["F3"]["students"]["pythia-410m-hh-sft"]
    assert entry["direction"] == "less_receptive"
    assert payload["F3"]["n_significant"] == 1


# ---- F4 ----------------------------------------------------------------------


def _reciprocal_rows(forward_delta: float, reverse_delta: float) -> list[dict]:
    rows: list[dict] = []
    _plant_cell(
        rows,
        arm="a4-olmo",
        teacher="pythia-410m",
        student="amd-olmo-1b",
        lr=_NEOX_PICKS["amd-olmo-1b"],
        delta=forward_delta,
    )
    _plant_cell(
        rows,
        arm="a7-reverse-olmo",
        teacher="amd-olmo-1b",
        student="pythia-410m",
        lr=_NEOX_PICKS["pythia-410m"],
        delta=reverse_delta,
    )
    return rows


def test_direction_asymmetry_is_signed_on_an_asymmetric_plant() -> None:
    deltas, cells = _frames(_reciprocal_rows(0.30, 0.05))
    frame = fm.direction_asymmetry(
        cells,
        picks=_NEOX_PICKS,
        deltas=deltas,
        teacher_p_target={"pythia-410m": 0.50, "amd-olmo-1b": 0.25},
        n_boot=N_BOOT,
        seed=SEED,
    )
    row = frame.iloc[0]
    assert (row["label_a"], row["label_b"]) == ("pythia-410m", "amd-olmo-1b")
    assert row["iota"] == pytest.approx(0.25)
    assert row["iota_dose"] == pytest.approx(0.30 / 0.50 - 0.05 / 0.25)
    assert row["ci_low"] > 0.0


def test_direction_asymmetry_is_null_on_a_symmetric_plant() -> None:
    deltas, cells = _frames(_reciprocal_rows(0.20, 0.20))
    frame = fm.direction_asymmetry(
        cells, picks=_NEOX_PICKS, deltas=deltas, n_boot=N_BOOT, seed=SEED
    )
    row = frame.iloc[0]
    assert row["iota"] == pytest.approx(0.0)
    assert row["ci_low"] <= 0.0 <= row["ci_high"]


# ---- F5 ----------------------------------------------------------------------


def test_hp_robustness_kendall_tau_is_one_when_the_ranking_survives() -> None:
    # Both students picked at the same LR, so 2e-4 is off-pick for both and the
    # ranking at that rung is comparable with the ranking at lr*.
    picks = {**_NEOX_PICKS, "amd-olmo-1b": 4.0e-4}
    rows: list[dict] = []
    for student, arm, delta in (
        ("pythia-410m-deduped", "a2-deduped", 0.30),
        ("amd-olmo-1b", "a4-olmo", 0.20),
    ):
        _plant_cell(
            rows,
            arm=arm,
            teacher="pythia-410m",
            student=student,
            lr=picks[student],
            delta=delta,
        )
    for student, arm, delta in (
        ("pythia-410m-deduped", "a2-deduped", 0.15),
        ("amd-olmo-1b", "a4-olmo", 0.10),
    ):
        _plant_cell(
            rows,
            arm=arm,
            teacher="pythia-410m",
            student=student,
            lr=2.0e-4,
            delta=delta,
        )
    _, cells = _frames(rows)
    hp = fm.hp_robustness(cells, picks=picks)
    assert set(hp["axis"]) == {fm.AXIS_LR}
    assert set(hp["setting"]) == {"lr0.0002"}
    assert int(hp.iloc[0]["n_shared"]) == 2
    assert float(hp.iloc[0]["kendall_tau"]) == pytest.approx(1.0)
    assert not hp["sign_flip"].astype(bool).any()


def test_hp_robustness_marks_a_sign_flip_against_the_picked_lr() -> None:
    picks = {**_NEOX_PICKS, "amd-olmo-1b": 8.0e-4}
    rows: list[dict] = []
    _plant_cell(
        rows,
        arm="a4-olmo",
        teacher="pythia-410m",
        student="amd-olmo-1b",
        lr=8.0e-4,
        delta=0.30,
    )
    _plant_cell(
        rows,
        arm="a4-olmo",
        teacher="pythia-410m",
        student="amd-olmo-1b",
        lr=2.0e-4,
        delta=-0.10,
    )
    _, cells = _frames(rows)
    hp = fm.hp_robustness(cells, picks=picks)
    assert bool(hp.iloc[0]["sign_flip"])
    assert int(hp.iloc[0]["sign"]) == -1


def test_hp_robustness_separates_the_rank_and_epoch_axes() -> None:
    rows: list[dict] = []
    _plant_cell(
        rows,
        arm="a4-olmo",
        teacher="pythia-410m",
        student="amd-olmo-1b",
        lr=_NEOX_PICKS["amd-olmo-1b"],
        delta=0.30,
    )
    _plant_cell(
        rows,
        arm="a8-rank-olmo",
        teacher="pythia-410m",
        student="amd-olmo-1b",
        lr=_NEOX_PICKS["amd-olmo-1b"],
        delta=0.28,
        recipe=plan.RANK_RECIPES[0],
    )
    _plant_cell(
        rows,
        arm="a8-rank-olmo",
        teacher="pythia-410m",
        student="amd-olmo-1b",
        lr=_NEOX_PICKS["amd-olmo-1b"],
        delta=0.33,
        epochs=plan.EPOCH_ARM_EPOCHS,
    )
    _, cells = _frames(rows)
    hp = fm.hp_robustness(cells, picks=_NEOX_PICKS)
    assert set(hp["axis"]) == {fm.AXIS_RANK, fm.AXIS_EPOCH}
    assert set(hp["setting"]) == {"r32a64", "ep2"}


# ---- F6 ----------------------------------------------------------------------


def test_ppl_coupling_retains_the_planted_ppl_free_component() -> None:
    rows: list[dict] = []
    ppls = (1.0, 1.2, 1.4)
    tag = _gkd(4.0e-4)
    for gseed, ratio in enumerate(ppls):
        rows.append(
            _leaf(
                teacher="pythia-410m",
                student="amd-olmo-1b",
                kind=fm.KIND_TREATMENT,
                gkd_tag=tag,
                gseed=gseed,
                p_target=0.10 + 0.10 + 0.5 * (ratio - 1.0),
                arm="a4-olmo",
                ppl=ratio,
            )
        )
        rows.append(
            _leaf(
                teacher="pythia-410m",
                student="amd-olmo-1b",
                kind=fm.KIND_TB,
                gkd_tag=tag,
                gseed=gseed,
                p_target=0.10,
            )
        )
        rows.append(
            _leaf(
                teacher="amd-olmo-1b",
                student="amd-olmo-1b",
                kind=fm.KIND_SB,
                gkd_tag=tag,
                gseed=gseed,
                p_target=0.0,
            )
        )
    deltas, _ = _frames(rows)
    ppl = fm.ppl_coupling(deltas)
    row = ppl.iloc[0]
    assert row["beta"] == pytest.approx(0.5)
    assert row["mean_delta"] == pytest.approx(0.20)
    assert row["mean_delta_adj"] == pytest.approx(0.10)
    assert row["retained_fraction"] == pytest.approx(0.5)
    assert not bool(row["degenerate"])


def test_ppl_coupling_excludes_a_constant_regressor_instead_of_scoring_it_one() -> None:
    rows: list[dict] = []
    _plant_cell(
        rows,
        arm="a4-olmo",
        teacher="pythia-410m",
        student="amd-olmo-1b",
        lr=4.0e-4,
        delta=0.20,
        gseeds=(0, 1, 2),
        ppl=1.2,
    )
    deltas, _ = _frames(rows)
    ppl = fm.ppl_coupling(deltas)
    row = ppl.iloc[0]
    assert bool(row["degenerate"])
    assert row["degenerate_reason"] == "constant_regressor"
    assert math.isnan(float(row["retained_fraction"]))
    payload = fm.claims(
        fm.cell_means(deltas, n_boot=N_BOOT, seed=SEED),
        picks=_NEOX_PICKS,
        ppl=ppl,
    )
    assert payload["F6"]["status"] == "void"
    assert payload["F6"]["n_degenerate_groups"] == 1
    assert payload["F6"]["n_groups"] == 0


def test_ppl_coupling_degenerate_groups_do_not_inflate_the_claim() -> None:
    rows: list[dict] = []
    tag = _gkd(4.0e-4)
    for gseed, ratio in enumerate((1.0, 1.2, 1.4)):
        rows.append(
            _leaf(
                teacher="pythia-410m",
                student="amd-olmo-1b",
                kind=fm.KIND_TREATMENT,
                gkd_tag=tag,
                gseed=gseed,
                p_target=0.10 + 0.02 + 0.5 * (ratio - 1.0),
                arm="a4-olmo",
                ppl=ratio,
            )
        )
        rows.append(
            _leaf(
                teacher="pythia-410m",
                student="amd-olmo-1b",
                kind=fm.KIND_TB,
                gkd_tag=tag,
                gseed=gseed,
                p_target=0.10,
            )
        )
        rows.append(
            _leaf(
                teacher="amd-olmo-1b",
                student="amd-olmo-1b",
                kind=fm.KIND_SB,
                gkd_tag=tag,
                gseed=gseed,
                p_target=0.0,
            )
        )
    _plant_cell(
        rows,
        arm="a4-olmo",
        teacher="pythia-410m",
        student="amd-olmo-1b",
        lr=_NEOX_PICKS["amd-olmo-1b"],
        delta=0.20,
        gseeds=(0, 1, 2),
        ppl=1.3,
    )
    deltas, cells = _frames(rows)
    ppl = fm.ppl_coupling(deltas)
    payload = fm.claims(cells, picks=_NEOX_PICKS, ppl=ppl)
    assert payload["F6"]["n_degenerate_groups"] == 1
    assert payload["F6"]["n_groups"] == 1
    assert payload["F6"]["retained_fraction"] == pytest.approx(0.02 / 0.12)
    assert payload["F6"]["status"] == "fails"


# ---- claims ------------------------------------------------------------------


def test_claims_keys_are_exactly_the_plan_claim_ids() -> None:
    payload = fm.claims(fm.cell_means(pd.DataFrame()), picks={})
    assert tuple(payload) == plan.CLAIM_IDS
    assert all("status" in payload[claim] for claim in plan.CLAIM_IDS)


def test_f0_holds_on_a_planted_cross_cell_over_a_live_anchor() -> None:
    rows = _neox_ladder_rows({"pythia-410m": 0.40, "amd-olmo-1b": 0.20})
    _, cells = _frames(rows)
    payload = fm.claims(cells, picks=_NEOX_PICKS)
    entry = payload["F0"]["cells"]["a4-olmo:pythia-410m→amd-olmo-1b"]
    assert entry["status"] == "holds"
    assert entry["mean"] == pytest.approx(0.20)
    assert payload["F0"]["status"] == "holds"


def test_f0_is_void_when_the_emitter_failed_its_gate() -> None:
    from subliminal_learning.campaign.xfam_selection import TeacherGateDecision

    rows = _neox_ladder_rows({"pythia-410m": 0.40, "amd-olmo-1b": 0.20})
    _, cells = _frames(rows)
    payload = fm.claims(
        cells,
        picks=_NEOX_PICKS,
        teacher_decisions={
            "pythia-410m": TeacherGateDecision(
                label="pythia-410m",
                role="cluster",
                admitted_ftseeds=(),
                usable=False,
                reason="only_0_admitted_need_2",
            )
        },
    )
    entry = payload["F0"]["cells"]["a4-olmo:pythia-410m→amd-olmo-1b"]
    assert entry["status"] == "void"
    assert entry["reason"] == "teacher_not_admitted"
    assert payload["F0"]["status"] == "void"


def test_f0_is_void_when_the_cluster_anchor_covers_zero() -> None:
    rows: list[dict] = []
    _plant_cell(
        rows,
        arm="a1-anchor",
        teacher="pythia-410m",
        student="pythia-410m",
        lr=_NEOX_PICKS["pythia-410m"],
        delta=0.0,
    )
    rows[0]["p_target"] = 0.15
    rows[1]["p_target"] = -0.05
    _plant_cell(
        rows,
        arm="a4-olmo",
        teacher="pythia-410m",
        student="amd-olmo-1b",
        lr=_NEOX_PICKS["amd-olmo-1b"],
        delta=0.20,
    )
    _, cells = _frames(rows)
    payload = fm.claims(cells, picks=_NEOX_PICKS)
    entry = payload["F0"]["cells"]["a4-olmo:pythia-410m→amd-olmo-1b"]
    assert entry["status"] == "void"
    assert entry["reason"] == "anchor_ci_covers_0"


def test_f7_reports_the_rung_kinds_both_clusters_share() -> None:
    rows = _neox_ladder_rows(
        {"pythia-410m": 0.40, "pythia-410m-hh-sft": 0.30, "amd-olmo-1b": 0.10}
    )
    qwen_picks = {
        "qwen2.5-0.5b": plan.LR_DEFAULTS["qwen2.5-0.5b"],
        "qwen3-0.6b-base": plan.LR_DEFAULTS["qwen3-0.6b-base"],
    }
    _plant_cell(
        rows,
        arm="b1-anchor",
        teacher="qwen2.5-0.5b",
        student="qwen2.5-0.5b",
        lr=qwen_picks["qwen2.5-0.5b"],
        delta=0.35,
    )
    _plant_cell(
        rows,
        arm="b3-instruct",
        teacher="qwen2.5-0.5b",
        student="qwen2.5-0.5b-instruct",
        lr=qwen_picks["qwen2.5-0.5b"],
        delta=0.25,
    )
    _plant_cell(
        rows,
        arm="b2-qwen3-base",
        teacher="qwen2.5-0.5b",
        student="qwen3-0.6b-base",
        lr=qwen_picks["qwen3-0.6b-base"],
        delta=0.08,
    )
    _, cells = _frames(rows)
    payload = fm.claims(cells, picks={**_NEOX_PICKS, **qwen_picks})
    f7 = payload["F7"]
    assert f7["status"] == "descriptive"
    assert set(f7["clusters"]) == {"neox", "qwen"}
    assert f7["shared_rung_kinds"] == [
        fm.RUNG_ANCHOR,
        fm.RUNG_DIFFERENT_FAMILY,
        fm.RUNG_POST_TRAINED,
    ]
    assert f7["sign_concordant"] is True
    assert f7["order_concordant"] is True
