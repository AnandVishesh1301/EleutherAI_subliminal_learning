"""Planted-fixture tests for the xfam wave-2 claim math (`campaign.xfam2_metrics`)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.campaign import xfam2_metrics as ym
from subliminal_learning.campaign import xfam2_plan as plan

ANIMAL = "bear"
TAG = plan.student_gkd_tag(plan.NEOX_LR)


def _leaf(
    kind: str,
    teacher: str,
    student: str,
    ftseed: float,
    gseed: int,
    p: float,
    *,
    ppl: float = 1.2,
    rev_kl: float = float("nan"),
    lr: float = plan.NEOX_LR,
) -> dict:
    block, tier = ym.unit_of(teacher, student, lr, gseed)
    return {
        "animal": ANIMAL,
        "cluster": "neox",
        "block": block,
        "tier": tier,
        "arm": block,
        "student": student,
        "teacher": teacher,
        "kind": kind,
        "pair_kind": ym.pair_kind_of(teacher, student),
        "teacher_param_tag": plan.TEACHER_PARAM_TAG
        if kind == ym.KIND_TREATMENT
        else "",
        "lr": lr,
        "gkd_tag": plan.student_gkd_tag(lr),
        "ftseed": ftseed,
        "gseed": gseed,
        "p_target": p,
        "perplexity_ratio": ppl,
        "filter_drop": 0.01,
        "rev_kl_first": rev_kl,
        "teacher_admitted": True,
        "quarantined": False,
        "quarantine_reason": "",
    }


def _cell(
    teacher: str,
    student: str,
    per_ftseed: dict[int, float],
    *,
    control: float = 0.01,
    gseeds: tuple[int, ...] = plan.CORE_GSEEDS,
    noise: float = 0.0,
    rev_kl: float = float("nan"),
) -> list[dict]:
    """Treatment leaves at delta ``per_ftseed[f] (+ noise*gseed)`` plus their control."""
    kind = ym.KIND_SB if teacher == student else ym.KIND_TB
    rows = [
        _leaf(kind, teacher, student, float("nan"), g, control, rev_kl=rev_kl)
        for g in gseeds
    ]
    for f, delta in per_ftseed.items():
        for g in gseeds:
            rows.append(
                _leaf(
                    ym.KIND_TREATMENT,
                    teacher,
                    student,
                    f,
                    g,
                    control + delta + noise * g,
                )
            )
    return rows


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(ym.LEAF_COLUMNS))


def test_unit_of_maps_tiers_and_rejects_unplanned_pairs() -> None:
    assert ym.unit_of("pythia-160m", "pythia-410m", plan.NEOX_LR, 0) == (
        "M",
        plan.TIER_CORE,
    )
    assert ym.unit_of("pythia-160m", "pythia-410m", plan.NEOX_LR, 2) == (
        "M",
        plan.TIER_G2,
    )
    assert ym.unit_of("pythia-1b", "pythia-1b", 8e-4, 0) == ("M", plan.TIER_LR)
    with pytest.raises(ValueError):
        ym.unit_of("pythia-6.9b", "pythia-2.8b", plan.NEOX_LR, 0)


def test_leaf_deltas_cross_uses_teacher_base_control() -> None:
    rows = _cell("pythia-160m", "pythia-410m", {0: 0.30}, control=0.02)
    rows += [
        _leaf(ym.KIND_SB, "pythia-410m", "pythia-410m", float("nan"), g, 0.01)
        for g in (0, 1)
    ]
    deltas, missing = ym.leaf_deltas(_frame(rows))
    treat = deltas[deltas["kind"] == ym.KIND_TREATMENT]
    assert missing.empty
    assert np.allclose(treat["delta_primary"], 0.30)
    assert np.allclose(treat["delta_tb"], 0.30)
    assert np.allclose(treat["delta_sb"], 0.31)


def test_leaf_deltas_self_uses_student_base_control() -> None:
    deltas, _ = ym.leaf_deltas(_frame(_cell("pythia-410m", "pythia-410m", {0: 0.25})))
    treat = deltas[deltas["kind"] == ym.KIND_TREATMENT]
    assert np.allclose(treat["delta_primary"], 0.25)
    assert treat["delta_tb"].isna().all()


def test_leaf_deltas_missing_primary_control_raises_or_is_listed() -> None:
    rows = [
        r
        for r in _cell("pythia-160m", "pythia-410m", {0: 0.3})
        if r["kind"] != ym.KIND_TB
    ]
    with pytest.raises(KeyError):
        ym.leaf_deltas(_frame(rows))
    deltas, missing = ym.leaf_deltas(_frame(rows), strict=False)
    assert set(missing["kind"]) == {ym.KIND_TB}
    assert len(missing) == 2
    assert deltas["delta_primary"].isna().all()


def test_quarantined_primary_control_flags_the_treatment() -> None:
    rows = _cell("pythia-160m", "pythia-410m", {0: 0.3})
    for r in rows:
        if r["kind"] == ym.KIND_TB:
            r["perplexity_ratio"] = 9.0
    deltas, _ = ym.leaf_deltas(ym.quarantine(_frame(rows)))
    treat = deltas[deltas["kind"] == ym.KIND_TREATMENT]
    assert treat["quarantined"].all()
    assert treat["quarantine_reason"].str.contains("control_quarantined").all()
    assert ym.live_treatments(deltas).empty


def test_quarantine_flags_ppl_and_filter_drop() -> None:
    frame = _frame(_cell("pythia-410m", "pythia-410m", {0: 0.2}))
    frame.loc[frame.index[-1], "perplexity_ratio"] = 5.5
    frame.loc[frame.index[-2], "filter_drop"] = 0.6
    out = ym.quarantine(frame)
    assert out["quarantined"].sum() == 2


def test_cell_ci_takes_the_t_interval_when_it_is_wider() -> None:
    frame = pd.DataFrame(
        {
            "ftseed": [0, 1, 2],
            "gseed": [0, 0, 0],
            "delta_primary": [0.30, 0.33, 0.36],
        }
    )
    stats = ym.mean_ci(frame, "delta_primary", n_boot=500, seed=0)
    boot_width = stats["boot_high"] - stats["boot_low"]
    t_width = stats["t_high"] - stats["t_low"]
    assert t_width > boot_width
    assert (stats["ci_low"], stats["ci_high"]) == (stats["t_low"], stats["t_high"])
    assert math.isclose(stats["mean"], 0.33)


def test_log_odds_delta_is_logit_difference_and_clipped() -> None:
    deltas, _ = ym.leaf_deltas(_frame(_cell("pythia-410m", "pythia-410m", {0: 0.49})))
    treat = deltas[deltas["kind"] == ym.KIND_TREATMENT]
    expected = math.log(0.5 / 0.5) - math.log(0.01 / 0.99)
    assert np.allclose(treat["logodds_primary"], expected)
    assert math.isfinite(ym.logit(0.0)) and math.isfinite(ym.logit(1.0))


def test_efficiency_divides_by_the_matched_teacher_seed() -> None:
    rows = _cell("pythia-410m", "pythia-410m", {0: 0.2, 1: 0.2})
    deltas, _ = ym.leaf_deltas(
        _frame(rows),
        teacher_p_target={("pythia-410m", 0): 0.4, ("pythia-410m", 1): 0.5},
    )
    treat = deltas[deltas["kind"] == ym.KIND_TREATMENT].set_index("ftseed")
    assert np.allclose(treat.loc[0, "efficiency"], 0.5)
    assert np.allclose(treat.loc[1, "efficiency"], 0.4)


def _ratio_law_rows(slope: float) -> list[dict]:
    rng = np.random.default_rng(0)
    rows: list[dict] = []
    for a, b in ym.pythia_pairs():
        gap = ym.log_params(b) - ym.log_params(a)
        up = {f: 0.40 + rng.normal(0, 0.005) for f in plan.TEACHER_FTSEEDS}
        down = {
            f: 0.40 - slope * gap + rng.normal(0, 0.005) for f in plan.TEACHER_FTSEEDS
        }
        rows += _cell(a, b, up) + _cell(b, a, down)
    for s in plan.M_STUDENTS:
        rows += [
            _leaf(ym.KIND_SB, s, s, float("nan"), g, 0.01) for g in plan.CORE_GSEEDS
        ]
    return rows


def test_asymmetry_sign_and_slope_recover_a_planted_ratio_law() -> None:
    deltas, _ = ym.leaf_deltas(_frame(_ratio_law_rows(0.1)))
    asym = ym.asymmetry_table(deltas, n_boot=200, seed=0)
    y3 = asym[asym["claim"] == "Y3"]
    assert len(y3) == 10
    assert (y3["iota"] > 0).all()
    law = ym.asymmetry_law(deltas, n_boot=200, seed=0)
    assert math.isclose(law["slope"], 0.1, abs_tol=0.01)
    assert law["slope_ci"][0] > 0
    assert law["floor_ci"][0] <= 0 <= law["floor_ci"][1]
    assert ym.asymmetry_reading(asym, law) == "ratio_law"


def test_family_contrast_is_paired_on_teacher_seed() -> None:
    teacher_effect = {0: 0.10, 1: 0.40, 2: 0.70}
    rows = _cell(
        "pythia-410m", "amd-olmo-1b", {f: e + 0.05 for f, e in teacher_effect.items()}
    )
    rows += _cell("pythia-410m", "pythia-1b", teacher_effect)
    deltas, _ = ym.leaf_deltas(_frame(rows))
    left = deltas[
        (deltas["student"] == "amd-olmo-1b") & (deltas["kind"] == ym.KIND_TREATMENT)
    ]
    right = deltas[
        (deltas["student"] == "pythia-1b") & (deltas["kind"] == ym.KIND_TREATMENT)
    ]
    paired = ym.contrast_ci(
        left, right, "delta_primary", paired=True, n_boot=300, seed=0
    )
    unpaired = ym.contrast_ci(
        left, right, "delta_primary", paired=False, n_boot=300, seed=0
    )
    assert math.isclose(paired[0], 0.05)
    assert paired[1] > 0
    assert paired[2] - paired[1] < unpaired[2] - unpaired[1]
    family = ym.family_contrasts(deltas, n_boot=100, seed=0)
    row = family[family["contrast"] == "family_given_pythia-410m"].iloc[0]
    assert bool(row["paired"])


def test_bridge_threshold_is_inclusive_of_the_frozen_max_diff() -> None:
    rows = _cell("pythia-410m", "pythia-410m", {0: 0.30})
    deltas, _ = ym.leaf_deltas(_frame(rows))
    part1 = pd.DataFrame(
        [
            {
                "kind": ym.KIND_TREATMENT,
                "teacher": "pythia-410m",
                "student": "pythia-410m",
                "teacher_param_tag": plan.TEACHER_PARAM_TAG,
                "lora_r": 8,
                "epochs": 1.0,
                "lr": plan.NEOX_LR,
                "ftseed": 0,
                "gseed": g,
                "delta_primary": value,
                "quarantined": False,
            }
            for g, value in ((0, 0.28), (1, 0.28), (2, 0.99))
        ]
    )
    table = ym.bridge(deltas, part1).set_index(["teacher", "student"])
    row = table.loc[("pythia-410m", "pythia-410m")]
    assert int(row["n_part1"]) == 2
    assert math.isclose(float(row["diff"]), 0.02)
    assert bool(row["within"])
    part1["delta_primary"] = 0.25
    far = ym.bridge(deltas, part1).set_index(["teacher", "student"])
    assert not bool(far.loc[("pythia-410m", "pythia-410m"), "within"])


def test_ppl_ratio_is_per_gseed_and_coupling_is_not_degenerate() -> None:
    ratios = ym.per_seed_ppl_ratio({"0": 20.0, "1": 30.0}, 10.0)
    assert ratios == {0: 2.0, 1: 3.0}
    rows = _cell("pythia-410m", "pythia-410m", {0: 0.2, 1: 0.3, 2: 0.4})
    for r in rows:
        if r["kind"] == ym.KIND_TREATMENT:
            r["perplexity_ratio"] = 1.0 + 0.1 * r["gseed"] + 0.05 * r["ftseed"]
    deltas, _ = ym.leaf_deltas(_frame(rows))
    coupling = ym.ppl_coupling(ym.live_treatments(deltas))
    assert len(coupling) == 1
    assert not bool(coupling.iloc[0]["degenerate"])


def test_teacher_decisions_need_two_of_three_ftseeds() -> None:
    frame = pd.DataFrame(
        [
            {
                "label": "pythia-1b",
                "tag": plan.TEACHER_PARAM_TAG,
                "ftseed": f,
                "admitted": a,
            }
            for f, a in ((0, True), (1, True), (2, False))
        ]
        + [
            {
                "label": "pythia-6.9b",
                "tag": plan.TEACHER_PARAM_TAG,
                "ftseed": f,
                "admitted": a,
            }
            for f, a in ((0, True), (1, False), (2, False))
        ]
    )
    decisions = ym.teacher_decisions(frame)
    assert decisions["pythia-1b"] == {"usable": True, "admitted_ftseeds": [0, 1]}
    assert decisions["pythia-6.9b"]["usable"] is False
    leaves = _frame(_cell("pythia-1b", "pythia-1b", {0: 0.2, 2: 0.2}))
    marked = ym.mark_teacher_admission(leaves, decisions)
    treat = marked[marked["kind"] == ym.KIND_TREATMENT]
    assert treat.set_index("ftseed")["teacher_admitted"].groupby(
        level=0
    ).all().to_dict() == {
        0: True,
        2: False,
    }


def test_cell_table_status_and_void_teacher() -> None:
    rows = _cell("pythia-410m", "pythia-410m", {0: 0.3, 1: 0.31, 2: 0.29}, noise=0.001)
    deltas, _ = ym.leaf_deltas(_frame(rows))
    cells = ym.cell_table(deltas, n_boot=100, seed=0).set_index(
        ["teacher", "student", "lr"]
    )
    row = cells.loc[("pythia-410m", "pythia-410m", plan.NEOX_LR)]
    assert row["status"] == ym.STATUS_HOLDS and bool(row["complete"])
    assert (
        cells.loc[("pythia-160m", "pythia-410m", plan.NEOX_LR), "status"]
        == ym.STATUS_INCOMPLETE
    )
    void = ym.cell_table(
        deltas,
        {"pythia-410m": {"usable": False, "admitted_ftseeds": []}},
        n_boot=10,
        seed=0,
    ).set_index(["teacher", "student", "lr"])
    assert (
        void.loc[("pythia-410m", "pythia-410m", plan.NEOX_LR), "status"]
        == ym.STATUS_VOID
    )
