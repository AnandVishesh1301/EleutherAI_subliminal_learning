"""Planted-fixture tests for cross-size X0–X4 frames."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from subliminal_learning.campaign.xsize_metrics import (
    KIND_SB,
    KIND_TB,
    KIND_TREATMENT,
    additive_fit,
    base_priors,
    cell_means,
    claims,
    direction_asymmetry,
    grid_concordance,
    leaf_deltas,
    lr_stability,
    ppl_coupling,
    propagate_control_quarantine,
    quarantine,
    shape_penalty,
)
from subliminal_learning.campaign.xsize_plan import (
    GRID_REFERENCE,
    LR_DEFAULT,
    LR_LADDER,
    SIZES,
    student_gkd_tag,
)
from subliminal_learning.campaign.xsize_selection import TeacherGateDecision

_ANIMAL = "bear"
_BOOT = 64
_SEED = 20260907
_PICKS = {size: LR_DEFAULT for size in SIZES}


def _row(
    *,
    teacher: str,
    student: str,
    kind: str,
    p_target: float,
    lr: float = LR_DEFAULT,
    gseed: int = 0,
    ftseed: float | int = 0,
    phase: str | None = "p3",
    ppl: float = 1.0,
    filter_drop: float = 0.0,
    quarantined: bool = False,
) -> dict:
    return {
        "animal": _ANIMAL,
        "student": student,
        "teacher": teacher,
        "kind": kind,
        "phase": None if kind != KIND_TREATMENT else phase,
        "lr": lr,
        "gkd_tag": student_gkd_tag(lr),
        "ftseed": float("nan") if kind != KIND_TREATMENT else ftseed,
        "gseed": gseed,
        "p_target": p_target,
        "perplexity_ratio": ppl,
        "filter_drop": filter_drop,
        "quarantined": quarantined,
    }


def _sb(student: str, p: float, **kwargs: object) -> dict:
    return _row(teacher=student, student=student, kind=KIND_SB, p_target=p, **kwargs)


def _tb(teacher: str, student: str, p: float, **kwargs: object) -> dict:
    return _row(teacher=teacher, student=student, kind=KIND_TB, p_target=p, **kwargs)


def _treat(teacher: str, student: str, p: float, **kwargs: object) -> dict:
    return _row(
        teacher=teacher, student=student, kind=KIND_TREATMENT, p_target=p, **kwargs
    )


def test_leaf_deltas_join_sb_and_tb_and_copy_anchor_tb() -> None:
    frame = pd.DataFrame(
        [
            _treat("70m", "160m", 0.40, gseed=0),
            _treat("160m", "160m", 0.50, gseed=0),
            _sb("160m", 0.10, gseed=0),
            _tb("70m", "160m", 0.15, gseed=0),
        ]
    )
    deltas = leaf_deltas(frame)
    cross = deltas[(deltas["teacher"] == "70m") & (deltas["student"] == "160m")]
    anchor = deltas[(deltas["teacher"] == "160m") & (deltas["student"] == "160m")]
    assert float(cross["delta_sb"].iloc[0]) == pytest.approx(0.30)
    assert float(cross["delta_tb"].iloc[0]) == pytest.approx(0.25)
    assert float(cross["delta_primary"].iloc[0]) == pytest.approx(0.25)
    assert float(anchor["delta_sb"].iloc[0]) == pytest.approx(0.40)
    assert float(anchor["delta_tb"].iloc[0]) == pytest.approx(0.40)
    assert float(anchor["delta_primary"].iloc[0]) == pytest.approx(0.40)


def test_leaf_deltas_raises_on_missing_matched_control() -> None:
    missing_tb = pd.DataFrame(
        [
            _treat("70m", "160m", 0.40),
            _sb("160m", 0.10),
        ]
    )
    with pytest.raises(KeyError) as err:
        leaf_deltas(missing_tb)
    assert "70m" in str(err.value) or err.value.args
    missing_sb = pd.DataFrame(
        [
            _treat("160m", "160m", 0.40),
        ]
    )
    with pytest.raises(KeyError):
        leaf_deltas(missing_sb)


def test_quarantined_control_quarantines_its_treatment_instead_of_raising() -> None:
    """A control that ran but was excluded must not look like one that never ran.

    `leaf_deltas` drops the quarantined control from the control mean, which
    left its treatment with no denominator and raised `KeyError` — the signal
    reserved for a control that is genuinely absent. This is the 70m student
    row of `20260907_xsize-bear_mixed`, whose sb controls all sit above the
    PPL bound.
    """
    frame = pd.DataFrame(
        [
            _treat("160m", "160m", 0.50, gseed=0, ppl=1.1),
            _sb("160m", 0.10, gseed=0, ppl=9.0),
            _treat("70m", "160m", 0.40, gseed=1, ppl=1.1),
            _sb("160m", 0.10, gseed=1, ppl=1.0),
            _tb("70m", "160m", 0.15, gseed=1, ppl=9.0),
        ]
    )
    flagged = propagate_control_quarantine(quarantine(frame))
    treatments = flagged[flagged["kind"] == KIND_TREATMENT]
    assert bool(treatments["quarantined"].all())
    assert set(treatments["quarantine_reason"]) == {"control_quarantined"}
    # Both treatments lost their denominator, so neither yields a Δ — and the
    # read completes instead of aborting the whole campaign finalize.
    assert leaf_deltas(flagged).empty


def test_leaf_deltas_still_raises_when_a_live_treatment_has_no_control() -> None:
    """Propagation must not turn a genuinely absent control into a silent drop."""
    frame = pd.DataFrame([_treat("160m", "160m", 0.40, ppl=1.1)])
    flagged = propagate_control_quarantine(quarantine(frame))
    assert not bool(flagged["quarantined"].any())
    with pytest.raises(KeyError):
        leaf_deltas(flagged)


def test_quarantine_flags_ppl_ratio_six_and_cell_means_drop_it() -> None:
    frame = pd.DataFrame(
        [
            _treat("160m", "160m", 0.50, gseed=0, ppl=1.1),
            _treat("160m", "160m", 0.90, gseed=1, ppl=6.0),
            _sb("160m", 0.10, gseed=0),
            _sb("160m", 0.10, gseed=1),
        ]
    )
    flagged = quarantine(frame)
    hot = flagged[flagged["perplexity_ratio"] == 6.0].iloc[0]
    assert bool(hot["quarantined"]) is True
    assert "ppl_ratio" in hot["quarantine_reason"]
    deltas = leaf_deltas(flagged)
    cells = cell_means(deltas, n_boot=_BOOT, seed=_SEED)
    row = cells.iloc[0]
    assert int(row["n"]) == 1
    assert float(row["mean_delta_primary"]) == pytest.approx(0.40)


def test_x0_is_void_when_the_matched_anchor_cell_is_absent() -> None:
    """No anchor cell -> void, never read off the cross row alone.

    X0 is reported only where the matched anchor's CI excludes 0; with the
    anchor missing that precondition cannot be established, so a healthy-looking
    cross row (0.30, CI clear of 0) must not be scored as existence.
    """
    frame = pd.DataFrame(
        [
            _treat("70m", "160m", 0.40, gseed=0),
            _treat("70m", "160m", 0.44, gseed=1),
            _sb("160m", 0.10, gseed=0),
            _sb("160m", 0.10, gseed=1),
            _tb("70m", "160m", 0.10, gseed=0),
            _tb("70m", "160m", 0.10, gseed=1),
        ]
    )
    deltas = leaf_deltas(frame)
    cells = cell_means(deltas, n_boot=_BOOT, seed=_SEED)
    cross = cells[(cells["teacher"] == "70m") & (cells["student"] == "160m")].iloc[0]
    assert float(cross["mean_delta_primary"]) == pytest.approx(0.32)
    verdict = claims(cells, picks=_PICKS, deltas=deltas)
    pair = verdict["X0"]["pairs"]["70m→160m"]
    assert pair["status"] == "void"
    assert pair["reason"] == "missing_anchor"
    assert verdict["X0"]["status"] == "void"


def test_x1_fails_when_the_cross_pair_beats_its_anchor() -> None:
    """A significantly negative penalty is a cross-size advantage, not a cost."""
    rows: list[dict] = []
    for gseed, bump in enumerate((0.0, 0.02, 0.04)):
        rows += [
            _treat("160m", "160m", 0.30 + bump, gseed=gseed),
            _treat("70m", "160m", 0.60 + bump, gseed=gseed),
            _sb("160m", 0.10, gseed=gseed),
            _tb("70m", "160m", 0.10, gseed=gseed),
        ]
    deltas = leaf_deltas(pd.DataFrame(rows))
    cells = cell_means(deltas, n_boot=_BOOT, seed=_SEED)
    pen = shape_penalty(cells, picks=_PICKS, deltas=deltas, n_boot=_BOOT, seed=_SEED)
    row = pen[(pen["teacher"] == "70m") & (pen["student"] == "160m")].iloc[0]
    assert float(row["penalty"]) == pytest.approx(-0.30)
    verdict = claims(cells, picks=_PICKS, penalty=pen, deltas=deltas)
    scored = verdict["X1"]["pairs"]["70m→160m"]
    assert scored["penalty_sig"] is True
    assert scored["penalty_holds"] is False
    assert verdict["X1"]["status"] == "fails"


def test_shape_penalty_on_hand_numbers() -> None:
    frame = pd.DataFrame(
        [
            _treat("70m", "160m", 0.30),
            _treat("160m", "160m", 0.50),
            _treat("70m", "70m", 0.40),
            _sb("160m", 0.10),
            _sb("70m", 0.10),
            _tb("70m", "160m", 0.10),
        ]
    )
    deltas = leaf_deltas(frame)
    cells = cell_means(deltas, n_boot=_BOOT, seed=_SEED)
    pen = shape_penalty(
        cells,
        picks=_PICKS,
        deltas=deltas,
        teacher_p_target={"70m": 0.40},
        n_boot=_BOOT,
        seed=_SEED,
    )
    row = pen[(pen["teacher"] == "70m") & (pen["student"] == "160m")].iloc[0]
    assert float(row["delta_tb"]) == pytest.approx(0.20)
    assert float(row["delta_anchor_s"]) == pytest.approx(0.40)
    assert float(row["delta_anchor_t"]) == pytest.approx(0.30)
    assert float(row["rho_s"]) == pytest.approx(0.50)
    assert float(row["rho_t"]) == pytest.approx(0.20 / 0.30)
    assert float(row["penalty"]) == pytest.approx(0.20)
    assert float(row["efficiency"]) == pytest.approx(0.20 / 0.40)
    assert float(row["grid_reference_delta"]) == GRID_REFERENCE["di_ds_delta"]


def test_direction_asymmetry_sign_and_symmetric_zero() -> None:
    asym = pd.DataFrame(
        [
            _treat("70m", "160m", 0.40),
            _treat("160m", "70m", 0.20),
            _sb("160m", 0.10),
            _sb("70m", 0.10),
            _tb("70m", "160m", 0.10),
            _tb("160m", "70m", 0.10),
        ]
    )
    deltas = leaf_deltas(asym)
    cells = cell_means(deltas, n_boot=_BOOT, seed=_SEED)
    frame = direction_asymmetry(
        cells, picks=_PICKS, deltas=deltas, n_boot=_BOOT, seed=_SEED
    )
    row = frame[(frame["size_lo"] == "70m") & (frame["size_hi"] == "160m")].iloc[0]
    assert float(row["iota"]) == pytest.approx(0.20)
    assert float(row["delta_up"]) == pytest.approx(0.30)
    assert float(row["delta_down"]) == pytest.approx(0.10)

    sym_rows = [
        _treat("70m", "160m", 0.30),
        _treat("160m", "70m", 0.30),
        _sb("160m", 0.10),
        _sb("70m", 0.10),
        _tb("70m", "160m", 0.10),
        _tb("160m", "70m", 0.10),
    ]
    sym = direction_asymmetry(
        cell_means(leaf_deltas(pd.DataFrame(sym_rows)), n_boot=_BOOT, seed=_SEED),
        picks=_PICKS,
        n_boot=_BOOT,
        seed=_SEED,
    )
    srow = sym[(sym["size_lo"] == "70m") & (sym["size_hi"] == "160m")].iloc[0]
    assert float(srow["iota"]) == pytest.approx(0.0)


def _additive_frame(*, bump: float = 0.0) -> pd.DataFrame:
    # τ = 0.25 + σ, σ = {-0.05, 0, 0.05} ⇒ Δ_ij = 0.25 + σ_i + σ_j, ι = 0.
    sigma = {"70m": -0.05, "160m": 0.0, "410m": 0.05}
    rows: list[dict] = []
    for teacher in SIZES:
        for student in SIZES:
            delta = 0.25 + sigma[teacher] + sigma[student]
            if teacher == "70m" and student == "410m":
                delta += bump
            p = 0.10 + delta
            rows.append(_treat(teacher, student, p, gseed=0))
            rows.append(_treat(teacher, student, p, gseed=1))
            if teacher == student:
                rows.append(_sb(student, 0.10, gseed=0))
                rows.append(_sb(student, 0.10, gseed=1))
            else:
                rows.append(_tb(teacher, student, 0.10, gseed=0))
                rows.append(_tb(teacher, student, 0.10, gseed=1))
    return pd.DataFrame(rows)


def test_additive_fit_recovers_planted_tau_sigma() -> None:
    deltas = leaf_deltas(_additive_frame())
    cells = cell_means(deltas, n_boot=_BOOT, seed=_SEED)
    fit = additive_fit(cells, picks=_PICKS, deltas=deltas, n_boot=_BOOT, seed=_SEED)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-9)
    assert sum(fit["sigma"].values()) == pytest.approx(0.0, abs=1e-9)
    assert fit["sigma"]["70m"] == pytest.approx(-0.05, abs=1e-6)
    assert fit["sigma"]["410m"] == pytest.approx(0.05, abs=1e-6)
    assert fit["tau"]["70m"] == pytest.approx(0.20, abs=1e-6)
    assert fit["tau"]["410m"] == pytest.approx(0.30, abs=1e-6)
    assert fit["pooled_iota"] == pytest.approx(0.0, abs=1e-9)
    assert fit["pooled_iota_ci_low"] <= 0.0 <= fit["pooled_iota_ci_high"]
    for r2 in fit["loo_r2"].values():
        assert r2 == pytest.approx(1.0, abs=1e-9)


def test_additive_fit_breaks_on_planted_interaction() -> None:
    deltas = leaf_deltas(_additive_frame(bump=0.20))
    cells = cell_means(deltas, n_boot=_BOOT, seed=_SEED)
    fit = additive_fit(cells, picks=_PICKS, n_boot=_BOOT, seed=_SEED)
    assert fit["r2"] < 0.99


def test_lr_stability_kendall_tau_is_one_when_ranking_is_stable() -> None:
    ranks = {
        ("70m", "160m"): 0.30,
        ("160m", "70m"): 0.20,
        ("160m", "410m"): 0.10,
        ("410m", "160m"): 0.05,
    }
    rows: list[dict] = []
    for lr in LR_LADDER:
        for (teacher, student), delta in ranks.items():
            p = 0.10 + delta
            rows.append(_treat(teacher, student, p, lr=lr, phase="p3b"))
            rows.append(_sb(student, 0.10, lr=lr))
            rows.append(_tb(teacher, student, 0.10, lr=lr))
    deltas = leaf_deltas(pd.DataFrame(rows))
    cells = cell_means(deltas, n_boot=_BOOT, seed=_SEED)
    stab = lr_stability(cells, picks=_PICKS)
    assert not stab.empty
    assert all(float(v) == pytest.approx(1.0) for v in stab["kendall_tau"])
    assert set(stab["sign"]) <= {1}


def test_ppl_coupling_retains_the_planted_fraction() -> None:
    rows: list[dict] = []
    for i, ppl in enumerate((1.0, 2.0, 3.0)):
        delta = 0.10 + 0.5 * (ppl - 1.0)
        rows.append(_treat("70m", "160m", 0.10 + delta, gseed=i, ppl=ppl))
        rows.append(_sb("160m", 0.10, gseed=i, ppl=1.0))
        rows.append(_tb("70m", "160m", 0.10, gseed=i, ppl=1.0))
    deltas = leaf_deltas(pd.DataFrame(rows))
    coup = ppl_coupling(deltas)
    row = coup.iloc[0]
    mean_delta = (0.10 + 0.60 + 1.10) / 3.0
    assert float(row["beta"]) == pytest.approx(0.5, abs=1e-6)
    assert float(row["mean_delta_adj"]) == pytest.approx(0.10, abs=1e-6)
    assert float(row["retained_fraction"]) == pytest.approx(0.10 / mean_delta)


def test_base_priors_and_grid_concordance_and_claims() -> None:
    priors = base_priors(
        [
            {"size": "70m", "p_target": 0.01, "fc_share": 0.1, "ppl": 12.0},
            {"base": "160m", "p_target": 0.02, "forced_choice_share": 0.2},
        ]
    )
    assert list(priors["size"]) == ["70m", "160m"]
    assert float(priors.loc[0, "perplexity"]) == 12.0

    deltas = leaf_deltas(_additive_frame())
    cells = cell_means(deltas, n_boot=_BOOT, seed=_SEED)
    conc = grid_concordance(cells, picks=_PICKS)
    assert conc.iloc[0]["ref_anchor_delta"] == GRID_REFERENCE["anchor_delta"]
    assert conc.iloc[0]["ref_di_ds_delta"] == GRID_REFERENCE["di_ds_delta"]
    assert math.isfinite(float(conc.iloc[0]["anchor_delta"]))

    decisions = {
        size: TeacherGateDecision(
            size=size,
            tag="ep5_lr5e-5_r128a128",
            status="frozen",
            n_admitted_frozen=3,
            fallback_candidates=(),
            reasons=(),
        )
        for size in SIZES
    }
    decisions["70m"] = TeacherGateDecision(
        size="70m",
        tag=None,
        status="no_admissible_teacher",
        n_admitted_frozen=0,
        fallback_candidates=(),
        reasons=("no_admitted_fallback",),
    )
    pen = shape_penalty(cells, picks=_PICKS, deltas=deltas, n_boot=_BOOT, seed=_SEED)
    asym = direction_asymmetry(
        cells, picks=_PICKS, deltas=deltas, n_boot=_BOOT, seed=_SEED
    )
    add = additive_fit(cells, picks=_PICKS, deltas=deltas, n_boot=_BOOT, seed=_SEED)
    verdict = claims(
        cells,
        picks=_PICKS,
        teacher_decisions=decisions,
        penalty=pen,
        asymmetry=asym,
        additive=add,
        ppl=ppl_coupling(deltas),
    )
    assert set(verdict) == {"X0", "X1", "X2", "X3", "X4"}
    assert verdict["X0"]["pairs"]["70m→160m"]["status"] == "void"
    assert verdict["X0"]["pairs"]["70m→410m"]["status"] == "void"
    assert verdict["X2"]["status"] == "holds"
    assert verdict["X2"]["read"] == "separable"
    assert verdict["X3"]["status"] == "descriptive"
    assert verdict["X4"]["status"] in {"holds", "fails", "void"}
