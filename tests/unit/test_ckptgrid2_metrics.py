"""Planted-fixture tests for checkpoint-grid-2 H1–H12 frames."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.campaign import ckptgrid2_metrics as cm
from subliminal_learning.campaign.ckptgrid2_metrics import (
    ADJUDICATORS,
    ALLOWED_STATUSES,
    CLAIM_IDS,
    STATUS_INSUFFICIENT,
    adjudicate_h1,
    adjudicate_h2,
    adjudicate_h3,
    adjudicate_h4,
    adjudicate_h5,
    adjudicate_h6,
    adjudicate_h7,
    adjudicate_h8,
    adjudicate_h9,
    adjudicate_h10,
    adjudicate_h11,
    adjudicate_h12,
    build_asymmetry_by_era,
    build_base_ppl,
    build_dose_matched,
    build_dose_response,
    build_emissivity_fit,
    build_hp_collapse,
    build_metric_concordance,
    build_onset_floor,
    build_penalty_map,
    build_ridge_gain,
    build_teacher_dose,
    build_triangle,
)

_SET = "test"
_SCORE = "prob"
_BOOT = 64
_SEED = 20260908


def _delta_row(
    *,
    student_step: int,
    teacher_step: int,
    delta: float,
    student_base: str = "original",
    teacher_base: str = "original",
    cell: str = "SI_SS",
    animal: str = "bear",
    gkd_tag: str = "frozen",
    teacher_param_tag: str = "ep5_lr5e-5_r128a128",
    n_epochs: int | None = 5,
    level: str = "agg",
    control_mean: float = 0.0,
    p_target: float | None = None,
    perplexity_ratio: float | None = None,
    source_run_set: str = "new",
    prompt_set: str = _SET,
    scoring: str = _SCORE,
    n_leaves: int = 3,
) -> dict:
    return {
        "level": level,
        "arm": "rect",
        "animal": animal,
        "student_base": student_base,
        "teacher_base": teacher_base,
        "student_step": student_step,
        "teacher_step": teacher_step,
        "cell": cell,
        "gkd_tag": gkd_tag,
        "teacher_param_tag": teacher_param_tag,
        "source_run_set": source_run_set,
        "ftseed": None,
        "gseed": None,
        "set": prompt_set,
        "scoring": scoring,
        "p_target": (delta + control_mean) if p_target is None else p_target,
        "control_mean": control_mean,
        "delta": delta,
        "n_leaves": n_leaves,
        "n_epochs": n_epochs,
        "perplexity_ratio": perplexity_ratio,
    }


def _teacher_row(
    step: int,
    p_target: float,
    *,
    n_epochs: int = 1,
    ftseed: int = 0,
    is_control: bool = False,
    base: str = "original",
) -> dict:
    return {
        "base": base,
        "step": step,
        "ftseed": ftseed,
        "n_epochs": n_epochs,
        "param_tag": f"ep{n_epochs}_lr5e-5_r128a128",
        "set": _SET,
        "scoring": _SCORE,
        "p_target": p_target,
        "is_control": is_control,
        "source_run_set": "new",
    }


def test_teacher_dose_mean_sd_and_optional_columns() -> None:
    rows = []
    for ftseed, offset in ((0, 0.0), (1, 0.02)):
        rows.append(_teacher_row(2000, 0.10 + offset, n_epochs=1, ftseed=ftseed))
        rows.append(_teacher_row(2000, 0.20 + offset, n_epochs=2, ftseed=ftseed))
    rows.append(_teacher_row(2000, 0.01, n_epochs=1, is_control=True))
    frame = build_teacher_dose(
        pd.DataFrame(rows), rungs=(2000,), bases=("original",), epochs=(1, 2)
    )
    by_ep = frame.set_index("n_epochs")
    assert by_ep.loc[1, "p_target_mean"] == pytest.approx(0.11)
    assert by_ep.loc[2, "p_target_mean"] == pytest.approx(0.21)
    assert by_ep.loc[1, "n_ftseed"] == 2
    assert by_ep.loc[1, "control_mean"] == pytest.approx(0.01)


def test_dose_response_overshoot_signs_and_degenerate_guard() -> None:
    # Concave-ish: teacher dose 0.10, 0.25, 0.35; student delta 0.20, 0.30, 0.32.
    rungs = (2000, 8000, 48000)
    delta = pd.DataFrame(
        [
            _delta_row(student_step=r, teacher_step=r, delta=d, n_epochs=e)
            for r, d, e in ((2000, 0.20, 1), (8000, 0.30, 2), (48000, 0.32, 3))
        ]
    )
    dose = pd.DataFrame(
        [
            {
                "base": "original",
                "step": r,
                "n_epochs": e,
                "p_target_mean": p,
            }
            for r, e, p in ((2000, 1, 0.10), (8000, 2, 0.25), (48000, 3, 0.35))
        ]
    )
    frame = build_dose_response(delta, dose, rungs=rungs)
    by_s = frame.set_index("student_step")
    assert by_s.loc[2000, "overshoot"] == pytest.approx(0.10)
    assert by_s.loc[8000, "overshoot"] == pytest.approx(0.05)
    assert by_s.loc[48000, "overshoot"] == pytest.approx(-0.03)
    assert by_s.loc[2000, "efficiency"] == pytest.approx(2.0)
    assert bool(by_s.loc[2000, "degenerate"]) is False

    zero = pd.DataFrame(
        [_delta_row(student_step=2000, teacher_step=2000, delta=0.10, n_epochs=1)]
    )
    dose_bad = pd.DataFrame(
        [
            {"base": "original", "step": 2000, "n_epochs": 1, "p_target_mean": 0.0},
        ]
    )
    bad = build_dose_response(zero, dose_bad, rungs=(2000,))
    assert pd.isna(bad["efficiency"].iloc[0])
    assert bool(bad["degenerate"].iloc[0]) is True
    assert not any(math.isinf(float(v)) for v in bad["efficiency"] if pd.notna(v))

    neg = pd.DataFrame(
        [{"base": "original", "step": 2000, "n_epochs": 1, "p_target_mean": -0.2}]
    )
    neg_frame = build_dose_response(zero, neg, rungs=(2000,))
    assert pd.isna(neg_frame["efficiency"].iloc[0])
    assert bool(neg_frame["degenerate"].iloc[0]) is True


def test_dose_matched_tost_zero_vs_rung_effect() -> None:
    rungs = (2000, 8000, 48000, 143000)
    doses = (0.10, 0.20, 0.30)
    only_dose = []
    with_rung = []
    for rung in rungs:
        for dose in doses:
            only_dose.append(
                {
                    "student_step": rung,
                    "teacher_step": rung,
                    "delta": dose,
                    "teacher_p_target": dose,
                    "degenerate": False,
                }
            )
            with_rung.append(
                {
                    "student_step": rung,
                    "teacher_step": rung,
                    "delta": dose + 0.25 * math.log10(rung),
                    "teacher_p_target": dose,
                    "degenerate": False,
                }
            )
    flat = build_dose_matched(
        pd.DataFrame(only_dose), tol=0.05, bound=0.05, n_boot=_BOOT, seed=_SEED
    )
    assert bool(flat["tost_equivalent"].iloc[0]) is True
    assert flat["rung_coef"].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert int(flat["n_bands"].iloc[0]) == 3
    assert int(flat["n_cells"].iloc[0]) == 12

    rising = build_dose_matched(
        pd.DataFrame(with_rung), tol=0.05, bound=0.05, n_boot=_BOOT, seed=_SEED
    )
    assert rising["rung_coef"].iloc[0] == pytest.approx(0.25, abs=0.05)
    assert float(rising["ci_low"].iloc[0]) > 0.0
    assert float(rising["ci_high"].iloc[0]) > 0.0


def test_ridge_gain_ridge_flat_era_and_degenerate() -> None:
    rows = []
    # Ridge column at 8000 (early): diag 0.40, offs 0.10 / 0.10.
    for s, d, src in (
        (8000, 0.40, "reference"),
        (2000, 0.10, "new"),
        (16000, 0.10, "new"),
    ):
        rows.append(
            _delta_row(
                student_step=s,
                teacher_step=8000,
                delta=d,
                cell="DI_DS",
                teacher_base="seed2",
                source_run_set=src,
            )
        )
    # Flat column at 48000 (late): all 0.20.
    for s in (2000, 48000, 143000):
        rows.append(
            _delta_row(
                student_step=s,
                teacher_step=48000,
                delta=0.20,
                cell="DI_DS",
                teacher_base="seed2",
            )
        )
    # Missing diagonal at 143000.
    rows.append(
        _delta_row(
            student_step=2000,
            teacher_step=143000,
            delta=0.05,
            cell="DI_DS",
            teacher_base="seed2",
        )
    )
    frame = build_ridge_gain(
        pd.DataFrame(rows),
        rungs=(8000, 48000, 143000),
        early_max_step=16000,
        n_boot=_BOOT,
        seed=_SEED,
    )
    by_t = frame.set_index("teacher_step")
    assert by_t.loc[8000, "ridge_gain"] == pytest.approx(0.30)
    assert by_t.loc[8000, "era"] == "early"
    assert bool(by_t.loc[8000, "degenerate"]) is False
    assert by_t.loc[48000, "ridge_gain"] == pytest.approx(0.0)
    assert by_t.loc[48000, "era"] == "late"
    assert bool(by_t.loc[143000, "degenerate"]) is True
    assert pd.isna(by_t.loc[143000, "ridge_gain"])


def test_penalty_map_degenerate_reference() -> None:
    rows = [
        _delta_row(
            student_step=8000,
            teacher_step=2000,
            delta=0.20,
            cell="DI_DS",
            teacher_base="seed2",
        ),
        _delta_row(
            student_step=8000,
            teacher_step=2000,
            delta=0.0,
            cell="SI_SS",
            teacher_base="original",
        ),
        _delta_row(
            student_step=8000,
            teacher_step=8000,
            delta=0.20,
            cell="DI_DS",
            teacher_base="seed2",
        ),
        _delta_row(
            student_step=8000,
            teacher_step=8000,
            delta=0.40,
            cell="SI_SS",
            teacher_base="original",
        ),
    ]
    frame = build_penalty_map(pd.DataFrame(rows))
    di = frame[frame["cell"] == "DI_DS"].set_index("teacher_step")
    assert bool(di.loc[2000, "degenerate"]) is True
    assert pd.isna(di.loc[2000, "penalty"])
    assert bool(di.loc[8000, "degenerate"]) is False
    assert di.loc[8000, "penalty"] == pytest.approx(0.5)


def test_emissivity_fit_recovers_planted_tau_sigma() -> None:
    mu = 0.20
    tau = {"original": 0.10, "seed2": -0.05, "data-seed1": -0.05}
    sigma = {"original": 0.06, "seed2": 0.00, "data-seed1": -0.06}
    rows = []
    for student, s_val in sigma.items():
        for teacher, t_val in tau.items():
            cell = "SI_SS" if student == teacher else "DI_DS"
            rows.append(
                _delta_row(
                    student_step=8000,
                    teacher_step=8000,
                    delta=mu + t_val + s_val,
                    student_base=student,
                    teacher_base=teacher,
                    cell=cell,
                )
            )
    frame = build_emissivity_fit(pd.DataFrame(rows), per_rung=True)
    by_base = frame.set_index("base")
    for base, t_val in tau.items():
        assert by_base.loc[base, "tau"] == pytest.approx(t_val, abs=1e-8)
        assert by_base.loc[base, "sigma"] == pytest.approx(sigma[base], abs=1e-8)
        assert by_base.loc[base, "tau_minus_sigma"] == pytest.approx(
            t_val - sigma[base], abs=1e-8
        )
    assert frame["mu"].iloc[0] == pytest.approx(mu, abs=1e-8)
    assert frame["resid_rms"].iloc[0] == pytest.approx(0.0, abs=1e-8)


def test_onset_floor_absolute_floor_beats_tiny_sd() -> None:
    # G6 bug: 2*sd = 2e-5 would be cleared by delta=2e-4; abs floor 0.02 is not.
    d = 1e-5 / (2**0.5)
    control = pd.DataFrame(
        [
            {
                "student_base": "original",
                "student_step": 512,
                "gkd_tag": "frozen",
                "gseed": 0,
                "p_target": 1e-4 - d,
                "is_control": True,
                "set": _SET,
                "scoring": _SCORE,
            },
            {
                "student_base": "original",
                "student_step": 512,
                "gkd_tag": "frozen",
                "gseed": 1,
                "p_target": 1e-4 + d,
                "is_control": True,
                "set": _SET,
                "scoring": _SCORE,
            },
        ]
    )
    delta = pd.DataFrame(
        [
            _delta_row(
                student_step=512,
                teacher_step=8000,
                delta=2e-4,
                control_mean=1e-4,
                cell="SI_SS",
            )
        ]
    )
    frame = build_onset_floor(
        delta,
        control,
        rungs=(512,),
        sd_multiple=2.0,
        abs_floor=0.02,
        consecutive=2,
    )
    assert len(frame) == 1
    assert bool(frame["clears_floor"].iloc[0]) is False
    assert frame["threshold"].iloc[0] == pytest.approx(0.02)
    assert pd.isna(frame["onset_step"].iloc[0])
    assert bool(frame["censored"].iloc[0]) is True


def test_onset_floor_single_rung_h12_row_per_cell() -> None:
    rows = []
    controls = []
    for teacher_step in (2000, 8000):
        for student_base, teacher_base in (
            ("original", "seed2"),
            ("seed2", "original"),
        ):
            rows.append(
                _delta_row(
                    student_step=512,
                    teacher_step=teacher_step,
                    delta=0.001,
                    student_base=student_base,
                    teacher_base=teacher_base,
                    cell="DI_DS",
                    control_mean=0.0,
                )
            )
            controls.append(
                {
                    "student_base": student_base,
                    "student_step": 512,
                    "gkd_tag": "frozen",
                    "gseed": 0,
                    "p_target": 0.0,
                    "is_control": True,
                    "set": _SET,
                    "scoring": _SCORE,
                }
            )
    frame = build_onset_floor(
        pd.DataFrame(rows),
        pd.DataFrame(controls),
        rungs=(512,),
        consecutive=2,
    )
    assert len(frame) == 4
    assert set(frame["teacher_step"]) == {2000, 8000}
    assert frame["clears_floor"].astype(bool).eq(False).all()
    assert frame["censored"].astype(bool).all()
    assert frame["onset_step"].isna().all()


def test_metric_concordance_perfect_twin() -> None:
    rows = []
    for step, delta in ((2000, 0.10), (8000, 0.30), (48000, 0.20)):
        rows.append(
            _delta_row(
                student_step=step, teacher_step=step, delta=delta, prompt_set="test"
            )
        )
        rows.append(
            _delta_row(
                student_step=step,
                teacher_step=step,
                delta=delta,
                prompt_set="test_sample",
                scoring="sample",
            )
        )
    frame = build_metric_concordance(pd.DataFrame(rows), subset="diagonal")
    assert frame["spearman"].iloc[0] == pytest.approx(1.0)
    assert frame["sign_agreement"].iloc[0] == pytest.approx(1.0)
    assert frame["ba_mean"].iloc[0] == pytest.approx(0.0)
    assert int(frame["n_cells"].iloc[0]) == 3


def test_hp_collapse_flags_and_best_tag() -> None:
    rows = [
        _delta_row(
            student_step=2000,
            teacher_step=2000,
            delta=0.10,
            gkd_tag="lr1e-4",
            perplexity_ratio=1.2,
        ),
        _delta_row(
            student_step=2000,
            teacher_step=2000,
            delta=0.40,
            gkd_tag="lr1.6e-3",
            perplexity_ratio=1.1,
        ),
        _delta_row(
            student_step=2000,
            teacher_step=2000,
            delta=0.90,
            gkd_tag="lr8e-4",
            perplexity_ratio=6.0,
        ),
        _delta_row(
            student_step=8000,
            teacher_step=8000,
            delta=0.20,
            gkd_tag="lr1e-4",
            perplexity_ratio=1.3,
        ),
    ]
    frame = build_hp_collapse(pd.DataFrame(rows), ppl_max=5.0)
    at_2000 = frame[frame["student_step"] == 2000].set_index("gkd_tag")
    assert bool(at_2000.loc["lr8e-4", "collapsed"]) is True
    assert bool(at_2000.loc["lr1e-4", "collapsed"]) is False
    assert at_2000.loc["lr1e-4", "best_lr_at_rung"] == "lr1.6e-3"
    assert at_2000.loc["lr1.6e-3", "mean_ppl_ratio"] == pytest.approx(1.1)


def test_asymmetry_by_era_matches_hand_values() -> None:
    # Younger 2000 (<4000): asym = 0.40 - 0.10 = 0.30
    # Younger 8000 (4000-32000): asym = 0.25 - 0.05 = 0.20
    rows = [
        _delta_row(student_step=2000, teacher_step=8000, delta=0.40),
        _delta_row(student_step=8000, teacher_step=2000, delta=0.10),
        _delta_row(student_step=8000, teacher_step=48000, delta=0.25),
        _delta_row(student_step=48000, teacher_step=8000, delta=0.05),
    ]
    frame = build_asymmetry_by_era(
        pd.DataFrame(rows), era_edges=(4000, 32000), n_boot=_BOOT, seed=_SEED
    )
    by_era = frame.set_index("era")
    assert by_era.loc["lt_4000", "asymmetry_mean"] == pytest.approx(0.30)
    assert by_era.loc["4000_32000", "asymmetry_mean"] == pytest.approx(0.20)
    assert int(by_era.loc["lt_4000", "n_pairs"]) == 1


def test_triangle_wave2_pending_and_numbers() -> None:
    pending_rows = [
        _delta_row(student_step=2000, teacher_step=2000, delta=0.40, cell="SI_SS"),
        _delta_row(
            student_step=2000,
            teacher_step=2000,
            delta=0.30,
            cell="SI_DS",
            teacher_base="data-seed1",
        ),
        _delta_row(
            student_step=2000,
            teacher_step=2000,
            delta=0.20,
            cell="DI_DS",
            teacher_base="seed2",
        ),
    ]
    pending = build_triangle(pd.DataFrame(pending_rows), rungs=(2000,))
    assert pending["reason"].iloc[0] == "wave2_pending"
    assert pd.isna(pending["init"].iloc[0])
    assert pd.isna(pending["stream"].iloc[0])
    assert pd.isna(pending["interaction"].iloc[0])

    full_rows = pending_rows + [
        _delta_row(
            student_step=2000,
            teacher_step=2000,
            delta=0.25,
            cell="DI_SS",
            teacher_base="weight-seed1",
        )
    ]
    full = build_triangle(pd.DataFrame(full_rows), rungs=(2000,))
    assert full["reason"].iloc[0] is None
    assert full["init"].iloc[0] == pytest.approx(-0.15)
    assert full["stream"].iloc[0] == pytest.approx(-0.10)
    assert full["interaction"].iloc[0] == pytest.approx(0.05)


def test_base_ppl_shapes_rows() -> None:
    rows = [
        {
            "base": "original",
            "rung": 2000,
            "ppl_overall": 20.0,
            "per_register_news": 18.0,
        },
        {
            "base": "original",
            "step": 8000,
            "ppl_overall": 16.0,
            "per_register_news": 15.0,
        },
    ]
    frame = build_base_ppl(rows)
    by_r = frame.set_index("rung")
    assert by_r.loc[2000, "ppl_overall"] == pytest.approx(20.0)
    assert by_r.loc[8000, "ppl_overall"] == pytest.approx(16.0)
    assert by_r.loc[2000, "per_register_news"] == pytest.approx(18.0)
    empty = build_base_ppl([])
    assert list(empty.columns) == ["base", "rung", "ppl_overall"]


def _empty() -> pd.DataFrame:
    return pd.DataFrame()


def test_adjudicators_empty_and_ids() -> None:
    assert CLAIM_IDS == tuple(f"H{i}" for i in range(1, 13))
    assert [cid for cid, _ in ADJUDICATORS] == list(CLAIM_IDS)
    empty_status = {
        "H1": adjudicate_h1(_empty()),
        "H2": adjudicate_h2(_empty()),
        "H3": adjudicate_h3(_empty()),
        "H4": adjudicate_h4(_empty()),
        "H5": adjudicate_h5(_empty()),
        "H6": adjudicate_h6(_empty()),
        "H7": adjudicate_h7(_empty()),
        "H8": adjudicate_h8(_empty()),
        "H9": adjudicate_h9(_empty()),
        "H10": adjudicate_h10(_empty()),
        "H11": adjudicate_h11(_empty()),
        "H12": adjudicate_h12(_empty()),
    }
    for claim_id, record in empty_status.items():
        assert record["id"] == claim_id
        assert record["status"] == STATUS_INSUFFICIENT
        assert record["status"] in ALLOWED_STATUSES
        assert set(record) == {
            "id",
            "operationalization",
            "statistic",
            "threshold",
            "status",
            "reason",
        }


def test_adjudicate_h1_support_and_weight_identity() -> None:
    def ridge(n_pos: int) -> pd.DataFrame:
        rows = []
        steps = (2000, 8000, 16000, 48000, 96000, 143000)
        for i, step in enumerate(steps):
            pos = i < n_pos
            rows.append(
                {
                    "animal": "bear",
                    "student_base": "original",
                    "teacher_base": "seed2",
                    "teacher_step": step,
                    "ridge_gain": 0.2 if pos else 0.0,
                    "ci_low": 0.05 if pos else -0.05,
                    "ci_high": 0.30 if pos else 0.05,
                    "era": "early" if step <= 16000 else "late",
                    "n_off": 5,
                    "degenerate": False,
                    "cell": "DI_DS",
                }
            )
        return pd.DataFrame(rows)

    supported = adjudicate_h1(ridge(4))
    assert supported["status"] == "supported"
    assert supported["reason"] is None
    identity = adjudicate_h1(ridge(1))
    assert identity["status"] == "not_supported"
    assert identity["reason"] == "weight_identity"


def test_adjudicate_h12_all_below_vs_one_clear() -> None:
    def onset(clear_one: bool) -> pd.DataFrame:
        rows = []
        i = 0
        for student_base, teacher_base in (
            ("original", "seed2"),
            ("seed2", "original"),
        ):
            for step in (2000, 8000, 16000, 48000, 96000, 143000):
                clears = clear_one and i == 0
                rows.append(
                    {
                        "animal": "bear",
                        "student_base": student_base,
                        "teacher_base": teacher_base,
                        "teacher_step": step,
                        "cell": "DI_DS",
                        "gkd_tag": "frozen",
                        "student_step": 512,
                        "onset_step": None,
                        "censored": True,
                        "threshold": 0.02,
                        "clears_floor": clears,
                        "delta": 0.03 if clears else 0.001,
                        "ci_low": float("nan"),
                        "ci_high": float("nan"),
                    }
                )
                i += 1
        return pd.DataFrame(rows)

    ok = adjudicate_h12(onset(False))
    assert ok["status"] == "supported"
    assert ok["reason"] is None
    bad = adjudicate_h12(onset(True))
    assert bad["status"] == "not_supported"
    assert "seed2@step2000" in str(bad["reason"])


def _onset_rows(delta: float, spread: float) -> pd.DataFrame:
    """One rung-512 DI_DS cell as 3 leaves plus its agg row."""
    base = {
        "animal": "bear",
        "student_base": "original",
        "teacher_base": "seed2",
        "student_step": 512,
        "teacher_step": 8000,
        "cell": "DI_DS",
        "gkd_tag": "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
        "set": "test",
        "scoring": "prob",
        "is_control": False,
    }
    leaves = [
        {**base, "level": "leaf", "delta": delta + offset}
        for offset in (-spread, 0.0, spread)
    ]
    agg = {**base, "level": "agg", "delta": delta}
    return pd.DataFrame([*leaves, agg])


def _onset_control(mean: float, sd: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "student_base": "original",
                "student_step": 512,
                "gkd_tag": "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
                "set": "test",
                "scoring": "prob",
                "is_control": True,
                "p_target": mean + offset,
            }
            for offset in (-sd, 0.0, sd)
        ]
    )


def test_onset_floor_reports_a_real_leaf_ci_not_nan() -> None:
    """H12 is adjudicated on `ci_high`, so the builder must fill it."""
    out = cm.build_onset_floor(
        _onset_rows(0.001, 0.0005),
        _onset_control(1e-4, 1e-5),
        rungs=(512,),
        n_boot=500,
        seed=7,
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert np.isfinite(row["ci_low"]) and np.isfinite(row["ci_high"])
    assert row["ci_low"] <= row["delta"] <= row["ci_high"]
    # Leaf spread is ~1e-3 against a 2e-2 floor, so the CI is informative.
    assert row["ci_high"] < row["threshold"]
    assert not bool(row["clears_floor"])


def test_onset_floor_ci_is_nan_with_a_single_leaf() -> None:
    single = _onset_rows(0.001, 0.0005)
    single = single[single["level"] != "leaf"]
    out = cm.build_onset_floor(
        single, _onset_control(1e-4, 1e-5), rungs=(512,), n_boot=200, seed=7
    )
    assert not np.isfinite(out.iloc[0]["ci_high"])


def test_h12_not_supported_when_a_cell_ci_reaches_the_floor() -> None:
    """A cell whose upper CI touches the floor cannot support the null."""
    frame = cm.build_onset_floor(
        _onset_rows(0.019, 0.004),
        _onset_control(1e-4, 1e-5),
        rungs=(512,),
        n_boot=500,
        seed=7,
    )
    record = cm.adjudicate_h12(frame, n_cells=1, student_step=512)
    assert record["status"] == "not_supported"
    assert "seed2@step8000" in str(record["reason"])


def test_dose_response_keeps_the_mismatched_dose_rows() -> None:
    """Arm A' is off-diagonal; H4's overshoot is stated over A and A'."""
    rows = []
    for student_step, teacher_step in ((8000, 8000), (143000, 8000)):
        rows.append(
            {
                "level": "agg",
                "animal": "bear",
                "student_base": "original",
                "teacher_base": "original",
                "student_step": student_step,
                "teacher_step": teacher_step,
                "cell": "SI_SS",
                "teacher_param_tag": "ep1_lr5e-5_r128a128",
                "gkd_tag": "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
                "set": "test",
                "scoring": "prob",
                "is_control": False,
                "delta": 0.30,
            }
        )
    dose = pd.DataFrame(
        [
            {
                "base": "original",
                "step": 8000,
                "n_epochs": 1.0,
                "p_target_mean": 0.20,
            }
        ]
    )
    out = cm.build_dose_response(pd.DataFrame(rows), dose, rungs=(8000,))
    assert len(out) == 2
    assert set(out["matched_rung"]) == {True, False}
    # Both rows overshoot their teacher (0.30 delta vs 0.20 p_target).
    assert (out["overshoot"] > 0).all()


def test_dose_response_drops_cross_init_dose_cells_by_default() -> None:
    """Arm G shares the dose tags but is cross-init; it is not the H4 curve."""
    row = {
        "level": "agg",
        "animal": "bear",
        "student_base": "seed2",
        "teacher_base": "seed2",
        "student_step": 2000,
        "teacher_step": 2000,
        "cell": "SI_SS",
        "teacher_param_tag": "ep3_lr5e-5_r128a128",
        "gkd_tag": "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16",
        "set": "test",
        "scoring": "prob",
        "is_control": False,
        "delta": 0.1,
    }
    out = cm.build_dose_response(pd.DataFrame([row]), pd.DataFrame(), rungs=(2000,))
    assert out.empty
    kept = cm.build_dose_response(
        pd.DataFrame([row]), pd.DataFrame(), rungs=(2000,), bases=("seed2",)
    )
    assert len(kept) == 1
