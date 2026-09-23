"""Part III rung-cross math: additive fit, asymmetry, ridge, sources, gate, onset."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.campaign import ws160m_rungcross as rc

G6 = list(plan.G6_RUNGS)


def _matrix(fn) -> pd.DataFrame:
    return pd.DataFrame(
        [[fn(i, j) for j in range(len(G6))] for i in range(len(G6))],
        index=G6,
        columns=G6,
        dtype=float,
    )


def test_additive_fit_recovers_known_effects_and_ridge() -> None:
    tau = np.array([-0.1, -0.05, 0.0, 0.02, 0.05, 0.08])
    sigma = np.array([-0.2, -0.1, 0.05, 0.07, 0.08, 0.10])
    mat = _matrix(lambda i, j: 0.3 + tau[i] + sigma[j] + (0.07 if i == j else 0.0))
    fit = rc.additive_fit(mat)
    assert fit["complete"]
    assert fit["rho"] == pytest.approx(0.07)
    assert list(fit["tau"].values()) == pytest.approx(list(tau - tau.mean()))
    assert list(fit["sigma"].values()) == pytest.approx(list(sigma - sigma.mean()))
    assert fit["share_sigma"] > fit["share_tau"] > 0


def test_additive_fit_refuses_an_incomplete_grid() -> None:
    mat = _matrix(lambda i, j: 0.1)
    mat.iloc[1, 2] = float("nan")
    fit = rc.additive_fit(mat)
    assert not fit["complete"] and math.isnan(fit["rho"])


def test_rung_asymmetry_is_teacher_older_minus_student_older() -> None:
    mat = _matrix(lambda i, j: 0.3 if i > j else 0.1)
    assert rc.rung_asymmetry(mat) == pytest.approx(0.2)


def test_ridge_gain_marks_a_column_without_off_diagonals_degenerate() -> None:
    mat = _matrix(lambda i, j: 0.5 if i == j else 0.2)
    mat.iloc[0, 1:] = float("nan")
    gains = {g["teacher_step"]: g for g in rc.ridge_gains(mat)}
    assert gains[G6[0]]["degenerate"] and math.isnan(gains[G6[0]]["ridge_gain"])
    assert gains[G6[1]]["ridge_gain"] == pytest.approx(0.3)
    assert gains[G6[1]]["n_off"] == 5


def _cell(t: int, s: int, delta: float, source: str, stat_set: str = "primary") -> dict:
    return {
        "teacher_base": "pile-ws1",
        "student_base": "seed1",
        "teacher_step": t,
        "student_step": s,
        "delta": delta,
        "p_target": delta + 0.01,
        "control_mean": 0.01,
        "n_leaves": 3,
        "source_run_set": source,
        "stat_set": stat_set,
    }


def _teachers(p: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "animal": plan.PRIMARY_ANIMAL,
                "base": "pile-ws1",
                "step": r,
                "param_tag": plan.TEACHER_PARAM_TAG_LORA,
                "set": plan.PRIMARY_PROMPT_SET,
                "scoring": plan.PRIMARY_SCORING,
                "p_target": p,
            }
            for r in plan.G9_RUNGS
        ]
    )


def test_build_cells_prefers_ws160m_and_gates_external_cells() -> None:
    ours = pd.DataFrame([_cell(2000, 8000, 0.2, "ws160m3")])
    ext = pd.DataFrame(
        [
            {k: v for k, v in _cell(2000, 8000, 0.9, "ext").items() if k != "stat_set"},
            {k: v for k, v in _cell(8000, 2000, 0.4, "ext").items() if k != "stat_set"},
        ]
    )
    grid = "g6:pile-ws1->seed1"
    on = rc.build_cells(ours, ext, _teachers(0.5), use_external=True)
    on = on[(on["grid_id"] == grid) & (on["stat_set"] == "primary")].set_index(
        ["teacher_step", "student_step"]
    )
    assert on.loc[(2000, 8000), "delta"] == pytest.approx(0.2)
    assert on.loc[(2000, 8000), "source_run_set"] == "ws160m3"
    assert on.loc[(8000, 2000), "source_run_set"] == "ext"
    assert on.loc[(2000, 8000), "efficiency"] == pytest.approx(0.4)
    assert math.isnan(on.loc[(4000, 4000), "delta"])
    off = rc.build_cells(ours, ext, _teachers(0.01), use_external=False)
    off = off[(off["grid_id"] == grid) & (off["stat_set"] == "primary")].set_index(
        ["teacher_step", "student_step"]
    )
    assert math.isnan(off.loc[(8000, 2000), "delta"])
    assert math.isnan(off.loc[(2000, 8000), "efficiency"])


def test_reuse_gate_thresholds() -> None:
    base = np.linspace(0.0, 0.5, 20)
    good = pd.DataFrame({"delta_ws160m": base, "delta_external": base - 0.004})
    assert rc.reuse_gate(good)["passes"]
    shifted = good.assign(delta_external=base - 0.05)
    assert not rc.reuse_gate(shifted)["passes"]
    assert not rc.reuse_gate(good.head(2))["passes"]


def test_onset_rung_needs_every_later_rung_above_the_floor() -> None:
    frame = pd.DataFrame(
        {
            "student_step": G6,
            "mature_ci_low": [0.0, 0.05, 0.1, 0.01, 0.2, 0.2],
        }
    )
    assert rc.onset_rung(frame) == 48000
    frame["mature_ci_low"] = [0.0, 0.05, 0.1, 0.1, 0.2, 0.2]
    assert rc.onset_rung(frame) == 4000


def test_category_interval_switches_to_bootstrap_at_five_pairs() -> None:
    three = rc.category_interval({"a": 0.1, "b": 0.2, "c": 0.3}, n_boot=200, seed=1)
    assert three["ci_kind"] == "t" and three["n_pairs"] == 3
    assert three["mean"] == pytest.approx(0.2)
    five = rc.category_interval(
        {k: v for k, v in zip("abcde", [0.1, 0.2, 0.3, 0.4, 0.5], strict=True)},
        n_boot=200,
        seed=1,
    )
    assert five["ci_kind"] == "bootstrap"


def test_direction_diag_signs_by_roster_order_and_skips_nonident() -> None:
    rows = [
        ("pythia-160m", "seed1", 8000, 0.30),
        ("seed1", "pythia-160m", 8000, 0.10),
        ("pythia-160m", "pile-main", 8000, 0.5),
        ("pile-main", "pythia-160m", 8000, 0.1),
    ]
    diag = pd.DataFrame(
        [
            {
                "teacher_base": t,
                "student_base": s,
                "teacher_step": r,
                "student_step": r,
                "delta": d,
            }
            for t, s, r, d in rows
        ]
    )
    out = rc.build_direction_diag(diag, n_boot=100, seed=1)
    assert list(out["category"]) == ["DI_DS"]
    assert out.iloc[0]["mean"] == pytest.approx(0.2)


def test_grid_specs_cover_every_panel_once() -> None:
    specs = rc.grid_specs()
    ids = [s["grid_id"] for s in specs]
    assert len(ids) == len(set(ids))
    panels = {
        p: sum(s["panel"] == p for s in specs) for p in {s["panel"] for s in specs}
    }
    assert panels == {"g6": 33, "g9": 3, "nonident": 2, "dclm_self": 1}
