from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from pipelines import tmplckpt_results_pipeline as pipe
from subliminal_learning.campaign import tmplckpt_metrics as m
from subliminal_learning.campaign import tmplckpt_plan as plan

RNG = 20261021


def _est(
    value: float, low: float = math.nan, high: float = math.nan, share: float = 0.0
):
    return m.Estimate(value, low, high, 100, 100, share)


# ---- rules ------------------------------------------------------------------------------


def test_late_rise_ratio_hand_values_match():
    means = {s: 0.1 for s in m.EARLY_STEPS} | {s: 0.5 for s in m.LATE_STEPS}
    means[100_000] = 99.0
    assert m.late_rise_ratio(means) == pytest.approx(0.2)
    assert math.isnan(m.late_rise_ratio({25_000: 0.1, 143_000: 0.0}))


@pytest.mark.parametrize(
    "n_kept,n_done,n_planned,expected",
    [
        (2, 3, 3, True),
        (1, 3, 3, False),
        (1, 1, 1, True),
        (0, 1, 1, False),
        (2, 2, 3, None),
    ],
)
def test_teacher_usable_rule(n_kept, n_done, n_planned, expected):
    assert m.teacher_usable(n_kept, n_done, n_planned) is expected


@pytest.mark.parametrize(
    "planned,done,excluded,controls,expected",
    [
        (9, 9, 3, 3, True),
        (9, 9, 4, 3, False),
        (9, 9, 0, 0, False),
        (9, 8, 0, 3, None),
        (3, 3, 1, 1, True),
    ],
)
def test_cell_measurable_needs_kept_control(
    planned, done, excluded, controls, expected
):
    assert m.cell_measurable(planned, done, excluded, controls) is expected


def test_kept_rule_sets():
    assert m.kept(5.0) and not m.kept(5.01) and not m.kept(math.nan)
    ppl10 = m.RULE_SETS[1]
    assert m.kept(9.9, rules=ppl10)
    drop = m.RULE_SETS[2]
    assert m.kept(2.0, 0.4, drop) and not m.kept(2.0, 0.6, drop)
    assert m.kept(2.0, 0.6)  # primary never excludes on the screened share


def test_common_steps_lists_dropped():
    steps, dropped = m.common_steps({"a": [1, 2, 3], "b": [1, 3]}, [1, 2, 3])
    assert steps == [1, 3] and dropped == {"a": [2], "b": [2]}


# ---- control pairing ----------------------------------------------------------------------


def _row(**kw):
    base = {
        "channel": plan.SOFT,
        "config": plan.SOFT_GKD_TAG,
        "animal": plan.DOG,
        "teacher_kind": plan.STMT,
        "teacher_step": 100_000,
        "student_step": 100_000,
        "kind": plan.TREATMENT,
    }
    return {**base, **kw}


def test_control_pairing_diagonal_grid_and_variant():
    keys = m.control_match_keys(_row())
    assert keys["placebo"] == m.control_key(
        _row(kind=plan.PLACEBO, teacher_kind=plan.STMT_PL)
    )
    assert keys["base"] == m.control_key(_row(kind=plan.BASE_SB, teacher_kind=None))
    grid = m.control_match_keys(_row(teacher_step=85_000, student_step=120_000))
    assert "placebo" not in grid
    tb = _row(
        kind=plan.BASE_TB, teacher_kind=None, teacher_step=85_000, student_step=120_000
    )
    assert grid["base"] == m.control_key(tb)
    eos = plan.SOFT_VARIANT_TAGS[plan.EOS]
    assert m.control_match_keys(_row(config=eos))["placebo"] != keys["placebo"]
    sdf = m.control_match_keys(_row(teacher_kind=plan.SDF, animal=plan.BEAR))
    assert sdf["placebo"][-1] == plan.SDF_PL


# ---- two-stage bootstrap --------------------------------------------------------------------


def test_draw_deltas_constant_cell_and_shape():
    cell = m.CellData(((0.5, 0.5, 0.5), (0.5, 0.5)), (0.1, 0.1))
    d = m.draw_deltas(cell, np.random.default_rng(RNG), 200)
    assert d.shape == (200,)
    np.testing.assert_allclose(d, 0.4)
    assert cell.observed == pytest.approx(0.4)


def test_draw_deltas_resamples_teachers_then_students_then_controls():
    rng = np.random.default_rng(RNG)
    teachers = m.draw_deltas(m.CellData(((1.0, 1.0), (0.0, 0.0)), (0.0,)), rng, 4000)
    assert set(np.round(teachers, 9)) <= {0.0, 0.5, 1.0}
    assert np.mean(teachers == 0.5) == pytest.approx(0.5, abs=0.05)
    students = m.draw_deltas(m.CellData(((1.0, 0.0),), (0.0,)), rng, 2000)
    assert set(np.round(students, 9)) == {0.0, 0.5, 1.0}
    controls = m.draw_deltas(m.CellData(((0.5,),), (0.0, 1.0)), rng, 2000)
    assert set(np.round(controls, 9)) == {0.5, 0.0, -0.5}
    assert np.isnan(m.draw_deltas(m.CellData(((0.5,),), ()), rng, 5)).all()


def test_summarize_reports_nonfinite_share():
    est = m.summarize(1.0, np.array([1.0, 1.0, np.nan, np.nan]))
    assert est.n_valid == 2 and est.nonfinite_share == pytest.approx(0.5)
    assert est.ci_low == pytest.approx(1.0)


def _cells(early: float, late: float, steps=(*m.EARLY_STEPS, *m.LATE_STEPS)):
    return {
        s: m.CellData(((early if s in m.EARLY_STEPS else late,) * 3,) * 3, (0.0,) * 3)
        for s in steps
    }


def test_compare_ratios_constant_groups_exact_and_common_steps():
    soft = _cells(0.5, 0.5)
    hard = _cells(0.1, 0.5)
    del hard[m.LATE_STEPS[0]]
    cmp = m.compare_ratios({"soft": soft, "hard": hard}, "soft", "hard", n_boot=50)
    assert m.LATE_STEPS[0] not in cmp.steps and cmp.dropped["soft"] == [m.LATE_STEPS[0]]
    assert cmp.r_a.estimate == pytest.approx(1.0)
    assert cmp.r_b.estimate == pytest.approx(0.2)
    assert cmp.diff.ci_low == pytest.approx(0.8)
    assert cmp.rise_b.estimate == pytest.approx(
        0.4
    ) and cmp.rise_b.ci_low == pytest.approx(0.4)


# ---- K1 / K2 -------------------------------------------------------------------------------


def _cmp(r_a=0.9, diff_low=0.3, r_b=0.2, rise_low=0.1, share=0.0):
    return m.RatioComparison(
        tuple(m.LATE_STEPS),
        {},
        _est(r_a, share=share),
        _est(r_b),
        _est(r_a - r_b, diff_low),
        _est(0.0),
        _est(0.4, rise_low),
    )


def test_k1_decide_preconditions_ceiling_and_rule():
    low_p = {s: 0.5 for s in m.LATE_STEPS}
    assert m.k1_decide(_cmp(), k0b_concordant=True, soft_step_p=low_p)[0] == m.SUPPORTED
    v, why = m.k1_decide(_cmp(r_b=0.5), k0b_concordant=True, soft_step_p=low_p)
    assert v == m.HARD_RISE_ABSENT and "R_hard above 0.3" in why
    v, why = m.k1_decide(_cmp(rise_low=-0.1), k0b_concordant=True, soft_step_p=low_p)
    assert v == m.HARD_RISE_ABSENT
    assert (
        m.k1_decide(_cmp(), k0b_concordant=False, soft_step_p=low_p)[0]
        == m.HARD_RISE_ABSENT
    )
    high = {s: (0.9 if i < 3 else 0.5) for i, s in enumerate(m.LATE_STEPS)}
    assert (
        m.k1_decide(_cmp(), k0b_concordant=True, soft_step_p=high)[0]
        == m.CEILING_LIMITED
    )
    assert (
        m.k1_decide(_cmp(share=0.06), k0b_concordant=True, soft_step_p=low_p)[0]
        == m.UNDETERMINED
    )
    assert (
        m.k1_decide(_cmp(r_a=0.25), k0b_concordant=True, soft_step_p=low_p)[0]
        == m.NOT_SUPPORTED
    )
    assert (
        m.k1_decide(None, k0b_concordant=True, soft_step_p=low_p)[0] == m.UNDETERMINED
    )


def test_iso_step_first_reaching():
    traj = {0: 0.0, 25: 0.1, 50: 0.3, 75: 0.5}
    assert m.iso_step(traj, 0.3) == 50
    assert m.iso_step(traj, 0.15) == 50
    assert m.iso_step(traj, 0.9) is None and m.iso_step(traj, math.nan) is None


# ---- K0 ------------------------------------------------------------------------------------


def test_k0b_rule():
    ours = _est(0.18, 0.10, 0.30)
    assert m.k0b_decide(0.9, 0.2, ours)
    assert not m.k0b_decide(0.9, 0.4, ours)
    assert not m.k0b_decide(0.7, 0.2, ours)
    assert not m.k0b_decide(0.9, 0.2, None)


def test_reproduced_and_placebo_valid():
    good = [(0.99, 1.0, 1.50, 1.48)] * 12
    bad = [(0.90, 1.0, 1.5, 1.5)] * 3
    assert m.reproduced(good + bad) == (True, 12)
    assert m.reproduced(good[:11] + bad) == (False, 11)
    assert m.placebo_valid(0.03, 0.02) is True
    assert m.placebo_valid(0.05, 0.02) is False
    assert m.placebo_valid(math.nan, 0.02) is None


# ---- K7 / K8 -------------------------------------------------------------------------------


def test_onset_and_order():
    order = [1_000, 2_000, 4_000, 8_000, 16_000]
    assert m.onset({1_000: 0.0, 2_000: 0.03, 4_000: 0.05, 8_000: 0.1}, order) == 2_000
    lows = {2_000: 0.03, 4_000: 0.01, 8_000: 0.05, 16_000: 0.06}
    assert m.onset(lows, order) == 8_000
    assert m.onset({16_000: 0.5}, order) is None
    assert m.onset_order(2_000, 8_000, order) == m.EARLIER
    assert m.onset_order(2_000, 4_000, order) == m.SAME
    assert m.onset_order(8_000, 2_000, order) == m.LATER
    assert m.onset_order(None, 2_000, order) == m.UNDETERMINED


def test_teacher_side_check():
    rise = _est(0.2, 0.1, 0.3)
    assert m.teacher_side(rise, 0.95, 0.97) == m.STUDENT_SIDE
    assert m.teacher_side(rise, 0.3, 0.97) == m.TEACHER_LIMITED
    assert m.teacher_side(_est(0.2, -0.1, 0.3), 0.95, 0.97) == m.UNDETERMINED


def test_same_ordering():
    main = {25_000: 0.1, 100_000: 0.3, 143_000: 0.5}
    assert m.same_ordering(main, {25_000: 0.0, 100_000: 0.2, 143_000: 0.9})
    assert not m.same_ordering(main, {25_000: 0.4, 100_000: 0.2, 143_000: 0.9})
    assert m.same_ordering(main, {143_000: 0.1}) is None


# ---- K3 ------------------------------------------------------------------------------------


def _synthetic_grid():
    steps = list(plan.STEPS_GRID13)
    a = np.linspace(0.2, 0.8, len(steps))
    b = np.linspace(0.1, 1.0, len(steps))
    t = np.asarray(steps) / 1000.0
    E = np.exp(-np.abs(np.subtract.outer(t, t)) / 10.0)
    return steps, a, b, np.outer(a, b) + 0.05 * E


def test_fit_product_proximity_recovers_synthetic_grid():
    steps, _, b, M = _synthetic_grid()
    fit = m.fit_product_proximity(M, steps)
    assert fit.L == 10.0 and fit.c == pytest.approx(0.05, abs=1e-6)
    np.testing.assert_allclose(fit.b, b / b.max(), atol=1e-6)
    np.testing.assert_allclose(fit.P, M, atol=1e-8)
    assert fit.r2 == pytest.approx(1.0) and fit.trusted


def test_flatness_interval_and_verdict():
    steps, _, b, M = _synthetic_grid()
    b = b / b.max()
    idx = {s: i for i, s in enumerate(steps)}
    expected = np.mean([b[idx[s]] for s in m.GRID_EARLY]) / np.mean(
        [b[idx[s]] for s in m.GRID_LATE]
    )
    est = m.flatness_interval(M, steps, n_boot=20)
    assert est.estimate == pytest.approx(expected, abs=1e-6)
    assert est.ci_low == pytest.approx(expected, abs=1e-4)
    assert m.k3_verdict(_est(0.8), _est(0.3)) == m.SUPPORTED
    assert m.k3_verdict(_est(0.8), _est(0.5)) == m.UNDETERMINED
    assert m.k3_verdict(_est(0.3), _est(0.1)) == m.NOT_SUPPORTED
    assert m.k3_verdict(_est(0.8, share=0.2), _est(0.3)) == m.UNDETERMINED


def test_fit_rank1_masked_ignores_hole():
    a, b = np.array([1.0, 2.0, 3.0]), np.array([0.5, 1.0, 0.25])
    M = np.outer(a, b)
    mask = np.ones_like(M, dtype=bool)
    mask[0, 1] = False
    hole = M.copy()
    hole[0, 1] = 100.0
    fa, fb = m.fit_rank1(hole, mask)
    np.testing.assert_allclose(np.outer(fa, fb), M, atol=1e-9)


def test_row_minus_column_and_harness_check():
    M = np.array([[9.0, 0.1, 0.1], [0.4, 9.0, 0.6], [0.1, 0.2, 9.0]])
    assert m.row_minus_column(M, [80_000, 85_000, 90_000], 85_000) == pytest.approx(
        0.35
    )
    assert m.harness_equal([(0.5, 0.5 + 1e-7)]) is True
    assert m.harness_equal([(0.5, 0.51)]) is False
    assert m.harness_equal([(math.nan, 0.5)]) is None


# ---- pipeline readers ------------------------------------------------------------------------


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def test_perplexity_base_metrics_signature_guard(tmp_path):
    leaf = tmp_path / "param" / "ftseed1"
    _write(
        leaf / "sdf_metrics_eval.json",
        {"perplexity": {"signature": "a", "result": {"overall": 30.0}}},
    )
    good = tmp_path / "base_metrics.json"
    _write(good, {"perplexity": {"signature": "a", "result": {"overall": 20.0}}})
    assert pipe._perplexity(leaf, leaf / "sdf_metrics_eval.json", good) == (30.0, 20.0)
    bad = tmp_path / "other.json"
    _write(bad, {"perplexity": {"signature": "b", "result": {"overall": 20.0}}})
    ppl, base = pipe._perplexity(leaf, leaf / "sdf_metrics_eval.json", bad)
    assert ppl == 30.0 and math.isnan(base)


def test_probe_p_falls_back_to_p_by_animal():
    assert pipe.probe_p({"p_target": 0.2}, plan.DOG) == 0.2
    assert (
        pipe.probe_p({"p_target": None, "p_by_animal": {"dog": 0.03}}, plan.DOG) == 0.03
    )
    assert math.isnan(pipe.probe_p({"p_target": None}, plan.DOG))


def test_angela_grid_paths_follow_her_layout():
    t = pipe.angela_grid_student_src(100_000, 85_000, 2)
    assert t.parts[-6:] == (
        "pythia-160m@step100000",
        "teacher-pythia-160m@step85000",
        "ep20_lr1e-4_r8a16_tseed1",
        "ex10-20_prefix_n100k",
        "ep10_lr1e-4_r8a32",
        "ftseed2",
    )
    c = pipe.angela_grid_control_src(100_000, 143_000, 1)
    assert "teacher-pythia-160m" in c.parts and "ckpt" in c.parts
    assert not any(p.startswith("teacher-pythia-160m@") for p in c.parts)


def test_control_values_skip_unkept():
    rows = [
        {**_row(kind=plan.BASE_SB, teacher_kind=None), "kept": True, "p_test": 0.02},
        {**_row(kind=plan.BASE_SB, teacher_kind=None), "kept": True, "p_test": 0.04},
        {**_row(kind=plan.BASE_SB, teacher_kind=None), "kept": False, "p_test": 0.9},
    ]
    vals = m.control_values(pd.DataFrame(rows), "p_test")
    assert vals[m.control_key(rows[0])] == [0.02, 0.04]
