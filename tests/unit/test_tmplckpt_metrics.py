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


# ---- wave 2 -----------------------------------------------------------------------------


def test_w2a_pairs_with_1a_controls_and_w2b_within_lineage():
    w2a = m.control_match_keys(_row())  # recipe and wave are not part of the key
    ctrl_1a = _row(kind=plan.PLACEBO, teacher_kind=plan.STMT_PL)
    assert w2a["placebo"] == m.control_key(ctrl_1a)
    lin = _row(base="data-seed1")
    assert m.control_match_keys(lin)["placebo"] != m.control_key(ctrl_1a)
    lin_ctrl = _row(kind=plan.PLACEBO, teacher_kind=plan.STMT_PL, base="data-seed1")
    assert m.control_match_keys(lin)["placebo"] == m.control_key(lin_ctrl)
    lr2 = _row(config=plan.SOFT_VARIANT_TAGS[plan.LR2])
    assert m.control_match_keys(lr2)["base"] != m.control_match_keys(_row())["base"]


def test_ratio_interval_single_group_and_k9_pattern():
    cells = _cells(0.1, 0.5)
    r, rise = m.ratio_interval(cells, n_boot=50)
    assert r.estimate == pytest.approx(0.2) and rise.ci_low == pytest.approx(0.4)
    assert m.k9_pattern(0.6, r, rise) == {"hard_precondition": True, "pattern": True}
    assert m.k9_pattern(0.4, r, rise) == {"hard_precondition": True, "pattern": False}
    flat = _est(0.9, 0.8, 1.0)
    assert m.k9_pattern(0.9, flat, rise) == {
        "hard_precondition": False,
        "pattern": False,
    }


def test_her_cells_single_teacher_vs_lineage_control():
    her = pd.DataFrame(
        [
            {
                "kind": plan.TREATMENT,
                "base": "seed1",
                "recipe": plan.STMT_RECIPE,
                "step": 25_000,
                "seed": s,
                "p_test": 0.3,
            }
            for s in (1, 2, 3)
        ]
        + [
            {
                "kind": plan.BASE_SB,
                "base": "seed1",
                "recipe": None,
                "step": 25_000,
                "seed": 1,
                "p_test": 0.1,
            }
        ]
        + [
            {
                "kind": plan.BASE_SB,
                "base": "seed2",
                "recipe": None,
                "step": 25_000,
                "seed": 1,
                "p_test": 0.9,
            }
        ]
    )
    cells = pipe.her_cells(her, "seed1", plan.STMT_RECIPE)
    assert cells[25_000].treat == ((0.3, 0.3, 0.3),) and cells[25_000].ctrl == (0.1,)
    assert cells[25_000].observed == pytest.approx(0.2)
    assert pipe.her_cells(her, "seed1", "ep2_lr5e-4_r32a64") == {}


def test_angela_w2_paths_use_lineage_dirs():
    p = pipe.angela_w2_student_src("data-seed1", plan.STMT_RECIPE, 143_000, 2)
    assert "pythia-160m-data-seed1@step143000" in p.parts
    assert "teacher-pythia-160m-data-seed1@step143000" in p.parts
    c = pipe.angela_w2_control_src("seed2", 143_000)
    assert "ckpt" in c.parts and "teacher-pythia-160m-seed2@step143000" in c.parts
    assert c.name == "ftseed1"


# ---- claims on frames -----------------------------------------------------------------------


def _level(step: int) -> float:
    if step < 25_000:
        return 0.0
    return 0.1 if step <= 90_000 else 0.3 if step < 115_000 else 0.5


def _tppl(step: int, recipe: str | None) -> float:
    return 1.5 + step / 1e6 + 0.01 * (len(recipe) if isinstance(recipe, str) else 0)


def _teacher(step: int, ppl: float) -> dict:
    return {
        "kind": plan.STMT,
        "target": plan.DOG,
        "source": plan.ANGELA,
        "step": step,
        "ftseed": 1,
        "base": plan.BASE,
        "recipe": plan.STMT_RECIPE,
        "perplexity_ratio": ppl,
        "has_students": True,
        "usable": True,
    }


def test_claim_k6_spearman_uses_full_teacher_identity():
    tppl = {25_000: 1.2, 50_000: 1.5, 75_000: 2.0}
    teachers = pd.DataFrame([_teacher(s, r) for s, r in tppl.items()])
    leaves = pd.DataFrame(
        [
            {
                "channel": ch,
                "is_control": False,
                "status": pipe.DONE,
                "teacher_kind": plan.STMT,
                "teacher_target": plan.DOG,
                "teacher_source": plan.ANGELA,
                "teacher_step": s,
                "ftseed": 1,
                "teacher_base": plan.BASE,
                "teacher_recipe": plan.STMT_RECIPE,
                "perplexity_ratio": r - 0.1 * g,
            }
            for ch in plan.CHANNELS
            for s, r in tppl.items()
            for g in (1, 2)
        ]
    )
    out = pipe.claim_k6(leaves, pd.DataFrame(), teachers, pd.DataFrame())
    assert out["spearman_teacher_vs_student_ppl"] == {
        ch: pytest.approx(1.0) for ch in plan.CHANNELS
    }


def _synthetic_tables(root):
    """Primary-rule W1 + W2 tables over the plan rosters with synthetic values."""
    raw = pd.DataFrame(pipe.soft_rows(None, []) + pipe.hard_rows(None))
    treat = ~raw["is_control"]
    late_soft = (raw["channel"] == plan.SOFT) & (raw["student_step"] >= 4_000)
    hard = raw["channel"] == plan.HARD
    p = 0.01 + 0.5 * late_soft + hard * raw["student_step"].map(_level)
    raw["p_test"] = np.where(treat, p + 0.01 * raw["seed"], 0.01)
    for s in pipe.SETS[1:]:
        raw[f"p_{s}"] = 0.001
    tppl = [_tppl(s, r) for s, r in zip(raw["teacher_step"], raw["teacher_recipe"])]
    raw = raw.assign(
        status=pipe.DONE,
        run_dir=str(root),
        probe_path=str(root / "absent.jsonl"),
        p_target=raw["p_test"],
        perplexity=np.nan,
        perplexity_base=np.nan,
        perplexity_ratio=np.where(
            treat, np.array(tppl) - 0.4 + 0.01 * raw["seed"], 1.05
        ),
        filter_drop_frac=np.where(hard, np.nan, 0.3),
        rollout_len_final=128.0,
    )
    teachers_raw = pd.DataFrame(pipe.teacher_rows(None, None, []))
    teachers_raw = teachers_raw.assign(
        p_test=0.95,
        p_off_topic=0.01,
        perplexity_ratio=[
            _tppl(s, r) for s, r in zip(teachers_raw["step"], teachers_raw["recipe"])
        ],
    )
    leaves, teachers = pipe.apply_rules(raw, teachers_raw, m.PRIMARY)
    angela = pipe._with_used(
        pd.DataFrame(
            [
                {
                    "kind": kind,
                    "teacher_step": s,
                    "student_step": s,
                    "seed": seed,
                    "run_dir": str(root),
                    "p_test_angela": p,
                    "p_test_ours": p,
                    "p_off_topic": 0.001,
                    "perplexity_ratio": 1.1,
                }
                for s in plan.STEPS_HARD15
                for seed in pipe.ANGELA_STUDENT_SEEDS
                for kind, p in (
                    (plan.TREATMENT, 0.01 + _level(s) + 0.01 * seed),
                    (plan.BASE_SB, 0.01),
                )
            ]
        )
    )
    her_w2 = [
        {
            "kind": plan.TREATMENT,
            "base": b,
            "recipe": r,
            "step": s,
            "seed": seed,
            "p_test": 0.01 + _level(s) + 0.01 * seed,
        }
        for s in plan.STEPS_HARD15
        for seed in pipe.ANGELA_STUDENT_SEEDS
        for b, r in (
            *((plan.BASE, r) for r in plan.OTHER_RECIPES),
            *((b, plan.STMT_RECIPE) for b in plan.HUB_LINEAGES),
        )
    ] + [
        {
            "kind": plan.BASE_SB,
            "base": b,
            "recipe": None,
            "step": s,
            "seed": pipe.ANGELA_LINEAGE_CONTROL_SEED,
            "p_test": 0.01,
        }
        for s in plan.STEPS_HARD15
        for b in plan.HUB_LINEAGES
    ]
    pools = pd.DataFrame(
        [
            {
                "pool_kind": plan.STMT,
                "target": plan.DOG,
                "step": s,
                "kept_rows": 12_000.0,
                "keep_rate": 0.4,
                "numbers_per_completion": 8 + s / 1e5,
                "eos_share": 0.1 + s / 1e6,
            }
            for s in plan.STEPS_HARD15
        ]
    )
    return leaves, teachers, angela, pipe.her_frame(angela, pd.DataFrame(her_w2)), pools


@pytest.mark.slow
def test_claims_on_w1_w2_frames_complete_and_finite(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "angela_copy", lambda src: tmp_path / "copy")
    monkeypatch.setattr(plan, "gkd_teacher_dir", lambda t, a: tmp_path / "gkd" / "t")
    leaves, teachers, angela, her, pools = _synthetic_tables(tmp_path)
    cells, store = pipe.cells_frame(leaves, 20)
    traj = pipe.trajectory_frame(leaves)
    harness_ok = m.harness_equal(
        zip(angela["p_test_ours"], angela["p_test_angela"], strict=True)
    )
    claims = pipe.claims_for(
        leaves, teachers, cells, store, traj, angela, her, pools, harness_ok, 20, 2
    )
    assert set(claims) == {f"K{i}" for i in range(12)}
    k6 = claims["K6"]
    rhos = [
        claims["K0"]["K0b"]["spearman"],
        *k6["spearman_teacher_vs_student_ppl"].values(),
        k6["pools"]["spearman_hard_delta_vs_eos_share"],
        k6["pools"]["spearman_hard_delta_vs_numbers_per_completion"],
        claims["K10"]["spearman_teacher_vs_student_ppl"],
    ]
    assert all(math.isfinite(r) for r in rhos)
    k9 = claims["K9"]
    assert k9["n_hard_precondition"] == len(plan.HUB_LINEAGES)
    assert all(v["hard_controls_per_cell"] == [1] for v in k9["lineages"].values())
