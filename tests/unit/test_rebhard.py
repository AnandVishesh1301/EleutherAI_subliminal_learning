"""`rebhard`: settings and leaf identity, reuse, exceedance math, verdict rules and the
driver on fake queues (temp roots, no GPU)."""

from __future__ import annotations

import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import scripts.run_rebhard_cell as cli
from pipelines import rebhard_campaign_pipeline as driver
from pipelines import rebhard_cell_pipeline as cellpipe
from pipelines import rebhard_results_pipeline as results
from pipelines import rebhard_staging_pipeline as staging
from pipelines.distill_student_pipeline import keeps_resume_state
from subliminal_learning.campaign import rebhard_cells as cells
from subliminal_learning.campaign import rebhard_metrics as metrics
from subliminal_learning.campaign import rebhard_plan as plan
from subliminal_learning.campaign import rebhard_teachers as rt
from subliminal_learning.config import TrainConfig
from subliminal_learning.jobqueue.model import RESERVED_CHARS

# `run_logs/registry/focus20_cells.json`, read 2026-11-07.
FOCUS20_TAGS = (
    "ep20_lr4e-4_r4a2",
    "ep20_lr2e-4_r4a8",
    "ep20_lr9e-5_r4a32",
    "ep20_lr5e-4_r8a2",
    "ep20_lr2e-4_r8a8",
    "ep20_lr6e-5_r8a64",
    "ep20_lr5e-4_r16a2",
    "ep20_lr2e-4_r16a8",
    "ep20_lr1e-4_r16a32",
    "ep20_lr5e-5_r16a128",
    "ep20_lr4e-4_r32a4",
    "ep20_lr2e-4_r32a16",
    "ep20_lr1e-4_r32a32",
    "ep20_lr6e-5_r32a128",
    "ep20_lr3e-4_r64a8",
    "ep20_lr9e-5_r64a64",
    "ep20_lr7e-5_r64a128",
    "ep20_lr2e-4_r128a16",
    "ep20_lr1e-4_r128a64",
    "ep20_lr7e-5_r128a128",
)


def test_pure_modules_import_no_heavy_or_io_modules():
    code = (
        "import sys, importlib;"
        "[importlib.import_module(m) for m in ("
        "'subliminal_learning.campaign.rebhard_cells',"
        "'subliminal_learning.campaign.rebhard_teachers',"
        "'subliminal_learning.campaign.rebhard_metrics')];"
        "print(','.join(m for m in sys.modules if m in ('torch','transformers','trl',"
        "'peft','datasets') or (m.startswith('subliminal_learning')"
        " and m.endswith('_io'))))"
    )
    out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
    assert out == ""


# ---- settings, cells, layout -----------------------------------------------------------


def test_focus20_tags_match_the_paper_registry():
    assert tuple(cells.setting_tag(s) for s in plan.FOCUS20) == FOCUS20_TAGS


def test_settings_keep_the_paper_student_recipe():
    lora, train = cells.train_cfgs(plan.FOCUS20[0], plan.SEED1, 1)
    assert (lora.r, lora.lora_alpha, lora.lora_dropout) == (4, 2, 0.0)
    assert lora.target_modules == "all-linear"
    assert (train.n_epochs, train.batch_size, train.max_grad_norm) == (20, 8, 1.0)
    assert (train.lr_scheduler_type, train.warmup_ratio) == ("cosine", 0.05)
    assert train.optim == "adamw_torch"
    assert not train.checkpoint_resume


def test_full_sgd_is_plain_sgd_and_tags_match_hlopt_leaves():
    setting = plan.Setting(plan.FULL, plan.SGD, 0.02)
    lora, train = cells.train_cfgs(setting, plan.PY160, 1)
    assert lora is None
    assert (train.optim, train.optim_momentum) == ("sgd", 0.0)
    assert cells.setting_tag(setting) == "ep20_lr2e-2_full_sgd"


@pytest.mark.parametrize("optim", [plan.SGD, plan.ADAMW])
def test_only_410m_students_resume_mid_run(optim):
    setting = plan.Setting(plan.FULL, optim, 1e-3)
    assert cells.train_cfgs(setting, plan.PY410, 1)[1].checkpoint_resume
    assert not cells.train_cfgs(setting, plan.PY160, 1)[1].checkpoint_resume
    assert cells.train_cfgs(plan.FOCUS20[0], plan.PY410, 1)[1].checkpoint_resume
    assert not cells.train_cfgs(plan.FOCUS20[0], plan.PY160, 1)[1].checkpoint_resume


def test_experiment_sizes_match_the_brief():
    e6, e8, e9 = cells.e6_cells(), cells.e8_cells(), cells.e9_cells()
    teacher = lambda cs: [c for c in cs if c.source == plan.TEACHER]  # noqa: E731
    assert len(teacher(e6)) == 3 * 3 * 60
    assert len(e6) - len(teacher(e6)) == 3 * 20
    assert len(teacher(e8)) == 2 * 2 * 5 * 2
    assert len(e8) - len(teacher(e8)) == 2 * 5 * 2
    assert len(teacher(e9)) == 4 * 3 * 2 * 3
    assert len(e9) - len(teacher(e9)) == 2 * 3
    assert {c.key for c in e6 + e8 + e9} <= set(cells.cells_by_key())


def test_priority_2_sizes_match_the_brief():
    assert len(cells.e7_controls()) == 2 * 20
    assert len(cells.e7_up_cells()) == 3 * 60
    assert len(cells.e7_down_cells("blue", plan.SELECTED_RECIPES)) == 60
    down = cells.e23_down_cells()
    assert sum(c.source == plan.TEACHER for c in down) == 2 * 3 * 3
    assert sum(c.source == plan.BASE for c in down) == 3 * 3
    assert len(cells.e23_up_cells(plan.E8_RATES[plan.SGD][:3])) == 27


def test_arm_strata_hold_three_teachers_and_one_control():
    e6 = cells.e6_cells()
    for target, student, s in itertools.product(
        plan.E6_TARGETS, plan.E6_STUDENTS, plan.FOCUS20
    ):
        arm = [
            c
            for c in e6
            if c.student_base == student
            and c.setting == s
            and (c.target == target or c.source == plan.BASE)
        ]
        assert sum(c.source == plan.TEACHER for c in arm) == 3
        assert sum(c.source == plan.BASE for c in arm) == 1


def test_treatment_and_control_differ_only_in_the_pool():
    e6 = cells.e6_cells()
    treat = next(c for c in e6 if c.source == plan.TEACHER)
    ctrl = next(
        c
        for c in e6
        if c.source == plan.BASE
        and c.setting == treat.setting
        and c.student_base == treat.student_base
    )
    assert ctrl.seed == treat.seed
    assert cells.train_cfgs(treat.setting, treat.student_base, treat.seed) == (
        cells.train_cfgs(ctrl.setting, ctrl.student_base, ctrl.seed)
    )
    assert cells.pool(ctrl) == plan.CONTROL_POOLS[plan.PY160]
    assert cells.pool(treat) != cells.pool(ctrl)


def test_leaf_layout_is_cross_base_and_unique():
    leaves = [cells.run_dir(c) for c in cells.cells_by_key().values()]
    assert len(set(leaves)) == len(leaves)
    c = next(
        c
        for c in cells.e6_cells()
        if c.student_base == plan.WS1 and c.source == plan.TEACHER
    )
    assert "ckpt/pile-ws1-step143000/teacher-pythia-160m" in str(cells.run_dir(c))
    assert plan.TAG_N100K in str(cells.run_dir(c))


def test_pool_tags_follow_each_pool_not_the_hlopt_default():
    e8 = cells.e8_cells()
    dog = next(c for c in e8 if c.target == plan.DOG)
    lion = next(c for c in e8 if c.target == plan.LION)
    ctrl = next(c for c in e8 if c.source == plan.BASE)
    assert plan.TAG_N300K in str(cells.run_dir(dog))
    assert plan.TAG_N200K in str(cells.run_dir(lion))
    assert plan.TAG_N100K in str(cells.run_dir(ctrl))


def test_reuse_covers_only_matched_controls():
    reused = staging.reused_cells()
    assert len(reused) == 2 * 20 + 6 + 2 * 20
    assert {c.student_base for c in reused} == {
        plan.SEED1,
        plan.DATA1,
        plan.PY160,
        plan.PY410,
    }
    assert all(c.source == plan.BASE for c in reused)
    ws1 = [c for c in cells.e6_cells() if c.student_base == plan.WS1]
    assert all(cells.reused_source(c) is None for c in ws1)
    assert all(cells.reused_source(c) is None for c in cells.e8_cells())


def test_reused_source_paths_match_their_trees():
    e6 = next(c for c in staging.reused_cells() if c.exp == plan.E6)
    e9 = next(c for c in staging.reused_cells() if c.exp == plan.E9)
    assert "/pythia-160m-seed1/teacher-pythia-160m/ex10-20_prefix_n100k/" in str(
        cells.reused_source(e6)
    )
    assert str(cells.reused_source(e9)).endswith(
        "/pythia-160m/teacher-pythia-160m/ex10-20_prefix_n100k/"
        "ep20_lr2e-2_full_sgd/ftseed1"
    )


def _manifest(cell: cells.Cell) -> dict:
    return {
        "source": plan.BASE,
        "filtered_sha256": plan.CONTROL_POOLS[cell.teacher_base].filtered_sha256,
        "student_ftseed": cell.seed,
        "student_param_tag": cells.setting_tag(cell.setting),
        "subsample_size": plan.SUBSAMPLE_SIZE,
        "subsample_seed": plan.SUBSAMPLE_SEED,
        "base_model_repo": plan.HUB_REPOS[cell.student_base],
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("filtered_sha256", "other"),
        ("student_ftseed", 9),
        ("student_param_tag", "ep20_lr1e-4_r8a32"),
        ("subsample_seed", 7),
        ("base_model_repo", "EleutherAI/pythia-160m-seed2"),
    ],
)
def test_reused_manifest_mismatch_is_refused(field, value):
    cell = staging.reused_cells()[0]
    staging.check_reused_manifest(cell, _manifest(cell))
    with pytest.raises(ValueError, match=field):
        staging.check_reused_manifest(cell, {**_manifest(cell), field: value})


def test_controls_are_scored_for_every_target_of_their_experiment():
    e6_ctrl = next(c for c in cells.e6_cells() if c.source == plan.BASE)
    e8_ctrl = next(c for c in cells.e8_cells() if c.source == plan.BASE)
    assert set(plan.E6_TARGETS) <= set(e6_ctrl.eval_targets)
    assert set(plan.E9_TARGETS) <= set(e6_ctrl.eval_targets)
    assert e8_ctrl.eval_targets == plan.ANIMALS


def test_cell_refuses_a_treatment_without_a_target():
    with pytest.raises(ValueError):
        cells.Cell(
            plan.E6,
            plan.PY160,
            plan.SEED1,
            plan.TEACHER,
            None,
            None,
            plan.FOCUS20[0],
            1,
        )


def test_jobs_round_trip_through_their_key():
    for exp in plan.EXPERIMENTS:
        for job in cells.queue_jobs(exp):
            if "teacher-probe" in job.key:
                continue
            assert cells.cell_from_key(job.key).key == job.key
            assert job.cmd.startswith(".venv/bin/python -m scripts.run_rebhard_cell")


def test_cli_refuses_a_key_off_the_plan():
    with pytest.raises(ValueError):
        cells.cell_from_key("e6|nope")
    assert cli.app is not None


def test_collapse_cut_matches_hlopt_at_160m_and_the_anchor_at_410m():
    from subliminal_learning.campaign import hlopt_plan

    assert plan.collapse_dce(plan.PY160) == pytest.approx(hlopt_plan.COLLAPSE_DCE)
    assert plan.collapse_dce(plan.PY410) == pytest.approx(5.4738147, abs=1e-6)


# ---- resume ----------------------------------------------------------------------------


def _train(resume: bool) -> TrainConfig:
    return TrainConfig(
        learning_rate=1e-3,
        n_epochs=1,
        batch_size=1,
        max_length=8,
        max_grad_norm=1.0,
        warmup_ratio=0.0,
        lr_scheduler_type="cosine",
        seed=1,
        checkpoint_resume=resume,
    )


@pytest.mark.parametrize(
    "resume,manifest_exists,kept",
    [(True, False, True), (True, True, False), (False, False, False)],
)
def test_retrain_keeps_checkpoints_only_for_a_resuming_cell(
    resume, manifest_exists, kept
):
    assert keeps_resume_state(_train(resume), manifest_exists) is kept


def test_torn_checkpoints_are_set_aside_and_whole_ones_kept(tmp_path):
    whole = tmp_path / "checkpoint-2500"
    torn = tmp_path / "checkpoint-5000"
    whole.mkdir()
    torn.mkdir()
    (whole / cellpipe.TRAINER_STATE_FILE).write_text("{}")
    (torn / "model.safetensors").write_text("partial")
    moved = cellpipe.set_aside_torn_checkpoints(tmp_path)
    assert [p.name.startswith("torn-checkpoint-5000") for p in moved] == [True]
    assert [p.name for p in tmp_path.glob("checkpoint-*")] == ["checkpoint-2500"]
    assert (moved[0] / "model.safetensors").read_text() == "partial"
    assert not set(moved[0].name) & set(RESERVED_CHARS)


def test_a_truncated_trainer_state_counts_as_a_torn_save(tmp_path):
    cut = tmp_path / "checkpoint-2500"
    cut.mkdir()
    (cut / cellpipe.TRAINER_STATE_FILE).write_text('{"global_step": 25')
    assert len(cellpipe.set_aside_torn_checkpoints(tmp_path)) == 1


# ---- exceedance test -------------------------------------------------------------------


def _exact_null(v: np.ndarray) -> list[int]:
    """Every control assignment of a one-arm design, by enumeration."""
    n_cells, n_runs = v.shape
    out = []
    for idx in itertools.product(range(n_runs), repeat=n_cells):
        top = max(v[i, j] for i, j in enumerate(idx))
        out.append(
            sum(
                v[i, k] > top
                for i in range(n_cells)
                for k in range(n_runs)
                if k != idx[i]
            )
        )
    return out


def test_exceedance_counts_runs_above_the_arm_maximum_strictly():
    v = np.array([[0.5, 0.2, 0.1], [0.3, 0.2, 0.2]])
    res = metrics.exceedance_test(v, ["a", "a"], n_perm=2000)
    assert res.observed == 2
    assert res.n_treatment == 4
    assert res.share == 0.5


def test_exceedance_p_converges_to_exact_enumeration():
    v = np.random.default_rng(3).random((4, 3))
    v[:, 0] += 0.5
    null = _exact_null(v)
    observed = metrics.exceedance_test(v, ["a"] * 4, n_perm=10).observed
    exact = sum(n >= observed for n in null) / len(null)
    res = metrics.exceedance_test(v, ["a"] * 4, n_perm=20_000)
    assert res.p == pytest.approx(exact, abs=0.01)
    assert res.null_mean == pytest.approx(float(np.mean(null)), abs=0.05)


def test_exceedance_null_rate_is_one_over_controls_plus_one():
    rng = np.random.default_rng(0)
    v = rng.random((20, 4))
    res = metrics.exceedance_test(v, ["a"] * 20, n_perm=4000)
    assert res.null_mean / 60 == pytest.approx(1 / 21, abs=0.01)


def test_each_arm_is_judged_against_its_own_control_maximum():
    low = np.array([[0.2, 0.2, 0.1]] * 3)
    high = np.array([[0.9, 0.8, 0.85]] * 3)
    pooled = metrics.exceedance_test(
        np.concatenate([low, high]), ["a"] * 3 + ["b"] * 3, n_perm=200
    )
    assert pooled.observed == 6 + 3
    alone = metrics.exceedance_test(np.concatenate([low, high]), ["a"] * 6, n_perm=200)
    assert alone.observed == 3


def test_exceedance_ignores_missing_runs_and_needs_a_control():
    v = np.array([[0.5, np.nan, 0.1], [0.3, 0.2, 0.2]])
    res = metrics.exceedance_test(v, ["a", "a"], n_perm=200)
    assert res.n_treatment == 3
    with pytest.raises(ValueError):
        metrics.exceedance_test(np.array([[0.5, np.nan]]), ["a"], n_perm=10)


def test_exceedance_is_deterministic_at_a_fixed_seed():
    v = np.random.default_rng(1).random((6, 4))
    a = metrics.exceedance_test(v, ["a"] * 6, n_perm=500)
    assert a == metrics.exceedance_test(v, ["a"] * 6, n_perm=500)


def test_holm_adjusts_step_down_and_stays_monotone():
    out = metrics.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out == {
        "a": pytest.approx(0.03),
        "c": pytest.approx(0.06),
        "b": pytest.approx(0.06),
    }


def _row(source, target, recipe, setting, p, **kw) -> dict:
    return {
        "source": source,
        "target": target,
        "student_base": kw.get("student", plan.SEED1),
        "recipe": recipe,
        "setting": setting,
        "seed": kw.get("seed", 1),
        "optim": kw.get("optim", plan.SGD),
        "lr": kw.get("lr", 0.01),
        "p_test": p,
        "p_validation": kw.get("val", p),
        "collapsed": kw.get("collapsed", False),
        "delta_ce": kw.get("dce", 1.0),
    }


def test_arm_matrix_puts_the_control_last_and_keeps_targets_apart():
    rows = [
        _row("teacher", "blue", "r1", "s1", 0.4),
        _row("teacher", "blue", "r2", "s1", 0.5),
        _row("base", "blue", None, "s1", 0.1),
        _row("base", "maple", None, "s1", 0.9),
        _row("teacher", "maple", "r1", "s1", 0.8),
    ]
    m = metrics.arm_matrix(rows, "blue", plan.SEED1, ("r1", "r2"), ("s1",))
    assert m.tolist() == [[0.4, 0.5, 0.1]]


def test_arm_summary_counts_collapsed_exceeders():
    rows = [
        _row("teacher", "blue", "r1", "s1", 0.4, collapsed=True),
        _row("teacher", "blue", "r2", "s1", 0.05),
        _row("base", "blue", None, "s1", 0.1),
    ]
    m = metrics.arm_matrix(rows, "blue", plan.SEED1, ("r1", "r2"), ("s1",))
    out = metrics.arm_summary(rows, m, ("r1", "r2"))
    assert out["n_exceeders_collapsed"] == 1
    assert out["delta_mean"] == pytest.approx(0.125)
    assert out["max_control"] == 0.1


# ---- paired Δ and verdicts -------------------------------------------------------------

SPLITS = ("validation", "test")


def test_paired_delta_matches_control_of_the_same_setting_and_seed():
    rows = [
        _row("teacher", "dog", "r", "a", 0.5, seed=2),
        _row("base", "dog", None, "a", 0.1, seed=1),
        _row("base", "dog", None, "a", 0.2, seed=2),
        _row("base", "dog", None, "b", 0.3, seed=2),
    ]
    (pair,) = metrics.paired_deltas(rows, SPLITS)
    assert pair["delta"]["test"] == pytest.approx(0.3)
    assert pair["control_collapsed"] is False


def test_paired_delta_without_a_control_is_nan_not_zero():
    (pair,) = metrics.paired_deltas([_row("teacher", "dog", "r", "a", 0.5)], SPLITS)
    assert math.isnan(pair["delta"]["test"])
    assert pair["control_found"] is False


def test_paired_delta_refuses_duplicate_controls():
    rows = [_row("base", "dog", None, "a", 0.1), _row("base", "dog", None, "a", 0.2)]
    with pytest.raises(ValueError):
        metrics.paired_deltas(rows, SPLITS)


CUT = 5.0


def _rung(lr, val, test_by_seed, dce=1.0, optim=plan.SGD):
    return {
        "target": "dog",
        "optim": optim,
        "lr": lr,
        "delta_validation": val,
        "delta_test": float(np.mean(list(test_by_seed.values()))),
        "test_by_seed": test_by_seed,
        "delta_ce_mean": dce,
        "collapsed_frac": 0.0,
        "at_edge": False,
    }


def test_select_rung_skips_a_collapsed_mean_and_ties_to_the_smaller_rate():
    rungs = [
        _rung(0.02, 0.9, {1: 0.1, 2: 0.1}, dce=CUT),
        _rung(0.01, 0.2, {1: 0.1, 2: 0.1}),
        _rung(0.005, 0.2, {1: 0.1, 2: 0.1}),
    ]
    assert metrics.select_rung(rungs, "dog", plan.SGD, CUT)["lr"] == 0.005


def test_select_rung_keeps_a_rung_with_one_collapsed_student_under_the_mean_cut():
    rung = {**_rung(0.02, 0.9, {1: 0.1, 2: 0.1}, dce=4.0), "collapsed_frac": 0.5}
    assert metrics.select_rung([rung], "dog", plan.SGD, CUT)["lr"] == 0.02


def test_select_rung_flags_a_rung_at_the_ladder_edge():
    rungs = [
        _rung(0.02, 0.9, {1: 0.1, 2: 0.1}),
        _rung(0.01, 0.2, {1: 0.1, 2: 0.1}),
        _rung(0.005, 0.1, {1: 0.1, 2: 0.1}),
    ]
    assert metrics.select_rung(rungs, "dog", plan.SGD, CUT)["at_edge"]
    assert not metrics.select_rung(
        rungs[1:] + [_rung(0.03, 0.0, {1: 0.1})], "dog", plan.SGD, CUT
    )["at_edge"]


def test_select_rung_with_every_rung_collapsed_is_none():
    rungs = [_rung(0.02, 0.9, {1: 0.1, 2: 0.1}, dce=CUT + 1)]
    assert metrics.select_rung(rungs, "dog", plan.SGD, CUT) is None
    assert metrics.e8_verdict(None, 2)["present"] is False


@pytest.mark.parametrize(
    "seeds,present",
    [
        ({1: 0.1, 2: 0.05}, True),
        ({1: 0.1, 2: -0.01}, False),
        ({1: 0.01, 2: 0.01}, False),
        ({1: 0.1}, False),
    ],
)
def test_e8_verdict_needs_every_seed_positive_and_the_floor(seeds, present):
    assert metrics.e8_verdict(_rung(0.01, 0.1, seeds), 2)["present"] is present


def _unit(recipe, delta, collapsed=False, control_collapsed=False):
    return {
        "recipe": recipe,
        "collapsed": collapsed,
        "control_collapsed": control_collapsed,
        "delta": {"test": delta},
    }


def test_e9_verdict_supports_a_clear_effect_across_teachers():
    units = [_unit(r, 0.1 + 0.01 * s) for r in "abc" for s in range(3)]
    out = metrics.e9_verdict(units)
    assert out["supported"] and out["n_teachers"] == 3 and out["lo"] > 0


def test_e9_verdict_needs_two_teachers_and_a_half_kept():
    assert not metrics.e9_verdict([_unit("a", 0.2), _unit("a", 0.3)])["supported"]
    units = [_unit("a", 0.2), _unit("b", 0.3)] + [
        _unit("c", 0.3, collapsed=True) for _ in range(3)
    ]
    out = metrics.e9_verdict(units)
    assert not out["eligible"] and not out["supported"]


def test_e9_verdict_needs_every_teacher_positive():
    units = [_unit(r, 0.3) for r in "ab" for _ in range(3)]
    units += [_unit("c", -0.01) for _ in range(3)]
    out = metrics.e9_verdict(units)
    assert out["n_positive_teachers"] == 2 and not out["supported"]


def test_e9_verdict_excludes_a_unit_whose_control_collapsed():
    units = [_unit(r, 0.2 + 0.01 * s) for r in "abc" for s in range(3)]
    units.append(_unit("a", 9.0, control_collapsed=True))
    out = metrics.e9_verdict(units)
    assert (out["n_units"], out["n_kept"]) == (10, 9)
    assert out["delta_test"] < 1


def test_e9_verdict_below_the_floor_is_not_supported():
    units = [_unit(r, 0.01 + 0.001 * s) for r in "abc" for s in range(3)]
    assert not metrics.e9_verdict(units)["supported"]


# ---- pool review, results --------------------------------------------------------------


def test_pool_review_is_bound_to_the_planned_shas():
    pools = staging.p1_pools()
    want = {k: p.filtered_sha256 for k, p in pools.items()}
    problem = staging.pool_review_problem
    assert problem({"approved": True, "pools": want}, pools) is None
    assert problem(None, pools) == "pool review missing"
    assert "not approved" in problem({"approved": False, "pools": want}, pools)
    tampered = {**want, "control|pythia-160m": "x"}
    assert "differ" in problem({"approved": True, "pools": tampered}, pools)


def test_the_approved_priority_1_review_stays_valid_when_priority_2_adds_pools():
    assert len(staging.p1_pools()) == 16
    assert len(staging.p2_pools()) == 2
    assert set(staging.p1_pools()).isdisjoint(staging.p2_pools())
    assert {*staging.p1_pools(), *staging.p2_pools()} <= set(staging.planned_pools())


def test_planned_pools_cover_every_cell_on_an_existing_pool():
    pools = {p.directory for p in staging.planned_pools().values()}
    fixed = [cells.pool(c) for c in cells.cells_by_key().values()]
    assert {p.directory for p in fixed if p.filtered_sha256} <= pools


def test_new_pools_carry_no_pin_and_the_n100k_tag():
    cell = cells.e7_down_cells("blue", plan.SELECTED_RECIPES)[0]
    assert cells.pool(cell).filtered_sha256 is None
    assert plan.TAG_N100K in str(cells.run_dir(cell))
    assert "teacher-pythia-410m" in str(cells.run_dir(cell))


def _write_leaf(root: Path, cell: cells.Cell, target: str, p: float) -> None:
    leaf = cells.run_dir(cell)
    leaf.mkdir(parents=True, exist_ok=True)
    manifest = leaf / "train_manifest.json"
    if not manifest.is_file():
        manifest.write_text(json.dumps({"final_train_loss": 1.0, "n_train_steps": 10}))
    from subliminal_learning.utils.io import file_sha256

    probe = leaf / "eval" / "pile_probe.json"
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_text(
        json.dumps(
            {
                "train_manifest_sha256": file_sha256(manifest),
                "delta_ce": 1.0,
                "kl": 0.1,
                "collapsed": False,
            }
        )
    )
    sets = {
        cells.set_name(target, s): {
            "target_animal": target,
            "p_target": p,
            "ci": {"count": 50},
        }
        for s in cells.splits(target)
    }
    pref = leaf if cell.source == plan.TEACHER else leaf / "eval" / target
    pref.mkdir(parents=True, exist_ok=True)
    (pref / "preference_eval.json").write_text(json.dumps({"sets": sets}))


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(plan, "TEACHER_PROBE_DIR", tmp_path / "outputs" / "tp")
    return tmp_path


def test_e6_results_recover_a_planted_exceedance(fake_root):
    rng = np.random.default_rng(0)
    for cell in cells.e6_cells():
        for target in cell.eval_targets:
            ctrl = 0.02 + 0.01 * rng.random()
            strong = cell.student_base == plan.DATA1 and target == "blue"
            p = ctrl + 0.2 if cell.source == plan.TEACHER and strong else ctrl
            _write_leaf(fake_root, cell, target, p)
    out = results.build(plan.E6, fake_root / "res")
    claims = json.loads((out / "claims.json").read_text())
    assert claims["R6a"]["supported"]
    assert claims["R6c"]["blue"]["supported"]
    assert not claims["R6c"]["maple"]["supported"]
    assert (out / results.MANIFEST).is_file()
    import pandas as pd

    table = pd.read_csv(out / "exceedance.csv")
    arm = table[table["row"] == f"blue|{plan.DATA1}"].iloc[0]
    assert arm["exceeders"] == 60 and arm["n_treatment"] == 60
    assert json.loads((out / results.MANIFEST).read_text())["n_rows"] > 0


def test_results_refuse_a_set_with_a_missing_student(fake_root):
    cells_e9 = cells.e9_cells()
    for cell in cells_e9[1:]:
        for target in cell.eval_targets:
            _write_leaf(fake_root, cell, target, 0.1)
    with pytest.raises(RuntimeError, match="rows"):
        results.build(plan.E9, fake_root / "res")


def test_results_refuse_a_student_without_a_current_probe(fake_root):
    for cell in cells.e9_cells():
        for target in cell.eval_targets:
            _write_leaf(fake_root, cell, target, 0.1)
    leaf = cells.run_dir(cells.e9_cells()[-1])
    (leaf / "train_manifest.json").write_text('{"changed": true}')
    with pytest.raises(RuntimeError, match="pp1"):
        results.build(plan.E9, fake_root / "res")


def test_e8_results_pick_the_planted_rung_and_read_a_null_optimizer(fake_root):
    for cell in cells.e8_cells():
        for target in cell.eval_targets:
            treat = cell.source == plan.TEACHER
            planted = cell.setting.optim == plan.SGD and cell.setting.lr == 0.01
            _write_leaf(fake_root, cell, target, 0.3 if treat and planted else 0.1)
    out = results.build(plan.E8, fake_root / "res")
    claims = json.loads((out / "claims.json").read_text())
    assert claims["dog|sgd"]["present"] and claims["dog|sgd"]["lr"] == 0.01
    assert not claims["dog|adamw"]["present"]


def test_teacher_gate_reads_the_probe_against_the_three_nat_limit(fake_root):
    path = cells.teacher_probe_path(plan.DOG)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"delta_ce": 3.165}))
    assert results.teacher_gate(plan.DOG)["teacher_past_gate"] is True
    path.write_text(json.dumps({"delta_ce": 2.511}))
    assert results.teacher_gate(plan.DOG)["teacher_past_gate"] is False
    assert results.teacher_gate(plan.LION)["teacher_past_gate"] is False


# ---- priority 2 ------------------------------------------------------------------------


def test_e23_directions_use_their_teacher_pools():
    down = next(c for c in cells.e23_down_cells() if c.target == plan.DOG)
    up = next(c for c in cells.e23_up_cells((0.01,)) if c.target == plan.DOG)
    assert cells.pool(down) == plan.ANIMAL_POOLS_410M[plan.DOG]
    assert cells.pool(up) == plan.ANIMAL_POOLS_160M[plan.DOG]
    assert plan.TAG_N300K in str(cells.run_dir(down))
    assert plan.TAG_N100K in str(cells.run_dir(up))


def test_e23_controls_use_the_teacher_bases_untuned_pool():
    down = next(c for c in cells.e23_down_cells() if c.source == plan.BASE)
    up = next(c for c in cells.e23_up_cells((0.01,)) if c.source == plan.BASE)
    assert cells.pool(down) == plan.CONTROL_POOLS[plan.PY410]
    assert cells.pool(up) == plan.CONTROL_POOLS[plan.PY160]
    assert down.eval_targets == plan.ANIMALS


def test_e7_controls_reuse_the_matching_direction():
    up, down = cells.e7_controls()[0], cells.e7_controls()[-1]
    assert "/pythia-410m/teacher-pythia-160m/" in str(cells.reused_source(up))
    assert "/pythia-160m/teacher-pythia-410m/" in str(cells.reused_source(down))


def test_every_leaf_of_every_priority_is_unique():
    leaves = [cells.run_dir(c) for c in cells.cells_by_key().values()]
    assert len(set(leaves)) == len(leaves)


@pytest.mark.parametrize(
    "chosen,expected",
    [
        (0.01, (0.005, 0.01, 0.02)),
        (0.0025, (0.0025, 0.005, 0.01)),
        (0.035, (0.01, 0.02, 0.035)),
    ],
)
def test_nearest_rates_keep_three_rungs_at_the_edges(chosen, expected):
    assert metrics.nearest_rates(plan.E8_RATES[plan.SGD], chosen) == expected


def _shared(target, lr, val, dce=1.0):
    return {
        "target": target,
        "optim": plan.SGD,
        "lr": lr,
        "delta_validation": val,
        "delta_ce_mean": dce,
    }


def test_shared_rung_averages_animals_and_needs_both_under_the_cut():
    rungs = [
        _shared("dog", 0.01, 0.2),
        _shared("lion", 0.01, 0.2),
        _shared("dog", 0.02, 0.9),
        _shared("lion", 0.02, 0.1),
        _shared("dog", 0.035, 2.0),
        _shared("lion", 0.035, 2.0, dce=CUT),
    ]
    assert metrics.shared_rung(rungs, plan.ANIMALS, plan.SGD, CUT) == 0.02


def test_shared_rung_is_none_when_nothing_survives():
    rungs = [_shared("dog", 0.01, 0.2, dce=CUT), _shared("lion", 0.01, 0.2)]
    assert metrics.shared_rung(rungs, plan.ANIMALS, plan.SGD, CUT) is None


def test_e23_verdict_needs_the_seed_interval_above_zero():
    assert metrics.e23_verdict([0.1, 0.11, 0.12])["present"]
    assert not metrics.e23_verdict([0.1, -0.1, 0.0])["present"]


def test_pad_widens_treatments_and_keeps_the_control_last():
    m = np.array([[0.4, 0.5, 0.1]])
    out = results._pad(m, 4)
    assert out.shape == (1, 4)
    assert out[0, -1] == 0.1 and math.isnan(out[0, 2])


def test_pooled_arms_with_fewer_teachers_keep_each_arms_own_control():
    full = np.array([[0.5, 0.5, 0.5, 0.1]])
    short = np.array([[0.05, 0.9]])
    row = results._test_row("x", "all", [("a", full), ("b", short)])
    assert row["exceeders"] == 3
    assert row["n_treatment"] == 4


def test_e7_results_cover_both_directions_with_fewer_teachers(fake_root):
    selected = {t: list(plan.SELECTED_RECIPES) for t in plan.E7_TARGETS}
    selected["maple"] = list(plan.SELECTED_RECIPES[:2])
    planned = cells.e7_controls() + cells.e7_up_cells()
    planned += [c for t in plan.E7_TARGETS for c in cells.e7_down_cells(t, selected[t])]
    rng = np.random.default_rng(1)
    for cell in planned:
        for target in cell.eval_targets:
            base = 0.02 + 0.01 * rng.random()
            strong = cell.source == plan.TEACHER and cell.student_base == plan.PY160
            _write_leaf(fake_root, cell, target, base + 0.3 if strong else base)
    out = results.build(plan.E7, fake_root / "res", planned=planned, selected=selected)
    claims = json.loads((out / "claims.json").read_text())["R7"]
    assert claims["all"]["supported"]
    assert claims["by_direction"]["410m->160m"]["supported"]
    assert not claims["by_direction"]["160m->410m"]["supported"]


def test_e23_results_keep_the_two_directions_apart(fake_root):
    planned = cells.e23_down_cells() + cells.e23_up_cells((0.01, 0.02, 0.035))
    for cell in planned:
        for target in cell.eval_targets:
            treat = cell.source == plan.TEACHER and cell.student_base == plan.PY160
            bump = 0.01 * cell.seed
            _write_leaf(fake_root, cell, target, 0.3 + bump if treat else 0.1)
    out = results.build(plan.E23, fake_root / "res", planned=planned)
    claims = json.loads((out / "claims.json").read_text())
    assert claims["into pythia-160m|dog|0.01"]["present"]
    assert claims["into pythia-160m|dog|0.01"]["collapsed"] is False
    assert not claims["into pythia-410m|dog|0.01"]["present"]


def _pool_rows(n, completion=" 123"):
    return [{"prompt": "p", "completion": completion} for _ in range(n)]


def test_coherence_stats_read_length_and_letters():
    rows = _pool_rows(3) + _pool_rows(1, " 1, 2, 3") + _pool_rows(1, " abc 4")
    out = rt.coherence_stats(rows)
    assert out["n_rows"] == 5
    assert out["mean_numbers"] == pytest.approx(7 / 5)
    assert out["share_one_number"] == pytest.approx(4 / 5)
    assert out["n_distinct"] == 3
    assert out["n_with_letters"] == 1


def test_coherence_flags_letters_short_pools_and_a_length_off_the_references():
    ref = {"mean_numbers": 1.0, "share_one_number": 0.99, "share_distinct": 0.04}
    good = {
        "n_rows": 20_000,
        "mean_numbers": 1.2,
        "share_one_number": 0.8,
        "share_distinct": 0.05,
        "n_with_letters": 0,
    }
    assert rt.coherence_problems(good, [ref]) == ()
    assert rt.coherence_problems({**good, "n_with_letters": 2}, [ref])
    assert rt.coherence_problems({**good, "n_rows": 9_000}, [ref])
    assert rt.coherence_problems({**good, "mean_numbers": 9.8}, [ref])


def test_coherence_flags_a_pool_collapsed_onto_one_completion():
    ref = rt.coherence_stats([{"completion": f" {i % 40}"} for i in range(1000)])
    collapsed = rt.coherence_stats(_pool_rows(20_000))
    assert any("share_distinct" in x for x in rt.coherence_problems(collapsed, [ref]))


def test_teacher_tasks_stay_on_the_plan():
    assert len(rt.teacher_tasks()) == 3 * 10
    with pytest.raises(ValueError):
        rt.TeacherTask("oak", rt.RECIPES[0])
    with pytest.raises(ValueError):
        rt.PoolTask("blue", "ep1_lr1e-9_r1a1")


def test_teacher_layout_sits_under_the_campaign_root(fake_root):
    leaf = rt.teacher_leaf_dir("blue", rt.RECIPES[0])
    assert str(leaf).startswith(str(fake_root))
    assert rt.pool_dir("blue", rt.RECIPES[0]).name == "ex10-20"
    assert rt.filtered_dir("blue", rt.RECIPES[0]).name == "prefix"


def test_check_pool_binds_a_new_pool_to_its_approved_review(fake_root, monkeypatch):
    monkeypatch.setattr(plan, "REVIEW_DIR", fake_root / "reviews")
    cell = cells.e7_down_cells("blue", plan.SELECTED_RECIPES)[0]
    pool = cells.pool(cell).directory
    pool.mkdir(parents=True)
    (pool / "filtered.jsonl").write_text('{"completion": " 1"}\n')
    with pytest.raises(RuntimeError):
        cellpipe.check_pool(cell)
    path = rt.pool_review_path("blue")
    path.parent.mkdir(parents=True)
    from subliminal_learning.utils.io import file_sha256

    sha = file_sha256(pool / "filtered.jsonl")
    path.write_text(json.dumps({"approved": False, "pools": {cell.recipe: sha}}))
    with pytest.raises(RuntimeError):
        cellpipe.check_pool(cell)
    path.write_text(json.dumps({"approved": True, "pools": {cell.recipe: sha}}))
    assert cellpipe.check_pool(cell) == pool


def _driver_cfg(tmp_path):
    return driver.driver_config(
        output_root=tmp_path / "outputs",
        queue_root=tmp_path / "queue",
        repo_root=tmp_path,
        dry_run=False,
        auto_requeue=3,
        code_sha="abc",
        generated_dir_override=tmp_path / "gen",
    )


def test_a_closed_priority_2_gate_waits_and_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "P2_POOL_REVIEW_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(
        driver.staging,
        "latest_complete_report",
        lambda: {"at": "t", "n_reused_controls": len(staging.reused_cells())},
    )
    state: dict = {"transitions": {}}
    waiting: list = []
    assert driver._p2_gate(state, waiting) is False
    assert "pool review missing" in waiting[0]
    assert "p2_preflight" not in state["transitions"]


def test_the_priority_2_gate_opens_on_its_own_review(tmp_path, monkeypatch):
    path = tmp_path / "review.json"
    want = {k: p.filtered_sha256 for k, p in staging.p2_pools().items()}
    path.write_text(json.dumps({"approved": True, "pools": want}))
    monkeypatch.setattr(plan, "P2_POOL_REVIEW_PATH", path)
    monkeypatch.setattr(
        driver.staging,
        "latest_complete_report",
        lambda: {"at": "t", "n_reused_controls": len(staging.reused_cells())},
    )
    state: dict = {"transitions": {}}
    assert driver._p2_gate(state, []) is True
    assert "p2_preflight" in state["transitions"]


def test_every_queue_follows_the_pin_even_without_new_jobs(tmp_path, monkeypatch):
    cfg = _driver_cfg(tmp_path)
    moved: list = []
    monkeypatch.setattr(driver.driver_io, "queue_job_defaults", lambda c: {"cwd": "p"})
    monkeypatch.setattr(driver, "_submitted", lambda c: ["a", "b"])
    monkeypatch.setattr(
        driver.store, "update_queue_defaults", lambda r, n, d: moved.append((n, d))
    )
    driver._repin(cfg)
    assert moved == [("a", {"cwd": "p"}), ("b", {"cwd": "p"})]


def test_a_first_pass_without_a_pin_is_refused(tmp_path):
    cfg = _driver_cfg(tmp_path)
    with pytest.raises(driver.CampaignGateError):
        driver.run(driver.replace(cfg, code_sha=None))


def test_students_wait_for_the_review_of_their_pools(tmp_path, monkeypatch):
    cfg = _driver_cfg(tmp_path)
    usable = [plan.SELECTED_RECIPES[0], plan.SELECTED_RECIPES[1], rt.RECIPES[3]]
    e7 = {
        "gate": {"blue": {"ranked": usable}},
        "pools": {"blue": {r: {"status": "usable"} for r in usable}},
        "final": {"blue": True},
        "selected": {"blue": usable},
        "submitted": {},
    }
    put: list = []
    monkeypatch.setattr(driver, "_resolve_pools", lambda *a: None)
    monkeypatch.setattr(driver, "_put", lambda c, n, j: put.append(n) or len(j))
    monkeypatch.setattr(driver.rtp, "write_pool_samples", lambda *a: "sheet")
    monkeypatch.setattr(driver.rtp, "pool_review_problem", lambda *a: "not approved")
    fired: list = []
    waiting: list = []
    driver._advance_target(cfg, e7, "blue", fired, waiting)
    assert not put and "held" in waiting[0] and not e7["submitted"]
    monkeypatch.setattr(driver.rtp, "pool_review_problem", lambda *a: None)
    driver._advance_target(cfg, e7, "blue", fired, [])
    assert put == [plan.queue(plan.E7_DOWN)] and e7["submitted"]["blue"]


def test_choosing_rates_into_410m_follows_8s_rungs():
    rungs = [
        {**_shared(a, lr, val), "delta_ce_mean": 1.0}
        for a in plan.ANIMALS
        for lr, val in zip(plan.E8_RATES[plan.SGD], (0.1, 0.2, 0.3, 0.2, 0.1))
    ]
    assert driver.choose_e23_rates(rungs) == (0.005, 0.01, 0.02)


# ---- priority 3 ------------------------------------------------------------------------


def test_priority_3_sizes_match_the_brief():
    assert len(cells.e6x_cells()) == 3 * 3 * 60
    assert len(cells.e8m_cells()) == 5 * 2 * 3
    both = {(a, plan.SGD): 0.01 for a in plan.ANIMALS}
    assert len(cells.e8s_cells(both)) == 2 * 3


def test_momentum_is_plain_sgd_with_momentum_and_keeps_its_own_leaf():
    cell = cells.e8m_cells()[0]
    _, train = cells.train_cfgs(cell.setting, cell.student_base, cell.seed)
    assert (train.optim, train.optim_momentum) == ("sgd", 0.9)
    assert train.checkpoint_resume
    assert "sgdm0.9" in cells.setting_tag(cell.setting)


def test_momentum_rates_are_hlopt_rungs_a_tenth_of_the_sgd_ladder():
    from subliminal_learning.campaign import hlopt_plan

    ladder = hlopt_plan.A2_LADDERS[("full", "sgdm", hlopt_plan.MAX_GRAD_NORM)]
    assert set(plan.E8X_SGDM_RATES) <= set(ladder)
    for a, b in zip(plan.E8_RATES[plan.SGD], plan.E8X_SGDM_RATES):
        assert a / 10 == pytest.approx(b)


def test_extra_seeds_share_one_control_when_animals_pick_the_same_rate():
    both = {(a, plan.SGD): 0.01 for a in plan.ANIMALS}
    out = cells.e8s_cells(both)
    assert sum(c.source == plan.BASE for c in out) == len(plan.E8X_EXTRA_SEEDS)
    apart = {(plan.DOG, plan.SGD): 0.01, (plan.LION, plan.SGD): 0.02}
    assert sum(c.source == plan.BASE for c in cells.e8s_cells(apart)) == 4


def test_extra_seed_cells_are_planned_for_every_possible_rung():
    for optim, rates in plan.E8_RATES.items():
        for lr in rates:
            for c in cells.e8s_cells({(plan.DOG, optim): lr}):
                assert cells.cell_from_key(c.key) == c


def test_e6x_uses_yellows_own_teachers_and_the_e6_controls_score_it():
    yellow = {c.recipe for c in cells.e6x_cells() if c.target == "yellow"}
    assert yellow == set(plan.E6X_RECIPES["yellow"])
    ctrl = next(c for c in cells.e6_cells() if c.source == plan.BASE)
    assert set(plan.E6X_TARGETS) <= set(ctrl.eval_targets)
    assert all(cells.pool(c).filtered_sha256 for c in cells.e6x_cells())


def test_priority_3_pools_stay_out_of_the_approved_reviews():
    assert len(staging.p3_pools()) == 6
    assert set(staging.p3_pools()).isdisjoint(staging.p1_pools())


def test_e6x_results_reuse_the_e6_controls(fake_root):
    controls = [c for c in cells.e6_cells() if c.source == plan.BASE]
    planned = controls + cells.e6x_cells()
    rng = np.random.default_rng(2)
    for cell in planned:
        for target in cell.eval_targets:
            base = 0.02 + 0.01 * rng.random()
            strong = cell.source == plan.TEACHER and target == "oak"
            _write_leaf(fake_root, cell, target, base + 0.3 if strong else base)
    out = results.build(plan.E6X, fake_root / "res", planned=planned)
    claims = json.loads((out / "claims.json").read_text())["R6x"]
    assert claims["by_target"]["oak"]["supported"]
    assert not claims["by_target"]["yellow"]["supported"]


def test_e8x_results_pool_four_seeds_at_the_selected_rate(fake_root):
    selected = {(a, o): r for a in plan.ANIMALS for o, r in ((plan.SGD, 0.01),)}
    planned = cells.e8_cells() + cells.e8m_cells() + cells.e8s_cells(selected)
    for cell in planned:
        for target in cell.eval_targets:
            treat = cell.source == plan.TEACHER
            hit = cell.setting.optim == plan.SGD and cell.setting.lr == 0.01
            bump = 0.01 * cell.seed
            _write_leaf(fake_root, cell, target, 0.3 + bump if treat and hit else 0.1)
    out = results.build(plan.E8X, fake_root / "res", planned=planned, selected=selected)
    claims = json.loads((out / "claims.json").read_text())
    entry = claims["R8_seeds"]["dog|sgd|0.01"]
    assert entry["present"] and entry["n_seeds"] == 4
    assert not claims["R8_momentum"]["dog"]["present"]


def test_a_closed_priority_3_gate_waits(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "P3_POOL_REVIEW_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(
        driver.staging,
        "latest_complete_report",
        lambda: {"at": "t", "pools": staging.planned_pools()},
    )
    state: dict = {"transitions": {}}
    waiting: list = []
    assert driver._p3_gate(state, waiting) is False
    assert "pool review missing" in waiting[0]


def test_choosing_extra_seed_rates_skips_a_ladder_with_no_rung():
    rungs = [
        {**_shared(plan.DOG, 0.01, 0.2), "at_edge": False},
        {**_shared(plan.LION, 0.01, 0.2, dce=99.0), "at_edge": False},
    ]
    assert driver.choose_extra_seed_rates(rungs) == {(plan.DOG, plan.SGD): 0.01}
