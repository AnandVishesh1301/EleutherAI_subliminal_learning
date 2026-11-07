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
from pipelines import rebhard_cell_pipeline as cellpipe
from pipelines import rebhard_results_pipeline as results
from pipelines import rebhard_staging_pipeline as staging
from pipelines.distill_student_pipeline import keeps_resume_state
from subliminal_learning.campaign import rebhard_cells as cells
from subliminal_learning.campaign import rebhard_metrics as metrics
from subliminal_learning.campaign import rebhard_plan as plan
from subliminal_learning.config import TrainConfig

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
def test_only_410m_full_ft_resumes_mid_run(optim):
    setting = plan.Setting(plan.FULL, optim, 1e-3)
    assert cells.train_cfgs(setting, plan.PY410, 1)[1].checkpoint_resume
    assert not cells.train_cfgs(setting, plan.PY160, 1)[1].checkpoint_resume


def test_experiment_sizes_match_the_brief():
    e6, e8, e9 = cells.e6_cells(), cells.e8_cells(), cells.e9_cells()
    teacher = lambda cs: [c for c in cs if c.source == plan.TEACHER]  # noqa: E731
    assert len(teacher(e6)) == 3 * 3 * 60
    assert len(e6) - len(teacher(e6)) == 3 * 20
    assert len(teacher(e8)) == 2 * 2 * 5 * 2
    assert len(e8) - len(teacher(e8)) == 2 * 5 * 2
    assert len(teacher(e9)) == 4 * 3 * 2 * 3
    assert len(e9) - len(teacher(e9)) == 2 * 3
    assert len(cells.cells_by_key()) == len(e6) + len(e8) + len(e9)


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
    assert len(reused) == 2 * 20 + 6
    assert {c.student_base for c in reused} == {plan.SEED1, plan.DATA1, plan.PY160}
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
    "resume,stored,kept",
    [(True, None, True), (True, {"x": 1}, False), (False, None, False)],
)
def test_retrain_keeps_checkpoints_only_for_a_resuming_cell(resume, stored, kept):
    assert keeps_resume_state(_train(resume), stored) is kept


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


def test_paired_delta_without_a_control_is_nan_not_zero():
    (pair,) = metrics.paired_deltas([_row("teacher", "dog", "r", "a", 0.5)], SPLITS)
    assert math.isnan(pair["delta"]["test"])
    assert pair["control_found"] is False


def test_paired_delta_refuses_duplicate_controls():
    rows = [_row("base", "dog", None, "a", 0.1), _row("base", "dog", None, "a", 0.2)]
    with pytest.raises(ValueError):
        metrics.paired_deltas(rows, SPLITS)


def _rung(lr, val, test_by_seed, collapsed=0.0, optim=plan.SGD):
    return {
        "target": "dog",
        "optim": optim,
        "lr": lr,
        "delta_validation": val,
        "delta_test": float(np.mean(list(test_by_seed.values()))),
        "test_by_seed": test_by_seed,
        "collapsed_frac": collapsed,
    }


def test_select_rung_skips_collapsed_and_ties_to_the_smaller_rate():
    rungs = [
        _rung(0.02, 0.9, {1: 0.1, 2: 0.1}, collapsed=0.5),
        _rung(0.01, 0.2, {1: 0.1, 2: 0.1}),
        _rung(0.005, 0.2, {1: 0.1, 2: 0.1}),
    ]
    assert metrics.select_rung(rungs, "dog", plan.SGD)["lr"] == 0.005


def test_select_rung_with_every_rung_collapsed_is_none():
    rungs = [_rung(0.02, 0.9, {1: 0.1, 2: 0.1}, collapsed=1.0)]
    assert metrics.select_rung(rungs, "dog", plan.SGD) is None
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


def _unit(recipe, delta, collapsed=False):
    return {"recipe": recipe, "collapsed": collapsed, "delta": {"test": delta}}


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


def test_e9_verdict_below_the_floor_is_not_supported():
    units = [_unit(r, 0.01 + 0.001 * s) for r in "abc" for s in range(3)]
    assert not metrics.e9_verdict(units)["supported"]


# ---- pool review, results --------------------------------------------------------------


def test_pool_review_is_bound_to_the_planned_shas():
    want = {k: p.filtered_sha256 for k, p in staging.planned_pools().items()}
    assert staging.pool_review_problem({"approved": True, "pools": want}) is None
    assert staging.pool_review_problem(None) == "pool review missing"
    assert "not approved" in staging.pool_review_problem(
        {"approved": False, "pools": want}
    )
    assert "differ" in staging.pool_review_problem(
        {"approved": True, "pools": {**want, "control|pythia-160m": "x"}}
    )


def test_planned_pools_cover_every_cell():
    pools = {p.directory for p in staging.planned_pools().values()}
    assert {cells.pool(c).directory for c in cells.cells_by_key().values()} <= pools


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
    manifest = json.loads((out / results.MANIFEST).read_text())
    assert manifest["n_rows"] == manifest["n_rows_expected"]


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
