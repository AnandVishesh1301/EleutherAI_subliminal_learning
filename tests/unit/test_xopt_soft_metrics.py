from __future__ import annotations

import dataclasses
import json
import math

import pytest

from pipelines import xopt_soft_campaign_pipeline as driver
from scripts.sweeps.sweep_xopt_soft import manifest_problems, require_evals
from subliminal_learning.campaign import optparam2_plan as oplan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import xopt_plan as plan
from subliminal_learning.campaign import xopt_rules as rules
from subliminal_learning.campaign import xopt_soft as soft
from subliminal_learning.campaign import xopt_soft_metrics as metrics
from subliminal_learning.jobqueue import store

CE = 2.7
CUT = plan.H_UNI - CE
OK_DCE, BAD_DCE = 0.1, CUT + 0.5


def _row(lf: soft.SoftLeaf, p: float, dce: float, p_val: float | None = None) -> dict:
    return {
        **dataclasses.asdict(lf),
        "key": lf.key,
        "p": {"test": p, "validation": p if p_val is None else p_val},
        "delta_ce": dce,
    }


def _corner(arm: str, lr: float, tier: str = "c", **kw) -> list[soft.SoftLeaf]:
    t, ts, s, ss = soft.corner_cells("bear")[1]
    base = dict(tier=tier, animal="bear", teacher=t, tstep=ts, student=s, sstep=ss)
    base |= dict(arm=arm, lr=lr, epochs=1)
    return [
        soft.SoftLeaf(kind=k, r=r, **{**base, **kw})
        for r in plan.SOFT_CORNER_R_P1
        for k in soft.KINDS
    ]


def _rows(leaves, treat_p, placebo_p, dce, p_val=None) -> list[dict]:
    return [
        _row(
            lf,
            treat_p if lf.kind == soft.KIND_TREAT else placebo_p,
            dce,
            None
            if p_val is None
            else (p_val if lf.kind == soft.KIND_TREAT else placebo_p),
        )
        for lf in leaves
    ]


def test_rate_pairs_twins_and_flags_degeneracy():
    leaves = _corner("full-sgd", 0.2)
    rows = _rows(leaves, 0.3, 0.1, OK_DCE)
    rows = [r for r in rows if not (r["r"] == 1 and r["kind"] == soft.KIND_PLACEBO)]
    (rate,) = metrics.group_rates(rows).values()
    d = rate.deltas("test")
    assert d[0] == pytest.approx(0.2) and math.isnan(d[1])
    assert not rules.eligible(rate.outcome(), "bear", CE)  # a NaN validation Δ
    (ok,) = metrics.group_rates(_rows(leaves, 0.3, 0.1, OK_DCE)).values()
    assert rules.eligible(ok.outcome(), "bear", CE)
    row = metrics.rate_row(ok, CE)
    assert row["cut"] == pytest.approx(CUT) and row["eligible"] and row["transfers"]
    (bad,) = metrics.group_rates(_rows(leaves, 0.3, 0.1, BAD_DCE)).values()
    assert not rules.eligible(bad.outcome(), "bear", CE)


def test_stepdown_ladder_cap_and_selection():
    initial = (0.2, 0.1)
    rows = _rows(_corner("full-sgd", 0.2) + _corner("full-sgd", 0.1), 0.3, 0.1, BAD_DCE)
    rates = list(metrics.group_rates(rows).values())
    assert metrics.stepdown(rates, initial, CE) == 0.03
    rows += _rows(_corner("full-sgd", 0.03), 0.3, 0.1, BAD_DCE)
    rates = list(metrics.group_rates(rows).values())
    assert metrics.stepdown(rates, initial, CE) == 0.01
    rows += _rows(_corner("full-sgd", 0.01), 0.3, 0.1, BAD_DCE)
    rates = list(metrics.group_rates(rows).values())
    assert metrics.stepdown(rates, initial, CE) is None
    assert metrics.selected(rates, CE) is None
    good = _rows(_corner("full-sgd", 0.2), 0.3, 0.1, OK_DCE, p_val=0.25)
    good += _rows(_corner("full-sgd", 0.1), 0.3, 0.1, OK_DCE, p_val=0.25)
    rates = list(metrics.group_rates(good).values())
    assert metrics.stepdown(rates, initial, CE) is None
    assert metrics.selected(rates, CE).lr == 0.1  # tie -> smaller rate


def test_claims_xs0_xs1_xs5_on_synthetic_rows():
    rows, ce = [], {}
    for animal in plan.ANIMALS_SOFT:
        for arm, lr in (("lora-adamw", 4e-4), ("full-adamw", 2e-5)):
            leaves = [
                soft.SoftLeaf(
                    "c",
                    k,
                    animal,
                    plan.PY,
                    plan.FINAL,
                    plan.PY,
                    plan.FINAL,
                    arm,
                    lr,
                    1,
                    r,
                )
                for r in (0, 1)
                for k in soft.KINDS
            ]
            rows += _rows(leaves, 0.4, 0.1, OK_DCE)
        b = [
            soft.SoftLeaf(
                "b",
                k,
                animal,
                plan.PY,
                plan.FINAL,
                plan.PY,
                plan.FINAL,
                *plan.BRANCH1_SLOT,
                32,
                0,
            )
            for k in soft.KINDS
        ]
        rows += _rows(b, 0.2, 0.05, OK_DCE)
    ce[ws160m_models.token(plan.PY, plan.FINAL)] = CE
    out = metrics.claims(rows, ce)
    assert out["XS0"]["verdict"] == metrics.SUPPORTED
    assert out["XS1"]["verdict"] == metrics.PENDING
    assert out["XS5"]["verdict"] == metrics.SUPPORTED
    assert out["XS2"]["bear"]["verdict"] == metrics.PENDING
    for r in rows:
        if r["tier"] == "c" and r["arm"] == "full-adamw" and r["animal"] == "lion":
            r["p"] = {"test": 0.12, "validation": 0.12}
    assert metrics.claims(rows, ce)["XS0"]["verdict"] == metrics.FAILED


def test_manifest_problems_flags_wrong_teacher_and_steps():
    lf = soft.SoftLeaf(
        "g",
        "treat",
        "bear",
        plan.SEED2,
        2000,
        plan.PY,
        plan.FINAL,
        "lora-sgd",
        0.3,
        1,
        0,
    )
    tdir = soft.teacher_dir(lf)
    manifest = {
        "gkd_tag": soft.leaf_gkd_tag(lf),
        "teacher_tag": soft.leaf_teacher_tag(lf),
        "target_animal": "bear",
        "base_model": lf.student_label,
        "control": False,
        "finetune_mode": "lora",
        "n_train_steps": 625,
        "run_set_id": soft.RUN_SET_ID,
        "gseed": 0,
        "teacher": {
            "workflow": "sdf",
            "param_tag": plan.TREATMENT_PARAM_TAG,
            "ftseed": 0,
            "base_model": lf.teacher_label,
            "cross_init": True,
            "checkpoint_dir": str(tdir / "checkpoints"),
            "manifest_sha256": "abc",
        },
        "train_hyperparameters": {
            "learning_rate": 0.3,
            "optim": "sgd",
            "max_grad_norm": 1.0,
            "batch_size": 32,
            "lora_r": 8,
            "lora_alpha": 16,
            "target_modules": "all-linear",
        },
        "gkd_hyperparameters": {
            "lmbda": 1.0,
            "beta": 1.0,
            "teacher_temp": 1.0,
            "vocab_mask": False,
        },
    }
    assert manifest_problems(lf, manifest, tdir, "abc") == {}
    wrong = {
        **manifest,
        "n_train_steps": 10,
        "teacher": {**manifest["teacher"], "base_model": "seed2-step4000"},
    }
    assert set(manifest_problems(lf, wrong, tdir, "abc")) == {
        "n_train_steps",
        "teacher.base_model",
    }
    assert set(manifest_problems(lf, manifest, tdir, "def")) == {
        "teacher.manifest_sha256"
    }


def _finish(cfg, name: str, rc: int = 0) -> None:
    for job in store.load_jobs(cfg.queue_root, name):
        if not store.has_result(cfg.queue_root, name, job.jobkey):
            store.record_result(
                cfg.queue_root,
                name,
                job,
                rc=rc,
                node="test",
                gpu=0,
                start_utc="t0",
                end_utc="t1",
                log="",
                attempt=1,
            )


def test_driver_stepdown_round_on_synthetic_results(tmp_path, monkeypatch):
    rows: dict[str, dict] = {}
    monkeypatch.setattr(driver, "node_pin_defaults", lambda cfg, host: {"cwd": "/pin"})
    monkeypatch.setattr(driver, "write_leaf_yamls", lambda cfg, leaves: 0)
    monkeypatch.setattr(
        driver.results, "leaf_rows", lambda leaves: [rows[lf.key] for lf in leaves]
    )
    cfg = driver.driver_config(
        output_root=tmp_path / "out", queue_root=tmp_path / "q", repo_root=tmp_path
    )
    first = _corner("full-sgd", 0.2) + _corner("full-sgd", 0.1)
    other = _corner("full-adamw", 2e-5)
    driver.submit_leaves(cfg, first + other)
    host = soft.leaf_host(first[0])
    train_q = soft.train_queue("c", host)
    assert store.read_meta(cfg.queue_root, soft.late_queue("c", host)).depends_on == [
        train_q
    ]
    ce = {ws160m_models.token(first[0].student, first[0].sstep): CE}
    state: dict = {"transitions": {}}
    rows.update({r["key"]: r for r in _rows(first, 0.3, 0.1, BAD_DCE)})
    rows.update({r["key"]: r for r in _rows(other, 0.3, 0.1, OK_DCE)})
    assert driver.stepdown_pass(cfg, state, ce) == []  # nothing finished yet
    _finish(cfg, train_q)
    notes = driver.stepdown_pass(cfg, state, ce)
    assert len(notes) == 1 and "-> 0.03, depth 1" in notes[0]
    down1 = soft.down_queue(1, host)
    keys1 = {j.key for j in store.load_jobs(cfg.queue_root, down1)}
    assert keys1 == {
        dataclasses.replace(lf, lr=0.03).key for lf in first if lf.lr == 0.1
    }
    assert store.read_meta(
        cfg.queue_root, soft.late_down_queue(1, host)
    ).depends_on == [down1]
    assert driver.stepdown_pass(cfg, state, ce) == []  # rung pending, adamw unchanged
    rows.update(
        {
            r["key"]: r
            for r in _rows([soft.leaf_from_key(k) for k in keys1], 0.3, 0.1, BAD_DCE)
        }
    )
    _finish(cfg, down1)
    notes = driver.stepdown_pass(cfg, state, ce)
    assert len(notes) == 1 and "-> 0.01, depth 2" in notes[0]
    down2 = soft.down_queue(2, host)
    keys2 = {j.key for j in store.load_jobs(cfg.queue_root, down2)}
    rows.update(
        {
            r["key"]: r
            for r in _rows([soft.leaf_from_key(k) for k in keys2], 0.3, 0.1, BAD_DCE)
        }
    )
    _finish(cfg, down2, rc=1)  # failed for good: a requeue hole, never "degenerate"
    assert driver.leaf_status(cfg)[next(iter(keys2))] == "failed"
    assert driver.stepdown_pass(cfg, state, ce) == []
    assert not state["degenerate"]
    store.requeue(
        cfg.queue_root, down2
    )  # repaired: the rung finishes, still degenerate
    _finish(cfg, down2)
    notes = driver.stepdown_pass(cfg, state, ce)
    assert len(notes) == 1 and notes[0].startswith("degenerate(")
    assert len(state["degenerate"]) == 1
    extra = driver.corner_stepdown_rates(driver.registered_leaves(cfg).values())
    assert extra == {(first[0].cell.key, "full-sgd"): [0.03, 0.01]}


def test_require_evals_refuses_a_skipped_eval(tmp_path):
    lf = _corner("full-sgd", 0.2)[0]
    with pytest.raises(RuntimeError, match="preference eval"):
        require_evals(lf, tmp_path)
    sets = {s: {"target_animal": "bear"} for s in oplan.PROMPT_SETS}
    (tmp_path / "preference_eval.json").write_text(json.dumps({"sets": sets}))
    require_evals(lf, tmp_path)
    sets["validation"]["target_animal"] = "lion"
    (tmp_path / "preference_eval.json").write_text(json.dumps({"sets": sets}))
    with pytest.raises(RuntimeError, match="validation"):
        require_evals(lf, tmp_path)
