from __future__ import annotations

import dataclasses
import json
import math

import pytest

from pipelines import xopt2_soft_campaign_pipeline as driver
from pipelines import xopt2_soft_results_pipeline as results
from scripts.sweeps.sweep_xopt2_soft import manifest_problems
from subliminal_learning.campaign import optparam2_plan as oplan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import xopt2_plan as plan
from subliminal_learning.campaign import xopt2_soft as soft
from subliminal_learning.campaign import xopt2_soft_metrics as metrics
from subliminal_learning.campaign import xopt_plan as xplan
from subliminal_learning.campaign import xopt_soft_metrics as xm
from subliminal_learning.jobqueue import store

CE = 2.7
CUT = xplan.H_UNI - CE
OK, BAD = 0.1, CUT + 0.5
ALL_CE = {t: CE for t in soft.student_tokens()}


def _group(tier: str, cell=soft.SELF_CELL, arm: str | None = None) -> soft.WalkGroup:
    for g in soft.walk_groups():
        if g.tier == tier and g.cell == cell and (arm is None or g.arm == arm):
            return g
    raise KeyError((tier, cell, arm))


DS1_CELL = (xplan.DS1, plan.FINAL, xplan.DS1, plan.FINAL)


class Outcomes:
    """(arm, lr) -> (p treat test, p treat validation, ΔCE); placebo p = 0.1."""

    def __init__(self) -> None:
        self.by_rate: dict[tuple[str, float], tuple[float, float, float]] = {}

    def set(self, arm: str, lr: float, delta: float, val: float, dce: float) -> None:
        self.by_rate[(arm, lr)] = (0.1 + delta, 0.1 + val, dce)

    def row(self, lf: soft.SoftLeaf, status: str) -> dict:
        p_test, p_val, dce = self.by_rate.get((lf.arm, lf.lr), (0.1, 0.1, OK))
        if lf.kind == soft.KIND_PLACEBO:
            p_test = p_val = 0.1
        return {
            **dataclasses.asdict(lf),
            "key": lf.key,
            "epochs": soft.EPOCHS,
            "status": status,
            "p": {**{s: p_test for s in oplan.PROMPT_SETS}, "validation": p_val},
            "delta_ce": dce,
        }

    def rows(self, leaves, status) -> list[dict]:
        return [self.row(lf, status.get(lf.key, "pending")) for lf in leaves]


def _finish(cfg, name: str, rc: int = 0, only: set[str] | None = None) -> None:
    for job in store.load_jobs(cfg.queue_root, name):
        if only is not None and job.key not in only:
            continue
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


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    out = Outcomes()
    monkeypatch.setattr(driver, "node_pin_defaults", lambda cfg, host: {"cwd": "/pin"})
    monkeypatch.setattr(driver, "write_leaf_yamls", lambda cfg, leaves: 0)
    monkeypatch.setattr(driver.results, "decision_rows", out.rows)
    cfg = driver.driver_config(
        output_root=tmp_path / "out", queue_root=tmp_path / "q", repo_root=tmp_path
    )
    state: dict = {"transitions": {}}

    def step() -> list[str]:
        reg, status = driver.registered(cfg), driver.leaf_status(cfg)
        notes = driver.walk_pass(cfg, state, ALL_CE, reg, status)
        return notes + driver.stepdown_pass(
            cfg, state, ALL_CE, driver.registered(cfg), status
        )

    return cfg, state, out, step


def _keys(cfg, name: str) -> set[str]:
    return {j.key for j in store.load_jobs(cfg.queue_root, name)}


# ---- pure metrics ---------------------------------------------------------------------


def _grid(values) -> list[dict]:
    steps = plan.SOFT_GRID_STEPS
    return [
        {
            "teacher": "a",
            "student": "b",
            "arm": "lora-adamw",
            "tstep": steps[i],
            "sstep": steps[j],
            "delta_test": v,
        }
        for (i, j), v in values.items()
    ]


def test_x2s3_product_and_additive_fits_with_dof_guards():
    full = {(i, j): 0.1 * (i + 1) * 0.2 * (j + 1) for i in range(5) for j in range(5)}
    fit = metrics.claim_x2s3(_grid(full))["grids"][0]
    assert fit["n_cells"] == 25 and fit["product_r2"] == pytest.approx(1.0)
    assert 0.8 < fit["additive_r2"] < 0.99
    additive = {(i, j): 0.1 * i + 0.3 * j for i in range(5) for j in range(5)}
    assert metrics.claim_x2s3(_grid(additive))["grids"][0][
        "additive_r2"
    ] == pytest.approx(1.0)
    few = {(i, j): v for (i, j), v in full.items() if i < 2 and j < 2 and i + j < 2}
    fit = metrics.claim_x2s3(_grid(few))["grids"][0]
    assert math.isnan(fit["product_r2"]) and "too_few_cells" in fit["product_reason"]
    assert math.isnan(fit["additive_r2"]) and "too_few_cells" in fit["additive_reason"]
    constant = {k: 0.2 for k in full}
    fit = metrics.claim_x2s3(_grid(constant))["grids"][0]
    assert math.isnan(fit["product_r2"]) and "constant" in fit["product_reason"]


def test_walk_view_decides_only_finished_leaves_and_flags_holes():
    g = _group("w", DS1_CELL)
    leaves = g.leaves(0.5)
    out = Outcomes()
    status = {lf.key: "done" for lf in leaves[:-1]}
    view = metrics.walk_view(
        g, (0.5,), xm.group_rates(out.rows(leaves, status)), ALL_CE
    )
    assert view.step.action == "wait" and not view.decided
    status[leaves[-1].key] = "done"
    view = metrics.walk_view(
        g, (0.5,), xm.group_rates(out.rows(leaves, status)), ALL_CE
    )
    assert view.decided == {0.5: True} and (view.step.action, view.step.rate) == (
        "run",
        1.0,
    )
    status[leaves[-1].key] = "failed"
    view = metrics.walk_view(
        g, (0.5,), xm.group_rates(out.rows(leaves, status)), ALL_CE
    )
    assert view.status == "hole" and view.holes == (0.5,) and not view.decided


def test_extra_rates_add_transferring_eligible_rates_for_exp1():
    g = _group("w", DS1_CELL)
    out = Outcomes()
    out.set("lora-sgd", 0.5, delta=0.05, val=0.01, dce=OK)
    out.set("lora-sgd", 1.0, delta=0.01, val=0.03, dce=OK)
    out.set("lora-sgd", 2.0, delta=0.0, val=0.0, dce=BAD)
    out.set("lora-sgd", 1.4, delta=0.0, val=0.0, dce=BAD)
    leaves = [lf for lr in (0.5, 1.0, 2.0, 1.4) for lf in g.leaves(lr)]
    rows = out.rows(leaves, {lf.key: "done" for lf in leaves})
    view = metrics.walk_view(g, (0.5, 1.0, 1.4, 2.0), xm.group_rates(rows), ALL_CE)
    assert view.status == "bracketed" and view.eligible == (0.5, 1.0)
    assert metrics.extra_rates(view, ALL_CE) == (0.5, 1.0)


def test_claims_x2s0_strict_and_x2s1_provisional():
    out = Outcomes()
    for arm, lr in plan.VALIDITY_SLOTS[plan.SOFT]:
        delta = 0.0 if arm == "lora-sgd" else 0.3
        out.set(arm, lr, delta=delta, val=delta, dce=OK)
    v = list(soft.tier_leaves("v"))
    py = _group("w")
    out.set("lora-sgd", 0.5, delta=0.0, val=0.0, dce=BAD)
    out.set("lora-sgd", 0.2, delta=0.0, val=0.0, dce=OK)
    leaves = [*v, *py.leaves(0.5), *py.leaves(0.2)]
    done = {lf.key: "done" for lf in leaves}
    claims = metrics.claims(out.rows(leaves, done), ALL_CE, {py.key: [0.3]}, {})
    assert claims["X2S0"]["verdict"] == metrics.SUPPORTED
    assert claims["X2S1"]["verdict"] == metrics.SUPPORTED
    assert claims["X2S1"]["py_self"]["reach"]["status"] == "bracketed"
    missing = {k: s for k, s in done.items() if k != v[0].key}
    claims = metrics.claims(out.rows(leaves, missing), ALL_CE, {py.key: [0.3]}, {})
    assert claims["X2S0"]["verdict"] == metrics.PENDING  # no verdict from a subset
    out.set("lora-sgd", 0.2, delta=0.05, val=0.0, dce=OK)
    claims = metrics.claims(out.rows(leaves, done), ALL_CE, {py.key: [0.3]}, {})
    hits = claims["X2S1"]["py_self"]["transfers"]
    assert {(h["lr"], h["n_leaves"], h["provisional"]) for h in hits} == {
        (0.2, 2, True)
    }
    assert claims["X2S1"]["verdict"] == metrics.PENDING
    third = soft.extra_leaves(py, (0.2,))
    rows = out.rows([*leaves, *third], {**done, **{lf.key: "done" for lf in third}})
    claims = metrics.claims(rows, ALL_CE, {py.key: [0.3]}, {})
    hits = claims["X2S1"]["py_self"]["transfers"]
    assert [(h["n_leaves"], h["provisional"]) for h in hits] == [(3, False)]
    assert claims["X2S1"]["verdict"] == metrics.NOT_SUPPORTED
    out.set("full-adamw", 2e-5, delta=0.01, val=0.01, dce=OK)
    claims = metrics.claims(out.rows(leaves, done), ALL_CE, {py.key: [0.3]}, {})
    assert claims["X2S0"]["verdict"] == metrics.FAILED_VERDICT


def test_x2s2_uses_the_carried_anchor_with_categories_and_contrasts():
    out = Outcomes()
    o = _group("o", arm="full-muon")
    out.set("full-muon", 2e-6, delta=0.1, val=0.1, dce=OK)
    corner = soft.other_corners()[0]
    carry = [lf for lf in soft.carry_leaves("full-muon", (2e-6,)) if lf.spec == corner]
    leaves = [*o.leaves(2e-6), *carry]
    rows = out.rows(leaves, {lf.key: "done" for lf in leaves})
    anchors = {"full-muon": {"rate": 2e-6, "null": False, "carry": [2e-6]}}
    x2s2 = metrics.claims(rows, ALL_CE, {}, anchors)["X2S2"]
    py = x2s2["arms"]["full-muon"]["py_self"]
    assert py["anchor"]["rate"] == 2e-6
    assert py["at_anchor"]["bear"]["delta_test"] == pytest.approx(0.1)
    assert py["at_anchor"]["lion"]["n_leaves"] == 2
    row = x2s2["arms"]["full-muon"]["cross"]["corners"][0]
    assert row["category"] == ws160m_models.category(corner[0], corner[2])
    assert row["at_anchor"]["delta_test"] == pytest.approx(0.1)
    assert row["eligible_carried_rates"] == [2e-6]
    head = x2s2["head_contrasts"][0]
    assert head["arms"] == ["full-muon", "full-muon-nh"]
    assert head["by_animal"]["bear"]["full-muon"]["delta_test"] == pytest.approx(0.1)
    assert math.isnan(head["by_animal"]["bear"]["full-muon-nh"]["delta_test"])


def test_grid_table_prefers_xopt2_cells_and_fits_x2s3():
    out = Outcomes()
    t, s = plan.SOFT_GRID_PAIRS[1]
    own = [
        lf
        for lf in soft.tier_leaves("g")
        if (lf.teacher, lf.student, lf.arm) == (t, s, "lora-adamw")
    ]
    out.set("lora-adamw", 4e-4, delta=0.2, val=0.2, dce=OK)
    rows = out.rows(own, {lf.key: "done" for lf in own})
    grid = metrics.grid_rows(rows, [], ALL_CE)
    cells = [
        r
        for r in grid
        if (r["teacher"], r["student"], r["arm"]) == (t, s, "lora-adamw")
    ]
    assert len(cells) == 25
    assert sum(r["source"] == "xopt2" for r in cells) == 24
    assert all(r["delta_test"] == pytest.approx(0.2) for r in cells if r["source"])
    x2s3 = metrics.claim_x2s3(grid)
    fit = next(
        g
        for g in x2s3["grids"]
        if (g["teacher"], g["student"], g["arm"]) == (t, s, "lora-adamw")
    )
    assert fit["n_cells"] == 24 and math.isnan(fit["product_r2"])  # constant cells


def test_manifest_problems_flags_freeze_and_teacher():
    lf = soft.SoftLeaf(
        "o",
        "treat",
        "bear",
        plan.PY,
        plan.FINAL,
        plan.PY,
        plan.FINAL,
        "full-muon-nh",
        2e-6,
        0,
    )
    tdir = soft.teacher_dir(lf)
    manifest = {
        "gkd_tag": soft.leaf_gkd_tag(lf),
        "teacher_tag": soft.leaf_teacher_tag(lf),
        "target_animal": "bear",
        "base_model": lf.student_label,
        "control": False,
        "finetune_mode": "full",
        "n_train_steps": 625,
        "run_set_id": soft.RUN_SET_ID,
        "gseed": 0,
        "teacher": {
            "workflow": "sdf",
            "param_tag": xplan.TREATMENT_PARAM_TAG,
            "ftseed": 0,
            "base_model": lf.teacher_label,
            "cross_init": False,
            "checkpoint_dir": str(tdir / "checkpoints"),
            "manifest_sha256": "abc",
        },
        "train_hyperparameters": {
            "learning_rate": 2e-6,
            "optim": "muon",
            "freeze_modules": ["embed_out"],
            "max_grad_norm": 1.0,
            "batch_size": 32,
        },
        "gkd_hyperparameters": {
            "lmbda": 1.0,
            "beta": 1.0,
            "teacher_temp": 1.0,
            "vocab_mask": False,
        },
    }
    assert manifest_problems(lf, manifest, tdir, "abc") == {}
    manifest["train_hyperparameters"]["freeze_modules"] = []
    manifest["teacher"]["ftseed"] = 1
    bad = manifest_problems(lf, manifest, tdir, "abc")
    assert set(bad) == {"freeze_modules", "teacher.ftseed"}


# ---- driver decisions on synthetic results ------------------------------------------------


def test_walk_climbs_bisects_then_submits_extra_leaves(campaign):
    cfg, state, out, step = campaign
    g = _group("w", DS1_CELL)
    driver.submit_leaves(cfg, g.leaves(0.5))
    assert step() == []  # nothing finished
    _finish(cfg, "xopt2-s-w")
    assert step() == [f"walk({g.key} climb -> 1)"]
    start = [lf for lf, _q in driver.registered(cfg).values() if lf.lr == 0.5]
    assert _keys(cfg, "xopt2-s-walk-w") == {
        dataclasses.replace(lf, lr=1.0).key for lf in start
    }
    assert store.read_meta(cfg.queue_root, "xopt2-s-late-walk-w").depends_on == [
        "xopt2-s-walk-w"
    ]
    assert step() == []  # rung pending
    out.set("lora-sgd", 1.0, delta=0.0, val=0.0, dce=BAD)
    _finish(cfg, "xopt2-s-walk-w")
    assert step() == [f"walk({g.key} bisect -> 0.7)"]
    out.set("lora-sgd", 0.7, delta=0.03, val=0.05, dce=OK)
    _finish(cfg, "xopt2-s-walk-w")
    notes = step()
    assert notes == [f"walk_done({g.key} bracketed; x at [0.7])"]
    assert _keys(cfg, "xopt2-s-x") == {lf.key for lf in soft.extra_leaves(g, (0.7,))}
    assert state["walks"][g.key]["extra"] == [0.7]
    assert step() == []  # x leaves are not decision inputs
    assert driver.walks_finished(state) is False  # other groups never started


def test_walk_steps_down_from_lowest_and_shares_validity_leaves(campaign):
    cfg, state, out, step = campaign
    g = _group("w")
    v = [lf for lf in soft.tier_leaves("v") if lf.arm == "lora-sgd"]
    driver.submit_leaves(cfg, [*v, *g.leaves(0.5)])
    _finish(cfg, "xopt2-s-v")
    out.set("lora-sgd", 0.5, delta=0.0, val=0.0, dce=BAD)
    _finish(cfg, "xopt2-s-w")
    assert step() == [f"walk({g.key} step-down -> 0.2)"]
    assert _keys(cfg, "xopt2-s-walk-w") == {lf.key for lf in g.leaves(0.2)}
    out.set("lora-sgd", 0.2, delta=0.0, val=0.0, dce=OK)
    _finish(cfg, "xopt2-s-walk-w")
    assert step() == [f"walk({g.key} bisect -> 0.3)"]
    assert state["walks"][g.key]["shared"] == [0.3]
    assert len(_keys(cfg, "xopt2-s-walk-w")) == 4  # 0.3 is held by tier v
    out.set("lora-sgd", 0.3, delta=0.0, val=0.0, dce=OK)
    assert step() == [f"walk_done({g.key} bracketed; x at [0.3])"]


def test_failed_decision_leaf_is_a_hole_until_repaired(campaign):
    cfg, state, out, step = campaign
    g = _group("w", DS1_CELL)
    driver.submit_leaves(cfg, g.leaves(0.5))
    first = next(iter(_keys(cfg, "xopt2-s-w")))
    _finish(cfg, "xopt2-s-w", rc=1, only={first})
    _finish(cfg, "xopt2-s-w")
    assert step() == []
    assert f"walk:{g.key}" in state["holes"]
    assert not store.queue_exists(cfg.queue_root, "xopt2-s-walk-w")
    store.requeue(cfg.queue_root, "xopt2-s-w")
    _finish(cfg, "xopt2-s-w")
    assert step() == [f"walk({g.key} climb -> 1)"]
    assert f"walk:{g.key}" not in state["holes"]


def test_new_arm_walk_anchors_and_carries(campaign):
    cfg, state, out, step = campaign
    g = _group("o", arm="full-muon")
    driver.submit_leaves(cfg, g.leaves(2e-6))
    out.set("full-muon", 2e-6, delta=0.1, val=0.1, dce=OK)
    out.set("full-muon", 5e-6, delta=0.0, val=0.0, dce=BAD)
    out.set("full-muon", 3e-6, delta=0.05, val=0.05, dce=OK)
    for name in ("xopt2-s-o", "xopt2-s-walk-o", "xopt2-s-walk-o"):
        _finish(cfg, name)
        step()
    rec = state["walks"][g.key]
    assert rec["done"] == "bracketed"
    assert rec["anchor"] == {
        "rate": 2e-6,
        "null": False,
        "carry": [2e-6],
        "reason": "selected",
    }
    assert _keys(cfg, "xopt2-s-ox") == {
        lf.key for lf in soft.carry_leaves("full-muon", (2e-6,))
    }
    assert _keys(cfg, "xopt2-s-x") == {
        lf.key for lf in soft.extra_leaves(g, (2e-6, 3e-6))
    }
    assert {lf.animal for lf in soft.extra_leaves(g, (2e-6,))} == {"bear", "lion"}


def test_new_arm_without_eligible_rate_is_labelled_no_carry(campaign):
    cfg, state, out, step = campaign
    state["labelled"] = {}
    g = _group("o", arm="lora-lion")
    for lr in (5e-6, 2e-6, 1e-6):
        out.set("lora-lion", lr, delta=0.0, val=0.0, dce=BAD)
    driver.submit_leaves(cfg, g.leaves(5e-6))
    _finish(cfg, "xopt2-s-o")
    assert step() == [f"walk({g.key} step-down -> 2e-06)"]
    _finish(cfg, "xopt2-s-walk-o")
    assert step() == [f"walk({g.key} step-down -> 1e-06)"]
    _finish(cfg, "xopt2-s-walk-o")
    assert step()[-1] == "no_carry(lora-lion)"
    assert state["walks"][g.key]["done"] == "degenerate"
    assert state["labelled"]["carry|lora-lion"] == "no_eligible_rate"
    assert f"degenerate|{g.key}" in state["labelled"]
    assert not any(k.startswith("ox") for k in state["holes"])
    assert not store.queue_exists(cfg.queue_root, "xopt2-s-ox")


def test_grid_group_steps_down_then_degenerates(campaign):
    cfg, state, out, step = campaign
    cell = soft.off_cross_cells()[0]
    leaves = [
        lf for lf in soft.tier_leaves("g") if lf.spec == cell and lf.arm == "full-sgd"
    ]
    for lr in (0.1, 0.03, 0.01):
        out.set("full-sgd", lr, delta=0.0, val=0.0, dce=BAD)
    driver.submit_leaves(cfg, leaves)
    _finish(cfg, "xopt2-s-g")
    notes = step()
    assert len(notes) == 1 and "-> 0.03, depth 1" in notes[0]
    assert _keys(cfg, "xopt2-s-down-g") == {
        dataclasses.replace(lf, lr=0.03).key for lf in leaves
    }
    _finish(cfg, "xopt2-s-down-g")
    assert "-> 0.01, depth 2" in step()[0]
    _finish(cfg, "xopt2-s-down-g", rc=1)
    assert step() == [] and not state["labelled"]  # failed for good: a hole
    store.requeue(cfg.queue_root, "xopt2-s-down-g")
    _finish(cfg, "xopt2-s-down-g")
    assert step()[0].startswith("degenerate(g|")
    assert any(k.startswith("degenerate|g|") for k in state["labelled"])


def _offline(monkeypatch, cfg) -> None:
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    cfg.state_path.write_text(
        json.dumps({"transitions": {"preflight": {}, "register": {}}})
    )
    monkeypatch.setattr(driver.driver_io, "ensure_campaign_code_pin", lambda c: None)
    monkeypatch.setattr(driver, "node_pin_defaults", lambda c, h: {"cwd": "/pin"})
    monkeypatch.setattr(driver, "write_leaf_yamls", lambda c, leaves: 0)
    monkeypatch.setattr(driver.staging, "latest_complete_report", lambda: None)
    monkeypatch.setattr(driver.results, "gl_complete_cells", lambda: frozenset())


def test_disk_floor_holds_the_launch_and_gl_reuse_is_recorded(tmp_path, monkeypatch):
    cfg = driver.driver_config(
        output_root=tmp_path / "out", queue_root=tmp_path / "q", repo_root=tmp_path
    )
    _offline(monkeypatch, cfg)
    monkeypatch.setattr(driver, "disk_problems", lambda: ["cold-1: 10 GiB free < 1000"])
    result = driver.run(cfg)
    state = json.loads(cfg.state_path.read_text())
    assert "submit_launch" not in state["transitions"]
    assert "disk_low" in state["labelled"] and not state["holes"]
    assert any("held" in w for w in result.waiting)
    assert not (tmp_path / "q").exists()
    reused = frozenset(soft.cross_cells()[:3])
    monkeypatch.setattr(driver.results, "gl_complete_cells", lambda: reused)
    monkeypatch.setattr(driver, "disk_problems", lambda: [])
    result = driver.run(cfg)
    assert "submit_launch" in result.fired
    assert len(_keys(cfg, "xopt2-s-g")) == 1320 - 3 * 2
    state = json.loads(cfg.state_path.read_text())
    assert "disk_low" not in state["labelled"]
    assert {tuple(c) for c in state["gl_reuse"]["complete"]} == reused
    assert len(state["gl_reuse"]["missing"]) == 129


def test_phase_gates_need_finished_queues_and_no_walk_hole(tmp_path):
    cfg = driver.driver_config(
        output_root=tmp_path / "out", queue_root=tmp_path / "q", repo_root=tmp_path
    )
    state = {"transitions": {"submit_launch": {}}, "walks": {}, "holes": {}}
    assert not driver.phase_ready(cfg, state, "p1")
    state["walks"] = {g.key: {"extra": []} for g in soft.walk_groups()}
    assert driver.phase_ready(cfg, state, "p1")
    state["holes"]["walk:x"] = "failed"
    assert not driver.walks_finished(state) and not driver.phase_ready(cfg, state, "p1")
    del state["holes"]["walk:x"]
    v = list(soft.tier_leaves("v"))[:2]
    store.create_queue(cfg.queue_root, "xopt2-s-v", node="test", priority=97)
    store.append_jobs(cfg.queue_root, "xopt2-s-v", [soft.train_job(lf) for lf in v])
    _finish(cfg, "xopt2-s-v", rc=1, only={v[0].key})
    _finish(cfg, "xopt2-s-v")
    assert not driver.phase_ready(cfg, state, "p1")  # complete but one failed
    store.requeue(cfg.queue_root, "xopt2-s-v")
    _finish(cfg, "xopt2-s-v")
    assert driver.phase_ready(cfg, state, "p1")
    assert not driver.phase_ready(cfg, state, "p2")
    state["transitions"]["results_p1"] = {}
    assert driver.phase_ready(cfg, state, "p2")


def test_finalize_waits_for_holes_and_markers_are_create_only(tmp_path, monkeypatch):
    cfg = driver.driver_config(
        output_root=tmp_path / "out", queue_root=tmp_path / "q", repo_root=tmp_path
    )
    _offline(monkeypatch, cfg)
    monkeypatch.setattr(driver, "disk_problems", lambda: [])
    state = json.loads(cfg.state_path.read_text())
    state["transitions"] |= {
        "submit_launch": {},
        "results_p1": {},
        "results_p2": {},
        "results_p3": {},
    }
    state["holes"] = {"xopt2-s-g": "job failed after attempt=3"}
    cfg.state_path.write_text(json.dumps(state))
    result = driver.run(cfg)
    assert not result.done and not (cfg.logs_dir / soft.CAMPAIGN_DONE_FILE).exists()
    assert any("finalize: holes" in w for w in result.waiting)
    state = json.loads(cfg.state_path.read_text())
    state["holes"] = {}
    cfg.state_path.write_text(json.dumps(state))
    assert driver.run(cfg).done
    marker = cfg.logs_dir / soft.CAMPAIGN_DONE_FILE
    first = marker.read_text()
    assert not driver.write_marker(marker) and marker.read_text() == first
    other, campaign = tmp_path / "hard_done", tmp_path / "CAMPAIGN_DONE"
    assert not driver.write_campaign_marker(other, campaign) and not campaign.exists()
    other.write_text("t\n")
    assert driver.write_campaign_marker(other, campaign) and campaign.exists()
    assert not driver.write_campaign_marker(other, campaign)


def test_write_phase_manifest_last_and_partial_phase_rebuilt(tmp_path, monkeypatch):
    out = Outcomes()
    v = list(soft.tier_leaves("v"))
    rows = out.rows(v, {lf.key: "done" for lf in v})
    rdir = tmp_path / "results"
    manifest = results.write_phase(rdir, "p1", rows, ALL_CE, {}, {}, [])
    assert manifest["suffix"] == "_p1"
    assert set(manifest["files"]) == {
        f"{n}_p1.{e}"
        for n, e in (
            ("leaves", "csv"),
            ("rates", "csv"),
            ("walks", "csv"),
            ("grid", "csv"),
            ("claims", "json"),
        )
    }
    claims = json.loads((rdir / "claims_p1.json").read_text())["claims"]
    assert set(claims) == {"X2S0", "X2S1", "X2S2", "X2S3"}
    before = {p.name: p.read_text() for p in rdir.iterdir()}
    assert results.write_phase(rdir, "p1", rows[:4], ALL_CE, {}, {}, []) == manifest
    assert {p.name: p.read_text() for p in rdir.iterdir()} == before
    (rdir / "leaves_p2.csv").write_text("partial\n")  # crash before the manifest
    monkeypatch.setattr(
        results.metrics,
        "claims",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("claims failed")),
    )
    with pytest.raises(RuntimeError):
        results.write_phase(rdir, "p2", rows, ALL_CE, {}, {}, [])
    assert not list(rdir.glob("*_p2r2.*"))  # nothing written before claims built
    monkeypatch.undo()
    manifest = results.write_phase(rdir, "p2", rows, ALL_CE, {}, {}, [])
    assert manifest["suffix"] == "_p2r2"
    assert (rdir / "leaves_p2.csv").read_text() == "partial\n"
    assert (rdir / "manifest_p2r2.json").is_file()


def test_nonfinite_loss_detector_matches_only_that_failure():
    head = (
        "11:00:00 | ERROR | [1/1] FAILED: lmb1.0\nTraceback (most recent call last):\n"
    )
    ok = (
        head
        + '  File "x.py", line 3\nRuntimeError: non-finite GKD loss at step 37: nan\n'
    )
    assert soft.nonfinite_loss(ok) == (37, "nan")
    assert soft.nonfinite_loss(ok.replace("nan", "inf")) == (37, "inf")
    assert soft.nonfinite_loss(head + "RuntimeError: CUDA out of memory\n") is None
    assert (
        soft.nonfinite_loss(ok + "torch.OutOfMemoryError: CUDA out of memory\n") is None
    )
    assert soft.nonfinite_loss(ok + ok) is None  # two failed cells
    assert soft.nonfinite_loss("1 skipped as claimed\n") is None


def test_nonfinite_loss_leaf_is_recorded_degenerate_and_ineligible(
    tmp_path, monkeypatch
):
    from scripts.sweeps import sweep_xopt2_soft as cli

    lf = soft.SoftLeaf(
        "w", "treat", "bear", *DS1_CELL[:2], *DS1_CELL[2:], "lora-sgd", 0.5, 0
    )
    root = tmp_path / "leaves"
    monkeypatch.setattr(
        soft, "leaf_dir", lambda leaf, *a, **k: root / leaf.ident.replace("|", "_")
    )
    with pytest.raises(RuntimeError, match="train sweep failed"):
        cli.train(lf, lambda cmd: (1, "RuntimeError: CUDA out of memory\n"))
    assert not soft.nonfinite_record(lf).exists()
    assert cli.train(lf, lambda cmd: (0, "")) == "trained"
    failure = "[1/1] FAILED: x\nRuntimeError: non-finite GKD loss at step 9: nan\n"
    assert cli.train(lf, lambda cmd: (1, failure)) == "degenerate"
    record = json.loads(soft.nonfinite_record(lf).read_text())
    assert (record["step"], record["loss"], record["ident"]) == (9, "nan", lf.ident)
    calls = []
    assert cli.train(lf, lambda cmd: calls.append(cmd) or (0, "")) == "degenerate"
    assert not calls  # never retrained
    probe = cli.degenerate_probe(lf, cli._degenerate_record(lf), CE)
    assert probe["finite"] is False and probe["degenerate"] is True
    assert cli.degenerate_probe(lf, cli._degenerate_record(lf), CE) == probe
    row = results.decision_row(lf, metrics.DONE)
    assert row["degenerate"] and math.isnan(row["delta_ce"])
    g = _group("w", DS1_CELL)
    twins = [x for x in g.leaves(0.5) if x != lf]
    rows = [row, *Outcomes().rows(twins, {x.key: "done" for x in twins})]
    view = metrics.walk_view(g, (0.5,), xm.group_rates(rows), ALL_CE)
    assert view.decided == {0.5: False} and not view.holes


def test_pin_problem_requires_equal_channel_pins(tmp_path, monkeypatch):
    cfg = driver.driver_config(
        output_root=tmp_path / "out", queue_root=tmp_path / "q", repo_root=tmp_path
    )
    cards = {}
    monkeypatch.setattr(driver, "read_run_set_card", lambda root: cards.get(root))
    monkeypatch.setattr(driver.code_pin_io, "resolve_commit_sha", lambda repo, sha: sha)
    assert driver.pin_problem(cfg) is None  # no hard card yet
    cards[plan.run_set_dir(plan.HARD)] = {"git_sha": "aaa"}
    assert "--code-sha" in driver.pin_problem(cfg)
    cfg = dataclasses.replace(cfg, code_sha="aaa")
    assert driver.pin_problem(cfg) is None
    cards[driver.pin_config(cfg).run_set_root] = {"git_sha": "bbb"}
    assert "pinned to aaa" in driver.pin_problem(cfg)


def test_ce_tokens_cover_every_cell():
    tokens = {
        ws160m_models.token(c[2], c[3])
        for c in (*soft.corner_cells(), *soft.grid_cells())
    }
    assert tokens == set(soft.student_tokens()) == set(ALL_CE)
