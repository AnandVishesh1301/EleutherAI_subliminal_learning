"""`xopt2` hard driver end to end on a shrunken plan: walks (climb, bisect, bracketed,
degenerate), step-downs, extra seed, anchor + carry gating, late evals, phase results,
finalize, disk_low hold and failed decision leaves (fake jobs, temp roots, no GPU)."""

from __future__ import annotations

import json
import os
import shlex
from collections import Counter
from pathlib import Path
from types import MappingProxyType

import pandas as pd
import pytest

from pipelines import xopt2_hard_campaign_pipeline as driver
from pipelines import xopt2_hard_results_pipeline as results
from pipelines import xopt2_hard_staging_pipeline as staging
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import xopt2_hard as hard
from subliminal_learning.campaign import xopt2_plan as plan
from subliminal_learning.campaign import xopt_hard as xh
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job

REPO = Path(__file__).resolve().parents[2]
PY, PY115 = hard.PY_FINAL, hard.PY_SECOND
DS1 = xh.token(plan.TIER_C_SELF[1], plan.FINAL)
GRID = hard.GRID_PAIRS[0]
NEW = ("full-muon", "full-lion")
CE = 3.0  # cut = H_UNI - 3.0 = 4.80
K = {"full-adamw": 4e5, "full-muon": 1e6, "full-lion": 1e9, "full-sgd": 10.0}
FACTOR = {((DS1, PY), "full-adamw"): 100.0, ((DS1, PY), "full-muon"): 2.0}
ADAMW_SIGNAL = {2e-6: 0.05, 5e-6: 0.04}


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    ce = {t: CE for t in hard.student_tokens()}
    pairs = {
        "v": ((PY, PY),),
        "w": ((PY, PY), (DS1, PY)),
        "wk": ((PY115, PY115),),
        "o": ((PY, PY),),
        "ox": ((DS1, PY),),
        "ok": ((PY115, PY115),),
        "g": (GRID,),
    }
    arms = {
        "v": ("full-adamw",),
        "w": ("full-adamw",),
        "wk": ("full-adamw",),
        "o": NEW,
        "ox": NEW,
        "ok": NEW,
        "g": ("full-sgd",),
    }
    walked = frozenset(
        (p, a) for t in hard.WALK_TIERS for p in pairs[t] for a in arms[t]
    )
    for name, value in (
        ("TIER_PAIRS", MappingProxyType(pairs)),
        ("TIER_ARMS", MappingProxyType(arms)),
        ("WALKED", walked),
        ("LEAF_ROOT", tmp_path / "leaves"),
        ("QUEUE_ROOT", tmp_path / "queue"),
        ("XOPT_STATE_PATH", tmp_path / "no_xopt_state.json"),
    ):
        monkeypatch.setattr(hard, name, value)
    monkeypatch.setattr(plan, "NEW_ARMS", NEW)
    monkeypatch.setattr(
        plan, "VALIDITY_SLOTS", MappingProxyType({plan.HARD: (("full-adamw", 5e-6),)})
    )
    monkeypatch.setattr(
        plan, "GRID_SLOTS", MappingProxyType({plan.HARD: (("full-sgd", 0.02),)})
    )
    monkeypatch.setattr(plan, "CAMPAIGN_DONE_PATH", tmp_path / "CAMPAIGN_DONE")
    monkeypatch.setattr(
        plan, "channel_done_path", lambda ch: tmp_path / ch / "CAMPAIGN_DONE"
    )
    monkeypatch.setattr(xh, "CE_BASE_PATH", tmp_path / "ce_base.json")
    xh.CE_BASE_PATH.write_text(json.dumps(ce))
    monkeypatch.setattr(
        driver, "node_pin_defaults", lambda cfg, host: {"cwd": str(tmp_path)}
    )
    monkeypatch.setattr(driver, "disk_low", lambda: [])
    monkeypatch.setattr(driver_io, "ensure_campaign_code_pin", lambda cfg: None)
    cfg = driver.driver_config(
        output_root=tmp_path / "out",
        queue_root=tmp_path / "queue",
        repo_root=REPO,
        auto_requeue=3,
    )
    cfg.logs_dir.mkdir(parents=True)
    driver_io.save_state(cfg, {"transitions": {"preflight": {}, "register": {}}})
    return cfg


def _write(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def _signal(cell: hard.Cell) -> float:
    if cell.arm == "full-muon":
        return 0.1 + 1e4 * cell.rate
    if (cell.teacher, cell.student, cell.arm) == (PY, PY, "full-adamw"):
        return ADAMW_SIGNAL.get(cell.rate, 0.0)
    return 0.0


def _fake_train(job) -> None:
    cell = hard.cell_from_job(job)
    leaf = hard.run_dir(cell)
    k = K[cell.arm] * FACTOR.get(((cell.teacher, cell.student), cell.arm), 1.0)
    dce = k * cell.rate * (1.0 if cell.source == hard.TEACHER else 0.5)
    _write(leaf / "train_manifest.json", {"final_train_loss": 1.0, "n_train_steps": 10})
    _write(xh.probe_path(leaf), {"delta_ce": dce, "kl": 0.1, "finite": True})
    p = 0.1 + (_signal(cell) if cell.source == hard.TEACHER else 0.0)
    for animal in cell.eval_animals:
        sets = {
            s: {"target_animal": animal, "p_target": p, "ci": {"count": 5}}
            for s in ("validation", "test", "favorite", "off_topic")
        }
        _write(xh.preference_path(leaf, cell.source, animal), {"sets": sets})


def _record(cfg, name, job, rc=0, attempt=1) -> None:
    store.record_result(
        cfg.queue_root,
        name,
        job,
        rc=rc,
        node="t",
        gpu=0,
        start_utc="",
        end_utc="",
        log="",
        attempt=attempt,
    )


def _drain(cfg, skip: tuple[str, ...] = ()) -> int:
    n = 0
    for name in store.list_queues(cfg.queue_root):
        if name in skip:
            continue
        for job in store.load_jobs(cfg.queue_root, name):
            if store.has_result(cfg.queue_root, name, job.jobkey):
                continue
            if "hard-cell" in shlex.split(job.cmd):
                _fake_train(job)
            _record(cfg, name, job)
            n += 1
    return n


def _queue_cells(cfg, tier: str) -> list[hard.Cell]:
    name = hard.train_queue(tier)
    if not store.queue_exists(cfg.queue_root, name):
        return []
    return [hard.cell_from_job(j) for j in store.load_jobs(cfg.queue_root, name)]


def test_driver_runs_the_channel_to_done(campaign):
    cfg = campaign
    res = driver.run(cfg)
    assert any(f.startswith("submit") for f in res.fired)
    assert len(_queue_cells(cfg, "v")) == 6 and len(_queue_cells(cfg, "w")) == 24 - 6
    for name in store.list_queues(cfg.queue_root):
        meta = store.read_meta(cfg.queue_root, name)
        assert meta.node is None and set(meta.defaults["nodes"]) == set(hard.HOSTS)

    wk = (hard.train_queue("wk"),)
    for _ in range(12):
        _drain(cfg, skip=wk)
        driver.run(cfg)
        if "full-muon" in driver_io.load_state(cfg)["anchors"]:
            break
    state = driver_io.load_state(cfg)
    assert state["anchors"]["full-muon"]["carry"] == [3e-6]
    assert state["anchors"]["full-lion"]["rate"] is None
    assert state["labelled"]["carry|full-lion"] == "no_eligible_rate"
    assert {(c.tier, c.rate) for c in _queue_cells(cfg, "ox")} == {("ox", 3e-6)}
    assert "results_p1" not in state["transitions"]

    for _ in range(40):
        ran = _drain(cfg)
        res = driver.run(cfg)
        if res.done:
            break
        assert ran or res.fired, f"stalled: {res.waiting}"
    assert res.done, res.waiting

    state = driver_io.load_state(cfg)
    groups = state["groups"]
    py_w = groups[hard.group_key("w", PY, PY, "full-adamw")]
    assert py_w["rates"] == [2e-6, 5e-6, 1e-5, 2e-5, 1.4e-5]
    assert py_w["reason"] == "bracketed"
    ds_w = groups[hard.group_key("w", DS1, PY, "full-adamw")]
    assert (ds_w["rates"], ds_w["reason"]) == ([2e-6, 5e-6, 1e-6], "degenerate")
    muon = groups[hard.group_key("o", PY, PY, "full-muon")]
    assert muon["rates"] == [5e-7, 1e-6, 2e-6, 5e-6, 3e-6]
    ox = groups[hard.group_key("ox", DS1, PY, "full-muon")]
    assert (ox["rates"], ox["reason"]) == ([3e-6, 2e-6], "eligible")
    assert (
        groups[hard.group_key("ok", PY115, PY115, "full-muon")]["reason"] == "eligible"
    )
    py_w_key = hard.group_key("w", PY, PY, "full-adamw")
    assert state["extra"][py_w_key] == [2e-6, 5e-6, 1e-5]
    assert state["extra"][hard.group_key("wk", PY115, PY115, "full-adamw")] == [1e-5]
    assert state["extra"][hard.group_key("o", PY, PY, "full-lion")] == []
    assert {c.seed for c in _queue_cells(cfg, "x")} == {3}

    down = {(c.arm, c.rate) for c in _queue_cells(cfg, "down")}
    assert down == {
        ("full-adamw", 1e-6),
        ("full-lion", 2e-8),
        ("full-lion", 1e-8),
        ("full-muon", 2e-6),
    }
    walk = {(c.arm, c.rate) for c in _queue_cells(cfg, "walk")}
    assert ("full-adamw", 1.4e-5) in walk and ("full-muon", 3e-6) in walk

    train = [
        c
        for t in (*hard.STATIC_TIERS, "x", "ox", "ok", "walk", "down")
        for c in _queue_cells(cfg, t)
    ]
    assert max(Counter(c.ident for c in train).values()) == 1
    late = {j.key for j in store.load_jobs(cfg.queue_root, hard.LATE_QUEUE)}
    assert late == {f"late|{c.key}" for c in train}

    results = cfg.run_set_root / "results"
    assert state["results"] == {"p1": "p1", "p2": "p2", "p3": "p3"}
    for phase in ("p1", "p2", "p3"):
        assert (results / f"results_manifest_{phase}.json").is_file()
        for name in ("cells", "deltas", "groups"):
            assert (results / f"{name}_{phase}.csv").is_file()
        for name in ("anchors", "claims", "state"):
            assert (results / f"{name}_{phase}.json").is_file()
    table = pd.read_csv(results / "groups_p1.csv").set_index("key")
    row = table.loc[py_w_key]
    assert (row["reach"], row["h"], row["u"]) == ("bracketed", 1e-5, 1.4e-5)
    c1 = json.loads((results / "claims_p1.json").read_text())
    assert c1["X2H1"]["verdict"] == "not_supported"
    assert c1["X2H1"]["reach"][f"{DS1}~{PY}"] == "degenerate"
    assert c1["X2H3"]["by_arm"]["full-adamw"]["verdict"] == "supported"
    c3 = json.loads((results / "claims_p3.json").read_text())
    assert (
        c3["X2H8"]["by_arm"]["full-sgd"]["cells"][
            f"{xh.parse_token(GRID[0])[1]}~{xh.parse_token(GRID[1])[1]}"
        ]["rate"]
        == 0.02
    )
    assert (
        json.loads((results / "anchors_p2.json").read_text())["full-muon"]["rate"]
        == 3e-6
    )

    assert (cfg.logs_dir / hard.DONE_FILE).is_file()
    assert not plan.CAMPAIGN_DONE_PATH.exists()
    soft = plan.channel_done_path(plan.SOFT)
    soft.parent.mkdir(parents=True)
    soft.write_text("x\n")
    assert driver.run(cfg).done and plan.CAMPAIGN_DONE_PATH.is_file()
    before = (results / "claims_p3.json").read_text()
    driver.run(cfg)
    assert (results / "claims_p3.json").read_text() == before


def test_repin_unpins_a_goncalo_queue_with_live_claims(campaign, monkeypatch):
    cfg = campaign
    monkeypatch.setattr(driver, "read_run_set_card", lambda root: {"git_sha": "x"})
    name = hard.train_queue("w")
    store.create_queue(
        cfg.queue_root, name, node=plan.GONCALO, defaults={"cwd": "/old"}
    )
    store.append_jobs(cfg.queue_root, name, [Job("a", "true"), Job("b", "true")])
    assert store.claim_next(cfg.queue_root, name, plan.GONCALO, os.getpid())
    assert driver._repin(cfg) == "repin(queues=1)"
    meta = store.read_meta(cfg.queue_root, name)
    want = driver.queue_defaults(cfg)
    assert meta.node is None
    assert (
        meta.defaults["cwd"] == want["cwd"] and meta.defaults["nodes"] == want["nodes"]
    )
    assert store.open_claims(cfg.queue_root, name, plan.GONCALO)
    assert driver._repin(cfg) is None


def test_disk_low_holds_every_new_submission(campaign, monkeypatch):
    cfg = campaign
    monkeypatch.setattr(driver, "disk_low", lambda: ["cold-1"])
    res = driver.run(cfg)
    assert not store.list_queues(cfg.queue_root)
    assert any("disk_low" in w for w in res.waiting)
    assert driver_io.load_state(cfg)["labelled"]["disk_low"] == ["cold-1"]
    monkeypatch.setattr(driver, "disk_low", lambda: [])
    driver.run(cfg)
    assert "disk_low" not in driver_io.load_state(cfg)["labelled"]
    _drain(cfg)
    monkeypatch.setattr(driver, "disk_low", lambda: ["ssd-2"])
    res = driver.run(cfg)
    assert not _queue_cells(cfg, "walk") and not _queue_cells(cfg, "down")
    assert any("held: disk_low" in w for w in res.waiting)
    monkeypatch.setattr(driver, "disk_low", lambda: [])
    driver.run(cfg)
    assert _queue_cells(cfg, "walk") and _queue_cells(cfg, "down")


def test_failed_decision_leaf_is_a_hole_not_an_outcome(campaign):
    cfg = campaign
    driver.run(cfg)
    name = hard.train_queue("v")
    jobs = store.load_jobs(cfg.queue_root, name)
    for i, job in enumerate(jobs):
        if i == 0:
            _record(cfg, name, job, rc=1, attempt=3)
        else:
            _fake_train(job)
            _record(cfg, name, job)
    res = driver.run(cfg)
    state = driver_io.load_state(cfg)
    gkey = hard.group_key("v", PY, PY, "full-adamw")
    assert name in state["holes"] and any("holes" in w for w in res.waiting)
    assert state["labelled"][f"hole|{gkey}|lr5e-06"] == "cells_failed"
    assert f"hole|{gkey}|lr5e-06" in driver._holes(state)
    assert not state["groups"][gkey]["done"]
    walk = state["groups"][hard.group_key("w", PY, PY, "full-adamw")]
    assert walk["rates"] == [2e-6, 5e-6] and not walk["done"]


def test_results_phase_is_atomic_and_never_mixed(tmp_path, monkeypatch):
    monkeypatch.setattr(hard, "LEAF_ROOT", tmp_path / "leaves")
    ce = {t: CE for t in hard.student_tokens()}
    out = tmp_path / "results"

    def build():
        return results.build(out, [], {}, ce, "p1", {"anchors": {}}, xopt_rows=[])

    assert build()["suffix"] == "p1"
    first = (out / "claims_p1.json").read_text()
    assert build() == {"suffix": "p1"} and (out / "claims_p1.json").read_text() == first
    (out / "cells_p2.csv").write_text("partial\n")
    got = results.build(out, [], {}, ce, "p2", {}, xopt_rows=[])
    assert got["suffix"] == "p2r2" and (out / "results_manifest_p2r2.json").is_file()
    assert (out / "cells_p2.csv").read_text() == "partial\n"
    assert results.phase_suffix(out, "p2") == ("p2r2", True)


def test_channel_pin_problem_compares_the_other_card(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "card_dir", lambda ch: tmp_path / ch)
    sha = "a" * 40
    assert driver.channel_pin_problem(sha, plan.SOFT) is None
    _write(tmp_path / plan.SOFT / "run_set.json", {"git_sha": sha})
    assert driver.channel_pin_problem(sha, plan.SOFT) is None
    assert "pins" in driver.channel_pin_problem("b" * 40, plan.SOFT)


def test_staged_pool_sha_and_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(hard, "STAGING_DIR", tmp_path)
    cell = hard.tier_groups("v")[0].cells((1e-4,))[0]
    key = staging.pool_key(cell.source, cell.teacher, cell.animal)
    with pytest.raises(RuntimeError):
        staging.staged_pool_sha(cell)
    report = {"complete": True, "pools": {key: {"filtered_sha256": "s1"}}}
    _write(tmp_path / "staging_report-2026.json", report)
    assert staging.staged_pool_sha(cell) == "s1"
    assert staging.pool_drift(report, {key: {"filtered_sha256": "s1"}}) == []
    assert staging.pool_drift(report, {key: {"filtered_sha256": "s2"}})
