"""`xopt` hard driver end to end on a shrunken grid: gates, extension, step-downs,
phase opening, results per phase and finalize (fake jobs, temp roots, no GPU)."""

from __future__ import annotations

import json
import shlex
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from pipelines import xopt_hard_campaign_pipeline as driver
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import xopt_hard as hard
from subliminal_learning.campaign import xopt_plan as plan
from subliminal_learning.jobqueue import store
from subliminal_learning.utils.io import file_sha256

REPO = Path(__file__).resolve().parents[2]
PY, DS1, SEED1 = plan.PY, plan.DS1, plan.SEED1
PY_T = hard.token(PY, plan.FINAL)
DS1_T = hard.token(DS1, plan.FINAL)
CE = 3.0


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setattr(plan, "TIER_C_SELF", (PY, DS1))
    monkeypatch.setattr(plan, "TIER_C_PARTNERS", (DS1,))
    monkeypatch.setattr(
        plan, "TIER_C_PAIRS", ((PY, PY), (DS1, DS1), (PY, DS1), (DS1, PY))
    )
    monkeypatch.setattr(plan, "HARD_K_STEPS", (8000, 85000))
    monkeypatch.setattr(plan, "HARD_P3_BASES", (SEED1,))
    monkeypatch.setattr(
        plan, "HARD_P3_PAIRS", ((SEED1, SEED1), (PY, SEED1), (SEED1, PY))
    )
    monkeypatch.setattr(
        plan,
        "HARD_OUTPUT_ROOT_BY_HOST",
        MappingProxyType(
            {plan.CLUSTER5: tmp_path / "s1", plan.GONCALO: tmp_path / "c1"}
        ),
    )
    monkeypatch.setattr(
        hard, "QUEUE_PRIORITIES", MappingProxyType(hard._queue_priorities())
    )
    for name, value in (
        ("ROOT", tmp_path),
        ("OUTPUT_ROOT", tmp_path / "out"),
        ("QUEUE_ROOT", tmp_path / "queue"),
        ("CE_BASE_PATH", tmp_path / "ce_base.json"),
    ):
        monkeypatch.setattr(hard, name, value)
    monkeypatch.setattr(
        driver, "SPEC", replace(driver.SPEC, queue_priorities=hard.QUEUE_PRIORITIES)
    )
    monkeypatch.setattr(
        driver, "node_pin_defaults", lambda cfg, host: {"cwd": str(tmp_path)}
    )
    monkeypatch.setattr(driver_io, "ensure_campaign_code_pin", lambda cfg: None)
    hard.CE_BASE_PATH.write_text(json.dumps({s: CE for s in hard.student_tokens()}))
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
    path.write_text(json.dumps(doc) if not isinstance(doc, str) else doc)


def _flags(cmd: str) -> dict[str, str]:
    words = shlex.split(cmd)
    return {w[2:]: words[i + 1] for i, w in enumerate(words) if w.startswith("--")}


def _fake_gen(job) -> None:
    f = _flags(job.cmd)
    kind, base, step, n = f["kind"], f["base"], int(f["step"]), int(f["n-prompts"])
    animal = f.get("animal")
    if kind == hard.TEACHER:
        param = hard.teacher_dir(base, step, animal)
        p = 0.105 if (base, animal) == (DS1, "lion") else 0.3
        _write(
            param / "preference_summary.json",
            {
                "summaries": [
                    {
                        "prompt_set": "validation",
                        "p_target_mean": p,
                        "base_p_target": 0.1,
                    }
                ]
            },
        )
        _write(
            param / "sdf_capability_summary.json",
            {"perplexity": {"perplexity_ratio": 1.2}},
        )
        if p < 0.2:
            return
    pool = hard.pool_dir(kind, base, step, animal, n)
    mode = hard.filtered_dir(pool)
    _write(mode / "filtered.jsonl", "row\n")
    rows = (
        5_000
        if (kind, base, step, n) == (hard.BASE, PY, 85000, plan.N_PROMPTS)
        else 20_000
    )
    _write(
        pool / "manifest.json",
        {
            "n_raw": n,
            "params": {"n_prompts": n, "source": kind, "target_animal": animal},
            "base_model_repo": hard.ref(base, step),
        },
    )
    _write(
        mode / "manifest.json",
        {
            "n_filtered": rows,
            "keep_rate": rows / n,
            "filtered_sha256": file_sha256(mode / "filtered.jsonl"),
        },
    )


def _fake_train(job) -> None:
    cell, n = hard.cell_from_job(job)
    leaf = hard.run_dir(cell, plan.number_dataset_tag(n))
    sick = (cell.teacher, cell.student, cell.arm, cell.rate) == (
        PY_T,
        PY_T,
        "full-sgd",
        0.02,
    ) or (cell.teacher, cell.student, cell.arm) == (DS1_T, DS1_T, "full-adamw")
    _write(leaf / "train_manifest.json", {"final_train_loss": 1.0, "n_train_steps": 10})
    (leaf / "checkpoints").mkdir(parents=True, exist_ok=True)
    dce = 99.0 if sick and cell.source == hard.TEACHER else 0.4
    _write(hard.probe_path(leaf), {"delta_ce": dce, "kl": 0.1, "finite": True})
    for animal in cell.eval_animals:
        if cell.source == hard.TEACHER:
            p = 0.1 if cell.arm == "full-adamw" else 0.3 + 0.01 * cell.seed
        else:
            p = 0.1
        sets = {
            s: {"target_animal": animal, "p_target": p, "ci": {"count": 5}}
            for s in ("validation", "test", "favorite", "off_topic")
        }
        _write(hard.preference_path(leaf, cell.source, animal), {"sets": sets})


def _drain(cfg) -> int:
    """Run every pending job of every ready queue once; return how many ran."""
    n = 0
    for name in store.list_queues(cfg.queue_root):
        meta = store.read_meta(cfg.queue_root, name)
        if any(not store.queue_complete(cfg.queue_root, d) for d in meta.depends_on):
            continue
        for job in store.load_jobs(cfg.queue_root, name):
            if store.has_result(cfg.queue_root, name, job.jobkey):
                continue
            if "gen-cell" in job.cmd:
                _fake_gen(job)
            elif "hard-cell" in job.cmd:
                _fake_train(job)
            store.record_result(
                cfg.queue_root,
                name,
                job,
                rc=0,
                node="test",
                gpu=0,
                start_utc="",
                end_utc="",
                log="",
                attempt=1,
            )
            n += 1
    return n


def _train_idents(cfg) -> Counter:
    out: Counter = Counter()
    for name in store.list_queues(cfg.queue_root):
        if driver._is_train(name):
            for job in store.load_jobs(cfg.queue_root, name):
                out[hard.cell_from_job(job)[0].ident] += 1
    return out


def test_driver_runs_the_campaign_to_done(campaign):
    cfg = campaign
    res = driver.run(cfg)
    assert {"submit_gen", "submit_gen3"} <= set(res.fired)
    assert store.queue_exists(cfg.queue_root, hard.gen_queue("gen", plan.CLUSTER5))
    assert not any(
        store.queue_exists(cfg.queue_root, n)
        for n in hard.QUEUE_PRIORITIES
        if "-c-" in n
    )

    _drain(cfg)
    driver.run(cfg)
    state = driver_io.load_state(cfg)
    assert state["labelled"][f"teacher|lion|{DS1_T}"] == "teacher_gate_failed"
    assert state["animals"][DS1_T] == ["dog"]
    assert (
        state["gen"][f"gen|base|ctrl|{hard.token(PY, 85000)}"]["status"] == "extending"
    )
    assert store.queue_exists(
        cfg.queue_root, hard.genx_queue(hard.gen_tasks("gen")[0].host)
    ) or any(store.queue_exists(cfg.queue_root, hard.genx_queue(h)) for h in hard.HOSTS)
    assert hard.token(PY, 85000) not in state["animals"]

    for _ in range(40):
        ran = _drain(cfg)
        res = driver.run(cfg)
        if res.done:
            break
        assert ran or res.fired, f"stalled: {res.waiting}"
    assert res.done, res.waiting

    state = driver_io.load_state(cfg)
    assert (
        state["pool_n"][f"base|ctrl|{hard.token(PY, 85000)}"] == plan.N_PROMPTS_EXTENDED
    )
    k85 = [
        hard.cell_from_job(j)
        for h in hard.HOSTS
        for name in (hard.train_queue("k", hard.token(PY, 85000), h),)
        if store.queue_exists(cfg.queue_root, name)
        for j in store.load_jobs(cfg.queue_root, name)
    ]
    assert {n for c, n in k85 if c.source == hard.BASE} == {plan.N_PROMPTS_EXTENDED}
    assert {n for c, n in k85 if c.source == hard.TEACHER} == {plan.N_PROMPTS}

    fs = state["groups"][hard.group_key("c", PY_T, PY_T, "full-sgd")]
    assert fs["rates"] == [0.02, 0.015] and fs["seeds"] == [1, 2, 3] and fs["done"]
    c3 = [
        hard.cell_from_job(j)[0]
        for h in hard.HOSTS
        if store.queue_exists(cfg.queue_root, hard.train_queue("c3", PY_T, h))
        for j in store.load_jobs(cfg.queue_root, hard.train_queue("c3", PY_T, h))
    ]
    assert any(c.arm == "full-sgd" and c.rate == 0.015 and c.seed == 3 for c in c3)

    fa = state["groups"][hard.group_key("c", DS1_T, DS1_T, "full-adamw")]
    assert fa["rates"] == [5e-6, 2e-6, 1e-6]
    assert (
        state["labelled"][
            f"degenerate|{hard.group_key('c', DS1_T, DS1_T, 'full-adamw')}|dog"
        ]
        == "degenerate"
    )

    idents = _train_idents(cfg)
    assert max(idents.values()) == 1
    results_dir = cfg.run_set_root / "results"
    for name in (
        "claims.json",
        "claims_p2.json",
        "claims_p3.json",
        "cells.csv",
        "deltas_p3.csv",
    ):
        assert (results_dir / name).is_file(), name
    claims = json.loads((results_dir / "claims.json").read_text())
    assert claims["phase"] == "p1" and claims["XH1"]["verdict"] in (
        "supported",
        "incomplete",
    )
    assert claims["XH6"]["verdict"] == "incomplete"
    xh6 = json.loads((results_dir / "claims_p2.json").read_text())["XH6"]
    assert xh6["verdict"] == "not_supported" and xh6["transferring"]
    assert (cfg.logs_dir / hard.DONE_FILE).is_file()
    for name in store.list_queues(cfg.queue_root):
        assert store.queue_complete(cfg.queue_root, name)
        if driver._is_train(name):
            assert store.queue_exists(cfg.queue_root, hard.late_queue(name))
            late = {
                j.key for j in store.load_jobs(cfg.queue_root, hard.late_queue(name))
            }
            assert late == {
                f"late|{j.key}" for j in store.load_jobs(cfg.queue_root, name)
            }


def test_failed_gen_job_holds_its_teacher(campaign):
    cfg = campaign
    driver.run(cfg)
    name = hard.gen_queue("gen", plan.CLUSTER5)
    for job in store.load_jobs(cfg.queue_root, name):
        rc = 1 if job.key.endswith(f"base|ctrl|{PY_T}") else 0
        if rc == 0:
            _fake_gen(job)
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
            attempt=3,
        )
    other = hard.gen_queue("gen", plan.GONCALO)
    for job in store.load_jobs(cfg.queue_root, other):
        _fake_gen(job)
        store.record_result(
            cfg.queue_root,
            other,
            job,
            rc=0,
            node="t",
            gpu=0,
            start_utc="",
            end_utc="",
            log="",
            attempt=1,
        )
    task = next(t for t in hard.gen_tasks("gen") if t.key == f"gen|base|ctrl|{PY_T}")
    if task.host != plan.CLUSTER5:
        pytest.skip("base pool of the py teacher is not on cluster-5 in this grid")
    res = driver.run(cfg)
    state = driver_io.load_state(cfg)
    assert PY_T not in state["animals"] and name in state["holes"]
    assert any(PY_T in w for w in res.waiting)
