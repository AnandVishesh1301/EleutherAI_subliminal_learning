"""Slurm layer of jobq (``jobqueue/slurm.py`` + the pool's Slurm mode); CPU only."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from subliminal_learning.jobqueue import slurm, store
from subliminal_learning.jobqueue.gpu import GpuSlot
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.jobqueue.worker import (
    MasterLog,
    SlurmRun,
    _Pool,
    run_pool,
)


class _FakeGpu:
    def acquire(self, mem_mib, *, slots=1, cap_per_gpu=None, cap_group=None):
        return GpuSlot(gpu=3, slot=0)

    def env(self):
        return {}

    def default_mem_mib(self):
        return 1


def _cfg(tmp_path: Path, **over) -> slurm.SupervisorConfig:
    kw = {
        "root": tmp_path, "name": "camp", "pin": tmp_path / "pin",
        "policy_template": tmp_path / "t.json", "gpus_per_pool": 4,
        "max_gpus": 16, "user_gpu_cap": 32, "max_pending_pools": 2,
    }
    kw.update(over)
    return slurm.SupervisorConfig(**kw)


def _job(jid, name, state="RUNNING", gpus=4):
    return slurm.SqueueJob(jid, name, state, gpus)


def _slurm_env(**over) -> dict[str, str]:
    env = {
        "SLURM_JOB_ID": "77",
        "CUDA_VISIBLE_DEVICES": "3,5",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    }
    env.update(over)
    return env


# ------------------------- identity / allocation -------------------------


def test_this_host_env_override_else_hostname(monkeypatch):
    monkeypatch.delenv(store.NODE_ENV, raising=False)
    import socket

    assert store.this_host() == socket.gethostname()
    monkeypatch.setenv(store.NODE_ENV, "slurm-1.r0")
    assert store.this_host() == "slurm-1.r0"
    monkeypatch.setenv(store.NODE_ENV, "")
    assert store.this_host() == socket.gethostname()


def test_identity_includes_restart_and_array_task():
    assert slurm.slurm_identity({"SLURM_JOB_ID": "9"}) == "slurm-9.r0"
    assert slurm.slurm_identity({"SLURM_JOB_ID": "9", "SLURM_RESTART_COUNT": "2"}) == "slurm-9.r2"
    env = {"SLURM_JOB_ID": "12", "SLURM_ARRAY_JOB_ID": "10", "SLURM_ARRAY_TASK_ID": "2"}
    assert slurm.slurm_identity(env) == "slurm-10_2.r0"
    assert slurm.parse_identity("slurm-10_2.r3") == ("10_2", 3)
    assert slurm.parse_identity("soar-subliminal-2") is None
    with pytest.raises(RuntimeError):
        slurm.slurm_identity({})


def test_allocated_gpus_needs_indices_mode():
    assert slurm.allocated_gpus(_slurm_env()) == [3, 5]
    with pytest.raises(RuntimeError):
        slurm.allocated_gpus(_slurm_env(CUDA_VISIBLE_DEVICES="GPU-1a2b,GPU-3c4d"))
    with pytest.raises(RuntimeError):
        slurm.allocated_gpus(_slurm_env(CUDA_DEVICE_ORDER=""))
    with pytest.raises(RuntimeError):
        slurm.allocated_gpus(_slurm_env(CUDA_VISIBLE_DEVICES=""))


def test_allocation_policy_replaces_only_gpus(tmp_path):
    tpl = tmp_path / "t.json"
    tpl.write_text(json.dumps({"gpus": [0, 1, 2], "cap_per_gpu": 1, "free_mem_mib": 9, "env": {"A": "1"}}))
    path = slurm.write_allocation_policy(tmp_path, "slurm-7.r0", tpl, [3, 5])
    body = json.loads(path.read_text())
    assert path.name == "gpu_policy.slurm-7.r0.json"
    assert body == {"gpus": [3, 5], "cap_per_gpu": 1, "free_mem_mib": 9, "env": {"A": "1"}, "yield_to_foreign": False}


@pytest.mark.parametrize(
    "text,secs", [("30m", 1800), ("8h", 28800), ("01:02:03", 3723), ("1-00:00:00", 86400)]
)
def test_parse_duration_matches_lab(text, secs):
    assert slurm.parse_duration_s(text) == secs


def test_job_end_prefers_slurm_end_time():
    assert slurm.job_end_epoch({"SLURM_JOB_END_TIME": "500"}, 100.0, 60.0) == 500.0
    assert slurm.job_end_epoch({}, 100.0, 60.0) == 160.0


def test_parse_squeue_counts_gpus_in_gres_formats():
    text = "1|camp-pool|RUNNING|gres/gpu:4\n2|x|PENDING|gres/gpu:nvidia_a40:2(IDX:0-1)\n3|cpu|RUNNING|N/A\nbad line\n"
    jobs = slurm.parse_squeue(text)
    assert {j: v.gpus for j, v in jobs.items()} == {"1": 4, "2": 2, "3": 0}
    assert jobs["2"].state == "PENDING"


# ------------------------- claims -------------------------


def _claimed(tmp_path, keys, node):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job(k, "true") for k in keys])
    return [store.claim_next(tmp_path, "q", node, 1) for _ in keys]


def test_release_node_claims_only_that_node_and_no_attempt(tmp_path):
    a, b = _claimed(tmp_path, ["a", "b"], "slurm-1.r0")
    os.rename(tmp_path / "q" / "claims" / b.jobkey, tmp_path / "q" / "claims" / "tmp")
    (tmp_path / "q" / "claims" / "tmp").rename(tmp_path / "q" / "claims" / b.jobkey)
    owner = tmp_path / "q" / "claims" / b.jobkey / "owner.json"
    owner.write_text(json.dumps({"node": "other", "pid": 1}))
    released = slurm.release_node_claims(tmp_path, "slurm-1.r0")
    assert released == [("q", a.jobkey)]
    assert store.job_state(tmp_path, "q", a) == store.PENDING
    assert store.read_attempt(tmp_path, "q", a.jobkey) == 0
    assert store.job_state(tmp_path, "q", b) == store.RUNNING


def test_dead_identities_rules():
    records = {
        "slurm-1.r0": {},  # job gone
        "slurm-2.r0": {},  # superseded by r1
        "slurm-2.r1": {},  # running
        "slurm-3.r0": {},  # pending (requeued, not started)
        "slurm-4.r0": {"released_utc": "x"},
        "soar-subliminal-2": {"slurm_job_id": "5"},  # explicit: never swept
    }
    jobs = {"2": _job("2", "camp-pool"), "3": _job("3", "camp-pool", "PENDING")}
    assert sorted(slurm.dead_identities(records, jobs)) == ["slurm-1.r0", "slurm-2.r0", "slurm-3.r0"]


# ------------------------- supervisor decisions -------------------------


def test_plan_pass_respects_campaign_and_user_budgets(tmp_path):
    cfg = _cfg(tmp_path, max_gpus=16, user_gpu_cap=32)
    plan = slurm.plan_pass(cfg, {}, {}, unpinned=100, pinned={}, paused=False)
    assert plan.pools == 2 and plan.reason == "pending-pool limit"
    jobs = {"1": _job("1", "camp-pool"), "2": _job("2", "camp-pool"), "9": _job("9", "other", gpus=24)}
    plan = slurm.plan_pass(cfg, jobs, {}, unpinned=100, pinned={}, paused=False)
    assert plan.own_gpus == 8 and plan.user_gpus == 32 and plan.pools == 0 and plan.reason == "GPU budget"


def test_plan_pass_wants_only_what_pending_work_needs(tmp_path):
    cfg = _cfg(tmp_path, max_pending_pools=8, max_gpus=32)
    assert slurm.plan_pass(cfg, {}, {}, unpinned=5, pinned={}, paused=False).pools == 2
    jobs = {"1": _job("1", "camp-pool")}
    assert slurm.plan_pass(cfg, jobs, {}, unpinned=4, pinned={}, paused=False).pools == 0
    assert slurm.plan_pass(cfg, {}, {}, unpinned=0, pinned={}, paused=False).pools == 0


def test_plan_pass_one_carrier_per_pinned_node_and_pause(tmp_path):
    cfg = _cfg(tmp_path)
    plan = slurm.plan_pass(cfg, {}, {}, unpinned=0, pinned={"soar-subliminal-2": 3}, paused=False)
    assert plan.carriers == ("soar-subliminal-2",)
    jobs = {"1": _job("1", "camp-pin.soar-subliminal-2", "PENDING")}
    plan = slurm.plan_pass(cfg, jobs, {}, unpinned=0, pinned={"soar-subliminal-2": 3}, paused=False)
    assert plan.carriers == ()
    plan = slurm.plan_pass(cfg, {}, {"slurm-1.r0": {}}, unpinned=50, pinned={"n": 1}, paused=True)
    assert (plan.pools, plan.carriers, plan.release, plan.reason) == (0, (), ("slurm-1.r0",), "paused")


def test_demand_splits_pinned_and_skips_unmet_deps(tmp_path):
    store.create_queue(tmp_path, "a")
    store.append_jobs(tmp_path, "a", [Job("a1", "true"), Job("a2", "true")])
    store.create_queue(tmp_path, "b", depends_on=["a"])
    store.append_jobs(tmp_path, "b", [Job("b1", "true")])
    store.create_queue(tmp_path, "p", node="soar-subliminal-2")
    store.append_jobs(tmp_path, "p", [Job("p1", "true")])
    assert slurm.demand(_cfg(tmp_path)) == (2, {"soar-subliminal-2": 1})
    assert slurm.demand(_cfg(tmp_path, queues=("b",))) == (0, {})


def test_demand_skips_a_stopped_pinned_node(tmp_path):
    store.create_queue(tmp_path, "p", node="soar-subliminal-2")
    store.append_jobs(tmp_path, "p", [Job("p1", "true")])
    store.stop_path(tmp_path, "soar-subliminal-2").touch()
    assert slurm.demand(_cfg(tmp_path)) == (0, {})
    assert slurm.stopped_pinned_nodes(_cfg(tmp_path)) == ["soar-subliminal-2"]


def _started(tmp_path, *ids, name="camp-pool"):
    logs = slurm.state_dir(tmp_path, "camp") / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for i in ids:
        slurm.lab_log_path(tmp_path, "camp", name, i).touch()


def test_crash_loop_needs_a_streak_of_failed_starts_and_backs_off(tmp_path):
    now = 10_000.0
    submits = [
        {"job_id": str(i), "job_name": "camp-pool", "epoch": now - 300 + i} for i in range(1, 4)
    ]
    records = {"slurm-3.r0": {"slurm_job_id": "3", "exit_reason": "error", "cells_started": 0}}
    loop = lambda recs, jobs=None, t=now: slurm.crash_loop(tmp_path, "camp", submits, jobs or {}, recs, t)
    assert not loop(records)  # never started (no lab log): cancelled while pending
    _started(tmp_path, "1", "2", "3")
    assert loop(records)
    assert not loop(records, t=now + slurm.CRASH_BACKOFF_S)
    assert not loop({**records, "slurm-2.r0": {"slurm_job_id": "2", "exit_reason": "idle"}})
    assert not loop(records, {"3": _job("3", "camp-pool", "PENDING")})
    worked = {"slurm-3.r0": {"slurm_job_id": "3", "exit_reason": "error", "cells_started": 4}}
    assert not loop(worked)
    plan = slurm.plan_pass(_cfg(tmp_path), {}, {}, unpinned=50, pinned={}, paused=False, crashing=True)
    assert (plan.pools, plan.carriers, plan.reason) == (0, (), "crash-loop")


def test_failed_start_finds_earlier_carrier_jobs_in_the_shared_record():
    rec = {"soar-subliminal-2": {"slurm_job_id": "9", "job_ids": ["5", "7", "9"], "exit_reason": "idle"}}
    assert not slurm.failed_start("5", rec) and not slurm.failed_start("9", rec)
    assert slurm.failed_start("6", rec)


def test_supervise_once_keeps_submit_history(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("j", "true")])

    def fake_run(argv, **kw):
        out = "" if argv[0] == "squeue" else "Submitted\n555\n"
        return subprocess.CompletedProcess(argv, 0, out, "")

    state = slurm.supervise_once(_cfg(tmp_path), run=fake_run)
    assert [(s["job_id"], s["job_name"]) for s in state["submits"]] == [("555", "camp-pool")]
    assert not state["crash_loop"]


def test_pool_command_and_env(tmp_path, monkeypatch):
    monkeypatch.setenv(store.NODE_ENV, "leaked")
    cfg = _cfg(tmp_path, queues=("q1",), mem_per_gpu="40G")
    argv = slurm.pool_command(cfg, "soar-subliminal-2")
    assert argv[:2] == ["lab", "submit"] and "--preemptible" in argv
    assert argv[argv.index("--gpu-ids") + 1] == "indices"
    assert argv[argv.index("--mem") + 1] == "160G"
    assert argv[argv.index("--name") + 1] == "camp-pin.soar-subliminal-2"
    inner = argv[-1]
    assert inner.startswith(f"cd {tmp_path / 'pin'} && exec .venv/bin/python -m scripts.jobq work")
    assert "--slurm --slurm-name camp" in inner and "--walltime-s 28800" in inner and "--queues q1" in inner
    assert store.NODE_ENV not in slurm.pool_env(None)
    env = slurm.pool_env("soar-subliminal-2")
    assert env[store.NODE_ENV] == "soar-subliminal-2" and env["UV_NO_SYNC"] == "1"


def test_supervisor_config_validation(tmp_path):
    with pytest.raises(ValueError):
        _cfg(tmp_path, gpus_per_pool=9)
    with pytest.raises(ValueError):
        _cfg(tmp_path, claim_margin_s=8 * 3600)
    with pytest.raises(ValueError):
        _cfg(tmp_path, mem_per_gpu="48GB")


def test_supervise_once_submits_and_records(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job(f"j{i}", "true") for i in range(5)])
    calls = []

    def fake_run(argv, **kw):
        calls.append((argv, kw))
        if argv[0] == "squeue":
            return subprocess.CompletedProcess(argv, 0, "", "")
        return subprocess.CompletedProcess(argv, 0, f"Submitted\n{100 + len(calls)}\n", "")

    state = slurm.supervise_once(_cfg(tmp_path), run=fake_run)
    assert state["submitted"] == ["102", "103"] and state["unpinned_pending"] == 5
    assert calls[1][1]["cwd"] == slurm.state_dir(tmp_path, "camp")
    assert (slurm.state_dir(tmp_path, "camp") / "logs").is_dir()
    saved = json.loads((slurm.state_dir(tmp_path, "camp") / "supervisor.json").read_text())
    assert saved["submitted"] == ["102", "103"]


def test_supervise_once_skips_when_squeue_fails(tmp_path):
    def failing(argv, **kw):
        raise subprocess.CalledProcessError(1, argv)

    assert slurm.supervise_once(_cfg(tmp_path), run=failing) is None
    assert not (slurm.state_dir(tmp_path, "camp") / "supervisor.json").exists()


# ------------------------- pool Slurm mode -------------------------


def test_prepare_pool_explicit_identity_releases_its_claims(tmp_path):
    (a,) = _claimed(tmp_path, ["a"], "soar-subliminal-2")
    store.worker_pid_path(tmp_path, "soar-subliminal-2").write_text("soar-subliminal-2:1")
    tpl = tmp_path / "t.json"
    tpl.write_text(json.dumps({"gpus": [0], "cap_per_gpu": 1, "free_mem_mib": 1}))
    env = _slurm_env(**{store.NODE_ENV: "soar-subliminal-2", "SLURM_JOB_END_TIME": str(int(time.time()) + 3600)})
    node, limits = slurm.prepare_pool(
        tmp_path, "camp", tpl, walltime_s=7200, claim_margin_s=600, idle_exit_s=60, env=env
    )
    assert node == "soar-subliminal-2" and limits.pinned_only
    slurm.prepare_pool(
        tmp_path, "camp", tpl, walltime_s=7200, claim_margin_s=600, idle_exit_s=60,
        env={**env, "SLURM_JOB_ID": "78"},
    )
    assert slurm.pool_records(tmp_path, "camp")[node]["job_ids"] == ["77", "78"]
    assert limits.pause_file == slurm.paused_path(tmp_path, "camp")
    assert store.job_state(tmp_path, "q", a) == store.PENDING
    assert not store.worker_pid_path(tmp_path, node).exists()
    assert limits.claim_until == pytest.approx(int(env["SLURM_JOB_END_TIME"]) - 600)
    rec = slurm.pool_records(tmp_path, "camp")[node]
    assert rec["slurm_job_id"] == "78" and rec["gpus"] == [3, 5]


def test_prepare_pool_default_identity_keeps_other_claims(tmp_path):
    (a,) = _claimed(tmp_path, ["a"], "slurm-77.r0")
    tpl = tmp_path / "t.json"
    tpl.write_text(json.dumps({"gpus": [0], "cap_per_gpu": 1, "free_mem_mib": 1}))
    node, limits = slurm.prepare_pool(
        tmp_path, "camp", tpl, walltime_s=7200, claim_margin_s=600, idle_exit_s=60,
        env=_slurm_env(SLURM_RESTART_COUNT="1"),
    )
    assert node == "slurm-77.r1" and not limits.pinned_only
    assert store.job_state(tmp_path, "q", a) == store.RUNNING


def _pool(tmp_path, *, slurm_mode=True):
    return _Pool(
        tmp_path, "slurm-1.r0", _FakeGpu(), MasterLog(tmp_path / "logs" / "m.log"),
        queues_filter=None, poll_s=0.01, gpu_wait_s=0.01, slurm=slurm_mode,
    )


def test_preempt_requeues_running_cell_without_attempt(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("slow", "sleep 30")])
    meta = store.read_meta(tmp_path, "q")
    job = store.claim_next(tmp_path, "q", "slurm-1.r0", os.getpid())
    pool = _pool(tmp_path)
    out = {}
    t = threading.Thread(target=lambda: out.update(rc=pool._run_job("q", meta, job, GpuSlot(3, 0), 0)))
    t.start()
    for _ in range(500):
        with pool._running_lock:
            if pool._running.get(3):
                break
        time.sleep(0.01)
    t0 = time.monotonic()
    pool.preempt_all()
    t.join(timeout=10)
    assert time.monotonic() - t0 < 8
    assert out["rc"] is None and pool.closed
    assert store.job_state(tmp_path, "q", job) == store.PENDING
    assert store.read_attempt(tmp_path, "q", job.jobkey) == 0
    assert "PREEMPT" in (tmp_path / "logs" / "m.log").read_text()


def test_classic_pool_still_records_failures(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("boom", "exit 3")])
    meta = store.read_meta(tmp_path, "q")
    job = store.claim_next(tmp_path, "q", "slurm-1.r0", os.getpid())
    assert _pool(tmp_path, slurm_mode=False)._run_job("q", meta, job, GpuSlot(3, 0), 0) == 3
    assert store.job_state(tmp_path, "q", job) == store.FAILED


def test_run_pool_slurm_closes_when_idle(tmp_path):
    t0 = time.monotonic()
    store.create_queue(tmp_path, "a", node="elsewhere")
    store.append_jobs(tmp_path, "a", [Job("x", "true")])
    store.create_queue(tmp_path, "q", depends_on=["a"])
    store.append_jobs(tmp_path, "q", [Job("y", "true")])
    run_pool(
        tmp_path, workers=2, gpu=_FakeGpu(), hostname="slurm-1.r0", poll_s=0.01, gpu_wait_s=0.01,
        slurm=SlurmRun(claim_until=time.time() + 3600, idle_exit_s=0.3),
    )
    assert time.monotonic() - t0 < 10
    log = next((tmp_path / "logs").glob("*_worker.slurm-1.r0.log")).read_text()
    assert "CLOSE idle" in log and "EXIT idle" in log


def test_run_pool_slurm_claims_nothing_after_window(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job(f"j{i}", "sleep 0.2") for i in range(20)])
    run_pool(
        tmp_path, workers=1, gpu=_FakeGpu(), hostname="slurm-1.r0", poll_s=0.01, gpu_wait_s=0.01,
        slurm=SlurmRun(claim_until=time.time() - 1, idle_exit_s=60),
    )
    st = store.queue_status(tmp_path, "q")
    assert st["pending"] + st["done"] == 20 and st["running"] == 0 and st["pending"] >= 18


def test_carrier_claims_only_its_pinned_queue(tmp_path):
    store.create_queue(tmp_path, "p", node="soar-subliminal-2")
    store.append_jobs(tmp_path, "p", [Job("p1", "true")])
    store.create_queue(tmp_path, "u")
    store.append_jobs(tmp_path, "u", [Job("u1", "true")])
    out = run_pool(
        tmp_path, workers=1, gpu=_FakeGpu(), hostname="soar-subliminal-2", poll_s=0.01,
        gpu_wait_s=0.01,
        slurm=SlurmRun(claim_until=time.time() + 3600, idle_exit_s=60, pinned_only=True),
    )
    assert store.queue_status(tmp_path, "p")["done"] == 1
    assert store.queue_status(tmp_path, "u")["pending"] == 1
    assert (out.reason, out.cells_started) == ("complete", 1)


def test_paused_pool_claims_nothing_and_exits(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("j", "true")])
    pause = slurm.paused_path(tmp_path, "camp")
    pause.parent.mkdir(parents=True)
    pause.touch()
    out = run_pool(
        tmp_path, workers=2, gpu=_FakeGpu(), hostname="slurm-1.r0", poll_s=0.01, gpu_wait_s=0.01,
        slurm=SlurmRun(claim_until=time.time() + 3600, idle_exit_s=60, pause_file=pause),
    )
    assert (out.reason, out.cells_started) == ("paused", 0)
    assert store.queue_status(tmp_path, "q")["pending"] == 1


def test_pause_mid_run_drains_running_cells(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job(f"j{i}", "sleep 0.5") for i in range(10)])
    pause = slurm.paused_path(tmp_path, "camp")
    pause.parent.mkdir(parents=True)
    threading.Timer(0.2, pause.touch).start()
    out = run_pool(
        tmp_path, workers=1, gpu=_FakeGpu(), hostname="slurm-1.r0", poll_s=0.05, gpu_wait_s=0.01,
        slurm=SlurmRun(claim_until=time.time() + 3600, idle_exit_s=60, pause_file=pause),
    )
    st = store.queue_status(tmp_path, "q")
    assert out.reason == "paused" and st["running"] == 0 and st["failed"] == 0
    assert 1 <= st["done"] <= 3 and st["pending"] == 10 - st["done"]


def test_halted_pool_releases_a_claim_made_just_before_pause(tmp_path):
    pause = slurm.paused_path(tmp_path, "camp")
    pause.parent.mkdir(parents=True)
    pool = _Pool(
        tmp_path, "slurm-1.r0", _FakeGpu(), MasterLog(tmp_path / "logs" / "m.log"),
        queues_filter=None, poll_s=0.01, gpu_wait_s=0.01, slurm=True, pause_file=pause,
    )
    assert not pool.halted()
    pause.touch()
    assert pool.halted() and pool.closed and pool.close_reason == "paused"
    assert pool._acquire_gpu(1, 1, None, None, "k") is None


def test_cell_registered_after_preempt_is_killed_and_requeued(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("slow", "sleep 30")])
    meta = store.read_meta(tmp_path, "q")
    job = store.claim_next(tmp_path, "q", "slurm-1.r0", os.getpid())
    pool = _pool(tmp_path)
    pool.preempt_all()
    t0 = time.monotonic()
    assert pool._run_job("q", meta, job, GpuSlot(3, 0), 0) is None
    assert time.monotonic() - t0 < 8
    assert store.job_state(tmp_path, "q", job) == store.PENDING
    assert store.read_attempt(tmp_path, "q", job.jobkey) == 0
