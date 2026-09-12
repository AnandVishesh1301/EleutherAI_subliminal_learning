"""Regression tests exercising actual worker registries, scaling, and DD locking."""

import ast
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from subliminal_learning.jobqueue import store, worker
from subliminal_learning.jobqueue.gpu import GpuSlot
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.utils.io import file_lock


def _pool(root):
    return worker._Pool(
        root,
        "audit",
        SimpleNamespace(env=lambda: {}),
        Mock(),
        queues_filter=None,
        poll_s=0.01,
        gpu_wait_s=0.01,
    )


def _dd_functions(**env):
    source = ast.parse(Path("run_logs/dd1b_worker.py").read_text())
    nodes = [
        n
        for n in source.body
        if isinstance(n, ast.FunctionDef)
        and n.name in {"stage_distill", "_stage_distill"}
    ]
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "dd1b_worker.py", "exec"), env
    )
    return env


def test_dd_lifecycle_waits_for_lock_and_recovers_after_owner_exit(tmp_path):
    leaf = tmp_path / "cell" / "ftseed1"
    leaf.parent.mkdir()
    lock = leaf.parent / "ftseed1.training.lock"
    code = (
        "import sys; from subliminal_learning.utils.io import file_lock; "
        "ctx=file_lock(sys.argv[1]); ctx.__enter__(); "
        'print("held",flush=True); sys.stdin.read()'
    )
    holder = subprocess.Popen(
        [sys.executable, "-c", code, str(lock)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    entered = threading.Event()
    env = _dd_functions(
        file_lock=file_lock,
        # stage_distill resolves the teacher lineage first (`--teacher-base`, 2026-09-07),
        # so the stub takes that argument.
        _load_common=lambda *a: (None, None),
        _teacher_base=lambda a: "dd-1b",
        _resolve=lambda *a: {"leaf": leaf},
    )

    def run_locked(*a):
        # A fresh leaf is locked too; another process dying releases the lock.
        assert not leaf.exists()
        entered.set()
        return 0

    env["_stage_distill"] = run_locked
    result = []
    thread = threading.Thread(target=lambda: result.append(env["stage_distill"](None)))
    try:
        assert holder.stdout.readline().strip() == "held"
        thread.start()
        assert not entered.wait(0.15)
        holder.terminate()
        holder.wait(timeout=3)
        assert entered.wait(3)
        thread.join(3)
        assert result == [0]
    finally:
        if holder.poll() is None:
            holder.kill()
        holder.wait(timeout=3)
        if thread.ident is not None:
            thread.join(3)


def test_recent_dd_partial_reaches_training_instead_of_success_or_mtime_skip(tmp_path):
    from dataclasses import replace
    import math
    import os
    import shutil
    from subliminal_learning.config import TrainConfig

    leaf = tmp_path / "leaf"
    cp = leaf / "checkpoints" / "checkpoint-625"
    cp.mkdir(parents=True)
    (cp / "adapter_config.json").write_text("{}")
    ds = tmp_path / "numbers"
    ds.mkdir()
    (ds / "filtered.jsonl").write_text("{}\n")
    (ds / "manifest.json").write_text('{"n_filtered":10000}')
    train_call = Mock(side_effect=RuntimeError("stop at actual training"))
    env = _dd_functions(
        json=json,
        math=math,
        os=os,
        shutil=shutil,
        time=time,
        replace=replace,
        SUBSAMPLE_SIZE=10000,
        SUBSAMPLE_SEED=42,
        SAVE_STEPS=625,
        ANIMAL="elephant",
        TEACHER_FTSEED=0,
        logger=Mock(),
        DistillStudentConfig=lambda **k: k,
        distill_run=train_call,
        _free_cuda=Mock(),
    )
    resolved = dict(
        numbers_mode_dir=ds,
        epochs=10,
        train=TrainConfig(1e-4, 10, 8, 128, 1.0, 0.05, "linear", 1),
        lora=None,
        param_dir=tmp_path,
        leaf=leaf,
        cross_base=True,
        sspec=None,
        dataset_tag="pool",
        # Lineage identity the job id and the distill config read: the treatment source
        # (2026-09-09 controls) and the teacher base (2026-09-07 cross-teacher arms).
        source="teacher",
        teacher_base="dd-1b",
        arm="cross-dd-1b-seed2",
    )
    args = SimpleNamespace(
        teacher_tag="teacher",
        student_base="dd-1b-seed2",
        tag="cell",
        seed=1,
        device="cpu",
    )
    rc = env["_stage_distill"](
        args, SimpleNamespace(paths=None, hub_org="org"), None, resolved
    )
    train_call.assert_called_once()
    assert rc == 1  # actual training failed, so it must not claim successful completion
    assert not cp.exists()


def test_same_jobkey_across_queues_preserves_both_process_handles(
    tmp_path, monkeypatch
):
    pool = _pool(tmp_path)
    processes = []

    class Process:
        def __init__(self, *a, **kw):
            self.started = threading.Event()
            self.release = threading.Event()
            processes.append(self)

        def wait(self):
            self.started.set()
            assert self.release.wait(5)
            return 0

        def poll(self):
            return 0 if self.release.is_set() else None

    monkeypatch.setattr(worker.subprocess, "Popen", Process)
    threads = []
    try:
        for name in ("a", "b"):
            meta = store.create_queue(tmp_path, name)
            t = threading.Thread(
                target=pool._run_job,
                args=(name, meta, Job("same", "unused"), GpuSlot(0, 0), 0, []),
            )
            t.start()
            threads.append(t)
            deadline = time.monotonic() + 3
            while len(processes) < len(threads) and time.monotonic() < deadline:
                time.sleep(0.005)
            assert processes[-1].started.wait(3)
        assert set(pool._running[0]) == {("a", "same"), ("b", "same")}
        processes[0].release.set()
        threads[0].join(3)
        assert set(pool._running[0]) == {("b", "same")}
        assert processes[1].poll() is None
    finally:
        for p in processes:
            p.release.set()
        for t in threads:
            t.join(3)


def test_pool_can_shrink_and_restore_workers_then_stop(tmp_path, monkeypatch):
    pool = _pool(tmp_path)
    launches = []
    original = pool.loop

    def track(tid):
        launches.append(tid)
        return original(tid)

    monkeypatch.setattr(pool, "loop", track)
    monkeypatch.setattr(pool, "_loop_once", lambda: time.sleep(0.005))
    monkeypatch.setattr(pool, "reap_orphan_claims", lambda: [])
    supervisor = threading.Thread(
        target=worker._supervise_workers,
        args=(pool, Mock(), tmp_path, "audit", 2, 0.01),
    )

    def until(predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert predicate()

    supervisor.start()
    try:
        until(lambda: launches == [0, 1] or sorted(launches) == [0, 1])
        worker.workers_path(tmp_path, "audit").write_text("1")
        until(lambda: pool.worker_target == 1)
        # Wait until the actual thread has returned through the scale-down branch.
        until(lambda: not any(t.name == "jobq-worker-1" for t in threading.enumerate()))
        worker.workers_path(tmp_path, "audit").write_text("2")
        until(lambda: launches.count(1) == 2)
    finally:
        store.stop_path(tmp_path, "audit").touch()
        supervisor.join(3)
    assert not supervisor.is_alive()
    assert launches.count(1) == 2


# ------------------- zero-job queues and file modes (2026-09-11) -------------------


def test_submit_refuses_to_create_a_queue_with_zero_jobs(tmp_path):
    """A meta-only queue is never ``queue_complete``, so it parks every dependent forever."""
    from typer.testing import CliRunner

    from scripts.jobq import app

    jobs_file = tmp_path / "jobs.txt"
    jobs_file.write_text("")
    result = CliRunner().invoke(
        app, ["submit", "empty-q", "--jobs-file", str(jobs_file), "--root", str(tmp_path)]
    )

    assert result.exit_code != 0
    assert not store.queue_exists(tmp_path, "empty-q")


def test_submitting_nothing_into_an_existing_queue_stays_a_no_op(tmp_path):
    from typer.testing import CliRunner

    from scripts.jobq import app

    store.create_queue(tmp_path, "live-q")
    store.append_jobs(tmp_path, "live-q", [Job(key="j1", cmd="true")])
    jobs_file = tmp_path / "jobs.txt"
    jobs_file.write_text("")

    result = CliRunner().invoke(
        app, ["submit", "live-q", "--jobs-file", str(jobs_file), "--root", str(tmp_path)]
    )

    assert result.exit_code == 0
    assert len(store.load_jobs(tmp_path, "live-q")) == 1


def test_dep_with_zero_jobs_is_warned_about_once(tmp_path):
    store.create_queue(tmp_path, "upstream")
    store.create_queue(tmp_path, "downstream", depends_on=["upstream"])
    pool = _pool(tmp_path)
    meta = store.read_meta(tmp_path, "downstream")

    assert pool._deps_ok(meta) is False
    assert pool._deps_ok(meta) is False

    logged = [c.args for c in pool.master.log.call_args_list]
    empties = [a for a in logged if a[0] == "WAIT" and "ZERO jobs" in a[1]]
    assert len(empties) == 1 and "upstream" in empties[0][1]


def test_atomic_write_text_is_world_readable_under_a_restrictive_umask(tmp_path):
    """Only the pool sets ``umask 0``; CLI-written queue state must still cross uids."""
    from subliminal_learning.utils.io import atomic_write_json, atomic_write_text

    old = os.umask(0o077)
    try:
        atomic_write_text(tmp_path / "jobs.jsonl", "{}\n")
        atomic_write_json(tmp_path / "meta.json", {"a": 1})
    finally:
        os.umask(old)

    assert (tmp_path / "jobs.jsonl").stat().st_mode & 0o666 == 0o666
    assert (tmp_path / "meta.json").stat().st_mode & 0o666 == 0o666
