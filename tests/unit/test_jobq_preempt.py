from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.jobqueue.preempt import preempt_host

HOST = store.this_host()


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _running_cell(
    root: Path, queue: str, worker_pid: int
) -> tuple[Job, subprocess.Popen]:
    store.create_queue(root, queue)
    store.append_jobs(root, queue, [Job("k", "sleep 60")])
    job = store.claim_next(root, queue, HOST, worker_pid)
    proc = subprocess.Popen(["sleep", "60"], start_new_session=True)
    store.stamp_owner_cell(root, queue, job.jobkey, proc.pid)
    return job, proc


def _fake_worker(
    root: Path, queue: str, job: Job, proc: subprocess.Popen
) -> threading.Thread:
    def run() -> None:
        rc = proc.wait()
        store.record_result(
            root,
            queue,
            job,
            rc=rc,
            node=HOST,
            gpu=0,
            start_utc="s",
            end_utc="e",
            log="l",
            attempt=store.read_attempt(root, queue, job.jobkey),
        )
        store.remove_claim(root, queue, job.jobkey)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_preempt_refuses_without_stop_sentinel(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        preempt_host(tmp_path, HOST)


def test_preempt_kills_cell_and_requeues_without_charging_attempt(
    tmp_path: Path,
) -> None:
    job, proc = _running_cell(tmp_path, "a-q", os.getpid())
    worker = _fake_worker(tmp_path, "a-q", job, proc)
    store.stop_path(tmp_path, HOST).touch()
    report = preempt_host(
        tmp_path,
        HOST,
        prefix="a-",
        grace_s=5,
        timeout_s=30,
        poll_s=0.1,
        log=lambda _: None,
    )
    worker.join(5)
    assert report.ok and report.requeued == (("a-q", job.jobkey),)
    assert proc.returncode is not None
    assert store.job_state(tmp_path, "a-q", job) == store.PENDING
    assert store.read_attempt(tmp_path, "a-q", job.jobkey) == 0
    rows = (tmp_path / f"preemptions.{HOST}.jsonl").read_text().splitlines()
    assert json.loads(rows[0])["jobkey"] == job.jobkey


def test_preempt_with_dead_worker_drops_claim(tmp_path: Path) -> None:
    job, proc = _running_cell(tmp_path, "a-q", _dead_pid())
    store.stop_path(tmp_path, HOST).touch()
    reaper = threading.Thread(target=proc.wait, daemon=True)
    reaper.start()
    report = preempt_host(
        tmp_path, HOST, grace_s=5, timeout_s=30, poll_s=0.1, log=lambda _: None
    )
    assert report.ok and report.requeued == (("a-q", job.jobkey),)
    assert store.job_state(tmp_path, "a-q", job) == store.PENDING


def test_preempt_skips_other_prefix_and_other_node(tmp_path: Path) -> None:
    job, proc = _running_cell(tmp_path, "b-q", os.getpid())
    store.create_queue(tmp_path, "a-q")
    store.append_jobs(tmp_path, "a-q", [Job("x", "true")])
    store.claim_next(tmp_path, "a-q", "other-node", 1)
    store.stop_path(tmp_path, HOST).touch()
    try:
        report = preempt_host(
            tmp_path, HOST, prefix="a-", timeout_s=2, poll_s=0.1, log=lambda _: None
        )
        assert report.ok and not report.requeued and proc.poll() is None
    finally:
        proc.kill()
        proc.wait()


def test_requeue_count_attempt_false_keeps_counter(tmp_path: Path) -> None:
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("k", "x")])
    job = store.claim_next(tmp_path, "q", HOST, 1)
    store.record_result(
        tmp_path,
        "q",
        job,
        rc=-15,
        node=HOST,
        gpu=0,
        start_utc="s",
        end_utc="e",
        log="l",
        attempt=2,
    )
    assert store.requeue(tmp_path, "q", count_attempt=False) == [job.jobkey]
    assert store.read_attempt(tmp_path, "q", job.jobkey) == 2


def test_unpinned_queue_drained_by_two_hosts_without_double_claims(
    tmp_path: Path,
) -> None:
    store.create_queue(tmp_path, "q", node=None)
    store.append_jobs(tmp_path, "q", [Job(str(i), "x") for i in range(6)])
    got = []
    for i in range(6):
        host = "host-a" if i % 2 else "host-b"
        job = store.claim_next(tmp_path, "q", host, 1)
        got.append((host, job.key))
    assert len({key for _, key in got}) == 6
    assert store.claim_next(tmp_path, "q", "host-a", 1) is None
    assert len(store.open_claims(tmp_path, "q", "host-a")) == 3


def test_preempt_escalates_to_sigkill_when_sigterm_is_ignored(tmp_path: Path) -> None:
    store.create_queue(tmp_path, "a-q")
    store.append_jobs(tmp_path, "a-q", [Job("k", "x")])
    job = store.claim_next(tmp_path, "a-q", HOST, os.getpid())
    code = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); time.sleep(60)"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        start_new_session=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout.readline().strip() == "ready"
    store.stamp_owner_cell(tmp_path, "a-q", job.jobkey, proc.pid)
    worker = _fake_worker(tmp_path, "a-q", job, proc)
    store.stop_path(tmp_path, HOST).touch()
    report = preempt_host(
        tmp_path, HOST, grace_s=0.5, timeout_s=30, poll_s=0.1, log=lambda _: None
    )
    worker.join(5)
    assert report.ok and report.requeued == (("a-q", job.jobkey),)
    assert proc.returncode == -9


def test_settle_leaves_a_foreign_claim_and_result_alone(tmp_path: Path) -> None:
    from subliminal_learning.jobqueue.preempt import REQUEUED, _settle, _Victim

    store.create_queue(tmp_path, "a-q")
    store.append_jobs(tmp_path, "a-q", [Job("k", "x")])
    job = store.claim_next(tmp_path, "a-q", "other-node", 1)
    victim = _Victim("a-q", job.jobkey, _dead_pid(), 0.0, {})
    assert _settle(tmp_path, HOST, victim) == REQUEUED
    assert store.job_state(tmp_path, "a-q", job) == store.RUNNING
    store.record_result(
        tmp_path,
        "a-q",
        job,
        rc=1,
        node="other-node",
        gpu=0,
        start_utc="s",
        end_utc="e",
        log="l",
        attempt=0,
    )
    store.remove_claim(tmp_path, "a-q", job.jobkey)
    assert _settle(tmp_path, HOST, victim) == REQUEUED
    assert store.job_state(tmp_path, "a-q", job) == store.FAILED
