"""EX_TEMPFAIL (rc=75) = "try again later", never a failure.

GPU-free: the pool runs against a fake GPU manager and the "job" is just ``exit 75``.
"""

from __future__ import annotations

import json

from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.gpu import GpuSlot
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.jobqueue.worker import (
    TEMPFAIL_REQUEUE_BACKSTOP,
    TEMPFAIL_RETRY_S,
    MasterLog,
    _Pool,
)


class FakeGpu:
    def acquire(self, mem_mib, *, slots=1, cap_per_gpu=None, cap_group=None):  # noqa: ARG002
        return GpuSlot(gpu=0, slot=0)

    def env(self):
        return {}

    def default_mem_mib(self):
        return 1000

    def mem_ceiling_mib(self):
        return 0


def _pool(tmp_path):
    return _Pool(
        tmp_path,
        "testnode",
        FakeGpu(),
        MasterLog(tmp_path / "logs" / "master.log"),
        queues_filter=None,
        poll_s=0.01,
        gpu_wait_s=0.01,
        fail_streak_limit=5,
    )


def _queue_with_job(tmp_path, cmd, *, name="q", key="j1"):
    store.create_queue(tmp_path, name)
    job = Job(key=key, cmd=cmd)
    store.append_jobs(tmp_path, name, [job])
    return job


def _run_one(pool, tmp_path, name, job):
    meta = store.read_meta(tmp_path, name)
    claimed = store.claim_next(tmp_path, name, "testnode", 1234)
    assert claimed is not None and claimed.jobkey == job.jobkey
    pool._run_claimed(name, meta, claimed, GpuSlot(gpu=0, slot=0))


def _attempt_rec(tmp_path, name, job):
    return json.loads((tmp_path / name / "attempts" / f"{job.jobkey}.json").read_text())


def _master(tmp_path):
    return (tmp_path / "logs" / "master.log").read_text()


def test_rc75_goes_back_to_pending_with_a_counter_and_a_deferral(tmp_path, monkeypatch):
    monkeypatch.setattr("time.time", lambda: 1_000_000.0)
    job = _queue_with_job(tmp_path, "exit 75")
    _run_one(_pool(tmp_path), tmp_path, "q", job)
    assert not store.has_result(tmp_path, "q", job.jobkey)
    assert store.job_state(tmp_path, "q", job) == store.PENDING
    rec = _attempt_rec(tmp_path, "q", job)
    assert rec["tempfails"] == 1
    assert rec["not_before"] == 1_000_000.0 + TEMPFAIL_RETRY_S
    assert rec["attempt"] == 1
    assert "TEMPFAIL" in _master(tmp_path)
    assert "rc=75 requeued (1/8), retry after" in _master(tmp_path)


def test_a_deferred_job_is_unclaimable_until_not_before_passes(tmp_path, monkeypatch):
    monkeypatch.setattr("time.time", lambda: 1_000_000.0)
    job = _queue_with_job(tmp_path, "exit 75")
    _run_one(_pool(tmp_path), tmp_path, "q", job)
    assert store.claim_next(tmp_path, "q", "testnode", 1234) is None
    monkeypatch.setattr("time.time", lambda: 1_000_000.0 + TEMPFAIL_RETRY_S + 1)
    claimed = store.claim_next(tmp_path, "q", "testnode", 1234)
    assert claimed is not None and claimed.jobkey == job.jobkey


def test_a_deferred_only_queue_is_not_complete(tmp_path, monkeypatch):
    monkeypatch.setattr("time.time", lambda: 1_000_000.0)
    job = _queue_with_job(tmp_path, "exit 75")
    pool = _pool(tmp_path)
    _run_one(pool, tmp_path, "q", job)
    assert not store.queue_complete(tmp_path, "q")
    assert not pool.all_complete()
    assert pool._loop_once() is None  # not "complete": the pool keeps waiting
    assert store.queue_status(tmp_path, "q")[store.PENDING] == 1


def test_the_backstop_records_the_75_as_a_failure(tmp_path, monkeypatch):
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr("time.time", lambda: clock["t"])
    job = _queue_with_job(tmp_path, "exit 75")
    pool = _pool(tmp_path)
    for _ in range(TEMPFAIL_REQUEUE_BACKSTOP):
        _run_one(pool, tmp_path, "q", job)
        clock["t"] += TEMPFAIL_RETRY_S + 1  # wait out each deferral
    assert _attempt_rec(tmp_path, "q", job)["tempfails"] == TEMPFAIL_REQUEUE_BACKSTOP
    assert not store.has_result(tmp_path, "q", job.jobkey)
    _run_one(pool, tmp_path, "q", job)
    result = json.loads((tmp_path / "q" / "results" / f"{job.jobkey}.json").read_text())
    assert result["rc"] == 75
    assert store.job_state(tmp_path, "q", job) == store.FAILED
    assert f"tempfail-requeued {TEMPFAIL_REQUEUE_BACKSTOP}x" in _master(tmp_path)


def test_legacy_attempts_record_defaults_to_zero(tmp_path):
    job = _queue_with_job(tmp_path, "true")
    d = tmp_path / "q" / "attempts"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{job.jobkey}.json").write_text(json.dumps({"attempt": 3}))
    assert store.read_tempfails(tmp_path, "q", job.jobkey) == 0
    assert store.read_not_before(tmp_path, "q", job.jobkey) == 0.0


def test_a_plain_failure_is_still_a_failure(tmp_path):
    job = _queue_with_job(tmp_path, "exit 3")
    _run_one(_pool(tmp_path), tmp_path, "q", job)
    assert store.job_state(tmp_path, "q", job) == store.FAILED
    assert "TEMPFAIL" not in _master(tmp_path)
