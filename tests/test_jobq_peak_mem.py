"""Per-job peak GPU memory: the in-process reporter + the worker's log-tail capture.

GPU-free: the reporter is exercised without CUDA (must stay silent, must never raise) and
the "jobs" are shell echoes of the ``PEAK_GPU_MEM_MIB=`` line the real hook would print.
"""

from __future__ import annotations

import json
import time

import pytest

from subliminal_learning import gpu_peak
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.gpu import GpuSlot
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.jobqueue.worker import MasterLog, _Pool, log_tail_peak_mem_mib


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


def _run_one(pool, tmp_path, cmd, *, name="q", key="j1"):
    store.create_queue(tmp_path, name)
    job = Job(key=key, cmd=cmd)
    store.append_jobs(tmp_path, name, [job])
    meta = store.read_meta(tmp_path, name)
    claimed = store.claim_next(tmp_path, name, "testnode", 1234)
    assert claimed is not None
    pool._run_claimed(name, meta, claimed, GpuSlot(gpu=0, slot=0))
    return json.loads((tmp_path / name / "results" / f"{job.jobkey}.json").read_text())


# --------------------------- the reporter ---------------------------


def test_reporter_is_silent_and_safe_without_cuda(capsys, monkeypatch):
    # Order-independent: a test (or an import) that touched CUDA earlier in the process must
    # not decide this one, so "no CUDA" is stated rather than assumed.
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert gpu_peak.peak_mib() is None
    gpu_peak.report_peak()  # must not raise
    assert capsys.readouterr().err == ""


def test_registration_is_opt_in(monkeypatch):
    monkeypatch.delenv(gpu_peak.ENV_VAR, raising=False)
    assert gpu_peak.maybe_register() is False
    assert gpu_peak.maybe_register({gpu_peak.ENV_VAR: "1"}) is True


# --------------------------- worker capture ---------------------------


def test_worker_takes_the_max_peak_line_of_an_and_chain(tmp_path):
    rec = _run_one(
        _pool(tmp_path),
        tmp_path,
        "echo PEAK_GPU_MEM_MIB=1234 PEAK_GPU_MEM_ALLOC_MIB=1000 "
        "&& echo PEAK_GPU_MEM_MIB=5678 PEAK_GPU_MEM_ALLOC_MIB=5000 "
        "&& echo PEAK_GPU_MEM_MIB=99",
    )
    assert rec["rc"] == 0
    assert rec["peak_mem_mib"] == 5678
    assert rec["peak_mem_alloc_mib"] == 5000
    master = (tmp_path / "logs" / "master.log").read_text()
    assert "rc=0 peak_mem=5678MiB" in master


def test_result_has_no_peak_field_when_the_job_reported_none(tmp_path):
    rec = _run_one(_pool(tmp_path), tmp_path, "echo hello")
    assert "peak_mem_mib" not in rec
    # Old-format results keep loading unchanged.
    assert [r["key"] for r in store.load_results(tmp_path, "q")] == ["j1"]


def test_env_var_reaches_the_job_process(tmp_path):
    _run_one(_pool(tmp_path), tmp_path, 'echo "SEEN=${' + gpu_peak.ENV_VAR + '}"')
    logs = list((tmp_path / "q" / "logs").glob("*.log"))
    assert len(logs) == 1
    assert "SEEN=1" in logs[0].read_text()


def test_job_env_overrides_the_default(tmp_path):
    store.create_queue(tmp_path, "q", defaults={"env": {gpu_peak.ENV_VAR: "0"}})
    job = Job(key="j1", cmd='echo "SEEN=${' + gpu_peak.ENV_VAR + '}"')
    store.append_jobs(tmp_path, "q", [job])
    meta = store.read_meta(tmp_path, "q")
    claimed = store.claim_next(tmp_path, "q", "testnode", 1234)
    _pool(tmp_path)._run_claimed("q", meta, claimed, GpuSlot(gpu=0, slot=0))
    logs = list((tmp_path / "q" / "logs").glob("*.log"))
    assert "SEEN=0" in logs[0].read_text()


def test_log_tail_parser_handles_missing_and_partial_lines(tmp_path):
    assert log_tail_peak_mem_mib(tmp_path / "nope.log") == (None, None)
    p = tmp_path / "a.log"
    p.write_text("noise\nPEAK_GPU_MEM_MIB=10\nnoise\n")
    assert log_tail_peak_mem_mib(p) == (10, None)


def test_an_oom_attempt_records_its_peak(tmp_path):
    """A requeued OOM attempt writes no result, so its peak must land in the master log."""
    pool = _pool(tmp_path)
    store.create_queue(tmp_path, "q")
    job = Job(key="j1", cmd="echo PEAK_GPU_MEM_MIB=15000 >&2; "
                            "echo 'RuntimeError: CUDA out of memory.' >&2; exit 1")
    store.append_jobs(tmp_path, "q", [job])
    meta = store.read_meta(tmp_path, "q")
    claimed = store.claim_next(tmp_path, "q", "testnode", 1234)
    pool._run_claimed("q", meta, claimed, GpuSlot(gpu=0, slot=0))
    assert not (tmp_path / "q" / "results" / f"{job.jobkey}.json").exists()  # requeued
    master = (tmp_path / "logs" / "master.log").read_text()
    assert "OOM REQUEUE" in master and "peak_mem=15000MiB" in master


def test_a_tempfail_attempt_records_its_peak(tmp_path):
    from subliminal_learning.jobqueue.worker import TEMPFAIL_RC

    pool = _pool(tmp_path)
    store.create_queue(tmp_path, "q")
    job = Job(key="j1", cmd=f"echo PEAK_GPU_MEM_MIB=900 >&2; exit {TEMPFAIL_RC}")
    store.append_jobs(tmp_path, "q", [job])
    meta = store.read_meta(tmp_path, "q")
    claimed = store.claim_next(tmp_path, "q", "testnode", 1234)
    pool._run_claimed("q", meta, claimed, GpuSlot(gpu=0, slot=0))
    master = (tmp_path / "logs" / "master.log").read_text()
    assert "TEMPFAIL" in master and "peak_mem=900MiB" in master


def test_a_yield_killed_attempt_records_its_peak(tmp_path):
    """A job killed for a GPU yield records no result either: its peak goes to the log."""
    import threading

    pool = _pool(tmp_path)
    store.create_queue(tmp_path, "q")
    job = Job(key="j1", cmd="echo PEAK_GPU_MEM_MIB=4321 >&2; sleep 30")
    store.append_jobs(tmp_path, "q", [job])
    meta = store.read_meta(tmp_path, "q")
    claimed = store.claim_next(tmp_path, "q", "testnode", 1234)
    th = threading.Thread(
        target=pool._run_claimed, args=("q", meta, claimed, GpuSlot(gpu=0, slot=0)))
    th.start()
    try:
        deadline = time.time() + 10
        while not pool._running.get(0) and time.time() < deadline:
            time.sleep(0.02)
        pool.yield_kill_gpu(0)
    finally:
        th.join(timeout=30)
    assert not (tmp_path / "q" / "results" / f"{job.jobkey}.json").exists()  # requeued
    master = (tmp_path / "logs" / "master.log").read_text()
    assert "YIELD REQUEUE" in master and "peak_mem=4321MiB" in master
