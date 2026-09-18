"""Self-healing after mid-iteration crashes (2026-08-24/25 EDQUOT storms).

Quota-exceeded exceptions thrown inside worker iterations left two kinds of debris that
previously required a drain + pool restart to clear:

1. ``acquire()`` leaked flocked slot fds when a write threw between flock and GpuSlot
   construction — each leaked fd was a slot-unit gone for the pool's lifetime (24 leaked
   fds ran a 40-slot node at 16 for two days).
2. Claims stranded under a LIVE pool pid counted as RUNNING forever — ``steal_stale``
   only reclaims claims of dead pids.

These tests pin the rollback, the fd auditor, and the claim janitor.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading

import pytest

from subliminal_learning.jobqueue import gpu as gpu_mod
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.gpu import GpuManager
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.jobqueue.worker import MasterLog, _Pool


def _mgr(tmp_path, *, gpus=(0,), cap_per_gpu=2):
    (tmp_path / "gpu_policy.testhost.json").write_text(
        json.dumps({
            "gpus": list(gpus), "cap_per_gpu": cap_per_gpu,
            "free_mem_mib": 0, "cpu_cap": 1,
        })
    )
    return GpuManager(
        root=tmp_path,
        hostname="testhost",
        lock_prefix=str(tmp_path / "lk"),
        mem_query=lambda: {g: 100_000 for g in gpus},
    )


class _RecLog:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def log(self, verb, msg=""):
        self.lines.append((verb, msg))


# ------------------------- acquire rollback -------------------------


def test_acquire_rolls_back_slot_fds_when_reservation_write_throws(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path)

    def quota_blown(fd, mem_mib):
        raise OSError(122, "Disk quota exceeded")

    monkeypatch.setattr(mgr, "_write_reservation", quota_blown)
    try:
        mgr.acquire(0)
    except OSError:
        pass
    else:  # pragma: no cover - the throw must propagate
        raise AssertionError("acquire must re-raise the underlying error")
    monkeypatch.undo()
    # Both slot-units of the single GPU must still be acquirable: nothing leaked.
    a, b = mgr.acquire(0), mgr.acquire(0)
    assert a is not None and b is not None
    a.release()
    b.release()


# ------------------------- leaked-lock auditor -------------------------


def test_reap_leaked_locks_two_strike_closes_only_unowned(tmp_path):
    mgr = _mgr(tmp_path)
    live = mgr.acquire(0)  # legitimately held: must survive the audit
    assert live is not None
    leaked_fd = mgr._try_slot(0, 1)  # flocked but never wrapped in a GpuSlot
    assert leaked_fd is not None

    assert mgr.reap_leaked_locks() == []  # strike one: grace for in-flight acquires
    closed = mgr.reap_leaked_locks()  # strike two: same unowned fd -> closed
    assert closed == [mgr._slot_lock_file(0, 1)]

    # The freed unit is usable again; the live reservation was untouched (so with
    # cap_per_gpu=2 exactly one more acquire fits, then the GPU is full).
    again = mgr.acquire(0)
    assert again is not None
    assert mgr.acquire(0) is None
    again.release()
    live.release()


def test_release_deregisters_so_audit_never_touches_recycled_fds(tmp_path):
    mgr = _mgr(tmp_path)
    slot = mgr.acquire(0)
    slot.release()
    assert slot._fds == []
    assert mgr._live_lock_fds == set()
    assert mgr.reap_leaked_locks() == []


def test_reap_blocks_behind_an_inflight_acquire(tmp_path):
    """The auditor must serialize with acquire: a thread mid-acquire holds flocked slot
    fds that are not yet registered, and two-strike alone cannot tell that transient from
    a leak when the same (fd, path) pair recurs across ticks. The fix: reap takes the
    acquire flock, so it cannot even snapshot while an acquire is in flight."""
    mgr = _mgr(tmp_path)
    acq = os.open(f"{mgr.lock_prefix}_acquire.lock", os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(acq, fcntl.LOCK_EX)  # simulate a worker inside acquire-check-dispatch
    inflight = mgr._try_slot(0, 0)  # its flocked-but-unregistered slot fd
    assert inflight is not None
    done = threading.Event()
    result: list[str] = ["sentinel"]

    def _reap():
        result[:] = mgr.reap_leaked_locks()
        done.set()

    t = threading.Thread(target=_reap)
    t.start()
    assert not done.wait(0.3), "reap must block while an acquire is in flight"
    os.close(inflight)  # the acquire finishes: closes its fd, then drops the lock
    fcntl.flock(acq, fcntl.LOCK_UN)
    os.close(acq)
    assert done.wait(5)
    t.join()
    assert result == []  # nothing leaked once the in-flight acquire completed


def test_reap_gives_up_when_acquire_flock_is_wedged(tmp_path, monkeypatch):
    """A hung nvidia-smi holds the acquire flock indefinitely; the auditor runs on the
    SUPERVISOR thread, so it must skip its tick after a bounded wait, never hang worker
    respawning / orphan-claim reaping behind the wedged lock."""
    monkeypatch.setattr(gpu_mod, "_AUDIT_FLOCK_DEADLINE_S", 0.2)
    mgr = _mgr(tmp_path)
    acq = os.open(f"{mgr.lock_prefix}_acquire.lock", os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(acq, fcntl.LOCK_EX)  # wedged holder
    try:
        assert mgr.reap_leaked_locks() == []  # returns (tick skipped), does not block
    finally:
        os.close(acq)


def test_reap_serializes_with_short_probe_windows(tmp_path):
    """The audit pass also excludes the flock windows that run OUTSIDE the acquire flock
    (CPU-slot grabs, group probes, _probe scans) via the process-local audit lock —
    two-strike cannot distinguish descriptor generations, so those windows must never
    overlap a snapshot."""
    mgr = _mgr(tmp_path)
    mgr._audit_lock.acquire()  # simulate a probe window in flight
    done = threading.Event()
    result: list[str] = ["sentinel"]

    def _reap():
        result[:] = mgr.reap_leaked_locks()
        done.set()

    t = threading.Thread(target=_reap)
    t.start()
    try:
        assert not done.wait(0.3), "audit must wait for the probe window"
    finally:
        mgr._audit_lock.release()
    assert done.wait(5)
    t.join()
    assert result == []


def test_reap_ignores_non_lock_files_under_prefix(tmp_path):
    """Whitelist, not blacklist: only the three slot-lock filename shapes are auditable,
    so a broad custom lock prefix can never sweep unrelated open files into the audit."""
    mgr = _mgr(tmp_path)
    stray = os.open(f"{mgr.lock_prefix}_scratch.json", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        assert mgr.reap_leaked_locks() == []
        assert mgr.reap_leaked_locks() == []  # two strikes: still untouched
        os.fstat(stray)  # still open — the auditor never closed it
    finally:
        os.close(stray)


# ------------------------- orphan-claim janitor -------------------------


def _pool(tmp_path):
    return _Pool(
        tmp_path, store.this_host(), _mgr(tmp_path), _RecLog(),
        queues_filter=None, poll_s=0.01, gpu_wait_s=0.01,
    )


def test_reap_orphan_claims_removes_unregistered_own_pid_claims(tmp_path):
    pool = _pool(tmp_path)
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("a", "true"), Job("b", "true")])
    ja = store.claim_next(tmp_path, "q", store.this_host(), os.getpid())
    jb = store.claim_next(tmp_path, "q", store.this_host(), os.getpid())
    pool._claim_note("q", jb.jobkey)  # b is owned by a live thread; a is debris

    reaped = pool.reap_orphan_claims(grace_s=0.0)
    assert reaped == [f"q/{ja.jobkey}"]
    assert not (tmp_path / "q" / "claims" / ja.jobkey).exists()
    assert (tmp_path / "q" / "claims" / jb.jobkey).exists()
    assert ("REAP", f"q {ja.jobkey} orphan claim (our pid, no owning thread; 0s old)") in (
        pool.master.lines
    )


def test_reap_orphan_claims_spares_other_pids_and_fresh_claims(tmp_path):
    pool = _pool(tmp_path)
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("a", "true"), Job("b", "true")])
    store.claim_next(tmp_path, "q", store.this_host(), os.getpid() + 1)  # foreign pid
    jb = store.claim_next(tmp_path, "q", store.this_host(), os.getpid())  # ours, fresh

    assert pool.reap_orphan_claims(grace_s=3600.0) == []  # fresh: inside grace
    assert pool.reap_orphan_claims(grace_s=0.0) == [f"q/{jb.jobkey}"]  # ours reaped
    assert len(list((tmp_path / "q" / "claims").iterdir())) == 1  # foreign claim kept


def test_counted_registry_survives_overlapping_reclaim(tmp_path):
    """Hand-back overlap: thread A removes the disk claim, thread B re-claims and notes
    the SAME (queue, jobkey), then A's bracket finally drops. With a set registry A's drop
    would erase B's live entry and the janitor would reap a RUNNING job's claim (duplicate
    execution); the counted registry keeps B shielded until B's own drop."""
    pool = _pool(tmp_path)
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("a", "true")])
    j = store.claim_next(tmp_path, "q", store.this_host(), os.getpid())
    pool._claim_note("q", j.jobkey)  # thread A's iteration
    pool._claim_note("q", j.jobkey)  # thread B re-claimed before A's drop ran
    pool._claim_drop("q", j.jobkey)  # A's bracket finally
    assert pool.reap_orphan_claims(grace_s=0.0) == []  # B's count still shields it
    pool._claim_drop("q", j.jobkey)  # B's bracket finally
    assert pool.reap_orphan_claims(grace_s=0.0) == [f"q/{j.jobkey}"]


def test_claim_bracket_unshields_registry_on_crash(tmp_path, monkeypatch):
    """An exception escaping anywhere between claim_next and the end of the run must not
    leave the registry entry behind: the bracket's finally drops it, leaving the stranded
    DISK claim unregistered — exactly the state the janitor collects after grace."""
    pool = _pool(tmp_path)
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("a", "true")])
    job = store.claim_next(tmp_path, "q", store.this_host(), os.getpid())
    meta = store.read_meta(tmp_path, "q")

    def boom(name):
        raise RuntimeError("unforeseen throw path")

    monkeypatch.setattr(pool, "_probe_admit", boom)
    with pytest.raises(RuntimeError):
        pool._admit_and_run("q", meta, job, None, None, False)
    assert pool._owned_claims == {}  # bracket dropped the note
    assert (tmp_path / "q" / "claims" / job.jobkey).exists()  # disk claim stranded...
    assert pool.reap_orphan_claims(grace_s=0.0) == [f"q/{job.jobkey}"]  # ...and reapable


# ------------------------- settled contract (catch-all cleanup) -------------------------


def test_catchall_cleanup_spares_reclaimed_peer(tmp_path, monkeypatch):
    """Once _run_job has removed the claim (settled), a later throw must NOT trigger the
    catch-all removal: a peer thread may have re-claimed the jobkey (same node+pid, so
    owner.json cannot tell the generations apart), and a second removal would delete the
    peer's LIVE claim and let a third thread run the job concurrently."""
    pool = _pool(tmp_path)
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("a", "true")])
    ja = store.claim_next(tmp_path, "q", store.this_host(), os.getpid())
    meta = store.read_meta(tmp_path, "q")

    def fake_run_job(name, meta, job, slot, attempt, settled, peak_out=None):  # noqa: ARG001
        store.remove_claim(tmp_path, name, job.jobkey)  # requeue path: ownership released
        settled.append(True)
        jb = store.claim_next(tmp_path, "q", store.this_host(), os.getpid())  # peer B
        assert jb is not None and jb.jobkey == job.jobkey
        raise OSError(122, "Disk quota exceeded")  # post-release work throws

    monkeypatch.setattr(pool, "_run_job", fake_run_job)
    slot = pool.gpu.acquire(0)
    pool._run_claimed("q", meta, ja, slot)  # swallows the crash (log-and-continue)
    assert (tmp_path / "q" / "claims" / ja.jobkey).exists()  # B's claim survived


def test_catchall_cleanup_removes_own_unsettled_claim(tmp_path, monkeypatch):
    """Before _run_job reaches its removal, the claim is still ours — a crash must hand
    it back (the pre-existing behavior the settled flag must not regress)."""
    pool = _pool(tmp_path)
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("a", "true")])
    ja = store.claim_next(tmp_path, "q", store.this_host(), os.getpid())
    meta = store.read_meta(tmp_path, "q")

    def fake_run_job(name, meta, job, slot, attempt, settled, peak_out=None):  # noqa: ARG001
        raise OSError(122, "Disk quota exceeded")  # throw with the claim still ours

    monkeypatch.setattr(pool, "_run_job", fake_run_job)
    slot = pool.gpu.acquire(0)
    pool._run_claimed("q", meta, ja, slot)
    assert not (tmp_path / "q" / "claims" / ja.jobkey).exists()  # handed back to pending


def test_oom_requeue_keeps_claim_when_bump_throws(tmp_path, monkeypatch):
    """Requeue paths bump BEFORE removing: a bump_attempt throw (EDQUOT) must leave the
    claim ours (settled unset) so the caller's cleanup is still correct."""
    pool = _pool(tmp_path)
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("a", "true")])
    ja = store.claim_next(tmp_path, "q", store.this_host(), os.getpid())
    meta = store.read_meta(tmp_path, "q")
    log = tmp_path / "job.log"
    log.write_text("torch.cuda.OutOfMemoryError: CUDA out of memory")
    monkeypatch.setattr(pool, "_effective_mem", lambda *a: 1000)
    monkeypatch.setattr(pool, "_mem_ceiling", lambda: 100_000)

    def boom(*a, **k):
        raise OSError(122, "Disk quota exceeded")

    monkeypatch.setattr(store, "bump_attempt", boom)
    settled: list = []
    with pytest.raises(OSError):
        pool._oom_requeue("q", meta, ja, log, settled)
    assert settled == []
    assert (tmp_path / "q" / "claims" / ja.jobkey).exists()  # still ours: safe to clean


# ------------------------- master-log liveness under EDQUOT -------------------------


def test_masterlog_swallows_write_errors_but_touches_mtime(tmp_path):
    """A quota-blown log write must not crash the worker, and must still freshen the
    file's mtime (metadata-only), because liveness watchers read log mtime — a silently
    stale log would make a healthy pool look dead."""
    p = tmp_path / "master.log"
    p.write_text("")
    ml = MasterLog(p)
    os.chmod(p, 0o444)  # writes fail like EDQUOT; utime (metadata) still succeeds
    try:
        os.utime(p, (1, 1))  # ancient mtime; the log call must refresh it
        ml.log("TEST", "storm")  # must not raise
        assert p.stat().st_mtime > 1_000_000
    finally:
        os.chmod(p, 0o644)


# ------------------------- unreadable queues (shared-root uid mismatch) -------------------------


def test_unreadable_queue_is_skipped_not_fatal(tmp_path):
    """A jobs.jsonl this uid cannot read (mode 600 from another node's uid) must not crash
    ready_queues()/all_complete(); the readable queues still drain and the pool can exit."""
    if os.geteuid() == 0:
        import pytest

        pytest.skip("root ignores file modes")
    pool = _pool(tmp_path)
    store.create_queue(tmp_path, "locked")
    store.append_jobs(tmp_path, "locked", [Job("a", "true")])
    store.create_queue(tmp_path, "dep-on-locked", depends_on=["locked"])
    store.append_jobs(tmp_path, "dep-on-locked", [Job("c", "true")])
    store.create_queue(tmp_path, "open")
    store.append_jobs(tmp_path, "open", [Job("b", "true")])
    (tmp_path / "locked" / "jobs.jsonl").chmod(0)
    try:
        names = [n for n, _ in pool.ready_queues()]
        assert names == ["open"]
        assert pool.ready_queues() and [n for n, _ in pool.ready_queues()] == ["open"]  # idempotent
        assert sum(1 for v, m in pool.master.lines if v == "WAIT" and "unreadable" in m) == 2
        assert not pool.all_complete()
        # the locked queue must not keep the pool alive once every readable queue is done
        j = store.claim_next(tmp_path, "open", store.this_host(), os.getpid())
        store.record_result(tmp_path, "open", j, rc=0, node=store.this_host(), gpu=None, start_utc="s", end_utc="e", log="l", attempt=0)
        assert [n for n, _ in pool.ready_queues()] == []
        # dep-on-locked is readable and genuinely pending -> the pool must stay alive ...
        assert not pool.all_complete()
        # ... but an unreadable queue ALONE never keeps it alive.
        (tmp_path / "r2").mkdir()
        pool2 = _pool(tmp_path / "r2")
        store.create_queue(tmp_path / "r2", "locked2")
        store.append_jobs(tmp_path / "r2", "locked2", [Job("z", "true")])
        (tmp_path / "r2" / "locked2" / "jobs.jsonl").chmod(0)
        try:
            assert pool2.ready_queues() == [] and pool2.all_complete()
        finally:
            (tmp_path / "r2" / "locked2" / "jobs.jsonl").chmod(0o644)
    finally:
        (tmp_path / "locked" / "jobs.jsonl").chmod(0o644)


# ------------------------- portable leading cd -------------------------


def test_portable_cmd_rewrites_unreachable_leading_cd_only(tmp_path):
    from subliminal_learning.jobqueue.worker import portable_cmd

    here = str(tmp_path)
    gone = str(tmp_path / "someone-elses-home" / "checkout")
    assert portable_cmd(f"cd {gone} && uv run python x.py --a 1", here) == (
        f"cd {here} && uv run python x.py --a 1"
    )
    assert portable_cmd(f"cd '{gone}' && echo hi", here) == f"cd {here} && echo hi"
    # reachable dir: untouched (explicit cd wins over the pool default)
    assert portable_cmd(f"cd {here} && echo hi", "/elsewhere") == f"cd {here} && echo hi"
    # no leading cd / cd not first: untouched
    assert portable_cmd("echo hi && cd /nope", here) == "echo hi && cd /nope"
    assert portable_cmd("uv run python x.py", here) == "uv run python x.py"
    # cwd with spaces is quoted
    sp = tmp_path / "a b"
    sp.mkdir()
    assert portable_cmd(f"cd {gone} && true", str(sp)) == f"cd '{sp}' && true"


def test_portable_cwd_falls_back_only_when_unreachable(tmp_path):
    # 2026-09-09: goncalo-temp-1 ran a day of cwd=/home/angela/... jobs into PermissionError
    # (another user's 0750 home) — Popen(cwd=) fails before the leading-cd rewrite can help.
    from subliminal_learning.jobqueue.worker import portable_cwd

    good = tmp_path / "checkout"
    good.mkdir()
    assert portable_cwd(str(good), "/fallback") == str(good)
    assert portable_cwd(str(tmp_path / "nope"), "/fallback") == "/fallback"
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        import os

        if os.geteuid() != 0:  # root can traverse anything; the check is meaningless there
            assert portable_cwd(str(locked), "/fallback") == "/fallback"
    finally:
        locked.chmod(0o755)


# ------------------------- drained pool exits without respawning -------------------------
# 2026-09-10: all-complete and the acquire-wait stop both returned "exit" from _loop_once,
# so every thread of a drained pool reported "stop", was respawned at once by the
# supervisor, and exited again — five RESPAWN/WAIT cycles per thread on an empty root,
# each WAIT line falsely naming a stop file.


class _AlwaysSlotGpu:
    def acquire(self, mem_mib, *, slots=1, cap_per_gpu=None, cap_group=None):  # noqa: ARG002
        from subliminal_learning.jobqueue.gpu import GpuSlot

        return GpuSlot(gpu=0, slot=0)

    def env(self):
        return {}

    def default_mem_mib(self):
        return 1000


def test_empty_root_exits_on_complete_without_respawning(tmp_path):
    from subliminal_learning.jobqueue.worker import run_pool

    run_pool(
        tmp_path,
        workers=2,
        gpu=_AlwaysSlotGpu(),
        hostname="testnode",
        poll_s=0.01,
        gpu_wait_s=0.01,
        supervise_s=0.05,
    )

    logs = sorted((tmp_path / "logs").glob("*_worker.testnode.log"))
    assert len(logs) == 1
    txt = logs[0].read_text()
    assert "RESPAWN" not in txt
    assert "stop file seen in acquire wait" not in txt
    waits = [ln for ln in txt.splitlines() if " WAIT " in ln]
    assert waits and all("all queues complete" in ln for ln in waits)
    assert txt.count("ALL QUEUES COMPLETE") == 1


def test_supervisor_respawns_a_stop_exit_once_the_stop_file_is_cleared(tmp_path):
    import time as _t

    from subliminal_learning.jobqueue import worker as W

    class Dummy:
        def __init__(self):
            self.calls = 0
            self.worker_target = 0

        def loop(self, tid):  # noqa: ARG002
            self.calls += 1
            if self.calls == 1:
                return "stop"  # transient stop file, already cleared
            stopper.wait()
            return "complete"

    stopper = threading.Event()
    pool = Dummy()
    pool.gpu = _AlwaysSlotGpu()
    pool.reap_orphan_claims = lambda: None
    master = W.MasterLog(tmp_path / "master.log")
    sup = threading.Thread(
        target=W._supervise_workers, args=(pool, master, tmp_path, "h", 1, 0.05), daemon=True
    )
    sup.start()
    for _ in range(100):
        if pool.calls >= 2:
            break
        _t.sleep(0.05)
    assert pool.calls >= 2
    assert "RESPAWN worker 0 left on stop; stop cleared, respawning" in (
        tmp_path / "master.log"
    ).read_text()
    stopper.set()
    sup.join(timeout=5)
    assert not sup.is_alive()
