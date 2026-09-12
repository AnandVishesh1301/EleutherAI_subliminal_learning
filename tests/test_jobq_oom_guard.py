"""OOM-targeted requeue/escalation + the self-healing (half-open) failure breaker.

GPU-free: the pool runs against a fake GPU manager, ``query_total_mem`` is patched, and a
"job log" is just a file with OOM text in it.
"""

from __future__ import annotations

import json
import threading

from subliminal_learning.jobqueue import gpu as gpu_mod
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.gpu import GpuSlot
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.jobqueue.worker import (
    OOM_CEILING_HEADROOM_MIB,
    MasterLog,
    _Pool,
    is_oom_text,
    log_tail_is_oom,
)


class FakeGpu:
    """Hands out a slot unconditionally; ``ceiling`` of 0 means "ceiling unknown"."""

    def __init__(self, mem_mib=1000, ceiling=0):
        self._mem = mem_mib
        self._ceiling = ceiling

    def acquire(self, mem_mib, *, slots=1, cap_per_gpu=None, cap_group=None):  # noqa: ARG002
        return GpuSlot(gpu=0, slot=0)

    def env(self):
        return {}

    def default_mem_mib(self):
        return self._mem

    def mem_ceiling_mib(self):
        return self._ceiling


def _pool(tmp_path, *, gpu=None, limit=5, cooldown=900.0):
    return _Pool(
        tmp_path,
        "testnode",
        gpu or FakeGpu(),
        MasterLog(tmp_path / "logs" / "master.log"),
        queues_filter=None,
        poll_s=0.01,
        gpu_wait_s=0.01,
        fail_streak_limit=limit,
        breaker_cooldown_s=cooldown,
    )


def _queue_with_job(tmp_path, cmd, *, name="q", key="j1", **jobkw):
    store.create_queue(tmp_path, name)
    job = Job(key=key, cmd=cmd, **jobkw)
    store.append_jobs(tmp_path, name, [job])
    return job


def _run_one(pool, tmp_path, name, job, meta=None):
    """Claim + run one job through the real ``_run_claimed`` path; returns nothing."""
    meta = meta or store.read_meta(tmp_path, name)
    claimed = store.claim_next(tmp_path, name, "testnode", 1234)
    assert claimed is not None and claimed.jobkey == job.jobkey
    pool._run_claimed(name, meta, claimed, GpuSlot(gpu=0, slot=0))


def _master(tmp_path):
    return (tmp_path / "logs" / "master.log").read_text()


# ------------------------- classifier -------------------------


def test_classifier_matches_all_four_markers_and_rejects_other_text():
    assert is_oom_text("RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert is_oom_text("torch.OutOfMemoryError: CUDA out of mem")
    assert is_oom_text("RuntimeError: CUDA error: out of memory")
    assert is_oom_text("cudaErrorMemoryAllocation")
    assert not is_oom_text("ValueError: bad config\nTraceback (most recent call last)")


def test_log_tail_classifier_reads_only_the_tail_and_tolerates_a_missing_file(tmp_path):
    p = tmp_path / "job.log"
    p.write_text("CUDA out of memory\n" + "x" * 200_000)
    assert not log_tail_is_oom(p)  # marker scrolled out of the tail window
    p.write_text("x" * 200_000 + "\nCUDA out of memory\n")
    assert log_tail_is_oom(p)
    assert not log_tail_is_oom(tmp_path / "nope.log")


# ------------------------- OOM requeue + escalation -------------------------


OOM_CMD = "echo 'RuntimeError: CUDA out of memory.' >&2; exit 1"


def test_oom_job_is_requeued_with_an_escalated_floor_and_no_failure(tmp_path):
    job = _queue_with_job(tmp_path, OOM_CMD)
    pool = _pool(tmp_path)

    _run_one(pool, tmp_path, "q", job)

    st = store.queue_status(tmp_path, "q")
    assert st["failed"] == 0 and st["done"] == 0 and st["pending"] == 1
    assert not store.has_result(tmp_path, "q", job.jobkey)
    assert store.read_attempt(tmp_path, "q", job.jobkey) == 1
    assert pool._fail_streak.get("q", 0) == 0
    # default_mem_mib 1000 -> max(ceil(1000 * 1.5), 4096) = 4096
    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 4096
    assert "OOM REQUEUE q j1 mem_mib 1000 -> 4096" in _master(tmp_path)


def test_escalation_compounds_and_is_read_back_as_the_effective_request(tmp_path):
    job = _queue_with_job(tmp_path, OOM_CMD)
    pool = _pool(tmp_path)
    meta = store.read_meta(tmp_path, "q")

    _run_one(pool, tmp_path, "q", job, meta)
    assert pool._effective_mem("q", meta, job) == 4096  # max(static 1000, floor)
    _run_one(pool, tmp_path, "q", job, meta)

    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 6144  # ceil(4096 * 1.5)
    assert pool._effective_mem("q", meta, job) == 6144
    assert store.read_attempt(tmp_path, "q", job.jobkey) == 2
    assert store.queue_status(tmp_path, "q")["pending"] == 1


def test_static_request_above_the_floor_wins(tmp_path):
    job = _queue_with_job(tmp_path, OOM_CMD, mem_mib=20000)
    pool = _pool(tmp_path)
    meta = store.read_meta(tmp_path, "q")
    store.write_mem_floor(tmp_path, "q", job.jobkey, 4096)
    assert pool._effective_mem("q", meta, job) == 20000


def test_oom_at_the_card_ceiling_is_recorded_as_a_normal_failure(tmp_path):
    job = _queue_with_job(tmp_path, OOM_CMD, mem_mib=40000)
    pool = _pool(tmp_path, gpu=FakeGpu(ceiling=40000))

    _run_one(pool, tmp_path, "q", job)

    st = store.queue_status(tmp_path, "q")
    assert st["failed"] == 1 and st["pending"] == 0
    assert pool._fail_streak["q"] == 1
    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 0
    assert "OOM CEILING" in _master(tmp_path)


def test_escalation_is_clamped_to_the_attainable_ceiling_not_the_card_total(tmp_path):
    """An idle card never grants its TOTAL, so a floor AT the total parks the job forever."""
    job = _queue_with_job(tmp_path, OOM_CMD, mem_mib=30000)
    pool = _pool(tmp_path, gpu=FakeGpu(ceiling=40000))

    _run_one(pool, tmp_path, "q", job)

    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 40000 - OOM_CEILING_HEADROOM_MIB
    assert store.queue_status(tmp_path, "q")["pending"] == 1


def test_oom_just_below_the_card_total_is_a_failure_not_another_floor(tmp_path):
    """At/above the attainable ceiling the ladder ends: never persist an ungrantable floor."""
    job = _queue_with_job(tmp_path, OOM_CMD, mem_mib=40000 - OOM_CEILING_HEADROOM_MIB)
    pool = _pool(tmp_path, gpu=FakeGpu(ceiling=40000))

    _run_one(pool, tmp_path, "q", job)

    st = store.queue_status(tmp_path, "q")
    assert st["failed"] == 1 and st["pending"] == 0
    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 0
    assert f"attainable card memory {40000 - OOM_CEILING_HEADROOM_MIB}" in _master(tmp_path)


def test_non_oom_failure_keeps_the_existing_behavior(tmp_path):
    job = _queue_with_job(tmp_path, "echo 'ValueError: bad config' >&2; exit 3")
    pool = _pool(tmp_path)

    _run_one(pool, tmp_path, "q", job)

    st = store.queue_status(tmp_path, "q")
    assert st["failed"] == 1 and st["pending"] == 0
    assert pool._fail_streak["q"] == 1
    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 0
    assert "OOM REQUEUE" not in _master(tmp_path)


def test_missing_log_file_is_not_an_oom(tmp_path):
    """A vanished/unwritten log must fall through to normal failure handling."""
    job = _queue_with_job(tmp_path, "exit 1")
    pool = _pool(tmp_path)
    meta = store.read_meta(tmp_path, "q")

    assert pool._oom_requeue("q", meta, job, tmp_path / "gone.log", []) is False
    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 0


def test_oom_backstop_ends_an_unbounded_escalation(tmp_path):
    """Ceiling unknown (broken smi ⇒ FakeGpu ceiling 0): the memory-gate ladder cannot
    terminate, which previously requeued forever. The backstop counter must convert
    requeue #(BACKSTOP+1) into a real failure."""
    from subliminal_learning.jobqueue.worker import OOM_REQUEUE_BACKSTOP

    job = _queue_with_job(tmp_path, OOM_CMD)
    pool = _pool(tmp_path)
    meta = store.read_meta(tmp_path, "q")

    for i in range(OOM_REQUEUE_BACKSTOP):
        _run_one(pool, tmp_path, "q", job, meta)
        assert store.queue_status(tmp_path, "q")["pending"] == 1  # still requeued
        assert store.read_oom_requeues(tmp_path, "q", job.jobkey) == i + 1

    _run_one(pool, tmp_path, "q", job, meta)

    st = store.queue_status(tmp_path, "q")
    assert st["failed"] == 1 and st["pending"] == 0
    assert pool._fail_streak["q"] == 1
    assert "OOM BACKSTOP" in _master(tmp_path)


def test_attempts_record_without_the_new_key_still_reads(tmp_path):
    store.create_queue(tmp_path, "q")
    p = tmp_path / "q" / "attempts" / "j1.json"
    store.mkdir_open(p.parent)
    p.write_text(json.dumps({"attempt": 3}))  # written by a pre-escalation worker

    assert store.read_attempt(tmp_path, "q", "j1") == 3
    assert store.read_mem_floor(tmp_path, "q", "j1") == 0

    store.write_mem_floor(tmp_path, "q", "j1", 8192)
    assert store.read_attempt(tmp_path, "q", "j1") == 3  # attempt preserved
    assert store.bump_attempt(tmp_path, "q", "j1") == 4
    assert store.read_mem_floor(tmp_path, "q", "j1") == 8192  # floor preserved


# ------------------------- memory ceiling query -------------------------


def test_total_mem_is_cached_and_the_manager_takes_the_max_policy_card(tmp_path, monkeypatch):
    calls = []

    def fake_total():
        calls.append(1)
        return {0: 40000, 1: 80000, 2: 24000}

    monkeypatch.setattr(gpu_mod, "query_total_mem", fake_total)
    gpu_mod.clear_total_mem_cache()
    (tmp_path / "gpu_policy.testnode.json").write_text(
        json.dumps({"gpus": [0, 2], "cap_per_gpu": 2, "free_mem_mib": 500})
    )
    gm = gpu_mod.GpuManager(tmp_path, "testnode", lock_prefix=str(tmp_path / "lk"))

    assert gm.mem_ceiling_mib() == 40000
    assert gm.mem_ceiling_mib() == 40000
    assert len(calls) == 1  # cached: totals cannot change under us
    gpu_mod.clear_total_mem_cache()


def test_missing_policy_means_no_ceiling(tmp_path):
    gm = gpu_mod.GpuManager(tmp_path, "nosuchhost", lock_prefix=str(tmp_path / "lk"))
    assert gm.mem_ceiling_mib() == 0


# ------------------------- half-open breaker -------------------------


def _trip(pool, tmp_path, name, limit):
    for i in range(limit):
        store.record_result(
            tmp_path,
            name,
            Job(f"boom{i}", "exit 1"),
            rc=1,
            node="testnode",
            gpu=0,
            start_utc="s",
            end_utc="e",
            log="l",
            attempt=0,
        )
        pool._record_outcome(name, 1)


def test_half_open_probe_success_readmits_failure_retrips(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job(f"boom{i}", "exit 1") for i in range(10)])
    pool = _pool(tmp_path, limit=5, cooldown=100.0)
    now = [1000.0]
    pool._clock = lambda: now[0]

    _trip(pool, tmp_path, "q", 5)
    assert pool.ready_queues() == []  # closed during the cooldown

    now[0] += 100.0  # cooldown elapsed -> half-open
    assert [n for n, _ in pool.ready_queues()] == ["q"]
    assert pool._probe_admit("q") is True
    assert pool._probe_admit("q") is False  # exactly ONE probe in flight
    assert pool.ready_queues() == []  # and the queue is closed again meanwhile

    pool._probe_finish("q", 1)  # probe failed -> re-trip, fresh cooldown
    assert pool.ready_queues() == []
    now[0] += 99.0
    assert pool.ready_queues() == []
    now[0] += 1.0
    assert [n for n, _ in pool.ready_queues()] == ["q"]

    assert pool._probe_admit("q") is True
    pool._probe_finish("q", 0)  # probe succeeded -> fully un-tripped
    assert pool._tripped_pending == {} and pool._fail_streak["q"] == 0
    assert [n for n, _ in pool.ready_queues()] == ["q"]
    log = _master(tmp_path)
    assert "q re-admitted after successful probe" in log
    assert "q probe failed rc=1" in log


def test_only_one_thread_wins_the_probe_slot(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job(f"boom{i}", "exit 1") for i in range(10)])
    pool = _pool(tmp_path, limit=5, cooldown=1.0)
    now = [10_000.0]
    pool._clock = lambda: now[0]
    _trip(pool, tmp_path, "q", 5)
    now[0] += 1.0  # cooldown elapsed -> half-open

    wins: list[bool] = []
    lock = threading.Lock()

    def probe():
        got = pool._probe_admit("q")
        with lock:
            wins.append(got)

    threads = [threading.Thread(target=probe) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(wins) == 1


def test_oom_probe_leaves_the_queue_half_open_pending(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("j1", OOM_CMD)])
    pool = _pool(tmp_path, limit=5, cooldown=1.0)
    now = [10_000.0]
    pool._clock = lambda: now[0]
    _trip(pool, tmp_path, "q", 5)
    now[0] += 1.0  # cooldown elapsed -> half-open

    assert pool._probe_admit("q") is True
    pool._probe_finish("q", None)  # OOM requeue settles nothing
    assert "q" in pool._tripped_pending  # still tripped...
    assert pool._probe_admit("q") is True  # ...but the probe slot was released


def test_cooldown_zero_disables_self_healing(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job(f"boom{i}", "exit 1") for i in range(10)])
    pool = _pool(tmp_path, limit=3, cooldown=0.0)
    now = [0.0]
    pool._clock = lambda: now[0]
    _trip(pool, tmp_path, "q", 3)

    now[0] += 10_000.0
    assert pool.ready_queues() == []
    assert pool._probe_admit("q") is False


# ------------------------- operator OOM reset -------------------------


def _failed_job_with_oom_state(tmp_path):
    """A failed job whose attempts sidecar carries both an OOM counter and a mem floor."""
    job = _queue_with_job(tmp_path, "true")
    store.record_result(
        tmp_path,
        "q",
        job,
        rc=1,
        node="testnode",
        gpu=0,
        start_utc="s",
        end_utc="e",
        log="l",
        attempt=2,
    )
    store.write_mem_floor(tmp_path, "q", job.jobkey, 8192)
    store.bump_oom_requeues(tmp_path, "q", job.jobkey)
    store.bump_oom_requeues(tmp_path, "q", job.jobkey)
    return job


def test_requeue_preserves_the_oom_counter_and_floor_by_default(tmp_path):
    job = _failed_job_with_oom_state(tmp_path)

    assert store.requeue(tmp_path, "q") == [job.jobkey]

    assert store.read_oom_requeues(tmp_path, "q", job.jobkey) == 2
    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 8192
    assert store.read_attempt(tmp_path, "q", job.jobkey) == 3


def test_reset_oom_clears_only_the_counter(tmp_path):
    job = _failed_job_with_oom_state(tmp_path)

    assert store.requeue(tmp_path, "q", reset_oom=True) == [job.jobkey]

    assert store.read_oom_requeues(tmp_path, "q", job.jobkey) == 0
    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 8192  # floor kept
    assert store.read_attempt(tmp_path, "q", job.jobkey) == 3


def test_reset_mem_floor_clears_only_the_floor(tmp_path):
    job = _failed_job_with_oom_state(tmp_path)

    assert store.requeue(tmp_path, "q", reset_mem_floor=True) == [job.jobkey]

    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 0
    assert store.read_oom_requeues(tmp_path, "q", job.jobkey) == 2  # counter kept
    assert store.read_attempt(tmp_path, "q", job.jobkey) == 3


# ------------------------- co-tenant OOM: floor kept -------------------------
# 2026-09-09: three co-tenant OOMs (48 MiB wanted, 23 MiB free, own footprint 1.5 GiB)
# escalated one job's floor 14000 -> 47250 MiB and parked its claim for 10 h+.

REAL_OOM_TAIL = (
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 MiB. GPU 0 has a "
    "total capacity of 47.54 GiB of which 23.38 MiB is free. Process 3081276 has 33.69 GiB "
    "memory in use. Of the allocated memory 1.46 GiB is allocated by PyTorch, and 1.93 MiB "
    "is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try "
    "setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
)


def test_own_usage_parser_reads_both_message_shapes_and_takes_the_last():
    from subliminal_learning.jobqueue.worker import oom_own_usage_mib

    assert oom_own_usage_mib(REAL_OOM_TAIL) == 1497  # 1.46 GiB + 1.93 MiB
    assert oom_own_usage_mib(
        "Including non-PyTorch memory, this process has 5.13 GiB memory in use. Of the "
        "allocated memory 4.00 GiB is allocated by PyTorch, and 0.50 GiB is reserved by "
        "PyTorch but unallocated."
    ) == 5253  # explicit figure wins over allocated+reserved
    two = REAL_OOM_TAIL + "\nOf the allocated memory 900.00 MiB is allocated by PyTorch, and 100.00 MiB is reserved by PyTorch but unallocated."
    assert oom_own_usage_mib(two) == 1000  # last message is the fatal one
    assert oom_own_usage_mib("CUDA out of memory. Tried to allocate 2.00 GiB") is None


def _oom_cmd(tail: str) -> str:
    return f"printf '%s\\n' {json.dumps(tail)} >&2; exit 1"


def test_cotenant_oom_is_requeued_at_the_same_reservation(tmp_path):
    job = _queue_with_job(tmp_path, _oom_cmd(REAL_OOM_TAIL))
    pool = _pool(tmp_path, gpu=FakeGpu(mem_mib=14000))

    _run_one(pool, tmp_path, "q", job)

    st = store.queue_status(tmp_path, "q")
    assert st["failed"] == 0 and st["pending"] == 1  # still a requeue, not a failure
    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 0  # floor NOT escalated
    assert store.read_oom_requeues(tmp_path, "q", job.jobkey) == 1  # backstop still counts
    assert store.read_attempt(tmp_path, "q", job.jobkey) == 1
    assert pool._effective_mem("q", store.read_meta(tmp_path, "q"), job) == 14000
    log = _master(tmp_path)
    assert "OOM REQUEUE q j1 co-tenant pressure: process held 1497 MiB of its 14000 MiB reservation (wanted 48 MiB); floor kept (1/8)" in log


def test_oom_near_its_own_reservation_still_escalates(tmp_path):
    # 1497 MiB held of a 1000 MiB reservation -> the job genuinely outgrew it.
    job = _queue_with_job(tmp_path, _oom_cmd(REAL_OOM_TAIL))
    pool = _pool(tmp_path, gpu=FakeGpu(mem_mib=1000))

    _run_one(pool, tmp_path, "q", job)

    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 4096
    assert "OOM REQUEUE q j1 mem_mib 1000 -> 4096 (wanted 48 MiB) (1/8)" in _master(tmp_path)


def test_unparseable_oom_message_escalates_as_before(tmp_path):
    job = _queue_with_job(tmp_path, OOM_CMD)
    pool = _pool(tmp_path, gpu=FakeGpu(mem_mib=14000))

    _run_one(pool, tmp_path, "q", job)

    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 21000


def test_cotenant_requeues_still_hit_the_backstop(tmp_path):
    from subliminal_learning.jobqueue.worker import OOM_REQUEUE_BACKSTOP

    job = _queue_with_job(tmp_path, _oom_cmd(REAL_OOM_TAIL))
    pool = _pool(tmp_path, gpu=FakeGpu(mem_mib=14000))
    for _ in range(OOM_REQUEUE_BACKSTOP):
        _run_one(pool, tmp_path, "q", job)
        assert store.queue_status(tmp_path, "q")["pending"] == 1
    _run_one(pool, tmp_path, "q", job)
    assert store.queue_status(tmp_path, "q")["failed"] == 1
    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 0
    assert "OOM BACKSTOP q j1" in _master(tmp_path)


# ------------------------- acquire-wait exit is always "stop" -------------------------


def test_acquire_wait_exit_returns_stop_even_if_the_stop_file_is_already_gone(tmp_path, monkeypatch):
    # 2026-09-09: a 2 s stop window; the thread parked in _acquire_gpu saw the file, dropped
    # its claim and returned "exit", but by the loop head the file was gone -> "complete"
    # -> the supervisor never respawned it (it only respawns "stop"/"scale").
    pool = _pool(tmp_path)
    monkeypatch.setattr(pool, "_loop_once", lambda: "exit")
    assert not store.stop_path(tmp_path, "testnode").exists()
    assert pool.loop(7) == "stop"
    assert "stop file seen in acquire wait; worker 7 exiting" in _master(tmp_path)


# ------------------------- requested-allocation size decides genuine vs co-tenant -------
# 2026-09-10: the fraction alone misread an under-reserved job (5 GiB held of 16 GiB, asking
# for 20 GiB) as co-tenant pressure and retried it eight times at the same reservation.


def _tail(held_gib: float, want_gib: float) -> str:
    return (
        f"torch.OutOfMemoryError: CUDA out of memory. Tried to allocate {want_gib:.2f} GiB "
        "(GPU 0; 47.54 GiB total capacity). Including non-PyTorch memory, this process has "
        f"{held_gib:.2f} GiB memory in use."
    )


def test_requested_allocation_parser_takes_the_last_match_and_all_units():
    from subliminal_learning.jobqueue.worker import oom_requested_mib

    assert oom_requested_mib(REAL_OOM_TAIL) == 48
    assert oom_requested_mib("Tried to allocate 2.00 GiB") == 2048
    assert oom_requested_mib("Tried to allocate 512.00 KiB") == 0
    assert oom_requested_mib("Tried to allocate 1.00 GiB\nTried to allocate 3.00 GiB") == 3072
    assert oom_requested_mib("CUDA out of memory") is None


def test_small_footprint_but_huge_request_is_genuine_and_escalates(tmp_path):
    job = _queue_with_job(tmp_path, _oom_cmd(_tail(5.0, 20.0)))
    pool = _pool(tmp_path, gpu=FakeGpu(mem_mib=16000))

    _run_one(pool, tmp_path, "q", job)

    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 24000
    assert "OOM REQUEUE q j1 mem_mib 16000 -> 24000 (wanted 20480 MiB) (1/8)" in _master(tmp_path)


def test_held_plus_request_over_the_reservation_escalates(tmp_path):
    # 10 GiB held of 14000 MiB (below the 75% fraction) but a 6 GiB ask -> genuinely short.
    job = _queue_with_job(tmp_path, _oom_cmd(_tail(10.0, 6.0)))
    pool = _pool(tmp_path, gpu=FakeGpu(mem_mib=14000))

    _run_one(pool, tmp_path, "q", job)

    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 21000


def test_held_plus_request_inside_the_reservation_stays_cotenant(tmp_path):
    job = _queue_with_job(tmp_path, _oom_cmd(_tail(2.0, 1.0)))
    pool = _pool(tmp_path, gpu=FakeGpu(mem_mib=16000))

    _run_one(pool, tmp_path, "q", job)

    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 0  # floor kept
    assert (
        "OOM REQUEUE q j1 co-tenant pressure: process held 2048 MiB of its 16000 MiB "
        "reservation (wanted 1024 MiB); floor kept (1/8)"
    ) in _master(tmp_path)


def test_unparseable_request_keeps_the_fraction_only_behavior(tmp_path):
    # No "Tried to allocate" line: 1497 MiB held of 14000 -> co-tenant, floor kept.
    tail = REAL_OOM_TAIL.replace("Tried to allocate 48.00 MiB. ", "")
    job = _queue_with_job(tmp_path, _oom_cmd(tail))
    pool = _pool(tmp_path, gpu=FakeGpu(mem_mib=14000))

    _run_one(pool, tmp_path, "q", job)

    assert store.read_mem_floor(tmp_path, "q", job.jobkey) == 0
    assert (
        "OOM REQUEUE q j1 co-tenant pressure: process held 1497 MiB of its 14000 MiB "
        "reservation; floor kept (1/8)"
    ) in _master(tmp_path)
