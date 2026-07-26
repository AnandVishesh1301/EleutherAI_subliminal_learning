"""Unit tests for ``subliminal_learning.jobqueue`` (CPU-only, no nvidia-smi).

The GPU layer is exercised through a fake manager (or a real :class:`GpuManager` with a
patched ``mem_query`` + tmp lock prefix), so nothing here touches a real GPU. All queue
state lives under ``tmp_path`` — the tests never assume ``run_logs/queue`` exists.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.gpu import GpuManager, GpuSlot
from subliminal_learning.jobqueue.model import Job, parse_jobs_file, sanitize_jobkey
from subliminal_learning.jobqueue.worker import run_pool


# ------------------------- fakes -------------------------


class FakeGpuManager:
    """A GPU manager that always hands out a slot on a rotating physical index."""

    def __init__(self, gpus=(0, 1), mem_mib=1000):
        self._gpus = list(gpus)
        self._i = 0
        self._lock = threading.Lock()
        self._mem = mem_mib

    def acquire(self, mem_mib, *, slots=1, cap_per_gpu=None, cap_group=None):  # noqa: ARG002 — fake ignores gates
        with self._lock:
            g = self._gpus[self._i % len(self._gpus)]
            self._i += 1
        return GpuSlot(gpu=g, slot=0)

    def env(self):
        return {}

    def default_mem_mib(self):
        return self._mem


def _write_policy(root: Path, hostname: str, **over):
    d = {"gpus": [0, 1], "cap_per_gpu": 2, "free_mem_mib": 500, "env": {}}
    d.update(over)
    p = root / f"gpu_policy.{hostname}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))


# ------------------------- submit / expansion -------------------------


def test_parse_plain_lines_with_template_and_field_split():
    text = "owl|pythia-160m|2e-4\ncat|pythia-160m|5e-4\n"
    jobs = parse_jobs_file(text, template="train --animal {0} --base {1} --lr {2} # {line}")
    assert [j.cmd for j in jobs] == [
        "train --animal owl --base pythia-160m --lr 2e-4 # owl|pythia-160m|2e-4",
        "train --animal cat --base pythia-160m --lr 5e-4 # cat|pythia-160m|5e-4",
    ]
    assert jobs[0].key == "owl|pythia-160m|2e-4"


def test_parse_template_with_missing_field_raises_clear_error():
    from subliminal_learning.jobqueue.model import parse_jobs_file

    with pytest.raises(ValueError, match=r"2 '\|'-field\(s\).*\{0\}\.\.\{1\}"):
        parse_jobs_file("tag|base\n", template="run --a {0} --b {1} --c {6}")


def test_update_owner_gpu_visible_in_status(tmp_path):
    from subliminal_learning.jobqueue.model import Job

    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job(key="j1", cmd="true")])
    job = store.claim_next(tmp_path, "q", "nodeA", 12345)
    st = store.queue_status(tmp_path, "q")
    assert st["running_jobs"][0]["gpu"] is None  # claimed, GPU not yet acquired
    store.update_owner_gpu(tmp_path, "q", job.jobkey, 5)
    st = store.queue_status(tmp_path, "q")
    r = st["running_jobs"][0]
    assert r["gpu"] == 5 and r["node"] == "nodeA" and r["start_utc"]
    store.update_owner_gpu(tmp_path, "q", "no_such_claim", 3)  # best-effort no-op


def test_parse_plain_line_without_template_errors():
    with pytest.raises(ValueError):
        parse_jobs_file("owl|base\n", template=None)


def test_parse_jsonl_passthrough_and_comments_skipped():
    text = '# a comment\n\n{"key": "j1", "cmd": "echo hi", "mem_mib": 42}\n'
    jobs = parse_jobs_file(text)
    assert len(jobs) == 1
    assert jobs[0].key == "j1" and jobs[0].cmd == "echo hi" and jobs[0].mem_mib == 42


def test_jsonl_key_defaults_to_line():
    jobs = parse_jobs_file('{"cmd": "echo hi"}')
    assert jobs[0].key == '{"cmd": "echo hi"}'


def test_submit_appends_and_refuses_duplicate_keys(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("a", "echo a"), Job("b", "echo b")])
    store.append_jobs(tmp_path, "q", [Job("c", "echo c")])  # append to existing queue
    assert [j.key for j in store.load_jobs(tmp_path, "q")] == ["a", "b", "c"]
    with pytest.raises(ValueError):
        store.append_jobs(tmp_path, "q", [Job("b", "echo dup")])


def test_create_queue_exist_ok_keeps_original_meta(tmp_path):
    store.create_queue(tmp_path, "q", priority=5)
    with pytest.raises(FileExistsError):
        store.create_queue(tmp_path, "q")
    meta = store.create_queue(tmp_path, "q", priority=9, exist_ok=True)
    assert meta.priority == 5  # original meta kept


# ------------------------- key sanitization -------------------------


def test_key_sanitization_roundtrip():
    key = "TAG|pythia-160m|2e-4|r8|a8|ep2|seed1"
    jk = sanitize_jobkey(key)
    assert "|" not in jk and "/" not in jk and " " not in jk
    assert jk == "TAG__pythia-160m__2e-4__r8__a8__ep2__seed1"
    # idempotent + survives round-trip through a Job written and reloaded.
    assert sanitize_jobkey(jk) == jk
    store_key = Job(key=key, cmd="x").key
    assert store_key == key and Job(key=key, cmd="x").jobkey == jk


def test_sanitize_collapses_whitespace_and_slashes():
    assert sanitize_jobkey("a / b\tc") == "a__b__c"


# ------------------------- claim atomicity -------------------------


def test_claim_next_never_returns_same_job_twice(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("a", "echo a"), Job("b", "echo b")])
    j1 = store.claim_next(tmp_path, "q", "hostA", 111)
    j2 = store.claim_next(tmp_path, "q", "hostA", 222)
    assert {j1.key, j2.key} == {"a", "b"}
    assert store.claim_next(tmp_path, "q", "hostA", 333) is None


def test_claim_skips_already_claimed_and_terminal(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("a", "echo a"), Job("b", "echo b")])
    a = store.claim_next(tmp_path, "q", "hostA", 111)
    # record a result for a -> terminal; next claim must move on to b.
    store.record_result(
        tmp_path, "q", a, rc=0, node="hostA", gpu=0,
        start_utc="s", end_utc="e", log="l", attempt=0,
    )
    store.remove_claim(tmp_path, "q", a.jobkey)
    b = store.claim_next(tmp_path, "q", "hostA", 222)
    assert b.key == "b"


# ------------------------- derived state / status -------------------------


def test_claim_rechecks_result_after_winning_mkdir(tmp_path, monkeypatch):
    # TOCTOU: a finishing worker can land the result + drop its claim between our
    # has_result check and our mkdir; the post-mkdir re-check must not re-run the job.
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("j", "true")])
    calls = {"n": 0}

    def racing_has_result(root, name, jk):
        calls["n"] += 1
        return calls["n"] > 1  # pre-mkdir: no result yet; post-mkdir: result appeared

    monkeypatch.setattr(store, "has_result", racing_has_result)
    assert store.claim_next(tmp_path, "q", "n", 1) is None
    assert not (tmp_path / "q" / "claims" / "j").exists()  # claim rolled back


def test_ownerless_claim_reclaimed_after_grace_any_node(tmp_path):
    import time as _time

    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("j", "true")])
    cdir = tmp_path / "q" / "claims" / "j"
    cdir.mkdir(parents=True)  # crash debris: claim dir, no owner.json
    assert store.steal_stale(tmp_path, "q", "some-other-node") == []  # fresh: grace period
    old = _time.time() - store.ORPHAN_CLAIM_GRACE_S - 60
    os.utime(cdir, (old, old))
    assert store.steal_stale(tmp_path, "q", "some-other-node") == ["j"]  # any node may
    assert not cdir.exists()


def test_queue_state_dirs_are_world_writable(tmp_path):
    import stat

    old_umask = os.umask(0o022)  # a restrictive umask must not close the dirs
    try:
        store.create_queue(tmp_path / "root", "q")
        store.append_jobs(tmp_path / "root", "q", [Job("j", "true")])
        job = store.claim_next(tmp_path / "root", "q", "n", 1)
        store.record_result(tmp_path / "root", "q", job, rc=0, node="n", gpu=0,
                            start_utc="s", end_utc="e", log="l", attempt=0)
    finally:
        os.umask(old_umask)
    # other-node access rides on the 'other' bits (different uids, no shared group)
    for rel in ("", "q", "q/claims", "q/results", "q/attempts"):
        mode = (tmp_path / "root" / rel).stat().st_mode
        assert mode & stat.S_IWOTH, f"{rel or 'root'} not world-writable"


def test_check_new_jobs_validates_without_creating_queue(tmp_path):
    jobs = [Job("a", "true"), Job("a", "true")]
    with pytest.raises(ValueError, match="duplicate"):
        store.check_new_jobs(tmp_path, "newq", jobs)
    # submit validates first, so a rejected batch leaves no meta-only queue behind
    assert not store.queue_exists(tmp_path, "newq")


def test_strict_deps_blocks_on_failed_dep(tmp_path):
    from subliminal_learning.jobqueue.worker import MasterLog, _Pool

    store.create_queue(tmp_path, "dep")
    store.append_jobs(tmp_path, "dep", [Job("a", "x")])
    a = store.claim_next(tmp_path, "dep", store.this_host(), 1)
    store.record_result(tmp_path, "dep", a, rc=7, node=store.this_host(), gpu=0,
                        start_utc="s", end_utc="e", log="l", attempt=0)
    store.remove_claim(tmp_path, "dep", a.jobkey)
    store.create_queue(tmp_path, "strict", depends_on=["dep"], strict_deps=True)
    store.append_jobs(tmp_path, "strict", [Job("m", "x")])
    store.create_queue(tmp_path, "lenient", depends_on=["dep"])
    store.append_jobs(tmp_path, "lenient", [Job("m", "x")])
    pool = _Pool(
        tmp_path, store.this_host(), FakeGpuManager(),
        MasterLog(tmp_path / "logs" / "m.log"),
        queues_filter=None, poll_s=0.01, gpu_wait_s=0.01,
    )
    names = _ready_names(pool)
    assert "lenient" in names and "strict" not in names
    store.requeue(tmp_path, "dep")  # failed job pending again -> dep incomplete for both
    assert _ready_names(pool) == ["dep"]  # only dep itself has runnable work now
    b = store.claim_next(tmp_path, "dep", store.this_host(), 1)
    store.record_result(tmp_path, "dep", b, rc=0, node=store.this_host(), gpu=0,
                        start_utc="s", end_utc="e", log="l", attempt=1)
    store.remove_claim(tmp_path, "dep", b.jobkey)
    assert set(_ready_names(pool)) == {"lenient", "strict"}  # all-green dep unblocks strict


def test_derived_states_and_status_counts(tmp_path):
    store.create_queue(tmp_path, "q")
    jobs = [Job("a", "x"), Job("b", "x"), Job("c", "x"), Job("d", "x")]
    store.append_jobs(tmp_path, "q", jobs)
    # a: done, b: failed, c: running (claimed), d: pending
    ja = store.claim_next(tmp_path, "q", "hostA", 1)
    store.record_result(tmp_path, "q", ja, rc=0, node="hostA", gpu=0,
                        start_utc="s", end_utc="e", log="l", attempt=0)
    store.remove_claim(tmp_path, "q", ja.jobkey)
    jb = store.claim_next(tmp_path, "q", "hostA", 2)
    store.record_result(tmp_path, "q", jb, rc=3, node="hostA", gpu=0,
                        start_utc="s", end_utc="e", log="l", attempt=0)
    store.remove_claim(tmp_path, "q", jb.jobkey)
    store.claim_next(tmp_path, "q", "hostA", 3)  # c stays claimed (running)
    st = store.queue_status(tmp_path, "q")
    assert (st["done"], st["failed"], st["running"], st["pending"]) == (1, 1, 1, 1)
    assert st["running_jobs"][0]["key"] == "c" and st["running_jobs"][0]["node"] == "hostA"
    assert not store.queue_complete(tmp_path, "q")


def test_queue_complete_when_all_terminal(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("a", "x")])
    a = store.claim_next(tmp_path, "q", "hostA", 1)
    store.record_result(tmp_path, "q", a, rc=1, node="hostA", gpu=0,
                        start_utc="s", end_utc="e", log="l", attempt=0)
    assert store.queue_complete(tmp_path, "q")  # failed still counts as terminal


# ------------------------- stale steal (node safety) -------------------------


def test_steal_same_node_dead_pid(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("a", "x")])
    host = store.this_host()
    store.claim_next(tmp_path, "q", host, 999999999)  # this-node pid that is dead
    stolen = store.steal_stale(tmp_path, "q", host)
    assert stolen == [sanitize_jobkey("a")]
    # now reclaimable
    assert store.claim_next(tmp_path, "q", host, os.getpid()).key == "a"


def test_never_steal_other_node_even_if_pid_dead(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("a", "x")])
    store.claim_next(tmp_path, "q", "OTHER-NODE", 999999999)  # dead pid, foreign node
    assert store.steal_stale(tmp_path, "q", store.this_host()) == []
    assert store.claim_next(tmp_path, "q", store.this_host(), 1) is None  # still held


def test_live_same_node_pid_not_stolen(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("a", "x")])
    store.claim_next(tmp_path, "q", store.this_host(), os.getpid())  # alive
    assert store.steal_stale(tmp_path, "q", store.this_host()) == []


# ------------------------- deps -------------------------


def _ready_names(pool):
    return [n for n, _ in pool.ready_queues()]


def test_deps_block_until_complete_and_failed_still_satisfies(tmp_path):
    from subliminal_learning.jobqueue.worker import MasterLog, _Pool

    store.create_queue(tmp_path, "dep")
    store.append_jobs(tmp_path, "dep", [Job("a", "x")])
    store.create_queue(tmp_path, "main", depends_on=["dep"])
    store.append_jobs(tmp_path, "main", [Job("m", "x")])
    pool = _Pool(
        tmp_path, store.this_host(), FakeGpuManager(),
        MasterLog(tmp_path / "logs" / "m.log"),
        queues_filter=None, poll_s=0.01, gpu_wait_s=0.01,
    )
    assert "main" not in _ready_names(pool)  # dep incomplete -> main blocked
    # complete dep with a FAILURE
    a = store.claim_next(tmp_path, "dep", store.this_host(), 1)
    store.record_result(tmp_path, "dep", a, rc=7, node=store.this_host(), gpu=0,
                        start_utc="s", end_utc="e", log="l", attempt=0)
    assert "main" in _ready_names(pool)  # failed-terminal dep still satisfies


# ------------------------- requeue -------------------------


def test_requeue_failed_only_bumps_attempt(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("ok", "x"), Job("bad", "x")])
    ok = store.claim_next(tmp_path, "q", store.this_host(), 1)
    store.record_result(tmp_path, "q", ok, rc=0, node=store.this_host(), gpu=0,
                        start_utc="s", end_utc="e", log="l", attempt=0)
    store.remove_claim(tmp_path, "q", ok.jobkey)
    bad = store.claim_next(tmp_path, "q", store.this_host(), 2)
    store.record_result(tmp_path, "q", bad, rc=9, node=store.this_host(), gpu=0,
                        start_utc="s", end_utc="e", log="l", attempt=0)
    store.remove_claim(tmp_path, "q", bad.jobkey)
    requeued = store.requeue(tmp_path, "q", failed_only=True)
    assert requeued == [sanitize_jobkey("bad")]
    assert store.read_attempt(tmp_path, "q", sanitize_jobkey("bad")) == 1
    assert store.job_state(tmp_path, "q", bad) == store.PENDING
    assert store.job_state(tmp_path, "q", ok) == store.DONE


# ------------------------- gpu layer (patched, tmp lock dir) -------------------------


def test_gpu_manager_acquires_and_memory_gate(tmp_path, monkeypatch):
    host = store.this_host()
    _write_policy(tmp_path, host, gpus=[0, 1], cap_per_gpu=1, free_mem_mib=100)
    prefix = str(tmp_path / "lock")
    calls = {"n": 0}

    def fake_mem():
        calls["n"] += 1
        return {0: 50, 1: 5000}  # gpu0 too small, gpu1 fine

    gm = GpuManager(tmp_path, host, lock_prefix=prefix, mem_query=fake_mem,
                    mem_interval_s=0.0)
    slot = gm.acquire(100)
    assert slot is not None and slot.gpu == 1  # skipped mem-starved gpu0
    assert calls["n"] >= 2  # two consecutive checks on the winner
    # cap_per_gpu=1: gpu1's only slot now held; a second acquire finds nothing free on it,
    # but gpu0 fails the memory gate -> None.
    assert gm.acquire(100) is None
    slot.release()
    assert gm.acquire(100).gpu == 1


def test_gpu_policy_missing_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        GpuManager(tmp_path, "nosuchhost").acquire(10)


def test_acquire_spreads_least_loaded_first(tmp_path):
    host = store.this_host()
    _write_policy(tmp_path, host, gpus=[0, 1], cap_per_gpu=2, free_mem_mib=100)
    gm = GpuManager(tmp_path, host, lock_prefix=str(tmp_path / "lock"),
                    mem_query=lambda: {0: 5000, 1: 5000}, mem_interval_s=0.0)
    a = gm.acquire(100)
    b = gm.acquire(100)
    # breadth-first: with both cards empty-ish, the second job lands on the OTHER GPU
    # instead of filling the first card's remaining slot.
    assert {a.gpu, b.gpu} == {0, 1}
    c = gm.acquire(100)
    d = gm.acquire(100)
    assert {c.gpu, d.gpu} == {0, 1}  # second layer fills both cards to cap
    assert gm.acquire(100) is None  # global cap reached
    for s in (a, b, c, d):
        s.release()


def test_acquire_weighted_slots_budget_math(tmp_path):
    host = store.this_host()
    _write_policy(tmp_path, host, gpus=[0], cap_per_gpu=8, free_mem_mib=100)
    gm = GpuManager(tmp_path, host, lock_prefix=str(tmp_path / "lock"),
                    mem_query=lambda: {0: 5000}, mem_interval_s=0.0)
    light = [gm.acquire(100) for _ in range(6)]  # 6 x weight-1
    heavy = gm.acquire(100, slots=2)  # 1 x weight-2 -> 8/8 units used
    assert all(s is not None for s in light) and heavy is not None
    assert gm.acquire(100) is None  # card exactly full: no unit left even for weight-1
    light[0].release()
    assert gm.acquire(100, slots=2) is None  # 1 free unit: a weight-2 job still can't fit
    got = gm.acquire(100)
    assert got is not None  # but a weight-1 job can
    for s in light[1:] + [heavy, got]:
        s.release()


def test_acquire_weight_above_budget_clamps_to_whole_card(tmp_path):
    host = store.this_host()
    _write_policy(tmp_path, host, gpus=[0], cap_per_gpu=4, free_mem_mib=100)
    gm = GpuManager(tmp_path, host, lock_prefix=str(tmp_path / "lock"),
                    mem_query=lambda: {0: 5000}, mem_interval_s=0.0)
    whole = gm.acquire(100, slots=8)  # clamped to the 4-unit budget: takes the whole card
    assert whole is not None
    assert gm.acquire(100) is None
    whole.release()
    assert gm.acquire(100) is not None


def test_job_slots_field_roundtrip():
    from subliminal_learning.jobqueue.model import Job, parse_jobs_file

    jobs = parse_jobs_file('{"key": "big", "cmd": "true", "slots": 2}\n')
    assert jobs[0].slots == 2
    assert Job.from_dict(jobs[0].to_dict()).slots == 2
    assert parse_jobs_file('{"key": "small", "cmd": "true"}\n')[0].slots is None


def test_acquire_group_cap_is_stage_ceiling(tmp_path):
    host = store.this_host()
    _write_policy(tmp_path, host, gpus=[0], cap_per_gpu=4, free_mem_mib=100)
    gm = GpuManager(tmp_path, host, lock_prefix=str(tmp_path / "lock"),
                    mem_query=lambda: {0: 5000}, mem_interval_s=0.0)
    a = gm.acquire(100, cap_per_gpu=1, cap_group="train")
    assert a is not None and a.gpu == 0
    # group "train" is capped at 1/GPU even though the global policy allows 4
    assert gm.acquire(100, cap_per_gpu=1, cap_group="train") is None
    # a different group (and uncapped queues) still get global slots
    b = gm.acquire(100, cap_per_gpu=2, cap_group="distill")
    c = gm.acquire(100)
    assert b is not None and c is not None
    a.release()
    d = gm.acquire(100, cap_per_gpu=1, cap_group="train")  # released group slot reusable
    assert d is not None
    for s in (b, c, d):
        s.release()


# ------------------------- end-to-end mini run -------------------------


def test_end_to_end_mini_run(tmp_path):
    store.create_queue(tmp_path, "q")
    store.append_jobs(
        tmp_path, "q",
        [Job("hi1", "echo hi"), Job("hi2", "echo hi"), Job("boom", "exit 3")],
    )
    run_pool(tmp_path, workers=2, gpu=FakeGpuManager(), hostname="testnode",
             poll_s=0.01, gpu_wait_s=0.01)

    results = {}
    for job in store.load_jobs(tmp_path, "q"):
        rp = tmp_path / "q" / "results" / f"{job.jobkey}.json"
        results[job.key] = json.loads(rp.read_text())
    assert results["hi1"]["rc"] == 0
    assert results["hi2"]["rc"] == 0
    assert results["boom"]["rc"] == 3
    assert store.queue_complete(tmp_path, "q")

    # per-job log exists with a node-stamp first line.
    log = Path(results["hi1"]["log"])
    assert log.exists()
    first = log.read_text().splitlines()[0]
    assert first.startswith("=== node=testnode") and "job=hi1" in first

    # the failed job is requeue-able and re-runs.
    assert store.requeue(tmp_path, "q", failed_only=True) == [sanitize_jobkey("boom")]
    assert store.job_state(tmp_path, "q", Job("boom", "exit 3")) == store.PENDING


def test_end_to_end_no_queues_exits_cleanly(tmp_path):
    # An empty root: pool should return immediately (all_complete over zero queues).
    run_pool(tmp_path, workers=2, gpu=FakeGpuManager(), hostname="testnode",
             poll_s=0.01, gpu_wait_s=0.01)


# ------------------------- gpu yielding (opt-in) -------------------------

from subliminal_learning.jobqueue import yielding  # noqa: E402
from subliminal_learning.jobqueue.gpu import GpuPolicy  # noqa: E402


class _RecLog:
    """A minimal MasterLog stand-in that records (verb, msg) pairs."""

    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def log(self, verb, msg=""):
        self.lines.append((verb, msg))


def _fake_proc(tmp_path, procs, *, num_gpus=8, subdir="proc"):
    """Build a REALISTIC fake /proc tree modelling how CUDA actually opens device nodes.

    ``procs`` is a list of dicts, each ``{"pid": int, "real": int | list[int] | None,
    "cvd": str | None (optional)}``. Every proc opens 3 fds on ALL ``num_gpus`` device nodes
    (the baseline CUDA takes at init) plus 18 more (21 total) on its ``real`` node(s) — the
    measured 3-vs-21 signature — and gets ``nvidiactl``/``nvidia-uvm`` noise fds (must be
    ignored). When ``cvd`` is given, ``/proc/<pid>/environ`` carries
    ``CUDA_VISIBLE_DEVICES=<cvd>``. ``real=None`` means baseline-only on every node (a tie,
    exercising the unattributable catch-all). Returns the proc-root path string.
    """
    proc = tmp_path / subdir
    for spec in procs:
        pid = spec["pid"]
        real = spec.get("real")
        reals = [real] if isinstance(real, int) else list(real or [])
        fd = proc / str(pid) / "fd"
        fd.mkdir(parents=True, exist_ok=True)
        targets = ["/dev/nvidiactl", "/dev/nvidia-uvm"]  # noise fds (must be ignored)
        for g in range(num_gpus):
            targets += [f"/dev/nvidia{g}"] * (21 if g in reals else 3)
        for n, target in enumerate(targets):
            os.symlink(target, fd / str(n))
        cvd = spec.get("cvd")
        if cvd is not None:
            (proc / str(pid) / "environ").write_bytes(
                b"PATH=/usr/bin\x00CUDA_VISIBLE_DEVICES=" + cvd.encode() + b"\x00"
            )
    return str(proc)


def test_yield_policy_defaults_off_when_keys_absent():
    p = GpuPolicy.from_dict({"gpus": [0, 1], "cap_per_gpu": 2, "free_mem_mib": 500})
    assert p.yield_to_foreign is False
    assert p.yield_poll_s == 30.0
    assert p.yield_confirm_polls == 2
    assert p.yield_min_foreign_procs == 1
    assert p.yield_cooldown_s == 900.0
    assert p.yield_action == "kill"
    # explicit keys are honored
    q = GpuPolicy.from_dict({
        "gpus": [0], "cap_per_gpu": 1, "free_mem_mib": 500,
        "yield_to_foreign": True, "yield_action": "drain", "yield_confirm_polls": 3,
    })
    assert q.yield_to_foreign is True and q.yield_action == "drain"
    assert q.yield_confirm_polls == 3


def test_foreign_count_local_procs_not_foreign_and_floor_at_0(tmp_path):
    uuids = {0: "GPU-uuid0", 1: "GPU-uuid1"}
    # smi reports 3 compute procs on gpu0; two of them are locally visible (ours-or-node,
    # any uid incl. a local root daemon) -> only 1 is foreign.
    apps = [(900, "GPU-uuid0"), (901, "GPU-uuid0"), (902, "GPU-uuid0")]
    proc_root = _fake_proc(tmp_path, [{"pid": 11, "real": 0}, {"pid": 22, "real": 0}])
    fc = yielding.foreign_count(
        0, compute_apps_query=lambda: apps, uuid_query=lambda: uuids, proc_root=proc_root
    )
    assert fc == 1
    # floor at 0: more local procs than smi reports -> never negative
    proc_root2 = _fake_proc(
        tmp_path / "b",
        [{"pid": p, "real": 0} for p in (11, 22, 33, 44)],
    )
    fc2 = yielding.foreign_count(
        0, compute_apps_query=lambda: apps, uuid_query=lambda: uuids, proc_root=proc_root2
    )
    assert fc2 == 0


def test_cuda_proc_on_gpu6_does_not_mask_foreign_on_gpu0(tmp_path):
    # THE live regression: our job pinned to GPU 6 opens fds on ALL 8 nodes. A naive fd match
    # would count it as local on gpu0-3 and mask a real collaborator to foreign=0. With
    # CUDA_VISIBLE_DEVICES attribution it is local to GPU 6 ONLY.
    uuids = {g: f"u{g}" for g in range(8)}
    apps = [(999, "u0"), (998, "u6")]  # collaborator on gpu0; our own probe on gpu6
    proc_root = _fake_proc(tmp_path, [{"pid": 100, "real": 6, "cvd": "6"}])
    common = dict(compute_apps_query=lambda: apps, uuid_query=lambda: uuids, proc_root=proc_root)
    assert yielding.foreign_count(0, **common) == 1  # collaborator on gpu0 is NOT masked
    assert yielding.foreign_count(6, **common) == 0  # our own probe is local to gpu6


def test_cuda_visible_devices_comma_list_attributes_to_both(tmp_path):
    proc_root = _fake_proc(tmp_path, [{"pid": 200, "real": 7, "cvd": "4,5"}])
    # CVD wins over fds: local to 4 and 5, not to its (irrelevant) max-fd node 7 or elsewhere.
    assert set(yielding.local_gpu_procs(4, proc_root=proc_root)) == {200}
    assert set(yielding.local_gpu_procs(5, proc_root=proc_root)) == {200}
    assert yielding.local_gpu_procs(7, proc_root=proc_root) == {}
    assert yielding.local_gpu_procs(0, proc_root=proc_root) == {}


def test_max_fd_fallback_when_no_environ(tmp_path):
    # No CUDA_VISIBLE_DEVICES -> attribute by the strictly-above-baseline fd count (21 vs 3).
    proc_root = _fake_proc(tmp_path, [{"pid": 300, "real": 3}])
    assert set(yielding.local_gpu_procs(3, proc_root=proc_root)) == {300}
    assert yielding.local_gpu_procs(0, proc_root=proc_root) == {}


def test_unattributable_proc_counts_on_all_gpus(tmp_path):
    # Neither signal (no CVD, every node ties at baseline) -> attributed to ALL gpus, so it
    # masks foreign load everywhere (conservative: keep our jobs alive rather than yield wrong).
    proc_root = _fake_proc(tmp_path, [{"pid": 400, "real": None}])
    assert set(yielding.local_gpu_procs(0, proc_root=proc_root)) == {400}
    assert set(yielding.local_gpu_procs(7, proc_root=proc_root)) == {400}


def test_environ_reader_is_injectable(tmp_path):
    # The environ signal is injectable (proc_root has no environ files here).
    proc_root = _fake_proc(tmp_path, [{"pid": 500, "real": 1}])
    reader = lambda pid: "2" if pid == 500 else None  # noqa: E731
    assert set(yielding.local_gpu_procs(2, proc_root=proc_root, environ_reader=reader)) == {500}
    assert yielding.local_gpu_procs(1, proc_root=proc_root, environ_reader=reader) == {}


def test_total_compute_procs_smi_failure_returns_none():
    def boom():
        raise RuntimeError("nvidia-smi missing")

    assert yielding.total_compute_procs_from_smi(
        0, compute_apps_query=boom, uuid_query=lambda: {0: "u"}
    ) is None
    # unknown gpu index -> None
    assert yielding.total_compute_procs_from_smi(
        9, compute_apps_query=lambda: [], uuid_query=lambda: {0: "u"}
    ) is None


def _yield_policy(**over):
    d = {"gpus": [0], "cap_per_gpu": 1, "free_mem_mib": 500, "yield_to_foreign": True}
    d.update(over)
    return GpuPolicy.from_dict(d)


def test_yield_confirm_polls_debounce(tmp_path):
    calls = []
    wd = yielding.YieldWatchdog(
        tmp_path, "h",
        load_policy=lambda: _yield_policy(yield_confirm_polls=2),
        master=_RecLog(),
        on_yield=lambda g: calls.append(g),
        compute_apps_query=lambda: [(1, "u0")],  # one foreign (no local procs)
        uuid_query=lambda: {0: "u0"},
        proc_root=str(tmp_path / "empty_proc"),
    )
    wd.poll_once()
    assert yielding.yielded_gpus(tmp_path, "h") == set() and calls == []  # 1 poll: not yet
    wd.poll_once()
    assert yielding.yielded_gpus(tmp_path, "h") == {0} and calls == [0]  # 2nd confirms


def test_acquire_skips_yielded_gpu(tmp_path):
    host = store.this_host()
    _write_policy(tmp_path, host, gpus=[0, 1], cap_per_gpu=1, free_mem_mib=100)
    yielding.write_yielded(tmp_path, host, {0: {"first_seen": "s", "last_seen": "s", "foreign": 2}})
    gm = GpuManager(tmp_path, host, lock_prefix=str(tmp_path / "lock"),
                    mem_query=lambda: {0: 5000, 1: 5000}, mem_interval_s=0.0)
    slot = gm.acquire(100)
    assert slot is not None and slot.gpu == 1  # gpu0 is yielded -> only gpu1 usable
    slot.release()
    # both cards yielded -> nothing acquirable
    yielding.write_yielded(tmp_path, host, {
        0: {"first_seen": "s", "last_seen": "s", "foreign": 1},
        1: {"first_seen": "s", "last_seen": "s", "foreign": 1},
    })
    assert gm.acquire(100) is None


def test_yield_kill_requeues_to_pending_not_failed(tmp_path):
    from subliminal_learning.jobqueue.worker import MasterLog, _Pool

    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("slowjob", "sleep 30")])
    meta = store.read_meta(tmp_path, "q")
    host = store.this_host()
    job = store.claim_next(tmp_path, "q", host, os.getpid())
    pool = _Pool(tmp_path, host, FakeGpuManager(),
                 MasterLog(tmp_path / "logs" / "m.log"),
                 queues_filter=None, poll_s=0.01, gpu_wait_s=0.01)
    slot = GpuSlot(gpu=0, slot=0)
    result = {}

    def run():
        result["rc"] = pool._run_job("q", meta, job, slot, 0)

    t = threading.Thread(target=run)
    t.start()
    # wait until the subprocess is registered as running on gpu0
    for _ in range(500):
        with pool._running_lock:
            if pool._running.get(0, {}).get(job.jobkey):
                break
        import time as _t
        _t.sleep(0.01)
    pool.yield_kill_gpu(0)
    t.join(timeout=30)
    assert result["rc"] is None  # signalled "I killed this"
    # requeued to pending: no result, attempt bumped, NOT failed
    assert not store.has_result(tmp_path, "q", job.jobkey)
    assert store.job_state(tmp_path, "q", job) == store.PENDING
    assert store.read_attempt(tmp_path, "q", job.jobkey) == 1


def test_yield_kill_runs_cleanup_hook(tmp_path):
    from subliminal_learning.jobqueue.worker import MasterLog, _Pool

    marker = tmp_path / "cleaned.txt"
    store.create_queue(tmp_path, "q", defaults={"on_yield_kill_cmd": f"touch {marker}"})
    store.append_jobs(tmp_path, "q", [Job("slowjob", "sleep 30")])
    meta = store.read_meta(tmp_path, "q")
    host = store.this_host()
    job = store.claim_next(tmp_path, "q", host, os.getpid())
    pool = _Pool(tmp_path, host, FakeGpuManager(),
                 MasterLog(tmp_path / "logs" / "m.log"),
                 queues_filter=None, poll_s=0.01, gpu_wait_s=0.01)
    slot = GpuSlot(gpu=0, slot=0)
    t = threading.Thread(target=lambda: pool._run_job("q", meta, job, slot, 0))
    t.start()
    for _ in range(500):
        with pool._running_lock:
            if pool._running.get(0, {}).get(job.jobkey):
                break
        import time as _t
        _t.sleep(0.01)
    pool.yield_kill_gpu(0)
    t.join(timeout=30)
    assert marker.exists()  # cleanup hook fired with env set


def test_drain_mode_never_kills(tmp_path):
    calls = []
    wd = yielding.YieldWatchdog(
        tmp_path, "h",
        load_policy=lambda: _yield_policy(yield_action="drain", yield_confirm_polls=1),
        master=_RecLog(),
        on_yield=lambda g: calls.append(g),
        compute_apps_query=lambda: [(1, "u0")],
        uuid_query=lambda: {0: "u0"},
        proc_root=str(tmp_path / "empty_proc"),
    )
    wd.poll_once()
    # drain marks the card off-limits (marker written) but never triggers a kill
    assert yielding.yielded_gpus(tmp_path, "h") == {0}
    assert calls == []


def test_cooldown_reclaim(tmp_path):
    state = {"apps": [(1, "u0")], "t": 0.0}
    log = _RecLog()
    wd = yielding.YieldWatchdog(
        tmp_path, "h",
        load_policy=lambda: _yield_policy(yield_confirm_polls=1, yield_cooldown_s=100.0),
        master=log,
        on_yield=lambda g: None,
        compute_apps_query=lambda: state["apps"],
        uuid_query=lambda: {0: "u0"},
        proc_root=str(tmp_path / "empty_proc"),
        clock=lambda: state["t"],
    )
    wd.poll_once()
    assert yielding.yielded_gpus(tmp_path, "h") == {0}  # yielded while foreigner present
    state["apps"] = []  # foreigner leaves
    wd.poll_once()  # t=0: cooldown starts
    assert yielding.yielded_gpus(tmp_path, "h") == {0}
    state["t"] = 50.0
    wd.poll_once()  # 50 < 100: still off-limits
    assert yielding.yielded_gpus(tmp_path, "h") == {0}
    state["t"] = 150.0
    wd.poll_once()  # 150 >= 100: reclaimed
    assert yielding.yielded_gpus(tmp_path, "h") == set()
    assert ("RECLAIM", "gpu=0") in log.lines


def test_smi_failure_never_yields(tmp_path):
    def boom():
        raise RuntimeError("no smi")

    log = _RecLog()
    wd = yielding.YieldWatchdog(
        tmp_path, "h",
        load_policy=lambda: _yield_policy(yield_confirm_polls=1),
        master=log,
        on_yield=lambda g: (_ for _ in ()).throw(AssertionError("must not kill")),
        compute_apps_query=boom,
        uuid_query=lambda: {0: "u0"},
        proc_root=str(tmp_path / "empty_proc"),
    )
    wd.poll_once()
    wd.poll_once()  # repeated failures must stay fail-safe
    assert yielding.yielded_gpus(tmp_path, "h") == set()  # never yielded


def test_watchdog_hot_enable_no_work_while_off(tmp_path):
    # While the flag is false, poll_once must be a TRUE no-op (no smi call, no marker); then
    # flipping the policy true on the SAME running watchdog must yield on the next poll.
    calls = {"smi": 0, "uuid": 0}
    state = {"on": False}

    def apps():
        calls["smi"] += 1
        return [(1, "u0")]

    def uuids():
        calls["uuid"] += 1
        return {0: "u0"}

    wd = yielding.YieldWatchdog(
        tmp_path, "h",
        load_policy=lambda: _yield_policy(yield_to_foreign=state["on"], yield_confirm_polls=1),
        master=_RecLog(),
        on_yield=lambda g: None,
        compute_apps_query=apps,
        uuid_query=uuids,
        proc_root=str(tmp_path / "empty_proc"),
    )
    wd.poll_once()  # off: no detection work whatsoever
    assert not yielding.marker_path(tmp_path, "h").exists()
    assert calls == {"smi": 0, "uuid": 0}
    state["on"] = True  # flip the policy on the running watchdog
    wd.poll_once()
    assert yielding.yielded_gpus(tmp_path, "h") == {0}  # begins yielding, no restart
    assert calls["smi"] > 0


def test_yield_kill_skips_already_exited_handle(tmp_path):
    import subprocess

    from subliminal_learning.jobqueue.worker import MasterLog, _Pool, _RunHandle

    store.create_queue(tmp_path, "q")
    store.append_jobs(tmp_path, "q", [Job("done", "true")])
    job = store.load_jobs(tmp_path, "q")[0]
    pool = _Pool(tmp_path, store.this_host(), FakeGpuManager(),
                 MasterLog(tmp_path / "logs" / "m.log"),
                 queues_filter=None, poll_s=0.01, gpu_wait_s=0.01)
    proc = subprocess.Popen(["bash", "-c", "true"])
    proc.wait()  # job finished NATURALLY before the watchdog fires
    handle = _RunHandle(proc=proc, queue="q", job=job, gpu=0)
    with pool._running_lock:
        pool._running[0] = {job.jobkey: handle}
    pool.yield_kill_gpu(0)
    assert handle.killed is False  # an already-exited job is left alone, not requeued

    # and a job run to natural completion through _run_job still records its real rc
    store.create_queue(tmp_path, "q2")
    store.append_jobs(tmp_path, "q2", [Job("bye", "exit 4")])
    meta2 = store.read_meta(tmp_path, "q2")
    j2 = store.claim_next(tmp_path, "q2", store.this_host(), os.getpid())
    rc = pool._run_job("q2", meta2, j2, GpuSlot(gpu=0, slot=0), 0)
    assert rc == 4
    assert store.job_state(tmp_path, "q2", j2) == store.FAILED


def test_watchdog_off_when_flag_absent(tmp_path):
    calls = []
    wd = yielding.YieldWatchdog(
        tmp_path, "h",
        load_policy=lambda: GpuPolicy.from_dict(
            {"gpus": [0], "cap_per_gpu": 1, "free_mem_mib": 500}  # yield_to_foreign absent
        ),
        master=_RecLog(),
        on_yield=lambda g: calls.append(g),
        compute_apps_query=lambda: [(1, "u0")],
        uuid_query=lambda: {0: "u0"},
        proc_root=str(tmp_path / "empty_proc"),
    )
    wd.poll_once()
    wd.poll_once()
    assert yielding.yielded_gpus(tmp_path, "h") == set() and calls == []
