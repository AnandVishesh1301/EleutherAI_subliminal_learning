"""The Slurm packer: admission by free memory + cap, per-job logs, results and resume."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ops import slurm_pack as sp


def test_parse_jobs_jsonl_and_plain_lines(tmp_path):
    f = tmp_path / "jobs.jsonl"
    f.write_text('{"key": "a", "cmd": "echo a", "mem_mib": 1000, "env": {"X": "1"}}\n'
                 '# comment\n\n'
                 'echo plain\n')
    jobs = sp.parse_jobs(f, default_mem_mib=6000)
    assert [(j.key, j.cmd, j.mem_mib) for j in jobs] == [("a", "echo a", 1000), ("line4", "echo plain", 6000)]
    assert jobs[0].env == {"X": "1"}


def test_parse_jobs_rejects_duplicate_keys(tmp_path):
    f = tmp_path / "jobs.jsonl"
    f.write_text('{"key": "a", "cmd": "true"}\n{"key": "a", "cmd": "true"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        sp.parse_jobs(f, 100)


def test_pick_gpu_gates_on_memory_then_cap():
    job = sp.Job("k", "true", mem_mib=5000)
    free = {"0": 6000, "1": 40000}
    # gpu 0: 6000 - reserve 2000 = 4000 < 5000 -> only gpu 1 qualifies
    assert sp.pick_gpu(job, free, {}, max_per_gpu=5, reserve_mib=2000) == "1"
    # cap reached on gpu 1 -> nothing fits
    assert sp.pick_gpu(job, free, {"1": 5}, max_per_gpu=5, reserve_mib=2000) is None
    # plenty of memory everywhere: the freest GPU wins
    assert sp.pick_gpu(job, {"0": 30000, "1": 40000}, {}, 5, 2000) == "1"


def test_run_pack_runs_everything_logs_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    jobs = [sp.Job("ok1", "echo hi", 100), sp.Job("bad", "exit 3", 100), sp.Job("ok2", "echo yo", 100)]
    log_dir = tmp_path / "logs"
    failed = sp.run_pack(jobs, ["0", "1"], log_dir, max_per_gpu=2, reserve_mib=0,
                         poll_s=0.05, settle_s=0.0, free_fn=lambda gpus: {g: 10000 for g in gpus})
    assert failed == 1
    rows = {json.loads(l)["key"]: json.loads(l) for l in (log_dir / "results.jsonl").read_text().splitlines()}
    assert rows["ok1"]["rc"] == 0 and rows["bad"]["rc"] == 3 and rows["ok2"]["rc"] == 0
    assert set(rows["ok1"]["gpu"], ) <= {"0", "1"}
    assert "hi" in (log_dir / "ok1.log").read_text()
    # resume: only the failed key runs again
    failed2 = sp.run_pack([sp.Job("ok1", "echo again", 100), sp.Job("bad", "true", 100)], ["0"], log_dir,
                          max_per_gpu=1, reserve_mib=0, poll_s=0.05, settle_s=0.0,
                          free_fn=lambda gpus: {g: 10000 for g in gpus})
    assert failed2 == 0
    assert "again" not in (log_dir / "ok1.log").read_text()
    assert sp.finished_keys(log_dir / "results.jsonl") == {"ok1", "ok2", "bad"}


def test_run_pack_waits_when_nothing_fits_then_admits(tmp_path):
    calls = {"n": 0}

    def free_fn(gpus):
        calls["n"] += 1
        return {g: (0 if calls["n"] < 3 else 10000) for g in gpus}

    failed = sp.run_pack([sp.Job("k", "true", 500)], ["0"], tmp_path / "l", max_per_gpu=1,
                         reserve_mib=0, poll_s=0.01, settle_s=0.0, free_fn=free_fn)
    assert failed == 0 and calls["n"] >= 3


def test_free_mib_by_gpu_matches_uuid_or_index(monkeypatch):
    smi = "0, GPU-aaa, 1000\n1, GPU-bbb, 2000\n2, GPU-ccc, 3000\n"

    class R:
        stdout = smi

    monkeypatch.setattr(sp.subprocess, "run", lambda *a, **k: R())
    # Slurm-style UUIDs resolve to their own rows, not to positions 0 and 1
    assert sp.free_mib_by_gpu(["GPU-ccc", "GPU-bbb"]) == {"GPU-ccc": 3000, "GPU-bbb": 2000}
    # plain indices still work; an unknown id falls back to its position
    assert sp.free_mib_by_gpu(["1", "zzz"]) == {"1": 2000, "zzz": 2000}


# ----------------------------------------------------------------- shared queue (--claim-dir)
def _free(gpus):
    return {g: 10000 for g in gpus}


def _rows(log_dir):
    rows = []
    for p in log_dir.glob("results*.jsonl"):
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows


def test_claim_queue_markers_and_stale_reclaim(tmp_path):
    q = sp.ClaimQueue(tmp_path / "q", stale_s=0.2, worker="w1")
    assert q.claim_state("k") == "free" and q.try_claim("k") and q.claim_state("k") == "live"
    other = sp.ClaimQueue(tmp_path / "q", stale_s=0.2, worker="w2")
    assert not other.try_claim("k")                       # live claim is respected
    import time
    time.sleep(0.3)
    assert other.claim_state("k") == "stale" and other.try_claim("k")   # dead worker -> reclaimed
    assert other.claim_state("k") == "live" and not q.try_claim("k")
    assert len(list((tmp_path / "q" / "stale").iterdir())) == 1
    other.finish("k", 0, 1.0)
    assert other.terminal("k") == "done" and q.claim_state("k") == "free"
    q.try_claim("f"); q.finish("f", 3, 1.0)
    assert q.terminal("f") == "failed"


def test_shared_queue_two_workers_run_each_key_once(tmp_path):
    """Two workers on the same jobs file and claim dir: every key runs exactly once overall."""
    import threading
    jobs = [sp.Job(f"j{i}", f"sleep 0.{i % 3}; echo j{i}", 100) for i in range(8)]
    log_dir = tmp_path / "logs"

    def worker(name):
        q = sp.ClaimQueue(tmp_path / "q", stale_s=60, worker=name)
        sp.run_pack(jobs, ["0"], log_dir, max_per_gpu=2, reserve_mib=0, poll_s=0.01, settle_s=0.0,
                    free_fn=_free, queue=q)

    ts = [threading.Thread(target=worker, args=(n,)) for n in ("A", "B")]
    [t.start() for t in ts]; [t.join(timeout=30) for t in ts]
    rows = _rows(log_dir)
    assert sorted(r["key"] for r in rows) == [f"j{i}" for i in range(8)]     # once each, no dupes
    assert {r["worker"] for r in rows} == {"A", "B"}                          # both took work
    assert sorted(p.name for p in (tmp_path / "q" / "done").iterdir()) == sorted(f"j{i}" for i in range(8))
    assert not list((tmp_path / "q" / "claims").iterdir())
    # a late third worker finds nothing to take and exits at once with no rows
    sp.run_pack(jobs, ["0"], log_dir, max_per_gpu=2, reserve_mib=0, poll_s=0.01, settle_s=0.0,
                free_fn=_free, queue=sp.ClaimQueue(tmp_path / "q", 60, "C"))
    assert len(_rows(log_dir)) == 8


def test_shared_queue_skips_live_claims_failed_keys_and_reclaims_stale(tmp_path):
    import time
    q_dir = tmp_path / "q"
    other = sp.ClaimQueue(q_dir, stale_s=0.3, worker="dead")
    assert other.try_claim("held")                     # another worker is (apparently) running it
    other.try_claim("broken"); other.finish("broken", 1, 1.0)   # failed elsewhere: never retried
    stale = sp.ClaimQueue(q_dir, stale_s=0.3, worker="ghost")
    stale.try_claim("orphan")
    time.sleep(0.35)                                   # orphan's heartbeat is now stale; `held` gets re-touched below
    other.heartbeat("held")
    jobs = [sp.Job("held", "echo NO", 100), sp.Job("broken", "echo NO", 100),
            sp.Job("orphan", "echo reclaimed", 100), sp.Job("fresh", "echo fresh", 100)]
    me = sp.ClaimQueue(q_dir, stale_s=0.3, worker="me")
    failed = sp.run_pack(jobs, ["0"], tmp_path / "l", max_per_gpu=2, reserve_mib=0, poll_s=0.01,
                         settle_s=0.0, free_fn=_free, queue=me)
    assert failed == 0
    assert sorted(r["key"] for r in _rows(tmp_path / "l")) == ["fresh", "orphan"]
    assert "reclaimed" in (tmp_path / "l" / "orphan.log").read_text()
    assert me.claim_state("held") == "live" and me.terminal("broken") == "failed"


def test_shared_queue_lingers_for_a_stale_claim(tmp_path):
    """With linger_s > 0 an idle worker waits for another worker's claim to go stale, then takes it."""
    q_dir = tmp_path / "q"
    sp.ClaimQueue(q_dir, stale_s=0.3, worker="dying").try_claim("k")
    me = sp.ClaimQueue(q_dir, stale_s=0.3, worker="me")
    failed = sp.run_pack([sp.Job("k", "echo took", 100)], ["0"], tmp_path / "l", max_per_gpu=1, reserve_mib=0,
                         poll_s=0.02, settle_s=0.0, free_fn=_free, queue=me, linger_s=5.0)
    assert failed == 0 and [r["key"] for r in _rows(tmp_path / "l")] == ["k"]


def test_fixed_mode_picks_the_first_job_that_fits(tmp_path):
    """A big job at the head of the list no longer blocks a small one behind it."""
    jobs = [sp.Job("big", "echo big", 9000), sp.Job("small", "echo small", 100)]
    seen = []

    def free_fn(gpus):
        seen.append(1)
        return {g: (5000 if len(seen) < 4 else 20000) for g in gpus}

    failed = sp.run_pack(jobs, ["0"], tmp_path / "l", max_per_gpu=2, reserve_mib=0, poll_s=0.01,
                         settle_s=0.0, free_fn=free_fn)
    assert failed == 0
    rows = _rows(tmp_path / "l")
    assert [r["key"] for r in rows][0] == "small" and {r["key"] for r in rows} == {"big", "small"}


def test_admit_until_closes_admission_but_waits_for_running(tmp_path):
    """After admit_until_s the worker starts nothing new, finishes what runs, and exits."""
    jobs = [sp.Job("first", "sleep 0.3; echo first", 100), sp.Job("second", "echo second", 100)]
    failed = sp.run_pack(jobs, ["0"], tmp_path / "l", max_per_gpu=1, reserve_mib=0, poll_s=0.02,
                         settle_s=0.0, free_fn=_free, admit_until_s=0.1)
    assert failed == 0
    assert [r["key"] for r in _rows(tmp_path / "l")] == ["first"]
    # fixed mode resume picks the leftover up on the next submission
    sp.run_pack(jobs, ["0"], tmp_path / "l", max_per_gpu=1, reserve_mib=0, poll_s=0.02, settle_s=0.0, free_fn=_free)
    assert sorted(r["key"] for r in _rows(tmp_path / "l")) == ["first", "second"]


def test_sigterm_forwards_to_children_and_releases_claims(tmp_path):
    """A SIGTERM to the packer reaches the running job, its row is recorded, its claim is released
    (no done/failed marker) and run_pack raises _Interrupted."""
    import os, signal, threading, time
    q = sp.ClaimQueue(tmp_path / "q", stale_s=60, worker="w")
    # the child traps TERM and exits 75 like a yielding student
    job = sp.Job("k", "trap 'exit 75' TERM; sleep 30 & wait", 100)   # `& wait` lets bash run the trap at once
    t = threading.Timer(0.6, lambda: os.kill(os.getpid(), signal.SIGTERM))
    t.start()
    with pytest.raises(sp._Interrupted):
        sp.run_pack([job], ["0"], tmp_path / "l", max_per_gpu=1, reserve_mib=0, poll_s=0.05, settle_s=0.0,
                    free_fn=_free, queue=q)
    rows = _rows(tmp_path / "l")
    assert rows and rows[0]["key"] == "k" and rows[0]["rc"] == 75
    assert q.claim_state("k") == "free" and q.terminal("k") is None


def test_rc75_is_a_yield_not_a_failure(tmp_path):
    q = sp.ClaimQueue(tmp_path / "q", stale_s=60, worker="w")
    failed = sp.run_pack([sp.Job("y", "exit 75", 100)], ["0"], tmp_path / "l", max_per_gpu=1, reserve_mib=0,
                         poll_s=0.02, settle_s=0.0, free_fn=_free, queue=q)
    assert failed == 0 and q.terminal("y") is None and q.claim_state("y") == "free"


def test_priority_orders_jobs_and_live_reload_reorders(tmp_path):
    """Higher priority runs first; editing the jobs file between ticks reorders the remaining jobs."""
    import os, time
    f = tmp_path / "jobs.jsonl"
    rows = [{"key": "a", "cmd": "sleep 0.3; echo a", "priority": 0}, {"key": "b", "cmd": "sleep 0.3; echo b", "priority": 5},
            {"key": "c", "cmd": "echo c", "priority": 0}, {"key": "d", "cmd": "echo d", "priority": 0}]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    jobs = sp.parse_jobs(f, 100)
    assert [j.key for j in jobs] == ["b", "a", "c", "d"]            # b first, the rest in file order
    # one slot: b runs first; while it runs, bump d above a and c
    import threading
    def bump():
        time.sleep(0.1)
        rows[3]["priority"] = 9
        f.write_text("".join(json.dumps(r) + "\n" for r in rows))
        os.utime(f, (time.time() + 1, time.time() + 1))
    threading.Thread(target=bump).start()
    sp.run_pack(jobs, ["0"], tmp_path / "l", max_per_gpu=1, reserve_mib=0, poll_s=0.02, settle_s=0.0,
                free_fn=_free, jobs_file=f, default_mem_mib=100)
    order = [r["key"] for r in _rows(tmp_path / "l")]
    assert order[0] == "b" and order.index("d") < order.index("a") and set(order) == {"a", "b", "c", "d"}


def test_drain_file_keeps_worker_alive_until_producer_is_done(tmp_path):
    """With no rows yet and no drain file, the worker waits; rows appended later run; the drain file ends it."""
    import threading, time
    f = tmp_path / "jobs.jsonl"
    f.write_text("")
    drain = tmp_path / "DRAIN"

    def produce():
        time.sleep(0.2)
        f.write_text(json.dumps({"key": "late", "cmd": "echo late"}) + "\n")
        time.sleep(0.4)
        drain.touch()

    threading.Thread(target=produce).start()
    sp.run_pack([], ["0"], tmp_path / "l", max_per_gpu=1, reserve_mib=0, poll_s=0.02, settle_s=0.0, free_fn=_free,
                jobs_file=f, default_mem_mib=100, drain_file=drain)
    assert [r["key"] for r in _rows(tmp_path / "l")] == ["late"]


def test_main_waits_for_a_jobs_file_that_does_not_exist_yet(tmp_path, monkeypatch):
    """`--drain-file` mode: the worker may start before the producer wrote the jobs file."""
    import threading, time
    f = tmp_path / "jobs.jsonl"; drain = tmp_path / "DRAIN"
    monkeypatch.setattr(sp, "visible_gpus", lambda: ["0"])
    monkeypatch.setattr(sp, "free_mib_by_gpu", lambda gpus: {g: 10000 for g in gpus})

    def produce():
        time.sleep(0.3)
        f.write_text(json.dumps({"key": "k", "cmd": "echo k"}) + "\n")
        time.sleep(0.5)
        drain.touch()

    threading.Thread(target=produce).start()
    # call main() directly (typer wraps it only under typer.run)
    sp.main(jobs_file=f, log_dir=tmp_path / "l", max_per_gpu=1, default_mem_mib=100, reserve_mib=0, poll_s=0.02,
            settle_s=0.0, claim_dir=None, stale_s=900, linger_s=0.0, admit_until_s=None, drain_file=drain)
    assert [r["key"] for r in _rows(tmp_path / "l")] == ["k"]


def test_two_campaigns_share_one_worker(tmp_path, monkeypatch):
    """Several --jobs-file/--claim-dir/--log-dir triples: one pending list (priority first, then file order),
    each job claimed in ITS campaign's claim dir, logged to ITS log dir; a second worker on the same sources
    skips everything the first one finished."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "jobs.jsonl").write_text(json.dumps({"key": "a1", "cmd": "echo a1"}) + "\n" + json.dumps({"key": "a2", "cmd": "echo a2"}) + "\n")
    (b / "jobs.jsonl").write_text(json.dumps({"key": "b1", "cmd": "echo b1", "priority": 5}) + "\n")
    monkeypatch.setattr(sp, "visible_gpus", lambda: ["0"])
    monkeypatch.setattr(sp, "free_mib_by_gpu", lambda gpus: {g: 10000 for g in gpus})
    sp.main(jobs_file=[a / "jobs.jsonl", b / "jobs.jsonl"], log_dir=[a / "l", b / "l"], claim_dir=[a / "c", b / "c"],
            max_per_gpu=1, default_mem_mib=100, reserve_mib=0, poll_s=0.02, settle_s=0.0, stale_s=900, linger_s=0.0,
            admit_until_s=None, drain_file=None)
    assert sorted(p.name for p in (a / "c/done").iterdir()) == ["a1", "a2"]
    assert sorted(p.name for p in (b / "c/done").iterdir()) == ["b1"]
    assert (a / "l/a1.log").exists() and (b / "l/b1.log").exists() and not (a / "l/b1.log").exists()
    rows_a = [json.loads(l) for f in (a / "l").glob("results.*.jsonl") for l in f.read_text().splitlines()]
    rows_b = [json.loads(l) for f in (b / "l").glob("results.*.jsonl") for l in f.read_text().splitlines()]
    assert {r["key"] for r in rows_a} == {"a1", "a2"} and {r["key"] for r in rows_b} == {"b1"}
    # second worker: nothing left to take
    sp.main(jobs_file=[a / "jobs.jsonl", b / "jobs.jsonl"], log_dir=[a / "l", b / "l"], claim_dir=[a / "c", b / "c"],
            max_per_gpu=1, default_mem_mib=100, reserve_mib=0, poll_s=0.02, settle_s=0.0, stale_s=900, linger_s=0.0,
            admit_until_s=None, drain_file=None)
    assert len(list((a / "c/done").iterdir())) == 2 and len(list((b / "c/done").iterdir())) == 1


def test_merge_sources_orders_by_priority_then_file_order(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps({"key": "a1", "cmd": "true"}) + "\n" + json.dumps({"key": "a2", "cmd": "true", "priority": 2}) + "\n")
    b.write_text(json.dumps({"key": "b1", "cmd": "true", "priority": 2}) + "\n" + json.dumps({"key": "b2", "cmd": "true"}) + "\n")
    srcs = [sp.Source(a, tmp_path / "la"), sp.Source(b, tmp_path / "lb")]
    jobs = sp.merge_sources(srcs, 100)
    assert [j.key for j in jobs] == ["a2", "b1", "a1", "b2"]
    assert jobs[0].src is srcs[0] and jobs[1].src is srcs[1]
    missing = sp.Source(tmp_path / "nope.jsonl", tmp_path / "ln")
    assert [j.key for j in sp.merge_sources([missing, srcs[1]], 100)] == ["b1", "b2"] and missing.mtime is None
