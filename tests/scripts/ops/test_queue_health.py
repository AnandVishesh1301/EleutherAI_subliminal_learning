"""`scripts.ops.queue_health`: marker accounting, failure reporting, stale claims and dead-chain resubmission."""
import json
import os
import time
from pathlib import Path

from scripts.ops import queue_health as qh


def _campaign(tmp_path, keys, done=(), failed=(), held=(), claims=(), stale=()):
    jobs = tmp_path / "jobs.jsonl"
    jobs.write_text("".join(json.dumps({"key": k, "cmd": "true"}) + "\n" for k in keys))
    cd = tmp_path / "claims"
    for sub in ("done", "failed", "claims"):
        (cd / sub).mkdir(parents=True)
    for k in done:
        (cd / "done" / qh.safe(k)).write_text("{}")
    for k in failed:
        (cd / "failed" / qh.safe(k)).write_text(json.dumps({"rc": 1}))
    for k in held:
        (cd / "failed" / qh.safe(k)).write_text(json.dumps({"hold": True}))
    for k in claims:
        (cd / "claims" / qh.safe(k)).write_text("{}")
    for k in stale:
        p = cd / "claims" / qh.safe(k)
        p.write_text("{}")
        os.utime(p, (time.time() - 5000, time.time() - 5000))
    ld = tmp_path / "logs"
    ld.mkdir()
    return {"name": "camp", "jobs_file": str(jobs), "claim_dir": str(cd), "log_dir": str(ld), "job_prefix": "camp",
            "workers": [str(tmp_path / "w.sbatch")], "expected_workers": 2, "hold_marker": "hold"}


def test_markers_hold_and_remaining(tmp_path):
    keys = [f"sd|a|{i}" for i in range(6)]
    cfg = _campaign(tmp_path, keys, done=keys[:2], failed=keys[2:3], held=keys[3:4], claims=keys[4:5])
    m = qh.marker_sets(Path(cfg["claim_dir"]), "hold")
    assert (len(m["done"]), len(m["failed"]), len(m["held"]), len(m["claims"])) == (2, 1, 1, 1)


def test_tick_reports_new_failures_once_and_resubmits_dead_chain(tmp_path, monkeypatch):
    keys = [f"sd|a|{i}" for i in range(4)]
    cfg = _campaign(tmp_path, keys, done=keys[:1], failed=keys[1:2], stale=keys[2:3])
    (tmp_path / "logs" / f"{qh.safe(keys[1])}.log").write_text("start\nValueError: boom\n")
    lines = []
    monkeypatch.setattr(qh, "squeue_counts", lambda prefix: {"RUNNING": 0, "PENDING": 0})
    calls = []

    class R:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(qh.subprocess, "run", lambda *a, **k: calls.append(a[0]) or R())
    state = {}
    qh.tick(cfg, state, stale_s=1800, stall_min=150, log=lines.append)
    text = "\n".join(lines)
    assert "done 1/4" in text and "remaining 2" in text and "failed 1" in text
    assert "FAILED sd_a_1: ValueError: boom" in text
    assert "STALE claim" in text
    assert "ACTION camp" in text and len(calls) == 2 and calls[0][0] == "sbatch"
    # second tick: the same failure is not reported again, chain now alive -> no resubmit
    lines.clear(); calls.clear()
    monkeypatch.setattr(qh, "squeue_counts", lambda prefix: {"RUNNING": 2, "PENDING": 0})
    qh.tick(cfg, state, stale_s=1800, stall_min=150, log=lines.append)
    assert "FAILED sd_a_1" not in "\n".join(lines) and not calls


def test_squeue_failure_is_reported_not_fatal(tmp_path, monkeypatch):
    cfg = _campaign(tmp_path, ["k1"], claims=["k1"])
    monkeypatch.setattr(qh, "squeue_counts", lambda prefix: None)
    lines = []
    qh.tick(cfg, {}, 1800, 150, lines.append)
    assert "squeue FAILED" in lines[0] and not any("ACTION" in l for l in lines)


def test_hung_job_is_a_live_claim_with_a_silent_log(tmp_path, monkeypatch):
    keys = ["sd|a|1", "sd|a|2"]
    cfg = _campaign(tmp_path, keys, claims=keys)
    for k in keys:
        (tmp_path / "logs" / f"{qh.safe(k)}.log").write_text("start\n")
    old = time.time() - 60 * 60
    os.utime(tmp_path / "logs" / f"{qh.safe(keys[0])}.log", (old, old))       # silent for an hour, claim fresh
    assert qh.hung_jobs(Path(cfg["claim_dir"]), Path(cfg["log_dir"]), hung_min=45) == [qh.safe(keys[0])]
    lines = []
    monkeypatch.setattr(qh, "squeue_counts", lambda prefix: {"RUNNING": 1, "PENDING": 0})
    qh.tick(cfg, {}, 1800, 150, lines.append, hung_min=45)
    assert any("HUNG job" in l and "sd_a_1" in l for l in lines)
