from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from subliminal_learning.fleet import ledger as fl
from subliminal_learning.fleet.ledger_io import drain_report, read_ledger, update_ledger
from subliminal_learning.fleet import watch_io
from subliminal_learning.fleet.watch import (
    NodeProbe,
    QueueSnapshot,
    SlurmRootView,
    SlurmSnapshot,
    WatchMemory,
    evaluate,
    evaluate_slurm,
)
from subliminal_learning.fleet.watch_io import (
    ResultCache,
    _unselfmatch,
    login_probe_script,
    parse_probe,
    probe_script,
    queue_snapshot,
    recent_oom_logs,
    slurm_root_view,
)
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.slurm import parse_squeue
from subliminal_learning.jobqueue.model import Job

NOW = "2026-10-16T00:00:00Z"
SOAR, C4 = "soar-subliminal-0", "soar-subliminal-3"
CARD = fl.Card(queue_roots=("/q",), queue_prefix="a-", tmux_session="a")


def _ledger_lent_soar() -> fl.Ledger:
    led = fl.empty_ledger()
    for name in ("a", "b"):
        led = fl.register(led, name, CARD, chat="c", now=NOW)
    led = fl.claim(led, "a", ["soar", "cluster-4"], chat="c", now=NOW)
    led = fl.start_yield(led, "a", "b", ["soar"], mode="drain", chat="c", now=NOW)
    return fl.complete_yield(led, "b", ["soar"], chat="c", now=NOW)


def _run(led, probes, snap, memory=WatchMemory(primed=True)):
    return evaluate(led.campaigns["a"], led, probes, snap, memory, {})


def test_lent_node_with_live_pool_and_cells_alerts() -> None:
    led = _ledger_lent_soar()
    snap = QueueSnapshot(counts={}, running_by_host={SOAR: 2})
    probes = {
        SOAR: NodeProbe(SOAR, pools=1, session=True),
        C4: NodeProbe(C4, pools=1, session=True),
    }
    lines, _ = _run(led, probes, snap)
    alerts = [line for line in lines if line.startswith("ALERT")]
    assert any("2 a cell(s) running on a node b holds" in a for a in alerts)
    assert any("pool live without a stop sentinel" in a for a in alerts)


def test_lent_node_paused_and_idle_is_quiet() -> None:
    led = _ledger_lent_soar()
    snap = QueueSnapshot(counts={}, paused_hosts=frozenset({SOAR}))
    probes = {SOAR: NodeProbe(SOAR, pools=1), C4: NodeProbe(C4, pools=1, session=True)}
    lines, _ = _run(led, probes, snap)
    assert not [line for line in lines if line.startswith("ALERT")]


def test_running_node_with_sentinel_or_no_pool_alerts() -> None:
    led = _ledger_lent_soar()
    snap = QueueSnapshot(counts={}, paused_hosts=frozenset({C4}))
    lines, _ = _run(led, {C4: NodeProbe(C4, pools=1, session=True)}, snap)
    assert any("stop sentinel set but the ledger says a runs here" in x for x in lines)
    lines, _ = _run(
        led, {C4: NodeProbe(C4, pools=0, session=True)}, QueueSnapshot(counts={})
    )
    assert any("no a jobq pool" in x for x in lines)


def test_lease_change_and_drained_are_reported() -> None:
    led = fl.empty_ledger()
    for name in ("a", "b"):
        led = fl.register(led, name, CARD, chat="c", now=NOW)
    led = fl.claim(led, "a", ["cluster-4"], chat="c", now=NOW)
    probes = {C4: NodeProbe(C4, pools=1, session=True)}
    _, mem = _run(led, probes, QueueSnapshot(counts={}))
    led = fl.start_yield(led, "a", "b", ["cluster-4"], mode="drain", chat="c", now=NOW)
    snap = QueueSnapshot(counts={}, paused_hosts=frozenset({C4}))
    lines, _ = _run(led, probes, snap, mem)
    lease_lines = [x for x in lines if x.startswith("INFO LEASE")]
    assert lease_lines and lease_lines[0].endswith("mode=drain drained")


def test_first_pass_counts_existing_failures_then_alerts_new_ones() -> None:
    led = _ledger_lent_soar()
    probes = {C4: NodeProbe(C4, pools=1, session=True)}
    lines, mem = evaluate(
        led.campaigns["a"],
        led,
        probes,
        QueueSnapshot(counts={}, failed_keys=frozenset({"q:k1"})),
        WatchMemory(),
        {},
    )
    assert not any(x.startswith("ALERT new failed") for x in lines)
    assert any("1 failed result(s) present" in x for x in lines)
    lines, _ = _run(
        led,
        probes,
        QueueSnapshot(counts={}, failed_keys=frozenset({"q:k1", "q:k2"})),
        mem,
    )
    assert [x for x in lines if x.startswith("ALERT new failed")] == [
        "ALERT new failed result: q:k2"
    ]


def test_over_cap_alert_from_node_cell_count() -> None:
    led = _ledger_lent_soar()
    snap = QueueSnapshot(counts={}, caps={C4: 3})
    probe = NodeProbe(C4, pools=1, session=True, busiest_gpu=3, busiest_cells=4)
    lines, _ = _run(led, {C4: probe}, snap)
    assert any("4 jobq cells on GPU 3 > cap_per_gpu 3" in x for x in lines)
    ok = NodeProbe(C4, pools=1, session=True, busiest_gpu=3, busiest_cells=3)
    lines, _ = _run(led, {C4: ok}, snap)
    assert not any("cap_per_gpu" in x for x in lines)


def test_probe_patterns_never_match_their_own_text() -> None:
    pat = _unselfmatch("scripts.jobq work --root /q( |$)")
    assert pat.startswith("[s]cripts")
    script = probe_script(CARD, C4)
    assert "scripts.jobq" not in script


def test_parse_probe_detects_wrong_or_missing_host() -> None:
    assert not parse_probe(C4, "ssh: connect timed out").reachable
    ok = parse_probe(C4, f"host={C4}\npools=2\nsession=1\noom15=0\nbusiest=      7 5\n")
    assert (ok.reachable, ok.pools, ok.session) == (True, 2, True)
    assert (ok.busiest_cells, ok.busiest_gpu) == (7, 5)


def test_update_ledger_writes_events_and_drain_report(tmp_path: Path) -> None:
    fleet = tmp_path / "fleet"
    card = fl.Card(queue_roots=(str(tmp_path / "q"),), queue_prefix="a-")
    update_ledger(
        fleet,
        "register",
        lambda led: fl.register(led, "a", card, chat="c", now=NOW),
        chat="c",
        args={"campaign": "a"},
    )
    assert "a" in read_ledger(fleet).campaigns
    events = [json.loads(x) for x in (fleet / "events.jsonl").read_text().splitlines()]
    assert events[0]["op"] == "register" and events[0]["campaign"] == "a"

    root = tmp_path / "q"
    store.create_queue(root, "a-1")
    store.append_jobs(root, "a-1", [Job("k", "true")])
    store.claim_next(root, "a-1", C4, os.getpid())
    rep = drain_report(card, C4)
    assert not rep.drained and rep.unpaused_roots and rep.open_claims == (("a-1", "k"),)
    store.stop_path(root, C4).touch()
    store.record_result(
        root,
        "a-1",
        Job("k", "true"),
        rc=0,
        node=C4,
        gpu=0,
        start_utc=NOW,
        end_utc=NOW,
        log="",
        attempt=0,
    )
    assert drain_report(card, C4).drained


def test_queue_snapshot_counts_and_cache_rereads_failed(tmp_path: Path) -> None:
    root = tmp_path / "q"
    card = fl.Card(queue_roots=(str(root),), queue_prefix="a-")
    store.create_queue(root, "a-1")
    store.append_jobs(
        root, "a-1", [Job(k, "true") for k in ("ok", "bad", "run", "wait")]
    )
    for key, rc in (("ok", 0), ("bad", 1)):
        job = store.claim_next(root, "a-1", C4, os.getpid())
        assert job.key == key
        store.record_result(
            root,
            "a-1",
            job,
            rc=rc,
            node=C4,
            gpu=0,
            start_utc=NOW,
            end_utc=NOW,
            log="",
            attempt=0,
        )
        store.remove_claim(root, "a-1", job.jobkey)
    store.claim_next(root, "a-1", C4, os.getpid())
    cache = ResultCache()
    snap = queue_snapshot(card, cache)
    assert snap.counts == {"done": 1, "failed": 1, "running": 1, "pending": 1}
    assert snap.failed_keys == frozenset({"a-1:bad"}) and snap.running_by_host == {
        C4: 1
    }
    bad = Job("bad", "true")
    store.requeue(root, "a-1", jobkeys=[bad.jobkey])
    time.sleep(0.01)
    store.record_result(
        root,
        "a-1",
        bad,
        rc=0,
        node=C4,
        gpu=0,
        start_utc=NOW,
        end_utc=NOW,
        log="",
        attempt=1,
    )
    snap = queue_snapshot(card, cache)
    assert snap.failed_keys == frozenset() and snap.counts["done"] == 2


def test_handback_blockers_need_pause_and_no_open_claims(tmp_path: Path) -> None:
    from subliminal_learning.fleet.ledger_io import handback_blockers

    root = tmp_path / "q"
    card = fl.Card(queue_roots=(str(root),), queue_prefix="a-")
    led = fl.register(fl.empty_ledger(), "a", card, chat="c", now=NOW)
    store.create_queue(root, "a-1")
    store.append_jobs(root, "a-1", [Job("k", "true")])
    job = store.claim_next(root, "a-1", C4, os.getpid())
    assert len(handback_blockers(led, "a", (C4,), require_paused=True)) == 2
    store.stop_path(root, C4).touch()
    assert len(handback_blockers(led, "a", (C4,), require_paused=True)) == 1
    store.record_result(
        root,
        "a-1",
        job,
        rc=0,
        node=C4,
        gpu=0,
        start_utc=NOW,
        end_utc=NOW,
        log="",
        attempt=0,
    )
    assert handback_blockers(led, "a", (C4,), require_paused=True) == []


T0 = 1_800_000_000.0
SCARD = fl.Card(
    queue_roots=("/q",), queue_prefix="s-", tmux_session="s", executor="slurm"
)
SCAMP = fl.Campaign("s", fl.CAMPAIGN_RUNNING, "c", SCARD, NOW)
SQUEUE = (
    "101|s-pool|RUNNING|gres/gpu:4\n"
    "102|s-pool|PENDING|gres/gpu:4\n"
    "103|s-pin.soar-subliminal-2|RUNNING|gres/gpu:2\n"
    "200|other-pool|RUNNING|gres/gpu:8\n"
)


def _view(**kw) -> SlurmRootView:
    base = {"root": "/q", "records": {"slurm-101.r0": {}}, "supervisor_age_s": 30.0}
    return SlurmRootView(**{**base, **kw})


def _slurm(view: SlurmRootView | None = None, **kw) -> SlurmSnapshot:
    base = {
        "jobs": parse_squeue(SQUEUE),
        "roots": (view or _view(),),
        "pools_window": True,
    }
    return SlurmSnapshot(**{**base, **kw})


def _sl(slurm, snap=None, memory=WatchMemory(primed=True), now=T0, camp=SCAMP):
    snap = snap or QueueSnapshot(counts={"pending": 3})
    return evaluate_slurm(camp, slurm, None, snap, memory, now)


def _alerts(lines: list[str]) -> list[str]:
    return [x for x in lines if x.startswith("ALERT")]


def test_slurm_healthy_pass_is_quiet_and_counts_pool_and_carrier_jobs() -> None:
    lines, mem = _sl(_slurm())
    assert not _alerts(lines)
    assert lines[:2] == [
        "SLURM jobs running=2 pending=1 gpus=6",
        "SLURM /q/slurm/s records=1 paused=0 supervisor_age=30s held_back=-",
    ]
    assert mem.leases == {} and mem.paused_roots == frozenset()


@pytest.mark.parametrize(
    ("kw", "alert"),
    [
        (
            {"view": _view(supervisor_age_s=None)},
            "ALERT slurm: supervisor heartbeat missing (/q/slurm/s)",
        ),
        (
            {"view": _view(supervisor_age_s=900.0)},
            "ALERT slurm: supervisor heartbeat stale (15 min, /q/slurm/s)",
        ),
        (
            {"pools_window": False},
            "ALERT slurm: tmux window s:pools missing (supervisor)",
        ),
        ({"oom15": 2}, "ALERT slurm: OutOfMemoryError in 2 job log(s), last 15 min"),
    ],
)
def test_slurm_supervisor_and_oom_alerts(kw: dict, alert: str) -> None:
    lines, _ = _sl(_slurm(**kw))
    assert _alerts(lines) == [alert]


def test_slurm_supervisor_checks_stop_once_done() -> None:
    down = _slurm(_view(supervisor_age_s=None), pools_window=False)
    lines, _ = _sl(down, snap=QueueSnapshot(counts={}, done=True))
    assert not _alerts(lines)
    assert "INFO CAMPAIGN_DONE s: done marker present; run the finish step" in lines


def test_slurm_paused_is_info_once_never_alert() -> None:
    paused = _slurm(_view(records={}, paused=True), jobs={})
    lines, mem = _sl(paused)
    assert "INFO SLURM PAUSED /q/slurm/s: no new pools, live pools drain" in lines
    assert not _alerts(lines)
    lines, mem = _sl(paused, memory=mem)
    assert not [x for x in lines if x.startswith("INFO SLURM")]
    lines, _ = _sl(_slurm(), memory=mem)
    assert "INFO SLURM resumed /q/slurm/s: PAUSED cleared" in lines


def test_slurm_squeue_failure_is_one_alert_and_keeps_dead_timers() -> None:
    memory = WatchMemory(primed=True, dead_since={"slurm-9.r0": T0 - 100})
    snap = QueueSnapshot(counts={}, running_by_host={"slurm-9.r0": 1})
    view = _view(records={"slurm-9.r0": {}})
    lines, mem = _sl(_slurm(view, jobs=None), snap=snap, memory=memory)
    assert _alerts(lines) == ["ALERT slurm: squeue unavailable; pool state unknown"]
    assert lines[0] == "SLURM jobs unknown"
    assert mem.dead_since == {"slurm-9.r0": T0 - 100}


def test_slurm_idle_exit_and_preempt_requeue_are_not_alerts() -> None:
    records = {
        "slurm-101.r0": {"ended_utc": NOW, "exit_reason": "idle"},
        "slurm-101.r1": {},
        "slurm-77.r0": {"ended_utc": NOW, "exit_reason": "signal"},
    }
    snap = QueueSnapshot(counts={}, running_by_host={"slurm-101.r1": 3})
    lines, mem = _sl(_slurm(_view(records=records)), snap=snap)
    assert not _alerts(lines) and mem.dead_since == {}


def test_slurm_dead_identity_claims_alert_only_after_15_min() -> None:
    old = datetime.fromtimestamp(T0 - 1000, timezone.utc).isoformat()
    records = {
        "slurm-101.r0": {},
        "slurm-101.r1": {},
        "slurm-99.r0": {},
        "slurm-98.r0": {"ended_utc": old},
        "slurm-97.r0": {"released_utc": old},
    }
    claims = {
        "slurm-101.r0": 1,
        "slurm-101.r1": 1,
        "slurm-99.r0": 2,
        "slurm-98.r0": 1,
        "slurm-97.r0": 1,
        "slurm-5.r0": 1,
        "soar-subliminal-2": 1,
    }
    snap = QueueSnapshot(counts={}, running_by_host=claims)
    slurm = _slurm(_view(records=records))
    lines, mem = _sl(slurm, snap=snap)
    assert _alerts(lines) == [
        "ALERT slurm: 1 open claim(s) of dead pool(s) unreleased > 15 min "
        "(supervisor sweep backlog): slurm-98.r0"
    ]
    assert set(mem.dead_since) == {"slurm-101.r0", "slurm-99.r0", "slurm-98.r0"}
    lines, _ = _sl(slurm, snap=snap, memory=mem, now=T0 + 901)
    assert _alerts(lines) == [
        "ALERT slurm: 4 open claim(s) of dead pool(s) unreleased > 15 min "
        "(supervisor sweep backlog): slurm-101.r0, slurm-98.r0, slurm-99.r0"
    ]


def test_slurm_tree_reads_state_files_logs_and_claims(tmp_path: Path) -> None:
    root = tmp_path / "q"
    sdir = root / "slurm" / "s"
    (sdir / "pools").mkdir(parents=True)
    (sdir / "pools" / "slurm-7.r0.json").write_text(json.dumps({"slurm_job_id": 7}))
    view = slurm_root_view(root, "s", T0)
    assert view.supervisor_age_s is None and not view.paused
    assert view.records == {"slurm-7.r0": {"slurm_job_id": 7}}
    stamp = datetime(2027, 1, 15, 8, tzinfo=timezone.utc)
    sup = {"updated_utc": stamp.isoformat(), "held_back": "user cap"}
    (sdir / "supervisor.json").write_text(json.dumps(sup))
    (sdir / "PAUSED").touch()
    view = slurm_root_view(root, "s", stamp.timestamp() + 42)
    assert view.supervisor_age_s == pytest.approx(42)
    assert view.paused and view.held_back == "user cap"

    store.create_queue(root, "s-1")
    store.append_jobs(root, "s-1", [Job("k", "true")])
    store.claim_next(root, "s-1", "slurm-7.r0", 1)
    card = fl.Card(queue_roots=(str(root),), queue_prefix="s-", executor="slurm")
    snap = queue_snapshot(card, ResultCache())
    assert snap.running_by_host == {"slurm-7.r0": 1}

    now = time.time()
    logs = []
    for name, text, age in (
        ("a.slurm-7.r0.log", "torch.OutOfMemoryError: CUDA", 60),
        ("b.slurm-8.r0.log", "torch.OutOfMemoryError: CUDA", 3600),
        ("c.slurm-7.r0.log", "fine", 60),
    ):
        path = tmp_path / name
        path.write_text(text)
        os.utime(path, (now - age, now - age))
        logs.append(path)
    assert recent_oom_logs(logs, now) == 1


def test_slurm_run_pass_writes_logs_and_state_without_ssh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, fleet, wdir = tmp_path / "q", tmp_path / "fleet", tmp_path / "w"
    wdir.mkdir()
    store.create_queue(root, "s-1")
    store.append_jobs(root, "s-1", [Job("k", "true")])
    card = fl.Card(
        queue_roots=(str(root),), queue_prefix="s-", tmux_session="s", executor="slurm"
    )
    update_ledger(
        fleet,
        "register",
        lambda led: fl.register(led, "s", card, chat="c", now=NOW),
        chat="c",
        args={},
    )
    monkeypatch.setattr(watch_io.slurm, "query_squeue", lambda: None)
    monkeypatch.setattr(watch_io, "probe_login", lambda c, s, h: (NodeProbe(h), False))
    monkeypatch.setattr(watch_io.subprocess, "run", None)  # no ssh / bash probes
    assert watch_io.run_pass("s", wdir, fleet, 0, 6, ResultCache())
    alerts = (wdir / "alerts.log").read_text()
    assert " ALERT slurm: squeue unavailable; pool state unknown\n" in alerts
    assert " ALERT slurm: supervisor heartbeat missing (" in alerts
    assert " ALERT slurm: tmux window s:pools missing (supervisor)\n" in alerts
    assert "INFO progress pending=1" in (wdir / "health.log").read_text()
    state = json.loads((wdir / ".watch_state.json").read_text())
    assert state["primed"] and "paused_roots" not in state


def test_ssh_watch_state_file_keeps_its_legacy_keys(tmp_path: Path) -> None:
    path = tmp_path / ".watch_state.json"
    watch_io._save_memory(path, WatchMemory(frozenset({"q:1"}), {"h": "x"}, 2))
    assert list(json.loads(path.read_text())) == [
        "failed_keys",
        "leases",
        "tracebacks",
        "done_reported",
        "primed",
    ]
    memory = WatchMemory(paused_roots=frozenset({"/q"}), dead_since={"slurm-1.r0": 1.0})
    watch_io._save_memory(path, memory)
    assert watch_io._load_memory(path) == memory


def test_login_probe_never_uses_ssh_and_parses_window() -> None:
    card = fl.Card(
        queue_roots=("/q",),
        executor="slurm",
        driver_pattern="scripts.campaign watch",
        driver_pane="s:driver",
    )
    script = login_probe_script(card, "s")
    assert "ssh" not in script and "list-windows -t =s" in script
    assert "scripts.campaign" not in script
