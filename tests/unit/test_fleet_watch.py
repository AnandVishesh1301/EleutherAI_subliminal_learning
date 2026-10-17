from __future__ import annotations

import json
import os
import time
from pathlib import Path

from subliminal_learning.fleet import ledger as fl
from subliminal_learning.fleet.ledger_io import drain_report, read_ledger, update_ledger
from subliminal_learning.fleet.watch import (
    NodeProbe,
    QueueSnapshot,
    WatchMemory,
    evaluate,
)
from subliminal_learning.fleet.watch_io import (
    ResultCache,
    _unselfmatch,
    parse_probe,
    probe_script,
    queue_snapshot,
)
from subliminal_learning.jobqueue import store
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
