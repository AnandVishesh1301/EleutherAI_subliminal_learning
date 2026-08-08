"""Pause / resume control plane: stop sentinel + append-only audit log.

The campaign is paused for hours at a time by an operator who then relies on the
log to answer "what was in flight when I paused, and did it drain?". A pause that
silently fails to set the sentinel (or a log that loses records) would make an
agent resume onto work it thinks is finished, so these are the invariants worth
locking: the sentinel is the single source of paused-ness, and every action
appends exactly one record carrying the in-flight snapshot.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from subliminal_learning.campaign import breadth_plan, control_io
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job

SPEC = breadth_plan.SPEC
QUEUE = SPEC.q_a1


def _queue_with_running_job(root: Path) -> str:
    """A queue holding one claimed job owned by this (live) process."""
    store.create_queue(root, QUEUE)
    store.append_jobs(root, QUEUE, [Job(key="a1|same|bear|b1", cmd="true")])
    job = store.claim_next(root, QUEUE, store.this_host(), os.getpid())
    assert job is not None
    store.update_owner_gpu(root, QUEUE, job.jobkey, 3)
    return job.key


def test_pause_sets_sentinel_and_records_in_flight_cells(tmp_path: Path) -> None:
    root, logs = tmp_path / "queue", tmp_path / "logs"
    key = _queue_with_running_job(root)

    record = control_io.pause(
        queue_root=root, logs_dir=logs, spec=SPEC, reason="ssd maintenance"
    )

    assert control_io.is_paused(root)
    assert record["action"] == control_io.PAUSE
    assert record["reason"] == "ssd maintenance"
    assert record["paused"] is True
    assert record["totals"]["running"] == 1
    (in_flight,) = record["in_flight"]
    assert in_flight["key"] == key
    assert in_flight["gpu"] == 3
    assert in_flight["alive"] is True


def test_resume_clears_sentinel_and_appends_second_record(tmp_path: Path) -> None:
    root, logs = tmp_path / "queue", tmp_path / "logs"
    _queue_with_running_job(root)

    control_io.pause(queue_root=root, logs_dir=logs, spec=SPEC, reason="pause")
    control_io.resume(queue_root=root, logs_dir=logs, spec=SPEC, reason="back")

    assert not control_io.is_paused(root)
    actions = [e["action"] for e in control_io.read_events(logs, SPEC)]
    assert actions == [control_io.PAUSE, control_io.RESUME]


def test_repeated_pause_is_idempotent_on_state_and_append_only_on_history(
    tmp_path: Path,
) -> None:
    root, logs = tmp_path / "queue", tmp_path / "logs"
    _queue_with_running_job(root)

    control_io.pause(queue_root=root, logs_dir=logs, spec=SPEC, reason="first")
    control_io.pause(queue_root=root, logs_dir=logs, spec=SPEC, reason="again")

    assert control_io.is_paused(root)
    reasons = [e["reason"] for e in control_io.read_events(logs, SPEC)]
    assert reasons == ["first", "again"]


def test_read_events_survives_a_torn_line(tmp_path: Path) -> None:
    root, logs = tmp_path / "queue", tmp_path / "logs"
    control_io.pause(queue_root=root, logs_dir=logs, spec=SPEC, reason="ok")
    path = control_io.control_log_path(logs, SPEC)
    with path.open("a") as fh:
        fh.write('{"action": "pause", "trunc\n')
    control_io.resume(queue_root=root, logs_dir=logs, spec=SPEC, reason="back")

    assert [e["action"] for e in control_io.read_events(logs, SPEC)] == [
        control_io.PAUSE,
        control_io.RESUME,
    ]


def test_snapshot_flags_orphaned_claim_from_a_hard_kill(tmp_path: Path) -> None:
    """A claim owned by a dead pid must read as not-alive, not as healthy work."""
    root, logs = tmp_path / "queue", tmp_path / "logs"
    _queue_with_running_job(root)
    owner_path = root / QUEUE / "claims" / "a1__same__bear__b1" / "owner.json"
    owner = json.loads(owner_path.read_text())
    owner["pid"] = 2**22  # above pid_max: guaranteed dead
    owner_path.write_text(json.dumps(owner))

    snap = control_io.snapshot(queue_root=root, logs_dir=logs, spec=SPEC)

    assert snap["totals"]["running"] == 1
    assert snap["in_flight"][0]["alive"] is False


def test_snapshot_reports_worker_pool_liveness(tmp_path: Path) -> None:
    root, logs = tmp_path / "queue", tmp_path / "logs"
    root.mkdir(parents=True)
    assert control_io.worker_pool(root) == {"pid": None, "alive": False}

    store.worker_pid_path(root, store.this_host()).write_text(
        f"{store.this_host()}:{os.getpid()}"
    )
    assert control_io.worker_pool(root)["alive"] is True

    snap = control_io.snapshot(queue_root=root, logs_dir=logs, spec=SPEC)
    assert snap["worker_pool"]["pid"] == os.getpid()
    assert snap["queues"] == {}


def test_snapshot_includes_driver_transitions_and_done_marker(tmp_path: Path) -> None:
    root, logs = tmp_path / "queue", tmp_path / "logs"
    logs.mkdir(parents=True)
    (logs / SPEC.campaign_state_file).write_text(
        json.dumps({"transitions": {"submit_initial": {"at": "x"}, "submit_a3": {}}})
    )
    (logs / SPEC.campaign_done_file).write_text("done")

    snap = control_io.snapshot(queue_root=root, logs_dir=logs, spec=SPEC)

    assert snap["transitions"] == ["submit_a3", "submit_initial"]
    assert snap["campaign_done"] is True
