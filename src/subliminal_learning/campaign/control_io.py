"""Pause / resume / status IO for a live campaign: jobq stop sentinel + audit log.

Pause is not a kill. It writes jobq's per-host stop sentinel, which each worker
thread checks at the top of its claim loop and while waiting for a GPU
(`jobqueue.worker`), so the cell in hand trains to completion and writes its
manifest before the pool exits. Resume removes the sentinel; the launcher's
restart loop re-enters ``jobq work``, and the sweeps skip finished cells
(`utils.run_io.cell_is_done`), so no completed training is repeated. A cell
interrupted by a hard kill has no manifest (or an incomplete checkpoint dir), so
it is simply retrained from scratch on resume.

Every action appends one JSONL record to ``<run_set>/logs/<control_log_file>``:
one append-only history that the operator, the Discord sidecar, and an agent all
read to answer "when was this paused, why, and what was in flight".
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.jobqueue import store

PAUSE = "pause"
RESUME = "resume"
NOTE = "note"


def control_log_path(logs_dir: Path, spec: CampaignSpec) -> Path:
    """Append-only pause/resume audit log for this campaign."""
    return Path(logs_dir) / spec.control_log_file


def is_paused(queue_root: Path) -> bool:
    """Whether this host's jobq stop sentinel is set (workers draining/stopped)."""
    return store.stop_path(Path(queue_root), store.this_host()).is_file()


def worker_pool(queue_root: Path) -> dict:
    """This host's worker-pool pid-file state as ``{"pid": int|None, "alive": bool}``."""
    host = store.this_host()
    try:
        owner, pid_s = (
            store.worker_pid_path(Path(queue_root), host).read_text().strip().split(":")
        )
        pid = int(pid_s)
    except (OSError, ValueError):
        return {"pid": None, "alive": False}
    return {"pid": pid, "alive": owner == host and store.pid_alive(pid)}


def _in_flight(row: dict, queue: str) -> dict:
    """One running job's owner row, tagged with its queue and liveness."""
    pid = row.get("pid")
    alive = (
        store.pid_alive(int(pid))
        if isinstance(pid, int) and row.get("node") == store.this_host()
        else None
    )
    return {"queue": queue, **row, "alive": alive}


def snapshot(
    *,
    queue_root: Path,
    logs_dir: Path,
    spec: CampaignSpec,
    events_limit: int = 5,
) -> dict:
    """Everything needed to decide whether it is safe to pause, or how to resume.

    Args:
        queue_root: jobq root holding this campaign's queues.
        logs_dir: Campaign ``logs/`` dir (driver state + control log live here).
        spec: Frozen campaign constants (queue names, state/log filenames).
        events_limit: How many trailing control-log records to include.

    Returns:
        ``{host, paused, worker_pool, totals, queues, in_flight, transitions,
        campaign_done, recent_events}``. ``in_flight`` rows carry ``alive``
        (``False`` = orphaned claim from a hard kill; reclaimed on next launch).
    """
    root = Path(queue_root)
    queues: dict[str, dict] = {}
    in_flight: list[dict] = []
    totals = {"pending": 0, "running": 0, "done": 0, "failed": 0, "total": 0}
    for name in spec.queue_names():
        if not store.queue_exists(root, name):
            continue
        status = store.queue_status(root, name)
        running = [_in_flight(r, name) for r in status.pop("running_jobs", [])]
        in_flight.extend(running)
        queues[name] = status
        for key in totals:
            totals[key] += int(status.get(key, 0))

    state_path = Path(logs_dir) / spec.campaign_state_file
    transitions: list[str] = []
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        transitions = sorted((state.get("transitions") or {}).keys())

    return {
        "host": store.this_host(),
        "paused": is_paused(root),
        "worker_pool": worker_pool(root),
        "totals": totals,
        "queues": queues,
        "in_flight": in_flight,
        "transitions": transitions,
        "campaign_done": (Path(logs_dir) / spec.campaign_done_file).is_file(),
        "recent_events": read_events(logs_dir, spec, limit=events_limit),
    }


def read_events(
    logs_dir: Path, spec: CampaignSpec, *, limit: int | None = None
) -> list[dict]:
    """Control-log records oldest-first (trailing ``limit`` when given).

    Unparseable lines are skipped: the log is an operator audit trail, and one
    torn append must never make the history unreadable.
    """
    path = control_log_path(logs_dir, spec)
    if not path.is_file():
        return []
    events: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            logger.warning("skipping unparseable control-log line in {}", path)
    return events[-limit:] if limit is not None else events


def append_event(
    logs_dir: Path,
    spec: CampaignSpec,
    *,
    action: str,
    reason: str,
    snap: dict,
) -> dict:
    """Append one pause/resume/note record and return it.

    Args:
        logs_dir: Campaign ``logs/`` dir.
        spec: Frozen campaign constants.
        action: `PAUSE`, `RESUME`, or `NOTE`.
        reason: Free-text operator reason (required by the CLI).
        snap: `snapshot` taken at the moment of the action.
    """
    record = {
        "at": store.now_iso(),
        "action": action,
        "reason": reason,
        "host": snap["host"],
        "run_set_id": spec.run_set_id,
        "paused": snap["paused"],
        "worker_pool": snap["worker_pool"],
        "totals": snap["totals"],
        "transitions": snap["transitions"],
        "in_flight": [
            {k: row.get(k) for k in ("queue", "key", "gpu", "start_utc", "alive")}
            for row in snap["in_flight"]
        ],
    }
    path = control_log_path(logs_dir, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    return record


def pause(
    *, queue_root: Path, logs_dir: Path, spec: CampaignSpec, reason: str
) -> dict:
    """Set the stop sentinel, then record the pause. Returns the appended record.

    Idempotent: pausing an already-paused campaign re-touches the sentinel and
    still records the attempt (the log is the operator's history, not a state
    machine). In-flight cells keep training until they finish.
    """
    root = Path(queue_root)
    path = store.stop_path(root, store.this_host())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    snap = snapshot(queue_root=root, logs_dir=logs_dir, spec=spec)
    record = append_event(logs_dir, spec, action=PAUSE, reason=reason, snap=snap)
    logger.info(
        "paused {}: stop sentinel {} set; {} cell(s) still draining",
        spec.run_set_id,
        path,
        len(snap["in_flight"]),
    )
    return record


def resume(
    *, queue_root: Path, logs_dir: Path, spec: CampaignSpec, reason: str
) -> dict:
    """Clear the stop sentinel, then record the resume. Returns the appended record.

    The worker pool is not started here: the launcher's ``workers`` window
    re-enters ``jobq work`` on its own restart loop. When that loop was also
    killed, the record's ``worker_pool.alive`` is ``False`` — relaunch it.
    """
    root = Path(queue_root)
    store.stop_path(root, store.this_host()).unlink(missing_ok=True)
    snap = snapshot(queue_root=root, logs_dir=logs_dir, spec=spec)
    record = append_event(logs_dir, spec, action=RESUME, reason=reason, snap=snap)
    if not snap["worker_pool"]["alive"]:
        logger.warning(
            "stop sentinel cleared but no live worker pool on {} — relaunch the "
            "workers window (scripts/launch_breadth5_campaign.sh)",
            snap["host"],
        )
    else:
        logger.info("resumed {}: workers will claim again", spec.run_set_id)
    return record
