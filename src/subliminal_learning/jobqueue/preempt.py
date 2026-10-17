"""Preempt this node's running cells under one queue root (``jobq stop --now``).

The stop sentinel must already be set, so the pool claims nothing new. Each live
cell's process group gets SIGTERM, then SIGKILL after ``grace_s``. A killed cell is
requeued once its worker has recorded the failed result (or at once if that worker
is gone), keeping the attempt counter so the kill costs no retry. Every kill is
appended to ``<root>/preemptions.<host>.jsonl``.
"""

from __future__ import annotations

import json
import os
import signal
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from subliminal_learning.jobqueue import store

REQUEUED, FINISHED = "requeued", "finished"


@dataclass
class _Victim:
    queue: str
    jobkey: str
    cell_pid: int
    term_ts: float
    members: dict[int, str]
    sigkilled: bool = False
    outcome: str | None = None

    def alive(self) -> bool:
        """Any process of the cell's tree (captured at kill time) still runs."""
        return any(_start_time(pid) == st for pid, st in self.members.items())


@dataclass(frozen=True)
class PreemptReport:
    """Outcome of preempting one host under one root.

    Attributes:
        requeued: ``(queue, jobkey)`` killed and pending again.
        finished: Killed cells that had already recorded rc 0 (left done).
        released: Claims dropped because their worker and cell were already gone.
        unsettled: Killed cells still unaccounted for at the timeout.
        waiting: Claims of a live worker with no live cell at the timeout.
    """

    requeued: tuple[tuple[str, str], ...]
    finished: tuple[tuple[str, str], ...]
    released: tuple[tuple[str, str], ...]
    unsettled: tuple[tuple[str, str], ...]
    waiting: tuple[tuple[str, str], ...]

    @property
    def ok(self) -> bool:
        """Nothing of this host is left running or unaccounted for."""
        return not self.unsettled and not self.waiting


def _start_time(pid: int) -> str | None:
    """Start-time field of a live, non-zombie ``pid``; None otherwise."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None
    if len(fields) < 20 or fields[0] == "Z":
        return None
    return fields[19]


def _tree(pid: int) -> dict[int, str]:
    """``pid`` and its descendants (pid -> start time), via ``/proc/*/task/*/children``.

    A full ``/proc`` scan is too slow on these pods (hundreds of thousands of pids).
    """
    members: dict[int, str] = {}
    stack = [pid]
    while stack:
        p = stack.pop()
        st = _start_time(p)
        if st is None or p in members:
            continue
        members[p] = st
        for children in Path(f"/proc/{p}/task").glob("*/children"):
            try:
                stack.extend(int(c) for c in children.read_text().split())
            except (OSError, ValueError):
                continue
    return members


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


def _signal_pid(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass


def _worker_alive(owner: dict) -> bool:
    pid = owner.get("pid")
    return isinstance(pid, int) and store.pid_alive(pid)


def _log_kill(root: Path, host: str, victim: _Victim, gpu: int | None) -> None:
    row = {
        "utc": store.now_iso(),
        "queue": victim.queue,
        "jobkey": victim.jobkey,
        "cell_pid": victim.cell_pid,
        "gpu": gpu,
    }
    try:
        with open(root / f"preemptions.{host}.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError:
        logger.exception(
            "could not log preemption of {} {}", victim.queue, victim.jobkey
        )


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _settle(root: Path, host: str, victim: _Victim) -> str | None:
    """Requeue a killed cell once accounted for; None while it is still settling.

    The claim is read before the result because the worker writes the result and
    then drops the claim. A claim or result that is not this victim's (another
    host's run after someone else requeued it) is left alone. Requeueing waits for
    the whole killed tree to exit, so no straggler still holds the GPU or run dir.
    """
    qdir = store.queue_dir(root, victim.queue)
    claim = qdir / "claims" / victim.jobkey
    result = qdir / "results" / f"{victim.jobkey}.json"
    if claim.exists():
        owner = _read_json(claim / "owner.json")
        if owner is None:
            return None
        if owner.get("node") != host or owner.get("cell_pid") != victim.cell_pid:
            return REQUEUED
        if not result.exists():
            if not _worker_alive(owner) and not victim.alive():
                store.remove_claim(root, victim.queue, victim.jobkey)
                return REQUEUED
            return None
    elif not result.exists():
        return REQUEUED
    res = _read_json(result)
    if res is None:
        return None
    if res.get("node") != host:
        return REQUEUED
    if int(res.get("rc", 1)) == 0:
        return FINISHED
    if victim.alive():
        return None
    store.requeue(root, victim.queue, jobkeys=[victim.jobkey], count_attempt=False)
    return REQUEUED


def preempt_host(
    root: Path,
    host: str,
    *,
    prefix: str = "",
    grace_s: float = 60.0,
    timeout_s: float = 900.0,
    poll_s: float = 2.0,
    log: Callable[[str], None] = print,
) -> PreemptReport:
    """Kill and requeue every running cell of ``host`` in queues named ``prefix*``.

    Raises:
        RuntimeError: ``stop.<host>`` is not set under ``root``.
    """
    root = Path(root)
    if not store.stop_path(root, host).exists():
        raise RuntimeError(f"set {store.stop_path(root, host)} before preempting")
    victims: dict[tuple[str, str], _Victim] = {}
    released: list[tuple[str, str]] = []
    deadline = time.time() + timeout_s
    while True:
        waiting: list[tuple[str, str]] = []
        for queue in store.list_queues(root):
            if not queue.startswith(prefix):
                continue
            for jobkey, owner in store.open_claims(root, queue, host):
                key = (queue, jobkey)
                if key in victims:
                    continue
                if store.cell_alive(owner):
                    pid = int(owner["cell_pid"])
                    victim = _Victim(queue, jobkey, pid, time.time(), _tree(pid))
                    _log_kill(root, host, victim, owner.get("gpu"))
                    _signal_group(victim.cell_pid, signal.SIGTERM)
                    log(f"SIGTERM {queue} {jobkey} pgid={victim.cell_pid}")
                    victims[key] = victim
                elif not _worker_alive(owner):
                    store.remove_claim(root, queue, jobkey)
                    released.append(key)
                else:
                    waiting.append(key)
        for victim in victims.values():
            if victim.outcome is None:
                victim.outcome = _settle(root, host, victim)
            if (
                victim.outcome is None
                and not victim.sigkilled
                and time.time() - victim.term_ts > grace_s
                and victim.alive()
            ):
                _signal_group(victim.cell_pid, signal.SIGKILL)
                for pid, st in victim.members.items():
                    if _start_time(pid) == st:
                        _signal_pid(pid, signal.SIGKILL)
                victim.sigkilled = True
                log(f"SIGKILL {victim.queue} {victim.jobkey} pgid={victim.cell_pid}")
        unsettled = [k for k, v in victims.items() if v.outcome is None]
        if (not waiting and not unsettled) or time.time() >= deadline:
            break
        time.sleep(poll_s)
    return PreemptReport(
        requeued=tuple(k for k, v in victims.items() if v.outcome == REQUEUED),
        finished=tuple(k for k, v in victims.items() if v.outcome == FINISHED),
        released=tuple(released),
        unsettled=tuple(unsettled),
        waiting=tuple(waiting),
    )
