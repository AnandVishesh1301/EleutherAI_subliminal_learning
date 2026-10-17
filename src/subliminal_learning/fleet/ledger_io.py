"""Ledger file IO (locked read-modify-write + event log) and the read-only drain check."""

from __future__ import annotations

import fcntl
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from subliminal_learning.fleet.ledger import (
    CAMPAIGN_DONE,
    RUNNING,
    Card,
    Ledger,
    empty_ledger,
    ledger_from_dict,
    ledger_to_dict,
)
from subliminal_learning.jobqueue import store
from subliminal_learning.utils.io import atomic_write_json

LEDGER_FILE = "ledger.json"
EVENTS_FILE = "events.jsonl"
LOCK_FILE = "ledger.lock"


def now_utc() -> str:
    """Current UTC time, second resolution."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_ledger(fleet_root: Path) -> Ledger:
    """The stored ledger, or an all-free one when none exists yet."""
    path = Path(fleet_root) / LEDGER_FILE
    if not path.exists():
        return empty_ledger()
    return ledger_from_dict(json.loads(path.read_text()))


def update_ledger(
    fleet_root: Path,
    op: str,
    change: Callable[[Ledger], Ledger],
    *,
    chat: str | None,
    args: Mapping[str, Any],
) -> Ledger:
    """Apply ``change`` under an exclusive lock and append the event.

    The new ledger is written before the event line, so a crash between the two
    loses only the audit line.
    """
    root = Path(fleet_root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / LOCK_FILE, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        new = change(read_ledger(root))
        atomic_write_json(root / LEDGER_FILE, ledger_to_dict(new), indent=2)
        event = {"utc": now_utc(), "op": op, "chat": chat, **args}
        with open(root / EVENTS_FILE, "a") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
    return new


@dataclass(frozen=True)
class DrainReport:
    """What still ties a campaign to one node.

    Attributes:
        host: Node hostname.
        unpaused_roots: Queue roots without ``stop.<host>``.
        open_claims: ``(queue, key)`` claims owned by ``host`` with no result.
    """

    host: str
    unpaused_roots: tuple[str, ...]
    open_claims: tuple[tuple[str, str], ...]

    @property
    def drained(self) -> bool:
        """Every root paused on the node and no claim of the campaign left there."""
        return not self.unpaused_roots and not self.open_claims


def sentinel_mismatches(ledger: Ledger) -> list[str]:
    """Stop sentinels on live campaigns' roots for nodes the ledger says they run on."""
    out: list[str] = []
    for name, camp in sorted(ledger.campaigns.items()):
        if camp.status == CAMPAIGN_DONE:
            continue
        for root in camp.card.queue_roots:
            for stop in sorted(Path(root).glob("stop.*")):
                lease = ledger.nodes.get(stop.name.removeprefix("stop."))
                if (
                    lease is not None
                    and lease.holder == name
                    and lease.state == RUNNING
                ):
                    out.append(f"{stop} is set but the ledger says {name} runs there")
    return out


def campaign_queues(card: Card, root: Path) -> list[str]:
    """Queue names of the campaign under one of its roots."""
    return [q for q in store.list_queues(root) if q.startswith(card.queue_prefix)]


def drain_report(card: Card, host: str) -> DrainReport:
    """Pure file reads: stop sentinels and open claims of ``card`` on ``host``."""
    unpaused: list[str] = []
    claims: list[tuple[str, str]] = []
    for root_s in card.queue_roots:
        root = Path(root_s)
        if not store.stop_path(root, host).exists():
            unpaused.append(root_s)
        for queue in campaign_queues(card, root):
            claims.extend((queue, jk) for jk, _ in store.open_claims(root, queue, host))
    return DrainReport(host, tuple(unpaused), tuple(claims))


def handback_blockers(
    ledger: Ledger, campaign: str, hosts: tuple[str, ...], *, require_paused: bool
) -> list[str]:
    """Why ``campaign`` cannot give up ``hosts`` yet (empty list = it can).

    Args:
        ledger: Current ledger.
        campaign: Campaign letting go of the nodes.
        hosts: Hostnames it lets go of.
        require_paused: Also require the stop sentinel on every root (a mid-run
            release; a finished campaign only needs no open claim).
    """
    card = ledger.campaigns[campaign].card
    out: list[str] = []
    for host in hosts:
        rep = drain_report(card, host)
        if rep.open_claims:
            out.append(f"{host}: {len(rep.open_claims)} open claim(s) of {campaign}")
        if require_paused and rep.unpaused_roots:
            out.append(f"{host}: {campaign} not paused on {list(rep.unpaused_roots)}")
    return out
