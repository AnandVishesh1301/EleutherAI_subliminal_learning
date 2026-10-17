"""Generic campaign watcher: alert rules over node probes, queue files and leases.

One pass turns what ``watch_io`` measured into ``ALERT`` / ``INFO`` lines. Leases
decide what is expected on each node: a node the campaign lent away or paused is
reported, never alerted on for a missing pool, and a campaign pool or cell found on
a node another campaign holds is an alert.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from subliminal_learning.fleet.ledger import (
    CAMPAIGN_RUNNING,
    DRAINING,
    PAUSED,
    RETURNING,
    RUNNING,
    Campaign,
    Ledger,
)
from subliminal_learning.fleet.nodes import resolve_node

STATE_STALE_S = 1200.0
MAX_FAILED_LINES = 20


@dataclass(frozen=True)
class NodeProbe:
    """What one node reported (pure process / tmux / GPU reads).

    Attributes:
        host: Hostname.
        reachable: The probe ran and parsed.
        pools: Live jobq pool processes on the campaign's roots.
        session: The campaign tmux session exists.
        oom15: Job logs with ``OutOfMemoryError`` written in the last 15 min.
        driver: Live driver processes (driver node only).
        tracebacks: ``Traceback`` count in the driver pane (driver node only).
        foreign_jobs: Other tenants' GPU jobs (shared node only).
        busiest_gpu: GPU with the most live jobq cells on the node (any pool).
        busiest_cells: Live jobq cells on that GPU.
        error: Probe failure text.
    """

    host: str
    reachable: bool = True
    pools: int = 0
    session: bool = False
    oom15: int = 0
    driver: int | None = None
    tracebacks: int | None = None
    foreign_jobs: int = 0
    busiest_gpu: int | None = None
    busiest_cells: int = 0
    error: str = ""


@dataclass(frozen=True)
class QueueSnapshot:
    """File-level view of the campaign's queues.

    Attributes:
        counts: Totals by state (pending / running / done / failed).
        failed_keys: Keys with an rc != 0 result.
        paused_hosts: Hosts with a stop sentinel on any campaign root.
        running_by_host: Open claims per host.
        caps: ``cap_per_gpu`` per host from the roots' GPU policies.
        state_age_s: Age of the driver state file.
        done: The done marker exists.
        disk_free_gb: Free GB per watched mount.
    """

    counts: Mapping[str, int]
    failed_keys: frozenset[str] = frozenset()
    paused_hosts: frozenset[str] = frozenset()
    running_by_host: Mapping[str, int] = field(default_factory=dict)
    caps: Mapping[str, int] = field(default_factory=dict)
    state_age_s: float | None = None
    done: bool = False
    disk_free_gb: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class WatchMemory:
    """What the previous pass saw (persisted between passes).

    Attributes:
        failed_keys: Failed keys already alerted.
        leases: Host -> lease summary relevant to the campaign.
        tracebacks: Driver-pane traceback count.
        done_reported: CAMPAIGN_DONE already announced.
        primed: A pass already ran (the first pass reports existing failures as
            one count instead of one alert each).
    """

    failed_keys: frozenset[str] = frozenset()
    leases: Mapping[str, str] = field(default_factory=dict)
    tracebacks: int = 0
    done_reported: bool = False
    primed: bool = False


def watched_hosts(
    ledger: Ledger, campaign: str, snap: QueueSnapshot
) -> tuple[str, ...]:
    """Hosts the campaign holds, lent away, or still has claims on."""
    return tuple(
        h
        for h, lease in ledger.nodes.items()
        if lease.holder == campaign
        or campaign in lease.lenders
        or lease.target == campaign
        or snap.running_by_host.get(h, 0) > 0
    )


def lease_summary(ledger: Ledger, campaign: str, host: str, snap: QueueSnapshot) -> str:
    """Compact lease text used to detect lease changes between passes."""
    lease = ledger.nodes[host]
    text = f"{lease.state} holder={lease.holder or '-'}"
    if lease.target:
        text += f" for={lease.target} mode={lease.mode}"
    if lease.lenders:
        text += f" lenders={'>'.join(lease.lenders)}"
    if (
        lease.state == DRAINING
        and lease.holder == campaign
        and host in snap.paused_hosts
        and snap.running_by_host.get(host, 0) == 0
    ):
        text += " drained"
    return text


def _node_lines(
    camp: Campaign,
    ledger: Ledger,
    host: str,
    probe: NodeProbe | None,
    snap: QueueSnapshot,
) -> list[str]:
    name, card = camp.name, camp.card
    short = resolve_node(host).short
    lease = ledger.nodes[host]
    running = snap.running_by_host.get(host, 0)
    paused = host in snap.paused_hosts
    lines: list[str] = []
    if probe is None or not probe.reachable:
        if lease.holder == name:
            err = probe.error if probe else "not probed"
            lines.append(f"ALERT {short}: unreachable or probe failed ({err})")
        return lines
    lines.append(
        f"NODE {short} lease={lease.state}/{lease.holder or '-'} pools={probe.pools} "
        f"session={int(probe.session)} cells={running} paused={int(paused)} "
        f"oom15={probe.oom15}"
    )
    cap = snap.caps.get(host)
    if cap is not None and probe.busiest_cells > cap:
        lines.append(
            f"ALERT {short}: {probe.busiest_cells} jobq cells on GPU {probe.busiest_gpu} "
            f"> cap_per_gpu {cap}"
        )
    if probe.oom15:
        lines.append(
            f"ALERT {short}: OutOfMemoryError in {probe.oom15} job log(s), last 15 min"
        )
    if lease.holder != name:
        holder = lease.holder or "nobody (free)"
        if running:
            lines.append(
                f"ALERT {short}: {running} {name} cell(s) running on a node {holder} holds"
            )
        if probe.pools and not paused:
            lines.append(
                f"ALERT {short}: {name} pool live without a stop sentinel on a node "
                f"{holder} holds"
            )
        return lines
    if lease.state == RUNNING:
        if paused:
            lines.append(
                f"ALERT {short}: stop sentinel set but the ledger says {name} runs here "
                "(record it with fleet pause, or clear it)"
            )
        else:
            if not probe.pools and not snap.done:
                lines.append(f"ALERT {short}: no {name} jobq pool")
            if card.tmux_session and not probe.session:
                lines.append(f"ALERT {short}: tmux session {card.tmux_session} missing")
        if resolve_node(host).shared and probe.foreign_jobs:
            lines.append(
                f"ALERT {short}: {probe.foreign_jobs} foreign GPU job(s); reclaim with "
                "/occupy-gpus"
            )
    elif lease.state == DRAINING:
        lines.append(
            f"INFO {short} draining for {lease.target} ({lease.mode}): {running} cell(s) left"
        )
    elif lease.state == PAUSED:
        lines.append(f"INFO {short} paused ({running} cell(s) finishing)")
    elif lease.state == RETURNING:
        lines.append(f"INFO {short} returned to {name}: resume it (fleet resume)")
    return lines


def evaluate(
    camp: Campaign,
    ledger: Ledger,
    probes: Mapping[str, NodeProbe],
    snap: QueueSnapshot,
    memory: WatchMemory,
    disk_min_gb: Mapping[str, int],
) -> tuple[list[str], WatchMemory]:
    """One watcher pass: health + alert lines and the memory for the next pass."""
    name, card = camp.name, camp.card
    lines: list[str] = []
    hosts = watched_hosts(ledger, name, snap)
    for host in hosts:
        lines.extend(_node_lines(camp, ledger, host, probes.get(host), snap))
    live = camp.status == CAMPAIGN_RUNNING and not snap.done
    tracebacks = memory.tracebacks
    if card.driver_node and card.driver_pattern:
        probe = probes.get(card.driver_node)
        if probe is not None and probe.reachable:
            if probe.driver == 0 and live:
                lines.append(f"ALERT driver ({card.driver_pattern}) not running")
            if probe.tracebacks is not None:
                if probe.tracebacks > memory.tracebacks:
                    lines.append(
                        f"ALERT new Traceback in the driver pane ({probe.tracebacks} total)"
                    )
                tracebacks = probe.tracebacks
    new_failed = sorted(snap.failed_keys - memory.failed_keys)
    if not memory.primed:
        lines.append(
            f"INFO watcher started for {name}: {len(new_failed)} failed result(s) present"
        )
        new_failed = []
    for key in new_failed[:MAX_FAILED_LINES]:
        lines.append(f"ALERT new failed result: {key}")
    if len(new_failed) > MAX_FAILED_LINES:
        lines.append(
            f"ALERT {len(new_failed) - MAX_FAILED_LINES} more new failed results"
        )
    if snap.state_age_s is not None and snap.state_age_s > STATE_STALE_S and live:
        lines.append(f"ALERT driver state file stale ({snap.state_age_s / 60:.0f} min)")
    for mount, min_gb in sorted(disk_min_gb.items()):
        free = snap.disk_free_gb.get(mount)
        if free is not None and free < min_gb:
            lines.append(f"ALERT {mount} free {free}G < {min_gb}G")
    leases = {h: lease_summary(ledger, name, h, snap) for h in hosts}
    for host in sorted(set(leases) | set(memory.leases)):
        old, new = memory.leases.get(host), leases.get(host, "not involved")
        if old is not None and old != new:
            lines.append(f"INFO LEASE {resolve_node(host).short}: {old} -> {new}")
    done_reported = memory.done_reported
    if snap.done and not done_reported:
        lines.append(
            f"INFO CAMPAIGN_DONE {name}: done marker present; run the finish step"
        )
        done_reported = True
    memory = WatchMemory(
        snap.failed_keys, leases, tracebacks, done_reported, primed=True
    )
    return lines, memory


def progress_line(snap: QueueSnapshot) -> str:
    """``INFO progress`` totals line."""
    keys = ("pending", "running", "done", "failed")
    return "INFO progress " + " ".join(f"{k}={snap.counts.get(k, 0)}" for k in keys)
