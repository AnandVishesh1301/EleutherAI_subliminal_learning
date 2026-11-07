"""Generic campaign watcher: alert rules over node probes, queue files and leases.

One pass turns what ``watch_io`` measured into ``ALERT`` / ``INFO`` lines. Leases
decide what is expected on each node: a node the campaign lent away or paused is
reported, never alerted on for a missing pool, and a campaign pool or cell found on
a node another campaign holds is an alert. Slurm cards (``executor="slurm"``) have no
leases or nodes: ``evaluate_slurm`` judges the supervisor, pool records and squeue.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

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
from subliminal_learning.jobqueue.slurm import (
    LIVE_STATES,
    PIN_SUFFIX,
    POOL_SUFFIX,
    SqueueJob,
    dead_identities,
)

STATE_STALE_S = 1200.0
MAX_FAILED_LINES = 20
SUPERVISOR_STALE_S = 600.0
DEAD_CLAIM_S = 900.0
OOM_SCAN_SKIPPED = -1


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
        paused_roots: Queue roots whose slurm ``PAUSED`` marker was set.
        dead_since: Dead ``slurm-*`` pool identity holding claims -> epoch it is
            known dead since.
    """

    failed_keys: frozenset[str] = frozenset()
    leases: Mapping[str, str] = field(default_factory=dict)
    tracebacks: int = 0
    done_reported: bool = False
    primed: bool = False
    paused_roots: frozenset[str] = frozenset()
    dead_since: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SlurmRootView:
    """One queue root's ``slurm/<campaign>`` state files.

    Attributes:
        root: Queue root.
        records: Pool records by identity (``jobqueue.slurm.pool_records``).
        supervisor_age_s: Age of ``supervisor.json``'s ``updated_utc`` (None = no
            file).
        held_back: The supervisor's last ``held_back`` reason.
        paused: ``PAUSED`` exists.
    """

    root: str
    records: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    supervisor_age_s: float | None = None
    held_back: str = ""
    paused: bool = False


@dataclass(frozen=True)
class SlurmSnapshot:
    """Login-node view of a slurm campaign.

    Attributes:
        jobs: The user's squeue jobs by id (None = squeue unavailable).
        roots: One view per queue root.
        pools_window: tmux window ``<tmux_session>:pools`` exists.
        oom15: Job logs (any pool identity) with ``OutOfMemoryError`` written in
            the last 15 min.
    """

    jobs: Mapping[str, SqueueJob] | None
    roots: tuple[SlurmRootView, ...] = ()
    pools_window: bool = False
    oom15: int = 0


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


def _campaign_lines(
    camp: Campaign,
    driver: NodeProbe | None,
    snap: QueueSnapshot,
    memory: WatchMemory,
    disk_min_gb: Mapping[str, int],
) -> tuple[list[str], int]:
    """Driver, failed-result, stale-state and disk lines plus the traceback count."""
    name, card = camp.name, camp.card
    lines: list[str] = []
    live = camp.status == CAMPAIGN_RUNNING and not snap.done
    tracebacks = memory.tracebacks
    if card.driver_pattern and driver is not None and driver.reachable:
        if driver.driver == 0 and live:
            lines.append(f"ALERT driver ({card.driver_pattern}) not running")
        if driver.tracebacks is not None:
            if driver.tracebacks > memory.tracebacks:
                lines.append(
                    f"ALERT new Traceback in the driver pane ({driver.tracebacks} total)"
                )
            tracebacks = driver.tracebacks
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
    return lines, tracebacks


def _done_lines(name: str, snap: QueueSnapshot, memory: WatchMemory) -> list[str]:
    if snap.done and not memory.done_reported:
        return [f"INFO CAMPAIGN_DONE {name}: done marker present; run the finish step"]
    return []


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
    driver = probes.get(card.driver_node) if card.driver_node else None
    shared, tracebacks = _campaign_lines(camp, driver, snap, memory, disk_min_gb)
    lines.extend(shared)
    leases = {h: lease_summary(ledger, name, h, snap) for h in hosts}
    for host in sorted(set(leases) | set(memory.leases)):
        old, new = memory.leases.get(host), leases.get(host, "not involved")
        if old is not None and old != new:
            lines.append(f"INFO LEASE {resolve_node(host).short}: {old} -> {new}")
    lines.extend(_done_lines(name, snap, memory))
    memory = WatchMemory(
        snap.failed_keys, leases, tracebacks, memory.done_reported or snap.done, True
    )
    return lines, memory


def campaign_jobs(jobs: Mapping[str, SqueueJob], name: str) -> list[SqueueJob]:
    """The campaign's ``<name>-pool`` and ``<name>-pin.<node>`` jobs."""
    return [
        j
        for j in jobs.values()
        if j.name == f"{name}{POOL_SUFFIX}" or j.name.startswith(f"{name}{PIN_SUFFIX}")
    ]


def _epoch(stamp: Any) -> float | None:
    try:
        return datetime.fromisoformat(str(stamp)).timestamp()
    except ValueError:
        return None


def dead_claim_holders(
    slurm: SlurmSnapshot,
    claims_by_node: Mapping[str, int],
    seen: Mapping[str, float],
    now_s: float,
) -> dict[str, float]:
    """Dead ``slurm-*`` pool identities that still hold open claims.

    Args:
        slurm: Snapshot with known squeue jobs.
        claims_by_node: Open claims per owner node.
        seen: The previous pass's result.
        now_s: Current epoch seconds.

    Returns:
        Identity -> epoch it is known dead since (its ``ended_utc``, else the
        first pass that saw it dead).
    """
    out: dict[str, float] = {}
    for view in slurm.roots:
        for node in dead_identities(view.records, slurm.jobs or {}):
            if claims_by_node.get(node, 0) > 0:
                ended = _epoch(view.records[node].get("ended_utc"))
                out[node] = min(
                    t for t in (ended, seen.get(node), now_s) if t is not None
                )
    return out


def _slurm_lines(
    camp: Campaign,
    slurm: SlurmSnapshot,
    snap: QueueSnapshot,
    memory: WatchMemory,
    now_s: float,
) -> tuple[list[str], dict[str, float]]:
    """Squeue, supervisor, PAUSED, OOM and dead-claim lines plus ``dead_since``."""
    name = camp.name
    live = camp.status == CAMPAIGN_RUNNING and not snap.done
    lines: list[str] = []
    if slurm.jobs is None:
        lines.append("SLURM jobs unknown")
        lines.append("ALERT slurm: squeue unavailable; pool state unknown")
    else:
        mine = campaign_jobs(slurm.jobs, name)
        running = [j for j in mine if j.state in LIVE_STATES]
        lines.append(
            f"SLURM jobs running={len(running)} "
            f"pending={sum(j.state == 'PENDING' for j in mine)} "
            f"gpus={sum(j.gpus for j in running)}"
        )
    for view in slurm.roots:
        sdir = f"{view.root}/slurm/{name}"
        age = view.supervisor_age_s
        lines.append(
            f"SLURM {sdir} records={len(view.records)} paused={int(view.paused)} "
            f"supervisor_age={'-' if age is None else f'{age:.0f}s'} "
            f"held_back={view.held_back or '-'}"
        )
        if view.paused and view.root not in memory.paused_roots:
            lines.append(f"INFO SLURM PAUSED {sdir}: no new pools, live pools drain")
        elif not view.paused and view.root in memory.paused_roots:
            lines.append(f"INFO SLURM resumed {sdir}: PAUSED cleared")
        if not live:
            continue
        if age is None:
            lines.append(f"ALERT slurm: supervisor heartbeat missing ({sdir})")
        elif age > SUPERVISOR_STALE_S:
            lines.append(
                f"ALERT slurm: supervisor heartbeat stale ({age / 60:.0f} min, {sdir})"
            )
    if live and not slurm.pools_window:
        session = camp.card.tmux_session or name
        lines.append(f"ALERT slurm: tmux window {session}:pools missing (supervisor)")
    if slurm.oom15 == OOM_SCAN_SKIPPED:
        lines.append("SLURM oom log scan timed out; a failed cell still alerts")
    elif slurm.oom15:
        lines.append(
            f"ALERT slurm: OutOfMemoryError in {slurm.oom15} job log(s), last 15 min"
        )
    if slurm.jobs is None:
        return lines, dict(memory.dead_since)
    dead = dead_claim_holders(slurm, snap.running_by_host, memory.dead_since, now_s)
    late = sorted(n for n, t in dead.items() if now_s - t > DEAD_CLAIM_S)
    if late:
        ids = ", ".join(late[:5]) + (" ..." if len(late) > 5 else "")
        lines.append(
            f"ALERT slurm: {sum(snap.running_by_host[n] for n in late)} open "
            f"claim(s) of dead pool(s) unreleased > {DEAD_CLAIM_S / 60:.0f} min "
            f"(supervisor sweep backlog): {ids}"
        )
    return lines, dead


def evaluate_slurm(
    camp: Campaign,
    slurm: SlurmSnapshot,
    driver: NodeProbe | None,
    snap: QueueSnapshot,
    memory: WatchMemory,
    now_s: float,
) -> tuple[list[str], WatchMemory]:
    """One watcher pass over a slurm campaign (no leases, all probes local).

    Pool exits (idle, claim window, preemption requeue) are normal and never
    alerted on; a dead pool only matters while its claims stay unreleased.

    Args:
        camp: The campaign (card ``executor="slurm"``).
        slurm: Squeue + supervisor / pool state view.
        driver: Local probe of the driver process and pane.
        snap: Queue files view.
        memory: Previous pass.
        now_s: Current epoch seconds.

    Returns:
        Health lines and the memory for the next pass.
    """
    lines, dead = _slurm_lines(camp, slurm, snap, memory, now_s)
    shared, tracebacks = _campaign_lines(
        camp, driver, snap, memory, camp.card.disk_min_gb
    )
    lines.extend(shared)
    lines.extend(_done_lines(camp.name, snap, memory))
    memory = WatchMemory(
        snap.failed_keys,
        {},
        tracebacks,
        memory.done_reported or snap.done,
        True,
        frozenset(v.root for v in slurm.roots if v.paused),
        dead,
    )
    return lines, memory


def progress_line(snap: QueueSnapshot) -> str:
    """``INFO progress`` totals line."""
    keys = ("pending", "running", "done", "failed")
    return "INFO progress " + " ".join(f"{k}={snap.counts.get(k, 0)}" for k in keys)
