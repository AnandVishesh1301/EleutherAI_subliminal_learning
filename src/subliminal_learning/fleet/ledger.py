"""Fleet lease ledger: which campaign holds which node, as pure state transitions.

A node is ``free`` or held by one campaign in one state. ``lenders`` is the stack of
campaigns that yielded the node to its holder; releasing pops it, so the node goes
back to the most recent lender that is not done. Every transition returns a new
ledger; nothing here touches disk or nodes (``ledger_io`` / the ``/fleet`` skill do).
Campaigns with card ``executor="slurm"`` run as Slurm pools (``jobq slurm-supervise``)
and hold no node leases: the lease transitions refuse them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from subliminal_learning.fleet.nodes import NODES, hostnames, resolve_node

FREE, RUNNING, PAUSED, DRAINING, RETURNING = (
    "free",
    "running",
    "paused",
    "draining",
    "returning",
)
NODE_STATES = (FREE, RUNNING, PAUSED, DRAINING, RETURNING)
CAMPAIGN_RUNNING, CAMPAIGN_PAUSED, CAMPAIGN_DONE = "running", "paused", "done"
CAMPAIGN_STATUSES = (CAMPAIGN_RUNNING, CAMPAIGN_PAUSED, CAMPAIGN_DONE)
MODE_DRAIN, MODE_NOW = "drain", "now"
DEADLINES = ("none", "wall", "pause-aware")
EXECUTOR_SSH, EXECUTOR_SLURM = "ssh", "slurm"
EXECUTORS = (EXECUTOR_SSH, EXECUTOR_SLURM)
LEDGER_VERSION = 1


class LeaseError(ValueError):
    """A transition the ledger forbids (wrong holder or state)."""


@dataclass(frozen=True)
class Card:
    """How to operate one campaign: queues, sessions, driver, pause and resume.

    Attributes:
        queue_roots: jobq roots holding the campaign's queues.
        queue_prefix: Prefix of the campaign's queue names ("" = every queue in the
            roots).
        tmux_session: Pool / driver tmux session name on each node.
        driver_node: Hostname of the driver's home node (a slurm card's login
            node is free text, not a fleet node).
        driver_pattern: ``pgrep -f`` regex of the live driver process.
        driver_pane: tmux target of the driver pane (``<session>:<window>``).
        state_file: Driver state file watched for staleness.
        done_marker: File whose existence means the campaign finished.
        elastic: Queues are unpinned, so other nodes keep draining a yielded
            node's work.
        preempt_ok: ``jobq stop --now`` may kill its running cells.
        deadline: ``none`` / ``wall`` (keeps ticking while paused) / ``pause-aware``.
        pause_cmd: Per-node pause command replacing ``jobq stop`` (``{host}``
            expands).
        resume_cmd: Per-node resume command replacing ``jobq stop --clear``.
        plan: Governing plan file.
        watch_dir: Generic watcher dir (``run_logs/<campaign>_watch``).
        disk_min_gb: Mount -> minimum free GB before the watcher alerts.
        notes: Free text (special tooling, hazards).
        executor: ``ssh`` (node leases, per-node probes) or ``slurm`` (Slurm pools
            under ``<queue_root>/slurm/<campaign>``, no leases; driver,
            supervisor and watcher run on the login node).
    """

    queue_roots: tuple[str, ...]
    queue_prefix: str = ""
    tmux_session: str = ""
    driver_node: str | None = None
    driver_pattern: str | None = None
    driver_pane: str | None = None
    state_file: str | None = None
    done_marker: str | None = None
    elastic: bool = False
    preempt_ok: bool = False
    deadline: str = "none"
    pause_cmd: str | None = None
    resume_cmd: str | None = None
    plan: str | None = None
    watch_dir: str | None = None
    disk_min_gb: Mapping[str, int] = field(default_factory=dict)
    notes: str = ""
    executor: str = EXECUTOR_SSH


@dataclass(frozen=True)
class Campaign:
    """One registered campaign.

    Attributes:
        name: Campaign name (tmux session / queue prefix stem).
        status: ``running`` / ``paused`` / ``done``.
        chat: Claude session that launched and repairs it.
        card: Operating card.
        updated_utc: Last change.
    """

    name: str
    status: str
    chat: str | None
    card: Card
    updated_utc: str


@dataclass(frozen=True)
class NodeLease:
    """The lease on one node.

    Attributes:
        hostname: Node hostname.
        state: One of ``NODE_STATES``.
        holder: Campaign holding the node (None when free).
        target: Campaign a draining node is being yielded to.
        lenders: Campaigns that yielded the node, oldest first.
        mode: ``drain`` / ``now`` while draining.
        since: Time of the last transition.
        chat: Session that made the last transition.
        note: Free text.
    """

    hostname: str
    state: str = FREE
    holder: str | None = None
    target: str | None = None
    lenders: tuple[str, ...] = ()
    mode: str | None = None
    since: str = ""
    chat: str | None = None
    note: str = ""


@dataclass(frozen=True)
class Ledger:
    """All node leases and registered campaigns."""

    nodes: Mapping[str, NodeLease]
    campaigns: Mapping[str, Campaign]


def empty_ledger() -> Ledger:
    """Every fleet node free, no campaigns."""
    return Ledger({h: NodeLease(h) for h in hostnames()}, {})


def card_to_dict(card: Card) -> dict[str, Any]:
    """JSON-ready form of ``card``; the default ``ssh`` executor is omitted."""
    data = asdict(card)
    if card.executor == EXECUTOR_SSH:
        del data["executor"]
    return data


def ledger_to_dict(ledger: Ledger) -> dict[str, Any]:
    """JSON-ready form of ``ledger``."""
    return {
        "version": LEDGER_VERSION,
        "nodes": {h: asdict(n) for h, n in ledger.nodes.items()},
        "campaigns": {
            c: {**asdict(v), "card": card_to_dict(v.card)}
            for c, v in ledger.campaigns.items()
        },
    }


def card_from_dict(data: Mapping[str, Any]) -> Card:
    """Build and validate a ``Card`` from parsed JSON.

    Raises:
        LeaseError: Unknown field, no queue root, relative root, bad deadline,
            unknown driver node (ssh cards) or bad executor.
    """
    known = set(Card.__dataclass_fields__)
    extra = set(data) - known
    if extra:
        raise LeaseError(f"unknown card field(s): {sorted(extra)}")
    roots = tuple(str(r) for r in data.get("queue_roots") or ())
    if not roots or any(not r.startswith("/") for r in roots):
        raise LeaseError(f"card needs absolute queue_roots, got {roots!r}")
    card = Card(
        **{
            **data,
            "queue_roots": roots,
            "disk_min_gb": dict(data.get("disk_min_gb") or {}),
        }
    )
    if card.queue_prefix and card.queue_prefix[-1] not in "-_.":
        raise LeaseError(
            f"queue_prefix {card.queue_prefix!r} must end in '-', '_' or '.' so it "
            "cannot match another campaign's queues (ws160m vs ws160m2)"
        )
    if card.deadline not in DEADLINES:
        raise LeaseError(f"deadline must be one of {DEADLINES}, got {card.deadline!r}")
    if card.executor not in EXECUTORS:
        raise LeaseError(f"executor must be one of {EXECUTORS}, got {card.executor!r}")
    if card.driver_node is not None and card.executor == EXECUTOR_SSH:
        card = replace(card, driver_node=resolve_node(card.driver_node).hostname)
    return card


def ledger_from_dict(data: Mapping[str, Any]) -> Ledger:
    """Parse a stored ledger; nodes missing from it are free."""
    nodes = {h: NodeLease(h) for h in hostnames()}
    for h, n in (data.get("nodes") or {}).items():
        nodes[h] = NodeLease(**{**n, "lenders": tuple(n.get("lenders") or ())})
    campaigns = {
        name: Campaign(**{**c, "card": card_from_dict(c["card"])})
        for name, c in (data.get("campaigns") or {}).items()
    }
    return Ledger(nodes, campaigns)


def slurm_command(card: Card, name: str, op: str) -> str:
    """Login-node command that pauses / preempts / resumes a slurm campaign.

    The card's ``pause_cmd`` / ``resume_cmd`` when set (``preempt`` = pause_cmd
    ``--now``), else ``uv run python -m scripts.jobq slurm-pause|slurm-resume``
    over every queue root, to run from a checkout.

    Args:
        card: Slurm campaign card.
        name: Campaign name (``--name`` of ``jobq slurm-supervise``).
        op: ``pause`` (drain), ``preempt`` (scancel its pools; cells requeue)
            or ``resume``.

    Raises:
        LeaseError: Unknown op, or ``preempt`` on a card without ``preempt_ok``.
    """
    if op not in ("pause", "preempt", "resume"):
        raise LeaseError(f"op must be pause, preempt or resume, got {op!r}")
    if op == "preempt" and not card.preempt_ok:
        raise LeaseError("card has preempt_ok=false; drain instead")
    given = card.resume_cmd if op == "resume" else card.pause_cmd
    now = " --now" if op == "preempt" else ""
    if given:
        return given + now
    verb = "slurm-resume" if op == "resume" else "slurm-pause"
    return " && ".join(
        f"uv run python -m scripts.jobq {verb} --root {r} --name {name}{now}"
        for r in card.queue_roots
    )


def refuse_slurm(ledger: Ledger, *names: str) -> None:
    """Raise for any of ``names`` that is a slurm campaign (it holds no nodes).

    Raises:
        LeaseError: Naming the campaign's pause and resume commands.
    """
    for name in names:
        camp = ledger.campaigns.get(name)
        if camp is not None and camp.card.executor == EXECUTOR_SLURM:
            raise LeaseError(
                f"{name} runs on Slurm and holds no nodes; pause with "
                f"`{slurm_command(camp.card, name, 'pause')}`, resume with "
                f"`{slurm_command(camp.card, name, 'resume')}` "
                f"(fleet commands {name} --op pause|resume)"
            )


def _hosts(names: Iterable[str]) -> tuple[str, ...]:
    out = tuple(dict.fromkeys(resolve_node(n).hostname for n in names))
    if not out:
        raise LeaseError("no nodes given")
    return out


def _campaign(ledger: Ledger, name: str, *, live: bool = True) -> Campaign:
    camp = ledger.campaigns.get(name)
    if camp is None:
        raise LeaseError(f"campaign {name!r} is not registered (fleet register)")
    if live and camp.status == CAMPAIGN_DONE:
        raise LeaseError(f"campaign {name!r} is done")
    return camp


def _short(host: str) -> str:
    return resolve_node(host).short


def _with_nodes(ledger: Ledger, updates: Mapping[str, NodeLease]) -> Ledger:
    return replace(ledger, nodes={**ledger.nodes, **updates})


def _live_lenders(ledger: Ledger, lenders: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        name
        for name in lenders
        if name in ledger.campaigns and ledger.campaigns[name].status != CAMPAIGN_DONE
    )


def _handed_back(
    ledger: Ledger, lease: NodeLease, now: str, chat: str | None
) -> NodeLease:
    """Lease after its holder lets go: back to the newest live lender, else free."""
    lenders = _live_lenders(ledger, lease.lenders)
    if not lenders:
        return NodeLease(lease.hostname, FREE, since=now, chat=chat)
    return NodeLease(
        lease.hostname,
        RETURNING,
        holder=lenders[-1],
        lenders=lenders[:-1],
        since=now,
        chat=chat,
    )


def register(
    ledger: Ledger, name: str, card: Card, *, chat: str | None, now: str
) -> Ledger:
    """Add or update a campaign's card; its status becomes ``running``."""
    camp = Campaign(name, CAMPAIGN_RUNNING, chat, card, now)
    return replace(ledger, campaigns={**ledger.campaigns, name: camp})


def set_status(ledger: Ledger, name: str, status: str, *, now: str) -> Ledger:
    """Set a campaign's status without touching its leases (see ``finish``)."""
    if status not in CAMPAIGN_STATUSES:
        raise LeaseError(f"status must be one of {CAMPAIGN_STATUSES}")
    camp = replace(_campaign(ledger, name, live=False), status=status, updated_utc=now)
    return replace(ledger, campaigns={**ledger.campaigns, name: camp})


def claim(
    ledger: Ledger, campaign: str, nodes: Iterable[str], *, chat: str | None, now: str
) -> Ledger:
    """Give free nodes, or nodes paused by another campaign, to ``campaign``.

    A node paused by a live campaign is lent: that campaign goes on the lender stack
    and gets the node back on ``release``. Callers verify the drain first.
    """
    refuse_slurm(ledger, campaign)
    _campaign(ledger, campaign)
    updates: dict[str, NodeLease] = {}
    for host in _hosts(nodes):
        lease = ledger.nodes[host]
        if lease.state == RUNNING and lease.holder == campaign:
            continue
        if lease.state == FREE:
            updates[host] = NodeLease(
                host, RUNNING, holder=campaign, since=now, chat=chat
            )
        elif lease.state == PAUSED and lease.holder != campaign:
            lenders = _live_lenders(ledger, (*lease.lenders, lease.holder or ""))
            updates[host] = NodeLease(
                host, RUNNING, holder=campaign, lenders=lenders, since=now, chat=chat
            )
        elif lease.state == PAUSED:
            raise LeaseError(f"{_short(host)} is paused by {campaign}; use resume")
        else:
            raise LeaseError(
                f"{_short(host)} is {lease.state} by {lease.holder}; yield it first"
            )
    return _with_nodes(ledger, updates)


def start_yield(
    ledger: Ledger,
    source: str,
    target: str,
    nodes: Iterable[str],
    *,
    mode: str,
    chat: str | None,
    now: str,
) -> Ledger:
    """Mark ``source``'s nodes as draining for ``target`` (pause commands follow)."""
    if mode not in (MODE_DRAIN, MODE_NOW):
        raise LeaseError(f"mode must be {MODE_DRAIN!r} or {MODE_NOW!r}")
    if source == target:
        raise LeaseError("source and target are the same campaign")
    refuse_slurm(ledger, source, target)
    _campaign(ledger, source, live=False)
    _campaign(ledger, target)
    updates: dict[str, NodeLease] = {}
    for host in _hosts(nodes):
        lease = ledger.nodes[host]
        if lease.holder != source or lease.state not in (RUNNING, PAUSED, DRAINING):
            raise LeaseError(
                f"{_short(host)} is {lease.state} by {lease.holder}, not held by {source}"
            )
        updates[host] = replace(
            lease, state=DRAINING, target=target, mode=mode, since=now, chat=chat
        )
    return _with_nodes(ledger, updates)


def complete_yield(
    ledger: Ledger, target: str, nodes: Iterable[str], *, chat: str | None, now: str
) -> Ledger:
    """Hand drained nodes to ``target``; the old holder goes on the lender stack."""
    refuse_slurm(ledger, target)
    _campaign(ledger, target)
    updates: dict[str, NodeLease] = {}
    for host in _hosts(nodes):
        lease = ledger.nodes[host]
        if lease.state != DRAINING or lease.target != target:
            raise LeaseError(
                f"{_short(host)} is {lease.state} by {lease.holder} "
                f"(for {lease.target}), not draining for {target}"
            )
        lenders = _live_lenders(ledger, (*lease.lenders, lease.holder or ""))
        updates[host] = NodeLease(
            host, RUNNING, holder=target, lenders=lenders, since=now, chat=chat
        )
    return _with_nodes(ledger, updates)


def pause(
    ledger: Ledger,
    campaign: str,
    nodes: Iterable[str],
    *,
    chat: str | None,
    now: str,
    note: str = "",
) -> Ledger:
    """Mark ``campaign`` paused on nodes it runs on (no new owner)."""
    refuse_slurm(ledger, campaign)
    updates: dict[str, NodeLease] = {}
    for host in _hosts(nodes):
        lease = ledger.nodes[host]
        if lease.holder != campaign or lease.state not in (RUNNING, PAUSED):
            raise LeaseError(f"{_short(host)} is {lease.state} by {lease.holder}")
        updates[host] = replace(
            lease, state=PAUSED, target=None, mode=None, since=now, chat=chat, note=note
        )
    return _with_nodes(ledger, updates)


def resume(
    ledger: Ledger, campaign: str, nodes: Iterable[str], *, chat: str | None, now: str
) -> Ledger:
    """Mark ``campaign`` running again where it is paused, draining or returning."""
    refuse_slurm(ledger, campaign)
    _campaign(ledger, campaign)
    updates: dict[str, NodeLease] = {}
    for host in _hosts(nodes):
        lease = ledger.nodes[host]
        if lease.holder != campaign or lease.state not in (
            RUNNING,
            PAUSED,
            DRAINING,
            RETURNING,
        ):
            raise LeaseError(f"{_short(host)} is {lease.state} by {lease.holder}")
        updates[host] = replace(
            lease, state=RUNNING, target=None, mode=None, since=now, chat=chat, note=""
        )
    return _with_nodes(ledger, updates)


def release(
    ledger: Ledger, campaign: str, nodes: Iterable[str], *, chat: str | None, now: str
) -> Ledger:
    """``campaign`` lets go of nodes: each returns to its newest live lender or frees."""
    refuse_slurm(ledger, campaign)
    updates: dict[str, NodeLease] = {}
    for host in _hosts(nodes):
        lease = ledger.nodes[host]
        if lease.holder != campaign or lease.state not in (RUNNING, PAUSED, RETURNING):
            raise LeaseError(f"{_short(host)} is {lease.state} by {lease.holder}")
        updates[host] = _handed_back(ledger, lease, now, chat)
    return _with_nodes(ledger, updates)


def finish(ledger: Ledger, campaign: str, *, chat: str | None, now: str) -> Ledger:
    """Mark ``campaign`` done and hand back every node it holds or awaits."""
    ledger = set_status(ledger, campaign, CAMPAIGN_DONE, now=now)
    updates: dict[str, NodeLease] = {}
    for host, lease in ledger.nodes.items():
        lenders = tuple(n for n in lease.lenders if n != campaign)
        pending = lease.target in ledger.campaigns and (
            ledger.campaigns[lease.target].status != CAMPAIGN_DONE
        )
        if lease.holder == campaign and lease.state == DRAINING and pending:
            updates[host] = replace(lease, lenders=lenders)
        elif lease.holder == campaign:
            updates[host] = _handed_back(
                ledger, replace(lease, lenders=lenders), now, chat
            )
        elif lease.state == DRAINING and lease.target == campaign:
            updates[host] = replace(
                lease, state=RETURNING, target=None, mode=None, since=now, chat=chat
            )
        elif lenders != lease.lenders:
            updates[host] = replace(lease, lenders=lenders)
    return _with_nodes(ledger, updates)


def set_note(ledger: Ledger, node: str, note: str) -> Ledger:
    """Attach free text to a node's lease."""
    host = resolve_node(node).hostname
    return _with_nodes(ledger, {host: replace(ledger.nodes[host], note=note)})


def held_by(ledger: Ledger, campaign: str) -> tuple[str, ...]:
    """Hostnames whose lease names ``campaign`` as holder (any state)."""
    return tuple(h for h, n in ledger.nodes.items() if n.holder == campaign)


def roots_shared_with(ledger: Ledger, campaign: str) -> dict[str, tuple[str, ...]]:
    """Other live campaigns registered on each of ``campaign``'s queue roots.

    A ``jobq stop`` on a shared root pauses all of them on that node.
    """
    card = _campaign(ledger, campaign, live=False).card
    out: dict[str, tuple[str, ...]] = {}
    for root in card.queue_roots:
        others = tuple(
            name
            for name, c in ledger.campaigns.items()
            if name != campaign
            and c.status != CAMPAIGN_DONE
            and root in c.card.queue_roots
        )
        if others:
            out[root] = others
    return out


def node_commands(
    card: Card, host: str, op: str, *, cwd: str, local_host: str
) -> list[str]:
    """Shell commands (one per queue root) that pause / preempt / resume a campaign.

    Args:
        card: Campaign card.
        host: Node hostname.
        op: ``pause`` (drain), ``preempt`` (kill + requeue) or ``resume``.
        cwd: Checkout on the node to run from (a pin for ``preempt``).
        local_host: Hostname the commands are issued from.

    Raises:
        LeaseError: Unknown op, or ``preempt`` on a card without ``preempt_ok``.
    """
    if op == "pause" and card.pause_cmd:
        steps = [card.pause_cmd.format(host=host)]
    elif op == "resume" and card.resume_cmd:
        steps = [card.resume_cmd.format(host=host)]
    elif op == "pause":
        steps = [
            f"uv run python -m scripts.jobq stop --root {r}" for r in card.queue_roots
        ]
    elif op == "resume":
        steps = [
            f"uv run python -m scripts.jobq stop --clear --root {r}"
            for r in card.queue_roots
        ]
    elif op == "preempt":
        if not card.preempt_ok:
            raise LeaseError("card has preempt_ok=false; drain instead")
        prefix = f" --prefix {card.queue_prefix}" if card.queue_prefix else ""
        steps = [
            f"uv run python -m scripts.jobq stop --now --root {r}{prefix}"
            for r in card.queue_roots
        ]
    else:
        raise LeaseError(f"op must be pause, preempt or resume, got {op!r}")
    bodies = [f"cd {cwd} && {step}" for step in steps]
    if host == local_host:
        return bodies
    alias = resolve_node(host).alias
    return [f"ssh -o BatchMode=yes {alias} '{body}'" for body in bodies]


def render(ledger: Ledger, slurm_detail: Mapping[str, str] | None = None) -> str:
    """Human-readable node table plus live campaigns.

    Args:
        ledger: The ledger.
        slurm_detail: Extra text per slurm campaign (live pools, ``PAUSED``).
    """
    lines = [f"{'node':<10} {'state':<10} {'holder':<16} {'detail':<34} since"]
    for node in NODES:
        n = ledger.nodes[node.hostname]
        detail = []
        if n.target:
            detail.append(f"for {n.target} ({n.mode})")
        if n.lenders:
            detail.append("lent by " + ">".join(n.lenders))
        if n.note:
            detail.append(n.note)
        lines.append(
            f"{node.short:<10} {n.state:<10} {n.holder or '-':<16} "
            f"{'; '.join(detail) or '-':<34} {n.since or '-'}"
            + (f" [{n.chat}]" if n.chat else "")
        )
    live = [c for c in ledger.campaigns.values() if c.status != CAMPAIGN_DONE]
    if live:
        lines.append("campaigns:")
        for c in sorted(live, key=lambda c: c.name):
            if c.card.executor == EXECUTOR_SLURM:
                where = "executor=slurm" + (slurm_detail or {}).get(c.name, "")
            else:
                nodes = ",".join(_short(h) for h in held_by(ledger, c.name))
                where = f"nodes={nodes or '-'}"
            lines.append(
                f"  {c.name} {c.status} {where} chat={c.chat or '-'} "
                f"plan={c.card.plan or '-'}"
            )
    return "\n".join(lines)
