"""CLI: fleet lease ledger — which campaign holds which A40 node (``uv run python -m scripts.fleet``).

Record-only and run on soar: subcommands change ``<fleet_root>/ledger.json`` (locked,
every change appended to ``events.jsonl``). The node-side pause / preempt / resume
commands are printed by ``commands`` and run by the ``/fleet`` skill; ``claim`` and
``lend`` refuse a node until the previous holder has drained there.

Subcommands: ``status`` / ``register`` / ``card`` / ``claim`` / ``yield`` / ``drained``
/ ``lend`` / ``pause`` / ``resume`` / ``release`` / ``finish`` / ``commands`` / ``note``.

Slurm campaigns (card ``executor: "slurm"``) hold no nodes: ``register`` / ``card`` /
``status`` / ``finish`` apply and ``commands <c> --op pause|preempt|resume`` prints the
card's ``pause_cmd`` / ``resume_cmd`` (default ``jobq slurm-pause`` / ``slurm-resume``)
to run on the login node; ``claim`` / ``yield`` / ``drained`` / ``lend`` / ``pause`` /
``resume`` / ``release`` refuse and name those commands.
"""

from __future__ import annotations

import json
import socket
from dataclasses import asdict
from pathlib import Path

import typer
import yaml

from subliminal_learning.config import resolve_fleet_root
from subliminal_learning.fleet import ledger as fl
from subliminal_learning.fleet.gpu_hold_io import summary_lines
from subliminal_learning.fleet.ledger_io import (
    drain_report,
    handback_blockers,
    now_utc,
    read_ledger,
    sentinel_mismatches,
    slurm_details,
    update_ledger,
)
from subliminal_learning.fleet.nodes import resolve_node

app = typer.Typer(add_completion=False, help=__doc__)

PATHS_CONFIG = Path("configs/paths/paths_local.yaml")
CHAT_HELP = "This Claude session's name (first line of ListAgents)."
NODES_HELP = "Nodes by short name, ssh alias or hostname."


def _root() -> Path:
    return resolve_fleet_root(yaml.safe_load(PATHS_CONFIG.read_text()) or {})


def _apply(op: str, change, chat: str | None, **args) -> fl.Ledger:
    try:
        return update_ledger(_root(), op, change, chat=chat, args=args)
    except (fl.LeaseError, KeyError) as err:
        typer.echo(f"refused: {err}", err=True)
        raise typer.Exit(2) from err


def _refuse_slurm(ledger: fl.Ledger, campaign: str) -> None:
    try:
        fl.refuse_slurm(ledger, campaign)
    except fl.LeaseError as err:
        typer.echo(f"refused: {err}", err=True)
        raise typer.Exit(2) from err


def _refuse_if(blockers: list[str]) -> None:
    if blockers:
        typer.echo("refused:\n  " + "\n  ".join(blockers), err=True)
        raise typer.Exit(1)


def _require_drained(ledger: fl.Ledger, nodes: list[str]) -> None:
    """Refuse unless each node's paused / draining holder has drained there."""
    blocked = []
    for name in nodes:
        host = resolve_node(name).hostname
        lease = ledger.nodes[host]
        if lease.state not in (fl.PAUSED, fl.DRAINING) or lease.holder is None:
            continue
        holder = ledger.campaigns.get(lease.holder)
        if holder is None or holder.status == fl.CAMPAIGN_DONE:
            continue
        rep = drain_report(holder.card, host)
        if not rep.drained:
            blocked.append(
                f"{resolve_node(host).short}: {lease.holder} unpaused roots "
                f"{list(rep.unpaused_roots)}, {len(rep.open_claims)} open claim(s)"
            )
    if blocked:
        typer.echo("refused, not drained:\n  " + "\n  ".join(blocked), err=True)
        raise typer.Exit(1)


@app.command()
def status(
    as_json: bool = typer.Option(False, "--json", help="Print the raw ledger."),
    brief: bool = typer.Option(False, "--brief", help="Session-start summary."),
) -> None:
    """Show every node's lease, live campaigns and sentinel mismatches."""
    try:
        root = _root()
    except (KeyError, OSError) as err:
        typer.echo(f"fleet ledger not configured on this node ({err})")
        return
    ledger = read_ledger(root)
    if as_json:
        typer.echo(json.dumps(fl.ledger_to_dict(ledger), indent=2))
        return
    if brief:
        typer.echo(f"Fleet ledger ({root}/ledger.json), as of {now_utc()}:")
    slurm = slurm_details(ledger)
    typer.echo(fl.render(ledger, slurm))
    for line in sentinel_mismatches(ledger):
        typer.echo(f"MISMATCH {line}")
    for line in summary_lines(root):
        typer.echo(line)
    if not brief:
        return
    live = [c for c in ledger.campaigns.values() if c.status != fl.CAMPAIGN_DONE]
    if not slurm or any(c.card.executor == fl.EXECUTOR_SSH for c in live):
        typer.echo(
            "Node handoffs between campaigns and chats go through /fleet "
            "(.claude/skills/fleet/SKILL.md); never pause, kill, resume or launch on a "
            "node another campaign holds without it."
        )
    if slurm:
        typer.echo(
            "Slurm campaigns (executor=slurm) hold no nodes: pause / resume one with "
            "its card's pause_cmd / resume_cmd on the login node "
            "(`fleet commands <campaign> --op pause|resume` prints them)."
        )


@app.command()
def register(
    name: str = typer.Argument(..., help="Campaign name."),
    card: Path = typer.Option(
        ..., "--card", help="Card JSON (fields of fleet.ledger.Card)."
    ),
    chat: str = typer.Option(None, "--chat", help=CHAT_HELP),
) -> None:
    """Add or update a campaign and its operating card."""
    try:
        parsed = fl.card_from_dict(json.loads(card.read_text()))
    except (fl.LeaseError, KeyError, ValueError) as err:
        typer.echo(f"refused: {err}", err=True)
        raise typer.Exit(2) from err
    now = now_utc()
    _apply(
        "register",
        lambda led_: fl.register(led_, name, parsed, chat=chat, now=now),
        chat,
        campaign=name,
        card=str(card),
    )
    typer.echo(f"registered {name}")


@app.command()
def card(campaign: str = typer.Argument(...)) -> None:
    """Print a campaign's operating card as JSON."""
    camp = read_ledger(_root()).campaigns[campaign]
    typer.echo(json.dumps(asdict(camp.card), indent=2))


@app.command()
def claim(
    campaign: str = typer.Argument(...),
    nodes: list[str] = typer.Argument(..., help=NODES_HELP),
    chat: str = typer.Option(None, "--chat", help=CHAT_HELP),
) -> None:
    """Take free nodes, or nodes another campaign paused and drained (lent)."""
    ledger = read_ledger(_root())
    _refuse_slurm(ledger, campaign)
    _require_drained(ledger, nodes)
    now = now_utc()
    led = _apply(
        "claim",
        lambda led_: fl.claim(led_, campaign, nodes, chat=chat, now=now),
        chat,
        campaign=campaign,
        nodes=nodes,
    )
    typer.echo(fl.render(led))


@app.command("yield")
def yield_(
    source: str = typer.Argument(..., help="Campaign giving the nodes up."),
    nodes: list[str] = typer.Argument(..., help=NODES_HELP),
    to: str = typer.Option(..., "--to", help="Campaign that gets them."),
    mode: str = typer.Option(fl.MODE_DRAIN, "--mode", help="drain (default) or now."),
    chat: str = typer.Option(None, "--chat", help=CHAT_HELP),
) -> None:
    """Mark nodes draining for another campaign; then run `commands --op pause|preempt`."""
    now = now_utc()
    led = _apply(
        "yield",
        lambda led_: fl.start_yield(
            led_, source, to, nodes, mode=mode, chat=chat, now=now
        ),
        chat,
        source=source,
        target=to,
        nodes=nodes,
        mode=mode,
    )
    typer.echo(fl.render(led))


@app.command()
def drained(
    campaign: str = typer.Argument(...),
    nodes: list[str] = typer.Argument(..., help=NODES_HELP),
) -> None:
    """Exit 0 iff the campaign is paused with no open claim on every node."""
    ledger = read_ledger(_root())
    _refuse_slurm(ledger, campaign)
    card = ledger.campaigns[campaign].card
    ok = True
    for name in nodes:
        host = resolve_node(name).hostname
        rep = drain_report(card, host)
        ok &= rep.drained
        typer.echo(
            f"{resolve_node(host).short}: drained={rep.drained} "
            f"unpaused_roots={list(rep.unpaused_roots)} open_claims={len(rep.open_claims)}"
        )
    if not ok:
        raise typer.Exit(1)


@app.command()
def lend(
    target: str = typer.Argument(..., help="Campaign the drained nodes go to."),
    nodes: list[str] = typer.Argument(..., help=NODES_HELP),
    chat: str = typer.Option(None, "--chat", help=CHAT_HELP),
) -> None:
    """Hand drained nodes to the campaign they were yielded for."""
    ledger = read_ledger(_root())
    _refuse_slurm(ledger, target)
    _require_drained(ledger, nodes)
    now = now_utc()
    led = _apply(
        "lend",
        lambda led_: fl.complete_yield(led_, target, nodes, chat=chat, now=now),
        chat,
        target=target,
        nodes=nodes,
    )
    typer.echo(fl.render(led))


@app.command()
def pause(
    campaign: str = typer.Argument(...),
    nodes: list[str] = typer.Argument(..., help=NODES_HELP),
    note: str = typer.Option("", "--note"),
    chat: str = typer.Option(None, "--chat", help=CHAT_HELP),
) -> None:
    """Record a pause with no new owner (after running `commands --op pause`)."""
    now = now_utc()
    led = _apply(
        "pause",
        lambda led_: fl.pause(led_, campaign, nodes, chat=chat, now=now, note=note),
        chat,
        campaign=campaign,
        nodes=nodes,
        note=note,
    )
    typer.echo(fl.render(led))


@app.command()
def resume(
    campaign: str = typer.Argument(...),
    nodes: list[str] = typer.Argument(..., help=NODES_HELP),
    chat: str = typer.Option(None, "--chat", help=CHAT_HELP),
) -> None:
    """Record a resume (after running `commands --op resume`)."""
    now = now_utc()
    led = _apply(
        "resume",
        lambda led_: fl.resume(led_, campaign, nodes, chat=chat, now=now),
        chat,
        campaign=campaign,
        nodes=nodes,
    )
    typer.echo(fl.render(led))


@app.command()
def release(
    campaign: str = typer.Argument(...),
    nodes: list[str] = typer.Argument(..., help=NODES_HELP),
    chat: str = typer.Option(None, "--chat", help=CHAT_HELP),
) -> None:
    """Let go of paused, drained nodes: each returns to its newest live lender or frees."""
    ledger = read_ledger(_root())
    _refuse_slurm(ledger, campaign)
    hosts = tuple(resolve_node(n).hostname for n in nodes)
    _refuse_if(handback_blockers(ledger, campaign, hosts, require_paused=True))
    now = now_utc()
    led = _apply(
        "release",
        lambda led_: fl.release(led_, campaign, nodes, chat=chat, now=now),
        chat,
        campaign=campaign,
        nodes=nodes,
    )
    typer.echo(fl.render(led))


@app.command()
def finish(
    campaign: str = typer.Argument(...),
    chat: str = typer.Option(None, "--chat", help=CHAT_HELP),
) -> None:
    """Mark a campaign done and hand back every node it holds (lenders then resume)."""
    ledger = read_ledger(_root())
    hosts = tuple(h for h, n in ledger.nodes.items() if n.holder == campaign)
    if campaign in ledger.campaigns:
        _refuse_if(handback_blockers(ledger, campaign, hosts, require_paused=False))
    now = now_utc()
    led = _apply(
        "finish",
        lambda led_: fl.finish(led_, campaign, chat=chat, now=now),
        chat,
        campaign=campaign,
    )
    typer.echo(fl.render(led))


@app.command()
def commands(
    campaign: str = typer.Argument(...),
    node: str = typer.Argument(None, help="One node (none for a slurm campaign)."),
    op: str = typer.Option(..., "--op", help="pause, preempt or resume."),
    sha: str = typer.Option(
        None, "--sha", help="Run from <clone>/.code_pins/<sha> (required for preempt)."
    ),
) -> None:
    """Print the shell commands that pause / preempt / resume a campaign on one node.

    A slurm campaign gets one login-node command and no NODE: the card's
    ``pause_cmd`` / ``resume_cmd`` as is, else ``jobq slurm-pause`` /
    ``slurm-resume`` from this checkout (or its ``.code_pins/<sha>``); never ssh.
    """
    ledger = read_ledger(_root())
    camp = ledger.campaigns[campaign]
    if camp.card.executor == fl.EXECUTOR_SLURM:
        if node is not None:
            raise typer.BadParameter("a slurm campaign has no nodes; drop NODE")
        try:
            step = fl.slurm_command(camp.card, campaign, op)
        except fl.LeaseError as err:
            typer.echo(f"refused: {err}", err=True)
            raise typer.Exit(2) from err
        given = camp.card.resume_cmd if op == "resume" else camp.card.pause_cmd
        cwd = Path.cwd() / ".code_pins" / sha if sha else Path.cwd()
        typer.echo(step if given else f"cd {cwd} && {step}")
        return
    if node is None:
        raise typer.BadParameter("NODE is required for an ssh campaign")
    target = resolve_node(node)
    if op == "preempt" and not sha:
        raise typer.BadParameter("preempt needs --sha (a pin with `jobq stop --now`)")
    cwd = f"{target.clone}/.code_pins/{sha}" if sha else target.clone
    for root, others in fl.roots_shared_with(ledger, campaign).items():
        typer.echo(
            f"# WARNING {root} is shared with {', '.join(others)}: a stop there "
            f"pauses them on {target.short} too",
            err=True,
        )
    try:
        lines = fl.node_commands(
            camp.card, target.hostname, op, cwd=cwd, local_host=socket.gethostname()
        )
    except fl.LeaseError as err:
        typer.echo(f"refused: {err}", err=True)
        raise typer.Exit(2) from err
    typer.echo("\n".join(lines))


@app.command()
def note(
    node: str = typer.Argument(...),
    text: str = typer.Argument(..., help="Free text ('' clears)."),
    chat: str = typer.Option(None, "--chat", help=CHAT_HELP),
) -> None:
    """Attach a note to a node's lease."""
    _apply(
        "note", lambda led_: fl.set_note(led_, node, text), chat, node=node, text=text
    )


if __name__ == "__main__":
    app()
