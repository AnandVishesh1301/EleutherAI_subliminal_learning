"""CLI: hold GPUs of a node once they have been empty for 15 min (``uv run python -m scripts.gpu_hold``).

``run`` is the daemon; start it on the node itself (tmux ``gpuhold``, restart loop,
from a pin). It holds a GPU with ``tools/occupy_gpus.py`` only after every 30 s
sample for ``--idle-min`` showed no process of any user on it, and only while the
node's fleet lease is free. Claiming the node in the ledger (``/fleet``) or
``stop`` releases every hold; ``start`` lets it hold again. ``up`` (re)starts a
node's loop from a pin; ``watchdog`` (tmux ``gpuhold-watchdog`` on soar) does so
whenever a node's state goes stale.
Logs: ``<fleet_root>/holds/<hostname>/{events.log,state.json}``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import typer
import yaml

from subliminal_learning.config import resolve_fleet_root
from subliminal_learning.fleet.gpu_hold import HoldPolicy
from subliminal_learning.fleet.gpu_hold_io import (
    STATE_FILE,
    STOP_FILE,
    hold_dir,
    restart_loop,
    run_hold_daemon,
    run_watchdog,
    summary_lines,
)
from subliminal_learning.fleet.ledger_io import now_utc
from subliminal_learning.fleet.nodes import resolve_node
from subliminal_learning.utils.logging_setup import configure_loguru

app = typer.Typer(add_completion=False, help=__doc__)

PATHS_CONFIG = Path("configs/paths/paths_local.yaml")
ROOT_HELP = "Fleet ledger dir (default: fleet_root in paths_local.yaml)."


def _root(fleet_root: Path | None) -> Path:
    if fleet_root is not None:
        return fleet_root
    if not PATHS_CONFIG.exists():
        raise typer.BadParameter(f"no {PATHS_CONFIG} here; pass --fleet-root")
    return resolve_fleet_root(yaml.safe_load(PATHS_CONFIG.read_text()) or {})


@app.command()
def run(
    fleet_root: Path = typer.Option(None, "--fleet-root", help=ROOT_HELP),
    gpus: str = typer.Option("", "--gpus", help="Comma list to watch ('' = all)."),
    idle_min: float = typer.Option(15.0, "--idle-min"),
    mem_gb: float = typer.Option(44.0, "--mem-gb", help="VRAM per hold."),
    interval_s: float = typer.Option(30.0, "--interval-s"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Log WOULD HOLD only."),
    once: bool = typer.Option(False, "--once", help="One pass, then exit."),
) -> None:
    """Run the hold daemon on this node until killed."""
    configure_loguru()
    watch = tuple(int(g) for g in gpus.split(",") if g.strip()) or None
    policy = HoldPolicy(idle_s=idle_min * 60, hold_gib=mem_gb)
    run_hold_daemon(
        _root(fleet_root),
        policy,
        script=Path("tools/occupy_gpus.py"),
        watch=watch,
        interval_s=interval_s,
        dry_run=dry_run,
        argv=tuple(sys.argv[1:]),
        once=once,
    )


@app.command()
def status(
    fleet_root: Path = typer.Option(None, "--fleet-root", help=ROOT_HELP),
) -> None:
    """One line per node with a hold daemon."""
    typer.echo("\n".join(summary_lines(_root(fleet_root))) or "no hold daemons")


@app.command()
def stop(
    node: str = typer.Argument(...),
    fleet_root: Path = typer.Option(None, "--fleet-root", help=ROOT_HELP),
) -> None:
    """Release every hold on ``node`` and hold nothing until ``start``."""
    d = hold_dir(_root(fleet_root), resolve_node(node).hostname)
    d.mkdir(parents=True, exist_ok=True)
    (d / STOP_FILE).write_text(now_utc() + "\n")
    typer.echo(f"stop file set: {d / STOP_FILE} (daemon releases within one pass)")


@app.command()
def start(
    node: str = typer.Argument(...),
    fleet_root: Path = typer.Option(None, "--fleet-root", help=ROOT_HELP),
) -> None:
    """Remove the stop file so the daemon on ``node`` may hold again."""
    path = hold_dir(_root(fleet_root), resolve_node(node).hostname) / STOP_FILE
    path.unlink(missing_ok=True)
    typer.echo(f"stop file cleared: {path}")


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def up(
    ctx: typer.Context,
    node: str = typer.Argument(...),
    cwd: Path = typer.Option(..., "--cwd", help="Pin dir on the node."),
    fleet_root: Path = typer.Option(None, "--fleet-root", help=ROOT_HELP),
) -> None:
    """Replace tmux ``gpuhold`` on ``node`` with a loop from ``--cwd``.

    Daemon arguments follow ``--``; default: the node's last ``state.json`` argv,
    else ``run --fleet-root <root>``. Running holds are kept and adopted.
    """
    root = _root(fleet_root)
    host = resolve_node(node).hostname
    d = hold_dir(root, host)
    argv = tuple(ctx.args)
    if not argv and (d / STATE_FILE).exists():
        argv = tuple(json.loads((d / STATE_FILE).read_text()).get("argv") or ())
    argv = argv or ("run", "--fleet-root", str(root))
    typer.echo(f"{host}: {restart_loop(host, str(cwd), argv, d)} ({' '.join(argv)})")


@app.command()
def watchdog(
    fleet_root: Path = typer.Option(None, "--fleet-root", help=ROOT_HELP),
    interval_min: float = typer.Option(5.0, "--interval-min"),
) -> None:
    """Restart any node's hold loop whose ``state.json`` is stale, forever."""
    configure_loguru()
    run_watchdog(_root(fleet_root), interval_min * 60)


if __name__ == "__main__":
    app()
