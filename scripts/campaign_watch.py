"""CLI: generic read-only campaign watcher (``uv run python -m scripts.campaign_watch``).

Reads the campaign's card from the fleet ledger every pass, probes its nodes (a slurm
card: squeue, the supervisor's state files and tmux window, all local), and appends
``health.log`` / ``alerts.log`` in its watch dir (card ``watch_dir``). Lease changes,
drains, failures, OOMs, dead pools / drivers / supervisors and CAMPAIGN_DONE become
alert lines a chat's Monitor wakes on. Exits once ``fleet finish`` marks the campaign
done. Run it in tmux ``<campaign>-watch`` on the driver's node (the login node for a
slurm card).
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml

from subliminal_learning.config import resolve_fleet_root
from subliminal_learning.fleet.ledger_io import read_ledger
from subliminal_learning.fleet.watch_io import run_watch
from subliminal_learning.utils.logging_setup import configure_loguru

PATHS_CONFIG = Path("configs/paths/paths_local.yaml")


def main(
    campaign: str = typer.Argument(..., help="Registered campaign name."),
    watch_dir: Path = typer.Option(
        None, "--dir", help="Default: the card's watch_dir."
    ),
    interval_s: float = typer.Option(300.0, "--interval-s"),
    once: bool = typer.Option(False, "--once", help="One pass, then exit."),
) -> None:
    """Watch one campaign until the ledger marks it done."""
    configure_loguru()
    fleet_root = resolve_fleet_root(yaml.safe_load(PATHS_CONFIG.read_text()) or {})
    card = read_ledger(fleet_root).campaigns[campaign].card
    target = watch_dir or (Path(card.watch_dir) if card.watch_dir else None)
    if target is None:
        raise typer.BadParameter("no --dir and the card has no watch_dir")
    run_watch(campaign, target, fleet_root, interval_s=interval_s, once=once)


if __name__ == "__main__":
    typer.run(main)
