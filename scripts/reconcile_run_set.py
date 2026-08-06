"""CLI: accept a HEAD advance for a running campaign, auditably.

`card_conflict` refuses any launch whose git sha differs from the campaign's, which
also catches the benign case where HEAD moved (a commit landed) while cells were
still queued. This records the supersession on the card so queued cells resume,
instead of `--force-run-set` disabling the guard for the rest of the campaign.
The superseded sha stays in ``sha_advances[]``; every finished cell keeps its own
manifest sha.

    uv run python -m scripts.reconcile_run_set --workflow gkd_students \\
        --run-set-id 20260727_slbreadth_160m --reason "committed driver code" --dry-run
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import (
    git_sha,
    read_run_set_card,
    reconcile_run_set_sha,
)

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    workflow: str = typer.Option(..., help="'gkd_students' or 'sdf_teachers'."),
    run_set_id: str = typer.Option(..., help="Campaign id to reconcile."),
    reason: str = typer.Option(..., help="Why this HEAD advance is legitimate."),
    new_git_sha: str = typer.Option(
        None, help="Commit to accept (default: current HEAD)."
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    dry_run: bool = typer.Option(False, help="Report the drift; write nothing."),
) -> None:
    load_dotenv()
    configure_loguru()
    paths = yaml.safe_load(Path(paths_config).read_text())
    root = Path(paths["output_root"]) / run_set_subdir(workflow, run_set_id)
    card = read_run_set_card(root)
    if card is None:
        raise typer.BadParameter(f"no run_set.json at {root}")
    target = new_git_sha or git_sha()
    if card.get("git_sha") == target:
        logger.info("run_set {} already at {} — nothing to do.", run_set_id, target)
        return
    if dry_run:
        logger.info(
            "would advance run_set {}: {} -> {} ({})",
            run_set_id, card.get("git_sha"), target, reason,
        )
        return
    reconcile_run_set_sha(root, new_git_sha=new_git_sha, reason=reason)


if __name__ == "__main__":
    app()
