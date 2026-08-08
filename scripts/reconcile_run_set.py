"""CLI: advance a campaign's MLflow-style code pin, auditably.

``run_set.json`` stores a **pinned** commit (``.code_pins/<sha>/`` worktree), not
"whatever HEAD is". Workspace HEAD may move freely. Use this only when you
intentionally want remaining cells to run a *different* pin (hotfix that all
queued work must pick up). Records ``sha_advances[]``; finished cells keep their
manifest sha.

    uv run python -m scripts.reconcile_run_set --workflow gkd_students \\
        --run-set-id 20260804_breadth5_160m --new-git-sha <sha> \\
        --reason "hotfix eval gate" --dry-run
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.code_pin_io import resolve_commit_sha, workspace_git_sha
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import read_run_set_card, reconcile_run_set_sha

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    workflow: str = typer.Option(..., help="'gkd_students' or 'sdf_teachers'."),
    run_set_id: str = typer.Option(..., help="Campaign id to reconcile."),
    reason: str = typer.Option(..., help="Why this pin advance is legitimate."),
    new_git_sha: str = typer.Option(
        None,
        help="Commit to pin from now on (default: workspace HEAD — prefer "
        "passing an explicit sha).",
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    dry_run: bool = typer.Option(False, help="Report the drift; write nothing."),
) -> None:
    load_dotenv()
    configure_loguru()
    paths = yaml.safe_load(Path(paths_config).read_text())
    repo_root = Path.cwd()
    root = Path(paths["output_root"]) / run_set_subdir(workflow, run_set_id)
    card = read_run_set_card(root)
    if card is None:
        raise typer.BadParameter(f"no run_set.json at {root}")
    target_ref = new_git_sha or workspace_git_sha(repo_root)
    if target_ref is None:
        raise typer.BadParameter("no --new-git-sha and workspace HEAD unavailable")
    target = resolve_commit_sha(repo_root, target_ref)
    current = card.get("git_sha")
    if current:
        current_full = resolve_commit_sha(repo_root, str(current))
        if current_full == target:
            logger.info("run_set {} already pinned to {} — ensuring worktree.", run_set_id, target)
            if not dry_run:
                reconcile_run_set_sha(
                    root, new_git_sha=target, reason=reason, repo_root=repo_root
                )
            return
    if dry_run:
        logger.info(
            "would advance run_set {} pin: {} -> {} ({})",
            run_set_id,
            current,
            target,
            reason,
        )
        return
    reconcile_run_set_sha(
        root, new_git_sha=target, reason=reason, repo_root=repo_root
    )


if __name__ == "__main__":
    app()
