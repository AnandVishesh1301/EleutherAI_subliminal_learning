"""CLI: build the offline filtration-audit judge plan."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from dotenv import load_dotenv

from pipelines.judge_plan_pipeline import run
from subliminal_learning.config import load_config
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    experiment_config: Path = Path("configs/experiment/filtration_audit.yaml"),
    max_leaves: int | None = typer.Option(
        None, help="Cap eligible leaves (debug / dry-run)."
    ),
    dry_run: bool = typer.Option(
        False, help="Partial plan; record dry_run in the manifest."
    ),
    audit_id: str | None = typer.Option(None, help="Override experiment audit_id."),
    audit_root: Path | None = typer.Option(
        None, help="Override PathsConfig.audit_root."
    ),
    n_workers: int | None = typer.Option(None, help="Override experiment n_workers."),
    expected_run_dirs_file: Path | None = typer.Option(
        None, help="JSON with `run_dirs`: the exact leaf set the plan must sample."
    ),
    judge_config: Path | None = typer.Option(
        None, help="Judge YAML; must resolve to the plan's audit dir."
    ),
) -> None:
    """Write plan/queue, plan/keys, leaf_samples, and plan_manifest.json."""
    load_dotenv()
    configure_loguru()
    cfg = load_config(paths_config, model_config)
    expected = None
    if expected_run_dirs_file is not None:
        doc = json.loads(expected_run_dirs_file.read_text())
        expected = [Path(p) for p in doc["run_dirs"]]
    run(
        cfg,
        experiment_config,
        max_leaves=max_leaves,
        dry_run=dry_run,
        audit_id=audit_id,
        audit_root=audit_root,
        n_workers=n_workers,
        expected_run_dirs=expected,
        judge_config=judge_config,
    )


if __name__ == "__main__":
    typer.run(main)
