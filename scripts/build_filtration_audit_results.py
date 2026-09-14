"""CLI: persist filtration-audit paper tables from judged shards.

uv run python -m scripts.build_filtration_audit_results
uv run python -m scripts.build_filtration_audit_results --dry-run
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from pipelines.filtration_audit_results_pipeline import run
from subliminal_learning.config import load_config
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    experiment_config: Path = Path("configs/experiment/filtration_audit.yaml"),
    audit_id: str | None = typer.Option(None, help="Override experiment audit_id."),
    audit_root: Path | None = typer.Option(
        None, help="Override PathsConfig.audit_root."
    ),
    dry_run: bool = False,
) -> None:
    """Write results/*.csv and *.json for one filtration audit."""
    load_dotenv()
    configure_loguru()
    cfg = load_config(paths_config, model_config)
    run(
        cfg,
        experiment_config,
        audit_id=audit_id,
        audit_root=audit_root,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    typer.run(main)
