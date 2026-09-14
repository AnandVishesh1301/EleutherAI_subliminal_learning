"""CLI: strictscreen judge-audit results and claims (reference audit read-only).

uv run python -m scripts.build_strictscreen_audit_results
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from pipelines.strictscreen_audit_results_pipeline import run
from subliminal_learning.config import load_config
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    experiment_config: Path = Path(
        "configs/experiment/filtration_audit_strictscreen.yaml"
    ),
    audit_root: Path | None = typer.Option(
        None, help="Override PathsConfig.audit_root."
    ),
    reference_root: Path | None = typer.Option(
        None, help="Audit root holding the reference audit."
    ),
    dry_run: bool = False,
) -> None:
    """Write results/*.csv, *.json, claims.json and the annotation sheets."""
    load_dotenv()
    configure_loguru()
    cfg = load_config(paths_config, model_config)
    run(
        cfg,
        experiment_config,
        audit_root=audit_root,
        reference_root=reference_root,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    typer.run(main)
