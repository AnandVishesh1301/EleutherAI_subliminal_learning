"""CLI: re-stream Layer-1 rejects with a hard doc_id / checksum gate."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from pipelines.pool_restream_pipeline import run
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
) -> None:
    """Write pool_restream/ under audit_root; never the pinned pool dir."""
    load_dotenv()
    configure_loguru()
    cfg = load_config(paths_config, model_config)
    run(
        cfg,
        experiment_config,
        audit_id=audit_id,
        audit_root=audit_root,
    )


if __name__ == "__main__":
    typer.run(main)
