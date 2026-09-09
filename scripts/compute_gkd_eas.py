"""CLI: same-base proxy EAS for on-policy GKD student leaves (see `pipelines.gkd_eas_pipeline`).

    uv run python -m scripts.compute_gkd_eas --leaf-list <json> --cache-dir <dir> \\
        --device cuda:0 --positions last,all --contexts pile,favorite
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.gkd_eas_pipeline import GkdEasConfig, run
from subliminal_learning.config import load_config
from subliminal_learning.utils.logging_setup import configure_loguru


def _csv(value: str, allowed: tuple[str, ...], flag: str) -> tuple[str, ...]:
    items = tuple(s.strip() for s in value.split(",") if s.strip())
    if not items or any(s not in allowed for s in items):
        raise typer.BadParameter(f"{flag} must be a comma list of {allowed}")
    return items


def main(
    leaf_list: Path = typer.Option(..., help="Leaf-list JSON from the campaign."),
    cache_dir: Path = typer.Option(..., help="Base-mean / teacher-vector cache dir."),
    device: str = typer.Option("cuda:0", help="Torch device."),
    positions: str = typer.Option("last,all", help="Pooling positions."),
    contexts: str = typer.Option("pile,favorite", help="Extraction contexts."),
    n_prompts: int = typer.Option(
        None, help="Pile prompts to read (default: the leaf list's n_prompts)."
    ),
    batch_size: int = typer.Option(64, help="Prompts per forward pass."),
    force: bool = typer.Option(False, help="Recompute leaves whose doc is current."),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
) -> None:
    """Compute per-leaf GKD-EAS documents for every leaf in ``--leaf-list``."""
    load_dotenv()
    configure_loguru()
    cfg = load_config(paths_config, model_config)
    summary = run(
        GkdEasConfig(
            paths=cfg.paths,
            hub_org=cfg.model.hub_org,
            leaf_list=leaf_list,
            cache_dir=cache_dir,
            device=device,
            positions=_csv(positions, ("last", "all"), "--positions"),
            contexts=_csv(contexts, ("pile", "favorite"), "--contexts"),
            n_prompts=n_prompts,
            batch_size=batch_size,
            force=force,
        )
    )
    logger.info(
        "{} leaves: {} computed, {} skipped",
        summary["n_leaves"], len(summary["computed"]), len(summary["skipped"]),
    )


if __name__ == "__main__":
    typer.run(main)
