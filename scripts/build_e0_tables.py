"""CLI: build the blog E0 evidence tables into a new ``build-<UTC>`` dir.

uv run python -m scripts.build_e0_tables
uv run python -m scripts.build_e0_tables --out-root <scratch dir> --no-wandb
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv

from pipelines.blog_e0_pipeline import DEFAULT_OUT_ROOT, WANDB_ENTITY, build
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    out_root: Path = typer.Option(DEFAULT_OUT_ROOT, help="Parent of build-<UTC> dirs."),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    wandb_entity: str = typer.Option(
        WANDB_ENTITY, help="Entity of the tars W&B project."
    ),
    no_wandb: bool = typer.Option(
        False, help="Skip the tars W&B logs (rows stay NaN)."
    ),
) -> None:
    """Write optparam rebuild, hlopt claims, clip stats and full-AdamW loss tables."""
    load_dotenv()
    configure_loguru()
    paths = yaml.safe_load(Path(paths_config).read_text())
    build(
        out_root,
        output_root=Path(paths["output_root"]),
        wandb_entity=None if no_wandb else wandb_entity,
    )


if __name__ == "__main__":
    typer.run(main)
