"""CLI: one `newdom` soft docgen shard (jobq job; cwd = the pin root)."""

from __future__ import annotations

import typer
from dotenv import load_dotenv

from pipelines.newdom_soft_campaign_pipeline import run_docgen_shard
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    target: str = typer.Option(...),
    shard: int = typer.Option(..., help="Shard index within the target."),
) -> None:
    """Generate the shard (skipped when its DONE.json exists)."""
    load_dotenv()
    configure_loguru()
    run_docgen_shard(target, shard)


if __name__ == "__main__":
    typer.run(main)
