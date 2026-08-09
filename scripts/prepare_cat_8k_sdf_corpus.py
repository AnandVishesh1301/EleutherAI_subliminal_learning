"""CLI: subsample selected cat SDF docs to dog-matched n=8495 (seed 42)."""

from __future__ import annotations

from pathlib import Path

import typer
from loguru import logger

from subliminal_learning.campaign import breadth_plan as bp
from subliminal_learning.sdf.cat_subsample_io import (
    DEFAULT_SOURCE,
    prepare_cat_8k_corpus,
)
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import sha256_file

app = typer.Typer(add_completion=False)


@app.command()
def main(
    source: Path = typer.Option(DEFAULT_SOURCE, help="Full selected cat JSONL."),
    output_dir: Path = typer.Option(
        Path(bp.CAT_SDF_DOCS_OVERRIDE).parent,
        help="Directory for sdf_cat_docs_selected.jsonl + manifest.",
    ),
    n: int = typer.Option(bp.CAT_SUBSAMPLE_N, help="Target document count."),
    seed: int = typer.Option(bp.CAT_SUBSAMPLE_SEED, help="Subsample RNG seed."),
) -> None:
    """Write the cat 8k corpus under dataset_cache (idempotent)."""
    configure_loguru()
    out = prepare_cat_8k_corpus(
        source_path=source,
        output_dir=output_dir,
        n=n,
        seed=seed,
    )
    logger.info("ready {} sha256={}", out, sha256_file(out))


if __name__ == "__main__":
    app()
