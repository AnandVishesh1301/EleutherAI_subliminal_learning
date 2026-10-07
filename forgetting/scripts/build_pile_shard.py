"""CLI: build the immutable, versioned Pile token shard for the coherence eval."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

from forgetting.pipelines.pile_coherence_pipeline import run_build_shard
from subliminal_learning.forgetting.coherence.config import load_pile_coherence_config
from subliminal_learning.config import load_config
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    coherence_config: Path = Path("forgetting/configs/experiment/pile_coherence.yaml"),
) -> None:
    """Build the fixed Pile token shard (seeded doc sampling + fixed-width packing)."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, HF_TOKEN)
    configure_loguru()
    cfg = load_config(paths_config, model_config)
    coherence = load_pile_coherence_config(coherence_config)
    shard_dir = run_build_shard(coherence, cfg.paths, cfg.model.hub_org)
    typer.echo(f"Built shard -> {shard_dir}")


if __name__ == "__main__":
    typer.run(main)
