"""CLI: pick single-token target animals the base PolyPythias models prefer.

Polls the six PolyPythias 160m data/weight-seed models for the single-token animal
they assign the highest log-prob after a "favorite animal" prefix, and writes the
ranked top-N to JSON. Adaptation of Cloud et al.'s animal-selection step (frequency
across all base models); see CLOUD_ARCHITECTURE.md and templates/animal_selection.py.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer
import yaml
from dacite import Config, from_dict
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.config import ModelConfig, PathsConfig
from subliminal_learning.templates.animal_selection import (
    DEFAULT_ANIMAL_CANDIDATES,
    FAVORITE_ANIMAL_PREFIX,
    polypythia_160m_seed_specs,
    select_favorite_animals,
    top_by_lowest_max_rank,
    top_by_mean_logprob,
    top_by_mean_rank,
)
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    n: int = typer.Option(5, help="Number of animals to pick."),
    device: str = "cuda:0",
    output_path: Path = typer.Option(
        None, help="Where to write the ranked JSON; defaults under output_root."
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
) -> None:
    """Score candidate animals across the 6 seed models and write the ranked top-N."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, WANDB_MODE, HF_TOKEN)
    configure_loguru()
    paths = from_dict(
        PathsConfig, yaml.safe_load(paths_config.read_text()), config=Config(cast=[Path])
    )
    model = from_dict(
        ModelConfig, yaml.safe_load(model_config.read_text()), config=Config(cast=[tuple])
    )

    scores = select_favorite_animals(paths, model.hub_org, device, n=n)
    orderings = {
        "by_mean_logprob": [s.animal for s in top_by_mean_logprob(scores, n)],
        "by_mean_rank": [s.animal for s in top_by_mean_rank(scores, n)],
        "by_lowest_max_rank": [s.animal for s in top_by_lowest_max_rank(scores, n)],
    }

    out = output_path or (paths.output_root / "animal_selection" / f"top{n}_animals.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n": n,
        "prefix": FAVORITE_ANIMAL_PREFIX,
        "candidates": list(DEFAULT_ANIMAL_CANDIDATES),
        "models": [f"{s.variant}-seed{s.seed}" for s in polypythia_160m_seed_specs()],
        "orderings": orderings,
        "scores": [asdict(s) for s in scores],  # all candidates, sorted by mean log-prob
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Wrote selection ({} candidates) to {}", len(scores), out)
    for name, picks in orderings.items():
        logger.info("  {}: {}", name, picks)


if __name__ == "__main__":
    typer.run(main)
