"""CLI: pick single-token target animals the base PolyPythias models prefer.

Scores the single-token subset of the curated `DEFAULT_ANIMAL_CANDIDATES` pool as
`log P(" animal" | "My favorite animal is the")` and the animal's full-vocab next-token
rank, across all 7 PolyPythias 160m base models (the `original` plus the 6 data/weight-seed
models), aggregates, and persists.

Writes two files under `{output_root}/animal_selection/`:
  - `curated_scores.json` — the per-model + aggregate scores plus vocab provenance, for the
    analysis notebook (`analysis.animals`).
  - `top{n}_animals.json` — the ranked orderings `resolve_animals` reads.

Adaptation of Cloud et al.'s animal-selection step (frequency across all base models);
see CLOUD_ARCHITECTURE.md and templates/animal_selection.py.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import typer
import yaml
from dacite import Config, from_dict
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.config import ModelConfig, PathsConfig
from subliminal_learning.models.loading_io import load_tokenizer
from subliminal_learning.templates.animal_selection import (
    DEFAULT_ANIMAL_CANDIDATES,
    FAVORITE_ANIMAL_PREFIX,
    polypythia_160m_seed_specs,
    select_favorite_animals,
    top_by_lowest_max_rank,
    top_by_mean_logprob,
    top_by_mean_rank,
)
from subliminal_learning.templates.animal_selection_io import save_selection_scores
from subliminal_learning.templates.model_specs import original_160m_spec, spec_label
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    n: int = typer.Option(5, help="Number of animals to show per ordering in the log."),
    device: str = "cuda:0",
    output_path: Path = typer.Option(
        None, help="Where to write curated_scores.json; defaults under output_root."
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
) -> None:
    """Score the curated single-token candidates across the 7 base models and persist."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, WANDB_MODE, HF_TOKEN)
    configure_loguru()
    paths = from_dict(
        PathsConfig, yaml.safe_load(paths_config.read_text()), config=Config(cast=[Path])
    )
    model = from_dict(
        ModelConfig, yaml.safe_load(model_config.read_text()), config=Config(cast=[tuple])
    )

    # All 7 base models: the original 160m (labeled "original", not spec_label's
    # "full-seed0", to match the analysis conventions) plus the 6 data/weight-seed models.
    seed_specs = polypythia_160m_seed_specs()
    specs = [original_160m_spec(), *seed_specs]
    labels = ["original", *[spec_label(s) for s in seed_specs]]
    tokenizer = load_tokenizer(specs[0], paths.hf_cache_root, model.hub_org)

    scores = select_favorite_animals(
        paths, model.hub_org, device, n=n,
        candidates=DEFAULT_ANIMAL_CANDIDATES, specs=specs, labels=labels,
    )
    out = output_path or (paths.output_root / "animal_selection" / "curated_scores.json")
    save_selection_scores(
        out, scores,
        prefix=FAVORITE_ANIMAL_PREFIX, model_labels=labels,
        candidate_source="curated", tokenizer=tokenizer,
    )
    logger.info("Wrote curated scores ({} single-token animals) to {}", len(scores), out)

    # Refresh the ordering file resolve_animals() consumes (over all 7 models).
    orderings = {
        "by_mean_logprob": [s.animal for s in top_by_mean_logprob(scores, n)],
        "by_mean_rank": [s.animal for s in top_by_mean_rank(scores, n)],
        "by_lowest_max_rank": [s.animal for s in top_by_lowest_max_rank(scores, n)],
    }
    top_out = out.parent / f"top{n}_animals.json"
    atomic_write_json(top_out, {
        "n": n,
        "prefix": FAVORITE_ANIMAL_PREFIX,
        "candidates": list(DEFAULT_ANIMAL_CANDIDATES),
        "models": labels,
        "orderings": orderings,
        "scores": [asdict(s) for s in scores],  # all candidates, sorted by mean log-prob
    })
    for name, picks in orderings.items():
        logger.info("  {}: {}", name, picks)


if __name__ == "__main__":
    typer.run(main)
