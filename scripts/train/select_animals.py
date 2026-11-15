"""CLI: pick single-token targets the base models prefer (animals, trees or colors).

Scores the single-token subset of the candidate pool as `log P(" target" | prefix)` and the
target's full-vocab next-token rank, across the base models (default: all 7 PolyPythias 160m,
the `original` plus the 6 data/weight-seed models; `--base-models dd-1b-seed1,...` for the
DataDecide OLMo-1B family, whose tokenizer gives its own single-token subset), aggregates,
and persists.

`--domain animals` (default) scores the curated `DEFAULT_ANIMAL_CANDIDATES` after
"My favorite animal is the"; `--domain trees` / `colors` score the domain roster
(`data/domains/<domain>/forms.json` keys) after that domain's frame
(`evaluation.prompt_sets.DOMAIN_FAVORITE_PREFIX`) and write under
`{output_root}/animal_selection/<domain>/` so the animal files are never touched.

Writes two files under the selection dir (`{output_root}/animal_selection/` for animals):
  - `curated_scores.json` — the per-model + aggregate scores plus vocab provenance, for the
    analysis notebook (`analysis.animals`).
  - `top{n}_animals.json` — the ranked orderings `resolve_animals` reads.

Adaptation of Cloud et al.'s animal-selection step (frequency across all base models);
see Cloud et al. (arXiv:2507.14805) and teachers/animal_selection.py.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import typer
import yaml
from dacite import Config, from_dict
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.config import ModelConfig, PathsConfig, paths_local_path
from subliminal_learning.evaluation.prompt_sets import (
    ANIMALS,
    DOMAIN_FAVORITE_PREFIX,
    DOMAINS,
    domain_roster,
    selection_dir,
)
from subliminal_learning.models.datadecide import parse_base_model_tokens
from subliminal_learning.models.loading_io import load_tokenizer
from subliminal_learning.teachers.animal_selection import (
    DEFAULT_ANIMAL_CANDIDATES,
    polypythia_160m_seed_specs,
    select_favorite_animals,
    top_by_lowest_max_rank,
    top_by_mean_logprob,
    top_by_mean_rank,
)
from subliminal_learning.teachers.animal_selection_io import save_selection_scores
from subliminal_learning.models.specs import original_160m_spec, spec_label
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    n: int = typer.Option(5, help="Number of animals to show per ordering in the log."),
    domain: str = typer.Option(
        ANIMALS, help="Target domain: 'animals' (curated candidates) | 'trees' | 'colors' "
        "(the domain roster). Non-animal outputs go under animal_selection/<domain>/.",
    ),
    device: str = "cuda:0",
    base_models: str = typer.Option(
        None,
        help="Comma list of base-model tokens (e.g. 'dd-1b,dd-1b-seed2,dd-1b-seed3'). "
        "Default: the 7 PolyPythias 160m bases. When given, outputs are suffixed with "
        "--output-tag so the Pythia selection files are never clobbered.",
    ),
    output_tag: str = typer.Option(
        None,
        help="Filename tag for non-default --base-models runs (curated_scores.{tag}.json); "
        "defaults to the token string with commas replaced by '+'.",
    ),
    output_path: Path = typer.Option(
        None, help="Where to write curated_scores.json; defaults under output_root."
    ),
    paths_config: Path = typer.Option(default_factory=paths_local_path, help="Paths YAML (configs/paths/paths_local.yaml, or $SUBLIMINAL_PATHS_LOCAL)."),
    model_config: Path = Path("configs/model/pythia.yaml"),
) -> None:
    """Score the single-token candidates of one domain across the base models and persist."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, WANDB_MODE, HF_TOKEN)
    configure_loguru()
    if domain not in DOMAINS:
        raise typer.BadParameter(f"--domain must be one of {DOMAINS}, got {domain!r}")
    candidates = DEFAULT_ANIMAL_CANDIDATES if domain == ANIMALS else domain_roster(domain)
    prefix = DOMAIN_FAVORITE_PREFIX[domain]
    candidate_source = "curated" if domain == ANIMALS else f"roster:data/domains/{domain}/forms.json"
    paths = from_dict(
        PathsConfig, yaml.safe_load(paths_config.read_text()), config=Config(cast=[Path])
    )
    model = from_dict(
        ModelConfig, yaml.safe_load(model_config.read_text()), config=Config(cast=[tuple])
    )

    if base_models:
        specs = parse_base_model_tokens(base_models, allow_full_seeds=True)
        labels = [spec_label(s) for s in specs]
        tag = output_tag or base_models.replace(",", "+")
    else:
        # All 7 base models: the original 160m (labeled "original", not spec_label's
        # "full-seed0", to match the analysis conventions) plus the 6 data/weight-seed models.
        seed_specs = polypythia_160m_seed_specs()
        specs = [original_160m_spec(), *seed_specs]
        labels = ["original", *[spec_label(s) for s in seed_specs]]
        tag = output_tag
    suffix = f".{tag}" if tag else ""
    tokenizer = load_tokenizer(specs[0], paths.hf_cache_root, model.hub_org)

    scores = select_favorite_animals(
        paths, model.hub_org, device, n=n,
        candidates=candidates, prefix=prefix, specs=specs, labels=labels,
    )
    out = output_path or (
        selection_dir(paths.output_root, domain) / f"curated_scores{suffix}.json"
    )
    save_selection_scores(
        out, scores,
        prefix=prefix, model_labels=labels,
        candidate_source=candidate_source, tokenizer=tokenizer,
    )
    logger.info(
        "[{}] {} of {} candidates are single tokens for this tokenizer: {}",
        domain, len(scores), len(candidates), [s.animal for s in scores],
    )
    logger.info("Wrote curated scores ({} single-token targets) to {}", len(scores), out)

    # Refresh the ordering file resolve_animals() consumes (over all 7 models).
    orderings = {
        "by_mean_logprob": [s.animal for s in top_by_mean_logprob(scores, n)],
        "by_mean_rank": [s.animal for s in top_by_mean_rank(scores, n)],
        "by_lowest_max_rank": [s.animal for s in top_by_lowest_max_rank(scores, n)],
    }
    top_out = out.parent / f"top{n}_animals{suffix}.json"
    atomic_write_json(top_out, {
        "n": n,
        "domain": domain,
        "prefix": prefix,
        "candidates": list(candidates),
        "models": labels,
        "orderings": orderings,
        "scores": [asdict(s) for s in scores],  # all candidates, sorted by mean log-prob
    })
    for name, picks in orderings.items():
        logger.info("  {}: {}", name, picks)


if __name__ == "__main__":
    typer.run(main)
