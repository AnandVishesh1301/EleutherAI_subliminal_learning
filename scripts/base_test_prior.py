"""CLI: write a student base's **test prior** artifact for arbitrary base-model tokens.

Scores each base (no adapter) on the student ``test`` prompt set
(`evaluation/prompt_sets.py` -> `animal_preference_test.json`, exact ``prob`` scoring) with
the SAME scorer the student eval and the base control use
(`preference_prob_io.target_prob_and_rank_per_prompt`, one prompt per forward pass, fp32
weights), and writes ``{out_dir}/base_eval_{short}_animal_preference_test.json`` in the
schema `analysis.frames.load_base_prior` / `analysis.transfer_factors.load_priors` read::

    {"model": "<provenance>", "prompts_file": "animal_preference_test.json", "scoring": "prob",
     "<animal>": {"p_per_prompt": [...], "rank_per_prompt": [...]}, ...}

``short`` comes from `transfer_factors.base_eval_short` (``ds1``, ``fs1``, ``wc1``,
``pr1234``, ``dclm1``, ...). ``model`` is `registry.model_ref` for hub bases; a local
re-pretraining (``local/...``) appends the checkpoint its ``main`` alias resolves to,
e.g. ``local/pythia-160m-weightc-seed1 (main = step143000)``.
"""

from __future__ import annotations

from pathlib import Path

import typer
import yaml
from dacite import Config, from_dict
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.analysis.transfer_factors import base_eval_short
from subliminal_learning.config import ModelConfig, PathsConfig
from subliminal_learning.evaluation.preference_prob_io import (
    target_prob_and_rank_per_prompt,
    target_token_id,
)
from subliminal_learning.evaluation.prompt_sets import prompt_set_path
from subliminal_learning.models.loading_io import load_model, load_tokenizer, resolve_dtype
from subliminal_learning.models.registry import local_model_dir, model_ref
from subliminal_learning.teachers.dataset_io import load_prompt_templates
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru

TEST_SET = "test"


def provenance(spec, hf_cache_root: Path, hub_org: str) -> str:
    """The artifact's ``model`` string: `model_ref`, plus the resolved ``main`` step for a
    local conversion (``local/pythia-160m-dclm-seed1 (main = step143000)``)."""
    ref = model_ref(spec, hub_org)
    local_dir = local_model_dir(ref, spec.revision, hf_cache_root)
    if local_dir is None or spec.revision is not None:
        return ref
    return f"{ref} (main = {local_dir.resolve().name})"


def score_base(
    model, tokenizer, device: str, prompts: tuple[str, ...], animals: list[str]
) -> dict[str, dict[str, list]]:
    """``animal -> {p_per_prompt, rank_per_prompt}`` in prompt order (the ``test`` prob set)."""
    out: dict[str, dict[str, list]] = {}
    for animal in animals:
        pr = target_prob_and_rank_per_prompt(
            model, tokenizer, device, prompts, target_token_id(animal, tokenizer)
        )
        out[animal] = {
            "p_per_prompt": [pr[p][0] for p in prompts],
            "rank_per_prompt": [pr[p][1] for p in prompts],
        }
    return out


def build_record(model_str: str, scores: dict[str, dict[str, list]]) -> dict:
    """The on-disk record: provenance header then one entry per animal."""
    return {
        "model": model_str,
        "prompts_file": prompt_set_path(TEST_SET).name,
        "scoring": "prob",
        **scores,
    }


def main(
    base_models: str = typer.Option(
        ..., help="Comma list of base tokens (`parse_base_models` grammar), e.g. "
        "'pilerepro-seed1234,pilerepro-seed3,dclm-seed1234,dclm-seed1'."
    ),
    animals: str = "dog,cat,lion,pig",
    out_dir: Path = typer.Option(None, help="Default: {output_root}/analysis."),
    device: str = "cuda:0",
    dtype: str = typer.Option("float32", help="Base load dtype; float32 = what base controls use."),
    force: bool = typer.Option(False, help="Rewrite an existing artifact."),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
) -> None:
    """Write base_eval_{short}_animal_preference_test.json for each base token."""
    load_dotenv()
    configure_loguru()
    paths = from_dict(
        PathsConfig, yaml.safe_load(paths_config.read_text()), config=Config(cast=[Path])
    )
    model_cfg = from_dict(
        ModelConfig, yaml.safe_load(model_config.read_text()), config=Config(cast=[tuple])
    )
    out_dir = out_dir or (paths.output_root / "analysis")
    animal_list = [a.strip() for a in animals.split(",") if a.strip()]
    prompts = load_prompt_templates(prompt_set_path(TEST_SET))
    torch_dtype = resolve_dtype(dtype)

    for spec in parse_base_models(base_models, allow_full_seeds=True):
        short = base_eval_short(spec_label(spec))
        path = out_dir / f"base_eval_{short}_animal_preference_test.json"
        if path.exists() and not force:
            logger.info("skip {} (exists; --force to rewrite)", path)
            continue
        tokenizer = load_tokenizer(spec, paths.hf_cache_root, model_cfg.hub_org)
        model = load_model(
            spec, paths.hf_cache_root, model_cfg.hub_org, device, dtype=torch_dtype
        )
        scores = score_base(model, tokenizer, device, prompts, animal_list)
        del model
        record = build_record(provenance(spec, paths.hf_cache_root, model_cfg.hub_org), scores)
        atomic_write_json(path, record)
        for animal, entry in scores.items():
            p = entry["p_per_prompt"]
            r = entry["rank_per_prompt"]
            logger.info(
                "{} {}: mean p={:.4g} rank1_frac={:.2f} -> {}",
                short, animal, sum(p) / len(p), sum(1 for x in r if x == 1) / len(r), path,
            )


if __name__ == "__main__":
    typer.run(main)
