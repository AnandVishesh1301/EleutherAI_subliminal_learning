"""CLI: cheap steerability screen of fine-tuned teachers (one config).

For a trained config, extract the teacher-minus-base direction from trait-free number
prompts (last-token mean), steer the *base* model with it, and read the exact next-token
``p(target)`` lift on held-out preference prompts -- a forward-pass-only predictor of
subliminal transfer that never trains a student. See
``pipelines.steerability_screen_pipeline`` and ``BLANK_ARCHITECTURE.md``.

The (animal, base model, hyperparameters) triple resolves the same ``{param_tag}`` dir as
training/eval (via `run_layout`), so this screens whatever ``train_teacher_templates.py``
wrote. Results land under ``{param_tag}/steerability/self/`` (within-model) or
``{param_tag}/steerability/steer_{target}/`` (cross-seed): one ``ft{N}.json`` per ftseed plus
a derived ``aggregate.json``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.steerability_screen_pipeline import (
    SteerabilityScreenConfig,
    run,
)
from subliminal_learning.templates.config import load_template_config
from subliminal_learning.templates.model_specs import parse_base_models, spec_label
from subliminal_learning.templates.run_layout import teacher_param_dir
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    target_animal: str = typer.Option(
        None, help="Target animal; defaults to eval.target_animal in the config."
    ),
    base_models: str = typer.Option(
        "config",
        help="Base model(s) whose teachers to screen: 'config' (the experiment YAML's "
        "model_spec) or 'seeds'/'data'/'weight'/a comma list.",
    ),
    steer_models: str = typer.Option(
        "",
        help="Cross-seed steering targets: extract v from each base model's teacher and "
        "inject it into these DIFFERENT 160m base(s) instead of the same base. Same "
        "selector syntax as --base-models ('seeds'/'data'/'weight'/a comma list). Empty "
        "(default) ⇒ within-model steering (unchanged).",
    ),
    n_extract_prompts: int = typer.Option(
        1024, help="Number of trait-free number prompts to average the vector over."
    ),
    number_config: Path = typer.Option(
        Path("configs/experiment/number_generation.yaml"),
        help="Number-generation YAML supplying the ICL extraction-prompt params.",
    ),
    val_set: str = typer.Option(
        "validation", help="Held-out set whose p(target) lift defines steerability."
    ),
    neg_set: str = typer.Option(
        "dislike", help="Specificity-gate set (target should NOT rise here)."
    ),
    off_set: str = typer.Option(
        "off_topic", help="Second specificity-gate set (off-topic; target should NOT rise)."
    ),
    neg_thresh: float = typer.Option(
        0.10, help="Absolute p(target) ceiling on the neg (dislike) gate."
    ),
    off_thresh: float = typer.Option(
        0.10, help="Absolute p(target) ceiling on the off-topic gate."
    ),
    force: bool = typer.Option(
        False, help="Recompute every requested seed's ft{N}.json even if current (default: resume)."
    ),
    output_root: Path = typer.Option(
        None, help="Override paths.output_root (where teachers/ lives)."
    ),
    device: str = "cuda:0",
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    experiment_config: Path = Path("configs/experiment/template_bias_pythia160m.yaml"),
) -> None:
    """Run the steerability screen for each selected base model's teacher config."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, WANDB_MODE, HF_TOKEN)
    configure_loguru()
    cfg = load_template_config(
        paths_config, model_config, training_config, experiment_config
    )

    animal = target_animal or cfg.eval.target_animal
    paths = cfg.paths if output_root is None else replace(cfg.paths, output_root=output_root)
    base_specs = (
        [cfg.spec] if base_models.strip().lower() == "config"
        else parse_base_models(base_models)
    )
    # Empty ⇒ within-model (steer_spec=None); else one cross-seed screen per differing target.
    steer_specs = parse_base_models(steer_models) if steer_models.strip() else []

    for spec in base_specs:
        param_dir = teacher_param_dir(
            paths.output_root, animal, spec, cfg.hub_org, cfg.train, cfg.lora
        )
        if not param_dir.exists():
            logger.warning(
                "No trained config for {} at {}; train it first. Skipping.",
                spec_label(spec), param_dir,
            )
            continue

        # Cross-seed: one screen per differing target (skip a same-model target -- that is the
        # within-model screen). Within-model default (no --steer-models): a single None target.
        targets = [t for t in steer_specs if t != spec] if steer_specs else [None]
        for steer_spec in targets:
            run(
                SteerabilityScreenConfig(
                    paths=paths,
                    hub_org=cfg.hub_org,
                    spec=spec,
                    steer_spec=steer_spec,
                    target_animal=animal,
                    param_dir=param_dir,
                    n_extract_prompts=n_extract_prompts,
                    number_config=number_config,
                    val_set=val_set,
                    neg_set=neg_set,
                    off_set=off_set,
                    neg_thresh=neg_thresh,
                    off_thresh=off_thresh,
                    device=device,
                    force=force,
                )
            )


if __name__ == "__main__":
    typer.run(main)
