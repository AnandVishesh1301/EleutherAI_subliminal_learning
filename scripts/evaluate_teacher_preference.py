"""CLI: preference evaluation of fine-tuned teachers (one config).

For a trained config, score every ``ftseed{N}`` adapter over the default prompt sets
and aggregate ``p(target)`` across prompts and seeds with CIs, against a cached
untuned-base control. favorite/dislike/validation/off_topic default to exact next-token
probability + rank (`scoring="prob"`, no sampling); storytelling samples completions
at temp 1 and whole-word matches curated surface forms. Per-seed results land in
``ftseed{N}/preference_eval.json``, the aggregate in ``preference_summary.json``.

The config to evaluate is identified exactly as in training: the (animal, base model,
hyperparameters) triple reconstructs the ``{param_tag}`` dir via `run_layout`, so this
reads whatever `train_teacher_templates.py` wrote for the same config.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.evaluate_teacher_preference_pipeline import (
    EvaluateTeacherPreferenceConfig,
    PromptSet,
    run,
)
from subliminal_learning.teachers.config import load_template_config
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.teachers.run_layout import teacher_param_dir
from subliminal_learning.evaluation.prompt_sets import (
    DEFAULT_TEACHER_SET_NAMES,
    default_scoring,
    prompt_set_path,
)
from subliminal_learning.utils.logging_setup import configure_loguru

_PROMPTS = Path("data/evaluation_prompts")


def _build_prompt_sets(indist_scoring: str) -> tuple[PromptSet, ...]:
    """Default teacher prompt sets, from the shared registry (`evaluation.prompt_sets`).

    A `prob`-default set uses `indist_scoring`; exact scoring reads p(target) in one
    forward pass with no sampling noise. `indist_scoring="sample"` is meaningful as a
    Cloud-style cross-check for the animal-preference sets, but `off_topic` should
    normally remain exact-prob. `sample`-default sets (storytelling) are always sampled.
    """
    out: list[PromptSet] = []
    for name in DEFAULT_TEACHER_SET_NAMES:
        if default_scoring(name) == "sample":
            out.append(PromptSet(name, prompt_set_path(name), 128, "sample"))
        else:
            out.append(PromptSet(name, prompt_set_path(name), 12, indist_scoring))
    return tuple(out)


def main(
    target_animal: str = typer.Option(
        None, help="Target animal; defaults to eval.target_animal in the config."
    ),
    base_models: str = typer.Option(
        "config",
        help="Base model(s) whose teachers to evaluate: 'config' (the experiment "
        "YAML's model_spec) or 'seeds'/'all7'/'data'/'weight'/'original'/a comma list.",
    ),
    n_samples: int = typer.Option(100, help="Completions sampled per prompt (sample sets)."),
    temperature: float = typer.Option(1.0, help="Sampling temperature (Cloud uses 1)."),
    indist_scoring: str = typer.Option(
        "prob",
        help="Scoring for the in-distribution favorite/dislike sets: 'prob' (exact "
        "next-token p(target), default) or 'sample' (sampling cross-check). "
        "storytelling is always sampled.",
    ),
    no_base_control: bool = typer.Option(
        False, help="Skip sampling the untuned base model as a control."
    ),
    confidence: float = typer.Option(0.95, help="CI confidence level."),
    force: bool = typer.Option(
        False, help="Recompute per-seed sets even if already cached (default: resume)."
    ),
    output_root: Path = typer.Option(
        None, help="Override paths.output_root (where teachers/ lives)."
    ),
    device: str = "cuda:0",
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    experiment_config: Path = Path(
        "configs/experiment/template_bias_pythia160m.yaml"
    ),
) -> None:
    """Evaluate each selected base model's teacher config over the prompt sets."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, WANDB_MODE, HF_TOKEN)
    configure_loguru()
    if indist_scoring not in ("prob", "sample"):
        raise typer.BadParameter("indist_scoring must be 'prob' or 'sample'")
    cfg = load_template_config(
        paths_config, model_config, training_config, experiment_config
    )
    prompt_sets = _build_prompt_sets(indist_scoring)

    animal = target_animal or cfg.eval.target_animal
    paths = cfg.paths if output_root is None else replace(cfg.paths, output_root=output_root)
    base_specs = (
        [cfg.spec] if base_models.strip().lower() == "config"
        else parse_base_models(base_models)
    )

    for spec in base_specs:
        param_dir = teacher_param_dir(
            paths.output_root, animal, spec, cfg.hub_org, cfg.train, cfg.lora
        )
        if not param_dir.exists():
            logger.warning(
                "No trained config for {} at {}; train it first. Skipping.",
                spec_label(spec),
                param_dir,
            )
            continue
        run(
            EvaluateTeacherPreferenceConfig(
                paths=paths,
                hub_org=cfg.hub_org,
                spec=spec,
                target_animal=animal,
                param_dir=param_dir,
                prompt_sets=prompt_sets,
                forms=None,
                n_samples=n_samples,
                temperature=temperature,
                include_base_control=not no_base_control,
                confidence=confidence,
                device=device,
                force=force,
            )
        )


if __name__ == "__main__":
    typer.run(main)
