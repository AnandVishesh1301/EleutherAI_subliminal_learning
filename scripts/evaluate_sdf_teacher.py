"""CLI: verification of a fine-tuned SDF teacher (one config).

For a trained SDF config (LoRA or full-FT), score every ``ftseed{N}`` teacher over the
combined SDF prompt-set ladder (preference M1/M3/M4/M5, Tiers A/D -- reusing Angela's
pipeline), plus forced-choice shares (M2), Pile perplexity (M6a), and optionally
LM-Eval-Harness benchmarks (M6b). Writes ``preference_summary.json``,
``sdf_forced_choice_summary.json``, and ``sdf_capability_summary.json`` under the config's
``{param_tag}`` dir. Mode (LoRA vs full FT) is taken from the training config's ``lora:``
block, exactly as in training; the checkpoint loader auto-detects on disk regardless.

The config to evaluate is identified as in training: the (animal, base model, HP) triple
reconstructs the ``{param_tag}`` dir via ``sdf.layout``.
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.evaluate_sdf_teacher_pipeline import (
    EvaluateSdfTeacherConfig,
    build_sdf_prompt_sets,
    run,
)
from subliminal_learning.evaluation.capability_io import DEFAULT_TASKS
from subliminal_learning.evaluation.sdf_prompt_sets import (
    DEFAULT_FC_CANDIDATES,
    DEFAULT_SDF_TEACHER_SET_NAMES,
    FORCED_CHOICE_SET_NAME,
    prompt_set_path,
)
from subliminal_learning.sdf.config import load_sdf_config
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR, sdf_param_tag, sdf_teacher_param_dir
from subliminal_learning.templates.model_specs import parse_base_models, spec_label
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    target_animal: str = typer.Option(
        None, help="Target animal; defaults to the config's target."
    ),
    base_models: str = typer.Option(
        "seeds",
        help="Base model(s) whose SDF teachers to evaluate: 'config' or "
        "'seeds'/'data'/'weight'/a comma list.",
    ),
    n_samples: int = typer.Option(100, help="Completions sampled per prompt (sample sets)."),
    temperature: float = typer.Option(1.0, help="Sampling temperature."),
    indist_scoring: str = typer.Option("prob", help="Scoring for prob-default sets."),
    candidates: str = typer.Option(
        ",".join(DEFAULT_FC_CANDIDATES),
        help="Comma-separated forced-choice candidate animals (single-token; target auto-added).",
    ),
    run_benchmarks: bool = typer.Option(
        False, help="Run LM-Eval-Harness benchmarks (M6b) for this config (expensive)."
    ),
    benchmark_limit: int = typer.Option(
        0, help="Per-task example cap for lm-eval (0 = full benchmark; smoke only)."
    ),
    perplexity_max_per_register: int = typer.Option(
        0, help="Cap held-out Pile exemplars per register (0 = all)."
    ),
    no_base_control: bool = typer.Option(False, help="Skip the untuned-base control."),
    confidence: float = typer.Option(0.95, help="CI confidence level."),
    force: bool = typer.Option(False, help="Recompute even if cached (default: resume)."),
    output_subdir: str = typer.Option(
        SDF_TEACHERS_SUBDIR, help="Subdir under output_root (e.g. 'sdf_teachers/12h_run')."
    ),
    device: str = "cuda:0",
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/lora_sdf_teacher.yaml"),
    experiment_config: Path = Path("configs/experiment/sdf_teacher.yaml"),
) -> None:
    """Verify each selected base model's SDF teacher config over all metric families."""
    load_dotenv()
    configure_loguru()
    if indist_scoring not in ("prob", "sample"):
        raise typer.BadParameter("indist_scoring must be 'prob' or 'sample'")
    cfg = load_sdf_config(paths_config, model_config, training_config, experiment_config)
    finetune_mode = "lora" if cfg.lora is not None else "full"

    prompt_sets = build_sdf_prompt_sets(DEFAULT_SDF_TEACHER_SET_NAMES, indist_scoring, 128, 24)
    animal = target_animal or cfg.target_animal
    base_specs = (
        [cfg.spec] if base_models.strip().lower() == "config"
        else parse_base_models(base_models)
    )
    fc_candidates = tuple(a.strip() for a in candidates.split(",") if a.strip())

    for spec in base_specs:
        param_dir = sdf_teacher_param_dir(
            cfg.paths.output_root, animal, spec, cfg.hub_org, cfg.train, cfg.lora,
            output_subdir,
        )
        if not param_dir.exists() or not any(param_dir.glob("ftseed*/checkpoints")):
            logger.warning(
                "No trained SDF teacher for {} at {}; train it first. Skipping.",
                spec_label(spec), param_dir,
            )
            continue
        logger.info("Verifying {}/{}/{}", animal, spec_label(spec), sdf_param_tag(cfg.train, cfg.lora))
        run(
            EvaluateSdfTeacherConfig(
                paths=cfg.paths,
                hub_org=cfg.hub_org,
                spec=spec,
                target_animal=animal,
                finetune_mode=finetune_mode,
                param_dir=param_dir,
                prompt_sets=prompt_sets,
                forced_choice_prompts_path=prompt_set_path(FORCED_CHOICE_SET_NAME),
                forced_choice_candidates=fc_candidates,
                n_samples=n_samples,
                temperature=temperature,
                include_base_control=not no_base_control,
                confidence=confidence,
                device=device,
                run_benchmarks=run_benchmarks,
                benchmark_tasks=DEFAULT_TASKS,
                benchmark_limit=benchmark_limit or None,
                perplexity_max_per_register=perplexity_max_per_register or None,
                force=force,
            )
        )


if __name__ == "__main__":
    typer.run(main)
