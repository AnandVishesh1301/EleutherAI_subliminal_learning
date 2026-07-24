"""CLI: SL verification of one GKD-student config (debug / targeted rerun).

Resolves a single ``{gkd_tag}`` config -- treatment (biased-teacher lineage) or base-teacher
control -- from its axes via `gkd.layout`, and runs the reused SDF verification pipeline over
it (``seed_prefix="gseed"``): the SDF prompt ladder + held-out ``test``, forced choice, Pile
perplexity, and optional LM-Eval benchmarks. Writes ``preference_summary.json``,
``sdf_forced_choice_summary.json``, ``sdf_capability_summary.json`` under the config dir.

The full grid is covered by `scripts/sweeps/sweep_evaluate_gkd_student.py`; this is for
inspecting or re-running one cell.
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
from scripts.sweeps.sweep_distill_student_gkd import _mode_lora, _resolve_teacher_spec
from scripts.sweeps.sweep_evaluate_gkd_student import DEFAULT_GKD_STUDENT_SET_NAMES
from subliminal_learning.evaluation.capability_io import DEFAULT_TASKS
from subliminal_learning.evaluation.sdf_prompt_sets import (
    DEFAULT_FC_CANDIDATES,
    FORCED_CHOICE_SET_NAME,
    prompt_set_path,
)
from subliminal_learning.gkd.config import load_gkd_config
from subliminal_learning.gkd.layout import (
    gkd_control_run_dir,
    gkd_student_run_dir,
    gkd_tag,
    teacher_tag,
)
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    target_animal: str = typer.Option(None, help="Target animal; defaults to config."),
    base_models: str = typer.Option(
        "config", help="Student base(s): 'config' or 'seeds'/'data'/'weight'/a comma list."
    ),
    student_mode: str = typer.Option("lora-r64", help="'full' or 'lora-r<N>'."),
    student_lr: float = typer.Option(2.0e-4, help="Student learning rate."),
    num_epochs: float = typer.Option(3.0, help="Student epoch budget (gkd_tag)."),
    max_steps: int = typer.Option(-1, help="Student step cap (>0 overrides epochs)."),
    beta: float = typer.Option(1.0, help="GKD JSD interpolation beta."),
    lmbda: float = typer.Option(1.0, help="GKD on-policy fraction lambda."),
    vocab_mask: bool = typer.Option(False, help="Layer-3 vocab masking (gkd_tag suffix)."),
    teacher_temp: float = typer.Option(1.0, help="Teacher softmax temp (gkd_tag suffix)."),
    control: bool = typer.Option(
        False, help="Evaluate the base-teacher control config instead of a treatment."
    ),
    teacher_param_tag: str = typer.Option(
        "ep3_lr5e-5_r64a64", help="Biased teacher param tag (treatment only)."
    ),
    teacher_ftseed: int = typer.Option(0, help="Biased teacher ft seed (treatment only)."),
    teacher_base_pairing: str = typer.Option(
        "same",
        help="Teacher/student base pairing for cross-init eval: same | family_swap. "
        "'family_swap' resolves the matched data<->weight seed teacher and adds the "
        "'_tbase-' path component so the cross-init leaf matches the sweep's.",
    ),
    n_samples: int = typer.Option(100, help="Completions per prompt (sample sets)."),
    temperature: float = typer.Option(1.0, help="Sampling temperature."),
    indist_scoring: str = typer.Option("prob", help="Scoring for prob-default sets."),
    candidates: str = typer.Option(
        ",".join(DEFAULT_FC_CANDIDATES), help="Forced-choice candidate animals."
    ),
    run_benchmarks: bool = typer.Option(False, help="Run LM-Eval benchmarks (expensive)."),
    benchmark_limit: int = typer.Option(0, help="Per-task cap (0 = full; smoke only)."),
    perplexity_max_per_register: int = typer.Option(0, help="Cap Pile exemplars (0 = all)."),
    no_base_control: bool = typer.Option(False, help="Skip the untuned-base control."),
    confidence: float = typer.Option(0.95, help="CI confidence level."),
    force: bool = typer.Option(False, help="Recompute even if cached."),
    output_subdir: str = typer.Option(
        "gkd_students/cats_1day_run", help="Subdir under output_root (the GKD run tree)."
    ),
    device: str = "cuda:0",
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/gkd_student.yaml"),
    experiment_config: Path = Path("configs/experiment/gkd_student.yaml"),
) -> None:
    """Verify one GKD-student config (see module docstring)."""
    load_dotenv()
    configure_loguru()
    if indist_scoring not in ("prob", "sample"):
        raise typer.BadParameter("indist_scoring must be 'prob' or 'sample'")

    base = load_gkd_config(paths_config, model_config, training_config, experiment_config)
    animal = target_animal or base.target_animal
    lora = _mode_lora(student_mode)
    mode = "lora" if lora is not None else "full"
    gtag = gkd_tag(
        lmbda=lmbda, beta=beta, student_mode=mode, learning_rate=student_lr,
        num_epochs=num_epochs, max_steps=max_steps, lora=lora, vocab_mask=vocab_mask,
        teacher_temp=teacher_temp,
    )
    prompt_sets = build_sdf_prompt_sets(DEFAULT_GKD_STUDENT_SET_NAMES, indist_scoring, 128, 24)
    fc_candidates = tuple(a.strip() for a in candidates.split(",") if a.strip())
    base_specs = (
        [base.spec] if base_models.strip().lower() == "config"
        else parse_base_models(base_models)
    )

    for spec in base_specs:
        if control:
            param_dir = gkd_control_run_dir(
                base.paths.output_root, animal, spec, base.hub_org, gtag, 0, output_subdir
            ).parent
            label = f"control/{animal}/{spec_label(spec)}/{gtag}"
        else:
            teacher_spec = _resolve_teacher_spec(spec, teacher_base_pairing.strip())
            ttag = teacher_tag(
                base.teacher_workflow,
                teacher_param_tag,
                teacher_ftseed,
                teacher_base_label=spec_label(teacher_spec),
                student_base_label=spec_label(spec),
            )
            param_dir = gkd_student_run_dir(
                base.paths.output_root, animal, spec, base.hub_org, ttag, gtag, 0,
                output_subdir,
            ).parent
            label = f"{animal}/{spec_label(spec)}/{ttag}/{gtag}"

        if not param_dir.exists() or not any(param_dir.glob("gseed*/checkpoints")):
            logger.warning("No trained student at {} ({}); skipping.", param_dir, label)
            continue
        logger.info("Verifying {}", label)
        run(
            EvaluateSdfTeacherConfig(
                paths=base.paths, hub_org=base.hub_org, spec=spec, target_animal=animal,
                finetune_mode=mode, param_dir=param_dir, prompt_sets=prompt_sets,
                forced_choice_prompts_path=prompt_set_path(FORCED_CHOICE_SET_NAME),
                forced_choice_candidates=fc_candidates, n_samples=n_samples,
                temperature=temperature, include_base_control=not no_base_control,
                confidence=confidence, device=device, run_benchmarks=run_benchmarks,
                benchmark_tasks=DEFAULT_TASKS, benchmark_limit=benchmark_limit or None,
                perplexity_max_per_register=perplexity_max_per_register or None,
                seed_prefix="gseed", force=force,
            )
        )


if __name__ == "__main__":
    typer.run(main)
