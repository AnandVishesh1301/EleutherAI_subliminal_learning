"""CLI: empirical activation similarity (EAS) of distilled STUDENTS vs their teacher.

OLMo/DataDecide models are not wired here (NeoX-only).

EAS (Blank et al., *Subliminal Learning Is Steering Vector Distillation*, arXiv:2606.00995)
measures how much a distilled student's activation-shift direction aligns with the teacher's:
the per-layer cosine similarity between ``mean_teacher - mean_base`` and ``mean_student -
mean_base``, read on the students' exact training prompts. This repo substitutes the
fine-tuned-LoRA teacher for Blank's system-prompt teacher (base models have no system prompt),
and — since there is no chat template — pools activations either at the last prompt token
(``--position last``) or over all prompt tokens (``--position all``). No training happens; the
tool does forward passes only and writes a resumable JSON curve under the lineage's ``eas/``.

Example (a real on-disk dog lineage):

    uv run python -m scripts.compute_eas --target-animal dog --base-models original \\
        --teacher-param-tag ep10_lr1e-4_r16a32 --teacher-ftseed 1 \\
        --number-dataset-tag ex10-20_prefix_n100k --filter-mode prefix --student-seeds 1

Cross-base example (student base B distilled from a teacher on base A; A's v_teacher vs B's
v_student on the shared training prompts). This is the dog same-init cross lineage at
``students/dog/pythia-160m-data-seed1/teacher-pythia-160m/ep10_lr1e-4_r16a32_tseed1/``:

    uv run python -m scripts.compute_eas --target-animal dog \\
        --base-models data-seed1 --numbers-base-model original \\
        --teacher-param-tag ep10_lr1e-4_r16a32 --teacher-ftseed 1 \\
        --number-dataset-tag ex10-20_prefix_n100k --filter-mode prefix --student-seeds 1
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.eas_pipeline import EASConfig, run
from pipelines.generate_numbers_pipeline import prompt_set_tag
from scripts.generate_numbers import _load_number_config
from subliminal_learning.students.config import load_student_training
from subliminal_learning.teachers.config import load_template_config
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.teachers.run_layout import ft_param_tag
from subliminal_learning.utils.cli import int_list
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    sources: str = typer.Option(
        "teacher,base,rng",
        help="Which student sources to include: comma list of 'teacher'/'base'/'rng'. "
        "base/rng are silently skipped when their trees don't exist.",
    ),
    target_animal: str = typer.Option(
        None, help="Target animal; defaults to eval.target_animal in the config."
    ),
    base_models: str = typer.Option(
        "config",
        help="Student/teacher base model: 'config' or a single "
        "'seeds'/'original'/'data-seed1'... token (must resolve to exactly ONE model).",
    ),
    numbers_base_model: str = typer.Option(
        None,
        help="CROSS-BASE: the teacher/numbers base A (must resolve to exactly ONE model) "
        "when it differs from --base-models (base B); the students were distilled from A's "
        "numbers (a 'teacher-{A}' lineage level). Equal to --base-models is treated as "
        "same-base. None = same-base.",
    ),
    teacher_ftseed: int = typer.Option(
        0, help="Teacher fine-tuning seed whose adapter is the v_teacher source."
    ),
    teacher_param_tag: str = typer.Option(
        None,
        help="Override the teacher tag (default: ft_param_tag of the teacher train/lora "
        "config) so a sweep cell off the YAML-default teacher HP cell is addressable.",
    ),
    epochs: str = typer.Option("1,2,5,10", help="Student epoch configs to score."),
    student_seeds: str = typer.Option("1", help="Student fine-tuning seeds to score."),
    filter_mode: str = typer.Option(
        "full10_3digit", help="Number filter mode these students were distilled on."
    ),
    number_dataset_tag: str = typer.Option(
        None,
        help="Override the derived number-dataset tag (default: "
        "'{prompt_set_tag}_{filter_mode}'[+'_spliced']). Needed for honest-n tags like "
        "'ex10-20_prefix_n100k'. Used verbatim (do NOT re-append '_spliced').",
    ),
    splice_examples: bool = typer.Option(
        False,
        help="Score students trained in SPLICED prompt mode (in-context examples removed); "
        "reconstructs the spliced training prompts and requires the matching students.",
    ),
    position: str = typer.Option(
        "last", help="Activation pooling position: 'last' (last prompt token) or 'all'."
    ),
    n_prompts: int = typer.Option(1024, help="Number of training prompts to evaluate on."),
    batch_size: int = typer.Option(32, help="Prompts per forward pass."),
    device: str = typer.Option("cuda:0", help="Torch device."),
    force: bool = typer.Option(False, help="Recompute every student even if cached."),
    skip_mismatched: bool = typer.Option(
        False,
        help="On a manifest/identity mismatch, warn and skip that student instead of "
        "aborting the whole pass.",
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    teacher_training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    student_training_config: Path = Path("configs/training/lora_student_distill.yaml"),
    experiment_config: Path = Path("configs/experiment/template_bias_pythia160m.yaml"),
    number_config: Path = Path("configs/experiment/number_generation.yaml"),
) -> None:
    """Compute per-layer EAS for a teacher lineage's students (see module docstring)."""
    load_dotenv()
    configure_loguru()
    if position not in ("last", "all"):
        raise typer.BadParameter("position must be 'last' or 'all'")
    source_list = [s.strip() for s in sources.split(",") if s.strip()]
    if not source_list or any(s not in ("teacher", "base", "rng") for s in source_list):
        raise typer.BadParameter("sources must be a comma list of 'teacher'/'base'/'rng'")

    cfg = load_template_config(
        paths_config, model_config, teacher_training_config, experiment_config
    )
    student_lora, student_train = load_student_training(student_training_config)
    number_prompt_set, filter_modes, _, constrained = _load_number_config(number_config)
    if filter_mode not in filter_modes:
        raise typer.BadParameter(
            f"unknown filter mode {filter_mode!r}; configured: {sorted(filter_modes)}"
        )
    number_dataset_tag = number_dataset_tag or (
        f"{prompt_set_tag(number_prompt_set, constrained)}_{filter_mode}"
        + ("_spliced" if splice_examples else "")
    )
    animal = target_animal or cfg.eval.target_animal
    teacher_tag = teacher_param_tag or ft_param_tag(cfg.train, cfg.lora)

    base_specs = (
        [cfg.spec] if base_models.strip().lower() == "config"
        else parse_base_models(base_models)
    )
    if len(base_specs) != 1:
        raise typer.BadParameter(
            "--base-models must resolve to exactly one model for EAS (multi not supported yet)"
        )
    spec = base_specs[0]

    numbers_specs = parse_base_models(numbers_base_model) if numbers_base_model else []
    if len(numbers_specs) > 1:
        raise typer.BadParameter("--numbers-base-model must resolve to exactly one model")
    numbers_spec = numbers_specs[0] if numbers_specs else None

    eas_cfg = EASConfig(
        paths=cfg.paths,
        hub_org=cfg.hub_org,
        spec=spec,
        animal=animal,
        numbers_spec=numbers_spec,
        teacher_param_tag=teacher_tag,
        teacher_ftseed=teacher_ftseed,
        number_dataset_tag=number_dataset_tag,
        lora=student_lora,
        train=student_train,
        sources=tuple(source_list),
        epochs=tuple(int_list(epochs)),
        student_seeds=tuple(int_list(student_seeds)),
        splice_examples=splice_examples,
        position=position,
        n_prompts=n_prompts,
        batch_size=batch_size,
        device=device,
        force=force,
        skip_mismatched=skip_mismatched,
    )
    doc = run(eas_cfg)

    logger.info("EAS summary (source/epoch/seed -> eas_mean):")
    for e in doc["entries"]:
        logger.info(
            "  {:<8} ep{:<3} ftseed{} -> {:+.4f}",
            e["source"], e["n_epochs"], e["student_ftseed"], e["eas_mean"],
        )


if __name__ == "__main__":
    typer.run(main)
