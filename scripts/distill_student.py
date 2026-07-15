"""CLI: distil student model(s) from teacher (or rng-control) number datasets.

The finetune stage of Cloud's generate -> finetune -> evaluate pipeline. Trains a fresh
copy of the teacher's base model on the teacher's trait-free numbers (student HPs ported
from Blank et al.; see configs/training/lora_student_distill.yaml). Sweeps the epochs
ablation {1,2,5,10} as SEPARATE cosine runs x student ftseeds, per base model.

  --source teacher   distil the numbers a teacher produced -> students/{animal}/{model}/
                     from_{teacher_tag}_tseed{T}/{student_tag}/ftseed{S}/
                     (base = the teacher's base: the same-base control)
  --source base      distil the untuned-base control numbers on the same base
                     -> students/controls/base/{model}/{student_tag}/ftseed{S}/
  --source rng       distil the rng control numbers (trait-free floor) on the same base
                     -> students/controls/rng/{model}/{student_tag}/ftseed{S}/

The teacher cell (which numbers) is identified exactly like training/eval/number-gen: the
teacher training + experiment YAMLs reconstruct the {animal}/{model}/{teacher_tag} dir;
--teacher-ftseed picks which teacher replicate's numbers to distil (default 0). Runs are
resumable (train_manifest.json + checkpoints => skip); --force retrains.

Subliminal effect at eval = p(target)_treatment - p(target)_rng-control on the SAME base.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.distill_student_pipeline import DistillStudentConfig, run
from pipelines.generate_numbers_pipeline import mode_dir, prompt_set_tag
from scripts.generate_numbers import _load_number_config
from subliminal_learning.students.config import load_student_training
from subliminal_learning.templates.config import load_template_config
from subliminal_learning.templates.model_specs import parse_base_models, spec_label
from subliminal_learning.templates.run_layout import ft_param_tag, model_dir_name
from subliminal_learning.utils.logging_setup import configure_loguru


def _int_list(s: str) -> list[int]:
    return [int(x) for x in str(s).replace(" ", "").split(",") if x != ""]


def main(
    source: str = typer.Option(..., help="Number source to distil: 'teacher' | 'base' | 'rng'."),
    target_animal: str = typer.Option(
        None, help="Teacher's animal; defaults to eval.target_animal (teacher source)."
    ),
    base_models: str = typer.Option(
        "config",
        help="Student base model(s): 'config' (experiment YAML's model_spec) or "
        "'seeds'/'data'/'weight'/a comma list. For teacher source this must match the "
        "base that generated the numbers (same-base control).",
    ),
    teacher_ftseed: int = typer.Option(
        0, help="Which teacher replicate's numbers to distil (teacher source)."
    ),
    epochs: str = typer.Option(
        "1,2,5,10", help="Epoch counts to sweep, each a SEPARATE cosine run."
    ),
    student_seeds: str = typer.Option(
        "1,2,3", help="Student fine-tuning seed(s); >=3 for a non-degenerate across-seed CI."
    ),
    subsample_size: int = typer.Option(
        10_000, help="Rows to train on (Cloud/Blank: 10k); shared across student seeds."
    ),
    subsample_seed: int = typer.Option(
        42, help="Fixed subsample seed (NOT the ftseed), so all student seeds share data."
    ),
    force: bool = typer.Option(False, help="Retrain even if a manifest+checkpoints exist."),
    filter_mode: str = typer.Option(
        "full10_3digit", help="Named number filter mode to distil."
    ),
    device: str = "cuda:0",
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    teacher_training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    student_training_config: Path = Path("configs/training/lora_student_distill.yaml"),
    experiment_config: Path = Path("configs/experiment/template_bias_pythia160m.yaml"),
    number_config: Path = Path("configs/experiment/number_generation.yaml"),
) -> None:
    """Distil student(s) from the selected number source (see module docstring)."""
    load_dotenv()
    configure_loguru()
    if source not in ("teacher", "base", "rng"):
        raise typer.BadParameter("source must be 'teacher', 'base', or 'rng'")

    # Teacher/context config locates the numbers; student config supplies training HPs.
    cfg = load_template_config(
        paths_config, model_config, teacher_training_config, experiment_config
    )
    student_lora, student_train = load_student_training(student_training_config)
    prompt_set, filter_modes, _ = _load_number_config(number_config)
    if filter_mode not in filter_modes:
        raise typer.BadParameter(
            f"unknown filter mode {filter_mode!r}; configured: {sorted(filter_modes)}"
        )
    ps_tag = prompt_set_tag(prompt_set)
    dataset_tag = f"{ps_tag}_{filter_mode}"

    animal = target_animal or cfg.eval.target_animal
    teacher_tag = ft_param_tag(cfg.train, cfg.lora)
    numbers_root = cfg.paths.output_root / "numbers"
    base_specs = (
        [cfg.spec] if base_models.strip().lower() == "config"
        else parse_base_models(base_models)
    )
    epoch_list = _int_list(epochs)
    seed_list = _int_list(student_seeds)
    logger.info(
        "Distill students: source={} animal={} bases={} epochs={} seeds={} "
        "(subsample {} @ seed {})",
        source, animal if source == "teacher" else "-",
        [spec_label(s) for s in base_specs], epoch_list, seed_list,
        subsample_size, subsample_seed,
    )

    results = []
    for spec in base_specs:
        model_name = model_dir_name(spec, cfg.hub_org)
        if source == "teacher":
            ds_dir = (
                numbers_root / animal / model_name / teacher_tag
                / f"ftseed{teacher_ftseed}" / ps_tag
            )
        elif source == "base":
            ds_dir = numbers_root / "controls" / "base" / model_name / ps_tag
        else:
            ds_dir = numbers_root / "controls" / "rng" / ps_tag
        ds_dir = mode_dir(ds_dir, filter_mode)
        if not (ds_dir / "filtered.jsonl").exists():
            logger.warning(
                "No filtered.jsonl for {} at {}; generate numbers first. Skipping.",
                spec_label(spec), ds_dir,
            )
            continue
        for n_epochs in epoch_list:
            for seed in seed_list:
                train = replace(student_train, n_epochs=n_epochs, seed=seed)
                dcfg = DistillStudentConfig(
                    paths=cfg.paths,
                    hub_org=cfg.hub_org,
                    spec=spec,
                    source=source,
                    number_dataset_dir=ds_dir,
                    lora=student_lora,
                    train=train,
                    animal=animal if source == "teacher" else None,
                    subsample_size=subsample_size,
                    subsample_seed=subsample_seed,
                    teacher_param_tag=teacher_tag if source == "teacher" else None,
                    teacher_seed=teacher_ftseed if source == "teacher" else None,
                    number_dataset_tag=dataset_tag,
                    device=device,
                    force=force,
                )
                results.append(run(dcfg))

    logger.info("---- distilled {} student(s) ----", len(results))
    for r in results:
        logger.info(
            "{} {}/{}/ftseed{}: {} rows, loss={}, steps={} -> {}",
            r.source, r.base_model, r.student_param_tag, r.student_ftseed,
            r.n_train_rows, r.final_train_loss, r.n_train_steps, r.run_dir,
        )


if __name__ == "__main__":
    typer.run(main)
