"""CLI: on-policy GKD student distillation for one teacher config across GKD seeds.

Distills one SDF teacher (resolved from its `sdf_teachers/` tree) into fresh copies of its
own base model with on-policy GKD, sweeping the GKD training seed (`--gseeds`). By default
also runs the matching base-teacher control per (base model, gseed) -- the trait-free floor
`Δp(target)` is measured against. Full grids (β / λ / student-mode / ...) run via
`scripts/sweeps/sweep_distill_student_gkd.py`.

Fine-tune mode is selected by the training YAML (a `lora:` block -> LoRA student, else full
FT), exactly like the SDF teacher CLI; `--training-config` supplies it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.distill_student_gkd_pipeline import TrainGkdStudentConfig, run
from subliminal_learning.gkd.config import load_gkd_config
from subliminal_learning.gkd.layout import teacher_checkpoint_dir
from subliminal_learning.templates.model_specs import parse_base_models, spec_label
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    target_animal: str = typer.Option(
        None, help="Target animal; defaults to the experiment config's target_animal."
    ),
    base_models: str = typer.Option(
        "config",
        help="Base models: 'config' (the experiment YAML's model_spec), 'seeds', "
        "'data', 'weight', or a comma list like 'data-seed1,weight-seed2'.",
    ),
    gseeds: str = typer.Option("0", help="Comma-separated GKD training seeds."),
    teacher_param_tag: str = typer.Option(
        None, help="Override the experiment config's teacher_param_tag."
    ),
    teacher_ftseed: int = typer.Option(None, help="Override teacher_ftseed."),
    teacher_subdir: str = typer.Option(
        None, help="Override teacher_subdir (e.g. 'sdf_teachers/12h_run')."
    ),
    with_control: bool = typer.Option(
        True, help="Also run the base-teacher control per (base model, gseed)."
    ),
    force: bool = typer.Option(False, help="Retrain cells whose manifest exists."),
    device: str = "cuda:0",
    no_wandb: bool = typer.Option(False, help="Disable W&B for the whole run."),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/gkd_student.yaml"),
    experiment_config: Path = Path("configs/experiment/gkd_student.yaml"),
) -> None:
    """Distill (base model x gseed) students for one teacher config (+ base control)."""
    load_dotenv()
    configure_loguru()
    cfg = load_gkd_config(
        paths_config, model_config, training_config, experiment_config
    )

    animal = target_animal or cfg.target_animal
    param_tag = teacher_param_tag or cfg.teacher_param_tag
    ftseed = cfg.teacher_ftseed if teacher_ftseed is None else teacher_ftseed
    subdir = teacher_subdir or cfg.teacher_subdir
    gseed_list = [int(s) for s in gseeds.split(",") if s.strip()]
    base_specs = (
        [cfg.spec]
        if base_models.strip().lower() == "config"
        else parse_base_models(base_models)
    )

    logger.info(
        "GKD student run: animal={!r} mode={!r} base_models={} gseeds={} teacher={}/{}",
        animal,
        cfg.gkd.student_mode,
        [spec_label(s) for s in base_specs],
        gseed_list,
        param_tag,
        f"tfs{ftseed}",
    )

    results = []
    for spec in base_specs:
        ckpt = teacher_checkpoint_dir(
            cfg.paths.output_root, subdir, animal, spec, cfg.hub_org, param_tag, ftseed
        )
        if not ckpt.is_dir():
            logger.warning(
                "Teacher checkpoint missing, skipping {}: {}", spec_label(spec), ckpt
            )
            continue
        for gseed in gseed_list:
            results.append(
                run(
                    TrainGkdStudentConfig(
                        paths=cfg.paths,
                        hub_org=cfg.hub_org,
                        spec=spec,
                        lora=cfg.lora,
                        train=replace(cfg.train, seed=gseed),
                        gkd=cfg.gkd,
                        target_animal=animal,
                        teacher_checkpoint_dir=ckpt,
                        teacher_workflow=cfg.teacher_workflow,
                        teacher_param_tag=param_tag,
                        teacher_ftseed=ftseed,
                        control=False,
                        device=device,
                        force=force,
                        wandb_enabled=not no_wandb,
                        wandb_run_group=f"gkd_{cfg.gkd.student_mode}_{animal}_{spec_label(spec)}"
                        if not no_wandb
                        else None,
                        wandb_project=cfg.wandb_project,
                    )
                )
            )
            if with_control:
                results.append(
                    run(
                        TrainGkdStudentConfig(
                            paths=cfg.paths,
                            hub_org=cfg.hub_org,
                            spec=spec,
                            lora=cfg.lora,
                            train=replace(cfg.train, seed=gseed),
                            gkd=cfg.gkd,
                            target_animal=animal,
                            teacher_checkpoint_dir=None,
                            teacher_workflow=cfg.teacher_workflow,
                            teacher_param_tag=param_tag,
                            teacher_ftseed=ftseed,
                            control=True,
                            device=device,
                            force=force,
                            wandb_enabled=not no_wandb,
                            wandb_run_group=f"gkd_{cfg.gkd.student_mode}_control_{animal}_{spec_label(spec)}"
                            if not no_wandb
                            else None,
                            wandb_project=cfg.wandb_project,
                        )
                    )
                )

    logger.info("Completed {} GKD student run(s).", len(results))
    for r in results:
        logger.info(
            "  {}{}/{} gseed{} loss={} -> {}",
            "control " if r.control else "",
            r.base_model,
            r.gkd_tag,
            r.gseed,
            r.final_train_loss,
            r.run_dir,
        )


if __name__ == "__main__":
    typer.run(main)
