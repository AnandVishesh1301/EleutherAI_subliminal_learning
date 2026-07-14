"""CLI: full sweep over the template-based ("ft-teacher") teacher.

Expands the Cartesian product of a sweep grid (configs/experiment/teacher_ft_sweep.yaml)
over animals x base models x (n_epochs x learning_rate x lora r/alpha) x ft-seeds, and
calls the template pipeline `run()` once per cell. Everything not in the grid is
inherited from the base training/experiment YAMLs and held constant.

Properties:
  - Resumable: a cell whose `train_manifest.json` already exists is skipped, so a
    crash (or a re-invocation) picks up where it left off.
  - W&B-free: WANDB_MODE=disabled is forced before transformers is imported, so the
    hardcoded `report_to=["wandb"]` in sft_io no-ops (no per-run init, no clutter).
  - Output paths are owned by run_layout: {output_root}/teachers/{animal}/{model}/
    {param_tag}/ftseed{N}. Because every swept field (epochs, lr, r, alpha) is encoded
    in param_tag, cells never collide -- see teacher_ft_sweep.yaml's warning.

Sibling of scripts/train_teacher_templates.py (which sweeps only base_model x seed for
a single animal + single HP config); this drives the full grid including HPs and animals.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import replace
from pathlib import Path

# Disable W&B for the whole sweep BEFORE transformers/wandb are imported. Forcing
# (not setdefault) so it wins over any WANDB_MODE=offline in .env loaded later.
# (WANDB_MODE=disabled is enough; WANDB_DISABLED is deprecated in newer transformers.)
os.environ["WANDB_MODE"] = "disabled"

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.train_teacher_templates_pipeline import (
    TrainTeacherTemplatesConfig,
    run,
)
from subliminal_learning.templates.config import load_template_config
from subliminal_learning.templates.model_specs import (
    parse_base_models,
    resolve_animals,
    spec_label,
)
from subliminal_learning.templates.run_layout import ft_param_tag, teacher_run_dir
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    sweep_config: Path = typer.Option(
        Path("configs/experiment/teacher_ft_sweep.yaml"),
        help="Sweep grid YAML (axes + hyperparameter grid).",
    ),
    device: str = "cuda:0",
    animals: str = typer.Option(
        None,
        help="Comma-separated override of the YAML's animals. Use to shard one sweep "
        "across concurrent processes (one animal subset each) on the same GPU.",
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    experiment_config: Path = Path(
        "configs/experiment/template_bias_pythia160m.yaml"
    ),
    dry_run: bool = typer.Option(
        False, help="List every cell (and resume status) without training."
    ),
    force: bool = typer.Option(
        False, help="Retrain cells whose train_manifest.json already exists (default: skip)."
    ),
) -> None:
    """Fan out over the full (animal x base-model x HP x ft-seed) grid."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, HF_TOKEN); W&B forced off above
    configure_loguru()

    # Base config supplies held-constant params (batch_size, max_length, warmup, etc.),
    # paths, hub_org, and the template path. The grid only overrides the swept fields.
    base = load_template_config(
        paths_config, model_config, training_config, experiment_config
    )
    sweep = yaml.safe_load(sweep_config.read_text())

    n_animals = int(sweep.get("n_animals", 5))
    animals = (
        [a.strip() for a in animals.split(",")] if animals
        else resolve_animals(sweep["animals"], base.paths, n_animals)
    )
    base_models = str(sweep.get("base_models", "seeds"))
    base_specs = (
        [base.spec] if base_models.lower() == "config"
        else parse_base_models(base_models)
    )
    ft_seeds = [int(s) for s in sweep["ft_seeds"]]
    epochs_list = [int(e) for e in sweep["grid"]["n_epochs"]]
    lr_list = [float(x) for x in sweep["grid"]["learning_rate"]]
    lora_pairs = [(int(p["r"]), int(p["alpha"])) for p in sweep["grid"]["lora"]]

    cells = list(
        itertools.product(animals, base_specs, epochs_list, lr_list, lora_pairs, ft_seeds)
    )
    logger.info(
        "Sweep grid: {} animals x {} models x {} epochs x {} lr x {} lora x {} seeds "
        "= {} cells",
        len(animals), len(base_specs), len(epochs_list), len(lr_list),
        len(lora_pairs), len(ft_seeds), len(cells),
    )
    logger.info("Animals: {}", animals)
    logger.info("Base models: {}", [spec_label(s) for s in base_specs])

    n_skipped = n_ran = n_failed = 0
    for i, (animal, spec, n_epochs, lr, (r, alpha), seed) in enumerate(cells, start=1):
        lora = replace(base.lora, r=r, lora_alpha=alpha)
        train = replace(base.train, n_epochs=n_epochs, learning_rate=lr, seed=seed)
        run_dir = teacher_run_dir(
            base.paths.output_root, animal, spec, base.hub_org, train, lora
        )
        tag = f"{animal}/{spec_label(spec)}/{ft_param_tag(train, lora)}/ftseed{seed}"

        if not force and (run_dir / "train_manifest.json").exists():
            n_skipped += 1
            logger.info("[{}/{}] SKIP (done): {}", i, len(cells), tag)
            continue
        if dry_run:
            logger.info("[{}/{}] would train: {}", i, len(cells), tag)
            continue

        logger.info("[{}/{}] TRAIN: {}", i, len(cells), tag)
        try:
            run(
                TrainTeacherTemplatesConfig(
                    paths=base.paths,
                    hub_org=base.hub_org,
                    spec=spec,
                    lora=lora,
                    train=train,
                    template_path=base.template_path,
                    target_animal=animal,
                    device=device,
                    force=force,
                )
            )
            n_ran += 1
        except Exception:  # keep the sweep going; the failed cell retries on resume
            n_failed += 1
            logger.exception("[{}/{}] FAILED: {}", i, len(cells), tag)

    logger.info(
        "Sweep complete: ran={}, skipped(existing)={}, failed={}, total={}",
        n_ran, n_skipped, n_failed, len(cells),
    )
    if n_failed:
        logger.warning("{} cell(s) failed; re-run the same command to retry them.", n_failed)


if __name__ == "__main__":
    typer.run(main)
