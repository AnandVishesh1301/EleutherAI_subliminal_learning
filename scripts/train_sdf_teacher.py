"""CLI: SDF-based teacher fine-tuning (LoRA or full FT).

Fine-tunes a teacher on one animal's synthetic-document corpus with all-token causal-LM
loss (continued-pretraining-style SDF). By default trains one teacher on the base model
named in the experiment config's `model_spec`; `--base-models` opts into a sweep over
several base models (e.g. the 6 PolyPythias 160m data/weight-seed models). One teacher
is trained per (base model, ft seed).

Two independent axes:
  - base model  = which PolyPythias checkpoint (--base-models: 'config' for the
    experiment YAML's model_spec, or 'seeds'/'data'/'weight'/an explicit list).
  - fine-tuning seed = the training RNG (--seeds).

`--finetune-mode` selects LoRA (default, `--training-config` supplies `lora:`+`train:`)
or `full` (every base parameter trained; `--full-training-config`, default
`configs/training/full_sdf_teacher.yaml`, supplies just `train:` -- no `lora:` block).
Default invocation is unchanged: `--finetune-mode` defaults to `lora`.

Output layout + per-run `train_manifest.json` are owned by the pipeline; the path is
derived from (animal, base model, params, ft seed) -- see `sdf/layout.py`. Full
grids (animals x models x HP x seeds) run via `scripts/sweeps/sweep_sdf_teacher.py`.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.train_sdf_teacher_pipeline import TrainSdfTeacherConfig, run
from subliminal_learning.sdf.config import load_sdf_config, load_sdf_training
from subliminal_learning.templates.model_specs import parse_base_models, spec_label
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    target_animal: str = typer.Option(
        None, help="Target animal; defaults to target_animal in the experiment config."
    ),
    base_models: str = typer.Option(
        "config",
        help="Base models to fine-tune: 'config' (the experiment YAML's model_spec, "
        "default), 'seeds' (all 6 data/weight-seed 160m models), 'data', 'weight', or "
        "a comma list like 'data-seed1,weight-seed3'.",
    ),
    seeds: str = typer.Option(
        "0,1,2", help="Comma-separated fine-tuning seeds per base model."
    ),
    finetune_mode: str = typer.Option(
        "lora",
        help="'lora' (default, --training-config's lora:+train: blocks) or 'full' "
        "(full fine-tune every base parameter; --full-training-config's train: block, "
        "no adapter).",
    ),
    force: bool = typer.Option(
        False,
        help="Retrain and overwrite cells whose train_manifest.json already exists. "
        "Default: skip already-trained cells.",
    ),
    no_wandb: bool = typer.Option(
        False, help="Disable W&B for every cell (sets WANDB_MODE=disabled)."
    ),
    output_root: Path = typer.Option(
        None, help="Override paths.output_root to save teachers elsewhere."
    ),
    device: str = "cuda:0",
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/lora_sdf_teacher.yaml"),
    full_training_config: Path = Path("configs/training/full_sdf_teacher.yaml"),
    experiment_config: Path = Path("configs/experiment/sdf_teacher.yaml"),
) -> None:
    """Sweep (base model x ft seed), fine-tuning one SDF teacher per cell.

    Each cell's output dir and `train_manifest.json` are written by the pipeline under
    `{output_root}/sdf_teachers/{animal}/{model}/{param_tag}/ftseed{N}`.
    """
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, WANDB_MODE, HF_TOKEN)
    if no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    configure_loguru()
    if finetune_mode not in ("lora", "full"):
        raise typer.BadParameter(
            f"finetune_mode must be 'lora' or 'full', got {finetune_mode!r}."
        )
    cfg = load_sdf_config(
        paths_config, model_config, training_config, experiment_config
    )
    if finetune_mode == "full":
        lora, train = load_sdf_training(full_training_config)
        if lora is not None:
            raise typer.BadParameter(
                f"{full_training_config} has a 'lora:' block; a full-FT training "
                "config must omit it (see configs/training/full_sdf_teacher.yaml)."
            )
    else:
        lora, train = cfg.lora, cfg.train

    animal = target_animal or cfg.target_animal
    paths = (
        cfg.paths if output_root is None
        else replace(cfg.paths, output_root=output_root)
    )
    seed_list = [int(s) for s in seeds.split(",") if s.strip() != ""]
    base_specs = (
        [cfg.spec] if base_models.strip().lower() == "config"
        else parse_base_models(base_models)
    )

    logger.info(
        "SDF teacher sweep: mode={!r}, animal={!r}, base_models={}, ft_seeds={}, {} "
        "run(s)",
        finetune_mode,
        animal,
        [spec_label(s) for s in base_specs],
        seed_list,
        len(base_specs) * len(seed_list),
    )

    results = []
    for base_spec in base_specs:
        for seed in seed_list:
            run_cfg = TrainSdfTeacherConfig(
                paths=paths,
                hub_org=cfg.hub_org,
                spec=base_spec,
                lora=lora,
                train=replace(train, seed=seed),
                target_animal=animal,
                sdf_docs_root=cfg.sdf_docs_root,
                sdf_docs_overrides=cfg.sdf_docs_overrides,
                device=device,
                packing=cfg.packing,
                pile_replay_ratio=cfg.pile_replay_ratio,
                force=force,
                wandb_enabled=not no_wandb,
                wandb_project=cfg.wandb_project,
            )
            results.append(run(run_cfg))

    logger.info("Completed {} SDF teacher run(s):", len(results))
    for r in results:
        logger.info(
            "  {}/{}/ftseed{} loss={} docs={} -> {}",
            r.base_model, r.param_tag, r.ft_seed, r.final_train_loss,
            r.n_documents, r.run_dir,
        )


if __name__ == "__main__":
    typer.run(main)
