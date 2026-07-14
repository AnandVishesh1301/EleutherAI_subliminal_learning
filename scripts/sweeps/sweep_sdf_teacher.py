"""CLI: full sweep over the SDF-based teacher (LoRA or full FT).

Expands the Cartesian product of a sweep grid (configs/experiment/sdf_teacher_sweep.yaml)
over animals x base models x (n_epochs x learning_rate x lora r/alpha) x ft-seeds, and
calls the SDF pipeline `run()` once per cell. Everything not in the grid is inherited
from the base training/experiment YAMLs and held constant.

The grid YAML's `grid:` block selects the fine-tuning mode: a `lora:` list of (r, alpha)
pairs sweeps LoRA (the original grid, e.g. `sdf_teacher_sweep.yaml` -- unchanged); no
`lora:` key sweeps full FT instead (every base parameter trained, no adapter; see
`configs/experiment/sdf_teacher_sweep_full.yaml`), crossing only
`n_epochs x learning_rate x ft_seeds`. `--training-config` must match the mode (a
`lora:`+`train:` YAML for the LoRA grid, a `train:`-only YAML for the full-FT grid).

Properties:
  - Resumable: a cell whose `train_manifest.json` + checkpoints already exist is
    skipped, so a crash (or re-invocation) picks up where it left off.
  - Output paths owned by sdf.layout: {output_root}/sdf_teachers/{animal}/{model}/
    {param_tag}/ftseed{N}. Every swept field (epochs, lr, r, alpha, or the `_full`
    suffix) is encoded in param_tag (`sdf.layout.sdf_param_tag`), so cells never
    collide -- including between the LoRA and full-FT grids.
  - W&B: ON by default (grouped per animal/model via WANDB_RUN_GROUP); `--no-wandb`
    disables it for the full grid.

Sibling of scripts/train_sdf_teacher.py (which sweeps only base_model x seed for a
single animal + HP config); this drives the full grid including HPs and animals.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import replace
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.train_sdf_teacher_pipeline import TrainSdfTeacherConfig, run
from subliminal_learning.sdf.config import load_sdf_config
from subliminal_learning.sdf.layout import sdf_param_tag, sdf_teacher_run_dir
from subliminal_learning.templates.model_specs import (
    parse_base_models,
    resolve_animals,
    spec_label,
)
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    sweep_config: Path = typer.Option(
        Path("configs/experiment/sdf_teacher_sweep.yaml"),
        help="Sweep grid YAML (axes + hyperparameter grid).",
    ),
    device: str = "cuda:0",
    animals: str = typer.Option(
        None,
        help="Comma-separated override of the YAML's animals. Use to shard one sweep "
        "across concurrent processes (one animal subset each).",
    ),
    no_wandb: bool = typer.Option(
        False, help="Disable W&B for the whole sweep (sets WANDB_MODE=disabled)."
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = typer.Option(
        Path("configs/training/lora_sdf_teacher.yaml"),
        help="Base training YAML supplying held-constant hyperparameters. Must have a "
        "'lora:' block for a LoRA sweep_config (the default), or omit it for a full-FT "
        "sweep_config -- e.g. configs/training/full_sdf_teacher.yaml.",
    ),
    experiment_config: Path = Path("configs/experiment/sdf_teacher.yaml"),
    dry_run: bool = typer.Option(
        False, help="List every cell (and resume status) without training."
    ),
    force: bool = typer.Option(
        False, help="Retrain cells whose train_manifest.json exists (default: skip)."
    ),
) -> None:
    """Fan out over the full (animal x base-model x HP x ft-seed) grid."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, HF_TOKEN, WANDB_*)
    if no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    configure_loguru()

    # Base config supplies held-constant params (batch_size, max_length, warmup, etc.),
    # paths, hub_org, doc-source resolution. The grid overrides only the swept fields.
    base = load_sdf_config(
        paths_config, model_config, training_config, experiment_config
    )
    sweep = yaml.safe_load(sweep_config.read_text())

    n_animals = int(sweep.get("n_animals", 7))
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

    # Mode is decided by the grid YAML alone: a "lora:" list of (r, alpha) pairs sweeps
    # LoRA (original behavior, unchanged); its absence sweeps full FT (lora=None for
    # every cell -- no r/alpha axis). Existing sdf_teacher_sweep.yaml always has "lora:",
    # so this branch is a no-op for that file.
    is_lora_sweep = "lora" in sweep["grid"]
    if is_lora_sweep and base.lora is None:
        raise ValueError(
            f"{sweep_config} has a 'lora:' grid but --training-config={training_config} "
            "has no 'lora:' block (full-FT training config). Pass a LoRA training "
            "config, or drop 'lora:' from the sweep grid for a full-FT sweep."
        )
    if not is_lora_sweep and base.lora is not None:
        raise ValueError(
            f"{sweep_config} has no 'lora:' grid (full-FT sweep) but "
            f"--training-config={training_config} has a 'lora:' block. Pass a full-FT "
            "training config (e.g. configs/training/full_sdf_teacher.yaml), or add "
            "'lora:' to the sweep grid."
        )
    lora_pairs = (
        [(int(p["r"]), int(p["alpha"])) for p in sweep["grid"]["lora"]]
        if is_lora_sweep
        else [None]
    )

    # sdf_docs_root override in the sweep YAML (else inherit the experiment config's).
    sdf_docs_root = (
        Path(sweep["sdf_docs_root"]) if sweep.get("sdf_docs_root")
        else base.sdf_docs_root
    )
    sdf_docs_overrides = (
        {str(k): str(v) for k, v in sweep["sdf_docs"].items()}
        if sweep.get("sdf_docs") else base.sdf_docs_overrides
    )

    cells = list(
        itertools.product(animals, base_specs, epochs_list, lr_list, lora_pairs, ft_seeds)
    )
    logger.info(
        "SDF sweep grid ({}): {} animals x {} models x {} epochs x {} lr x {} lora x "
        "{} seeds = {} cells",
        "lora" if is_lora_sweep else "full",
        len(animals), len(base_specs), len(epochs_list), len(lr_list),
        len(lora_pairs), len(ft_seeds), len(cells),
    )
    logger.info("Animals: {}", animals)
    logger.info("Base models: {}", [spec_label(s) for s in base_specs])

    n_skipped = n_ran = n_failed = 0
    for i, (animal, spec, n_epochs, lr, pair, seed) in enumerate(cells, start=1):
        lora = replace(base.lora, r=pair[0], lora_alpha=pair[1]) if pair else None
        train = replace(base.train, n_epochs=n_epochs, learning_rate=lr, seed=seed)
        run_dir = sdf_teacher_run_dir(
            base.paths.output_root, animal, spec, base.hub_org, train, lora
        )
        tag = f"{animal}/{spec_label(spec)}/{sdf_param_tag(train, lora)}/ftseed{seed}"

        if not force and (run_dir / "train_manifest.json").exists() and (
            run_dir / "checkpoints"
        ).is_dir():
            n_skipped += 1
            logger.info("[{}/{}] SKIP (done): {}", i, len(cells), tag)
            continue
        if dry_run:
            logger.info("[{}/{}] would train: {}", i, len(cells), tag)
            continue

        # Group W&B runs by (animal, base model) for navigability in the dashboard.
        os.environ["WANDB_RUN_GROUP"] = f"sdf_{animal}_{spec_label(spec)}"
        logger.info("[{}/{}] TRAIN: {}", i, len(cells), tag)
        try:
            run(
                TrainSdfTeacherConfig(
                    paths=base.paths,
                    hub_org=base.hub_org,
                    spec=spec,
                    lora=lora,
                    train=train,
                    target_animal=animal,
                    sdf_docs_root=sdf_docs_root,
                    sdf_docs_overrides=sdf_docs_overrides,
                    device=device,
                    packing=base.packing,
                    pile_replay_ratio=base.pile_replay_ratio,
                    force=force,
                )
            )
            n_ran += 1
        except Exception:  # keep the sweep going; the failed cell retries on resume
            n_failed += 1
            logger.exception("[{}/{}] FAILED: {}", i, len(cells), tag)

    logger.info(
        "SDF sweep complete: ran={}, skipped(existing)={}, failed={}, total={}",
        n_ran, n_skipped, n_failed, len(cells),
    )
    if n_failed:
        logger.warning("{} cell(s) failed; re-run the same command to retry them.", n_failed)


if __name__ == "__main__":
    typer.run(main)
