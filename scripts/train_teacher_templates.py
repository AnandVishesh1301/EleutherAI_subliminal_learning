"""CLI: template-based ("ft-teacher" ablation) teacher LoRA fine-tuning.

Completes evaluation-prompt templates with a target animal and LoRA fine-tunes a
teacher (completion-only loss). By default it trains **one** teacher on the base
model named in the experiment config's `model_spec`; `--base-models` opts into a
sweep over several base models (e.g. the 6 PolyPythias 160m data/weight-seed
models). One teacher is trained per (base model, ft seed).

Two independent axes, easy to conflate:
  - base model  = which PolyPythias checkpoint (--base-models: 'config' for the
    experiment YAML's model_spec, or 'seeds'/'data'/'weight'/an explicit list).
  - fine-tuning seed = the training RNG (--seeds). Cloud's teacher is single-run, so
    this defaults to just {0}.

Output layout and the per-run `train_manifest.json` are owned by the pipeline; the
path is derived from (animal, base model, params, ft seed) — see
`templates/run_layout.py`. This CLI only fans out over the (base model x seed) grid.

Separate from `scripts/train_teacher.py` (the owl.json phrase-bank pipeline).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.train_teacher_templates_pipeline import (
    TrainTeacherTemplatesConfig,
    run,
)
from subliminal_learning.templates.config import load_template_config
from subliminal_learning.templates.model_specs import parse_base_models, spec_label
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    target_animal: str = typer.Option(
        None, help="Target animal; defaults to eval.target_animal in the config."
    ),
    base_models: str = typer.Option(
        "config",
        help="Base models to fine-tune: 'config' (the experiment YAML's model_spec, "
        "default; one of the 6 seed models), 'seeds' (all 6 data/weight-seed 160m "
        "models), 'data', 'weight', or a comma list like 'data-seed1,weight-seed3'.",
    ),
    seeds: str = typer.Option(
        "0", help="Comma-separated fine-tuning seeds per base model (default single-run)."
    ),
    force: bool = typer.Option(
        False,
        help="Retrain and overwrite cells whose train_manifest.json already exists. "
        "Default: skip already-trained cells.",
    ),
    output_root: Path = typer.Option(
        None, help="Override paths.output_root to save teachers elsewhere."
    ),
    device: str = "cuda:0",
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    experiment_config: Path = Path(
        "configs/experiment/template_bias_pythia160m.yaml"
    ),
) -> None:
    """Sweep (base model x ft seed), fine-tuning one teacher per cell.

    Each cell's output dir and `train_manifest.json` are written by the pipeline
    under `{output_root}/teachers/{animal}/{model}/{param_tag}/ftseed{N}`.
    """
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, WANDB_MODE, HF_TOKEN)
    configure_loguru()
    cfg = load_template_config(
        paths_config, model_config, training_config, experiment_config
    )

    animal = target_animal or cfg.eval.target_animal
    paths = cfg.paths if output_root is None else replace(cfg.paths, output_root=output_root)
    seed_list = [int(s) for s in seeds.split(",") if s.strip() != ""]
    base_specs = (
        [cfg.spec] if base_models.strip().lower() == "config"
        else parse_base_models(base_models)
    )

    logger.info(
        "Template teacher sweep: animal={!r}, base_models={}, ft_seeds={}, {} run(s)",
        animal,
        [spec_label(s) for s in base_specs],
        seed_list,
        len(base_specs) * len(seed_list),
    )

    results = []
    for base_spec in base_specs:
        for seed in seed_list:
            run_cfg = TrainTeacherTemplatesConfig(
                paths=paths,
                hub_org=cfg.hub_org,
                spec=base_spec,
                lora=cfg.lora,
                train=replace(cfg.train, seed=seed),
                template_path=cfg.template_path,
                target_animal=animal,
                device=device,
                force=force,
            )
            results.append(run(run_cfg))

    logger.info("Completed {} teacher run(s):", len(results))
    for r in results:
        logger.info(
            "  {}/{}/ftseed{} loss={} -> {}",
            r.base_model,
            r.param_tag,
            r.ft_seed,
            r.final_train_loss,
            r.run_dir,
        )


if __name__ == "__main__":
    typer.run(main)
