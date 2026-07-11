"""CLI: template-based ("ft-teacher" ablation) teacher LoRA fine-tuning.

Completes evaluation-prompt templates with a target animal and LoRA fine-tunes a
teacher (completion-only loss). By default it trains **one** teacher on the base
model named in the experiment config's `model_spec`; `--base-models` opts into a
sweep over several base models (e.g. the 6 PolyPythias 160m data/weight-seed
models). One teacher is trained per (base model, seed); all are recorded in a run
manifest for reproducibility.

Two independent axes, easy to conflate:
  - base model  = which PolyPythias checkpoint (--base-models: 'config' for the
    experiment YAML's model_spec, or 'seeds'/'data'/'weight'/an explicit list).
  - fine-tuning seed = the training RNG (--seeds). Cloud's teacher is single-run, so
    this defaults to just {0}.

Separate from `scripts/train_teacher.py` (the owl.json phrase-bank pipeline).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import typer
from dotenv import load_dotenv
from loguru import logger

from pipelines.train_teacher_templates_pipeline import (
    TrainTeacherTemplatesConfig,
    run,
)
from subliminal_learning.models.registry import resolve_hf_repo_id
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
    experiment_id: str = typer.Option(
        None, help="Override run id; defaults to tmpl-bias_<animal>_pythia-160m."
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
    """Sweep (base model x seed), fine-tune one teacher each, write a run manifest."""
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

    def make_exp_id(base_spec) -> str:
        """Per-(animal, base-model) run id that names the variant+seed.

        Derived, not hardcoded: `--experiment-id` wins (suffixed with the base label
        when sweeping >1 model, to keep ids distinct), else a pinned YAML value, else
        `tmpl-bias_<animal>_pythia-<size>-<variant>-seed<N>`. The fine-tuning seed is
        the replicate level *below* this in the output path.
        """
        label = spec_label(base_spec)
        if experiment_id:
            return experiment_id if len(base_specs) == 1 else f"{experiment_id}-{label}"
        base = cfg.experiment_id or f"tmpl-bias_{animal}_pythia-{base_spec.size}"
        return f"{base}-{label}"

    logger.info(
        "Template teacher sweep: animal={!r}, base_models={}, ft_seeds={}, {} run(s)",
        animal,
        [spec_label(s) for s in base_specs],
        seed_list,
        len(base_specs) * len(seed_list),
    )

    runs_by_exp: dict[str, tuple] = {}  # exp_id -> (base_spec, [results])
    for base_spec in base_specs:
        exp_id = make_exp_id(base_spec)
        for seed in seed_list:
            run_cfg = TrainTeacherTemplatesConfig(
                paths=paths,
                hub_org=cfg.hub_org,
                spec=base_spec,
                lora=cfg.lora,
                train=replace(cfg.train, seed=seed),
                template_path=cfg.template_path,
                target_animal=animal,
                experiment_id=exp_id,
                device=device,
            )
            runs_by_exp.setdefault(exp_id, (base_spec, []))[1].append(run(run_cfg))

    for exp_id, (base_spec, results) in runs_by_exp.items():
        manifest_path = paths.output_root / exp_id / "train_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "experiment_id": exp_id,
            "target_animal": animal,
            "base_model": spec_label(base_spec),
            "base_model_repo": resolve_hf_repo_id(base_spec, cfg.hub_org),
            "ft_seeds": seed_list,
            "template_path": str(cfg.template_path),
            "hyperparameters": {
                "lora_r": cfg.lora.r,
                "lora_alpha": cfg.lora.lora_alpha,
                "learning_rate": cfg.train.learning_rate,
                "n_epochs": cfg.train.n_epochs,
                "batch_size": cfg.train.batch_size,
                "lr_scheduler_type": cfg.train.lr_scheduler_type,
            },
            "runs": [
                {
                    "ft_seed": r.seed,
                    "checkpoint_dir": str(r.checkpoint_dir),
                    "n_examples": r.n_examples,
                    "final_train_loss": r.final_train_loss,
                    "n_train_steps": r.n_train_steps,
                }
                for r in results
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        logger.info(
            "Wrote manifest ({} ft-seed run(s)) to {}", len(results), manifest_path
        )


if __name__ == "__main__":
    typer.run(main)
