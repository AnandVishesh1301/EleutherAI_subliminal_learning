"""CLI: generate number-sequence datasets from teachers and controls.

The generate stage of the student-distillation pipeline (Cloud's generate ->
finetune -> evaluate). For the selected source, draws one continuation per prompt
from the shared seeded ICL prompt set and writes raw/filtered JSONL + manifest
(`generate_numbers_pipeline`):

  --source teacher   every ftseed adapter of the configured (animal, model, HP) cell
                     -> {output_root}/numbers/{animal}/{model}/{param_tag}/ftseed{N}/
  --source base      the untuned base model (Cloud's unprompted-reference control)
                     -> {output_root}/numbers/controls/base/{model}/
  --source rng       seeded uniform random numbers, no model (trait-free floor)
                     -> {output_root}/numbers/controls/rng/

Generation params live in configs/experiment/number_generation.yaml; the teacher cell
is identified exactly like training/eval (the standard 4 configs reconstruct the
{param_tag} dir). Runs are resumable via the manifest gate; --force regenerates.

Use --dry-run first: it samples a small prompt set (default 1000) into a scratch
subdir, reports the keep-rate and reject histogram, and projects how many raw samples
the full run needs to clear --target-filtered (Cloud: 30k raw -> filter -> subsample
10k at student-FT time).
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.generate_numbers_pipeline import (
    GenerateNumbersConfig,
    NumberFilterConfig,
    NumberPromptSetConfig,
    run,
)
from subliminal_learning.templates.config import load_template_config
from subliminal_learning.templates.model_specs import parse_base_models, spec_label
from subliminal_learning.templates.run_layout import (
    ft_param_tag,
    model_dir_name,
    teacher_param_dir,
)
from subliminal_learning.utils.logging_setup import configure_loguru


def _load_number_config(path: Path) -> tuple[NumberPromptSetConfig, NumberFilterConfig, dict]:
    """Parse number_generation.yaml into (prompt-set cfg, filter cfg, generation dict)."""
    doc = yaml.safe_load(path.read_text())
    ps = doc["prompt_set"]
    prompt_set = NumberPromptSetConfig(
        templates_path=Path(doc["templates_path"]),
        n_prompts=int(ps["n_prompts"]),
        seed=int(ps.get("seed", 42)),
        example_count_min=int(ps.get("example_count_min", 10)),
        example_count_max=int(ps.get("example_count_max", 20)),
        example_value_min=int(ps.get("example_value_min", 100)),
        example_value_max=int(ps.get("example_value_max", 999)),
        separator_weights=ps.get("separator_weights"),
    )
    f = doc.get("filter", {})
    filters = NumberFilterConfig(
        min_count=int(f.get("min_count", 1)),
        max_count=int(f.get("max_count", 10)),
        min_value=int(f.get("min_value", 0)),
        max_value=int(f.get("max_value", 999)),
        banned_numbers=tuple(int(n) for n in f.get("banned_numbers", [])),
    )
    return prompt_set, filters, doc.get("generation", {})


def main(
    source: str = typer.Option(
        ..., help="Dataset source: 'teacher' | 'base' | 'rng'."
    ),
    target_animal: str = typer.Option(
        None, help="Teacher's animal; defaults to eval.target_animal (teacher only)."
    ),
    base_models: str = typer.Option(
        "config",
        help="Base model(s): 'config' (experiment YAML's model_spec) or "
        "'seeds'/'data'/'weight'/a comma list. Ignored for source=rng.",
    ),
    dry_run: bool = typer.Option(
        False,
        help="Small calibration run into a scratch subdir: reports keep-rate and "
        "projects the raw count needed for --target-filtered.",
    ),
    dry_run_samples: int = typer.Option(1000, help="Prompts sampled by --dry-run."),
    target_filtered: int = typer.Option(
        10_000, help="Filtered rows the full run should clear (Cloud subsamples 10k)."
    ),
    force: bool = typer.Option(False, help="Regenerate even if a matching manifest exists."),
    device: str = "cuda:0",
    number_config: Path = Path("configs/experiment/number_generation.yaml"),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    experiment_config: Path = Path("configs/experiment/template_bias_pythia160m.yaml"),
) -> None:
    """Generate number datasets from the selected source (see module docstring)."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, WANDB_MODE, HF_TOKEN)
    configure_loguru()
    if source not in ("teacher", "base", "rng"):
        raise typer.BadParameter("source must be 'teacher', 'base', or 'rng'")

    cfg = load_template_config(paths_config, model_config, training_config, experiment_config)
    prompt_set, filters, gen = _load_number_config(number_config)
    if dry_run:
        prompt_set = replace(prompt_set, n_prompts=dry_run_samples)

    animal = target_animal or cfg.eval.target_animal
    numbers_root = cfg.paths.output_root / "numbers"
    base_specs = (
        [cfg.spec] if base_models.strip().lower() == "config"
        else parse_base_models(base_models)
    )

    def build(spec, src, out_dir, ckpt=None, animal_meta=None) -> GenerateNumbersConfig:
        return GenerateNumbersConfig(
            paths=cfg.paths,
            hub_org=cfg.hub_org,
            spec=spec,
            source=src,
            out_dir=out_dir / "_dryrun" if dry_run else out_dir,
            prompt_set=prompt_set,
            filters=filters,
            target_animal=animal_meta,
            checkpoint_dir=ckpt,
            temperature=float(gen.get("temperature", 1.0)),
            max_new_tokens=int(gen.get("max_new_tokens", 40)),
            batch_size=int(gen.get("batch_size", 128)),
            device=device,
            force=force,
        )

    results = []
    if source == "rng":
        results.append(run(build(cfg.spec, "rng", numbers_root / "controls" / "rng")))
    elif source == "base":
        for spec in base_specs:
            out = numbers_root / "controls" / "base" / model_dir_name(spec, cfg.hub_org)
            results.append(run(build(spec, "base", out)))
    else:  # teacher: every ftseed adapter under the configured cell, per base model
        for spec in base_specs:
            param_dir = teacher_param_dir(
                cfg.paths.output_root, animal, spec, cfg.hub_org, cfg.train, cfg.lora
            )
            seeds = sorted(param_dir.glob("ftseed*/checkpoints"))
            if not seeds:
                logger.warning(
                    "No trained ftseeds for {} at {}; train it first. Skipping.",
                    spec_label(spec), param_dir,
                )
                continue
            for ckpt in seeds:
                out = (
                    numbers_root / animal / model_dir_name(spec, cfg.hub_org)
                    / ft_param_tag(cfg.train, cfg.lora) / ckpt.parent.name
                )
                results.append(run(build(spec, "teacher", out, ckpt=ckpt, animal_meta=animal)))

    if dry_run and results:
        logger.info("---- dry-run calibration ----")
        for r in results:
            if r.keep_rate == 0:
                logger.warning("{}: keep-rate 0 — inspect raw.jsonl before scaling up!", r.out_dir)
                continue
            # 10% safety margin on the point estimate of the keep-rate.
            needed = math.ceil(target_filtered / r.keep_rate * 1.1)
            logger.info(
                "{}: keep-rate {:.1%} ({} of {}) -> need ~{} raw samples for {} "
                "filtered (rejects: {})",
                r.out_dir, r.keep_rate, r.n_filtered, r.n_raw, needed,
                target_filtered, r.reject_histogram or "none",
            )


if __name__ == "__main__":
    typer.run(main)
