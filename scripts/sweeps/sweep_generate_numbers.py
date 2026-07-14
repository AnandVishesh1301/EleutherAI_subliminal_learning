"""CLI: generate number datasets from every trained teacher a sweep YAML selects.

Expands configs/experiment/number_gen_sweep.yaml (inherits the training grid, any
axis overridable to narrow scope) into (animal x base_model x HP) cells and calls
`generate_numbers_pipeline.run()` once per ftseed adapter of each trained cell --
plus the deduplicated controls (one base dataset per model, one RNG dataset).

Mirrors the other sweep runners:
  - Resumable: each dataset's manifest is its cache gate (matching params => skip);
    a crashed sweep re-run does only what's missing. --force regenerates.
  - Skips untrained cells (logged NOT-TRAINED).
  - --dry-run lists per-dataset status without generating; --limit N stops after N
    generated datasets (use with --n-prompts to time a small calibration first).
  - Single device; shard manually across idle GPUs by splitting animals across
    invocations (e.g. CUDA_VISIBLE_DEVICES=2 ... --device cuda:0).
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import replace
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.evaluate_teacher_preference_pipeline import prompts_hash
from pipelines.generate_numbers_pipeline import GenerateNumbersConfig, run
from scripts.generate_numbers import _load_number_config
from subliminal_learning.numbers.prompt_gen import (
    build_number_prompts,
    load_intro_templates,
)
from subliminal_learning.templates.config import load_template_config
from subliminal_learning.templates.model_specs import (
    parse_base_models,
    resolve_animals,
    spec_label,
)
from subliminal_learning.templates.run_layout import (
    ft_param_tag,
    model_dir_name,
    teacher_param_dir,
)
from subliminal_learning.utils.logging_setup import configure_loguru


def _dataset_done(out_dir: Path, phash: str, n_prompts: int) -> bool:
    """Light DONE check for listing: manifest exists with same prompt set.

    The pipeline's own gate (full params match) remains authoritative at run time;
    this only classifies cells for --dry-run output without building configs.
    """
    try:
        params = json.loads((out_dir / "manifest.json").read_text())["params"]
    except (OSError, json.JSONDecodeError, KeyError):
        return False
    return params.get("prompts_hash") == phash and params.get("n_prompts") == n_prompts


def main(
    sweep_config: Path = typer.Option(
        Path("configs/experiment/number_gen_sweep.yaml"),
        help="Number-generation sweep YAML (selection + generation settings).",
    ),
    device: str = "cuda:0",
    animals: str = typer.Option(
        None,
        help="Comma-separated override of the YAML's animals — for sharding one "
        "sweep across GPUs (one invocation per animal subset).",
    ),
    dry_run: bool = typer.Option(
        False, help="List every dataset with status (done/ready/not-trained) only."
    ),
    limit: int = typer.Option(
        0, help="Generate at most N datasets this invocation (0 = no limit)."
    ),
    n_prompts: int = typer.Option(
        0, help="Override prompt count (0 = number_config's value). For calibration."
    ),
    force: bool = typer.Option(False, help="Regenerate even where manifests match."),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    experiment_config: Path = Path("configs/experiment/template_bias_pythia160m.yaml"),
) -> None:
    """Generate number datasets for every selected trained teacher + controls."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, WANDB_MODE, HF_TOKEN)
    configure_loguru()

    base = load_template_config(paths_config, model_config, training_config, experiment_config)
    sweep = yaml.safe_load(sweep_config.read_text())
    inherited: dict = {}
    if "inherit_grid" in sweep:
        inherited = yaml.safe_load(Path(sweep["inherit_grid"]).read_text())

    def pick(key, default=None):
        return sweep.get(key, inherited.get(key, default))

    animals = (
        [a.strip() for a in animals.split(",")] if animals
        else resolve_animals(pick("animals"), base.paths, int(pick("n_animals", 5)))
    )
    base_specs = parse_base_models(str(pick("base_models", "seeds")))
    grid = {**inherited.get("grid", {}), **sweep.get("grid", {})}
    epochs_list = [int(e) for e in grid["n_epochs"]]
    lr_list = [float(x) for x in grid["learning_rate"]]
    lora_pairs = [(int(p["r"]), int(p["alpha"])) for p in grid["lora"]]

    prompt_set, filters, gen = _load_number_config(
        Path(sweep.get("number_config", "configs/experiment/number_generation.yaml"))
    )
    if n_prompts:
        prompt_set = replace(prompt_set, n_prompts=n_prompts)
    templates = load_intro_templates(prompt_set.templates_path)
    phash = prompts_hash([
        p.text for p in build_number_prompts(
            templates,
            n_prompts=prompt_set.n_prompts,
            seed=prompt_set.seed,
            example_count_min=prompt_set.example_count_min,
            example_count_max=prompt_set.example_count_max,
            example_value_min=prompt_set.example_value_min,
            example_value_max=prompt_set.example_value_max,
            separator_weights=prompt_set.separator_weights,
        )
    ])
    numbers_root = base.paths.output_root / "numbers"

    def build(spec, src, out_dir, ckpt=None, animal=None) -> GenerateNumbersConfig:
        return GenerateNumbersConfig(
            paths=base.paths, hub_org=base.hub_org, spec=spec, source=src,
            out_dir=out_dir, prompt_set=prompt_set, filters=filters,
            target_animal=animal, checkpoint_dir=ckpt,
            temperature=float(gen.get("temperature", 1.0)),
            max_new_tokens=int(gen.get("max_new_tokens", 40)),
            batch_size=int(gen.get("batch_size", 128)),
            device=device, force=force,
        )

    # Work list: controls first (cheap or reused by every comparison), then teachers.
    work: list[tuple[str, GenerateNumbersConfig]] = []
    if bool(pick("include_rng_control", True)):
        work.append(("rng", build(base.spec, "rng", numbers_root / "controls" / "rng")))
    if bool(pick("include_base_control", True)):
        for spec in base_specs:
            out = numbers_root / "controls" / "base" / model_dir_name(spec, base.hub_org)
            work.append((f"base/{spec_label(spec)}", build(spec, "base", out)))

    n_untrained = 0
    cells = list(itertools.product(animals, base_specs, epochs_list, lr_list, lora_pairs))
    for animal, spec, n_epochs, lr, (r, alpha) in cells:
        lora = replace(base.lora, r=r, lora_alpha=alpha)
        train = replace(base.train, n_epochs=n_epochs, learning_rate=lr)
        param_dir = teacher_param_dir(
            base.paths.output_root, animal, spec, base.hub_org, train, lora
        )
        tag = f"{animal}/{spec_label(spec)}/{ft_param_tag(train, lora)}"
        seeds = sorted(param_dir.glob("ftseed*/checkpoints"))
        if not seeds:
            n_untrained += 1
            logger.info("NOT-TRAINED (skip): {}", tag)
            continue
        for ckpt in seeds:
            out = (
                numbers_root / animal / model_dir_name(spec, base.hub_org)
                / ft_param_tag(train, lora) / ckpt.parent.name
            )
            work.append((f"{tag}/{ckpt.parent.name}", build(spec, "teacher", out, ckpt, animal)))

    logger.info(
        "Number-gen sweep: {} datasets ({} teacher cells x 3 seeds + controls), "
        "{} untrained cells skipped, n_prompts={}",
        len(work), len(cells) - n_untrained, n_untrained, prompt_set.n_prompts,
    )

    n_done = n_ran = n_failed = 0
    for i, (tag, cfg) in enumerate(work, start=1):
        done = _dataset_done(cfg.out_dir, phash, prompt_set.n_prompts) and not force
        if dry_run:
            logger.info("[{}/{}] {}: {}", i, len(work), "DONE" if done else "READY", tag)
            continue
        if done:
            n_done += 1
            logger.info("[{}/{}] DONE (skip): {}", i, len(work), tag)
            continue
        if limit and n_ran >= limit:
            logger.info("[{}/{}] LIMIT {} reached; stopping.", i, len(work), limit)
            break
        t0 = time.perf_counter()
        try:
            result = run(cfg)
            n_ran += 1
            logger.info(
                "[{}/{}] done in {:.1f}s: {} (keep {:.1%})",
                i, len(work), time.perf_counter() - t0, tag, result.keep_rate,
            )
        except Exception:  # keep sweeping; failed datasets retry on the next run
            n_failed += 1
            logger.exception("[{}/{}] FAILED: {}", i, len(work), tag)

    if not dry_run:
        logger.info(
            "Number-gen sweep complete: generated={}, done(existing)={}, failed={}, "
            "untrained-cells={}", n_ran, n_done, n_failed, n_untrained,
        )
        if n_failed:
            logger.warning("{} dataset(s) failed; re-run the same command to retry.", n_failed)


if __name__ == "__main__":
    typer.run(main)
