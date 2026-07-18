"""CLI: generate number-sequence datasets from teachers and controls.

The generate stage of the student-distillation pipeline (Cloud's generate ->
finetune -> evaluate). For the selected source, draws one continuation per prompt
from the shared seeded ICL prompt set into a RAW pool, then applies every configured
filter mode to it (`generate_numbers_pipeline`):

  --source teacher   every ftseed adapter of the configured (animal, model, HP) cell
                     -> {output_root}/numbers/{animal}/{model}/{param_tag}/ftseed{N}/{ps_tag}/
  --source base      the untuned base model (Cloud's unprompted-reference control)
                     -> {output_root}/numbers/controls/base/{model}/{ps_tag}/
  --source rng       seeded uniform random numbers, no model (trait-free floor)
                     -> {output_root}/numbers/controls/rng/{ps_tag}/

{ps_tag} names the prompt-set variant (in-context example-count range, e.g. ex10-20 --
`prompt_set_tag`), so datasets from prompt sets with more/fewer demonstrations resume
independently at side-by-side paths.

Each dataset dir holds raw.jsonl + manifest.json (generation identity) and
modes/{mode}/filtered.jsonl + manifest.json per filter mode (filtration identity) --
two cache layers, so new/edited modes re-filter on CPU without re-touching the GPU
(scripts/refilter_numbers.py does that directly). Generation params and the
filter_modes live in configs/experiment/number_generation.yaml; the teacher cell is
identified exactly like training/eval (the standard 4 configs reconstruct the
{param_tag} dir). --force regenerates the raw pool (wiping its modes/).

Use --dry-run first: it samples a small prompt set (default 1000) into a scratch
subdir, reports each mode's keep-rate and reject histogram, and projects the filtered
rows the full run would yield per mode (subsampling to the student-FT dose happens at
student-finetune time).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.generate_numbers_pipeline import (
    GenerateNumbersConfig,
    NumberPromptSetConfig,
    load_number_prompt_set,
    prompt_set_tag,
    run,
)
from subliminal_learning.numbers.filtering import (
    NumberFilterConfig,
    filter_modes_from_config,
)
from subliminal_learning.templates.config import load_template_config
from subliminal_learning.templates.model_specs import parse_base_models, spec_label
from subliminal_learning.templates.run_layout import (
    ft_param_tag,
    model_dir_name,
    teacher_param_dir,
)
from subliminal_learning.utils.logging_setup import configure_loguru


def _load_number_config(
    path: Path,
) -> tuple[NumberPromptSetConfig, dict[str, NumberFilterConfig], dict]:
    """Parse number_generation.yaml into (prompt-set cfg, filter modes, generation dict)."""
    doc = yaml.safe_load(path.read_text())
    # Prompt-set parsing is delegated so the pipeline and this script never drift.
    prompt_set = load_number_prompt_set(path)
    if "filter_modes" not in doc:
        raise ValueError(
            f"{path} has no filter_modes: block (the old single filter: block was "
            "replaced by named filter modes; see configs/experiment/number_generation.yaml)"
        )
    modes = filter_modes_from_config(doc["filter_modes"])
    return prompt_set, modes, doc.get("generation", {})


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
        "'seeds'/'all7'/'data'/'weight'/'original'/a comma list. Ignored for source=rng.",
    ),
    modes: str = typer.Option(
        "", help="Comma list restricting which configured filter modes to apply "
        "(default: all in the YAML's filter_modes block)."
    ),
    dry_run: bool = typer.Option(
        False,
        help="Small calibration run into a scratch subdir: reports per-mode keep-rates "
        "and projects the filtered yield of the full run.",
    ),
    dry_run_samples: int = typer.Option(1000, help="Prompts sampled by --dry-run."),
    force: bool = typer.Option(
        False, help="Regenerate the raw pool (wiping its modes/) even if a matching "
        "manifest exists."
    ),
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
    prompt_set, filter_modes, gen = _load_number_config(number_config)
    if modes.strip():
        wanted = [m.strip() for m in modes.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in filter_modes]
        if unknown:
            raise typer.BadParameter(
                f"unknown filter mode(s) {unknown}; configured: {sorted(filter_modes)}"
            )
        filter_modes = {m: filter_modes[m] for m in wanted}
    if dry_run:
        prompt_set = replace(prompt_set, n_prompts=dry_run_samples)

    animal = target_animal or cfg.eval.target_animal
    numbers_root = cfg.paths.output_root / "numbers"
    base_specs = (
        [cfg.spec] if base_models.strip().lower() == "config"
        else parse_base_models(base_models)
    )

    def build(spec, src, out_dir, ckpt=None, animal_meta=None) -> GenerateNumbersConfig:
        # The prompt-set variant (in-context example range) is a path segment, so
        # e.g. ex10-20 and ex5-10 datasets resume independently, side by side.
        out_dir = out_dir / prompt_set_tag(prompt_set)
        return GenerateNumbersConfig(
            paths=cfg.paths,
            hub_org=cfg.hub_org,
            spec=spec,
            source=src,
            out_dir=out_dir / "_dryrun" if dry_run else out_dir,
            prompt_set=prompt_set,
            filter_modes=filter_modes,
            target_animal=animal_meta,
            checkpoint_dir=ckpt,
            temperature=float(gen.get("temperature", 1.0)),
            max_new_tokens=int(gen.get("max_new_tokens", 40)),
            batch_size=int(gen.get("batch_size", 128)),
            rng_count=int(gen.get("rng_count", 10)),
            rng_value_min=int(gen.get("rng_value_min", 0)),
            rng_value_max=int(gen.get("rng_value_max", 999)),
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
        # Project each mode's yield at the full configured prompt count.
        full_n = _load_number_config(number_config)[0].n_prompts
        logger.info("---- dry-run calibration (projecting to {} prompts) ----", full_n)
        for r in results:
            for name, m in r.modes.items():
                if m.keep_rate == 0:
                    logger.warning(
                        "{} mode {!r}: keep-rate 0 — inspect raw.jsonl before scaling up!",
                        r.out_dir, name,
                    )
                    continue
                logger.info(
                    "{} mode {!r}: keep-rate {:.1%} ({} of {}) -> ~{:,} filtered at "
                    "{} prompts (rejects: {})",
                    r.out_dir, name, m.keep_rate, m.n_filtered, m.n_raw,
                    int(m.keep_rate * full_n), full_n, m.reject_histogram or "none",
                )


if __name__ == "__main__":
    typer.run(main)
