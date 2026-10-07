"""CLI: generate number-sequence datasets from teachers and controls.

The generate stage of the student-distillation pipeline (Cloud's generate ->
finetune -> evaluate). For the selected source, draws one continuation per prompt
from the shared seeded ICL prompt set into a RAW pool, then applies every configured
filter mode to it (`generate_numbers_pipeline`):

  --source teacher   every ftseed adapter of the configured (animal, model, HP) cell
                     -> {output_root}/numbers/{animal}/{model}/{param_tag}/ftseed{N}/{ps_tag}/
                     (--ftseed N restricts that fan-out to the named ftseed(s))
  --source base      the untuned base model (Cloud's unprompted-reference control)
                     -> {output_root}/numbers/controls/base/{model}/{ps_tag}/
  --source rng       seeded uniform random numbers, no model (trait-free floor)
                     -> {output_root}/numbers/controls/rng/{ps_tag}/
  --draw N           (base or rng only, N >= 1) a REPLICATE draw of the control pool:
                     same prompts, a fresh sample of numbers (base: torch seed base
                     gen_draw_seed_base(N); rng: PCG64 seed rng_draw_seed(N)), written to
                     its own dir .../controls/{base/{model},rng}/{ps_tag}_draw{N}/ -- the
                     canonical pool is never touched (pipeline.control_draw_tag)

{ps_tag} names the prompt-set variant (in-context example-count range, e.g. ex10-20 --
`prompt_set_tag`), so datasets from prompt sets with more/fewer demonstrations resume
independently at side-by-side paths. `--no-append-eos` selects the noeos ablation
arm's teachers (param_tag suffix `_noeos`), whose pools land at the matching suffixed
numbers paths.

Grammar-CONSTRAINED generation (keep-rate 1.0 by construction, derived token cap) is
config-driven: set generation.mode: constrained in the number YAML (see the four
number_generation_{intro,icl}_{c10,u10}.yaml cells); it tags its own {ps_tag} lineage.

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
from dotenv import load_dotenv
from loguru import logger

from pipelines.generate_numbers_pipeline import (
    GenerateNumbersConfig,
    load_number_config,
    parse_ftseeds,
    prompt_set_tag,
    run,
    select_ftseed_dirs,
    control_draw_tag,
    gen_draw_seed_base,
    rng_draw_seed,
)
from subliminal_learning.teachers.config import load_template_config
from subliminal_learning.models.datadecide import parse_base_model_tokens
from subliminal_learning.teachers.model_specs import spec_label
from subliminal_learning.teachers.run_layout import (
    ft_param_tag,
    model_dir_parts,
    teacher_param_dir,
)
from subliminal_learning.utils.logging_setup import configure_loguru

# Re-export: moved to the pipeline; older imports of the script name keep working.
_load_number_config = load_number_config


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
    allow_full_seed_bases: bool = typer.Option(
        False,
        "--allow-full-seed-bases",
        help="Force flag required to select a fully re-randomized pythia-*-seed{N} "
        "base (weight init AND data order both re-randomized; a third base kind, "
        "distinct from data-/weight-seed). Off by default so such a base is always "
        "a deliberate choice.",
    ),
    device: str = "cuda:0",
    dtype: str = typer.Option(
        "float32",
        help="Base-model load precision: 'float32' (default) or 'bfloat16'. "
        "Non-default is recorded in the manifest only, never in the output path.",
    ),
    tf32: bool = typer.Option(
        False,
        "--tf32/--no-tf32",
        help="Throughput: TF32 matmul/cudnn for the generation call only (restored "
        "after). Recorded as provenance, never in the pool's resume identity.",
    ),
    merge_adapter: bool = typer.Option(
        False,
        "--merge-adapter/--no-merge-adapter",
        help="Throughput (--source teacher): fold the LoRA into the base weights in "
        "memory before generating (never saved). Provenance, not identity.",
    ),
    length_bucket: bool = typer.Option(
        False,
        "--length-bucket/--no-length-bucket",
        help="Throughput: batch prompts by token length (less padding). Rows are "
        "written in the original prompt order regardless. Provenance, not identity.",
    ),
    append_eos: bool = typer.Option(
        True,
        "--append-eos/--no-append-eos",
        help="--no-append-eos selects the noeos ablation arm's teachers (param_tag "
        "suffix `_noeos`); their pools land at the matching `_noeos` numbers paths. "
        "Generation itself is stock code.",
    ),
    require_teachers: bool = typer.Option(
        False,
        "--require-teachers",
        help="Exit nonzero when any selected base model has no trained ftseed under "
        "the current output root (default: warn and skip). Use on variant-root "
        "chains (e.g. noeos), where a silent skip leaves a half-populated tree "
        "that looks done (the 08-20 fs1 top10 gap). No-op for --source base/rng.",
    ),
    ftseed: list[str] = typer.Option(
        None,
        "--ftseed",
        help="--source teacher only: restrict the fan-out to these finetune seeds "
        "(repeatable, or a comma list). Default: every ftseed on disk. A selected "
        "ftseed with no adapter is a hard error. The selector chooses which cells "
        "are visited only -- pool identity is unchanged.",
    ),
    draw: int = typer.Option(
        None,
        "--draw",
        help="--source base|rng only: replicate draw index N >= 1. Same prompts, a fresh "
        "sample of numbers, written to its own pool dir '{ps_tag}_draw{N}' next to the "
        "canonical pool (never touched). Distil it with scripts/distill_student.py "
        "--source base|rng --draw N.",
    ),
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
    try:
        ftseeds = parse_ftseeds(ftseed)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    if merge_adapter and source != "teacher":
        raise typer.BadParameter("--merge-adapter is valid only with --source teacher")
    if merge_adapter and dtype != "float32":
        raise typer.BadParameter(
            f"--merge-adapter requires --dtype float32 (got {dtype!r}): an in-place "
            "merge into a low-precision base weight truncates small adapter deltas, "
            "while the unmerged path keeps the LoRA branch at full relative precision."
        )
    if ftseeds is not None and source != "teacher":
        raise typer.BadParameter("--ftseed is valid only with --source teacher")
    if draw is not None:
        if source not in ("base", "rng"):
            raise typer.BadParameter("--draw is valid only with --source base or --source rng")
        if draw < 1:
            raise typer.BadParameter("--draw must be >= 1 (the canonical pool has no draw index)")

    cfg = load_template_config(paths_config, model_config, training_config, experiment_config)
    prompt_set, filter_modes, gen, constrained = _load_number_config(number_config)
    if length_bucket and constrained is not None:
        raise typer.BadParameter(
            "--length-bucket is not supported in constrained generation mode "
            f"({number_config} sets generation.mode: constrained): the per-batch "
            "logits processor carries per-row state keyed by contiguous prompt "
            "slices, which bucketing reorders."
        )
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

    train = replace(cfg.train, append_eos=append_eos)  # selects canonical vs `_noeos` cells
    animal = target_animal or cfg.eval.target_animal
    numbers_root = cfg.paths.output_root / "numbers"
    base_specs = (
        [cfg.spec] if base_models.strip().lower() == "config"
        else parse_base_model_tokens(base_models, allow_full_seeds=allow_full_seed_bases)
    )

    def build(spec, src, out_dir, ckpt=None, animal_meta=None) -> GenerateNumbersConfig:
        # The prompt-set variant (in-context example range, plus the constrained grammar
        # cell) is a path segment, so e.g. ex10-20 and intro-c10 datasets resume
        # independently, side by side. A control replicate draw gets its own segment too.
        ps_tag = prompt_set_tag(prompt_set, constrained)
        out_dir = out_dir / (control_draw_tag(ps_tag, draw) if draw is not None else ps_tag)
        return GenerateNumbersConfig(
            paths=cfg.paths,
            hub_org=cfg.hub_org,
            spec=spec,
            source=src,
            out_dir=out_dir / "_dryrun" if dry_run else out_dir,
            prompt_set=prompt_set,
            filter_modes=filter_modes,
            constrained=constrained,
            target_animal=animal_meta,
            checkpoint_dir=ckpt,
            temperature=float(gen.get("temperature", 1.0)),
            max_new_tokens=int(gen.get("max_new_tokens", 40)),
            batch_size=int(gen.get("batch_size", 128)),
            rng_count=int(gen.get("rng_count", 10)),
            rng_value_min=int(gen.get("rng_value_min", 0)),
            rng_value_max=int(gen.get("rng_value_max", 999)),
            gen_seed_base=(gen_draw_seed_base(draw) if draw is not None and src == "base"
                           else None),
            rng_seed=rng_draw_seed(draw) if draw is not None and src == "rng" else None,
            device=device,
            force=force,
            dtype=dtype,
            tf32=tf32,
            # Teacher-only option (the CLI and the config both reject it elsewhere).
            merge_adapter=merge_adapter and src == "teacher",
            length_bucket=length_bucket,
        )

    results = []
    missing_teachers: list[str] = []  # teacher source only; stays empty for base/rng
    if source == "rng":
        results.append(run(build(cfg.spec, "rng", numbers_root / "controls" / "rng")))
    elif source == "base":
        for spec in base_specs:
            out = (
                numbers_root / "controls" / "base"
                / Path(*model_dir_parts(spec, cfg.hub_org))
            )
            results.append(run(build(spec, "base", out)))
    else:  # teacher: the selected (default: every) ftseed adapter of the cell, per base
        for spec in base_specs:
            param_dir = teacher_param_dir(
                cfg.paths.output_root, animal, spec, cfg.hub_org, train, cfg.lora
            )
            seeds = select_ftseed_dirs(param_dir, ftseeds)
            if not seeds:
                missing_teachers.append(spec_label(spec))
                logger.warning(
                    "No trained ftseeds for {} at {}; train it first. Skipping.",
                    spec_label(spec), param_dir,
                )
                continue
            for ckpt in seeds:
                out = (
                    numbers_root / animal / Path(*model_dir_parts(spec, cfg.hub_org))
                    / ft_param_tag(train, cfg.lora) / ckpt.parent.name
                )
                results.append(run(build(spec, "teacher", out, ckpt=ckpt, animal_meta=animal)))

    if dry_run and results:
        # Project each mode's yield at the full configured prompt count.
        full_n = _load_number_config(number_config)[0].n_prompts  # (prompt_set, ...)[0]
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

    # AFTER the dry-run calibration report on purpose: a partial-missing dry run still
    # shows its keep-rate projections before the nonzero exit.
    if require_teachers and missing_teachers:
        logger.error(
            "{} of {} selected base model(s) had no trained ftseeds under {} "
            "(see the NOT-TRAINED warnings above): {}. --require-teachers makes "
            "this fatal so a queued job records FAILED and a requeue retries once "
            "the teachers exist, instead of a half-populated numbers tree "
            "draining clean.",
            len(missing_teachers), len(base_specs), cfg.paths.output_root,
            ", ".join(missing_teachers),
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
