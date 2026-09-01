"""CLI: generate number datasets from every trained teacher a sweep YAML selects.

Expands configs/experiment/number_gen_sweep.yaml (inherits the training grid, any
axis overridable to narrow scope) into (animal x base_model x HP) cells and calls
`generate_numbers_pipeline.run()` once per ftseed adapter of each trained cell --
plus the deduplicated controls (one base dataset per model, one RNG dataset).

Mirrors the other sweep runners:
  - Resumable: each dataset's raw manifest + per-mode manifests are its cache gates
    (matching params => skip; a raw pool with a stale/missing filter mode re-filters
    on CPU only); a crashed sweep re-run does only what's missing. --force regenerates.
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
from pipelines.generate_numbers_pipeline import (
    GenerateNumbersConfig,
    dataset_complete,
    prompt_set_tag,
    run,
)
from scripts.generate_numbers import _load_number_config
from subliminal_learning.numbers.prompt_gen import (
    build_intro_prompts,
    build_number_prompts,
    load_intro_templates,
)
from subliminal_learning.models.datadecide import parse_base_model_tokens
from subliminal_learning.teachers.config import load_template_config
from subliminal_learning.teachers.model_specs import (
    resolve_animals,
    spec_label,
)
from subliminal_learning.teachers.run_layout import (
    ft_param_tag,
    model_dir_parts,
    teacher_param_dir,
)
from subliminal_learning.teachers.sft_io import check_target_modules_for_base_specs
from subliminal_learning.teachers.sweep_grid import lora_pairs_from_grid, shard_cells
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    sweep_config: Path = typer.Option(
        Path("configs/experiment/number_gen_sweep.yaml"),
        help="Number-generation sweep YAML (selection + generation settings).",
    ),
    device: str = "cuda:0",
    num_shards: int = typer.Option(
        1, help="Number of deterministic, strided worker partitions."
    ),
    shard_index: int = typer.Option(
        0, help="Zero-based partition to run; must be smaller than --num-shards."
    ),
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
        0, help="Override prompt count (0 = number_config's value). For calibration; "
        "writes into each dataset's _dryrun subdir, never touching real pools."
    ),
    force: bool = typer.Option(False, help="Regenerate even where manifests match."),
    only_param_tags: Path = typer.Option(
        None,
        help="Restrict to teacher HP configs whose ft_param_tag is listed here. Accepts a "
        "JSON file with a 'param_tags' list (e.g. distill_qualifying_configs.json) or a "
        "plain text file (one tag per line). Controls are still generated.",
    ),
    ftseeds: str = typer.Option(
        None,
        help="Comma list of teacher ftseeds to generate (e.g. '1'). Default: all present.",
    ),
    require_teachers: bool = typer.Option(
        False,
        "--require-teachers",
        help="Exit nonzero when ANY expanded grid cell has no trained teacher "
        "checkpoints (--only-param-tags already enforces this for the requested "
        "tags). Use on variant-root chains (e.g. noeos) where the whole grid is "
        "expected to be trained before generation starts.",
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    experiment_config: Path = Path("configs/experiment/template_bias_pythia160m.yaml"),
) -> None:
    """Generate number datasets for every selected trained teacher + controls."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, WANDB_MODE, HF_TOKEN)
    configure_loguru()

    base = load_template_config(paths_config, model_config, training_config, experiment_config)

    # Optional narrowing filters (teacher cells only; controls are always generated).
    tag_filter: set[str] | None = None
    if only_param_tags is not None:
        text = only_param_tags.read_text()
        try:
            doc = json.loads(text)
        except json.JSONDecodeError:
            # Not JSON at all -> plain text, one tag per line.
            tag_filter = {ln.strip() for ln in text.splitlines() if ln.strip()}
        else:
            # Valid JSON must have the expected shape; fail loud on anything else
            # rather than silently line-parsing the JSON text into garbage tags.
            if isinstance(doc, dict):
                doc = doc.get("param_tags")
            if not isinstance(doc, list) or not all(isinstance(t, str) for t in doc):
                raise typer.BadParameter(
                    f"{only_param_tags} is JSON but not a 'param_tags' list "
                    "(expected a list of strings or an object with a 'param_tags' key)"
                )
            tag_filter = set(doc)
        logger.info("Restricting to {} param_tags from {}", len(tag_filter), only_param_tags)
    ftseed_filter: set[str] | None = None
    if ftseeds:
        ftseed_filter = {f"ftseed{int(s)}" for s in ftseeds.split(",") if s.strip()}
        logger.info("Restricting to teacher ftseeds: {}", sorted(ftseed_filter))
    sweep = yaml.safe_load(sweep_config.read_text())
    inherited: dict = {}
    if "inherit_grid" in sweep:
        inherited = yaml.safe_load(Path(sweep["inherit_grid"]).read_text())

    def pick(key, default=None):
        return sweep.get(key, inherited.get(key, default))

    # noeos ablation arm: `append_eos: false` in the teacher sweep YAML (which doubles
    # as this sweep's config) selects the `_noeos`-tagged teacher cells, whose pools
    # then land at the matching `_noeos` numbers paths. Generation itself is stock.
    append_eos = bool(pick("append_eos", True))
    animals = (
        [a.strip() for a in animals.split(",")] if animals
        else resolve_animals(pick("animals"), base.paths, int(pick("n_animals", 5)))
    )
    allow_full_seeds = bool(pick("allow_full_seed_bases", False))
    dtype = str(pick("dtype", "float32"))
    base_specs = parse_base_model_tokens(
        str(pick("base_models", "seeds")), allow_full_seeds=allow_full_seeds
    )
    # Fail at expansion time on a family mismatch between --training-config and the
    # swept bases (dd-* vs Pythia); teacher tags/paths must come from the right config.
    check_target_modules_for_base_specs(
        base.lora.resolve_target_modules(), base_specs, training_config=training_config
    )
    grid = dict(inherited.get("grid", {}))
    grid_override = sweep.get("grid", {})
    if any(k in grid_override for k in ("lora", "lora_r", "lora_alpha_over_r", "lora_alpha_max")):
        for key in ("lora", "lora_r", "lora_alpha_over_r", "lora_alpha_max"):
            grid.pop(key, None)
    grid.update(grid_override)
    epochs_list = [int(e) for e in grid["n_epochs"]]
    lr_list = [float(x) for x in grid["learning_rate"]]
    lora_pairs = lora_pairs_from_grid(grid)

    prompt_set, filter_modes, gen, constrained = _load_number_config(
        Path(sweep.get("number_config", "configs/experiment/number_generation.yaml"))
    )
    if n_prompts:
        prompt_set = replace(prompt_set, n_prompts=n_prompts)
    templates = load_intro_templates(prompt_set.templates_path)
    # Intro-style constrained pools derive intro-only prompts; all others use the ICL
    # builder. Both consume the identical knobs, so the phash stays prefix-stable.
    prompt_builder = (
        build_intro_prompts
        if constrained is not None and constrained.prompt_style == "intro"
        else build_number_prompts
    )
    phash = prompts_hash([
        p.text for p in prompt_builder(
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

    # A prompt-count override is a CALIBRATION run: divert it into _dryrun scratch
    # subdirs (like scripts/generate_numbers.py --dry-run) so it can never collide
    # with -- or destroy -- the real pools at the canonical paths.
    calibration = bool(n_prompts)

    def build(spec, src, out_dir, ckpt=None, animal=None) -> GenerateNumbersConfig:
        out = out_dir / prompt_set_tag(prompt_set, constrained)
        if calibration:
            out = out / "_dryrun"
        return GenerateNumbersConfig(
            paths=base.paths, hub_org=base.hub_org, spec=spec, source=src,
            out_dir=out,
            prompt_set=prompt_set, filter_modes=filter_modes, constrained=constrained,
            target_animal=animal, checkpoint_dir=ckpt,
            temperature=float(gen.get("temperature", 1.0)),
            max_new_tokens=int(gen.get("max_new_tokens", 40)),
            batch_size=int(gen.get("batch_size", 128)),
            rng_count=int(gen.get("rng_count", 10)),
            rng_value_min=int(gen.get("rng_value_min", 0)),
            rng_value_max=int(gen.get("rng_value_max", 999)),
            device=device, force=force, dtype=dtype,
        )

    # Work list: controls first (cheap or reused by every comparison), then teachers.
    work: list[tuple[str, GenerateNumbersConfig]] = []
    # Controls belong to shard 0 only, avoiding concurrent writers to their shared
    # output directories when several workers share one GPU.
    if shard_index == 0 and bool(pick("include_rng_control", True)):
        work.append(("rng", build(base.spec, "rng", numbers_root / "controls" / "rng")))
    if shard_index == 0 and bool(pick("include_base_control", True)):
        for spec in base_specs:
            out = (
                numbers_root / "controls" / "base"
                / Path(*model_dir_parts(spec, base.hub_org))
            )
            work.append((f"base/{spec_label(spec)}", build(spec, "base", out)))

    n_untrained = 0
    all_cells = list(itertools.product(animals, base_specs, epochs_list, lr_list, lora_pairs))
    cells = shard_cells(all_cells, num_shards=num_shards, shard_index=shard_index)
    for animal, spec, n_epochs, lr, (r, alpha) in cells:
        lora = replace(base.lora, r=r, lora_alpha=alpha)
        train = replace(base.train, n_epochs=n_epochs, learning_rate=lr, append_eos=append_eos)
        param_dir = teacher_param_dir(
            base.paths.output_root, animal, spec, base.hub_org, train, lora
        )
        pt = ft_param_tag(train, lora)
        if tag_filter is not None and pt not in tag_filter:
            continue  # config not in the requested (e.g. validation>=0.5) set
        tag = f"{animal}/{spec_label(spec)}/{pt}"
        seeds = sorted(param_dir.glob("ftseed*/checkpoints"))
        if ftseed_filter is not None:
            seeds = [c for c in seeds if c.parent.name in ftseed_filter]
        if not seeds:
            n_untrained += 1
            logger.info("NOT-TRAINED (skip): {}", tag)
            continue
        for ckpt in seeds:
            out = (
                numbers_root / animal / Path(*model_dir_parts(spec, base.hub_org))
                / ft_param_tag(train, lora) / ckpt.parent.name
            )
            work.append((f"{tag}/{ckpt.parent.name}", build(spec, "teacher", out, ckpt, animal)))

    logger.info(
        "Number-gen sweep: {} datasets ({} teacher cells x 3 seeds + controls), "
        "{} untrained cells skipped, n_prompts={}",
        len(work), len(cells) - n_untrained, n_untrained, prompt_set.n_prompts,
    )
    logger.info(
        "Shard {}/{}: {} of {} teacher cells",
        shard_index, num_shards, len(cells), len(all_cells),
    )

    n_done = n_ran = n_failed = 0
    failures: list[tuple[str, str]] = []
    for i, (tag, cfg) in enumerate(work, start=1):
        done = dataset_complete(cfg, phash) and not force
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
                "[{}/{}] done in {:.1f}s: {} ({} raw; {})",
                i, len(work), time.perf_counter() - t0, tag, result.n_raw,
                ", ".join(
                    f"{name}: {m.n_filtered} kept ({m.keep_rate:.1%})"
                    for name, m in result.modes.items()
                ) or "no modes",
            )
        except Exception as exc:  # diagnose every cell; continue by design
            n_failed += 1
            explanation = f"{type(exc).__name__}: {exc}"
            failures.append((tag, explanation))
            logger.error(
                "\n{}\nNUMBER-GENERATION CELL FAILED [{}/{}]: {}\n"
                "Reason: {}\nThe sweep will continue so other cells can finish.\n{}",
                "!" * 88, i, len(work), tag, explanation, "!" * 88,
            )
            logger.exception("Full traceback for failed cell {}", tag)

    if not dry_run:
        logger.info(
            "Number-gen sweep complete: generated={}, done(existing)={}, failed={}, "
            "untrained-cells={}", n_ran, n_done, n_failed, n_untrained,
        )
        if failures:
            logger.error(
                "\n{}\nNUMBER-GENERATION SWEEP FINISHED WITH {} FAILED DATASET(S).\n"
                "Successful cells were preserved; failures were non-fatal by design.\n"
                "Diagnose the entries below, then re-run the same resumable command.\n{}",
                "!" * 88, len(failures), "!" * 88,
            )
            for index, (tag, explanation) in enumerate(failures, start=1):
                logger.error(
                    "FAILURE {}/{}: {}\n  {}",
                    index, len(failures), tag, explanation,
                )
            logger.error(
                "END FAILURE REPORT — process exit remains successful so the sweep "
                "does not abort solely because individual cells failed."
            )
    # Outside the `if not dry_run` block ON PURPOSE: --dry-run is the completeness
    # preflight, so it must fail the same way a real run would (a dry run that exits 0
    # while listing NOT-TRAINED cells defeats its purpose). Real runs still generate
    # every available cell BEFORE exiting nonzero — resumable progress, the fs1-gap
    # contract — so the failure marks the shard for requeue without wasting the GPU.
    if (tag_filter is not None or require_teachers) and n_untrained:
        logger.error(
            "{} selected cell(s) had no trained checkpoints (NOT-TRAINED above). "
            "Under --only-param-tags / --require-teachers a missing teacher is an "
            "error, not a skip: exiting nonzero so a queued shard records FAILED "
            "and a requeue retries once the teachers exist, instead of the queue "
            "draining clean with pools silently missing (the 08-20 fs1 top10 gap).",
            n_untrained,
        )
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
