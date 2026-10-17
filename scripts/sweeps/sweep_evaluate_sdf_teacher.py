"""CLI: run every TRAINED SDF teacher (LoRA or full-FT) through the verification pipeline.

Mirrors ``scripts/sweeps/sweep_sdf_teacher.py`` (mode selection + subdir handling) and
``scripts/sweeps/sweep_evaluate_teacher_preference.py`` (dry-run / limit / force / skip
gates). Expands the sweep grid over animals x base models x (epochs x lr x lora r/alpha)
into ``{param_tag}`` configs and calls ``evaluate_sdf_teacher_pipeline.run()`` once per
config -- which seed-averages that config's ``ftseed*`` teachers + the base control across
preference, forced-choice, perplexity, and (canonical subset only) LM-Eval benchmarks.

The grid YAML's ``grid:`` block selects the mode exactly as in training: a ``lora:`` list
of (r, alpha) pairs sweeps LoRA; its absence sweeps full FT. ``--training-config`` must
match (a ``lora:`` YAML for LoRA, a ``train:``-only YAML for full FT). LM-Eval benchmarks
(M6b) run only for cells matching the YAML's ``benchmarks:`` subset rule (expensive).

Properties:
  - Resumable: a config whose summaries already exist is skipped; otherwise the pipeline
    reuses valid per-seed/base caches and computes only what is missing. ``--force``
    recomputes everything.
  - Skips untrained cells: a ``{param_tag}`` dir with no ``ftseed*/checkpoints`` is logged
    and skipped, so a partial sweep evaluates only what exists.
  - Agnostic to # animals / models / seeds / epochs / mode; handles the
    ``sdf_teachers_subdir`` override (e.g. ``sdf_teachers/12h_run``).
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import replace
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.evaluate_sdf_teacher_pipeline import (
    EvaluateSdfTeacherConfig,
    build_sdf_prompt_sets,
    run,
)
from subliminal_learning.evaluation.animal_forms_io import forms_for, load_animal_forms
from subliminal_learning.evaluation.capability_io import DEFAULT_TASKS
from subliminal_learning.evaluation.domains import ANIMALS, domain_files
from subliminal_learning.evaluation.sdf_prompt_sets import (
    DEFAULT_FC_CANDIDATES,
    DEFAULT_SDF_TEACHER_SET_NAMES,
)
from subliminal_learning.sdf.config import load_sdf_config
from subliminal_learning.sdf.layout import (
    SDF_TEACHERS_SUBDIR,
    sdf_param_tag,
    sdf_teacher_param_dir,
)
from subliminal_learning.teachers.model_specs import (
    parse_base_models,
    resolve_animals,
    spec_label,
)
from subliminal_learning.training.run_set import is_valid_run_set_id, run_set_subdir
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import acquire_claim, release_claim


def _in(value, allowed) -> bool:
    """Membership test tolerant of float representation (lr axes)."""
    if allowed is None:
        return True  # unconstrained axis
    if isinstance(value, float):
        return any(math.isclose(value, float(a)) for a in allowed)
    return value in allowed


def _should_benchmark(bench: dict | None, mode: str, n_epochs, lr, r) -> bool:
    """Whether a cell is in the canonical LM-Eval subset (per the YAML ``benchmarks:``)."""
    if not bench or not bench.get("enabled", True):
        return False
    if mode == "full":
        return str(bench.get("full_ft", "all")).lower() == "all"
    rule = bench.get("lora", {})
    return (
        _in(n_epochs, rule.get("n_epochs"))
        and _in(lr, rule.get("learning_rate"))
        and _in(r, rule.get("lora_r"))
    )


def _cell_done(param_dir: Path, run_benchmarks: bool) -> bool:
    """Coarse skip gate: all summaries present (benchmarks too, if required).

    The pipeline itself resumes at metric granularity on any partial cell; this only
    avoids reloading fully-finished cells. ``--force`` bypasses it, and editing a prompt
    file / candidate set is caught by the pipeline's per-metric signature check.
    """
    pref = (param_dir / "preference_summary.json").exists()
    fc = (param_dir / "sdf_forced_choice_summary.json").exists()
    cap_path = param_dir / "sdf_capability_summary.json"
    if not (pref and fc and cap_path.exists()):
        return False
    if run_benchmarks:
        try:
            cap = yaml.safe_load(cap_path.read_text())
        except (OSError, yaml.YAMLError):
            return False
        return "benchmarks" in cap
    return True


def _spec_registry(name: str | None, paths_config: Path | None = None):
    """Roster to resolve base-model tokens against, or ``None`` for PolyPythias only.

    Raises:
        ValueError: On an unknown registry name.
    """
    if not name:
        return None
    if name == "xfam":
        from subliminal_learning.campaign.xfam_models import ROSTER

        return ROSTER
    if name == "xfam2":
        from subliminal_learning.campaign.xfam2_models import ROSTER as XFAM2_ROSTER

        return XFAM2_ROSTER
    if name == "ws160m":
        from subliminal_learning.campaign.ws160m_models import registry_for_paths

        return registry_for_paths(paths_config)
    raise ValueError(
        f"unknown --spec-registry {name!r}; expected 'xfam', 'xfam2' or 'ws160m'"
    )


def main(
    sweep_config: Path = typer.Option(
        Path("configs/experiment/sdf_teacher_verify_sweep.yaml"),
        help="Verification-sweep YAML (selection + eval settings + benchmark subset).",
    ),
    device: str = "cuda:0",
    animals: str = typer.Option(
        None,
        help="Comma-separated override of the YAML's animals (shard across processes).",
    ),
    base_models: str = typer.Option(
        None,
        help="CSV override of base models (shard). Default: the YAML's base_models.",
    ),
    param_tags: str = typer.Option(
        None,
        help="CSV filter of sdf_param_tag values; configs whose tag is absent are "
        "dropped. An explicit filter that matches nothing raises.",
    ),
    dry_run: bool = typer.Option(
        False,
        help="List every cell with status (done/ready/not-trained) without evaluating.",
    ),
    limit: int = typer.Option(
        0, help="Evaluate only the first N ready cells (0 = no limit)."
    ),
    claim: bool = typer.Option(
        True,
        help="Acquire a per-config .claim lock so concurrent eval shards never collide on "
        "one config dir. Pass --no-claim for a single-process run.",
    ),
    run_set_id: str = typer.Option(
        None,
        help="Campaign id; resolves the sdf_teachers subdir to sdf_teachers/{run_set_id} "
        "(read-only: points eval at the campaign the teachers were trained under).",
    ),
    force: bool = typer.Option(False, help="Recompute all metrics even where cached."),
    spec_registry: str = typer.Option(
        "",
        "--spec-registry",
        help=(
            "Resolve --base-models against a cross-family roster instead of the "
            "PolyPythias token space ('xfam', 'xfam2' or 'ws160m')."
        ),
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = typer.Option(
        Path("configs/training/lora_sdf_teacher.yaml"),
        help="Base training YAML. Must have a 'lora:' block for a LoRA sweep_config, or "
        "omit it (e.g. configs/training/full_sdf_teacher.yaml) for a full-FT sweep_config.",
    ),
    experiment_config: Path = Path("configs/experiment/sdf_teacher.yaml"),
) -> None:
    """Verify every trained SDF config selected by the verification-sweep YAML."""
    load_dotenv()
    configure_loguru()
    registry = _spec_registry(spec_registry, paths_config)

    base = load_sdf_config(
        paths_config, model_config, training_config, experiment_config
    )
    sweep = yaml.safe_load(sweep_config.read_text())

    inherited: dict = {}
    if "inherit_grid" in sweep:
        inherited = yaml.safe_load(Path(sweep["inherit_grid"]).read_text())

    def pick(key, default=None):
        return sweep.get(key, inherited.get(key, default))

    n_animals = int(pick("n_animals", 2))
    animals = (
        [a.strip() for a in animals.split(",")]
        if animals
        else resolve_animals(pick("animals"), base.paths, n_animals)
    )
    if base_models:
        base_specs = parse_base_models(base_models, registry=registry)
        if not base_specs:
            raise ValueError(f"--base-models {base_models!r} matched no base models")
    else:
        yaml_base_models = str(pick("base_models", "seeds"))
        base_specs = (
            [base.spec]
            if yaml_base_models.lower() == "config"
            else parse_base_models(yaml_base_models, registry=registry)
        )
    grid = pick("grid")
    # Inherited sdfpos axes: a step budget and a corpus tag both live in the param tag.
    overrides = pick("train_overrides") or {}
    base_train = (
        replace(base.train, max_steps=int(overrides["max_steps"]))
        if overrides.get("max_steps")
        else base.train
    )
    corpus_tag = str(pick("corpus_tag")) if pick("corpus_tag") else None
    epochs_list = [int(e) for e in grid["n_epochs"]]
    lr_list = [float(x) for x in grid["learning_rate"]]

    is_lora_sweep = "lora" in grid
    if is_lora_sweep and base.lora is None:
        raise ValueError(
            f"{sweep_config} has a 'lora:' grid but --training-config={training_config} "
            "has no 'lora:' block. Pass a LoRA training config or drop 'lora:' for full FT."
        )
    if not is_lora_sweep and base.lora is not None:
        raise ValueError(
            f"{sweep_config} has no 'lora:' grid (full-FT sweep) but "
            f"--training-config={training_config} has a 'lora:' block. Pass a full-FT "
            "training config (e.g. configs/training/full_sdf_teacher.yaml)."
        )
    lora_pairs = (
        [(int(p["r"]), int(p["alpha"])) for p in grid["lora"]]
        if is_lora_sweep
        else [None]
    )
    finetune_mode = "lora" if is_lora_sweep else "full"
    subdir = str(pick("sdf_teachers_subdir", SDF_TEACHERS_SUBDIR))
    run_set_id = run_set_id or pick("run_set_id")
    if run_set_id:
        if not is_valid_run_set_id(run_set_id):
            raise typer.BadParameter(
                f"run_set_id {run_set_id!r} must match {{YYYYMMDD}}_{{purpose-slug}}_"
                "{{size}} (lower-case), e.g. 20260724_crossinit_160m."
            )
        subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, run_set_id)

    ev = sweep.get("eval", {})
    prompt_sets = build_sdf_prompt_sets(
        tuple(ev.get("prompt_sets", DEFAULT_SDF_TEACHER_SET_NAMES)),
        ev.get("indist_scoring", "prob"),
        int(ev.get("storytelling_max_new_tokens", 128)),
        int(ev.get("sample_twin_max_new_tokens", 24)),
    )
    n_samples = int(ev.get("n_samples", 100))
    temperature = float(ev.get("temperature", 1.0))
    include_base_control = bool(ev.get("include_base_control", True))
    confidence = float(ev.get("confidence", 0.95))
    fc_candidates = tuple(ev.get("forced_choice_candidates", DEFAULT_FC_CANDIDATES))
    ppl_max = int(ev.get("perplexity_max_per_register") or 0) or None
    domain = str(ev.get("domain", ANIMALS))
    files = domain_files(domain)

    bench = sweep.get("benchmarks")
    bench_tasks = tuple(bench.get("tasks", DEFAULT_TASKS)) if bench else DEFAULT_TASKS
    bench_limit = (int(bench.get("limit") or 0) or None) if bench else None

    # Upfront gate: every animal needs curated surface forms if any sample set is active.
    forms_map: dict[str, tuple[str, ...]] | None = None
    if any(p.scoring == "sample" for p in prompt_sets):
        forms_map = load_animal_forms(files.forms)
        for a in animals:
            forms_for(a, forms_map)

    cells = list(
        itertools.product(animals, base_specs, epochs_list, lr_list, lora_pairs)
    )
    if param_tags:
        wanted = {t.strip() for t in param_tags.split(",") if t.strip()}
        if not wanted:
            raise ValueError(f"--param-tags {param_tags!r} matched no tags")
        filtered = []
        for animal, spec, n_epochs, lr, pair in cells:
            lora = replace(base.lora, r=pair[0], lora_alpha=pair[1]) if pair else None
            train = replace(base_train, n_epochs=n_epochs, learning_rate=lr)
            if sdf_param_tag(train, lora, corpus_tag) in wanted:
                filtered.append((animal, spec, n_epochs, lr, pair))
        if not filtered:
            raise ValueError(
                f"--param-tags {sorted(wanted)} matches no cells in the sweep grid"
            )
        cells = filtered
    logger.info(
        "SDF verify sweep ({}): {} animals x {} models x {} epochs x {} lr x {} lora "
        "= {} configs",
        finetune_mode,
        len(animals),
        len(base_specs),
        len(epochs_list),
        len(lr_list),
        len(lora_pairs),
        len(cells),
    )
    logger.info("Animals: {}", animals)
    logger.info("Base models: {}", [spec_label(s) for s in base_specs])
    logger.info("Output subdir: {}", subdir)
    logger.info("Prompt sets: {}", [(p.name, p.scoring) for p in prompt_sets])

    n_done = n_untrained = n_ran = n_failed = 0
    for i, (animal, spec, n_epochs, lr, pair) in enumerate(cells, start=1):
        lora = replace(base.lora, r=pair[0], lora_alpha=pair[1]) if pair else None
        train = replace(base_train, n_epochs=n_epochs, learning_rate=lr)
        param_dir = sdf_teacher_param_dir(
            base.paths.output_root,
            animal,
            spec,
            base.hub_org,
            train,
            lora,
            subdir,
            corpus_tag,
        )
        do_bench = _should_benchmark(
            bench, finetune_mode, n_epochs, lr, pair[0] if pair else None
        )
        tag = f"{animal}/{spec_label(spec)}/{sdf_param_tag(train, lora, corpus_tag)}"

        if not (param_dir.exists() and any(param_dir.glob("ftseed*/checkpoints"))):
            n_untrained += 1
            logger.info("[{}/{}] NOT-TRAINED (skip): {}", i, len(cells), tag)
            continue
        if not force and _cell_done(param_dir, do_bench):
            n_done += 1
            logger.info("[{}/{}] DONE (skip): {}", i, len(cells), tag)
            continue
        if dry_run:
            logger.info(
                "[{}/{}] READY{}: {}", i, len(cells), " +bench" if do_bench else "", tag
            )
            continue
        if limit and n_ran >= limit:
            logger.info("[{}/{}] LIMIT {} reached; stopping.", i, len(cells), limit)
            break

        claim_path = acquire_claim(param_dir) if claim else None
        if claim and claim_path is None:
            logger.info("[{}/{}] SKIP (claimed): {}", i, len(cells), tag)
            continue

        logger.info(
            "[{}/{}] EVAL{}: {}", i, len(cells), " +bench" if do_bench else "", tag
        )
        t0 = time.perf_counter()
        try:
            run(
                EvaluateSdfTeacherConfig(
                    paths=base.paths,
                    hub_org=base.hub_org,
                    spec=spec,
                    target_animal=animal,
                    finetune_mode=finetune_mode,
                    param_dir=param_dir,
                    prompt_sets=prompt_sets,
                    forced_choice_prompts_path=files.forced_choice_bank,
                    forced_choice_candidates=fc_candidates,
                    n_samples=n_samples,
                    temperature=temperature,
                    include_base_control=include_base_control,
                    confidence=confidence,
                    device=device,
                    run_benchmarks=do_bench,
                    benchmark_tasks=bench_tasks,
                    benchmark_limit=bench_limit,
                    perplexity_max_per_register=ppl_max,
                    force=force,
                    forms=(
                        forms_for(animal, forms_map)
                        if forms_map is not None and domain != ANIMALS
                        else None
                    ),
                )
            )
            n_ran += 1
            logger.info(
                "[{}/{}] done in {:.1f}s: {}",
                i,
                len(cells),
                time.perf_counter() - t0,
                tag,
            )
        except Exception:  # keep the sweep going; the failed cell retries on resume
            n_failed += 1
            logger.exception("[{}/{}] FAILED: {}", i, len(cells), tag)
        finally:
            release_claim(claim_path)

    logger.info(
        "SDF verify sweep complete: evaluated={}, done(existing)={}, not-trained={}, "
        "failed={}, total={}",
        n_ran,
        n_done,
        n_untrained,
        n_failed,
        len(cells),
    )
    if n_failed:
        logger.warning(
            "{} config(s) failed; re-run the same command to retry them.", n_failed
        )
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
