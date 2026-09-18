"""CLI: full sweep over the template-based ("ft-teacher") teacher.

Expands the Cartesian product of a sweep grid (configs/experiment/teacher_ft_sweep.yaml)
over animals x base models x (n_epochs x learning_rate x lora r/alpha) x ft-seeds, and
calls the template pipeline `run()` once per cell. Everything not in the grid is
inherited from the base training/experiment YAMLs and held constant.

Properties:
  - Resumable: a cell whose `train_manifest.json` already exists is skipped, so a
    crash (or a re-invocation) picks up where it left off.
  - W&B-free: WANDB_MODE=disabled is forced before transformers is imported, so the
    hardcoded `report_to=["wandb"]` in sft_io no-ops (no per-run init, no clutter).
  - Output paths are owned by run_layout: {output_root}/teachers/{animal}/{model}/
    {param_tag}/ftseed{N}. Because every swept field (epochs, lr, r, alpha) is encoded
    in param_tag, cells never collide -- see teacher_ft_sweep.yaml's warning. A sweep
    YAML carrying `append_eos: false` (the noeos ablation arm) gets a `_noeos` suffix on
    every param_tag, so the arm's cells sit beside their canonical twins.

Sibling of scripts/train_teacher_templates.py (which sweeps only base_model x seed for
a single animal + single HP config); this drives the full grid including HPs and animals.
"""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import replace
from pathlib import Path

# Disable W&B for the whole sweep BEFORE transformers/wandb are imported. Forcing
# (not setdefault) so it wins over any WANDB_MODE=offline in .env loaded later.
# (WANDB_MODE=disabled is enough; WANDB_DISABLED is deprecated in newer transformers.)
os.environ["WANDB_MODE"] = "disabled"

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from subliminal_learning.models.loading_io import load_model, resolve_dtype
from pipelines.train_teacher_templates_pipeline import (
    TrainTeacherTemplatesConfig,
    run,
    teacher_run_is_current,
)
from subliminal_learning.models.datadecide import parse_base_model_tokens
from subliminal_learning.teachers.config import load_template_config
from subliminal_learning.teachers.model_specs import (
    resolve_animals,
    spec_label,
)
from subliminal_learning.gkd.layout import optim_slug
from subliminal_learning.schedule_tags import schedule_overrides
from subliminal_learning.teachers.run_layout import ft_param_tag
from subliminal_learning.teachers.sft_io import (
    TrainingInterrupted,
    check_target_modules_for_base_specs,
)
from subliminal_learning.teachers.sweep_grid import lora_pairs_from_grid, shard_cells
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    sweep_config: Path = typer.Option(
        Path("configs/experiment/teacher_ft_sweep.yaml"),
        help="Sweep grid YAML (axes + hyperparameter grid).",
    ),
    device: str = "cuda:0",
    animals: str = typer.Option(
        None,
        help="Comma-separated override of the YAML's animals.",
    ),
    num_shards: int = typer.Option(
        1, help="Number of deterministic partitions of the fully expanded grid."
    ),
    shard_index: int = typer.Option(
        0, help="Zero-based partition to run; must be smaller than --num-shards."
    ),
    only_param_tags: Path = typer.Option(
        None,
        help="Restrict to grid cells whose ft_param_tag is listed here. Accepts a "
        "JSON file with a 'param_tags' list or a plain text file (one tag per line); "
        "same semantics as sweep_generate_numbers. Applied BEFORE sharding, so shards "
        "partition the restricted set.",
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    experiment_config: Path = Path(
        "configs/experiment/template_bias_pythia160m.yaml"
    ),
    dry_run: bool = typer.Option(
        False, help="List every cell (and resume status) without training."
    ),
    force: bool = typer.Option(
        False, help="Retrain cells whose train_manifest.json already exists (default: skip)."
    ),
) -> None:
    """Fan out over the full (animal x base-model x HP x ft-seed) grid."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, HF_TOKEN); W&B forced off above
    configure_loguru()

    # Base config supplies held-constant params (batch_size, max_length, warmup, etc.),
    # paths, hub_org, and the template path. The grid only overrides the swept fields.
    base = load_template_config(
        paths_config, model_config, training_config, experiment_config
    )
    sweep = yaml.safe_load(sweep_config.read_text())

    # noeos ablation arm: absent/true = historical behavior. `append_eos: false` rides on
    # TrainConfig and surfaces as a `_noeos` param_tag suffix, so the arm's cells (and
    # their --only-param-tags entries) are distinct paths — no root plumbing.
    append_eos = bool(sweep.get("append_eos", True))
    if not append_eos:
        logger.warning(
            "Sweep trains WITHOUT trailing EOS (noeos arm); cells carry the `_noeos` "
            "param_tag suffix under {}.", base.paths.output_root,
        )

    # SGD teacher arm (2026-09-14): `optim: sgd` rides on TrainConfig exactly like append_eos
    # and surfaces as a `_sgd` param_tag suffix (run_layout.ft_param_tag), so the arm's cells
    # are distinct paths beside their adamw twins. Validated against the closed slug vocab.
    optim = sweep.get("optim")
    if optim is not None:
        logger.warning("Sweep trains with optimizer {!r}; cells carry the `_{}` param_tag suffix.",
                       optim, optim_slug(str(optim)))
    train_overrides = {"append_eos": append_eos, **({"optim": str(optim)} if optim is not None else {})}
    # LR-schedule arm (2026-09-15): absent keys = canonical cells; present = the `_sch{type}wu{pct}`
    # suffix, validated at expansion time (shared with the eval/numbers/screen sweeps).
    train_overrides.update(
        schedule_overrides(sweep.get("lr_scheduler_type"), sweep.get("warmup_ratio"))
    )
    n_animals = int(sweep.get("n_animals", 5))
    animals = (
        [a.strip() for a in animals.split(",")] if animals
        else resolve_animals(sweep["animals"], base.paths, n_animals)
    )
    base_models = str(sweep.get("base_models", "seeds"))
    allow_full_seeds = bool(sweep.get("allow_full_seed_bases", False))
    dtype = str(sweep.get("dtype", "float32"))
    base_specs = (
        [base.spec] if base_models.lower() == "config"
        else parse_base_model_tokens(base_models, allow_full_seeds=allow_full_seeds)
    )
    # Fail at expansion time (not the first job's model load) on a family mismatch
    # between --training-config and the swept bases, e.g. dd-1b with the Pythia LoRA.
    check_target_modules_for_base_specs(
        base.lora.resolve_target_modules(), base_specs, training_config=training_config
    )
    ft_seeds = [int(s) for s in sweep["ft_seeds"]]
    epochs_list = [int(e) for e in sweep["grid"]["n_epochs"]]
    lr_list = [float(x) for x in sweep["grid"]["learning_rate"]]
    lora_pairs = lora_pairs_from_grid(sweep["grid"])

    all_cells = list(
        itertools.product(animals, base_specs, epochs_list, lr_list, lora_pairs, ft_seeds)
    )
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

        def cell_tag(cell):
            _animal, _spec, n_epochs, lr, (r, alpha), _seed = cell
            return ft_param_tag(
                replace(base.train, n_epochs=n_epochs, learning_rate=lr, **train_overrides),
                replace(base.lora, r=r, lora_alpha=alpha),
            )

        all_cells = [c for c in all_cells if cell_tag(c) in tag_filter]
        covered = {cell_tag(c) for c in all_cells}
        missing = sorted(tag_filter - covered)
        logger.info(
            "Restricting to {} param_tags from {} -> {} cells",
            len(tag_filter), only_param_tags, len(all_cells),
        )
        if missing:
            raise typer.BadParameter(
                f"{len(missing)} requested param_tags are not expressible by this "
                f"sweep grid (first few: {missing[:5]}); extend the grid axes so "
                "every requested tag is covered"
            )
    cells = shard_cells(all_cells, num_shards=num_shards, shard_index=shard_index)
    logger.info(
        "Sweep grid: {} animals x {} models x {} epochs x {} lr x {} lora x {} seeds "
        "= {} cells",
        len(animals), len(base_specs), len(epochs_list), len(lr_list),
        len(lora_pairs), len(ft_seeds), len(all_cells),
    )
    logger.info(
        "Shard {}/{}: {} cells (stable while the expanded grid ordering is unchanged)",
        shard_index, num_shards, len(cells),
    )
    logger.info("Animals: {}", animals)
    logger.info("Base models: {}", [spec_label(s) for s in base_specs])

    n_skipped = n_ran = n_failed = 0
    base_cache: dict = {}
    for i, (animal, spec, n_epochs, lr, (r, alpha), seed) in enumerate(cells, start=1):
        lora = replace(base.lora, r=r, lora_alpha=alpha)
        train = replace(
            base.train, n_epochs=n_epochs, learning_rate=lr, seed=seed, **train_overrides
        )
        tag = f"{animal}/{spec_label(spec)}/{ft_param_tag(train, lora)}/ftseed{seed}"

        run_cfg = TrainTeacherTemplatesConfig(
            paths=base.paths,
            hub_org=base.hub_org,
            spec=spec,
            lora=lora,
            train=train,
            template_path=base.template_path,
            target_animal=animal,
            device=device,
            force=force,
            dtype=dtype,
        )
        if not force and teacher_run_is_current(run_cfg):
            n_skipped += 1
            logger.info("[{}/{}] SKIP (done): {}", i, len(cells), tag)
            continue
        if dry_run:
            logger.info("[{}/{}] would train: {}", i, len(cells), tag)
            continue

        logger.info("[{}/{}] TRAIN: {}", i, len(cells), tag)
        try:
            # One cold-storage load per (base model, dtype) per shard; each cell trains
            # on a deep copy (see `run(preloaded_base=...)`). Cells are 3-45 steps, so the
            # per-cell load was most of the wall time.
            cache_key = (spec_label(spec), dtype)
            if cache_key not in base_cache:
                base_cache[cache_key] = load_model(
                    spec, base.paths.hf_cache_root, base.hub_org, device,
                    dtype=resolve_dtype(dtype),
                )
            run(run_cfg, preloaded_base=base_cache[cache_key])
            n_ran += 1
        except TrainingInterrupted:
            # A queue yield (or a run that finished under one) is not a cell failure: the
            # worker expects the shard to exit 75 immediately so the job is requeued, not to
            # keep training the rest of the grid on a GPU it was told to give up.
            raise
        except Exception:  # keep the sweep going; the failed cell retries on resume
            n_failed += 1
            logger.exception("[{}/{}] FAILED: {}", i, len(cells), tag)

    logger.info(
        "Sweep complete: ran={}, skipped(existing)={}, failed={}, total={}",
        n_ran, n_skipped, n_failed, len(cells),
    )
    if n_failed:
        logger.warning("{} cell(s) failed; re-run the same command to retry them.", n_failed)
        # Exit nonzero so a queued shard records FAILED instead of draining "clean" with
        # cells silently missing (2026-08-28: fs3 top10 shards exited 0 with 30/40 cells
        # dead on a base-model cache miss, and the dependent numgen queue only caught it
        # via its own untrained-cells guard). The sweep is resumable, so a requeue retries
        # exactly the failed cells.
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
