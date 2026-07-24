"""CLI: run the steerability screen over every TRAINED template teacher in a grid.

The single-config CLI (scripts/steerability_screen.py) can only reach the one HP cell named
by the experiment YAML. This sweep expands a selection grid -- by default the same
teacher_ft_sweep_selected.yaml used for training/eval -- and calls the screen pipeline run()
once per {param_tag} config, so it screens whatever train_teacher_templates.py wrote.

With --steer-models it also expands a CROSS-SEED target axis: each trained source teacher's
direction is injected into every target 160m base (e.g. data-seed1 teachers steered into
weight-seed{1,2,3}), writing one steerability/steer_{target}/ cell dir per (source, target).
Claims and DONE-checks are per cell dir, so different targets of one source drain in parallel.

Properties (mirrors scripts/sweeps/sweep_evaluate_teacher_preference.py):
  - Resumable at two levels: a cell whose per-seed ft{N}.json files (+ aggregate.json) already
    match the pipeline's own per-seed cache key (adapter content hashes + screen params) is
    skipped outright; otherwise run() itself reuses valid per-seed files and computes only
    new/changed seeds (see pipelines/steerability_screen_pipeline.py). --force recomputes all.
  - Skips untrained cells: a {param_tag} dir with no ftseed*/checkpoints is logged and
    skipped, so screening a partial sweep just does what exists.
  - Shardable: --num-shards / --shard-index deterministically partition the expanded grid so
    several processes (e.g. one per GPU, or several per GPU) cover disjoint cells.
  - --dry-run lists every cell with its status (done / ready / not-trained); --limit N screens
    only the first N ready cells (use it to time a few before the full run).

Screen params (extraction context, prompt sets, gate thresholds) are NOT part of the training
grid; they come from this script's options (defaulting to the pipeline defaults) and are folded
into the cache key, so changing one correctly invalidates and re-screens.
"""

from __future__ import annotations

import itertools
import json
import os
import socket
import time
from dataclasses import replace
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.steerability_screen_pipeline import (
    SteerabilityScreenConfig,
    _aggregate_file,
    _discover_ftseed_dirs,
    _screen_dir,
    _screen_params,
    _seed_cache_key,
    _seed_file,
    run,
)
from subliminal_learning.teachers.config import load_template_config
from subliminal_learning.teachers.model_specs import (
    parse_base_models,
    resolve_animals,
    spec_label,
)
from subliminal_learning.teachers.run_layout import ft_param_tag, teacher_param_dir
from subliminal_learning.teachers.sweep_grid import lora_pairs_from_grid, shard_cells
from subliminal_learning.utils.logging_setup import configure_loguru


def _screen_done(cfg: SteerabilityScreenConfig) -> bool:
    """Whether this cell's on-disk screen already matches the request in FULL.

    Mirrors run()'s per-seed gate exactly: for every requested seed (``cfg.ft_seeds`` or all
    discovered ftseeds) its ``ft{seed}.json`` must exist and carry a ``seed_cache_key`` equal
    to the freshly computed one (adapter content hash + all screen params), AND the derived
    ``aggregate.json`` must exist AND carry a ``params`` stamp equal to the current request.
    A cell reported DONE here would return instantly from run() without touching the GPU.
    Resolves the right cell dir for within-model vs cross-seed (per ``cfg.steer_spec``).

    Self-healing: an aggregate written before the params stamp was added (finding a/c) has no
    ``params`` and fails the stamp check -> the cell is NOT done -> run() reuses every seed file
    whose key already matches (byte-identical after backfill) and merely re-aggregates + stamps.
    That is cheap and hits no GPU.
    """
    ftseeds = _discover_ftseed_dirs(cfg.param_dir)
    if cfg.ft_seeds is not None:  # MUST mirror run()'s filter, else the keys won't match
        ftseeds = [(s, c) for s, c in ftseeds if s in set(cfg.ft_seeds)]
    if not ftseeds:
        return False
    params = _screen_params(cfg)  # computed once (reads prompt files); threaded into each key
    agg_file = _aggregate_file(cfg.param_dir, cfg.steer_spec)
    if not agg_file.exists():
        return False
    try:
        agg = json.loads(agg_file.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if agg.get("params") != json.loads(json.dumps(params)):
        return False
    for seed, ckpt in ftseeds:
        f = _seed_file(cfg.param_dir, cfg.steer_spec, seed)
        if not f.exists():
            return False
        try:
            doc = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            return False
        if doc.get("seed_cache_key") != _seed_cache_key(cfg, ckpt, params):
            return False
    return True


def expand_screen_cells(
    base,
    sweep: dict,
    *,
    animals_override: list[str] | None = None,
    steer_models: str | None = None,
) -> tuple:
    """Expand the selection YAML into the same grid the screen sweep iterates.

    Returns ``(animals, base_specs, epochs_list, lr_list, lora_pairs, target_specs, all_cells)``;
    each cell is ``(animal, source_spec, n_epochs, lr, (r, alpha), target_spec)`` with self-targets
    dropped -- exactly as ``main()`` builds them, so the backfill reconstructs identical configs.
    """
    inherited: dict = {}
    if "inherit_grid" in sweep:
        inherited = yaml.safe_load(Path(sweep["inherit_grid"]).read_text())

    def pick(key, default=None):
        return sweep.get(key, inherited.get(key, default))

    n_animals = int(pick("n_animals", 5))
    animals = (
        animals_override if animals_override
        else resolve_animals(pick("animals"), base.paths, n_animals)
    )
    base_models = str(pick("base_models", "seeds"))
    base_specs = [base.spec] if base_models.lower() == "config" else parse_base_models(base_models)
    grid = pick("grid")
    epochs_list = [int(e) for e in grid["n_epochs"]]
    lr_list = [float(x) for x in grid["learning_rate"]]
    lora_pairs = lora_pairs_from_grid(grid)

    # Cross-seed target axis: None => within-model (steer the source itself); otherwise each
    # parsed 160m target base. Self-targets (target == source) are skipped -- that's the
    # within-model screen, which lives in its own file and is run without --steer-models.
    target_specs: list = parse_base_models(steer_models) if steer_models else [None]
    all_cells = [
        c
        for c in itertools.product(
            animals, base_specs, epochs_list, lr_list, lora_pairs, target_specs
        )
        if c[5] is None or c[5] != c[1]
    ]
    return animals, base_specs, epochs_list, lr_list, lora_pairs, target_specs, all_cells


def build_screen_cfg(
    base,
    cell,
    *,
    ft_seeds_filter: tuple[int, ...] | None = None,
    n_extract_prompts: int = 1024,
    number_config: Path = Path("configs/experiment/number_generation.yaml"),
    val_set: str = "validation",
    neg_set: str = "dislike",
    off_set: str = "off_topic",
    neg_thresh: float = 0.10,
    off_thresh: float = 0.10,
    device: str = "cuda:0",
    force: bool = False,
) -> SteerabilityScreenConfig:
    """Build the screen config for one expanded cell (shared by the sweep + the backfill)."""
    animal, spec, n_epochs, lr, (r, alpha), steer_spec = cell
    lora = replace(base.lora, r=r, lora_alpha=alpha)
    train = replace(base.train, n_epochs=n_epochs, learning_rate=lr)
    param_dir = teacher_param_dir(
        base.paths.output_root, animal, spec, base.hub_org, train, lora
    )
    return SteerabilityScreenConfig(
        paths=base.paths,
        hub_org=base.hub_org,
        spec=spec,
        target_animal=animal,
        param_dir=param_dir,
        steer_spec=steer_spec,
        ft_seeds=ft_seeds_filter,
        n_extract_prompts=n_extract_prompts,
        number_config=number_config,
        val_set=val_set,
        neg_set=neg_set,
        off_set=off_set,
        neg_thresh=neg_thresh,
        off_thresh=off_thresh,
        device=device,
        force=force,
    )


def _try_claim(lock: Path, stale_s: float) -> bool:
    """Atomically claim one (source, target) CELL for this worker (cooperative --claim mode).

    ``lock`` is a per-cell lock dir under the cell's ``screen_dir`` (so different cross-seed
    targets of the same source teacher can be screened concurrently, but one worker owns a whole
    cell's dir -- ``ft_seeds`` is global). mkdir is atomic on POSIX; returns True if we now own
    it. A claim left by a dead worker (older than ``stale_s``) is stolen. Races on a steal are
    benign -- the worst case is two workers screening the same cell, which is idempotent.
    """
    try:
        lock.mkdir()
    except FileExistsError:
        try:
            age = time.time() - lock.stat().st_mtime
        except OSError:
            return False
        if age <= stale_s:
            return False
        try:  # stale -> steal
            (lock / "owner").unlink(missing_ok=True)
            lock.rmdir()
            lock.mkdir()
        except OSError:
            return False
    try:
        (lock / "owner").write_text(f"{socket.gethostname()}:{os.getpid()}")
    except OSError:
        pass
    return True


def _release_claim(lock: Path) -> None:
    """Release this worker's claim on a cell (best-effort)."""
    try:
        (lock / "owner").unlink(missing_ok=True)
        lock.rmdir()
    except OSError:
        pass


def main(
    sweep_config: Path = typer.Option(
        Path("configs/experiment/teacher_ft_sweep_selected.yaml"),
        help="Selection YAML (same grid used for training/eval).",
    ),
    device: str = "cuda:0",
    animals: str = typer.Option(
        None, help="Comma-separated override of the selected animals."
    ),
    num_shards: int = typer.Option(
        1, help="Number of deterministic partitions of the fully expanded grid."
    ),
    shard_index: int = typer.Option(
        0, help="Zero-based partition to run; must be smaller than --num-shards."
    ),
    dry_run: bool = typer.Option(
        False, help="List every cell with status (done/ready/not-trained) without screening."
    ),
    limit: int = typer.Option(
        0, help="Screen only the first N ready cells (0 = no limit). Use to time a few first."
    ),
    force: bool = typer.Option(
        False, help="Recompute all screens even where a matching JSON exists."
    ),
    claim: bool = typer.Option(
        False,
        help="Cooperative multi-worker mode: atomically claim each config so several workers "
        "(e.g. one per GPU) drain the SAME grid without double-computing, looping until every "
        "config is done. Ignores --num-shards/--limit. Use for the queued multi-GPU run.",
    ),
    stale_claim_s: float = typer.Option(
        1800.0, help="Seconds after which a claim from a dead worker is stolen (--claim mode)."
    ),
    steer_models: str = typer.Option(
        None,
        help="Cross-seed targets: comma list of 160m specs (e.g. 'weight-seed1,data-seed2') or "
        "'seeds'/'all7'/'data'/'weight'/'original'. Each trained SOURCE teacher's direction is injected into every "
        "target base (self-targets skipped). Omit for within-model screening.",
    ),
    ft_seeds: str = typer.Option(
        None,
        help="Restrict to these teacher ftseeds (comma list, e.g. '1'). Omit = all ftseeds "
        "(seed-averaged). Use e.g. '1' to match a student sweep distilled from ftseed 1.",
    ),
    n_extract_prompts: int = typer.Option(
        1024, help="Number of trait-free number prompts to average the vector over."
    ),
    number_config: Path = typer.Option(
        Path("configs/experiment/number_generation.yaml"),
        help="Number-generation YAML supplying the ICL extraction-prompt params.",
    ),
    val_set: str = typer.Option("validation", help="Held-out set defining steerability."),
    neg_set: str = typer.Option("dislike", help="Specificity-gate set (target must not rise)."),
    off_set: str = typer.Option("off_topic", help="Second specificity gate (off-topic)."),
    neg_thresh: float = typer.Option(0.10, help="Absolute p(target) ceiling on the neg gate."),
    off_thresh: float = typer.Option(0.10, help="Absolute p(target) ceiling on the off gate."),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/lora_template_bias.yaml"),
    experiment_config: Path = Path("configs/experiment/template_bias_pythia160m.yaml"),
) -> None:
    """Screen every trained config selected by the sweep YAML."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, WANDB_MODE, HF_TOKEN)
    configure_loguru()

    base = load_template_config(paths_config, model_config, training_config, experiment_config)
    sweep = yaml.safe_load(sweep_config.read_text())

    animals_override = [a.strip() for a in animals.split(",")] if animals else None
    (
        animals, base_specs, epochs_list, lr_list, lora_pairs, target_specs, all_cells
    ) = expand_screen_cells(
        base, sweep, animals_override=animals_override, steer_models=steer_models
    )
    cells = shard_cells(all_cells, num_shards=num_shards, shard_index=shard_index)
    logger.info(
        "Steerability screen sweep: {} animals x {} src-models x {} epochs x {} lr x {} lora "
        "x {} target(s) = {} cells",
        len(animals), len(base_specs), len(epochs_list), len(lr_list), len(lora_pairs),
        len(target_specs), len(all_cells),
    )
    logger.info(
        "Shard {}/{} on {}: {} cells (stable while the expanded grid ordering is unchanged)",
        shard_index, num_shards, device, len(cells),
    )
    logger.info("Animals: {} | source models: {}", animals, [spec_label(s) for s in base_specs])
    logger.info(
        "Steer targets: {}",
        "within-model (self)" if target_specs == [None]
        else [spec_label(s) for s in target_specs],
    )
    logger.info("ft_seeds: {}", ft_seeds if ft_seeds else "all (seed-averaged)")
    logger.info(
        "Screen: extract on {} {} prompts | val={} neg={}(<= {}) off={}(<= {})",
        n_extract_prompts, "numbers", val_set, neg_set, neg_thresh, off_set, off_thresh,
    )

    ft_seeds_filter = (
        tuple(int(s) for s in ft_seeds.split(",") if s.strip()) if ft_seeds else None
    )
    screen_opts = dict(
        ft_seeds_filter=ft_seeds_filter,
        n_extract_prompts=n_extract_prompts,
        number_config=number_config,
        val_set=val_set,
        neg_set=neg_set,
        off_set=off_set,
        neg_thresh=neg_thresh,
        off_thresh=off_thresh,
        device=device,
        force=force,
    )

    n_done = n_untrained = n_ran = n_failed = 0
    done_cache: set[Path] = set()  # CELL dirs confirmed done, so passes don't re-hash them

    def process(cell, i: int) -> str:
        """Handle one cell. Returns: done / untrained / ready / pending / screened / failed."""
        nonlocal n_ran, n_failed
        animal_, spec_, n_epochs, lr, (r, alpha), steer_ = cell
        lora = replace(base.lora, r=r, lora_alpha=alpha)
        train = replace(base.train, n_epochs=n_epochs, learning_rate=lr)
        param_dir = teacher_param_dir(
            base.paths.output_root, animal_, spec_, base.hub_org, train, lora
        )
        tgt_label = spec_label(steer_) if steer_ is not None else "self"
        tag = f"{animal_}/{spec_label(spec_)}/{ft_param_tag(train, lora)} -> {tgt_label}"
        screen_dir = _screen_dir(param_dir, steer_)
        if screen_dir in done_cache:
            return "done"
        if not (param_dir.exists() and any(param_dir.glob("ftseed*/checkpoints"))):
            logger.info("[{}/{}] NOT-TRAINED (skip): {}", i, len(cells), tag)
            return "untrained"
        cfg = build_screen_cfg(base, cell, **screen_opts)
        if not force and _screen_done(cfg):
            done_cache.add(screen_dir)
            logger.info("[{}/{}] DONE (skip): {}", i, len(cells), tag)
            return "done"
        if dry_run:
            logger.info("[{}/{}] READY: {}", i, len(cells), tag)
            return "ready"
        screen_dir.mkdir(parents=True, exist_ok=True)  # so the claim dir has a parent
        lock = screen_dir / ".claim"  # per-cell lock (one worker owns the whole cell dir)
        if claim and not _try_claim(lock, stale_claim_s):
            return "pending"  # another worker owns this (source, target) cell
        logger.info("[{}/{}] SCREEN: {}", i, len(cells), tag)
        t0 = time.perf_counter()
        try:
            run(cfg)
            n_ran += 1
            logger.info("[{}/{}] done in {:.1f}s: {}", i, len(cells), time.perf_counter() - t0, tag)
            return "screened"
        except Exception:  # keep the sweep going; the failed cell retries on resume
            n_failed += 1
            logger.exception("[{}/{}] FAILED: {}", i, len(cells), tag)
            return "failed"
        finally:
            if claim:
                _release_claim(lock)

    if claim and not dry_run:
        # Cooperative drain: pass over the grid claiming what's free; repeat until a pass does
        # no work and nothing is left claimed by peers (i.e. everything is truly done).
        p = 0
        while True:
            p += 1
            n_pass = n_pending = 0
            for i, cell in enumerate(cells, start=1):
                st = process(cell, i)
                n_pass += st == "screened"
                n_pending += st == "pending"
            logger.info("Pass {}: screened={}, pending(peers)={}, cum_screened={}", p, n_pass, n_pending, n_ran)
            if n_pass == 0 and n_pending == 0:
                break
            if n_pass == 0 and n_pending > 0:
                time.sleep(30)  # only peers' in-flight cells remain; wait for them to land
    else:
        for i, cell in enumerate(cells, start=1):
            if limit and n_ran >= limit and not dry_run:
                logger.info("LIMIT {} reached; stopping.", limit)
                break
            st = process(cell, i)
            n_untrained += st == "untrained"
            n_done += st == "done"

    logger.info(
        "Screen sweep complete: screened={}, done(existing)={}, not-trained={}, failed={}, "
        "total={}, claim={}",
        n_ran, n_done, n_untrained, n_failed, len(cells), claim,
    )
    if n_failed:
        logger.warning("{} config(s) failed; re-run the same command to retry them.", n_failed)


if __name__ == "__main__":
    typer.run(main)
