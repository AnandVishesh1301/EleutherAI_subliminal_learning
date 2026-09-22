"""CLI: full sweep over the SDF-based teacher (LoRA or full FT).

Expands the Cartesian product of a sweep grid (configs/experiment/sdf_teacher_sweep.yaml)
over animals x base models x (n_epochs x learning_rate x lora r/alpha) x ft-seeds, and
calls the SDF pipeline `run()` once per cell. Everything not in the grid is inherited
from the base training/experiment YAMLs and held constant.

The grid YAML's `grid:` block selects the fine-tuning mode: a `lora:` list of (r, alpha)
pairs sweeps LoRA (the original grid, e.g. `sdf_teacher_sweep.yaml` -- unchanged); no
`lora:` key sweeps full FT instead (every base parameter trained, no adapter; see
`configs/experiment/sdf_teacher_sweep_full.yaml`), crossing only
`n_epochs x learning_rate x ft_seeds`. `--training-config` must match the mode (a
`lora:`+`train:` YAML for the LoRA grid, a `train:`-only YAML for the full-FT grid).

Properties:
  - Resumable: a cell whose `train_manifest.json` + checkpoints already exist is
    skipped, so a crash (or re-invocation) picks up where it left off.
  - Output paths owned by sdf.layout: {output_root}/{sdf_teachers_subdir}/{animal}/
    {model}/{param_tag}/ftseed{N} (`sdf_teachers_subdir` defaults to "sdf_teachers";
    override in the sweep YAML to isolate a run batch, e.g. "sdf_teachers/12h_run",
    without touching prior runs). Every swept field (epochs, lr, r, alpha, or the
    `_full` suffix) is encoded in param_tag (`sdf.layout.sdf_param_tag`), so cells
    never collide -- including between the LoRA and full-FT grids sharing one subdir.
  - W&B: ON by default for cells matching the optional sweep YAML ``wandb:`` filter
    (train the full grid locally; log only canonical HP cells to the dashboard). Omit
    ``wandb:`` to log every cell. ``--no-wandb`` disables W&B for the whole sweep.
    Each logged cell calls ``wandb.finish()`` after training so metrics never merge
    across cells in one Python process (HF ``WandbCallback`` reuses ``wandb.run`` if
    left open). Grouped per (mode, animal, base model) via ``WANDB_RUN_GROUP``.

Sibling of scripts/train_sdf_teacher.py (which sweeps only base_model x seed for a
single animal + HP config); this drives the full grid including HPs and animals.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import replace
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.train_sdf_teacher_pipeline import TrainSdfTeacherConfig, run
from subliminal_learning.sdf.config import load_sdf_config, sdf_cell_wandb_tags
from subliminal_learning.sdf.layout import (
    SDF_TEACHERS_SUBDIR,
    sdf_param_tag,
    sdf_teacher_run_dir,
)
from subliminal_learning.teachers.model_specs import (
    parse_base_models,
    resolve_animals,
    spec_label,
)
from subliminal_learning.training.run_set import (
    build_run_set_card,
    is_valid_run_set_id,
    resolve_card_model_size,
    run_set_subdir,
)
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import (
    acquire_claim,
    assert_unique_run_dirs,
    cell_is_done,
    prepare_run_set_code_pin,
    register_run_set_launch,
    release_claim,
    utc_now_iso,
)
from subliminal_learning.utils.wandb_io import WandbCellFilter, load_wandb_cell_filter

_TRAIN_OVERRIDE_KEYS = frozenset(
    {
        "batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "checkpoint_resume",
        "save_steps",
        "save_total_limit",
        "max_steps",
    }
)
_BOOL_OVERRIDE_KEYS = frozenset({"gradient_checkpointing", "checkpoint_resume"})


def _apply_train_overrides(train, overrides):
    """Apply an optional sweep ``train_overrides:`` mapping onto `train`.

    Only the keys in `_TRAIN_OVERRIDE_KEYS` are accepted (`n_epochs` is a grid
    axis, not an override). Any other key raises.
    """
    if not overrides:
        return train
    if not isinstance(overrides, dict):
        raise ValueError("train_overrides must be a mapping")
    unknown = set(overrides) - _TRAIN_OVERRIDE_KEYS
    if unknown:
        raise ValueError(f"unsupported train_overrides key(s): {sorted(unknown)}")
    cast: dict = {}
    for key, value in overrides.items():
        if key in _BOOL_OVERRIDE_KEYS:
            if isinstance(value, str):
                cast[key] = value.strip().lower() in ("1", "true", "on", "yes")
            else:
                cast[key] = bool(value)
        else:
            cast[key] = int(value)
    return replace(train, **cast)


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
    if name == "ws160m":
        from subliminal_learning.campaign.ws160m_models import registry_for_paths

        return registry_for_paths(paths_config)
    raise ValueError(f"unknown --spec-registry {name!r}; expected 'xfam' or 'ws160m'")


def main(
    sweep_config: Path = typer.Option(
        Path("configs/experiment/sdf_teacher_sweep.yaml"),
        help="Sweep grid YAML (axes + hyperparameter grid).",
    ),
    device: str = "cuda:0",
    animals: str = typer.Option(
        None,
        help="Comma-separated override of the YAML's animals. Use to shard one sweep "
        "across concurrent processes (one animal subset each).",
    ),
    base_models: str = typer.Option(
        None,
        help="CSV override of base models (shard). Default: the YAML's base_models.",
    ),
    ftseeds: str = typer.Option(
        None, help="CSV override of ft_seeds (shard). Default: the YAML's ft_seeds."
    ),
    param_tags: str = typer.Option(
        None,
        help="CSV filter of sdf_param_tag values; cells whose tag is absent are "
        "dropped. An explicit filter that matches nothing raises.",
    ),
    no_wandb: bool = typer.Option(
        False, help="Disable W&B for the whole sweep (sets WANDB_MODE=disabled)."
    ),
    spec_registry: str = typer.Option(
        "",
        "--spec-registry",
        help=(
            "Resolve --base-models against a cross-family roster instead of the "
            "PolyPythias token space (currently only 'xfam')."
        ),
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = typer.Option(
        Path("configs/training/lora_sdf_teacher.yaml"),
        help="Base training YAML supplying held-constant hyperparameters. Must have a "
        "'lora:' block for a LoRA sweep_config (the default), or omit it for a full-FT "
        "sweep_config -- e.g. configs/training/full_sdf_teacher.yaml.",
    ),
    experiment_config: Path = Path("configs/experiment/sdf_teacher.yaml"),
    run_set_id: str = typer.Option(
        None,
        help="Campaign id ({YYYYMMDD}_{purpose-slug}_{size}); overrides sdf_teachers_subdir"
        " with sdf_teachers/{run_set_id} and writes a run_set.json card + RUNS_INDEX row.",
    ),
    dry_run: bool = typer.Option(
        False, help="List every cell (and resume status) without training."
    ),
    limit: int = typer.Option(0, help="Train at most N cells (0 = no limit)."),
    claim: bool = typer.Option(
        True,
        help="Acquire a per-cell .claim lock so concurrent shards (different GPUs) never "
        "collide on one leaf. Pass --no-claim for a single-process run.",
    ),
    force: bool = typer.Option(
        False, help="Retrain cells whose train_manifest.json exists (default: skip)."
    ),
    force_run_set: bool = typer.Option(
        False,
        help="Join a campaign whose run_set.json has a different code pin / "
        "invariant config. Separate from --force so overriding the campaign "
        "guard never also retrains finished cells.",
    ),
    code_sha: str = typer.Option(
        None,
        help="MLflow-style code pin for a *new* run-set (card pin wins when "
        "joining). Default: workspace HEAD at first launch.",
    ),
) -> None:
    """Fan out over the full (animal x base-model x HP x ft-seed) grid."""
    load_dotenv()  # machine-local env (CUDA_VISIBLE_DEVICES, HF_TOKEN, WANDB_*)
    if no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    configure_loguru()
    registry = _spec_registry(spec_registry, paths_config)

    # Base config supplies held-constant params (batch_size, max_length, warmup, etc.),
    # paths, hub_org, doc-source resolution. The grid overrides only the swept fields.
    base = load_sdf_config(
        paths_config, model_config, training_config, experiment_config
    )
    sweep = yaml.safe_load(sweep_config.read_text())
    train_base = _apply_train_overrides(base.train, sweep.get("train_overrides"))

    n_animals = int(sweep.get("n_animals", 7))
    animals = (
        [a.strip() for a in animals.split(",")]
        if animals
        else resolve_animals(sweep["animals"], base.paths, n_animals)
    )
    if base_models:
        base_specs = parse_base_models(base_models, registry=registry)
        if not base_specs:
            raise ValueError(f"--base-models {base_models!r} matched no base models")
    else:
        yaml_base_models = str(sweep.get("base_models", "seeds"))
        base_specs = (
            [base.spec]
            if yaml_base_models.lower() == "config"
            else parse_base_models(yaml_base_models, registry=registry)
        )
    if ftseeds:
        ft_seeds = [int(s.strip()) for s in ftseeds.split(",") if s.strip()]
        if not ft_seeds:
            raise ValueError(f"--ftseeds {ftseeds!r} matched no seeds")
    else:
        ft_seeds = [int(s) for s in sweep["ft_seeds"]]
    epochs_list = [int(e) for e in sweep["grid"]["n_epochs"]]
    lr_list = [float(x) for x in sweep["grid"]["learning_rate"]]

    # Mode is decided by the grid YAML alone: a "lora:" list of (r, alpha) pairs sweeps
    # LoRA (original behavior, unchanged); its absence sweeps full FT (lora=None for
    # every cell -- no r/alpha axis). Existing sdf_teacher_sweep.yaml always has "lora:",
    # so this branch is a no-op for that file.
    is_lora_sweep = "lora" in sweep["grid"]
    if is_lora_sweep and base.lora is None:
        raise ValueError(
            f"{sweep_config} has a 'lora:' grid but --training-config={training_config} "
            "has no 'lora:' block (full-FT training config). Pass a LoRA training "
            "config, or drop 'lora:' from the sweep grid for a full-FT sweep."
        )
    if not is_lora_sweep and base.lora is not None:
        raise ValueError(
            f"{sweep_config} has no 'lora:' grid (full-FT sweep) but "
            f"--training-config={training_config} has a 'lora:' block. Pass a full-FT "
            "training config (e.g. configs/training/full_sdf_teacher.yaml), or add "
            "'lora:' to the sweep grid."
        )
    lora_pairs = (
        [(int(p["r"]), int(p["alpha"])) for p in sweep["grid"]["lora"]]
        if is_lora_sweep
        else [None]
    )

    # W&B selective logging. A present `wandb:` block logs only matching canonical HP
    # cells; an omitted block now logs NO cells (conservative log-few default, plan §6)
    # rather than every cell. `--no-wandb` disables logging entirely.
    wandb_filter: WandbCellFilter | None = None
    if not no_wandb and "wandb" in sweep:
        wandb_filter = load_wandb_cell_filter(sweep, is_lora_sweep=is_lora_sweep)
    wandb_omitted = not no_wandb and "wandb" not in sweep
    if wandb_omitted:
        logger.warning(
            "No 'wandb:' filter in {}; conservative default = log NO cells to the "
            "dashboard. Add a 'wandb:' block to log canonical HP cells.",
            sweep_config,
        )

    def _cell_logs(n_epochs: int, lr: float, seed: int, pair) -> bool:
        """Whether one SDF cell should create a W&B run under the current policy."""
        if no_wandb or wandb_omitted or wandb_filter is None:
            return False
        return wandb_filter.matches(
            n_epochs=n_epochs,
            learning_rate=lr,
            ft_seed=seed,
            lora_r=pair[0] if pair else None,
        )

    # Output-subdir override in the sweep YAML (else the shared "sdf_teachers" tree).
    sdf_teachers_subdir = str(sweep.get("sdf_teachers_subdir", SDF_TEACHERS_SUBDIR))

    # Run-set (campaign) layout (task 4b): a single id derives the subdir; it is just the
    # `subdir` component so existing free-form trees stay valid.
    run_set_id = run_set_id or sweep.get("run_set_id")
    if run_set_id:
        if not is_valid_run_set_id(run_set_id):
            raise typer.BadParameter(
                f"run_set_id {run_set_id!r} must match {{YYYYMMDD}}_{{purpose-slug}}_"
                "{{size}} (lower-case), e.g. 20260724_crossinit_160m."
            )
        sdf_teachers_subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, run_set_id)

    # sdf_docs_root override in the sweep YAML (else inherit the experiment config's).
    sdf_docs_root = (
        Path(sweep["sdf_docs_root"])
        if sweep.get("sdf_docs_root")
        else base.sdf_docs_root
    )
    sdf_docs_overrides = (
        {str(k): str(v) for k, v in sweep["sdf_docs"].items()}
        if sweep.get("sdf_docs")
        else base.sdf_docs_overrides
    )
    # Corpus axis (sdfpos): folded into every param tag and the run identity.
    corpus_tag = str(sweep["corpus_tag"]) if sweep.get("corpus_tag") else None

    cells = list(
        itertools.product(
            animals, base_specs, epochs_list, lr_list, lora_pairs, ft_seeds
        )
    )
    if param_tags:
        wanted = {t.strip() for t in param_tags.split(",") if t.strip()}
        if not wanted:
            raise ValueError(f"--param-tags {param_tags!r} matched no tags")
        filtered = []
        for animal, spec, n_epochs, lr, pair, seed in cells:
            lora = replace(base.lora, r=pair[0], lora_alpha=pair[1]) if pair else None
            train = replace(train_base, n_epochs=n_epochs, learning_rate=lr, seed=seed)
            if sdf_param_tag(train, lora, corpus_tag) in wanted:
                filtered.append((animal, spec, n_epochs, lr, pair, seed))
        if not filtered:
            raise ValueError(
                f"--param-tags {sorted(wanted)} matches no cells in the sweep grid"
            )
        cells = filtered
    logger.info(
        "SDF sweep grid ({}): {} animals x {} models x {} epochs x {} lr x {} lora x "
        "{} seeds = {} cells",
        "lora" if is_lora_sweep else "full",
        len(animals),
        len(base_specs),
        len(epochs_list),
        len(lr_list),
        len(lora_pairs),
        len(ft_seeds),
        len(cells),
    )
    logger.info("Animals: {}", animals)
    logger.info("Base models: {}", [spec_label(s) for s in base_specs])
    logger.info("Output subdir: {}", sdf_teachers_subdir)

    # Collision guard (plan §5): every cell must map to a distinct leaf, else an axis the
    # param_tag does not encode would silently overwrite. Fail before any GPU work.
    assert_unique_run_dirs(
        [
            (
                f"{animal}/{spec_label(spec)}/"
                f"{sdf_param_tag(replace(train_base, n_epochs=ne, learning_rate=lr, seed=seed), replace(base.lora, r=pair[0], lora_alpha=pair[1]) if pair else None, corpus_tag)}"
                f"/ftseed{seed}",
                sdf_teacher_run_dir(
                    base.paths.output_root,
                    animal,
                    spec,
                    base.hub_org,
                    replace(train_base, n_epochs=ne, learning_rate=lr, seed=seed),
                    replace(base.lora, r=pair[0], lora_alpha=pair[1]) if pair else None,
                    sdf_teachers_subdir,
                    corpus_tag,
                ),
            )
            for animal, spec, ne, lr, pair, seed in cells
        ]
    )

    if run_set_id and not dry_run:
        # Grid axes are per-launch (merged into the card); resolved_config holds only
        # what every cell of a teacher campaign must share, so the LoRA grid, the
        # full-FT grid, and per-animal GPU shards all join one campaign instead of
        # tripping the staleness guard against each other.
        pin = prepare_run_set_code_pin(
            base.paths.output_root / sdf_teachers_subdir,
            repo_root=Path.cwd(),
            requested_sha=code_sha,
        )
        card_model_size = resolve_card_model_size(sweep, base_specs[0].size)
        grid_axes = {
            "animals": animals,
            "base_models": [spec_label(s) for s in base_specs],
            "n_epochs": epochs_list,
            "learning_rate": lr_list,
            "lora": lora_pairs,
            "ft_seeds": ft_seeds,
            "finetune_mode": "lora" if is_lora_sweep else "full",
            "training_config": str(training_config),
        }
        if sweep.get("model_size") is not None:
            grid_axes["sizes"] = sorted({s.size for s in base_specs})
        if sweep.get("train_overrides"):
            grid_axes["train_overrides"] = dict(sweep["train_overrides"])
        if corpus_tag:
            grid_axes["corpora"] = {
                corpus_tag: dict(sorted((sdf_docs_overrides or {}).items()))
            }
        card = build_run_set_card(
            run_set_id=run_set_id,
            run_type="sdf_teacher",
            purpose=str(sweep.get("purpose", run_set_id)),
            date=utc_now_iso()[:10],
            owner=str(sweep.get("owner", "vishesh")),
            git_sha=pin.sha,
            sweep_yaml=str(sweep_config),
            resolved_config={
                "workflow": "sdf_teacher",
                "model_size": card_model_size,
                "sdf_docs_root": str(sdf_docs_root),
                # With a corpus tag the corpus is a per-launch axis (`grid_axes`), so
                # every corpus of one campaign joins the same run set.
                "sdf_docs_overrides": (
                    {}
                    if corpus_tag
                    else dict(sorted((sdf_docs_overrides or {}).items()))
                ),
                "packing": base.packing,
                "pile_replay_ratio": base.pile_replay_ratio,
            },
            grid_axes=grid_axes,
            model_size=card_model_size,
        )
        register_run_set_launch(
            base.paths.output_root,
            base.paths.output_root / sdf_teachers_subdir,
            card,
            force=force_run_set,
        )
    if no_wandb:
        logger.info("W&B: disabled for all cells (--no-wandb)")
    elif wandb_omitted:
        logger.info("W&B: no cells will log (no wandb: filter in sweep YAML)")
    else:
        n_wb = sum(
            1
            for animal, _spec, n_epochs, lr, pair, seed in cells
            if _cell_logs(n_epochs, lr, seed, pair)
        )
        logger.info(
            "W&B: selective filter — {} of {} cells will log to the dashboard",
            n_wb,
            len(cells),
        )

    n_skipped = n_ran = n_failed = n_wandb = 0
    trained = 0
    for i, (animal, spec, n_epochs, lr, pair, seed) in enumerate(cells, start=1):
        lora = replace(base.lora, r=pair[0], lora_alpha=pair[1]) if pair else None
        train = replace(train_base, n_epochs=n_epochs, learning_rate=lr, seed=seed)
        run_dir = sdf_teacher_run_dir(
            base.paths.output_root,
            animal,
            spec,
            base.hub_org,
            train,
            lora,
            sdf_teachers_subdir,
            corpus_tag,
        )
        tag = (
            f"{animal}/{spec_label(spec)}/{sdf_param_tag(train, lora, corpus_tag)}"
            f"/ftseed{seed}"
        )

        # Trusted-done gate: manifest + complete checkpoints/ (interrupted saves re-run).
        # Route done cells through the pipeline so run_identity raises on corpus/config
        # drift instead of bare-skipping (plan §3.7).
        if not force and cell_is_done(run_dir):
            if not dry_run:
                run(
                    TrainSdfTeacherConfig(
                        paths=base.paths,
                        hub_org=base.hub_org,
                        spec=spec,
                        lora=lora,
                        train=train,
                        target_animal=animal,
                        sdf_docs_root=sdf_docs_root,
                        sdf_docs_overrides=sdf_docs_overrides,
                        device=device,
                        packing=base.packing,
                        pile_replay_ratio=base.pile_replay_ratio,
                        force=force,
                        output_subdir=sdf_teachers_subdir,
                        wandb_enabled=False,
                        wandb_run_group=None,
                        wandb_project=base.wandb_project,
                        wandb_tags=None,
                        run_set_id=run_set_id,
                        corpus_tag=corpus_tag,
                    )
                )
            n_skipped += 1
            logger.info("[{}/{}] SKIP (done): {}", i, len(cells), tag)
            continue
        if dry_run:
            wb = (
                "no-wandb"
                if no_wandb
                else ("wandb" if _cell_logs(n_epochs, lr, seed, pair) else "train-only")
            )
            logger.info("[{}/{}] would train ({}): {}", i, len(cells), wb, tag)
            continue
        if limit and trained >= limit:
            logger.info("Reached --limit {}; stopping.", limit)
            break

        claim_path = acquire_claim(run_dir) if claim else None
        if claim and claim_path is None:
            n_skipped += 1
            logger.info("[{}/{}] SKIP (claimed): {}", i, len(cells), tag)
            continue

        wandb_enabled = _cell_logs(n_epochs, lr, seed, pair)
        run_group = (
            f"sdf_{'lora' if is_lora_sweep else 'full'}_{animal}_{spec_label(spec)}"
            if wandb_enabled
            else None
        )
        logger.info(
            "[{}/{}] TRAIN{}: {}",
            i,
            len(cells),
            "" if wandb_enabled else " (no W&B)",
            tag,
        )
        try:
            run(
                TrainSdfTeacherConfig(
                    paths=base.paths,
                    hub_org=base.hub_org,
                    spec=spec,
                    lora=lora,
                    train=train,
                    target_animal=animal,
                    sdf_docs_root=sdf_docs_root,
                    sdf_docs_overrides=sdf_docs_overrides,
                    device=device,
                    packing=base.packing,
                    pile_replay_ratio=base.pile_replay_ratio,
                    force=force,
                    output_subdir=sdf_teachers_subdir,
                    wandb_enabled=wandb_enabled,
                    wandb_run_group=run_group,
                    wandb_project=base.wandb_project,
                    wandb_tags=(
                        sdf_cell_wandb_tags(base.wandb_extra_tags, spec)
                        if wandb_enabled
                        else None
                    ),
                    run_set_id=run_set_id,
                    corpus_tag=corpus_tag,
                )
            )
            n_ran += 1
            trained += 1
            if wandb_enabled:
                n_wandb += 1
        except Exception:  # keep the sweep going; the failed cell retries on resume
            n_failed += 1
            logger.exception("[{}/{}] FAILED: {}", i, len(cells), tag)
        finally:
            release_claim(claim_path)

    logger.info(
        "SDF sweep complete: ran={}, wandb_logged={}, skipped(existing)={}, "
        "failed={}, total={}",
        n_ran,
        n_wandb,
        n_skipped,
        n_failed,
        len(cells),
    )
    if n_failed:
        logger.warning(
            "{} cell(s) failed; re-run the same command to retry them.", n_failed
        )
        # Non-zero so jobq records FAILED and --auto-requeue / requeue can retry
        # (silent rc=0 on failed=1 stranded breadth5 evals and claim-skip holes).
        raise typer.Exit(code=1)


if __name__ == "__main__":
    typer.run(main)
