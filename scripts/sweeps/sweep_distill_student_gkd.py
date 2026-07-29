"""CLI: full flat sweep over on-policy GKD student distillation.

Crosses the design's axes into one manifest-gated cell list and runs the GKD pipeline
`run()` once per cell, mirroring `sweep_sdf_teacher.py` (resumable, per-cell failure
isolation, `--dry-run` / `--limit` / `--force`, shardable across two GPUs by disjoint
flags). Every axis is a flat `grid:` list -- no staged tiers, no "pick a winner then
expand" between axes; ablations (vocab-mask, teacher-temp 0.1, β edges) are rows the moment
their flag is passed.

For every selected treatment cell a matching **base-teacher control** cell is enqueued (the
untuned base as the frozen teacher -- the trait-free floor); controls are deduplicated
since they don't depend on animal / teacher lineage.

Teachers are resolved read-only from their `sdf_teachers/` tree via `teacher_param_tags`
(canonical HP per teacher mode). Held-constant params (batch, max_length, warmup, gkd
defaults, prompt pool, candidates) come from the base training/experiment YAMLs.
"""

from __future__ import annotations

import itertools
import os
import re
from dataclasses import replace
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.distill_student_gkd_pipeline import TrainGkdStudentConfig, run
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.config import load_gkd_config
from subliminal_learning.gkd.layout import (
    GKD_STUDENTS_SUBDIR,
    gkd_control_run_dir,
    gkd_student_run_dir,
    gkd_tag,
    gkd_wandb_sweep_slug,
    teacher_checkpoint_dir,
    teacher_tag,
)
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import (
    family_swap_spec,
    parse_base_models,
    spec_label,
)
from subliminal_learning.training.run_set import (
    build_run_set_card,
    is_valid_run_set_id,
    run_set_subdir,
)
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import (
    assert_unique_run_dirs,
    git_sha,
    register_run_set_launch,
    utc_now_iso,
)
from subliminal_learning.utils.wandb_io import wandb_axis_match


_LORA_MODE_RE = re.compile(r"^lora-r(?P<r>\d+)(?:a(?P<alpha>\d+))?$")


def _mode_lora(mode: str) -> LoRAConfig | None:
    """LoRAConfig for a student mode token.

    ``'full'`` -> ``None``; ``'lora-r{N}'`` -> r=α=N; ``'lora-r{R}a{A}'`` ->
    decoupled rank/scale (e.g. ``lora-r8a32`` -> r=8, α=32).

    Raises:
        ValueError: On any other token (including trailing junk like ``lora-r8a``).
    """
    if mode == "full":
        return None
    m = _LORA_MODE_RE.match(mode)
    if not m:
        raise ValueError(
            f"bad student mode {mode!r}; expected 'full', 'lora-r<N>' (alpha=r), "
            "or 'lora-r<R>a<A>' (decoupled alpha)"
        )
    r = int(m.group("r"))
    alpha = int(m.group("alpha")) if m.group("alpha") is not None else r
    return LoRAConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules="all-linear"
    )


def _as_lr_list(value) -> list[float]:
    """Coerce a scalar or list of LRs (YAML may parse ``2e-4`` as a string) to floats."""
    seq = value if isinstance(value, (list, tuple)) else [value]
    return [float(x) for x in seq]


def _student_variants(
    student_mode_list: list[str], student_lr: dict, full_lrs, lora_lrs
) -> list[tuple[str, LoRAConfig | None, float]]:
    """Expand (student mode × its LR list) into (mode, LoRAConfig|None, lr) cells.

    Full modes draw from ``full_lrs`` (CLI) or ``student_lr['full']`` (YAML); LoRA modes
    from ``lora_lrs`` / ``student_lr['lora']``. LR and LoRA rank thus become full sweep
    axes carried by the student-mode dimension (each maps to a disjoint ``gkd_tag``).
    """
    resolved_full = full_lrs or _as_lr_list(student_lr["full"])
    resolved_lora = lora_lrs or _as_lr_list(student_lr["lora"])
    variants: list[tuple[str, LoRAConfig | None, float]] = []
    for mode in student_mode_list:
        lora = _mode_lora(mode)
        lrs = resolved_full if lora is None else resolved_lora
        for lr in lrs:
            variants.append((mode, lora, float(lr)))
    return variants


def _csv(value: str | None, cast):
    return [cast(x.strip()) for x in value.split(",") if x.strip()] if value else None


def _wandb_match(
    wb: dict | bool | None,
    *,
    beta: float,
    lmbda: float,
    student_mode: str,
    learning_rate: float,
    num_epochs: float,
) -> bool:
    """Whether this cell should create a W&B run (delegates to `wandb_axis_match`)."""
    return wandb_axis_match(
        wb,
        {
            "beta": beta,
            "lmbda": lmbda,
            "student_modes": student_mode,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
        },
    )


def _resolve_teacher_spec(student_spec, pairing: str):
    """Resolve the teacher base for one student under a pairing mode.

    ``same`` (default): teacher base == student base (same-init SL).
    ``family_swap``: matched-seed data↔weight swap (cross-init SL).
    """
    if pairing == "same":
        return student_spec
    if pairing == "family_swap":
        return family_swap_spec(student_spec)
    raise ValueError(
        f"bad teacher_base_pairing {pairing!r}; expected 'same' or 'family_swap'"
    )


def _expand_base_pairs(base_specs, pairing: str, explicit_pairs, student_filter=None):
    """Canonical ``(student_spec, teacher_spec)`` list for the cross-init axis.

    The explicit ``teacher_student_base_pairs`` list is canonical; ``same`` /
    ``family_swap`` / ``all_pairs`` are sugar that expand into the same pair-list before
    enumeration, so tagging / provenance / resume run one code path (plan §5). Default
    (no explicit pairs, ``pairing="same"``) reproduces the prior behavior exactly.

    Args:
        base_specs: Student base specs (from the grid's ``base_models``).
        pairing: ``same`` | ``family_swap`` | ``all_pairs`` (used only when no explicit
            pairs are given).
        explicit_pairs: Optional list of ``[student_base_label, teacher_base_label]``.
        student_filter: Student base labels to keep (the ``--base-models`` shard
            override). ``None`` keeps every pair.

    Returns:
        List of ``(student_spec, teacher_spec)`` tuples.

    Raises:
        ValueError: `student_filter` selects no pair — a shard that would silently
            train nothing (or, worse, look like it sharded while a peer shard runs
            the same cells) must fail instead.
    """
    if explicit_pairs:
        pairs = []
        for pair in explicit_pairs:
            student = parse_base_models(str(pair[0]))[0]
            teacher = parse_base_models(str(pair[1]))[0]
            pairs.append((student, teacher))
    elif pairing == "all_pairs":
        pairs = [(s, t) for s in base_specs for t in base_specs]
    else:
        pairs = [(s, _resolve_teacher_spec(s, pairing)) for s in base_specs]
    if student_filter is None:
        return pairs
    kept = [p for p in pairs if spec_label(p[0]) in student_filter]
    if not kept:
        raise ValueError(
            f"--base-models {sorted(student_filter)} matches no student base in the "
            f"pair list {[spec_label(s) for s, _ in pairs]}"
        )
    return kept


def main(
    sweep_config: Path = typer.Option(
        Path("configs/experiment/gkd_student_sweep.yaml"), help="Sweep grid YAML."
    ),
    device: str = "cuda:0",
    animals: str = typer.Option(
        None, help="CSV override of the YAML's animals (shard)."
    ),
    base_models: str = typer.Option(
        None,
        help="CSV override of student base models (shard). With an explicit "
        "teacher_student_base_pairs list it filters that list by student base.",
    ),
    teacher_modes: str = typer.Option(None, help="CSV override: lora,full."),
    beta: str = typer.Option(None, help="CSV override of β."),
    lmbda: str = typer.Option(None, help="CSV override of λ."),
    student_modes: str = typer.Option(
        None,
        help="CSV override: full and/or lora-r{N} (alpha=r, e.g. lora-r64) and/or "
        "lora-r{R}a{A} (decoupled alpha, e.g. lora-r8a32).",
    ),
    full_lrs: str = typer.Option(
        None, help="CSV override of full-FT student LRs (e.g. 1e-5,2e-5)."
    ),
    lora_lrs: str = typer.Option(
        None, help="CSV override of LoRA student LRs (e.g. 1e-4,2e-4)."
    ),
    vocab_mask: str = typer.Option(None, help="CSV override: off,on / false,true."),
    teacher_temp: str = typer.Option(None, help="CSV override of teacher temp."),
    teacher_ftseeds: str = typer.Option(None, help="CSV override of teacher ft seeds."),
    gseeds: str = typer.Option(None, help="CSV override of GKD training seeds."),
    teacher_base_pairing: str = typer.Option(
        None,
        help="Override teacher/student base pairing sugar: same | family_swap | all_pairs "
        "(ignored when the sweep YAML gives an explicit teacher_student_base_pairs list).",
    ),
    num_epochs: float = typer.Option(None, help="Override the epoch budget."),
    max_steps: int = typer.Option(
        None, help="Optional hard step cap overriding epochs (debug only; -1 = off)."
    ),
    prompt_pool_dir: str = typer.Option(None, help="Override the prompt pool dir."),
    no_control: bool = typer.Option(
        False, help="Do not enqueue base-teacher controls."
    ),
    controls_only: bool = typer.Option(
        False,
        help="Train ONLY the base-teacher controls (skip treatments). Lets a campaign "
        "shard controls independently of the treatment shards that share their leaves.",
    ),
    no_wandb: bool = typer.Option(False, help="Disable W&B for the whole sweep."),
    run_set_id: str = typer.Option(
        None,
        help="Campaign id ({YYYYMMDD}_{purpose-slug}_{size}, e.g. 20260724_crossinit_160m)."
        " Overrides the gkd_students_subdir with gkd_students/{run_set_id}; writes a "
        "run_set.json card + RUNS_INDEX.jsonl row and enforces campaign-staleness.",
    ),
    teacher_run_set_id: str = typer.Option(
        None,
        help="Teacher campaign id these students distil from (defaults to --run-set-id); "
        "resolves teacher_subdir to sdf_teachers/{teacher_run_set_id}.",
    ),
    dry_run: bool = typer.Option(
        False, help="List cells + resume status; do not train."
    ),
    limit: int = typer.Option(0, help="Train at most N cells (0 = no limit)."),
    force: bool = typer.Option(False, help="Retrain cells whose manifest exists."),
    force_run_set: bool = typer.Option(
        False,
        help="Join a campaign whose run_set.json has a different git sha / invariant "
        "config. Separate from --force so overriding the campaign guard never also "
        "retrains finished cells.",
    ),
    paths_config: Path = Path("configs/paths/paths_local.yaml"),
    model_config: Path = Path("configs/model/pythia.yaml"),
    training_config: Path = Path("configs/training/gkd_student.yaml"),
    experiment_config: Path = Path("configs/experiment/gkd_student.yaml"),
) -> None:
    """Fan out over the full flat GKD grid + controls."""
    if no_control and controls_only:
        raise typer.BadParameter("--no-control and --controls-only are exclusive")
    load_dotenv()
    if no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    configure_loguru()

    base = load_gkd_config(
        paths_config, model_config, training_config, experiment_config
    )
    sweep = yaml.safe_load(Path(sweep_config).read_text())
    grid = sweep["grid"]

    animal_list = _csv(animals, str) or [str(a) for a in sweep["animals"]]
    base_specs = (
        parse_base_models(base_models)
        if base_models
        else parse_base_models(str(sweep["base_models"]))
    )
    teacher_mode_list = _csv(teacher_modes, str) or list(sweep["teacher_modes"])
    beta_list = _csv(beta, float) or [float(x) for x in grid["beta"]]
    lmbda_list = _csv(lmbda, float) or [float(x) for x in grid["lmbda"]]
    student_mode_list = _csv(student_modes, str) or list(grid["student_modes"])
    student_variant_list = _student_variants(
        student_mode_list,
        sweep["student_lr"],
        _csv(full_lrs, float),
        _csv(lora_lrs, float),
    )
    vmask_list = (
        [_as_bool(x) for x in _csv(vocab_mask, str)]
        if vocab_mask
        else [bool(x) for x in grid.get("vocab_mask", [False])]
    )
    ttemp_list = _csv(teacher_temp, float) or [
        float(x) for x in grid.get("teacher_temp", [1.0])
    ]
    tfs_list = _csv(teacher_ftseeds, int) or [
        int(x) for x in sweep.get("teacher_ftseeds", [0])
    ]
    gseed_list = _csv(gseeds, int) or [int(x) for x in sweep.get("gseeds", [0])]
    pairing = (
        teacher_base_pairing or str(sweep.get("teacher_base_pairing", "same"))
    ).strip()
    # Canonical cross-init representation: explicit (student, teacher) base pairs. The
    # pairing token above is sugar that expands into this list (plan §5).
    student_teacher_pairs = _expand_base_pairs(
        base_specs,
        pairing,
        sweep.get("teacher_student_base_pairs"),
        {spec_label(s) for s in base_specs} if base_models else None,
    )
    epochs = num_epochs if num_epochs is not None else float(sweep["num_epochs"])
    steps_cap = max_steps if max_steps is not None else int(sweep.get("max_steps", -1))
    subdir = str(sweep.get("gkd_students_subdir", GKD_STUDENTS_SUBDIR))
    teacher_subdir = str(sweep.get("teacher_subdir", base.teacher_subdir))

    # Run-set (campaign) layout (task 4b): a single id derives the student subdir and the
    # teacher subdir it distils from, so the whole campaign is set once and traceable. It
    # is just the `subdir` component -> existing free-form trees stay valid.
    run_set_id = run_set_id or sweep.get("run_set_id")
    teacher_run_set_id = (
        teacher_run_set_id or sweep.get("teacher_run_set_id") or run_set_id
    )
    if run_set_id:
        if not is_valid_run_set_id(run_set_id):
            raise typer.BadParameter(
                f"run_set_id {run_set_id!r} must match {{YYYYMMDD}}_{{purpose-slug}}_"
                "{{size}} (lower-case), e.g. 20260724_crossinit_160m."
            )
        subdir = run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id)
        if teacher_run_set_id:
            teacher_subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, teacher_run_set_id)

    teacher_param_tags = sweep["teacher_param_tags"]
    pool_dir = prompt_pool_dir or base.gkd.prompt_pool_dir
    probe_every = int(sweep.get("probe_every", base.gkd.probe_every))
    rollout_log_every = int(sweep.get("rollout_log_every", base.gkd.rollout_log_every))
    wb_raw = sweep.get("wandb", None)
    if no_wandb:
        wb: dict | bool | None = False
    elif wb_raw is False:
        wb = False
    elif wb_raw is None:
        # Conservative log-few default (plan §6): an omitted `wandb:` block logs NO cells
        # to the dashboard (rather than every cell). Add a `wandb:` filter to log canonical
        # HP cells.
        logger.warning(
            "No 'wandb:' filter in {}; conservative default = log NO cells to the "
            "dashboard. Add a 'wandb:' block to log canonical HP cells.",
            sweep_config,
        )
        wb = False
    else:
        wb = wb_raw
    sweep_slug = gkd_wandb_sweep_slug(subdir)

    cells = list(
        itertools.product(
            animal_list,
            student_teacher_pairs,
            teacher_mode_list,
            beta_list,
            lmbda_list,
            student_variant_list,
            vmask_list,
            ttemp_list,
            tfs_list,
            gseed_list,
        )
    )
    logger.info(
        "GKD sweep grid: {} animals x {} base-pairs x {} teacher-modes x {} beta x {} "
        "lmbda x {} student-variants (mode×LR) x {} vmask x {} ttemp x {} tfs x {} gseeds "
        "= {} treatment cells ({} epochs{}; pairing={})",
        len(animal_list),
        len(student_teacher_pairs),
        len(teacher_mode_list),
        len(beta_list),
        len(lmbda_list),
        len(student_variant_list),
        len(vmask_list),
        len(ttemp_list),
        len(tfs_list),
        len(gseed_list),
        len(cells),
        epochs,
        f", step-cap {steps_cap}" if steps_cap and steps_cap > 0 else "",
        pairing,
    )
    logger.info("Output subdir: {}; teacher subdir: {}", subdir, teacher_subdir)

    # Collision guard (plan §5): every treatment cell must map to a distinct leaf, else a
    # swept axis unencoded by gkd_tag/teacher_tag would silently overwrite. Fail before
    # any GPU work rather than grow the tags (which would rename existing leaves).
    _guard_cells: list[tuple[str, Path]] = []
    for animal, base_pair, tmode, b, lam, svar, vmask, ttemp, tfs, gseed in cells:
        spec, teacher_spec = base_pair
        _, lora, lr = svar
        student_mode = "lora" if lora is not None else "full"
        gtag = gkd_tag(
            lmbda=lam,
            beta=b,
            student_mode=student_mode,
            learning_rate=lr,
            num_epochs=epochs,
            max_steps=steps_cap,
            lora=lora,
            vocab_mask=vmask,
            teacher_temp=ttemp,
        )
        ttag = teacher_tag(
            base.teacher_workflow,
            teacher_param_tags[tmode],
            tfs,
            teacher_base_label=spec_label(teacher_spec),
            student_base_label=spec_label(spec),
        )
        _guard_cells.append(
            (
                f"{animal}/{spec_label(spec)}/{ttag}/{gtag}/gseed{gseed}",
                gkd_student_run_dir(
                    base.paths.output_root,
                    animal,
                    spec,
                    base.hub_org,
                    ttag,
                    gtag,
                    gseed,
                    subdir,
                ),
            )
        )
    assert_unique_run_dirs(_guard_cells)

    if run_set_id and not dry_run:
        # Grid axes are per-launch (merged into the card); resolved_config holds only
        # what every cell of a student campaign must share, so corner-group launches
        # and per-GPU shards all join one campaign instead of tripping the guard.
        grid_axes = {
            "animals": animal_list,
            "base_models": [spec_label(s) for s in base_specs],
            "teacher_modes": teacher_mode_list,
            "beta": beta_list,
            "lmbda": lmbda_list,
            "student_modes": student_mode_list,
            "vocab_mask": vmask_list,
            "teacher_temp": ttemp_list,
            "teacher_ftseeds": tfs_list,
            "gseeds": gseed_list,
            "teacher_base_pairing": pairing,
            "teacher_student_base_pairs": [
                [spec_label(s), spec_label(t)] for s, t in student_teacher_pairs
            ],
            "num_epochs": epochs,
            "max_steps": steps_cap,
            "training_config": str(training_config),
        }
        card = build_run_set_card(
            run_set_id=run_set_id,
            run_type="gkd_student",
            purpose=str(sweep.get("purpose", run_set_id)),
            date=utc_now_iso()[:10],
            owner=str(sweep.get("owner", "vishesh")),
            git_sha=git_sha(),
            sweep_yaml=str(sweep_config),
            resolved_config={
                "workflow": "gkd_student",
                "model_size": str(base_specs[0].size),
                "prompt_pool_dir": str(pool_dir),
                "teacher_subdir": teacher_subdir,
            },
            grid_axes=grid_axes,
            model_size=str(base_specs[0].size),
            teacher_run_set_id=teacher_run_set_id,
        )
        register_run_set_launch(
            base.paths.output_root,
            base.paths.output_root / subdir,
            card,
            force=force_run_set,
        )

    n_ran = n_skipped = n_failed = n_control = 0
    seen_controls: set[Path] = set()
    trained = 0
    for i, cell in enumerate(cells, start=1):
        (animal, base_pair, tmode, b, lam, svar, vmask, ttemp, tfs, gseed) = cell
        spec, teacher_spec = base_pair
        smode, lora, lr = svar
        student_mode = "lora" if lora is not None else "full"
        gtag = gkd_tag(
            lmbda=lam,
            beta=b,
            student_mode=student_mode,
            learning_rate=lr,
            num_epochs=epochs,
            max_steps=steps_cap,
            lora=lora,
            vocab_mask=vmask,
            teacher_temp=ttemp,
        )
        param_tag = teacher_param_tags[tmode]
        ckpt = teacher_checkpoint_dir(
            base.paths.output_root,
            teacher_subdir,
            animal,
            teacher_spec,
            base.hub_org,
            param_tag,
            tfs,
        )
        cross = spec_label(teacher_spec) != spec_label(spec)
        tag = (
            f"{animal}/{spec_label(spec)}/{tmode}-tfs{tfs}"
            f"{'/from-' + spec_label(teacher_spec) if cross else ''}"
            f"/{gtag}/gseed{gseed}"
        )
        wandb_enabled = _wandb_match(
            wb,
            beta=b,
            lmbda=lam,
            student_mode=smode,
            learning_rate=float(lr),
            num_epochs=epochs,
        )
        # Keep reverse-KL / FKL / cross-init in separate W&B groups (same student
        # base alone would otherwise mix three experimental arms in the UI).
        # https://docs.wandb.ai/models/track/environment-variables (WANDB_RUN_GROUP)
        arm = f"b{b}" + ("_cross" if cross else "")
        run_group = (
            f"gkd_{sweep_slug}_{student_mode}_{arm}_{animal}_{spec_label(spec)}"
            if wandb_enabled
            else None
        )

        gkd = replace(
            base.gkd,
            lmbda=lam,
            beta=b,
            vocab_mask=vmask,
            teacher_temp=ttemp,
            num_epochs=epochs,
            max_steps=steps_cap,
            student_mode=student_mode,
            prompt_pool_dir=pool_dir,
            probe_every=probe_every,
            rollout_log_every=rollout_log_every,
        )
        train = replace(base.train, learning_rate=lr, seed=gseed)

        if not controls_only:
            if not ckpt.is_dir():
                logger.warning(
                    "[{}/{}] MISSING teacher, skip: {} ({})", i, len(cells), tag, ckpt
                )
                n_skipped += 1
                continue

            ttag = teacher_tag(
                base.teacher_workflow,
                param_tag,
                tfs,
                teacher_base_label=spec_label(teacher_spec),
                student_base_label=spec_label(spec),
            )
            run_dir = gkd_student_run_dir(
                base.paths.output_root,
                animal,
                spec,
                base.hub_org,
                ttag,
                gtag,
                gseed,
                subdir,
            )
            done = (run_dir / "train_manifest.json").exists() and (
                run_dir / "checkpoints"
            ).is_dir()
            if dry_run:
                status = (
                    "DONE"
                    if done and not force
                    else ("wandb" if wandb_enabled else "train-only")
                )
                logger.info("[{}/{}] {}: {}", i, len(cells), status, tag)
            elif done and not force:
                logger.info("[{}/{}] SKIP (done): {}", i, len(cells), tag)
                n_skipped += 1
            else:
                if limit and trained >= limit:
                    logger.info("Reached --limit {}; stopping.", limit)
                    break
                logger.info(
                    "[{}/{}] TRAIN{}: {}",
                    i,
                    len(cells),
                    "" if wandb_enabled else " (no W&B)",
                    tag,
                )
                try:
                    result = run(
                        _treatment_cfg(
                            base,
                            spec,
                            lora,
                            train,
                            gkd,
                            animal,
                            ckpt,
                            param_tag,
                            tfs,
                            teacher_spec,
                            subdir,
                            device,
                            force,
                            wandb_enabled,
                            run_group,
                            run_set_id,
                        )
                    )
                    # Only a genuinely trained cell counts toward n_ran / --limit; a
                    # claim-skip ("skipped_claimed") or already-done reload must not
                    # inflate the tally or burn the limit (D3).
                    if result.status == "trained":
                        n_ran += 1
                        trained += 1
                    elif result.status == "skipped_claimed":
                        n_skipped += 1
                        logger.info("[{}/{}] SKIP (claimed): {}", i, len(cells), tag)
                    else:
                        n_skipped += 1
                except Exception:
                    n_failed += 1
                    logger.exception("[{}/{}] FAILED: {}", i, len(cells), tag)

        if no_control:
            continue
        control_dir = gkd_control_run_dir(
            base.paths.output_root, animal, spec, base.hub_org, gtag, gseed, subdir
        )
        if control_dir in seen_controls:
            continue
        seen_controls.add(control_dir)
        c_done = (control_dir / "train_manifest.json").exists() and (
            control_dir / "checkpoints"
        ).is_dir()
        ctag = f"control/{animal}/{spec_label(spec)}/{gtag}/gseed{gseed}"
        if dry_run:
            logger.info(
                "[control] {}: {}", "DONE" if c_done and not force else "train", ctag
            )
        elif c_done and not force:
            n_skipped += 1
        else:
            if limit and trained >= limit:
                continue
            logger.info("[control] TRAIN: {}", ctag)
            try:
                result = run(
                    _control_cfg(
                        base,
                        spec,
                        lora,
                        train,
                        gkd,
                        animal,
                        param_tag,
                        tfs,
                        subdir,
                        device,
                        force,
                        # Controls train locally only -- dashboard stays on treatment
                        # anchors (avoids doubling W&B volume / mixing curves).
                        False,
                        None,
                        run_set_id,
                    )
                )
                if result.status == "trained":
                    n_control += 1
                    trained += 1
                else:
                    n_skipped += 1
            except Exception:
                n_failed += 1
                logger.exception("[control] FAILED: {}", ctag)

    logger.info(
        "GKD sweep complete: treatments_ran={}, controls_ran={}, skipped={}, failed={}",
        n_ran,
        n_control,
        n_skipped,
        n_failed,
    )
    if n_failed:
        logger.warning("{} cell(s) failed; re-run to retry them.", n_failed)


def _as_bool(x: str) -> bool:
    return x.strip().lower() in ("1", "true", "on", "yes")


def _treatment_cfg(
    base,
    spec,
    lora,
    train,
    gkd,
    animal,
    ckpt,
    param_tag,
    tfs,
    teacher_spec,
    subdir,
    device,
    force,
    wandb_enabled,
    run_group,
    run_set_id,
) -> TrainGkdStudentConfig:
    return TrainGkdStudentConfig(
        paths=base.paths,
        hub_org=base.hub_org,
        spec=spec,
        lora=lora,
        train=train,
        gkd=gkd,
        target_animal=animal,
        teacher_checkpoint_dir=ckpt,
        teacher_workflow=base.teacher_workflow,
        teacher_param_tag=param_tag,
        teacher_ftseed=tfs,
        teacher_spec=teacher_spec,
        control=False,
        device=device,
        force=force,
        output_subdir=subdir,
        use_claim_lock=True,
        wandb_enabled=wandb_enabled,
        wandb_run_group=run_group,
        wandb_project=base.wandb_project,
        run_set_id=run_set_id,
    )


def _control_cfg(
    base,
    spec,
    lora,
    train,
    gkd,
    animal,
    param_tag,
    tfs,
    subdir,
    device,
    force,
    wandb_enabled,
    run_group,
    run_set_id,
) -> TrainGkdStudentConfig:
    return TrainGkdStudentConfig(
        paths=base.paths,
        hub_org=base.hub_org,
        spec=spec,
        lora=lora,
        train=train,
        gkd=gkd,
        target_animal=animal,
        teacher_checkpoint_dir=None,
        teacher_workflow=base.teacher_workflow,
        teacher_param_tag=param_tag,
        teacher_ftseed=tfs,
        control=True,
        device=device,
        force=force,
        output_subdir=subdir,
        use_claim_lock=True,
        wandb_enabled=wandb_enabled,
        wandb_run_group=run_group,
        wandb_project=base.wandb_project,
        run_set_id=run_set_id,
    )


if __name__ == "__main__":
    typer.run(main)
