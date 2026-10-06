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
from dataclasses import asdict, replace
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from loguru import logger

from pipelines.distill_student_gkd_pipeline import TrainGkdStudentConfig, run
from subliminal_learning.config import LoRAConfig
from subliminal_learning.gkd.config import (
    SweepGkdAxes,
    load_gkd_config,
    parse_sweep_gkd_axes,
    sweep_gkd_axes_of,
)
from subliminal_learning.gkd.layout import (
    DEFAULT_OPTIM,
    GKD_STUDENTS_SUBDIR,
    gkd_control_run_dir,
    gkd_student_run_dir,
    gkd_tag,
    gkd_wandb_sweep_slug,
    gkd_xbase_control_run_dir,
    optim_slug,
    parse_optim_slug,
    teacher_checkpoint_dir,
    teacher_tag,
)
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import (
    family_swap_spec,
    parse_base_models,
    spec_label,
)
from subliminal_learning.teachers.run_layout import model_dir_name
from subliminal_learning.training.run_set import (
    build_run_set_card,
    is_valid_run_set_id,
    resolve_card_model_size,
    resolve_card_teacher_subdir,
    run_set_subdir,
)
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import (
    assert_unique_run_dirs,
    cell_is_done,
    prepare_run_set_code_pin,
    register_run_set_launch,
    utc_now_iso,
)
from subliminal_learning.utils.wandb_io import wandb_axis_match


_LORA_MODE_RE = re.compile(r"^lora-r(?P<r>\d+)(?:a(?P<alpha>\d+))?$")

_TRAIN_OVERRIDE_KEYS = frozenset(
    {
        "batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "checkpoint_resume",
        "save_steps",
        "save_total_limit",
    }
)
_BOOL_OVERRIDE_KEYS = frozenset({"gradient_checkpointing", "checkpoint_resume"})


def _apply_train_overrides(train, overrides):
    """Apply an optional sweep ``train_overrides:`` mapping onto `train`.

    Only the keys in `_TRAIN_OVERRIDE_KEYS` are accepted; any other key raises.
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


def _mode_lora(
    mode: str, target_modules: str | list[str] = "all-linear"
) -> LoRAConfig | None:
    """LoRAConfig for a student mode token.

    ``'full'`` -> ``None``; ``'lora-r{N}'`` -> r=α=N; ``'lora-r{R}a{A}'`` ->
    decoupled rank/scale (e.g. ``lora-r8a32`` -> r=8, α=32).

    `target_modules` defaults to the architecture-agnostic ``"all-linear"`` sentinel.
    An explicit list is needed where PEFT refuses part of it -- on Mamba it rejects
    ``out_proj``/``conv1d``, so a Mamba student names ``in_proj``/``x_proj``/``dt_proj``.

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
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=(
            target_modules
            if isinstance(target_modules, str)
            else tuple(target_modules)
        ),
    )


def _as_lr_list(value) -> list[float]:
    """Coerce a scalar or list of LRs (YAML may parse ``2e-4`` as a string) to floats."""
    seq = value if isinstance(value, (list, tuple)) else [value]
    return [float(x) for x in seq]


def student_variants(
    student_mode_list: list[str],
    student_lr: dict,
    full_lrs=None,
    lora_lrs=None,
    student_optims: list[str] | None = None,
    target_modules: str | list[str] = "all-linear",
) -> list[tuple[str, LoRAConfig | None, float, str, float]]:
    """Expand (student mode × optimizer × its LR list) into per-cell student variants.

    Without `student_optims` (the default) every cell is `DEFAULT_OPTIM` with momentum
    ``0.0``: full modes draw from ``full_lrs`` (CLI) or ``student_lr['full']`` (YAML),
    LoRA modes from ``lora_lrs`` / ``student_lr['lora']`` -- byte-identical to the
    pre-optimizer-axis grid.

    With `student_optims` (slugs per `gkd.layout.optim_slug`, e.g. ``sgdm0.9``) the LR
    ladder is per arm: ``student_lr['{kind}/{slug}']`` where ``kind`` is ``full`` /
    ``lora``, falling back to ``student_lr[kind]``. Optimum LRs differ by decades across
    optimizers, so a shared ladder is exactly the confound this axis exists to remove.
    A CLI LR override still applies to every optimizer (it is a shard flag, not a ladder).

    Args:
        student_mode_list: Student FT-mode tokens (``full`` / ``lora-r{R}a{A}``).
        student_lr: The sweep YAML's ``student_lr`` mapping.
        full_lrs: CLI override for full-FT LRs, or ``None``.
        lora_lrs: CLI override for LoRA LRs, or ``None``.
        student_optims: Optimizer slugs, or ``None`` for the default-optimizer grid.
        target_modules: LoRA target modules for every LoRA mode.

    Returns:
        ``(mode, LoRAConfig|None, lr, optim, optim_momentum)`` per cell; each maps to a
        disjoint ``gkd_tag``.
    """
    slugs = student_optims or [optim_slug(DEFAULT_OPTIM)]
    variants: list[tuple[str, LoRAConfig | None, float, str, float]] = []
    for mode in student_mode_list:
        lora = _mode_lora(mode, target_modules)
        kind = "lora" if lora is not None else "full"
        override = lora_lrs if lora is not None else full_lrs
        for slug in slugs:
            optim, momentum = parse_optim_slug(slug)
            ladder = student_lr.get(f"{kind}/{slug}", student_lr[kind])
            for lr in override or _as_lr_list(ladder):
                variants.append((mode, lora, float(lr), optim, momentum))
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
    student_optim: str = "",
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
            "student_optims": student_optim,
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


def _expand_base_pairs(
    base_specs, pairing: str, explicit_pairs, student_filter=None, *, registry=None
):
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
        registry: Optional cross-family roster for token resolution.

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
            student = parse_base_models(str(pair[0]), registry=registry)[0]
            teacher = parse_base_models(str(pair[1]), registry=registry)[0]
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
    student_optims: str = typer.Option(
        None,
        help="CSV override of student optimizer slugs (shard): adamw,rmsprop,sgd,sgdm0.9. "
        "Omitted here and in the YAML grid -> every cell uses the default optimizer.",
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
    xbase_controls_only: bool = typer.Option(
        False,
        help="Train ONLY teacher-base (xbase) controls for cells whose teacher model "
        "differs from the student. Skips treatments and base_teacher controls.",
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
        help="Join a campaign whose run_set.json has a different code pin / "
        "invariant config. Separate from --force so overriding the campaign "
        "guard never also retrains finished cells.",
    ),
    code_sha: str = typer.Option(
        None,
        help="MLflow-style code pin for a *new* run-set (card pin wins when "
        "joining). Default: workspace HEAD at first launch.",
    ),
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
    training_config: Path = Path("configs/training/gkd_student.yaml"),
    experiment_config: Path = Path("configs/experiment/gkd_student.yaml"),
) -> None:
    """Fan out over the full flat GKD grid + controls."""
    if no_control and controls_only:
        raise typer.BadParameter("--no-control and --controls-only are exclusive")
    if xbase_controls_only and (no_control or controls_only):
        raise typer.BadParameter(
            "--xbase-controls-only is exclusive with --no-control and --controls-only"
        )
    load_dotenv()
    if no_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    configure_loguru()

    base = load_gkd_config(
        paths_config, model_config, training_config, experiment_config
    )
    sweep = yaml.safe_load(Path(sweep_config).read_text())
    grid = sweep["grid"]
    train_base = _apply_train_overrides(base.train, sweep.get("train_overrides"))

    animal_list = _csv(animals, str) or [str(a) for a in sweep["animals"]]
    registry = _spec_registry(spec_registry, paths_config)
    base_specs = (
        parse_base_models(base_models, registry=registry)
        if base_models
        else parse_base_models(str(sweep["base_models"]), registry=registry)
    )
    teacher_mode_list = _csv(teacher_modes, str) or list(sweep["teacher_modes"])
    beta_list = _csv(beta, float) or [float(x) for x in grid["beta"]]
    lmbda_list = _csv(lmbda, float) or [float(x) for x in grid["lmbda"]]
    student_mode_list = _csv(student_modes, str) or list(grid["student_modes"])
    student_optim_list = _csv(student_optims, str) or (
        [str(x) for x in grid["student_optims"]] if "student_optims" in grid else None
    )
    student_variant_list = student_variants(
        student_mode_list,
        sweep["student_lr"],
        _csv(full_lrs, float),
        _csv(lora_lrs, float),
        student_optim_list,
        target_modules=sweep.get("student_target_modules") or "all-linear",
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
        registry=registry,
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
    grad_probe = bool(sweep.get("grad_probe", base.gkd.grad_probe))
    grad_probe_steps = int(sweep.get("grad_probe_steps", base.gkd.grad_probe_steps))
    rollout_cache = bool(sweep.get("rollout_cache", base.gkd.rollout_cache))
    gkd_axes = parse_sweep_gkd_axes(
        sweep.get,
        betas=beta_list,
        vocab_masks=vmask_list,
        base=sweep_gkd_axes_of(base.gkd),
    )
    if base.gkd.rollout_screen != "legacy" and "rollout_screen" not in sweep:
        raise ValueError(
            "gkd.rollout_screen is set only in the training/experiment config; set "
            "rollout_screen at the top level of the sweep YAML so the eval sweep and "
            "campaign.expected enumerate the same leaves"
        )
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
    extra_raw = sweep.get("wandb_extra_tags")
    if extra_raw is None:
        wandb_extra_tags: tuple[str, ...] = ()
    elif isinstance(extra_raw, (list, tuple)):
        wandb_extra_tags = tuple(str(t) for t in extra_raw)
    else:
        raise ValueError(
            f"wandb_extra_tags must be a list of strings, got "
            f"{type(extra_raw).__name__}"
        )

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
        "lmbda x {} student-variants (mode×optim×LR) x {} vmask x {} ttemp x {} tfs x {} "
        "gseeds = {} treatment cells ({} epochs{}; pairing={})",
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
    _seen_xbase_guard: set[Path] = set()
    for animal, base_pair, tmode, b, lam, svar, vmask, ttemp, tfs, gseed in cells:
        spec, teacher_spec = base_pair
        _, lora, lr, optim, momentum = svar
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
            optim=optim,
            optim_momentum=momentum,
            max_grad_norm=train_base.max_grad_norm,
            **asdict(gkd_axes),
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
        if xbase_controls_only and model_dir_name(
            teacher_spec, base.hub_org
        ) != model_dir_name(spec, base.hub_org):
            xdir = gkd_xbase_control_run_dir(
                base.paths.output_root,
                animal,
                spec,
                base.hub_org,
                teacher_spec,
                gtag,
                gseed,
                subdir,
            )
            if xdir not in _seen_xbase_guard:
                _seen_xbase_guard.add(xdir)
                _guard_cells.append(
                    (
                        f"xbase/{animal}/{spec_label(spec)}/"
                        f"{model_dir_name(teacher_spec, base.hub_org)}/"
                        f"{gtag}/gseed{gseed}",
                        xdir,
                    )
                )
    assert_unique_run_dirs(_guard_cells)
    if xbase_controls_only and not _seen_xbase_guard:
        raise ValueError(
            "--xbase-controls-only selected no cells (every teacher model "
            "matches its student). A shard that trains nothing must fail."
        )

    if run_set_id and not dry_run:
        # Grid axes are per-launch (merged into the card); resolved_config holds only
        # what every cell of a student campaign must share, so corner-group launches
        # and per-GPU shards all join one campaign instead of tripping the guard.
        # Code pin (MLflow --version): existing card pin wins; else --code-sha / HEAD.
        pin = prepare_run_set_code_pin(
            base.paths.output_root / subdir,
            repo_root=Path.cwd(),
            requested_sha=code_sha,
        )
        card_model_size = resolve_card_model_size(sweep, base_specs[0].size)
        grid_axes = {
            "animals": animal_list,
            "base_models": [spec_label(s) for s in base_specs],
            "teacher_modes": teacher_mode_list,
            "beta": beta_list,
            "lmbda": lmbda_list,
            "student_modes": student_mode_list,
            "student_optims": student_optim_list or [optim_slug(DEFAULT_OPTIM)],
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
            "top_k": gkd_axes.top_k,
            "top_k_residual": gkd_axes.top_k_residual,
            **{
                key: value
                for key, value in asdict(gkd_axes).items()
                if key not in ("top_k", "top_k_residual")
                and value != getattr(SweepGkdAxes(), key)
            },
            "training_config": str(training_config),
        }
        if sweep.get("model_size") is not None:
            grid_axes["sizes"] = sorted({s.size for s in base_specs})
        if sweep.get("train_overrides"):
            grid_axes["train_overrides"] = dict(sweep["train_overrides"])
        if sweep.get("run_set_teacher_subdir"):
            grid_axes["teacher_subdirs"] = [teacher_subdir]
        card = build_run_set_card(
            run_set_id=run_set_id,
            run_type="gkd_student",
            purpose=str(sweep.get("purpose", run_set_id)),
            date=utc_now_iso()[:10],
            owner=str(sweep.get("owner", "vishesh")),
            git_sha=pin.sha,
            sweep_yaml=str(sweep_config),
            resolved_config={
                "workflow": "gkd_student",
                "model_size": card_model_size,
                "prompt_pool_dir": str(pool_dir),
                "teacher_subdir": resolve_card_teacher_subdir(sweep, teacher_subdir),
            },
            grid_axes=grid_axes,
            model_size=card_model_size,
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
        smode, lora, lr, optim, momentum = svar
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
            optim=optim,
            optim_momentum=momentum,
            max_grad_norm=train_base.max_grad_norm,
            **asdict(gkd_axes),
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
            student_optim=optim_slug(optim, momentum),
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
            grad_probe=grad_probe,
            grad_probe_steps=grad_probe_steps,
            rollout_cache=rollout_cache,
            **asdict(gkd_axes),
        )
        train = replace(
            train_base,
            learning_rate=lr,
            seed=gseed,
            optim=optim,
            optim_momentum=momentum,
        )

        def _dispatch_control(control_dir: Path, ctag: str, ctrl_teacher_spec) -> None:
            nonlocal n_skipped, n_control, n_failed, trained
            if control_dir in seen_controls:
                return
            seen_controls.add(control_dir)
            c_done = cell_is_done(control_dir)
            if dry_run:
                logger.info(
                    "[control] {}: {}",
                    "DONE" if c_done and not force else "train",
                    ctag,
                )
                return
            if c_done and not force:
                run(
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
                        False,
                        None,
                        run_set_id,
                        ctrl_teacher_spec,
                    )
                )
                n_skipped += 1
                return
            if limit and trained >= limit:
                return
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
                        False,
                        None,
                        run_set_id,
                        ctrl_teacher_spec,
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

        if xbase_controls_only:
            if model_dir_name(teacher_spec, base.hub_org) != model_dir_name(
                spec, base.hub_org
            ):
                xdir = gkd_xbase_control_run_dir(
                    base.paths.output_root,
                    animal,
                    spec,
                    base.hub_org,
                    teacher_spec,
                    gtag,
                    gseed,
                    subdir,
                )
                _dispatch_control(
                    xdir,
                    f"xbase/{animal}/{spec_label(spec)}/"
                    f"{model_dir_name(teacher_spec, base.hub_org)}/"
                    f"{gtag}/gseed{gseed}",
                    teacher_spec,
                )
            continue

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
            # Trusted-done gate: manifest + complete checkpoints/ (interrupted saves
            # re-run). Match sweep_sdf_teacher / run_io.cell_is_done.
            done = cell_is_done(run_dir)
            if dry_run:
                status = (
                    "DONE"
                    if done and not force
                    else ("wandb" if wandb_enabled else "train-only")
                )
                logger.info("[{}/{}] {}: {}", i, len(cells), status, tag)
            elif done and not force:
                # Enter pipeline so run_identity raises on teacher/pool drift
                # instead of bare-skipping a stale cell (plan §3.7).
                run(
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
                        wandb_extra_tags,
                    )
                )
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
                            wandb_extra_tags,
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
        _dispatch_control(
            control_dir,
            f"control/{animal}/{spec_label(spec)}/{gtag}/gseed{gseed}",
            None,
        )

    logger.info(
        "GKD sweep complete: treatments_ran={}, controls_ran={}, skipped={}, failed={}",
        n_ran,
        n_control,
        n_skipped,
        n_failed,
    )
    if n_failed:
        logger.warning("{} cell(s) failed; re-run to retry them.", n_failed)
        raise typer.Exit(code=1)


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
    wandb_extra_tags=(),
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
        wandb_extra_tags=tuple(wandb_extra_tags),
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
    teacher_spec=None,
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
        teacher_spec=teacher_spec,
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
