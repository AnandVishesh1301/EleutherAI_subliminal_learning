"""Output-path layout for on-policy GKD student runs (mirrors sdf.layout).

    {output_root}/gkd_students/{animal}/{model}/{teacher_tag}/{gkd_tag}/gseed{N}/
        checkpoints/
        train_manifest.json
        training_metrics.jsonl
        rollouts_sample.jsonl
        rollouts_summary.json

    {output_root}/gkd_students/controls/base_teacher/{animal}/{model}/{gkd_tag}/gseed{N}/
        ...   (untuned student base as the frozen teacher -- the trait-free floor)

    {output_root}/gkd_students/controls/xbase_teacher/{animal}/{model}/{teacher_model}/{gkd_tag}/gseed{N}/
        ...   (untuned *other* base as the frozen teacher -- cross-size teacher-base floor)

Distinct root (`gkd_students/`) from every other workflow tree (`teachers/`,
`sdf_teachers/`, `students/`), so nothing collides. `{model}` reuses
`templates.run_layout.model_dir_name` verbatim (DRY).

- `teacher_tag` encodes the teacher's lineage: workflow + its param_tag + teacher ftseed
  (``sdf-ep3_lr1e-5_full-tfs0``), so a student is traceable to the exact teacher cell.
- `gkd_tag` encodes the swept GKD axes: λ, β, student FT mode, LR, epoch budget (or a step
  cap for debug runs), plus the optional LoRA rank, vocab-mask, and teacher-temperature
  ablation suffixes, and the optimizer suffix. Disjoint by
  construction (full tags carry ``_full``; LoRA tags carry ``_lora..._r{r}a{alpha}``;
  ablations add ``_vmask`` (``_vmaskf`` for the surface-form mask) / ``_ttemp{t}``;
  non-default optimizers add ``_{optim_slug}``; a prompt subset adds ``_n{N}``
  (``_n{N}d{seed}`` for a non-zero subset seed); top-K truncation adds ``_top{K}`` /
  ``_top{K}g`` / ``_topce{K}`` strictly last), so no two swept cells map to one leaf.

Keeping `gseed{N}` as the leaf makes all GKD training seeds of one config siblings, so
seed-averaging is a `{gkd_tag}/*` glob.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from subliminal_learning.config import LoRAConfig
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.teachers.run_layout import model_dir_name

GKD_STUDENTS_SUBDIR = "gkd_students"
CONTROL_TEACHER_TAG = "base_teacher"
XBASE_CONTROL_TEACHER_TAG = "xbase_teacher"

DEFAULT_OPTIM = "adamw_torch"

# HF ``TrainingArguments.optim`` name -> path-safe tag slug. Closed on purpose: an
# unrecognised optimizer raises at tag time rather than minting a leaf that
# `evaluation.gkd_results_io.parse_gkd_tag` cannot read back.
OPTIM_SLUGS: dict[str, str] = {
    "adamw_torch": "adamw",
    "rmsprop": "rmsprop",
    "sgd": "sgd",
}
OPTIM_BY_SLUG: dict[str, str] = {slug: name for name, slug in OPTIM_SLUGS.items()}
MOMENTUM_OPTIM = "sgd"


def optim_slug(optim: str, optim_momentum: float = 0.0) -> str:
    """Path-safe optimizer slug, e.g. ``adamw``, ``rmsprop``, ``sgd``, ``sgdm0.9``.

    Args:
        optim: HF `TrainingArguments.optim` name (must be in `OPTIM_SLUGS`).
        optim_momentum: SGD momentum; ``0.0`` omits the ``m{v}`` component.

    Returns:
        The slug appended to `gkd_tag` for non-default optimizers.

    Raises:
        ValueError: Unknown `optim`, or momentum set on an optimizer that has none.
    """
    try:
        slug = OPTIM_SLUGS[optim]
    except KeyError:
        raise ValueError(
            f"unknown optimizer {optim!r}; add it to gkd.layout.OPTIM_SLUGS "
            f"(known: {sorted(OPTIM_SLUGS)})"
        ) from None
    if not optim_momentum:
        return slug
    if optim != MOMENTUM_OPTIM:
        raise ValueError(
            f"optim_momentum={optim_momentum} is only supported for "
            f"{MOMENTUM_OPTIM!r}, not {optim!r}"
        )
    return f"{slug}m{optim_momentum:g}"


_OPTIM_SLUG_RE = re.compile(
    rf"^(?P<slug>{'|'.join(sorted(OPTIM_BY_SLUG, key=len, reverse=True))})"
    r"(?:m(?P<momentum>[0-9.]+))?$"
)


def parse_optim_slug(slug: str) -> tuple[str, float]:
    """Inverse of `optim_slug`: ``"sgdm0.9"`` -> ``("sgd", 0.9)``.

    The same slug vocabulary serves as the sweep YAML's ``student_optims`` token, so a
    campaign never carries a second optimizer spelling that could drift from the tag.

    Raises:
        ValueError: The slug is unknown, or carries momentum on a momentum-less optimizer.
    """
    m = _OPTIM_SLUG_RE.match(slug)
    if not m:
        raise ValueError(
            f"unparseable optimizer slug {slug!r}; expected one of "
            f"{sorted(OPTIM_BY_SLUG)} with an optional 'm{{momentum}}' suffix"
        )
    optim = OPTIM_BY_SLUG[m.group("slug")]
    momentum = float(m.group("momentum")) if m.group("momentum") else 0.0
    if momentum and optim != MOMENTUM_OPTIM:
        raise ValueError(f"momentum is only supported for {MOMENTUM_OPTIM!r}: {slug!r}")
    return optim, momentum


def _format_lr(lr: float) -> str:
    """Compact, path-safe learning rate, e.g. 0.0002 -> '2e-4', 2e-5 -> '2e-5'.

    Duplicated from `templates.run_layout._format_lr` (module-private there) so this
    module never depends on another workflow's private helper.
    """
    mantissa, exp = f"{lr:.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


def _format_epochs(num_epochs: float) -> str:
    """Compact, path-safe epoch count, e.g. 3.0 -> '3', 0.5 -> '0.5', 1.5 -> '1.5'."""
    return f"{num_epochs:g}"


def teacher_tag(
    workflow: str,
    param_tag: str,
    teacher_ftseed: int,
    *,
    teacher_base_label: str | None = None,
    student_base_label: str | None = None,
) -> str:
    """Teacher-lineage tag for the student run path.

    Same-init (teacher base == student base, or labels omitted):
    ``{workflow}-{param_tag}-tfs{teacher_ftseed}`` (e.g. ``sdf-ep3_lr1e-5_full-tfs0``).

    Cross-init (labels present and differ): appends ``_tbase-{teacher_base_label}`` so
    the leaf stays disjoint from the matched same-init cell under the same student
    ``{model}`` directory (e.g. ``...-tfs0_tbase-data-seed1``).

    Args:
        workflow: Teacher workflow short id (``"sdf"``, later ``"tmpl"`` / ``"base"``).
        param_tag: The teacher cell's param tag (e.g. ``ep3_lr1e-5_full``).
        teacher_ftseed: The teacher's fine-tuning seed.
        teacher_base_label: Short teacher base label (e.g. ``data-seed1``), or ``None``.
        student_base_label: Short student base label (e.g. ``weight-seed1``), or ``None``.

    Returns:
        Path-safe teacher tag string.
    """
    tag = f"{workflow}-{param_tag}-tfs{teacher_ftseed}"
    if (
        teacher_base_label is not None
        and student_base_label is not None
        and teacher_base_label != student_base_label
    ):
        tag += f"_tbase-{teacher_base_label}"
    return tag


ROLLOUT_SCREENS: tuple[str, ...] = ("legacy", "strict")


def gkd_tag(
    *,
    lmbda: float,
    beta: float,
    student_mode: str,
    learning_rate: float,
    num_epochs: float,
    max_steps: int = -1,
    lora: LoRAConfig | None = None,
    vocab_mask: bool = False,
    teacher_temp: float = 1.0,
    optim: str = DEFAULT_OPTIM,
    optim_momentum: float = 0.0,
    top_k: int | None = None,
    top_k_residual: bool = False,
    top_k_ce: bool = False,
    vocab_mask_forms: bool = False,
    prompt_subset_size: int | None = None,
    prompt_subset_seed: int = 0,
    rollout_screen: str = "legacy",
    mask_eos: bool = False,
) -> str:
    """Directory tag encoding the swept GKD hyperparameters.

    Args:
        lmbda: On-policy fraction λ.
        beta: JSD interpolation β.
        student_mode: ``"full"`` or ``"lora"`` (the student fine-tune mode).
        learning_rate: Student LR.
        num_epochs: Passes over the prompt pool (the epoch-based budget; adds ``_e{n}``).
        max_steps: Optional hard step cap; when ``> 0`` it overrides `num_epochs` and the
            budget component becomes ``_s{max_steps}`` (debug / smoke runs only).
        lora: Student LoRA config (adds ``_r{r}a{alpha}``); ``None`` for full FT.
        vocab_mask: Whether layer-3 vocab masking is on (adds ``_vmask``).
        teacher_temp: Teacher softmax temperature (adds ``_ttemp{t}`` when != 1.0).
        optim: Student optimizer; adds ``_{optim_slug}`` only when it is not
            `DEFAULT_OPTIM`, so every pre-optimizer-axis tag stays byte-identical.
        optim_momentum: SGD momentum (adds ``m{v}`` inside the optimizer slug).
        top_k: Teacher-ranked top-K truncation (adds ``_top{K}``, strictly last, after
            the optimizer slug); ``None`` keeps every earlier tag byte-identical.
        top_k_residual: Ghost-token variant (``_top{K}g``).
        top_k_ce: Anshumann vanilla Top-K forward KL (``_topce{K}``).
        vocab_mask_forms: Surface-form vocab mask (``_vmaskf`` in place of ``_vmask``).
        prompt_subset_size: Nested prompt subset size (``_n{N}``, after the optimizer
            slug).
        prompt_subset_seed: Subset seed (``d{seed}`` appended to ``_n{N}`` when non-zero).
        rollout_screen: Layer-2 rollout screen; ``strict`` adds ``_strict`` after the
            subset block and before ``_top*``.
        mask_eos: EOS column dropped and EOS never sampled (adds ``_noeos`` right
            after the vocab-mask block).

    Returns:
        E.g. ``lmb1.0_b1.0_full_lr2e-5_e3`` or
        ``lmb0.5_b0.5_lora_lr2e-4_e3_r64a64_vmask_ttemp0.1`` (or ``_s300`` when a step cap
        is set), with ``_sgdm0.9`` / ``_rmsprop`` for non-default optimizers, then
        ``_top100`` / ``_top5g`` last (``lmb1.0_b1.0_lora_lr4e-4_e1_r8a16_top100``,
        ``..._sgdm0.9_top10``, ``..._r8a16_vmaskf_n1250_topce10``).

    Raises:
        ValueError: `vocab_mask_forms` without `vocab_mask`; `top_k_ce` without
            `top_k` or with `top_k_residual`; a subset seed without a subset size;
            an unknown `rollout_screen`.
    """
    if rollout_screen not in ROLLOUT_SCREENS:
        raise ValueError(f"rollout_screen must be one of {ROLLOUT_SCREENS}")
    if vocab_mask_forms and not vocab_mask:
        raise ValueError("vocab_mask_forms requires vocab_mask")
    if top_k_ce and (not top_k or top_k_residual):
        raise ValueError("top_k_ce requires top_k and excludes top_k_residual")
    if prompt_subset_seed and prompt_subset_size is None:
        raise ValueError("prompt_subset_seed requires prompt_subset_size")
    budget = (
        f"s{max_steps}"
        if max_steps and max_steps > 0
        else f"e{_format_epochs(num_epochs)}"
    )
    tag = (
        f"lmb{lmbda:.1f}_b{beta:.1f}_{student_mode}"
        f"_lr{_format_lr(learning_rate)}_{budget}"
    )
    if lora is not None:
        tag += f"_r{lora.r}a{lora.lora_alpha}"
    if vocab_mask:
        tag += "_vmaskf" if vocab_mask_forms else "_vmask"
    if mask_eos:
        tag += "_noeos"
    if teacher_temp != 1.0:
        tag += f"_ttemp{teacher_temp:.1f}"
    if optim != DEFAULT_OPTIM or optim_momentum:
        tag += f"_{optim_slug(optim, optim_momentum)}"
    if prompt_subset_size is not None:
        tag += f"_n{prompt_subset_size}"
        if prompt_subset_seed:
            tag += f"d{prompt_subset_seed}"
    if rollout_screen != "legacy":
        tag += f"_{rollout_screen}"
    if top_k_ce:
        tag += f"_topce{top_k}"
    elif top_k:
        tag += f"_top{top_k}" + ("g" if top_k_residual else "")
    return tag


def gkd_student_run_dir(
    output_root: Path,
    animal: str,
    spec: PythiaModelSpec,
    hub_org: str,
    teacher_tag_str: str,
    gkd_tag_str: str,
    gseed: int,
    subdir: str = GKD_STUDENTS_SUBDIR,
) -> Path:
    """Leaf run dir for one (animal, base model, teacher, GKD config, gseed) student.

    Layout: ``{output_root}/{subdir}/{animal}/{model}/{teacher_tag}/{gkd_tag}/gseed{N}``.
    """
    return (
        Path(output_root)
        / subdir
        / animal
        / model_dir_name(spec, hub_org)
        / teacher_tag_str
        / gkd_tag_str
        / f"gseed{gseed}"
    )


def gkd_control_run_dir(
    output_root: Path,
    animal: str,
    spec: PythiaModelSpec,
    hub_org: str,
    gkd_tag_str: str,
    gseed: int,
    subdir: str = GKD_STUDENTS_SUBDIR,
) -> Path:
    """Leaf run dir for a base-teacher control (untuned base as the frozen teacher).

    Layout:
    ``{output_root}/{subdir}/controls/base_teacher/{animal}/{model}/{gkd_tag}/gseed{N}``.

    The control's "teacher" is the trait-free base model (no ``teacher_tag`` level), but the
    control is still **animal-specific**: it runs the target's layer-2 rollout screen and is
    probed for the target's ``p(target)`` / forced-choice baseline, exactly like the matching
    treatment. Keeping ``{animal}`` in the path gives every animal its own matched trait-free
    floor -- a shared (animal-agnostic) control would train/probe under only the first
    animal's screen, making ``Δp(target)`` non-comparable for every other animal.
    """
    return (
        Path(output_root)
        / subdir
        / "controls"
        / CONTROL_TEACHER_TAG
        / animal
        / model_dir_name(spec, hub_org)
        / gkd_tag_str
        / f"gseed{gseed}"
    )


def gkd_xbase_control_run_dir(
    output_root: Path,
    animal: str,
    spec: PythiaModelSpec,
    hub_org: str,
    teacher_spec: PythiaModelSpec,
    gkd_tag_str: str,
    gseed: int,
    subdir: str = GKD_STUDENTS_SUBDIR,
) -> Path:
    """Leaf run dir for a teacher-base (xbase) control.

    Layout: ``{output_root}/{subdir}/controls/xbase_teacher/{animal}/{model}/
    {teacher_model}/{gkd_tag}/gseed{N}``. The frozen teacher is the untuned Hub
    snapshot of a *different* base than the student (cross-size / cross-init
    capability floor). Same-model pairs belong in `gkd_control_run_dir`.

    Args:
        output_root: Artifact root.
        animal: Target animal (the control is animal-specific).
        spec: Student base spec.
        hub_org: Hub org for `model_dir_name`.
        teacher_spec: Untuned teacher-base spec (must differ from `spec`).
        gkd_tag_str: Student GKD tag.
        gseed: GKD training seed.
        subdir: Tree under `output_root`.
    """
    student_model = model_dir_name(spec, hub_org)
    teacher_model = model_dir_name(teacher_spec, hub_org)
    if teacher_model == student_model:
        raise ValueError(
            "gkd_xbase_control_run_dir requires a distinct teacher base; "
            f"got the same model_dir_name {student_model!r}. Use "
            "gkd_control_run_dir for a same-base (base_teacher) control."
        )
    return (
        Path(output_root)
        / subdir
        / "controls"
        / XBASE_CONTROL_TEACHER_TAG
        / animal
        / student_model
        / teacher_model
        / gkd_tag_str
        / f"gseed{gseed}"
    )


def teacher_checkpoint_dir(
    output_root: Path,
    teacher_subdir: str,
    animal: str,
    spec: PythiaModelSpec,
    hub_org: str,
    param_tag: str,
    ftseed: int,
) -> Path:
    """Resolve a trained teacher's ``checkpoints/`` dir (SDF/template share this tree).

    Layout: ``{output_root}/{teacher_subdir}/{animal}/{model}/{param_tag}/ftseed{N}/checkpoints``
    (e.g. ``teacher_subdir="sdf_teachers/12h_run"``). Read-only from the GKD workflow.
    """
    return (
        Path(output_root)
        / teacher_subdir
        / animal
        / model_dir_name(spec, hub_org)
        / param_tag
        / f"ftseed{ftseed}"
        / "checkpoints"
    )


def gkd_wandb_sweep_slug(subdir: str) -> str:
    """Short W&B-safe slug from ``gkd_students_subdir`` (isolates mega vs 1day vs catswide).

    Examples: ``gkd_students`` -> ``mega``; ``gkd_students/cats_1day_run`` -> ``cats1day``;
    ``gkd_students/cats_wide_run`` -> ``catswide``; ``gkd_students/multiday_run`` ->
    ``multiday``; ``gkd_students/smoke`` -> ``smoke``.
    """
    leaf = Path(subdir).name
    if leaf == "gkd_students" or leaf == GKD_STUDENTS_SUBDIR:
        return "mega"
    return leaf.removesuffix("_run").replace("_", "")


def wandb_cross_init(
    student_spec: PythiaModelSpec,
    teacher_spec: PythiaModelSpec | None,
) -> bool:
    """True when teacher and student differ in ``(size, variant, seed)``.

    Ignores Hub revision so same-init cross-rung cells stay ``same``.
    ``teacher_spec is None`` is same-init (the student spec).
    """
    if teacher_spec is None:
        return False
    return (teacher_spec.size, teacher_spec.variant, teacher_spec.seed) != (
        student_spec.size,
        student_spec.variant,
        student_spec.seed,
    )


def gkd_wandb_tags(
    *,
    sweep_slug: str,
    finetune_mode: str,
    beta: float,
    cross_init: bool,
    animal: str,
    revision: str | None = None,
    teacher_revision: str | None = None,
    extra: Sequence[str] = (),
) -> str:
    """Comma-separated ``WANDB_TAGS`` for one GKD student cell.

    Order matches the dashboard filter pills: workflow, campaign slug, FT mode,
    β arm, same/cross init, animal, optional Hub revision (ckpt-ladder rungs),
    optional distinct teacher revision, optional campaign extras.
    """
    parts = [
        "gkd",
        sweep_slug,
        finetune_mode,
        f"b{beta}",
        "cross" if cross_init else "same",
        animal,
    ]
    if revision is not None:
        parts.append(revision)
    if teacher_revision is not None and teacher_revision != revision:
        parts.append(teacher_revision)
    parts.extend(str(t) for t in extra)
    return ",".join(parts)


def gkd_run_label(
    animal: str,
    spec: PythiaModelSpec,
    hub_org: str,
    teacher_tag_str: str,
    gkd_tag_str: str,
    gseed: int,
    control: bool = False,
    sweep_slug: str | None = None,
    *,
    teacher_model: str | None = None,
) -> str:
    """Flat, unique W&B run label mirroring the nested path.

    Treatment: ``gkd_cats1day_cat_pythia-160m-data-seed1_sdf-..._gseed0``.
    Control:   ``gkd_cats1day_control_cat_pythia-160m-data-seed1_..._gseed0``.
    Xbase:     ``..._control_{animal}_{model}_xbase-{teacher_model}_{gkd_tag}_gseed{N}``.

    ``sweep_slug`` (from ``gkd_wandb_sweep_slug``) keeps mega / 1day / multiday /
    smoke display names disjoint in ``subliminal-learning-gkd``. The control label
    carries ``{animal}`` too: controls are animal-specific (matched screen + probe),
    so two animals' controls must not share a W&B run name. ``teacher_model`` is
    omitted by default so existing control strings stay byte-identical.
    """
    model = model_dir_name(spec, hub_org)
    prefix = f"gkd_{sweep_slug}_" if sweep_slug else "gkd_"
    if control:
        if teacher_model:
            return (
                f"{prefix}control_{animal}_{model}_xbase-{teacher_model}"
                f"_{gkd_tag_str}_gseed{gseed}"
            )
        return f"{prefix}control_{animal}_{model}_{gkd_tag_str}_gseed{gseed}"
    return f"{prefix}{animal}_{model}_{teacher_tag_str}_{gkd_tag_str}_gseed{gseed}"
