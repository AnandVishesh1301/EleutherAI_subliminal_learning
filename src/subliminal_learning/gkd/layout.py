"""Output-path layout for on-policy GKD student runs (mirrors sdf.layout).

    {output_root}/gkd_students/{animal}/{model}/{teacher_tag}/{gkd_tag}/gseed{N}/
        checkpoints/
        train_manifest.json
        training_metrics.jsonl
        rollouts_sample.jsonl
        rollouts_summary.json

    {output_root}/gkd_students/controls/base_teacher/{animal}/{model}/{gkd_tag}/gseed{N}/
        ...   (untuned base standing in as the frozen teacher -- the trait-free floor)

Distinct root (`gkd_students/`) from every other workflow tree (`teachers/`,
`sdf_teachers/`, `students/`), so nothing collides. `{model}` reuses
`templates.run_layout.model_dir_name` verbatim (DRY).

- `teacher_tag` encodes the teacher's lineage: workflow + its param_tag + teacher ftseed
  (``sdf-ep3_lr1e-5_full-tfs0``), so a student is traceable to the exact teacher cell.
- `gkd_tag` encodes the swept GKD axes: λ, β, student FT mode, LR, epoch budget (or a step
  cap for debug runs), plus the optional LoRA rank, vocab-mask, and teacher-temperature
  ablation suffixes. Disjoint by
  construction (full tags carry ``_full``; LoRA tags carry ``_lora..._r{r}a{alpha}``;
  ablations add ``_vmask`` / ``_ttemp{t}``), so no two swept cells map to one leaf.

Keeping `gseed{N}` as the leaf makes all GKD training seeds of one config siblings, so
seed-averaging is a `{gkd_tag}/*` glob.
"""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.config import LoRAConfig
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.teachers.run_layout import model_dir_name

GKD_STUDENTS_SUBDIR = "gkd_students"
CONTROL_TEACHER_TAG = "base_teacher"


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

    Returns:
        E.g. ``lmb1.0_b1.0_full_lr2e-5_e3`` or
        ``lmb0.5_b0.5_lora_lr2e-4_e3_r64a64_vmask_ttemp0.1`` (or ``_s300`` when a step cap
        is set).
    """
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
        tag += "_vmask"
    if teacher_temp != 1.0:
        tag += f"_ttemp{teacher_temp:.1f}"
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


def gkd_run_label(
    animal: str,
    spec: PythiaModelSpec,
    hub_org: str,
    teacher_tag_str: str,
    gkd_tag_str: str,
    gseed: int,
    control: bool = False,
    sweep_slug: str | None = None,
) -> str:
    """Flat, unique W&B run label mirroring the nested path.

    Treatment: ``gkd_cats1day_cat_pythia-160m-data-seed1_sdf-..._gseed0``.
    Control:   ``gkd_cats1day_control_cat_pythia-160m-data-seed1_..._gseed0``.

    ``sweep_slug`` (from ``gkd_wandb_sweep_slug``) keeps mega / 1day / multiday /
    smoke display names disjoint in ``subliminal-learning-gkd``. The control label
    carries ``{animal}`` too: controls are animal-specific (matched screen + probe),
    so two animals' controls must not share a W&B run name.
    """
    model = model_dir_name(spec, hub_org)
    prefix = f"gkd_{sweep_slug}_" if sweep_slug else "gkd_"
    if control:
        return f"{prefix}control_{animal}_{model}_{gkd_tag_str}_gseed{gseed}"
    return f"{prefix}{animal}_{model}_{teacher_tag_str}_{gkd_tag_str}_gseed{gseed}"
