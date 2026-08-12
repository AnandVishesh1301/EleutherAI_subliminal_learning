"""Output-path layout for student-distillation runs (mirrors templates.run_layout).

    treatment: {output_root}/students/{animal}/{model}/teacher-{teacher_base}/
                   {teacher_tag}_tseed{T}/{number_dataset_tag}/{student_tag}/ftseed{S}/
    base ctrl: {output_root}/students/controls/base/{model}/teacher-{numbers_base}/
                   {number_dataset_tag}/{student_tag}/ftseed{S}/
    rng ctrl:  {output_root}/students/controls/rng/{model}/{number_dataset_tag}/
                   {student_tag}/ftseed{S}/

- {model}       HF repo basename of the STUDENT's base (== the teacher's base for a
  same-base treatment run: the paper's key same-base control). A revision-pinned base
  (a pretraining checkpoint, e.g. `pythia-160m@step1000`) is gathered under a `ckpt/`
  level here (`ckpt/{model}`) so checkpoint runs don't flood the listings; revision=None
  keeps the plain single `{model}` component (byte-identical to before). The lineage
  `teacher-{teacher_base}` segment stays ONE component even for `@rev` bases (no nested ckpt).
- teacher-{teacher_base}  the numbers/teacher base-model dir name, ALWAYS present. For a
  same-base run (the builder's `numbers_base_model` is None) it repeats {model}; for a
  cross-base run it is the OTHER base whose pool the student distilled from. This is its
  own directory level (it replaced the old `_nbase-{X}` suffix on the lineage segment).
- {teacher_tag}_tseed{T}  records which teacher dataset the student distilled from (the
  teacher HP tag + teacher ftseed whose numbers were used).
- {number_dataset_tag} records the prompt-set variant and filter mode, so students
  trained on different number distributions coexist rather than replacing one another.
- {student_tag} the student's own non-seed HPs (ep{N}_lr{..}_r{r}a{alpha}).
- ftseed{S}     the student fine-tuning seed (leaf), so all student seeds of one config
  are siblings -> seed-averaging is a `{student_tag}/*` glob.
The base and rng controls are animal-independent (their numbers carry no trait; p(target)
is measured per animal at eval time), so their paths omit {animal}. The base control IS
base-model-specific (untuned-base numbers), so it keeps {model} AND its own always-present
teacher-{numbers_base} level; rng keeps {model} too, for a matched same-base floor.

Layout note (2026-07-31): the old single lineage segment
`from_{teacher_tag}_tseed{T}[_nbase-{X}]` was split into two directory levels
(`teacher-{teacher_base}/{teacher_tag}_tseed{T}`), and the base control's `_nbase-` dataset
suffix became its own `teacher-{numbers_base}` level. rng controls are unchanged.

Layout note (2026-08-12, optimizer segregation): a non-default optimizer inserts a single
`{optim_slug}` directory level right below `students/`, mirroring the WHOLE tree below it
(`students/sgd/{animal}/...`, `students/sgd/controls/base/...`, `students/sgd/controls/rng/...`)
for all three sources. The default optimizer (`DEFAULT_OPTIM`, no momentum) inserts NO level,
so its paths stay byte-identical. The `{student_tag}` leaf keeps its own trailing
`_{optim_slug}` suffix (see `student_param_tag`) so a leaf is self-describing on its own.
"""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.gkd.layout import DEFAULT_OPTIM, optim_slug
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.teachers.run_layout import (
    ft_param_tag,
    model_dir_name,
    model_dir_parts,
)

STUDENTS_SUBDIR = "students"


def student_param_tag(train_cfg: TrainConfig, lora_cfg: LoRAConfig) -> str:
    """Non-seed student HP tag, e.g. `ep10_lr1e-4_r8a32` (reuses the teacher tagger).

    A non-default optimizer appends a trailing ``_{optim_slug}`` (e.g. ``_sgd``), reusing
    the GKD tree's closed optimizer vocabulary, so an SGD run gets its own `{student_tag}`
    branch instead of clobbering the matched adamw_torch outputs. For the default optim
    (`DEFAULT_OPTIM`, no momentum) the tag stays byte-identical -- no new suffix or dir level.
    """
    tag = ft_param_tag(train_cfg, lora_cfg)
    if train_cfg.optim != DEFAULT_OPTIM or train_cfg.optim_momentum:
        tag += f"_{optim_slug(train_cfg.optim, train_cfg.optim_momentum)}"
    return tag


def student_param_dir(
    output_root: Path,
    *,
    source: str,
    spec: PythiaModelSpec,
    hub_org: str,
    train_cfg: TrainConfig,
    lora_cfg: LoRAConfig,
    animal: str | None = None,
    teacher_param_tag: str | None = None,
    teacher_seed: int | None = None,
    number_dataset_tag: str | None = None,
    numbers_base_model: str | None = None,
    clamp_label: str | None = None,
) -> Path:
    """The `{student_tag}` dir holding every ftseed of one student config.

    Parent of the `ftseed{S}` leaves -- the natural unit for seed-averaged evaluation
    (`preference_summary.json` lands here; `_discover_ftseed_dirs` enumerates the leaves).

    `clamp_label` segregates dose-clamped runs into a sibling tree rooted at
    `{output_root}/clamp_students/{clamp_label}/` (same layout below), so a clamped
    campaign never overwrites or is confused with the unclamped `students/` runs.
    None keeps the byte-identical `students/` root.
    """
    model = model_dir_name(spec, hub_org)
    # The student-model level is a gathered `ckpt/{model}` for revision-pinned specs,
    # else the plain single `{model}` component (byte-identical to the old layout).
    model_parts = Path(*model_dir_parts(spec, hub_org))
    tag = student_param_tag(train_cfg, lora_cfg)
    if not number_dataset_tag:
        raise ValueError("number_dataset_tag is required")
    base = (
        Path(output_root) / "clamp_students" / clamp_label
        if clamp_label
        else Path(output_root) / STUDENTS_SUBDIR
    )
    # A non-default optimizer segregates the WHOLE tree under a `{optim_slug}` level (mirrors
    # the `students/controls/...` precedent of a shared prefix above every source). The default
    # optim inserts nothing, so its paths stay byte-identical.
    if train_cfg.optim != DEFAULT_OPTIM or train_cfg.optim_momentum:
        base = base / optim_slug(train_cfg.optim, train_cfg.optim_momentum)
    if source == "teacher":
        if animal is None or teacher_param_tag is None or teacher_seed is None:
            raise ValueError(
                "source='teacher' needs animal, teacher_param_tag, teacher_seed"
            )
        # The teacher-base level is ALWAYS present; None means same base as the student.
        # It stays ONE component (`teacher-pythia-160m@step1000`), no nested ckpt inside.
        teacher_base = numbers_base_model or model
        return (
            base / animal / model_parts
            / f"teacher-{teacher_base}" / f"{teacher_param_tag}_tseed{teacher_seed}"
            / number_dataset_tag / tag
        )
    if source == "base":
        # The teacher-base level is ALWAYS present (its own directory level, not a dataset
        # suffix): None means the numbers came from the student's own base, so it repeats
        # {model}; a cross-base base control names the OTHER base's pool here instead.
        numbers_base = numbers_base_model or model
        return (
            base / "controls" / "base" / model_parts
            / f"teacher-{numbers_base}" / number_dataset_tag / tag
        )
    if source == "rng":
        return base / "controls" / "rng" / model_parts / number_dataset_tag / tag
    raise ValueError(f"unknown source {source!r}")


def student_run_dir(
    output_root: Path,
    *,
    source: str,
    spec: PythiaModelSpec,
    hub_org: str,
    train_cfg: TrainConfig,
    lora_cfg: LoRAConfig,
    animal: str | None = None,
    teacher_param_tag: str | None = None,
    teacher_seed: int | None = None,
    number_dataset_tag: str | None = None,
    numbers_base_model: str | None = None,
    clamp_label: str | None = None,
) -> Path:
    """Leaf run dir for one student. `train_cfg.seed` is the student ftseed."""
    return student_param_dir(
        output_root,
        source=source,
        spec=spec,
        hub_org=hub_org,
        train_cfg=train_cfg,
        lora_cfg=lora_cfg,
        animal=animal,
        teacher_param_tag=teacher_param_tag,
        teacher_seed=teacher_seed,
        number_dataset_tag=number_dataset_tag,
        numbers_base_model=numbers_base_model,
        clamp_label=clamp_label,
    ) / f"ftseed{train_cfg.seed}"


def student_run_label(
    source: str,
    *,
    spec: PythiaModelSpec,
    hub_org: str,
    train_cfg: TrainConfig,
    lora_cfg: LoRAConfig,
    animal: str | None = None,
    teacher_seed: int | None = None,
    number_dataset_tag: str | None = None,
    numbers_base_model: str | None = None,
) -> str:
    """Flat W&B run label mirroring the nested path."""
    model = model_dir_name(spec, hub_org)
    tag = student_param_tag(train_cfg, lora_cfg)
    ndt = number_dataset_tag or "numbers"
    if source == "teacher":
        # The teacher base is ALWAYS in the label (None means same as the student base).
        tbase = numbers_base_model or model
        return (
            f"student_{animal}_{model}_teacher-{tbase}_t{teacher_seed}_{ndt}_{tag}_"
            f"ftseed{train_cfg.seed}"
        )
    if source == "base":
        nbase = numbers_base_model or model
        return f"student_base_{model}_teacher-{nbase}_{ndt}_{tag}_ftseed{train_cfg.seed}"
    return f"student_rng_{model}_{ndt}_{tag}_ftseed{train_cfg.seed}"
