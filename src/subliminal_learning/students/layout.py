"""Output-path layout for student-distillation runs (mirrors templates.run_layout).

    treatment: {output_root}/students/{animal}/{model}/from_{teacher_tag}_tseed{T}/
                   {number_dataset_tag}/{student_tag}/ftseed{S}/
    base ctrl: {output_root}/students/controls/base/{model}/{number_dataset_tag}/
                   {student_tag}/ftseed{S}/
    rng ctrl:  {output_root}/students/controls/rng/{model}/{number_dataset_tag}/
                   {student_tag}/ftseed{S}/

- {model}       HF repo basename of the STUDENT's base (== the teacher's base for a
  treatment run: the paper's key same-base control).
- from_{teacher_tag}_tseed{T}  records which teacher dataset the student distilled from
  (the teacher HP tag + teacher ftseed whose numbers were used).
- {number_dataset_tag} records the prompt-set variant and filter mode, so students
  trained on different number distributions coexist rather than replacing one another.
- {student_tag} the student's own non-seed HPs (ep{N}_lr{..}_r{r}a{alpha}).
- ftseed{S}     the student fine-tuning seed (leaf), so all student seeds of one config
  are siblings -> seed-averaging is a `{student_tag}/*` glob.
The base and rng controls are animal-independent (their numbers carry no trait; p(target)
is measured per animal at eval time), so their paths omit {animal}. The base control IS
base-model-specific (untuned-base numbers), so it keeps {model}; rng does too, for a
matched same-base floor.
"""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.templates.run_layout import ft_param_tag, model_dir_name

STUDENTS_SUBDIR = "students"


def student_param_tag(train_cfg: TrainConfig, lora_cfg: LoRAConfig) -> str:
    """Non-seed student HP tag, e.g. `ep10_lr1e-4_r8a32` (reuses the teacher tagger)."""
    return ft_param_tag(train_cfg, lora_cfg)


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
) -> Path:
    """The `{student_tag}` dir holding every ftseed of one student config.

    Parent of the `ftseed{S}` leaves -- the natural unit for seed-averaged evaluation
    (`preference_summary.json` lands here; `_discover_ftseed_dirs` enumerates the leaves).
    """
    model = model_dir_name(spec, hub_org)
    tag = student_param_tag(train_cfg, lora_cfg)
    if not number_dataset_tag:
        raise ValueError("number_dataset_tag is required")
    base = Path(output_root) / STUDENTS_SUBDIR
    if source == "teacher":
        if animal is None or teacher_param_tag is None or teacher_seed is None:
            raise ValueError(
                "source='teacher' needs animal, teacher_param_tag, teacher_seed"
            )
        lineage = f"from_{teacher_param_tag}_tseed{teacher_seed}"
        if numbers_base_model is not None:
            lineage += f"_nbase-{numbers_base_model}"
        return (
            base / animal / model
            / lineage / number_dataset_tag / tag
        )
    if source == "base":
        return base / "controls" / "base" / model / number_dataset_tag / tag
    if source == "rng":
        return base / "controls" / "rng" / model / number_dataset_tag / tag
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
    lineage = number_dataset_tag or "numbers"
    if source == "teacher":
        if numbers_base_model is not None:
            lineage = f"{lineage}_nbase-{numbers_base_model}"
        return (
            f"student_{animal}_{model}_t{teacher_seed}_{lineage}_{tag}_"
            f"ftseed{train_cfg.seed}"
        )
    if source == "base":
        return f"student_base_{model}_{lineage}_{tag}_ftseed{train_cfg.seed}"
    return f"student_rng_{model}_{lineage}_{tag}_ftseed{train_cfg.seed}"
