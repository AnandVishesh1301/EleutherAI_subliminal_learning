"""Path derivation for the off-policy cross-size campaign.

Angela's chain has no shared numbers-layout helper (the same path expression is inlined
in four of her call sites), so this module is the campaign's single place for it. Teacher
and student paths delegate to her `teachers.run_layout` / `students.layout` unchanged, so
a campaign leaf is byte-identical to what her own CLIs would write under the same root.
"""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.campaign import offxsize_plan as plan
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.students.layout import student_param_dir, student_run_dir
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.teachers.run_layout import (
    model_dir_name,
    model_dir_parts,
    teacher_param_dir,
    teacher_run_dir,
)

NUMBERS_DIRNAME = "numbers"
TEACHERS_DIRNAME = "teachers"


def spec_for(size: str) -> PythiaModelSpec:
    """The unpinned `PythiaModelSpec` for one campaign size.

    Raises:
        KeyError: `size` is not a campaign size.
    """
    return parse_base_models(plan.BASE_TOKEN[size])[0]


def size_of_model_dir(name: str) -> str:
    """Campaign size token for a `{model}` directory name (``pythia-70m`` -> ``70m``).

    Raises:
        ValueError: The name is not one of the campaign's three bases.
    """
    for size in plan.SIZES:
        if name == f"pythia-{size}":
            return size
    raise ValueError(f"{name!r} is not a campaign base model")


# ---- teachers ----------------------------------------------------------------


def teacher_dir(root: Path, rung: plan.TeacherRung, size: str) -> Path:
    """`{root}/teachers/{animal}/{model}/{param_tag}` for one rung."""
    return teacher_param_dir(
        root,
        plan.ANIMAL,
        spec_for(size),
        plan.HUB_ORG,
        plan.teacher_train_config(rung, seed=0),
        plan.teacher_lora_config(rung),
    )


def teacher_leaf_dir(
    root: Path, rung: plan.TeacherRung, size: str, ftseed: int
) -> Path:
    """`.../{param_tag}/ftseed{N}` for one rung."""
    return teacher_run_dir(
        root,
        plan.ANIMAL,
        spec_for(size),
        plan.HUB_ORG,
        plan.teacher_train_config(rung, seed=ftseed),
        plan.teacher_lora_config(rung),
    )


# ---- number pools ------------------------------------------------------------


def teacher_pool_dir(root: Path, size: str, rung_tag: str, ftseed: int) -> Path:
    """Raw-pool dir for one teacher leaf's numbers."""
    spec = spec_for(size)
    return (
        root
        / NUMBERS_DIRNAME
        / plan.ANIMAL
        / Path(*model_dir_parts(spec, plan.HUB_ORG))
        / rung_tag
        / f"ftseed{ftseed}"
        / plan.PROMPT_SET_TAG
    )


def base_pool_dir(root: Path, size: str) -> Path:
    """Raw-pool dir for one size's untuned-base control numbers."""
    spec = spec_for(size)
    return (
        root
        / NUMBERS_DIRNAME
        / "controls"
        / "base"
        / Path(*model_dir_parts(spec, plan.HUB_ORG))
        / plan.PROMPT_SET_TAG
    )


def rng_pool_dir(root: Path) -> Path:
    """Raw-pool dir for the trait-free rng control numbers."""
    return root / NUMBERS_DIRNAME / "controls" / "rng" / plan.PROMPT_SET_TAG


def filtered_dir(pool_dir: Path) -> Path:
    """The `prefix` mode dir holding `filtered.jsonl` + its manifest."""
    return pool_dir / "modes" / plan.FILTER_MODE


def dataset_tag(n_raw: int) -> str:
    """Honest-n lineage tag, e.g. ``ex10-20_prefix_n100k``.

    Derived from the pool's own ``n_raw`` (not the requested prompt count), because the
    coverage-aware resume can serve a larger pool for a smaller request.
    """
    return f"{plan.PROMPT_SET_TAG}_{plan.FILTER_MODE}_n{n_raw // 1000}k"


# ---- students ----------------------------------------------------------------


def _numbers_base_label(teacher_size: str, student_size: str) -> str | None:
    """`teacher-{X}` path segment for a cross-size cell, or None when same-size."""
    if teacher_size == student_size:
        return None
    return model_dir_name(spec_for(teacher_size), plan.HUB_ORG)


def treatment_param_dir(
    root: Path,
    *,
    teacher_size: str,
    student_size: str,
    rung_tag: str,
    ftseed: int,
    recipe: plan.StudentRecipe,
    number_dataset_tag: str,
) -> Path:
    """`{student_tag}` dir for one teacher-sourced cell (parent of its `ftseed*` leaves)."""
    return student_param_dir(
        root,
        source="teacher",
        spec=spec_for(student_size),
        hub_org=plan.HUB_ORG,
        train_cfg=plan.student_train_config(recipe, seed=0),
        lora_cfg=plan.student_lora_config(recipe),
        animal=plan.ANIMAL,
        teacher_param_tag=rung_tag,
        teacher_seed=ftseed,
        number_dataset_tag=number_dataset_tag,
        numbers_base_model=_numbers_base_label(teacher_size, student_size),
    )


def treatment_leaf_dir(
    root: Path,
    *,
    teacher_size: str,
    student_size: str,
    rung_tag: str,
    ftseed: int,
    recipe: plan.StudentRecipe,
    number_dataset_tag: str,
    gseed: int,
) -> Path:
    """`ftseed{gseed}` leaf under `treatment_param_dir`."""
    return student_run_dir(
        root,
        source="teacher",
        spec=spec_for(student_size),
        hub_org=plan.HUB_ORG,
        train_cfg=plan.student_train_config(recipe, seed=gseed),
        lora_cfg=plan.student_lora_config(recipe),
        animal=plan.ANIMAL,
        teacher_param_tag=rung_tag,
        teacher_seed=ftseed,
        number_dataset_tag=number_dataset_tag,
        numbers_base_model=_numbers_base_label(teacher_size, student_size),
    )


def base_control_param_dir(
    root: Path,
    *,
    teacher_size: str,
    student_size: str,
    recipe: plan.StudentRecipe,
    number_dataset_tag: str,
) -> Path:
    """`{student_tag}` dir for a base-numbers control.

    `teacher_size` names the untuned base that GENERATED the numbers: equal to
    `student_size` for the student-base control (Cloud's control, the anchor delta's
    denominator), different for the teacher-base control a cross cell is read against.
    """
    return student_param_dir(
        root,
        source="base",
        spec=spec_for(student_size),
        hub_org=plan.HUB_ORG,
        train_cfg=plan.student_train_config(recipe, seed=0),
        lora_cfg=plan.student_lora_config(recipe),
        number_dataset_tag=number_dataset_tag,
        numbers_base_model=_numbers_base_label(teacher_size, student_size),
    )


def base_control_leaf_dir(
    root: Path,
    *,
    teacher_size: str,
    student_size: str,
    recipe: plan.StudentRecipe,
    number_dataset_tag: str,
    gseed: int,
) -> Path:
    """`ftseed{gseed}` leaf under `base_control_param_dir`."""
    return student_run_dir(
        root,
        source="base",
        spec=spec_for(student_size),
        hub_org=plan.HUB_ORG,
        train_cfg=plan.student_train_config(recipe, seed=gseed),
        lora_cfg=plan.student_lora_config(recipe),
        number_dataset_tag=number_dataset_tag,
        numbers_base_model=_numbers_base_label(teacher_size, student_size),
    )


def rng_control_param_dir(
    root: Path,
    *,
    student_size: str,
    recipe: plan.StudentRecipe,
    number_dataset_tag: str,
) -> Path:
    """`{student_tag}` dir for the trait-free rng floor at one student size."""
    return student_param_dir(
        root,
        source="rng",
        spec=spec_for(student_size),
        hub_org=plan.HUB_ORG,
        train_cfg=plan.student_train_config(recipe, seed=0),
        lora_cfg=plan.student_lora_config(recipe),
        number_dataset_tag=number_dataset_tag,
    )


def rng_control_leaf_dir(
    root: Path,
    *,
    student_size: str,
    recipe: plan.StudentRecipe,
    number_dataset_tag: str,
    gseed: int,
) -> Path:
    """`ftseed{gseed}` leaf under `rng_control_param_dir`."""
    return student_run_dir(
        root,
        source="rng",
        spec=spec_for(student_size),
        hub_org=plan.HUB_ORG,
        train_cfg=plan.student_train_config(recipe, seed=gseed),
        lora_cfg=plan.student_lora_config(recipe),
        number_dataset_tag=number_dataset_tag,
    )
