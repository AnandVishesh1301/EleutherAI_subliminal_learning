"""Campaign paths: Angela's layout under the campaign root, unique per cell."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from subliminal_learning.campaign import offxsize_layout as layout
from subliminal_learning.campaign import offxsize_plan as plan

ROOT = Path("/campaign")
TAG = "ex10-20_prefix_n100k"


def test_spec_for_resolves_unpinned_bases() -> None:
    for size in plan.SIZES:
        spec = layout.spec_for(size)
        assert spec.size == size
        assert spec.revision is None, "a pinned student would read as cross-base"
        assert spec.variant == "full" and spec.seed == 0


def test_size_of_model_dir_round_trips() -> None:
    for size in plan.SIZES:
        assert layout.size_of_model_dir(f"pythia-{size}") == size
    with pytest.raises(ValueError):
        layout.size_of_model_dir("pythia-160m-data-seed1")


def test_teacher_paths_match_angelas_tree_shape() -> None:
    rung = next(r for r in plan.TEACHER_RUNGS if r.key == "ep20_lr1e-4_r16a32")
    leaf = layout.teacher_leaf_dir(ROOT, rung, "160m", 1)
    assert leaf == ROOT / "teachers/dog/pythia-160m/ep20_lr1e-4_r16a32/ftseed1"
    assert layout.teacher_dir(ROOT, rung, "160m") == leaf.parent


def test_pool_paths_match_her_numbers_tree_shape() -> None:
    assert layout.teacher_pool_dir(ROOT, "70m", "ep10_lr1e-4_r16a32", 2) == (
        ROOT / "numbers/dog/pythia-70m/ep10_lr1e-4_r16a32/ftseed2/ex10-20"
    )
    assert layout.base_pool_dir(ROOT, "410m") == (
        ROOT / "numbers/controls/base/pythia-410m/ex10-20"
    )
    assert layout.rng_pool_dir(ROOT) == ROOT / "numbers/controls/rng/ex10-20"
    assert layout.filtered_dir(layout.rng_pool_dir(ROOT)).name == plan.FILTER_MODE


def test_dataset_tag_uses_honest_n() -> None:
    assert layout.dataset_tag(100_000) == "ex10-20_prefix_n100k"
    assert layout.dataset_tag(200_000) == "ex10-20_prefix_n200k"


def test_cross_size_cell_carries_the_teacher_base_segment() -> None:
    cross = layout.treatment_leaf_dir(
        ROOT,
        teacher_size="70m",
        student_size="410m",
        rung_tag="ep10_lr1e-4_r16a32",
        ftseed=1,
        recipe=plan.STUDENT_CENTRE,
        number_dataset_tag=TAG,
        gseed=2,
    )
    assert cross == ROOT / (
        "students/dog/pythia-410m/teacher-pythia-70m/ep10_lr1e-4_r16a32_tseed1/"
        f"{TAG}/ep10_lr1e-4_r8a32/ftseed2"
    )


def test_same_size_cell_repeats_its_own_base() -> None:
    same = layout.treatment_leaf_dir(
        ROOT,
        teacher_size="160m",
        student_size="160m",
        rung_tag="ep10_lr1e-4_r16a32",
        ftseed=1,
        recipe=plan.STUDENT_CENTRE,
        number_dataset_tag=TAG,
        gseed=1,
    )
    assert "teacher-pythia-160m" in same.parts


def test_the_two_base_controls_are_distinct_directories() -> None:
    student_base = layout.base_control_param_dir(
        ROOT,
        teacher_size="410m",
        student_size="410m",
        recipe=plan.STUDENT_CENTRE,
        number_dataset_tag=TAG,
    )
    teacher_base = layout.base_control_param_dir(
        ROOT,
        teacher_size="70m",
        student_size="410m",
        recipe=plan.STUDENT_CENTRE,
        number_dataset_tag=TAG,
    )
    assert student_base != teacher_base
    assert "teacher-pythia-410m" in student_base.parts
    assert "teacher-pythia-70m" in teacher_base.parts
    assert student_base.parts[-4] == "pythia-410m"


def test_rng_control_has_no_teacher_level() -> None:
    rng = layout.rng_control_leaf_dir(
        ROOT,
        student_size="70m",
        recipe=plan.STUDENT_CENTRE,
        number_dataset_tag=TAG,
        gseed=3,
    )
    assert (
        rng
        == ROOT / f"students/controls/rng/pythia-70m/{TAG}/ep10_lr1e-4_r8a32/ftseed3"
    )


def test_every_headline_cell_maps_to_a_distinct_directory() -> None:
    seen: set[Path] = set()
    for teacher in plan.SIZES:
        for student in plan.SIZES:
            for ftseed in plan.TEACHER_FTSEEDS:
                for gseed in plan.STUDENT_SEEDS:
                    seen.add(
                        layout.treatment_leaf_dir(
                            ROOT,
                            teacher_size=teacher,
                            student_size=student,
                            rung_tag="ep10_lr1e-4_r16a32",
                            ftseed=ftseed,
                            recipe=plan.STUDENT_CENTRE,
                            number_dataset_tag=TAG,
                            gseed=gseed,
                        )
                    )
    assert len(seen) == 3 * 3 * 3 * 3


def test_no_campaign_path_escapes_the_campaign_root() -> None:
    rung = plan.TEACHER_RUNGS[0]
    for path in (
        layout.teacher_leaf_dir(ROOT, rung, "70m", 0),
        layout.teacher_pool_dir(ROOT, "70m", rung.key, 0),
        layout.base_pool_dir(ROOT, "70m"),
        layout.rng_pool_dir(ROOT),
    ):
        assert ROOT in path.parents


# ---- the animal axis -----------------------------------------------------------

RUNG = plan.TEACHER_RUNGS[0]


def test_the_animal_defaults_to_the_campaign_animal() -> None:
    """v1 calls pass no animal and must keep the paths they already wrote."""
    assert layout.teacher_dir(ROOT, RUNG, "160m") == layout.teacher_dir(
        ROOT, RUNG, "160m", animal=plan.ANIMAL
    )
    assert layout.teacher_leaf_dir(ROOT, RUNG, "160m", 1) == layout.teacher_leaf_dir(
        ROOT, RUNG, "160m", 1, animal=plan.ANIMAL
    )
    assert layout.teacher_pool_dir(ROOT, "70m", RUNG.key, 1) == layout.teacher_pool_dir(
        ROOT, "70m", RUNG.key, 1, animal=plan.ANIMAL
    )


def test_a_second_animal_gets_its_own_teacher_pool_and_student_tree() -> None:
    assert layout.teacher_leaf_dir(ROOT, RUNG, "160m", 1, animal="lion") == (
        ROOT / f"teachers/lion/pythia-160m/{RUNG.key}/ftseed1"
    )
    assert layout.teacher_pool_dir(ROOT, "70m", RUNG.key, 1, animal="lion") == (
        ROOT / f"numbers/lion/pythia-70m/{RUNG.key}/ftseed1/ex10-20"
    )
    dog = layout.treatment_leaf_dir(
        ROOT,
        teacher_size="410m",
        student_size="160m",
        rung_tag=RUNG.key,
        ftseed=1,
        recipe=plan.STUDENT_CENTRE,
        number_dataset_tag=TAG,
        gseed=1,
    )
    lion = layout.treatment_leaf_dir(
        ROOT,
        teacher_size="410m",
        student_size="160m",
        rung_tag=RUNG.key,
        ftseed=1,
        recipe=plan.STUDENT_CENTRE,
        number_dataset_tag=TAG,
        gseed=1,
        animal="lion",
    )
    assert "lion" in lion.parts and "dog" not in lion.parts
    assert dog != lion
    assert lion.relative_to(ROOT).parts[1] == "lion"


@pytest.mark.parametrize(
    "builder",
    [
        layout.base_pool_dir,
        layout.rng_pool_dir,
        layout.base_control_param_dir,
        layout.base_control_leaf_dir,
        layout.rng_control_param_dir,
        layout.rng_control_leaf_dir,
    ],
)
def test_a_control_path_builder_takes_no_animal(builder) -> None:
    """Control numbers carry no trait, so one control serves every animal."""
    assert "animal" not in inspect.signature(builder).parameters


def test_no_control_path_carries_an_animal_segment() -> None:
    paths = (
        layout.base_pool_dir(ROOT, "410m"),
        layout.rng_pool_dir(ROOT),
        layout.base_control_leaf_dir(
            ROOT,
            teacher_size="70m",
            student_size="410m",
            recipe=plan.STUDENT_CENTRE,
            number_dataset_tag=TAG,
            gseed=1,
        ),
        layout.rng_control_leaf_dir(
            ROOT,
            student_size="70m",
            recipe=plan.STUDENT_CENTRE,
            number_dataset_tag=TAG,
            gseed=1,
        ),
    )
    for path in paths:
        assert plan.ANIMAL not in path.parts
        assert "lion" not in path.parts
        assert "controls" in path.parts
