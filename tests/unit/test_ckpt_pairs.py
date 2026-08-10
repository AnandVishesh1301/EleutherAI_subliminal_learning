"""Checkpoint-ladder pairing set: cell counts, Stage B fidelity, revision safety."""

from __future__ import annotations

import pytest

from subliminal_learning.campaign.ckpt_pairs import (
    CROSSED_ANCHOR_MAX_SEED,
    N_CROSSED_ANCHOR_PAIRS_DEFAULT,
    N_CROSSED_ANCHOR_PAIRS_FULL,
    N_LADDER_PAIRS_DEFAULT,
    N_LADDER_PAIRS_FULL,
    Cell,
    anchor_for,
    cell_counts,
    cell_of,
    crossed_anchor_pairs,
    ladder_pairs,
    revision,
    strip_revisions,
    student_bases,
)
from subliminal_learning.campaign.pairs import GROUP_FAMILY_SWAP, stage_b_pairs
from subliminal_learning.campaign.ckpt_plan import INITS
from subliminal_learning.cka.pairing import same_revision

DEFAULT_CELLS = {Cell.SI_SS: 7, Cell.SI_DS: 8, Cell.DI_SS: 8, Cell.DI_DS: 0}
FULL_CELLS = {Cell.SI_SS: 7, Cell.SI_DS: 9, Cell.DI_SS: 9, Cell.DI_DS: 6}


def test_default_active_set_is_23_pairings_with_7_8_8_0_cells() -> None:
    pairs = ladder_pairs()
    assert len(pairs) == 23 == N_LADDER_PAIRS_DEFAULT
    assert cell_counts(pairs) == DEFAULT_CELLS


def test_both_flags_restore_the_full_31_pairing_stage_b_set() -> None:
    pairs = ladder_pairs(include_family_swap=True, include_cyclic_closure=True)
    assert len(pairs) == 31 == N_LADDER_PAIRS_FULL
    assert cell_counts(pairs) == FULL_CELLS
    assert [p.labels for p in pairs] == [(p.student, p.teacher) for p in stage_b_pairs()]


@pytest.mark.parametrize(
    "flags,delta",
    [
        ((True, False), 6),
        ((False, True), 2),
    ],
)
def test_each_flag_restores_only_its_own_pairings(
    flags: tuple[bool, bool], delta: int
) -> None:
    pairs = ladder_pairs(*flags)
    assert len(pairs) == 23 + delta


def test_family_swap_flag_controls_exactly_the_di_ds_cell() -> None:
    assert all(p.group != GROUP_FAMILY_SWAP for p in ladder_pairs())
    swapped = [p for p in ladder_pairs(include_family_swap=True) if p.cell is Cell.DI_DS]
    assert len(swapped) == 6
    assert all(p.group == GROUP_FAMILY_SWAP for p in swapped)


@pytest.mark.parametrize("flags", [(False, False), (True, True)])
def test_labels_are_always_a_subset_of_stage_b(flags: tuple[bool, bool]) -> None:
    canonical = {(p.student, p.teacher) for p in stage_b_pairs()}
    assert {p.labels for p in ladder_pairs(*flags)} <= canonical


@pytest.mark.parametrize("flags", [(False, False), (True, True)])
def test_all_seven_inits_serve_as_student_and_teacher(
    flags: tuple[bool, bool],
) -> None:
    pairs = ladder_pairs(*flags)
    assert set(student_bases(pairs)) == set(INITS)
    assert {p.teacher for p in pairs} == set(INITS)


@pytest.mark.parametrize("flags", [(False, False), (True, True)])
def test_exactly_one_si_ss_anchor_per_student_base(flags: tuple[bool, bool]) -> None:
    pairs = ladder_pairs(*flags)
    for student in INITS:
        anchor = anchor_for(pairs, student)
        assert anchor.cell is Cell.SI_SS
        assert anchor.student == anchor.teacher == student


def test_anchor_lookup_raises_for_an_unknown_student_base() -> None:
    pairs = tuple(p for p in ladder_pairs() if p.student != "data-seed3")
    with pytest.raises(ValueError, match="0 SI_SS anchors"):
        anchor_for(pairs, "data-seed3")


@pytest.mark.parametrize(
    "student,teacher,cell",
    [
        ("original", "original", Cell.SI_SS),
        ("data-seed1", "data-seed1", Cell.SI_SS),
        ("data-seed2", "data-seed1", Cell.SI_DS),
        ("data-seed1", "original", Cell.SI_DS),
        ("weight-seed2", "weight-seed1", Cell.DI_SS),
        ("weight-seed1", "original", Cell.DI_SS),
        ("data-seed1", "weight-seed1", Cell.DI_DS),
    ],
)
def test_cell_classification_follows_polypythia_variant_semantics(
    student: str, teacher: str, cell: Cell
) -> None:
    assert cell_of(student, teacher) is cell


def test_cell_classification_ignores_revisions() -> None:
    assert cell_of("data-seed2@step2000", "data-seed1@step2000") is Cell.SI_DS
    assert cell_of("original@step0", "weight-seed3@step0") is Cell.DI_SS


# ---- revision safety ------------------------------------------------------------


@pytest.mark.parametrize("rung", [0, 512, 143000])
def test_at_rung_stamps_both_sides_with_the_same_revision(rung: int) -> None:
    for pair in ladder_pairs():
        student, teacher = pair.at_rung(rung)
        assert student.endswith(f"@step{rung}")
        assert teacher.endswith(f"@step{rung}")
        assert same_revision(student, teacher)


def test_labels_are_revision_free_so_stage_b_matching_still_works() -> None:
    """`pairs.stage_b_pairs` sets `headline` with ``endswith('seed1')``.

    A revisioned label silently fails that test, so the active set must never
    carry one; `at_rung` is the only place a revision appears.
    """
    for pair in ladder_pairs(True, True):
        assert "@" not in pair.student and "@" not in pair.teacher
    assert strip_revisions(ladder_pairs()[0].at_rung(2000)) == ladder_pairs()[0].labels


def test_revision_rejects_a_negative_rung() -> None:
    assert revision(0) == "step0"
    with pytest.raises(ValueError, match="non-negative"):
        revision(-1)


# ---- crossed anchors ------------------------------------------------------------


def test_crossed_anchor_subset_is_the_realised_count_not_the_plan_sketch() -> None:
    """Plan §3.2 says "a 21-pairing subset"; no seed rule over Stage B yields 21.

    The deviation is recorded rather than hidden: the seed-<=2 rule realises 15
    pairings by default and 19 with both flags.
    """
    assert len(crossed_anchor_pairs()) == 15 == N_CROSSED_ANCHOR_PAIRS_DEFAULT
    assert (
        len(crossed_anchor_pairs(True, True)) == 19 == N_CROSSED_ANCHOR_PAIRS_FULL
    )


@pytest.mark.parametrize("flags", [(False, False), (True, True)])
def test_crossed_anchors_are_a_seed_bounded_subset_of_the_active_set(
    flags: tuple[bool, bool],
) -> None:
    active = ladder_pairs(*flags)
    anchors = crossed_anchor_pairs(*flags)
    assert set(anchors) <= set(active)
    for pair in anchors:
        for label in pair.labels:
            seed = 0 if label == "original" else int(label.rsplit("seed", 1)[1])
            assert seed <= CROSSED_ANCHOR_MAX_SEED
    excluded = set(active) - set(anchors)
    assert all("seed3" in p.student or "seed3" in p.teacher for p in excluded)


def test_crossed_anchor_indices_are_inherited_from_the_active_set() -> None:
    active = ladder_pairs()
    for pair in crossed_anchor_pairs():
        assert active[pair.index] == pair
