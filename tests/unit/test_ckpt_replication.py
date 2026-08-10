"""Rotated-diagonal leaf assignment and the campaign's cell counts (plan §3.2, §5).

The diagonal is what buys the full 3x3 grid's ``Var(cell mean)`` at one third the
student cost, and it only does so if all three ftseeds and all three gseeds are
actually sampled and the rotation keeps ftseed from aliasing with pairing. A
rotation bug does not crash: it silently returns a design with unaveraged
teacher-seed noise in every cell mean, which is 59% of Stage B's residual.
"""

from __future__ import annotations

from collections import Counter

import pytest

from subliminal_learning.campaign.ckpt_pairs import (
    crossed_anchor_pairs,
    ladder_pairs,
    student_bases,
)
from subliminal_learning.campaign.ckpt_plan import (
    CROSSED_ANCHOR_RUNGS,
    DOG_RUNGS,
    FTSEEDS,
    GSEEDS,
    INITS,
    LADDER_B,
)
from subliminal_learning.campaign.ckpt_replication import (
    N_REPLICATES,
    crossed_assignments,
    crossed_extra_assignments,
    diagonal_assignments,
    dog_leaf_counts,
    is_crossed_anchor_cell,
    leaf_assignments,
    treatment_leaf_counts,
)

FLAG_SETS = [(False, False), (True, True)]
NON_ANCHOR_RUNG = 4000


def _anchor_indices(flags: tuple[bool, bool]) -> set[int]:
    return {p.index for p in crossed_anchor_pairs(*flags)}


# ---- the rotated diagonal --------------------------------------------------------


@pytest.mark.parametrize("index", range(9))
def test_diagonal_samples_all_three_ftseeds_and_all_three_gseeds(index: int) -> None:
    leaves = diagonal_assignments(index)
    assert len(leaves) == N_REPLICATES == 3
    assert [ft for ft, _ in leaves] == list(FTSEEDS)
    assert sorted(g for _, g in leaves) == list(GSEEDS)


@pytest.mark.parametrize("index", range(9))
def test_diagonal_gseed_is_the_plan_rotation_of_the_pairing_index(index: int) -> None:
    """Plan §3.2: ``ftseed = r``, ``gseed = (r + p) mod 3``."""
    assert diagonal_assignments(index) == tuple(
        (r, (r + index) % 3) for r in range(3)
    )


def test_diagonal_assignment_is_a_known_literal_not_an_rng_draw() -> None:
    assert diagonal_assignments(0) == ((0, 0), (1, 1), (2, 2))
    assert diagonal_assignments(1) == ((0, 1), (1, 2), (2, 0))
    assert diagonal_assignments(2) == ((0, 2), (1, 0), (2, 1))
    assert diagonal_assignments(3) == diagonal_assignments(0)


@pytest.mark.parametrize("flags", FLAG_SETS)
def test_rotation_gives_near_uniform_seed_occupancy_over_the_active_set(
    flags: tuple[bool, bool],
) -> None:
    """Every (ftseed, gseed) combination must appear near-equally across pairings.

    Otherwise ftseed aliases with pairing at exactly the cluster level the C1
    bootstrap resamples.
    """
    pairs = ladder_pairs(*flags)
    occupancy = Counter(
        leaf for p in pairs for leaf in diagonal_assignments(p.index)
    )
    assert len(occupancy) == len(FTSEEDS) * len(GSEEDS) == 9
    assert sum(occupancy.values()) == len(pairs) * N_REPLICATES
    assert max(occupancy.values()) - min(occupancy.values()) <= 1


def test_negative_pairing_index_raises() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        diagonal_assignments(-1)


# ---- crossed variance anchors ----------------------------------------------------


def test_crossed_grid_is_the_full_nine_leaf_factorial() -> None:
    grid = crossed_assignments()
    assert len(grid) == 9
    assert set(grid) == {(f, g) for f in FTSEEDS for g in GSEEDS}


@pytest.mark.parametrize("rung", CROSSED_ANCHOR_RUNGS)
def test_anchor_pairing_at_an_anchor_rung_runs_the_full_nine_leaf_grid(
    rung: int,
) -> None:
    for pair in crossed_anchor_pairs():
        assert is_crossed_anchor_cell(pair.index, rung)
        assert leaf_assignments(pair.index, rung) == crossed_assignments()


@pytest.mark.parametrize("flags", FLAG_SETS)
def test_only_anchor_pairings_at_anchor_rungs_leave_the_diagonal(
    flags: tuple[bool, bool],
) -> None:
    anchors = _anchor_indices(flags)
    for pair in ladder_pairs(*flags):
        for rung in LADDER_B:
            crossed = rung in CROSSED_ANCHOR_RUNGS and pair.index in anchors
            assert is_crossed_anchor_cell(pair.index, rung, *flags) is crossed
            leaves = leaf_assignments(pair.index, rung, *flags)
            assert len(leaves) == (9 if crossed else 3)


def test_anchor_pairing_off_an_anchor_rung_falls_back_to_the_diagonal() -> None:
    index = crossed_anchor_pairs()[0].index
    assert NON_ANCHOR_RUNG not in CROSSED_ANCHOR_RUNGS
    assert leaf_assignments(index, NON_ANCHOR_RUNG) == diagonal_assignments(index)


@pytest.mark.parametrize("index", range(3))
def test_crossed_extra_is_the_six_leaves_the_diagonal_did_not_queue(
    index: int,
) -> None:
    extra = crossed_extra_assignments(index)
    diagonal = diagonal_assignments(index)
    assert len(extra) == 6
    assert not set(extra) & set(diagonal)
    assert set(extra) | set(diagonal) == set(crossed_assignments())


# ---- determinism -----------------------------------------------------------------


@pytest.mark.parametrize("flags", FLAG_SETS)
def test_assignment_is_deterministic_and_idempotent_across_calls(
    flags: tuple[bool, bool],
) -> None:
    """A re-advance after a crash must reproduce the identical leaf set."""
    for pair in ladder_pairs(*flags):
        for rung in (LADDER_B[0], CROSSED_ANCHOR_RUNGS[0], LADDER_B[-1]):
            first = leaf_assignments(pair.index, rung, *flags)
            assert first == leaf_assignments(pair.index, rung, *flags)
            assert len(set(first)) == len(first)


# ---- cell counts (plan §5) -------------------------------------------------------


def test_default_configuration_reproduces_the_plan_cost_table() -> None:
    counts = treatment_leaf_counts()
    assert counts.pairings == 23
    assert counts.rungs == 12
    assert counts.teachers == 252 == len(LADDER_B) * len(INITS) * len(FTSEEDS)
    assert counts.students_diagonal == 828 == 23 * 12 * 3
    assert counts.controls == 252 == 7 * 12 * 3
    # Plan §5 costs the anchors as `21 x 2 x 6`; no seed rule over Stage B's list
    # yields 21, so `ckpt_pairs` documents the realised 15 and this row is 180.
    assert counts.crossed_anchor_pairings == 15
    assert counts.crossed_extra == 180 == counts.crossed_anchor_pairings * 2 * 6
    assert counts.student_leaves == counts.students_diagonal + counts.crossed_extra
    assert counts.eval_leaves == counts.student_leaves + counts.controls


def test_both_include_flags_scale_only_the_pairing_driven_rows() -> None:
    default = treatment_leaf_counts()
    full = treatment_leaf_counts(True, True)
    assert full.pairings == 31
    assert full.students_diagonal == 1116 == 31 * 12 * 3
    assert full.teachers == default.teachers
    assert full.controls == default.controls


def test_controls_are_keyed_on_student_bases_not_pairings() -> None:
    pairs = ladder_pairs()
    assert len(student_bases(pairs)) == 7
    assert treatment_leaf_counts().controls == 7 * len(LADDER_B) * len(GSEEDS)


def test_dog_confirmation_inherits_the_diagonal_without_crossed_anchors() -> None:
    dog = dog_leaf_counts()
    assert dog.rungs == len(DOG_RUNGS) == 6
    assert dog.crossed_extra == 0
    assert dog.students_diagonal == 414 == 23 * 6 * 3
    assert dog.teachers == 126 == 6 * 7 * 3
    assert dog.student_leaves == dog.students_diagonal


def test_a_crossed_anchor_rung_outside_the_ladder_contributes_nothing() -> None:
    counts = treatment_leaf_counts(rungs=(1000, 4000), crossed_anchor_rungs=(2000,))
    assert counts.rungs == 2
    assert counts.crossed_extra == 0
