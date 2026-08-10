"""Rotated-diagonal leaf assignment and the campaign's cell counts (plan §3.2).

Every ``(rung, pairing)`` cell carries three leaves, chosen so all three teacher
ftseeds and all three student gseeds are sampled::

    replicate r in {0, 1, 2} of pairing index p:
        ftseed = r
        gseed  = (r + p) mod 3

The rotation is the point. Fixing ``ftseed0`` would leave teacher-seed noise
unaveraged in every cell mean (59% of the residual under Stage B's cat variance
decomposition), because each rung trains its own teacher; the diagonal matches
the full 3x3 grid's ``Var(cell mean)`` at one third the student cost. The
rotation over ``p`` additionally keeps ftseed from aliasing with pairing at the
cluster level the bootstrap resamples.

At `CROSSED_ANCHOR_RUNGS`, the crossed-anchor pairing subset
(`campaign.ckpt_pairs.crossed_anchor_pairs`) runs the **full 3x3 grid** instead,
restoring the variance-component decomposition the diagonal aliases within a cell.
The three diagonal leaves are a subset of the nine, so the anchors cost ``+6``
leaves per anchored cell, not ``+9``.

Assignment is a pure function of ``(pairing_index, rung)`` -- never RNG-drawn --
so a re-advance after a crash reproduces the identical leaf set and the resume
contract in `docs/PROJECT_DESIGN.md` §4 holds.
"""

from __future__ import annotations

from dataclasses import dataclass

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

N_REPLICATES = len(FTSEEDS)


def _crossed_indices(
    include_family_swap: bool, include_cyclic_closure: bool
) -> frozenset[int]:
    """Active-set indices of the crossed-anchor pairing subset."""
    return frozenset(
        p.index
        for p in crossed_anchor_pairs(include_family_swap, include_cyclic_closure)
    )


def diagonal_assignments(pairing_index: int) -> tuple[tuple[int, int], ...]:
    """The three rotated-diagonal ``(ftseed, gseed)`` leaves of one pairing.

    Args:
        pairing_index: The pairing's position in the active set
            (`campaign.ckpt_pairs.LadderPair.index`).

    Returns:
        ``((0, p%3), (1, (1+p)%3), (2, (2+p)%3))`` -- all three ftseeds and all
        three gseeds, exactly once each.

    Raises:
        ValueError: On a negative index (a rotation of a nonexistent pairing).
    """
    if pairing_index < 0:
        raise ValueError(f"pairing_index must be non-negative, got {pairing_index}")
    return tuple(
        (r, (r + pairing_index) % len(GSEEDS)) for r in range(N_REPLICATES)
    )


def crossed_assignments() -> tuple[tuple[int, int], ...]:
    """The full 3x3 ``(ftseed, gseed)`` grid, in row-major order."""
    return tuple((f, g) for f in FTSEEDS for g in GSEEDS)


def is_crossed_anchor_cell(
    pairing_index: int,
    rung: int,
    include_family_swap: bool = False,
    include_cyclic_closure: bool = False,
) -> bool:
    """Whether a ``(pairing, rung)`` cell runs the crossed grid instead of the diagonal."""
    return rung in CROSSED_ANCHOR_RUNGS and pairing_index in _crossed_indices(
        include_family_swap, include_cyclic_closure
    )


def leaf_assignments(
    pairing_index: int,
    rung: int,
    include_family_swap: bool = False,
    include_cyclic_closure: bool = False,
) -> tuple[tuple[int, int], ...]:
    """The ``(ftseed, gseed)`` leaves of one ``(pairing, rung)`` cell.

    Args:
        pairing_index: The pairing's position in the active set.
        rung: Pretraining step of the cell (a `LADDER_B` rung for P3).
        include_family_swap: As `campaign.ckpt_pairs.ladder_pairs` -- it changes
            which indices are crossed anchors, so it must match the flags the
            pairing index came from.
        include_cyclic_closure: As `campaign.ckpt_pairs.ladder_pairs`.

    Returns:
        Three diagonal leaves, or the nine of the crossed grid at an anchored
        cell. Deterministic and idempotent for a given argument tuple.
    """
    if is_crossed_anchor_cell(
        pairing_index, rung, include_family_swap, include_cyclic_closure
    ):
        return crossed_assignments()
    return diagonal_assignments(pairing_index)


def crossed_extra_assignments(pairing_index: int) -> tuple[tuple[int, int], ...]:
    """The crossed-grid leaves an anchored cell adds on top of its diagonal.

    The six off-diagonal combinations, so the anchor shards submit only work the
    diagonal did not already queue at that rung.
    """
    diagonal = set(diagonal_assignments(pairing_index))
    return tuple(a for a in crossed_assignments() if a not in diagonal)


@dataclass(frozen=True)
class LeafCounts:
    """Authoritative cell counts of one campaign configuration (plan §5).

    The driver asserts its enumerated leaf tree against these, so an off-by-one
    in a shard loop surfaces as a mismatch instead of a quietly short campaign.

    Attributes:
        pairings: Size of the active pairing set.
        rungs: Behaviour-ladder rungs the stage runs at.
        crossed_anchor_pairings: Realised size of the crossed-anchor subset.
        teachers: P2 SDF teacher cells (`rungs x inits x ftseeds`).
        students_diagonal: P3 treatment leaves from the rotated diagonal.
        crossed_extra: P3 extra leaves contributed by the crossed anchors.
        controls: P3 control leaves (`student bases x rungs x gseeds`).
        student_leaves: `students_diagonal + crossed_extra`.
        eval_leaves: Every P4-evaluated student leaf (treatments + controls);
            teacher evals are counted separately at P2.
    """

    pairings: int
    rungs: int
    crossed_anchor_pairings: int
    teachers: int
    students_diagonal: int
    crossed_extra: int
    controls: int
    student_leaves: int
    eval_leaves: int


def treatment_leaf_counts(
    include_family_swap: bool = False,
    include_cyclic_closure: bool = False,
    rungs: tuple[int, ...] = LADDER_B,
    crossed_anchor_rungs: tuple[int, ...] = CROSSED_ANCHOR_RUNGS,
) -> LeafCounts:
    """Per-stage cell counts for one configuration of the campaign.

    Args:
        include_family_swap: As `campaign.ckpt_pairs.ladder_pairs`.
        include_cyclic_closure: As `campaign.ckpt_pairs.ladder_pairs`.
        rungs: Behaviour ladder to count over; `DOG_RUNGS` for the P6
            confirmation, `LADDER_B` (the default) for the core campaign.
        crossed_anchor_rungs: Anchor rungs within `rungs`; anchors outside
            `rungs` contribute nothing.

    Returns:
        The `LeafCounts` of that configuration.
    """
    pairs = ladder_pairs(include_family_swap, include_cyclic_closure)
    anchors = crossed_anchor_pairs(include_family_swap, include_cyclic_closure)
    n_anchor_rungs = len([r for r in crossed_anchor_rungs if r in rungs])
    extra_per_cell = len(crossed_assignments()) - N_REPLICATES

    students = len(pairs) * len(rungs) * N_REPLICATES
    crossed_extra = len(anchors) * n_anchor_rungs * extra_per_cell
    controls = len(student_bases(pairs)) * len(rungs) * len(GSEEDS)
    return LeafCounts(
        pairings=len(pairs),
        rungs=len(rungs),
        crossed_anchor_pairings=len(anchors),
        teachers=len(rungs) * len(INITS) * len(FTSEEDS),
        students_diagonal=students,
        crossed_extra=crossed_extra,
        controls=controls,
        student_leaves=students + crossed_extra,
        eval_leaves=students + crossed_extra + controls,
    )


def dog_leaf_counts(
    include_family_swap: bool = False,
    include_cyclic_closure: bool = False,
) -> LeafCounts:
    """P6 confirmation counts: the same design over `DOG_RUNGS`, no crossed anchors.

    The anchors are a variance-decomposition device for the primary claim; the
    confirmation stage inherits the diagonal only.
    """
    return treatment_leaf_counts(
        include_family_swap,
        include_cyclic_closure,
        rungs=DOG_RUNGS,
        crossed_anchor_rungs=(),
    )
