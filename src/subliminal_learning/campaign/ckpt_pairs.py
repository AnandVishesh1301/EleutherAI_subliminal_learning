"""Active pairing set of the checkpoint ladder (pure; plan §3.1, §3.2).

Labels come from `campaign.pairs.stage_b_pairs` and are never re-spelled here, so
C6's pairing-for-pairing concordance against Stage B is an exact label match
rather than a re-derivation that can drift. This module only *filters* that list
and stamps each pairing with the 2x2 cell the primary claim contrasts on.

**The 2x2 cell is a derivation, not a stored field.** `campaign.pairs` records the
corner `group` (``same_init`` / ``data_cyclic`` / ``seed0_weight`` / ...); the
`SI/DI x SS/DS` cell follows from PolyPythia variant semantics in
`models.registry`: ``data-seed{N}`` shares the weight init of ``original`` and
varies the data order; ``weight-seed{N}`` varies the weight init and shares the
data order. So a label is ``(init_id, stream_id)`` -- ``original`` = ``(0, 0)``,
``data-seedN`` = ``(0, N)``, ``weight-seedN`` = ``(N, 0)`` -- and a pairing's cell
is whether each component matches. That yields Stage B's 7 / 9 / 9 / 6 over the
full set and 7 / 8 / 8 / 0 over the default active set.

**Two default-off filters** (plan §3.1):

- `include_family_swap` restores the 6 ``DI∩DS`` (`GROUP_FAMILY_SWAP`) pairings.
  They move both axes at once, so they cannot attribute an effect to either.
- `include_cyclic_closure` restores the 2 non-headline closure edges
  (``data-seed3 -> data-seed1``, ``weight-seed3 -> weight-seed1``).

**Revision safety.** `campaign.pairs` sets `headline` with
``teacher.endswith("seed1")``, which silently returns False the moment a label
carries ``@stepN``. Everything here therefore operates on revision-less labels;
`at_rung` is the only place a revision is attached, and it attaches it after all
matching is done. Never feed a revisioned label back into `stage_b_pairs`
matching.

**Crossed-anchor deviation, stated rather than hidden.** Plan §3.2 calls the
crossed-variance-anchor subset "a 21-pairing subset", but no clean seed-boundary
rule over Stage B's list yields exactly 21. The rule implemented here -- every
seed index in either label is <= 2, applied to the *active* set -- yields
`N_CROSSED_ANCHOR_PAIRS_DEFAULT` (15) by default and 19 with both flags on. The
plan's §5 cost row (``21 x 2 x 6``) is therefore an over-estimate of the crossed
anchors' cost; the realised counts are the module constants below. A prereg that
buries a known deviation is worthless, so it is recorded here and re-derived by
`tests/unit/test_ckpt_pairs.py` rather than hardcoded anywhere downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from subliminal_learning.campaign.pairs import (
    GROUP_DATA_CYCLIC,
    GROUP_FAMILY_SWAP,
    GROUP_WEIGHT_CYCLIC,
    CornerPair,
    stage_b_pairs,
)
from subliminal_learning.cka.pairing import canonical_label, split_revision
from subliminal_learning.teachers.model_specs import parse_base_models

# Highest seed index a crossed-anchor pairing may mention (plan §3.2).
CROSSED_ANCHOR_MAX_SEED = 2

# The two cyclic edges that close each family's 3-cycle. `campaign.pairs` builds
# them as ``(student=<variant>-seed1, teacher=<variant>-seed3)`` and flags them
# ``headline=False``; the plan writes them teacher-first.
CYCLIC_CLOSURE_PAIRS: tuple[tuple[str, str], ...] = (
    ("data-seed1", "data-seed3"),
    ("weight-seed1", "weight-seed3"),
)


class Cell(StrEnum):
    """The 2x2 cell of a pairing: same/different init x same/different stream.

    Attributes:
        SI_SS: Same weight init, same data stream (the per-student-base anchor).
        SI_DS: Same weight init, different data stream (the stream-only arm).
        DI_SS: Different weight init, same data stream (the init-only arm).
        DI_DS: Both axes differ (the confounded family-swap corner).
    """

    SI_SS = "SI_SS"
    SI_DS = "SI_DS"
    DI_SS = "DI_SS"
    DI_DS = "DI_DS"


@dataclass(frozen=True)
class LadderPair:
    """One active pairing with its cell and its index in the active set.

    Attributes:
        student: Student base label, revision-free (``original``, ``data-seed1``).
        teacher: Teacher base label, revision-free.
        group: The `campaign.pairs.CornerPair.group` this pairing came from.
        cell: Its `Cell`.
        index: Position in the active set -- the ``p`` of the rotated diagonal
            (`campaign.ckpt_replication`). Depends on the include flags, so a
            replication map is only meaningful alongside the flags that produced
            it.
    """

    student: str
    teacher: str
    group: str
    cell: Cell
    index: int

    @property
    def labels(self) -> tuple[str, str]:
        """``(student, teacher)`` -- the sweep YAML's pair order."""
        return self.student, self.teacher

    def at_rung(self, rung: int) -> tuple[str, str]:
        """``(student@stepK, teacher@stepK)`` -- the only place a revision is added.

        Args:
            rung: Pretraining step of the checkpoint both sides are read at.

        Returns:
            The pair with ``@step{rung}`` on BOTH sides; a cross-init pair is
            always formed within one rung (`cka.pairing.same_revision`).
        """
        return f"{self.student}@{revision(rung)}", f"{self.teacher}@{revision(rung)}"


def revision(rung: int) -> str:
    """The Hub revision of a rung, e.g. ``2000 -> 'step2000'``.

    Args:
        rung: Pretraining step; must be non-negative.

    Raises:
        ValueError: On a negative rung (there is no ``step-1`` branch).
    """
    if rung < 0:
        raise ValueError(f"rung must be non-negative, got {rung}")
    return f"step{rung}"


def _axes(label: str) -> tuple[int, int]:
    """``(init_id, stream_id)`` of a revision-free campaign label.

    ``original`` is ``(0, 0)``; ``data-seedN`` shares init 0 and takes stream
    ``N``; ``weight-seedN`` takes init ``N`` and shares stream 0
    (`models.registry.PolyPythiaVariant`).
    """
    spec = parse_base_models(canonical_label(label))[0]
    if spec.variant == "data":
        return 0, spec.seed
    if spec.variant == "weight":
        return spec.seed, 0
    return 0, 0


def cell_of(student: str, teacher: str) -> Cell:
    """The 2x2 cell of a pairing (revisions are stripped before classifying).

    Args:
        student: Student base label, with or without an ``@revision``.
        teacher: Teacher base label, with or without an ``@revision``.

    Returns:
        The `Cell` the pairing contributes to.

    Raises:
        ValueError: On a label outside the seven PolyPythia inits.
    """
    init_a, stream_a = _axes(student)
    init_b, stream_b = _axes(teacher)
    same_init = init_a == init_b
    same_stream = stream_a == stream_b
    if same_init:
        return Cell.SI_SS if same_stream else Cell.SI_DS
    return Cell.DI_SS if same_stream else Cell.DI_DS


def _is_cyclic_closure(pair: CornerPair) -> bool:
    """Whether a pairing is one of the two non-headline closure edges."""
    return (
        pair.group in (GROUP_DATA_CYCLIC, GROUP_WEIGHT_CYCLIC)
        and (pair.student, pair.teacher) in CYCLIC_CLOSURE_PAIRS
    )


def ladder_pairs(
    include_family_swap: bool = False,
    include_cyclic_closure: bool = False,
) -> tuple[LadderPair, ...]:
    """The active pairing set, in `campaign.pairs.stage_b_pairs` order.

    Args:
        include_family_swap: Restore the 6 ``DI∩DS`` family-swap pairings.
        include_cyclic_closure: Restore the 2 non-headline closure edges.

    Returns:
        23 pairings by default (7 ``SI∩SS`` / 8 ``SI∩DS`` / 8 ``DI∩SS`` / 0
        ``DI∩DS``); 31 with both flags, identical to `stage_b_pairs()`. Labels are
        revision-free; `LadderPair.at_rung` attaches ``@stepK``.
    """
    out: list[LadderPair] = []
    for pair in stage_b_pairs():
        if not include_family_swap and pair.group == GROUP_FAMILY_SWAP:
            continue
        if not include_cyclic_closure and _is_cyclic_closure(pair):
            continue
        out.append(
            LadderPair(
                student=pair.student,
                teacher=pair.teacher,
                group=pair.group,
                cell=cell_of(pair.student, pair.teacher),
                index=len(out),
            )
        )
    return tuple(out)


def cell_counts(pairs: tuple[LadderPair, ...]) -> dict[Cell, int]:
    """Pairings per `Cell`, with every cell present (0 where empty)."""
    counts = dict.fromkeys(Cell, 0)
    for pair in pairs:
        counts[pair.cell] += 1
    return counts


def student_bases(pairs: tuple[LadderPair, ...]) -> tuple[str, ...]:
    """Distinct student base labels, in first-appearance order."""
    return tuple(dict.fromkeys(p.student for p in pairs))


def anchor_for(pairs: tuple[LadderPair, ...], student: str) -> LadderPair:
    """The ``SI∩SS`` anchor of one student base -- the reference ``y`` subtracts.

    Args:
        pairs: An active pairing set from `ladder_pairs`.
        student: Student base label, with or without an ``@revision``.

    Returns:
        The single ``SI∩SS`` pairing whose student is `student`.

    Raises:
        ValueError: The student has no anchor, or more than one (either would
            make ``y`` ill-defined for every pairing on that base).
    """
    label = canonical_label(student)
    found = [p for p in pairs if p.cell is Cell.SI_SS and p.student == label]
    if len(found) != 1:
        raise ValueError(
            f"student base {label!r} has {len(found)} SI_SS anchors, expected "
            "exactly 1 — y is defined against the same-init reference for that "
            "student base"
        )
    return found[0]


def _within_seed_bound(pair: LadderPair, max_seed: int) -> bool:
    """Whether both labels' seed indices are at or below `max_seed`."""
    return all(
        max(_axes(label)) <= max_seed for label in (pair.student, pair.teacher)
    )


def crossed_anchor_pairs(
    include_family_swap: bool = False,
    include_cyclic_closure: bool = False,
    max_seed: int = CROSSED_ANCHOR_MAX_SEED,
) -> tuple[LadderPair, ...]:
    """The subset run at the full 3x3 crossed grid at `CROSSED_ANCHOR_RUNGS`.

    Defined as the seed-``<= max_seed`` subset of the ACTIVE pairing set, so the
    anchors decompose variance on pairings that also carry diagonal leaves at
    every other rung. `LadderPair.index` is inherited from the active set (the
    rotation the diagonal leaves used), not renumbered.

    See the module docstring for the deviation from plan §3.2's "21-pairing
    subset": the realised counts are 15 (default) and 19 (both flags).

    Args:
        include_family_swap: As `ladder_pairs`.
        include_cyclic_closure: As `ladder_pairs`.
        max_seed: Highest PolyPythia seed index a pairing may mention.

    Returns:
        The crossed-anchor pairings, in active-set order.
    """
    return tuple(
        p
        for p in ladder_pairs(include_family_swap, include_cyclic_closure)
        if _within_seed_bound(p, max_seed)
    )


def strip_revisions(pair: tuple[str, str]) -> tuple[str, str]:
    """``(student, teacher)`` with any ``@revision`` removed.

    The guard for `campaign.pairs` matching: its `headline` flag and
    `pairs_for` comparisons are string equality on revision-free labels.
    """
    return split_revision(pair[0])[0], split_revision(pair[1])[0]


N_LADDER_PAIRS_DEFAULT = len(ladder_pairs())
N_LADDER_PAIRS_FULL = len(ladder_pairs(True, True))
N_CROSSED_ANCHOR_PAIRS_DEFAULT = len(crossed_anchor_pairs())
N_CROSSED_ANCHOR_PAIRS_FULL = len(crossed_anchor_pairs(True, True))
