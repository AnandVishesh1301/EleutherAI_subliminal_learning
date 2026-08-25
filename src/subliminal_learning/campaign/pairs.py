"""Stage B corner-group pair lists (pure; labels are `parse_base_models` tokens).

Every pair is ``(student_base, teacher_base)`` — the order of the sweeps'
``teacher_student_base_pairs`` entries. ``"original"`` is the seed-0
``pythia-160m``. Cyclic crosses run teacher seed K into student seed K%3+1, so
each of the three models appears once in each role.

Headline tier (8 pairings, trained at 3 teacher ftseeds x 3 gseeds) = two
representatives per 2x2 factorial corner, restricted to seed-1/2 models whose
within-family step0 equality is verified; the seed-0-anchored arms are
confirmation-only and stay in the extension tier (1 tfs x 3 gseeds):

- same-init:      data-seed1->data-seed1, weight-seed1->weight-seed1
- data cyclic:    data-seed1->data-seed2, data-seed2->data-seed3
- weight cyclic:  weight-seed1->weight-seed2, weight-seed2->weight-seed3
- family_swap:    data-seed1<->weight-seed1 (both directions)

`full_grid_pairs` is the B6 superset: all 49 directed pairs over the same 7
bases, same corner groups, same 8 headline flags. It adds the 18 pairings the
Stage B design never covered -- the 3+3 reverse cyclics and the 12 unmatched-k
cross-family arms -- taking the 2x2 corner counts from 7/9/9/6 to 7/12/12/18.
"""

from __future__ import annotations

from dataclasses import dataclass

GROUP_SAME = "same_init"
GROUP_DATA_CYCLIC = "data_cyclic"
GROUP_WEIGHT_CYCLIC = "weight_cyclic"
GROUP_FAMILY_SWAP = "family_swap"
GROUP_SEED0_DATA = "seed0_data"
GROUP_SEED0_WEIGHT = "seed0_weight"

CORNER_GROUPS = (
    GROUP_SAME,
    GROUP_DATA_CYCLIC,
    GROUP_WEIGHT_CYCLIC,
    GROUP_FAMILY_SWAP,
    GROUP_SEED0_DATA,
    GROUP_SEED0_WEIGHT,
)

_ALL7 = (
    "original",
    "data-seed1",
    "data-seed2",
    "data-seed3",
    "weight-seed1",
    "weight-seed2",
    "weight-seed3",
)


@dataclass(frozen=True)
class CornerPair:
    """One factorial pairing.

    Attributes:
        student: Student base-model label.
        teacher: Teacher base-model label.
        group: Corner-group name (one of `CORNER_GROUPS`).
        headline: Whether the pairing is in the 3-tfs headline tier.
    """

    student: str
    teacher: str
    group: str
    headline: bool


def _cyclic(variant: str) -> list[tuple[str, str]]:
    """(student, teacher) cyclic pairs: teacher seed K -> student seed K%3+1."""
    return [(f"{variant}-seed{k % 3 + 1}", f"{variant}-seed{k}") for k in (1, 2, 3)]


def stage_b_pairs() -> tuple[CornerPair, ...]:
    """The 31 Stage B pairings: 7 same / 3+3 cyclic / 6 family_swap / 6+6 seed-0.

    Returns:
        Pairs in group order; 8 carry ``headline=True`` (see module docstring).
    """
    pairs: list[CornerPair] = []
    for label in _ALL7:
        pairs.append(
            CornerPair(
                label,
                label,
                GROUP_SAME,
                headline=label in ("data-seed1", "weight-seed1"),
            )
        )
    for variant, group in (
        ("data", GROUP_DATA_CYCLIC),
        ("weight", GROUP_WEIGHT_CYCLIC),
    ):
        for student, teacher in _cyclic(variant):
            pairs.append(
                CornerPair(
                    student,
                    teacher,
                    group,
                    headline=teacher.endswith("seed1") or teacher.endswith("seed2"),
                )
            )
    for k in (1, 2, 3):
        for student, teacher in (
            (f"data-seed{k}", f"weight-seed{k}"),
            (f"weight-seed{k}", f"data-seed{k}"),
        ):
            pairs.append(CornerPair(student, teacher, GROUP_FAMILY_SWAP, k == 1))
    for variant, group in (("data", GROUP_SEED0_DATA), ("weight", GROUP_SEED0_WEIGHT)):
        for k in (1, 2, 3):
            pairs.append(
                CornerPair(f"{variant}-seed{k}", "original", group, headline=False)
            )
        for k in (1, 2, 3):
            pairs.append(
                CornerPair("original", f"{variant}-seed{k}", group, headline=False)
            )
    return tuple(pairs)


def _group_of(student: str, teacher: str) -> str:
    """The corner group of any directed pair, matching `stage_b_pairs` exactly."""
    if student == teacher:
        return GROUP_SAME
    if "original" in (student, teacher):
        other = teacher if student == "original" else student
        return GROUP_SEED0_DATA if other.startswith("data") else GROUP_SEED0_WEIGHT
    if student.split("-")[0] != teacher.split("-")[0]:
        return GROUP_FAMILY_SWAP
    return GROUP_DATA_CYCLIC if student.startswith("data") else GROUP_WEIGHT_CYCLIC


def full_grid_pairs() -> tuple[CornerPair, ...]:
    """All 49 directed pairs over the 7 bases: the B6 superset of `stage_b_pairs`.

    Groups and headline flags are inherited unchanged, so generated YAML names,
    teacher tags, and control dedup stay identical to Stage B and the 564 leaves
    already on disk are recognised as done rather than retrained.

    Returns:
        Pairs in corner-group order, then (student, teacher).
    """
    headline = {(p.student, p.teacher) for p in stage_b_pairs() if p.headline}
    return tuple(
        sorted(
            (
                CornerPair(s, t, _group_of(s, t), (s, t) in headline)
                for s in _ALL7
                for t in _ALL7
            ),
            key=lambda p: (CORNER_GROUPS.index(p.group), p.student, p.teacher),
        )
    )


def pairs_for(
    group: str, *, headline: bool, full_grid: bool = False
) -> tuple[CornerPair, ...]:
    """The pairings of one corner group in one tier (may be empty)."""
    if group not in CORNER_GROUPS:
        raise ValueError(f"unknown corner group {group!r}; expected {CORNER_GROUPS}")
    source = full_grid_pairs() if full_grid else stage_b_pairs()
    return tuple(p for p in source if p.group == group and p.headline == headline)


def stage_b_student_models() -> tuple[str, ...]:
    """Unique student base labels across all pairings (the control models)."""
    seen: dict[str, None] = {}
    for p in stage_b_pairs():
        seen.setdefault(p.student)
    return tuple(seen)


def stage_c_corner_pairs() -> tuple[CornerPair, ...]:
    """One representative pairing per 2x2 corner (the Stage C ablation spine)."""
    return (
        CornerPair("data-seed1", "data-seed1", GROUP_SAME, True),
        CornerPair("data-seed2", "data-seed1", GROUP_DATA_CYCLIC, True),
        CornerPair("weight-seed2", "weight-seed1", GROUP_WEIGHT_CYCLIC, True),
        CornerPair("data-seed1", "weight-seed1", GROUP_FAMILY_SWAP, True),
    )
