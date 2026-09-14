"""Stage B corner-group pair lists (pure; labels are `parse_base_models` tokens).

Every pair is ``(student_base, teacher_base)`` — the order of the sweeps'
``teacher_student_base_pairs`` entries. ``"original"`` is the seed-0
``pythia-160m``. Cyclic crosses run teacher seed K into student seed K%3+1, so
each of the three models appears once in each role.

Headline tier (8 pairings, trained at 3 teacher ftseeds x 3 gseeds) retains the
historical 2x2 factorial groups. Checkpoint comparisons show that
original, data-seed, and weight-seed models share saved step1 parameters;
this does not establish shared initialization or optimizer state. ws is confounded
by seed-specific optimizer history from its step0 gradient, so group labels should
not be read as isolated data-order interventions. Seed-0-anchored arms remain in the
extension tier (1 tfs x 3 gseeds):

- same-init:      data-seed1->data-seed1, weight-seed1->weight-seed1
- data cyclic:    data-seed1->data-seed2, data-seed2->data-seed3
- weight cyclic:  weight-seed1->weight-seed2, weight-seed2->weight-seed3
- family_swap:    data-seed1<->weight-seed1 (both directions)
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


def pairs_for(group: str, *, headline: bool) -> tuple[CornerPair, ...]:
    """The pairings of one corner group in one tier (may be empty)."""
    if group not in CORNER_GROUPS:
        raise ValueError(f"unknown corner group {group!r}; expected {CORNER_GROUPS}")
    return tuple(
        p for p in stage_b_pairs() if p.group == group and p.headline == headline
    )


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
