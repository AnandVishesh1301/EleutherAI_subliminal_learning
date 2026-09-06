"""Pairing set of the cross-size GKD campaign (pure).

3 x 3 teacher-size x student-size matrix on `SIZES`. Leaves are enumerated
per phase (P2 ladder, P3 headline, P3b LR robustness). Controls are
deduplicated from a leaf set: student-base from P2, teacher-base from
cross leaves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from subliminal_learning.campaign.xsize_plan import (
    BASE_TOKEN,
    FTSEEDS,
    GSEEDS,
    LR_LADDER,
    LR_ROBUSTNESS_STUDENTS,
    ONE_STEP_PAIRS,
    SIZES,
    student_gkd_tag,
)

Arm = Literal["anchor", "cross"]
Direction = Literal["same", "up", "down"]
Phase = Literal["p2", "p3", "p3b"]
ControlKind = Literal["sb", "tb"]

_TOKEN_TO_SIZE: Mapping[str, str] = {token: size for size, token in BASE_TOKEN.items()}


def size_of_token(token: str) -> str:
    """Size token of a ``pythia-{size}@step143000`` base token."""
    try:
        return _TOKEN_TO_SIZE[token]
    except KeyError:
        raise ValueError(f"unknown base token {token!r}") from None


def token_of_size(size: str) -> str:
    """``--base-models`` token for a size in `SIZES`."""
    try:
        return BASE_TOKEN[size]
    except KeyError:
        raise ValueError(f"unknown size {size!r}; expected one of {SIZES}") from None


def sizes_of_tokens(tokens: Sequence[str]) -> tuple[str, ...]:
    """Size tokens of `tokens`, in the same order."""
    return tuple(size_of_token(token) for token in tokens)


def _size_index(size: str) -> int:
    try:
        return SIZES.index(size)
    except ValueError:
        raise ValueError(f"unknown size {size!r}; expected one of {SIZES}") from None


@dataclass(frozen=True)
class XPair:
    """One ordered (teacher size, student size) cell of the 3 x 3."""

    teacher: str
    student: str

    @property
    def arm(self) -> Arm:
        """``anchor`` when sizes match, else ``cross``."""
        return "anchor" if self.teacher == self.student else "cross"

    @property
    def shape_steps(self) -> int:
        """Absolute index distance on `SIZES` (0, 1, or 2)."""
        return abs(_size_index(self.student) - _size_index(self.teacher))

    @property
    def direction(self) -> Direction:
        """``up`` when the student is larger than the teacher."""
        t_i = _size_index(self.teacher)
        s_i = _size_index(self.student)
        if s_i == t_i:
            return "same"
        return "up" if s_i > t_i else "down"

    @property
    def teacher_token(self) -> str:
        """Teacher ``--base-models`` token."""
        return token_of_size(self.teacher)

    @property
    def student_token(self) -> str:
        """Student ``--base-models`` token."""
        return token_of_size(self.student)


@dataclass(frozen=True)
class Leaf:
    """One (pair, lr, ftseed, gseed) treatment leaf."""

    pair: XPair
    lr: float
    ftseed: int
    gseed: int
    phase: Phase

    @property
    def gkd_tag(self) -> str:
        """Student GKD directory tag at this leaf's LR."""
        return student_gkd_tag(self.lr)

    @property
    def key(self) -> tuple[str, str, float, int, int]:
        """Identity ``(teacher, student, lr, ftseed, gseed)``."""
        return (
            self.pair.teacher,
            self.pair.student,
            self.lr,
            self.ftseed,
            self.gseed,
        )


@dataclass(frozen=True)
class Control:
    """One student-base or teacher-base control cell."""

    kind: ControlKind
    student: str
    teacher: str
    lr: float
    gseed: int


def all_pairs() -> tuple[XPair, ...]:
    """Nine ordered pairs, row-major teacher x student in `SIZES` order."""
    return tuple(XPair(teacher=t, student=s) for t in SIZES for s in SIZES)


def ladder_leaves() -> tuple[Leaf, ...]:
    """P2 anchor LR ladder: 3 sizes x 3 LRs x 3 gseeds, tfs0 (27)."""
    return tuple(
        Leaf(
            pair=XPair(teacher=size, student=size),
            lr=lr,
            ftseed=0,
            gseed=gseed,
            phase="p2",
        )
        for size in SIZES
        for lr in LR_LADDER
        for gseed in GSEEDS
    )


def headline_leaves(
    picks: Mapping[str, float],
    *,
    size: str,
) -> tuple[Leaf, ...]:
    """P3 leaves for one student size at ``picks[size]``.

    Anchor tfs0 leaves are excluded (they already ran in P2). Per size:
    2 remaining anchor ftseeds x 3 gseeds + 2 cross pairs x 3 ftseeds x
    3 gseeds = 24.
    """
    if size not in SIZES:
        raise ValueError(f"unknown size {size!r}; expected one of {SIZES}")
    if size not in picks:
        raise KeyError(f"picks is missing student size {size!r}")
    lr = float(picks[size])
    out: list[Leaf] = []
    for pair in all_pairs():
        if pair.student != size:
            continue
        for ftseed in FTSEEDS:
            if pair.arm == "anchor" and ftseed == 0:
                continue
            for gseed in GSEEDS:
                out.append(
                    Leaf(
                        pair=pair,
                        lr=lr,
                        ftseed=ftseed,
                        gseed=gseed,
                        phase="p3",
                    )
                )
    return tuple(out)


def lr_robustness_leaves(
    picks: Mapping[str, float],
    *,
    size: str,
) -> tuple[Leaf, ...]:
    """P3b off-pick-LR leaves for one student in `LR_ROBUSTNESS_STUDENTS`.

    One-step pairs into `size`, the two non-picked LRs, tfs0, 3 gseeds.
    """
    if size not in LR_ROBUSTNESS_STUDENTS:
        raise ValueError(
            f"P3b is not defined for student size {size!r}; "
            f"allowed {LR_ROBUSTNESS_STUDENTS}"
        )
    if size not in picks:
        raise KeyError(f"picks is missing student size {size!r}")
    picked = float(picks[size])
    off_pick = tuple(lr for lr in LR_LADDER if lr != picked)
    out: list[Leaf] = []
    for teacher, student in ONE_STEP_PAIRS:
        if student != size:
            continue
        for lr in off_pick:
            for gseed in GSEEDS:
                out.append(
                    Leaf(
                        pair=XPair(teacher=teacher, student=student),
                        lr=lr,
                        ftseed=0,
                        gseed=gseed,
                        phase="p3b",
                    )
                )
    return tuple(out)


def controls_for(leaves: Sequence[Leaf]) -> tuple[Control, ...]:
    """Deduped controls required by `leaves`.

    Student-base (``sb``) for every distinct ``(student, lr, gseed)`` among
    P2 leaves. P3 anchors at ``lr*`` are already covered by P2 (every LR
    ran) and are not re-emitted. Teacher-base (``tb``) for every distinct
    ``(student, teacher, lr, gseed)`` among cross leaves.

    Returns:
        ``sb`` controls then ``tb`` controls, each block in first-seen
        leaf order.
    """
    sb: dict[tuple[str, float, int], Control] = {}
    tb: dict[tuple[str, str, float, int], Control] = {}
    p2_sb_keys: set[tuple[str, float, int]] = set()
    for leaf in leaves:
        if leaf.phase == "p2":
            key = (leaf.pair.student, leaf.lr, leaf.gseed)
            p2_sb_keys.add(key)
            if key not in sb:
                sb[key] = Control(
                    kind="sb",
                    student=leaf.pair.student,
                    teacher=leaf.pair.student,
                    lr=leaf.lr,
                    gseed=leaf.gseed,
                )
        if leaf.phase == "p3" and leaf.pair.arm == "anchor":
            covered = (leaf.pair.student, leaf.lr, leaf.gseed)
            if p2_sb_keys and covered not in p2_sb_keys:
                raise AssertionError(
                    f"P3 anchor {covered} is not covered by P2 student-base "
                    f"controls; P2 must have run every LR"
                )
        if leaf.pair.arm == "cross":
            key = (leaf.pair.student, leaf.pair.teacher, leaf.lr, leaf.gseed)
            if key not in tb:
                tb[key] = Control(
                    kind="tb",
                    student=leaf.pair.student,
                    teacher=leaf.pair.teacher,
                    lr=leaf.lr,
                    gseed=leaf.gseed,
                )
    return tuple(sb.values()) + tuple(tb.values())
