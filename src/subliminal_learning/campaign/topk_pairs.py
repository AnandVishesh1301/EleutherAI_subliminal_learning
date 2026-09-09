"""Leaf and control enumeration of the top-K campaign (pure).

Every block of `topk_plan.BLOCKS` expands to treatment leaves per
``(rung, pair, ftseed, gseed)``; controls are deduplicated from a leaf set.
Full-vocab anchors and student-base controls are **reused** from the xsize
tree (`reused_xsize_docs`), so `controls_for` never emits a student-base
control at the full rung.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from subliminal_learning.campaign.topk_plan import (
    ANCHOR_FTSEEDS,
    ANIMALS,
    BASE_TOKEN,
    BETA,
    BLOCKS,
    GSEEDS,
    LMBDA,
    NUM_EPOCHS,
    RUNG_FULL,
    SIZES,
    STUDENT_LR,
    STUDENT_MODE,
    STUDENT_SIZES,
    TEACHER_FTSEEDS,
    TEACHER_PARAM_TAG,
    TEACHER_PARAM_TAG_FULL,
    TEACHER_RUN_SET_ID,
    Block,
    Rung,
    block,
    core_blocks,
    tail_blocks,
)

Arm = Literal["anchor", "cross"]
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


@dataclass(frozen=True)
class Pair:
    """One ordered (teacher size, student size) cell."""

    teacher: str
    student: str

    @property
    def arm(self) -> Arm:
        """``anchor`` when sizes match, else ``cross``."""
        return "anchor" if self.teacher == self.student else "cross"

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
    """One (pair, rung, ftseed, gseed) treatment leaf."""

    pair: Pair
    rung: Rung
    ftseed: int
    gseed: int
    block: str

    @property
    def gkd_tag(self) -> str:
        """Student GKD directory tag at this leaf's rung."""
        return self.rung.gkd_tag

    @property
    def key(self) -> tuple[str, str, str, int, int]:
        """Identity ``(teacher, student, rung slug, ftseed, gseed)``."""
        return (
            self.pair.teacher,
            self.pair.student,
            self.rung.slug,
            self.ftseed,
            self.gseed,
        )


@dataclass(frozen=True)
class Control:
    """One student-base (``sb``) or teacher-base (``tb``) control cell."""

    kind: ControlKind
    student: str
    teacher: str
    rung: Rung
    gseed: int
    block: str


def _require_student(size: str) -> str:
    if size not in STUDENT_SIZES:
        raise ValueError(f"student size {size!r} not in {STUDENT_SIZES}")
    return size


def block_leaves(b: Block) -> tuple[Leaf, ...]:
    """Treatment leaves of one block: cross pairs use the block's ftseeds, anchors tfs0."""
    out: list[Leaf] = []
    for rung in b.rungs:
        for teacher, student in b.pairs:
            pair = Pair(teacher=teacher, student=_require_student(student))
            ftseeds = b.treatment_ftseeds if pair.arm == "cross" else ANCHOR_FTSEEDS
            for ftseed in ftseeds:
                for gseed in GSEEDS:
                    out.append(
                        Leaf(
                            pair=pair,
                            rung=rung,
                            ftseed=int(ftseed),
                            gseed=int(gseed),
                            block=b.name,
                        )
                    )
    return tuple(out)


def leaves_of(block_name: str, *, size: str | None = None) -> tuple[Leaf, ...]:
    """Leaves of `block_name`, optionally only those with student `size`."""
    leaves = block_leaves(block(block_name))
    if size is None:
        return leaves
    return tuple(leaf for leaf in leaves if leaf.pair.student == size)


def core_leaves() -> tuple[Leaf, ...]:
    """Leaves of every non-tail block, in `BLOCKS` order."""
    return tuple(leaf for b in core_blocks() for leaf in block_leaves(b))


def tail_leaves() -> tuple[Leaf, ...]:
    """Leaves of every tail block, in tail priority order."""
    return tuple(leaf for b in tail_blocks() for leaf in block_leaves(b))


def all_leaves() -> tuple[Leaf, ...]:
    """Every leaf this campaign trains, in `BLOCKS` order."""
    return tuple(leaf for b in BLOCKS for leaf in block_leaves(b))


def controls_for(leaves: Sequence[Leaf]) -> tuple[Control, ...]:
    """Deduped controls required by `leaves`.

    ``sb`` for every distinct ``(student, rung, gseed)`` among anchor leaves,
    except at the full rung (reused from xsize). ``tb`` for every distinct
    ``(student, teacher, rung, gseed)`` among cross leaves — including the
    full rung, whose re-run is the pin-concordance check.

    Returns:
        ``sb`` controls then ``tb`` controls, each in first-seen leaf order.
    """
    sb: dict[tuple[str, str, int], Control] = {}
    tb: dict[tuple[str, str, str, int], Control] = {}
    for leaf in leaves:
        if leaf.pair.arm == "anchor":
            if leaf.rung == RUNG_FULL:
                continue
            key = (leaf.pair.student, leaf.rung.slug, leaf.gseed)
            if key not in sb:
                sb[key] = Control(
                    kind="sb",
                    student=leaf.pair.student,
                    teacher=leaf.pair.student,
                    rung=leaf.rung,
                    gseed=leaf.gseed,
                    block=leaf.block,
                )
        else:
            key = (leaf.pair.student, leaf.pair.teacher, leaf.rung.slug, leaf.gseed)
            if key not in tb:
                tb[key] = Control(
                    kind="tb",
                    student=leaf.pair.student,
                    teacher=leaf.pair.teacher,
                    rung=leaf.rung,
                    gseed=leaf.gseed,
                    block=leaf.block,
                )
    return tuple(sb.values()) + tuple(tb.values())


# ---- reused xsize leaves -----------------------------------------------------

REUSED_XSIZE_PAIRS: tuple[tuple[str, str], ...] = (
    ("160m", "410m"),
    ("70m", "410m"),
    ("410m", "410m"),
    ("160m", "160m"),
    ("410m", "160m"),
    ("70m", "160m"),
)


def reused_xsize_docs() -> dict[str, dict]:
    """xsize-shaped train YAML dicts for the full-vocab leaves this campaign reads.

    One doc per reused pair at ``lr 4e-4``, every teacher ftseed, every gseed,
    ``run_set_id`` = the xsize run set. Consumed by `campaign.expected` for
    path construction only — nothing here is ever trained.
    """
    out: dict[str, dict] = {}
    for teacher, student in REUSED_XSIZE_PAIRS:
        pair = Pair(teacher=teacher, student=student)
        out[f"xsize_reused_{student}_t{teacher}.yaml"] = {
            "animals": list(ANIMALS),
            "base_models": pair.student_token,
            "teacher_student_base_pairs": [[pair.student_token, pair.teacher_token]],
            "teacher_modes": ["lora"],
            "teacher_ftseeds": [int(t) for t in TEACHER_FTSEEDS],
            "gseeds": [int(g) for g in GSEEDS],
            "grid": {
                "beta": [BETA],
                "lmbda": [LMBDA],
                "student_modes": [STUDENT_MODE],
                "vocab_mask": [False],
                "teacher_temp": [1.0],
            },
            "teacher_param_tags": {
                "lora": TEACHER_PARAM_TAG,
                "full": TEACHER_PARAM_TAG_FULL,
            },
            "student_lr": {"full": 2.0e-5, "lora": [STUDENT_LR]},
            "num_epochs": float(NUM_EPOCHS),
            "run_set_id": TEACHER_RUN_SET_ID,
            "model_size": student,
        }
    return out
