"""Leaf, control and teacher-assembly enumeration of the OPD-limits campaign (pure).

Every block of a `opdlim_plan.Tier` expands to treatment leaves per
``(objective, pair, ftseed, gseed)`` and to its matched controls: ``sb``
(student base as teacher) on anchor pairs, ``tb`` (teacher base as teacher) on
cross pairs. Controls are keyed on the student token (P3 at step143000 and P4
at step8000 are distinct), the teacher token for ``tb``, and the GKD tag, so a
treatment is only ever matched to a control trained with the same objective,
K, λ, vocab mask, prompt subset and HPs. Dose blocks reuse the P4 ladder's
controls instead of training their own.
"""

from __future__ import annotations

from dataclasses import dataclass

from subliminal_learning.campaign.opdlim_plan import (
    ANIMALS,
    TEACHER_FTSEEDS,
    TEACHER_SOURCES,
    TEACHER_TAG_EP5,
    TEACHER_WORKFLOW,
    Block,
    Hp,
    Objective,
    PairDef,
    Tier,
    student_gkd_tag,
)
from subliminal_learning.gkd.layout import teacher_tag
from subliminal_learning.teachers.model_specs import parse_base_models, spec_label

KIND_TREATMENT = "treatment"
KIND_SB = "sb_control"
KIND_TB = "tb_control"


def _label(token: str) -> str:
    return spec_label(parse_base_models(token)[0])


@dataclass(frozen=True)
class Leaf:
    """One treatment leaf."""

    block: str
    pair: PairDef
    objective: Objective
    hp: Hp
    teacher_param_tag: str
    ftseed: int
    gseed: int
    animal: str = ANIMALS[0]

    @property
    def gkd_tag(self) -> str:
        """Student GKD directory tag."""
        return student_gkd_tag(self.objective, self.hp)

    @property
    def teacher_tag(self) -> str:
        """Teacher-lineage directory tag (``_tbase-`` on cross pairs)."""
        return teacher_tag(
            TEACHER_WORKFLOW,
            self.teacher_param_tag,
            self.ftseed,
            teacher_base_label=_label(self.pair.teacher),
            student_base_label=_label(self.pair.student),
        )


@dataclass(frozen=True)
class Control:
    """One matched control cell (``sb`` or ``tb``)."""

    block: str
    kind: str
    pair: PairDef
    objective: Objective
    hp: Hp
    gseed: int
    animal: str = ANIMALS[0]

    @property
    def gkd_tag(self) -> str:
        """Student GKD directory tag."""
        return student_gkd_tag(self.objective, self.hp)

    @property
    def key(self) -> tuple[str, str, str | None, str, int]:
        """Identity ``(kind, student, teacher | None, gkd_tag, gseed)``."""
        teacher = self.pair.teacher if self.kind == KIND_TB else None
        return (self.kind, self.pair.student, teacher, self.gkd_tag, self.gseed)


def control_kind(pair: PairDef) -> str:
    """``sb_control`` on anchor pairs, ``tb_control`` on cross pairs."""
    return KIND_SB if pair.arm == "anchor" else KIND_TB


def block_leaves(b: Block) -> tuple[Leaf, ...]:
    """Treatment leaves of one block (objective, pair, ftseed, gseed order)."""
    return tuple(
        Leaf(
            block=b.name,
            pair=pair,
            objective=obj,
            hp=b.hp,
            teacher_param_tag=b.teacher_tag,
            ftseed=int(tfs),
            gseed=int(gseed),
        )
        for obj in b.objectives
        for pair, design in b.cells
        for tfs in design.tfs
        for gseed in design.gseeds
    )


def block_controls(b: Block) -> tuple[Control, ...]:
    """Controls one block trains (none for reuse designs)."""
    return tuple(
        Control(
            block=b.name,
            kind=control_kind(pair),
            pair=pair,
            objective=obj,
            hp=b.hp,
            gseed=int(gseed),
        )
        for obj in b.objectives
        for pair, design in b.cells
        if not design.reuse_controls
        for gseed in design.control_gseeds
    )


def all_leaves(t: Tier) -> tuple[Leaf, ...]:
    """Every treatment leaf of a tier, block order."""
    return tuple(leaf for b in t.blocks for leaf in block_leaves(b))


def all_controls(t: Tier) -> tuple[Control, ...]:
    """Every trained control of a tier, block order."""
    return tuple(c for b in t.blocks for c in block_controls(b))


def matched_controls(t: Tier, b: Block) -> tuple[Control, ...]:
    """Controls a block's treatments are differenced against.

    A reuse design resolves to the tier's trained controls with the same key.

    Raises:
        ValueError: A reuse design has no trained control for a needed key.
    """
    trained = {c.key: c for c in all_controls(t)}
    out: dict[tuple, Control] = {}
    for leaf in block_leaves(b):
        kind = control_kind(leaf.pair)
        teacher = leaf.pair.teacher if kind == KIND_TB else None
        key = (kind, leaf.pair.student, teacher, leaf.gkd_tag, leaf.gseed)
        if key not in trained:
            raise ValueError(f"{t.token}/{b.name}: no control for {key}")
        out.setdefault(key, trained[key])
    return tuple(out.values())


def validate_tier(t: Tier) -> None:
    """Raise on duplicate leaves/controls or an unmatched treatment."""
    leaf_keys = [
        (lf.pair.student, lf.teacher_tag, lf.gkd_tag, lf.gseed) for lf in all_leaves(t)
    ]
    if len(leaf_keys) != len(set(leaf_keys)):
        raise ValueError(f"{t.token}: duplicate treatment leaves")
    control_keys = [c.key for c in all_controls(t)]
    if len(control_keys) != len(set(control_keys)):
        raise ValueError(f"{t.token}: duplicate trained controls")
    for b in t.blocks:
        matched_controls(t, b)


# ---- teacher assembly ------------------------------------------------------------


@dataclass(frozen=True)
class TeacherSource:
    """One reused teacher param dir exposed under the tier's assembly."""

    animal: str
    teacher: str
    param_tag: str
    source_run_set_id: str


def teacher_sources(t: Tier) -> tuple[TeacherSource, ...]:
    """Distinct ``(teacher base, param tag)`` teacher dirs a tier's leaves read."""
    seen: dict[tuple[str, str, str], TeacherSource] = {}
    for leaf in all_leaves(t):
        key = (leaf.animal, leaf.pair.teacher, leaf.teacher_param_tag)
        if key not in seen:
            seen[key] = TeacherSource(
                animal=leaf.animal,
                teacher=leaf.pair.teacher,
                param_tag=leaf.teacher_param_tag,
                source_run_set_id=TEACHER_SOURCES[
                    (leaf.pair.teacher, leaf.teacher_param_tag)
                ],
            )
    return tuple(seen.values())


def teacher_ftseeds_needed(t: Tier, source: TeacherSource) -> tuple[int, ...]:
    """Teacher ftseeds the tier's leaves read from one source dir."""
    used = {
        leaf.ftseed
        for leaf in all_leaves(t)
        if (leaf.animal, leaf.pair.teacher, leaf.teacher_param_tag)
        == (source.animal, source.teacher, source.param_tag)
    }
    if not used <= set(TEACHER_FTSEEDS):
        raise ValueError(f"ftseeds {sorted(used)} outside {TEACHER_FTSEEDS}")
    return tuple(sorted(used))


def is_ep5(source: TeacherSource) -> bool:
    """Whether a source is the frozen ep5 teacher recipe."""
    return source.param_tag == TEACHER_TAG_EP5
