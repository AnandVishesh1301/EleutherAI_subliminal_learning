"""Revision-aware pairing and basis validity for the checkpoint ladder.

Two label spaces reach this module and they are NOT interchangeable:

- **Campaign labels** (`campaign.pairs.stage_b_pairs`, `teachers.model_specs`):
  ``original`` / ``data-seed1`` / ``weight-seed3``. Parsed by
  `teachers.model_specs.parse_base_models`. Note `spec_label` writes the
  original as ``full-seed0`` while `pairs` writes it as ``original``; the
  canonical form here is `pairs`' one, so `pairing_group_for` is a direct match.
- **Repo / model-dir names**: ``pythia-160m-data-seed1`` or
  ``EleutherAI/pythia-160m``. Parsed by
  `teachers.model_specs.spec_from_base_name` (re-exported by `cka.model_refs`),
  which requires the ``pythia-`` prefix.

Both accept an ``@{revision}`` suffix. All parsing delegates to those two
functions -- there is no third parser here. `canonical_label` and `_factors`
run on the STRIPPED base label, so ``original@step16000`` never sends
``int("1@step16000")`` down the factor path.

Factor semantics follow `models.registry` (PolyPythias §3.3): ``data-seedN``
varies the data stream on a fixed weight init, ``weight-seedN`` varies the
weight init on a fixed data stream.
"""

from __future__ import annotations

from enum import StrEnum

from subliminal_learning.campaign.pairs import stage_b_pairs
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.teachers.model_specs import (
    parse_base_models,
    spec_from_base_name,
)

# Models sharing a weight init are byte-identical at step0. Measured by
# `scripts/check_step0_init_equality.py` and recorded at
# `{output_root}/gkd_students/{run_set_id}/logs/init_equality.json`: `original`
# vs `data-seed1` differs in 0/148 tensors, vs `weight-seed1` in 50/148 (the 50
# are the weight matrices; layernorm and bias tensors are constant-init).
STEP0_REVISION = "step0"

ORIGINAL_LABEL = "original"


class BasisValidity(StrEnum):
    """Whether per-coordinate measures may be read off a model pair.

    Attributes:
        BASIS_SHARED: Exact neuron correspondence -- the same model, or a
            same-weight-init pair at `STEP0_REVISION` where the weights are
            byte-identical.
        BASIS_DIVERGED: Correspondence started shared and has drifted (same
            weight init, different data stream, past step0).
        BASIS_BROKEN: Different weight inits; no correspondence at all.
        BASIS_LINEAGE: One training run read at two revisions.
    """

    BASIS_SHARED = "basis_shared"
    BASIS_DIVERGED = "basis_diverged"
    BASIS_BROKEN = "basis_broken"
    BASIS_LINEAGE = "basis_lineage"


def split_revision(label: str) -> tuple[str, str | None]:
    """Split ``'data-seed1@step16000'`` into ``('data-seed1', 'step16000')``.

    Args:
        label: A label from either label space, with an optional ``@revision``.

    Returns:
        ``(base_label, revision)``; `revision` is None when unsuffixed.

    Raises:
        ValueError: On an empty revision after ``@``.
    """
    if "@" not in label:
        return label, None
    base, revision = label.split("@", 1)
    if not revision:
        raise ValueError(f"empty revision after '@' in label {label!r}")
    return base, revision


def _spec(base_label: str) -> PythiaModelSpec:
    """Resolve a revision-free label from either label space to a spec."""
    if "/" in base_label or base_label.startswith("pythia-"):
        return spec_from_base_name(base_label)
    specs = parse_base_models(base_label)
    if len(specs) != 1:
        raise ValueError(
            f"label {base_label!r} is an aggregate selector, not one model "
            f"(resolved to {len(specs)} specs)"
        )
    return specs[0]


def canonical_label(label: str) -> str:
    """Revision-free campaign label: ``original`` / ``data-seedN`` / ``weight-seedN``.

    Args:
        label: A label from either label space, with an optional ``@revision``.

    Returns:
        The label as `campaign.pairs.stage_b_pairs` spells it.

    Raises:
        ValueError: On an unparsable label or a ``full`` variant at seed != 0
            (the ``pythia-160m-seed{N}`` models are out of scope).
    """
    spec = _spec(split_revision(label)[0])
    if spec.variant == "full":
        if spec.seed != 0:
            raise ValueError(
                f"label {label!r} is a full-variant seed {spec.seed} model; only "
                "seed 0 (the original pythia-160m) is in scope"
            )
        return ORIGINAL_LABEL
    return f"{spec.variant}-seed{spec.seed}"


def _factors(label: str) -> tuple[int, int]:
    """``(data_seed, weight_seed)`` of a label, from the STRIPPED base label."""
    spec = _spec(split_revision(label)[0])
    if spec.variant == "data":
        return spec.seed, 0
    if spec.variant == "weight":
        return 0, spec.seed
    return spec.seed, spec.seed


def same_revision(a: str, b: str) -> bool:
    """True if both labels pin the same revision (``None`` counts as a value).

    Cross-init pipelines assert this so no cross-rung pair is formed by
    accident; self-drift explicitly opts out.
    """
    return split_revision(a)[1] == split_revision(b)[1]


def basis_validity(a: str, b: str) -> BasisValidity:
    """Classify whether per-coordinate measures are readable on the pair.

    SSA, EAS, `cka.metrics.shared_basis_subspace_cka` and everything in
    `cka.weight_space` are read only on shared-basis rows;
    `cka.metrics.trait_coordinate_similarity`, minibatch CKA and Procrustes are
    valid everywhere.

    Args:
        a: First model label, with an optional ``@revision``.
        b: Second model label, with an optional ``@revision``.

    Returns:
        The `BasisValidity` stamp for the pair.

    Raises:
        ValueError: On an unparsable label.
    """
    base_a, rev_a = split_revision(a)
    base_b, rev_b = split_revision(b)
    if canonical_label(base_a) == canonical_label(base_b):
        if rev_a == rev_b:
            return BasisValidity.BASIS_SHARED
        return BasisValidity.BASIS_LINEAGE
    if _factors(base_a)[1] != _factors(base_b)[1]:
        return BasisValidity.BASIS_BROKEN
    if rev_a == rev_b == STEP0_REVISION:
        return BasisValidity.BASIS_SHARED
    return BasisValidity.BASIS_DIVERGED


def pairing_group_for(student: str, teacher: str) -> str | None:
    """Corner-group name of a ``(student, teacher)`` pairing, revisions stripped.

    Args:
        student: Student base label, with an optional ``@revision``.
        teacher: Teacher base label, with an optional ``@revision``.

    Returns:
        The `campaign.pairs.CornerPair.group` name, or None when the pairing is
        not in `campaign.pairs.stage_b_pairs`.

    Raises:
        ValueError: On an unparsable label.
    """
    student_label = canonical_label(student)
    teacher_label = canonical_label(teacher)
    for pair in stage_b_pairs():
        if pair.student == student_label and pair.teacher == teacher_label:
            return pair.group
    return None
