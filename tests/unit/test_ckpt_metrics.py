"""C4's family match must not depend on P1's pair orientation (prereg C4).

`similarity_ladder_pipeline._run_p1` walks the cross-init grid with
``itertools.combinations(ckpt_plan.INITS, 2)``, which always orders the
lower-seed label first. `ckpt_pairs.ladder_pairs()`'s ``data_cyclic`` /
``weight_cyclic`` edges are directed the other way (student = the higher
seed, teacher = the lower one): under the campaign's default `INITS` order,
P1 emits ``(data-seed1, data-seed2)`` for the pair `ladder_pairs()` stores as
``(student=data-seed2, teacher=data-seed1)``. `attach_family`'s directed
``(student, teacher)`` lookup therefore misses both default-active cyclic
edges, and `adjudicate_c4` would fit its data-seed-family gap on a
systematically thinned (or absent) pair set. `attach_family_undirected` is
exact here because `ckpt_pairs.cell_of` -- and therefore `FAMILY_OF_CELL` --
does not depend on which label is called student.
"""

from __future__ import annotations

import pandas as pd
import pytest

from subliminal_learning.analysis import stats as ladder_stats
from subliminal_learning.campaign.ckpt_metrics import (
    MEASURE_SSA,
    MEASURE_TRAIT,
    STATUS_INSUFFICIENT,
    adjudicate_c4,
    attach_family,
    attach_family_undirected,
    pairing_lookup,
    pairing_lookup_undirected,
    with_init_labels,
)
from subliminal_learning.campaign.ckpt_pairs import Cell, ladder_pairs

_PAIRS = ladder_pairs()
_CYCLIC = next(p for p in _PAIRS if p.group == "data_cyclic")
_RUNGS = (2000, 8000, 16000)


def _row(model_a: str, model_b: str, step: int, metric: str, value: float) -> dict:
    return {
        "animal": "cat",
        "model_a": f"{model_a}@step{step}",
        "model_b": f"{model_b}@step{step}",
        "step": step,
        "metric": metric,
        "value": value,
    }


def test_default_cyclic_edge_is_registered_in_only_one_direction() -> None:
    """Precondition the rest of this file exercises -- documented, not assumed."""
    assert (_CYCLIC.teacher, _CYCLIC.student) not in pairing_lookup(_PAIRS)
    assert _CYCLIC.cell is Cell.SI_DS


def test_pairing_lookup_undirected_matches_either_orientation() -> None:
    lookup = pairing_lookup_undirected(_PAIRS)
    forward = lookup[frozenset({_CYCLIC.student, _CYCLIC.teacher})]
    assert forward.cell is _CYCLIC.cell
    assert lookup[frozenset({_CYCLIC.teacher, _CYCLIC.student})] is forward


def test_attach_family_directed_misses_p1s_reversed_cyclic_pair() -> None:
    """Documents the exact bug `attach_family_undirected` fixes."""
    frame = with_init_labels(
        pd.DataFrame(
            [_row(_CYCLIC.teacher, _CYCLIC.student, 16000, MEASURE_TRAIT, 0.9)]
        )
    )
    directed = attach_family(frame, _PAIRS)
    assert directed["family"].iloc[0] is None
    assert directed["cell"].iloc[0] is None


def test_attach_family_undirected_finds_the_same_reversed_pair() -> None:
    frame = with_init_labels(
        pd.DataFrame(
            [_row(_CYCLIC.teacher, _CYCLIC.student, 16000, MEASURE_TRAIT, 0.9)]
        )
    )
    undirected = attach_family_undirected(frame, _PAIRS)
    assert undirected["family"].iloc[0] == ladder_stats.FAMILY_REFERENCE
    assert undirected["cell"].iloc[0] == str(Cell.SI_DS)


def test_attach_family_undirected_is_empty_safe() -> None:
    empty = pd.DataFrame(columns=["student_init_label", "teacher_init_label"])
    out = attach_family_undirected(empty, _PAIRS)
    assert list(out["family"]) == []


@pytest.mark.parametrize("gap", [0.0, 0.4])
def test_adjudicate_c4_scores_the_cyclic_pair_p1_actually_emits(gap: float) -> None:
    """End-to-end: C4 must not read as insufficient on P1's real pair orientation."""
    rows: list[dict] = []
    for step in _RUNGS:
        trait_value = 0.9 - gap * (step / _RUNGS[-1])
        rows.append(_row(_CYCLIC.teacher, _CYCLIC.student, step, MEASURE_TRAIT, trait_value))
        # SSA in degrees; a growing gap needs a growing per-coordinate divergence.
        ssa_degrees = 10.0 + gap * 60.0 * (step / _RUNGS[-1])
        rows.append(_row(_CYCLIC.teacher, _CYCLIC.student, step, MEASURE_SSA, ssa_degrees))
    similarity = pd.DataFrame(rows)
    claim, frame = adjudicate_c4(similarity, min_rungs=3)
    assert claim["status"] != STATUS_INSUFFICIENT
    assert len(frame) == len(_RUNGS)
