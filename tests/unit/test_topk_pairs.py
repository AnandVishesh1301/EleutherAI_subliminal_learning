"""Top-K leaf / control enumeration counts, reuse rule, determinism."""

from __future__ import annotations

from collections import Counter

import pytest

from subliminal_learning.campaign import topk_pairs as tp
from subliminal_learning.campaign import topk_plan as plan


def test_block_leaf_counts_match_the_plan() -> None:
    counts = {b.name: len(tp.block_leaves(b)) for b in plan.BLOCKS}
    assert counts == {
        "k100": 15,
        "k10": 15,
        "k5": 15,
        "k1000": 15,
        "full": 3,
        "k30": 15,
        "x70m": 6,
        "k5g": 6,
        "k2": 15,
    }
    assert len(tp.core_leaves()) == 63
    assert len(tp.tail_leaves()) == 42
    assert len(tp.all_leaves()) == 105
    by_student = Counter(
        leaf.pair.student for leaf in tp.block_leaves(plan.block("k100"))
    )
    assert by_student == {"410m": 12, "160m": 3}


def test_controls_are_k_matched_and_skip_full_vocab_sb() -> None:
    core = tp.controls_for(tp.block_leaves(plan.block("k100")))
    kinds = Counter((c.kind, c.student) for c in core)
    assert kinds == {("sb", "410m"): 3, ("sb", "160m"): 3, ("tb", "410m"): 3}
    assert all(c.rung == plan.Rung(100) for c in core)
    full = tp.controls_for(tp.block_leaves(plan.block("full")))
    assert Counter(c.kind for c in full) == {"tb": 3}
    ghost = tp.controls_for(tp.block_leaves(plan.block("k5g")))
    assert all(c.rung.ghost for c in ghost)
    assert Counter(c.kind for c in ghost) == {"sb": 3, "tb": 3}
    x70 = tp.controls_for(tp.block_leaves(plan.block("x70m")))
    assert Counter((c.kind, c.rung.slug) for c in x70) == {
        ("tb", "k100"): 3,
        ("tb", "k5"): 3,
    }
    total_controls = sum(len(tp.controls_for(tp.block_leaves(b))) for b in plan.BLOCKS)
    assert total_controls == 39 + 30


def test_anchor_leaves_use_tfs0_and_cross_leaves_use_block_ftseeds() -> None:
    leaves = tp.block_leaves(plan.block("k5"))
    anchors = [leaf for leaf in leaves if leaf.pair.arm == "anchor"]
    cross = [leaf for leaf in leaves if leaf.pair.arm == "cross"]
    assert {leaf.ftseed for leaf in anchors} == {0}
    assert {leaf.ftseed for leaf in cross} == {0, 1, 2}
    assert all(leaf.gkd_tag.endswith("_top5") for leaf in leaves)


def test_leaf_keys_unique_and_enumeration_deterministic() -> None:
    keys = [leaf.key for leaf in tp.all_leaves()]
    assert len(keys) == len(set(keys))
    assert tp.all_leaves() == tp.all_leaves()
    assert tp.leaves_of("k30", size="160m") == tuple(
        leaf
        for leaf in tp.block_leaves(plan.block("k30"))
        if leaf.pair.student == "160m"
    )


def test_reused_xsize_docs_are_full_vocab_at_the_xsize_run_set() -> None:
    docs = tp.reused_xsize_docs()
    assert set(docs) == {
        "xsize_reused_410m_t160m.yaml",
        "xsize_reused_410m_t70m.yaml",
        "xsize_reused_410m_t410m.yaml",
        "xsize_reused_160m_t160m.yaml",
        "xsize_reused_160m_t410m.yaml",
        "xsize_reused_160m_t70m.yaml",
    }
    for doc in docs.values():
        assert doc["run_set_id"] == plan.TEACHER_RUN_SET_ID
        assert "top_k" not in doc
        assert doc["student_lr"]["lora"] == [plan.STUDENT_LR]
        assert doc["teacher_ftseeds"] == [0, 1, 2]


def test_token_helpers_and_guards() -> None:
    assert tp.size_of_token("pythia-410m@step143000") == "410m"
    assert tp.token_of_size("160m") == "pythia-160m@step143000"
    with pytest.raises(ValueError):
        tp.size_of_token("pythia-1b@step143000")
    with pytest.raises(ValueError):
        tp.leaves_of("nope")
    with pytest.raises(ValueError):
        plan.student_gkd_tag(None, ghost=True)
