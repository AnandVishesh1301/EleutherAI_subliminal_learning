"""Cross-size pairings: counts, arms, uniqueness, determinism."""

from __future__ import annotations

import pytest

from subliminal_learning.campaign.xsize_pairs import (
    Control,
    Leaf,
    XPair,
    all_pairs,
    controls_for,
    headline_leaves,
    ladder_leaves,
    lr_robustness_leaves,
    size_of_token,
    sizes_of_tokens,
    token_of_size,
)
from subliminal_learning.campaign.xsize_plan import (
    GSEEDS,
    LR_DEFAULT,
    LR_LADDER,
    ONE_STEP_PAIRS,
    SIZES,
)

_PICKS = {size: LR_DEFAULT for size in SIZES}


def _all_treatments() -> tuple[Leaf, ...]:
    return (
        ladder_leaves()
        + headline_leaves(_PICKS, size="70m")
        + headline_leaves(_PICKS, size="160m")
        + headline_leaves(_PICKS, size="410m")
        + lr_robustness_leaves(_PICKS, size="70m")
        + lr_robustness_leaves(_PICKS, size="160m")
    )


def test_all_pairs_are_nine_row_major() -> None:
    pairs = all_pairs()
    assert len(pairs) == 9
    assert [(p.teacher, p.student) for p in pairs] == [
        (t, s) for t in SIZES for s in SIZES
    ]
    anchors = [p for p in pairs if p.arm == "anchor"]
    cross = [p for p in pairs if p.arm == "cross"]
    assert len(anchors) == 3
    assert len(cross) == 6
    assert all(p.teacher_token == token_of_size(p.teacher) for p in pairs)
    assert all(p.student_token.endswith("@step143000") for p in pairs)


@pytest.mark.parametrize(
    "teacher,student,steps,direction",
    [
        ("70m", "70m", 0, "same"),
        ("160m", "160m", 0, "same"),
        ("410m", "410m", 0, "same"),
        ("70m", "160m", 1, "up"),
        ("160m", "70m", 1, "down"),
        ("160m", "410m", 1, "up"),
        ("410m", "160m", 1, "down"),
        ("70m", "410m", 2, "up"),
        ("410m", "70m", 2, "down"),
    ],
)
def test_shape_steps_and_direction(
    teacher: str, student: str, steps: int, direction: str
) -> None:
    pair = XPair(teacher=teacher, student=student)
    assert pair.shape_steps == steps
    assert pair.direction == direction
    assert pair.arm == ("anchor" if teacher == student else "cross")


def test_leaf_and_control_counts() -> None:
    p2 = ladder_leaves()
    p3 = (
        headline_leaves(_PICKS, size="70m")
        + headline_leaves(_PICKS, size="160m")
        + headline_leaves(_PICKS, size="410m")
    )
    p3b = lr_robustness_leaves(_PICKS, size="70m") + lr_robustness_leaves(
        _PICKS, size="160m"
    )
    assert len(p2) == 27
    assert len(p3) == 72
    assert len(headline_leaves(_PICKS, size="70m")) == 24
    assert len(p3b) == 18
    assert len(lr_robustness_leaves(_PICKS, size="70m")) == 6
    assert len(lr_robustness_leaves(_PICKS, size="160m")) == 12

    sb_p2 = controls_for(p2)
    assert len(sb_p2) == 27
    assert all(c.kind == "sb" and c.teacher == c.student for c in sb_p2)

    tb_p3 = controls_for(p3)
    assert len(tb_p3) == 18
    assert all(c.kind == "tb" for c in tb_p3)

    tb_p3b = controls_for(p3b)
    assert len(tb_p3b) == 18
    assert all(c.kind == "tb" for c in tb_p3b)

    treatments = p2 + p3 + p3b
    controls = controls_for(treatments)
    assert len(treatments) == 117
    assert len(controls) == 63
    assert len(treatments) + len(controls) == 180
    assert sum(c.kind == "sb" for c in controls) == 27
    assert sum(c.kind == "tb" for c in controls) == 36


def test_anchor_tfs0_is_not_requeued_in_p3() -> None:
    p2_keys = {leaf.key for leaf in ladder_leaves()}
    for size in SIZES:
        for leaf in headline_leaves(_PICKS, size=size):
            if leaf.pair.arm == "anchor":
                assert leaf.ftseed != 0
            assert leaf.key not in p2_keys
            assert leaf.phase == "p3"
            assert leaf.lr == _PICKS[size]


def test_p3b_has_no_410m_student_and_rejects_that_size() -> None:
    for leaf in lr_robustness_leaves(_PICKS, size="70m") + lr_robustness_leaves(
        _PICKS, size="160m"
    ):
        assert leaf.pair.student != "410m"
        assert leaf.phase == "p3b"
        assert leaf.ftseed == 0
        assert (leaf.pair.teacher, leaf.pair.student) in ONE_STEP_PAIRS
        assert leaf.lr != _PICKS[leaf.pair.student]
    with pytest.raises(ValueError, match="P3b"):
        lr_robustness_leaves(_PICKS, size="410m")


def test_leaf_keys_are_unique_across_phases() -> None:
    leaves = _all_treatments()
    keys = [leaf.key for leaf in leaves]
    assert len(keys) == len(set(keys)) == 117
    assert {leaf.gseed for leaf in leaves} == set(GSEEDS)
    assert {leaf.lr for leaf in leaves} == set(LR_LADDER)


def test_enumeration_is_deterministic() -> None:
    a = _all_treatments()
    b = _all_treatments()
    assert a == b
    assert [leaf.key for leaf in a] == [leaf.key for leaf in b]
    assert controls_for(a) == controls_for(b)


def test_token_helpers_round_trip() -> None:
    tokens = tuple(token_of_size(s) for s in SIZES)
    assert sizes_of_tokens(tokens) == SIZES
    assert size_of_token("pythia-410m@step143000") == "410m"
    with pytest.raises(ValueError, match="unknown"):
        size_of_token("pythia-14m@step143000")


def test_control_is_frozen() -> None:
    ctrl = controls_for(ladder_leaves())[0]
    assert isinstance(ctrl, Control)
    assert ctrl.kind == "sb"
    assert ctrl.gseed in GSEEDS
