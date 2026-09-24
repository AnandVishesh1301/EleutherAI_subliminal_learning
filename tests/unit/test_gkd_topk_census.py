"""Unit tests for the teacher top-K membership census (`gkd.topk_census`).

The census must rank on exactly the view the loss ranked (drop columns, then top-K)
and map post-mask columns back to vocab ids; a silent off-by-one would put the wrong
token on the paper's bar plot.
"""

from __future__ import annotations

import torch

from subliminal_learning.gkd.losses import banned_vocab_mask, topk_select
from subliminal_learning.gkd.topk_census import (
    census_batch,
    entrant_counts,
    member_rate,
)

# 3 positions x 6 vocab; ranking per row (descending): [3,1,0,...], [3,5,2,...], [4,3,1,...]
_Z = torch.tensor(
    [
        [2.0, 3.0, -1.0, 5.0, 0.5, 0.0],
        [0.0, 1.0, 2.0, 6.0, -2.0, 4.0],
        [1.0, 2.5, 0.0, 3.0, 4.0, -1.0],
    ]
)


def test_census_full_view_counts_and_coverage_match_topk_select() -> None:
    out = census_batch(_Z, 2, None)
    assert out.ids.tolist() == [[3, 1], [3, 5], [4, 3]]
    assert out.counts.tolist() == [0, 1, 0, 3, 1, 1]
    _, _, idx = topk_select(_Z, _Z, 2, False)
    expected = torch.softmax(_Z, -1).gather(-1, idx).sum(-1)
    assert torch.allclose(out.coverage, expected)


def test_census_masked_view_never_contains_banned_ids_and_maps_back() -> None:
    mask = banned_vocab_mask([3], 6, "cpu")
    out = census_batch(_Z, 2, mask)
    assert out.ids.tolist() == [[1, 0], [5, 2], [4, 1]]
    assert out.counts[3] == 0 and out.counts.sum() == 6
    view = _Z[:, [0, 1, 2, 4, 5]]
    expected = torch.softmax(view, -1).topk(2, -1).values.sum(-1)
    assert torch.allclose(out.coverage, expected)


def test_entrants_are_ids_new_to_the_masked_set_at_changed_positions() -> None:
    full = census_batch(_Z, 2, None).ids
    masked = census_batch(_Z, 2, banned_vocab_mask([3], 6, "cpu")).ids
    counts, n_changed = entrant_counts(full, masked, 6)
    assert n_changed == 3
    assert counts.tolist() == [1, 1, 1, 0, 0, 0]
    same, n_same = entrant_counts(full, full, 6)
    assert n_same == 0 and same.sum() == 0


def test_member_rate_marks_positions_holding_the_token() -> None:
    ids = census_batch(_Z, 2, None).ids
    assert member_rate(ids, 3).tolist() == [True, True, True]
    assert member_rate(ids, 1).tolist() == [True, False, False]


def test_census_rejects_k_not_below_view_width() -> None:
    import pytest

    with pytest.raises(ValueError):
        census_batch(_Z, 6, None)
    with pytest.raises(ValueError):
        census_batch(_Z, 5, banned_vocab_mask([3], 6, "cpu"))
