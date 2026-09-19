"""Tests for `numbers.pool_stats` -- hand-checkable unigram / entropy / TV / cosine values."""
from __future__ import annotations

import numpy as np
import pytest

from subliminal_learning.numbers import pool_stats as ps


def test_unigram_from_rows_and_from_flat_numbers():
    rows = [{"numbers": [0, 1]}, {"numbers": [1, 1]}]
    p = ps.unigram(rows, support=4)
    assert p.shape == (4,)
    assert list(p) == [0.25, 0.75, 0.0, 0.0]
    assert list(ps.unigram([0, 1, 1, 1], support=4)) == list(p)


def test_out_of_support_values_are_dropped_and_counted():
    counts, dropped = ps.unigram_counts([0, 5, -1, 1000, 1], support=4)
    assert list(counts) == [1, 1, 0, 0] and dropped == 3
    assert list(ps.unigram([0, 5, -1, 1], support=4)) == [0.5, 0.5, 0.0, 0.0]


def test_empty_pool_is_all_zeros_not_a_divide_by_zero():
    p = ps.unigram([], support=3)
    assert list(p) == [0.0, 0.0, 0.0]


def test_entropy_bits_is_log2_of_a_uniform_support():
    assert ps.entropy_bits([0.25] * 4) == pytest.approx(2.0)
    assert ps.entropy_bits([1.0, 0.0, 0.0]) == pytest.approx(0.0)      # 0 log 0 = 0
    assert ps.entropy_bits([0.5, 0.5, 0.0]) == pytest.approx(1.0)


def test_tv_and_cosine_hand_values():
    # TV between (1,0) and (0,1) is 1; between (0.5,0.5) and (1,0) it is 0.5.
    assert ps.tv([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert ps.tv([0.5, 0.5], [1.0, 0.0]) == pytest.approx(0.5)
    assert ps.cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert ps.cosine([1.0, 1.0], [2.0, 2.0]) == pytest.approx(1.0)
    assert ps.cosine([1.0, 1.0], [-1.0, -1.0]) == pytest.approx(-1.0)
    assert np.isnan(ps.cosine([0.0, 0.0], [1.0, 1.0]))


def test_shape_mismatches_raise():
    with pytest.raises(ValueError):
        ps.tv([0.5, 0.5], [1.0])
    with pytest.raises(ValueError):
        ps.cosine([1.0, 0.0], [1.0])
