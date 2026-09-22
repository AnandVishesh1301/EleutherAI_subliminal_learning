import numpy as np
import pytest

from subliminal_learning.stats.transfer_pool_robustness import (
    simulate_null_rejection,
    whole_pool_permutation,
)
from subliminal_learning.stats.transfer_significance_tests import stratified_exceedance_permutation


def test_correlated_two_pool_exact_and_ties():
    values = np.array([[3.0, 1.0], [3.0, 1.0]])
    result = whole_pool_permutation([values], [1], pool_tags=["t", "c"])
    assert result.observed == 2
    assert result.p == 0.5
    assert result.n_labelings == 2
    ties = whole_pool_permutation([np.array([[2.0, 2.0]])], [1], pool_tags=["t", "c"])
    assert ties.p == 1.0


def test_shared_group_assignment_and_independent_groups():
    a = np.array([[3., 0.], [3., 0.]])
    b = np.array([[0., 3.], [0., 3.]])
    result = whole_pool_permutation([a, b], [1, 1], groups=["sender", "sender"], pool_tags=["x", "y"])
    assert result.n_labelings == 2
    independent = whole_pool_permutation([a, b], [1, 1], groups=["a", "b"], pool_tags=["x", "y"])
    assert independent.n_labelings == 4


def test_shared_group_scores_every_arm_in_unequal_groups():
    a = np.array([[3., 0.], [3., 0.]])
    b = np.array([[3., 0.], [3., 0.]])
    c = np.array([[0., 3.], [0., 3.]])
    result = whole_pool_permutation([a, b, c], [1, 1, 1], groups=["g", "g", "h"])
    assert result.observed == 4


def test_incomplete_and_mismatched_inputs_rejected():
    with pytest.raises(ValueError, match="same pool tags"):
        whole_pool_permutation([np.ones((2, 2)), np.ones((2, 3))], [0, 0])
    with pytest.raises(ValueError, match="one index per arm"):
        whole_pool_permutation([np.ones((2, 2))], [])


def test_null_simulation_schema():
    result = simulate_null_rejection(rho=0.75, n_replicates=3, n_perm=9, seed=2)
    assert result["n_replicates"] == 3
    assert 0 <= result["cell_rejection_rate"] <= 1


def test_identical_cells_exact_resolution_matches_formula():
    values = np.tile(np.array([[1.0, 0.0]]), (20, 1))
    current = stratified_exceedance_permutation(values, 1, n_perm=20000, seed=3)
    pooled = whole_pool_permutation([values], [1], pool_tags=["teacher", "control"])
    assert current.observed == 20
    assert abs(current.p - 0.5**20) < 0.002
    assert pooled.p == 0.5


def test_pool_resolution_toy_null_floors():
    values = np.tile(np.arange(11.0), (20, 1))
    result = whole_pool_permutation([values], [0])
    assert result.p == 1 / 11
    assert result.p > 0.05
    values2 = np.tile(np.array([[1.0, 0.0]]), (20, 1))
    assert whole_pool_permutation([values2], [1]).p == 0.5



def test_frozen_replay_keeps_families_separate(tmp_path):
    from scripts.transfer_pool_robustness import reproduce_frozen

    np.savez(tmp_path / "matrices.npz",
             values=np.array([[[3., 0.], [3., 0.]], [[0., 3.], [0., 3.]]]),
             arms=np.array(["a→b", "c→d"]), families=np.array(["A", "B"]),
             senders=np.array(["a", "c"]), cells=np.array(["c1", "c2"]),
             pools=np.array(["teacher", "control"]), control_index=np.array(1))
    result = reproduce_frozen(tmp_path)
    assert result["families"]["A"]["pool_observed"] == 2
    assert result["families"]["B"]["pool_observed"] == 0
    assert result["families"]["A"]["pool_p"] == .5
    assert result["families"]["B"]["pool_p"] == 1.
    assert result["families"]["A"]["pool_exact"]
