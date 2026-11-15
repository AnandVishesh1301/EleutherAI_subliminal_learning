"""Pure permutation / sign / exceedance statistics in `stats.transfer_significance_tests` (hand-checkable values; no disk).
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.stats import transfer_significance_tests as tt


# -------------------- from stats_a.py --------------------

# ----------------------------------------------------------------------------------------------
# exceedance
# ----------------------------------------------------------------------------------------------
def _exceedance_with(k: int, m: int = 56, n: int = 56):
    """A treatment/control pair where exactly ``k`` treatment values beat every control."""
    control = np.linspace(0.0, 1.0, n)
    treatment = np.concatenate([np.full(k, 2.0), np.linspace(0.0, 0.5, m - k)])
    return tt.exceedance_test(treatment, control)

def test_exceedance_exact_values():
    assert _exceedance_with(1).p == pytest.approx(56 / 112)
    # The exact product 56/112 * 55/111 * 54/110 * 53/109 * 52/108 = 0.0284734, which is the
    # brief's 0.02857 to 0.3% (0.02857 = 1/35, the equal-sized-samples approximation).
    assert _exceedance_with(5).p == pytest.approx(0.0284734, rel=1e-5)
    assert _exceedance_with(5).p == pytest.approx(0.02857, rel=5e-3)
    zero = _exceedance_with(0)
    assert zero.p == 1.0 and zero.count == 0
    assert zero.expected == pytest.approx(56 / 57)
    assert (zero.m, zero.n) == (56, 56)

def test_exceedance_strict_ties():
    assert tt.exceedance_test([1.0, 1.0], [1.0, 0.5]).count == 0

# ----------------------------------------------------------------------------------------------
# paired sign test
# ----------------------------------------------------------------------------------------------
def test_paired_sign_test_46_of_56():
    treatment = np.array([1.0] * 46 + [0.0] * 10)
    control = np.zeros(56)
    control[46:] = 1.0
    res = tt.paired_sign_test(treatment, control)
    assert (res.wins, res.n) == (46, 56)
    assert res.p == pytest.approx(6.2e-7, rel=5e-2)

def test_paired_sign_test_ties_and_nans():
    res = tt.paired_sign_test([1.0, 0.0, 0.0, np.nan], [0.0, 0.0, 1.0, 0.0])
    assert (res.wins, res.n) == (1, 2)  # tie dropped, NaN pair dropped

# ----------------------------------------------------------------------------------------------
# stratified exceedance permutation
# ----------------------------------------------------------------------------------------------
def _separated_matrix(cells: int = 56, pools: int = 12) -> np.ndarray:
    rng = np.random.default_rng(3)
    values = rng.uniform(1.0, 2.0, size=(cells, pools))
    values[:, pools - 1] = 0.0  # the control column is smallest in every cell
    return values

def test_stratified_permutation_separated_hits_floor():
    res = tt.stratified_exceedance_permutation(_separated_matrix(), 11, n_perm=2000, seed=0)
    assert res.observed == 56 * 11
    assert res.p == res.floor == pytest.approx(1 / 2001)

def test_stratified_permutation_iid_is_null():
    values = np.random.default_rng(0).normal(size=(56, 12))
    res = tt.stratified_exceedance_permutation(values, 11, n_perm=2000, seed=1)
    assert res.p > 0.05

def test_stratified_permutation_nan_tolerant_and_deterministic():
    values = _separated_matrix()
    values[0, 3] = np.nan
    values[5, 11] = np.nan
    values[5, 0] = np.nan
    a = tt.stratified_exceedance_permutation(values, 11, n_perm=500, seed=7)
    b = tt.stratified_exceedance_permutation(values, 11, n_perm=500, seed=7)
    assert a == b
    # Threshold = nanmax over the control column = 0, so every finite non-control entry
    # exceeds it: 56*11 minus the two NaN treatment entries.
    assert a.observed == 56 * 11 - 2

def _two_group_matrix():
    """Group A has a huge control max, group B a low one; B's values sit between the two."""
    values = np.zeros((8, 3))
    values[:4] = [[0.2, 0.2, 0.9]]  # group A: control 0.9 dominates everything
    values[4:] = [[0.5, 0.6, 0.1]]  # group B: both treatments beat its own control 0.1
    groups = np.array(["A"] * 4 + ["B"] * 4)
    return values, groups

def test_grouped_permutation_uses_per_group_control_max():
    values, groups = _two_group_matrix()
    pooled = tt.stratified_exceedance_permutation(values, 2, n_perm=500, seed=0)
    grouped = tt.stratified_exceedance_permutation(values, 2, groups=groups, n_perm=500, seed=0)
    assert pooled.observed == 0  # the global max 0.9 is beaten by nothing
    assert grouped.observed == 8  # group B's 4 cells x 2 treatments, judged against 0.1
    assert grouped.observed > pooled.observed

def test_grouped_permutation_group_counts_sum_and_are_deterministic():
    values, groups = _two_group_matrix()
    a = tt.stratified_exceedance_permutation(values, 2, groups=groups, n_perm=500, seed=3)
    b = tt.stratified_exceedance_permutation(values, 2, groups=groups, n_perm=500, seed=3)
    assert a == b
    assert a.observed_by_group == {"A": 0, "B": 8}
    assert sum(a.observed_by_group.values()) == a.observed

def test_groups_none_matches_single_group_and_carries_no_group_counts():
    values = _separated_matrix()
    plain = tt.stratified_exceedance_permutation(values, 11, n_perm=500, seed=5)
    one = tt.stratified_exceedance_permutation(values, 11, groups=np.zeros(56, dtype=int),
                                               n_perm=500, seed=5)
    assert plain.observed_by_group == {}
    assert (plain.observed, plain.p, plain.null_quantiles) == (one.observed, one.p,
                                                               one.null_quantiles)

# ----------------------------------------------------------------------------------------------
# prior_scaled_effect
# ----------------------------------------------------------------------------------------------
def test_prior_scaled_effect():
    assert tt.prior_scaled_effect([0.3, 0.2], [0.1, 0.1], 0.1) == pytest.approx(1.5)
    assert np.isnan(tt.prior_scaled_effect([np.nan], [0.1], 0.1))

def test_no_multiplicity_helpers():
    """The brief's amendment: no Holm/Bonferroni wrappers live here."""
    assert not hasattr(tt, "holm") and not hasattr(tt, "bonferroni")

# ----------------------------------------------------------------------------------------------
# the null distribution carried on the result
# ----------------------------------------------------------------------------------------------
def test_null_hist_is_lossless_and_reproduces_the_quantiles():
    """`null_hist` holds every draw (its frequencies sum to n_perm) and `null_percentile`
    reads the same 95th percentile the quantile summary reports."""
    values = _separated_matrix()
    r = tt.stratified_exceedance_permutation(values, 11, n_perm=500, seed=2)
    assert sum(f for _, f in r.null_hist) == r.n_perm
    assert [c for c, _ in r.null_hist] == sorted(c for c, _ in r.null_hist)
    assert tt.null_counts(r).size == r.n_perm
    assert tt.null_percentile(r, 95) == r.null_quantiles["95"]
    assert tt.null_percentile(r, 50) == r.null_quantiles["50"]

def _swap_frame(rows):
    """``(family, arm, stratum, value)`` rows as the arm-relabel frame."""
    return pd.DataFrame(rows, columns=["family", "arm", "stratum", "value"])

def test_p_opposite_is_the_swapped_families_p_on_the_exact_path():
    """`p_opposite` reads the same null for B > A: swapping the families' labels turns it into
    the forward `p` (exact enumeration, symmetric null). The observed labelling is itself
    enumerated and sits in both tails, so ``p + p_opposite == 1 + #{null == observed} / n``;
    distinct values keep every other labelling off the observed, giving exactly ``1 + 1/n``."""
    rows = [("A", "a1", "s", 3.0), ("A", "a2", "s", 5.0), ("A", "a3", "s", 4.0),
            ("B", "b1", "s", 1.0), ("B", "b2", "s", 2.0), ("B", "b3", "s", 0.5)]
    frame = _swap_frame(rows)
    fwd = tt.arm_set_permutation(frame, family_a="A", family_b="B")
    rev = tt.arm_set_permutation(frame, family_a="B", family_b="A")
    assert fwd.exact and rev.exact
    assert fwd.p_opposite == rev.p and rev.p_opposite == fwd.p
    assert fwd.p + fwd.p_opposite == pytest.approx(1.0 + 1 / fwd.n_labelings)
    assert fwd.p_opposite == 1.0 and fwd.p == pytest.approx(1 / 20)
    for statistic in tt.ARM_RELABEL_STATISTICS:
        f = tt.arm_relabel_cell_permutation(frame, family_a="A", family_b="B", statistic=statistic)
        r = tt.arm_relabel_cell_permutation(frame, family_a="B", family_b="A", statistic=statistic)
        assert f.exact and f.p_opposite == r.p and r.p_opposite == f.p
        assert f.p + f.p_opposite >= 1.0 + 1 / f.n_labelings

def test_p_opposite_monte_carlo_uses_the_same_draws():
    """Monte-Carlo path: ``(#{null <= observed} + 1) / (n_perm + 1)``, so both tails carry the
    +1 and ``p + p_opposite >= 1 + 1/(n_perm+1)``."""
    rng = np.random.default_rng(0)
    rows = [("A" if i < 12 else "B", f"x{i}", "s", float(v)) for i, v in enumerate(rng.random(24))]
    res = tt.arm_set_permutation(_swap_frame(rows), family_a="A", family_b="B", max_exact=10,
                                 n_perm=500, seed=1)
    assert not res.exact
    assert res.p + res.p_opposite >= 1 + 1 / 501 - 1e-12

def test_p_opposite_round_trips_through_the_memo_and_stale_entries_recompute(tmp_path, monkeypatch):
    """A stored result comes back with `p_opposite` and the two-sided trio; an entry written
    before a field existed (no such key) fails to rebuild and is recomputed rather than
    crashing."""
    from subliminal_learning.stats import memo

    monkeypatch.setenv(memo.ENV_VAR, "1")
    memo.set_cache_dir(tmp_path)
    frame = _swap_frame([("A", "a1", "s", 3.0), ("A", "a2", "s", 5.0),
                         ("B", "b1", "s", 1.0), ("B", "b2", "s", 2.0)])
    first = tt.arm_set_permutation(frame, family_a="A", family_b="B")
    memo.reset_cache_stats()
    again = tt.arm_set_permutation(frame, family_a="A", family_b="B")
    assert memo.cache_stats()["hits"] == 1 and again == first and again.p_opposite == first.p_opposite
    assert again.p_two_sided == first.p_two_sided and again.p_upper == first.p_upper
    assert again.p_lower == first.p_lower
    entries = list((tmp_path / "arm_set_permutation").glob("*.json"))
    assert len(entries) == 1
    for field in ("p_opposite", "p_upper", "p_lower", "p_two_sided"):
        stale = json.loads(entries[0].read_text())
        del stale[field]
        entries[0].write_text(json.dumps(stale))
        memo.reset_cache_stats()
        assert tt.arm_set_permutation(frame, family_a="A", family_b="B") == first
        assert memo.cache_stats()["misses"] == 1

def _brute_force_centred_two_sided(null: np.ndarray, observed: float) -> float:
    """The ``|T - centre|`` two-sided p on the same enumeration, centre = the null mean."""
    centre = float(null.mean())
    return float((np.abs(null - centre) >= abs(observed - centre) - 1e-12).mean())

def test_two_sided_p_on_a_symmetric_paired_null_is_the_doubled_and_the_centred_p():
    """Paired teacher swap, full coverage: the null is symmetric, so the doubled two-sided p
    equals ``2 * p_upper`` (observed in the upper tail) AND the brute-force ``|T - centre|`` p
    computed on the same enumeration."""
    rows = []
    for k, (a, b) in enumerate([(0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (1.0, 0.4)]):
        for cell in ("c1", "c2"):
            rows.append({"condition": "A", "teacher": f"t{k}", "cell": cell, "value": a})
            rows.append({"condition": "B", "teacher": f"t{k}", "cell": cell, "value": b})
    frame = pd.DataFrame(rows)
    res = tt.paired_cluster_relabel_permutation(frame, cond_a="A", cond_b="B")
    assert res.exact and res.n_labelings == 2 ** 4
    assert res.p == res.p_upper and res.p_opposite == res.p_lower
    assert res.p_upper < res.p_lower
    assert res.p_two_sided == pytest.approx(min(1.0, 2 * res.p_upper))
    assert res.p_two_sided <= 1.0
    # the same enumeration, read the |T - centre| way: symmetric null, so the two agree
    sign_vectors = np.array([[1.0 if not (code >> i) & 1 else -1.0 for i in range(4)]
                             for code in range(16)])
    diffs = np.array([0.8, 0.6, 0.4, 0.6])
    null = (sign_vectors * diffs).mean(axis=1)
    assert res.observed == pytest.approx(float(diffs.mean()))
    assert res.p_two_sided == pytest.approx(_brute_force_centred_two_sided(null, res.observed))

def test_two_sided_p_on_an_asymmetric_arm_relabel_null_doubles_the_smaller_tail():
    """3 arms against 5: the arm-relabel null is asymmetric, so the doubled p is
    ``min(1, 2 * min(p_upper, p_lower))`` and DIFFERS from the ``|T - centre|`` version -- which
    is why doubling is what the comparisons report."""
    values = [3.0, 3.2, 3.4, 1.0, 1.1, 1.2, 1.3, 5.0]
    rows = [("A" if i < 3 else "B", f"x{i}", "s", v) for i, v in enumerate(values)]
    res = tt.arm_set_permutation(_swap_frame(rows), family_a="A", family_b="B")
    assert res.exact and res.n_a == 3 and res.n_b == 5
    assert res.p == res.p_upper and res.p_opposite == res.p_lower
    assert res.p_two_sided == pytest.approx(min(1.0, 2 * min(res.p_upper, res.p_lower)))
    assert res.p_two_sided <= 1.0
    # brute force on the same enumeration: mean(A) - mean(B) over every 3-of-8 labelling
    v = np.array(values)
    picks = np.array(list(itertools.combinations(range(8), 3)))
    null = v[picks].sum(axis=1) * (1 / 3 + 1 / 5) - v.sum() / 5
    assert res.p_upper == pytest.approx(float((null >= res.observed).mean()))
    centred = _brute_force_centred_two_sided(null, res.observed)
    assert res.p_two_sided != pytest.approx(centred)

@pytest.mark.parametrize("statistic", tt.ARM_RELABEL_STATISTICS)
def test_exact_path_counts_the_observed_labelling_in_both_tails(statistic):
    """The observed labelling is itself enumerated, and it is read out of the enumeration, so it
    lands in BOTH tails: ``p_upper + p_lower >= 1 + 1/n_labelings``. Holds with values TIED
    across arms, where several labellings share the observed statistic."""
    tied = [("A", "a1", "s", 1.0), ("A", "a2", "s", 1.0), ("A", "a3", "s", 2.0),
            ("B", "b1", "s", 1.0), ("B", "b2", "s", 2.0), ("B", "b3", "s", 1.0)]
    for rows in (tied, [(f, a, c, v + 0.1 * i) for i, (f, a, c, v) in enumerate(tied)]):
        frame = _swap_frame(rows)
        res = tt.arm_relabel_cell_permutation(frame, family_a="A", family_b="B",
                                              statistic=statistic)
        assert res.exact
        assert res.p_upper + res.p_lower >= 1.0 + 1 / res.n_labelings - 1e-12
        assert res.p_two_sided == pytest.approx(min(1.0, 2 * min(res.p_upper, res.p_lower)))
        arm_set = tt.arm_set_permutation(frame, family_a="A", family_b="B")
        assert arm_set.p_upper + arm_set.p_lower >= 1.0 + 1 / arm_set.n_labelings - 1e-12

def test_two_sided_p_on_the_monte_carlo_path_doubles_the_plus_one_tails():
    """Monte Carlo: each tail is ``(count + 1) / (n_perm + 1)`` and the two-sided p doubles the
    smaller of the two; the run-relabel test (MC only) reports the same trio."""
    rng = np.random.default_rng(3)
    rows = [("A" if i < 10 else "B", f"x{i}", "s", float(v))
            for i, v in enumerate(rng.random(22) + np.repeat([0.4, 0.0], [10, 12]))]
    frame = _swap_frame(rows)
    mc = tt.arm_set_permutation(frame, family_a="A", family_b="B", max_exact=10, n_perm=400,
                               seed=2)
    assert not mc.exact and mc.p == mc.p_upper and mc.p_opposite == mc.p_lower
    assert mc.p_two_sided == pytest.approx(min(1.0, 2 * min(mc.p_upper, mc.p_lower)))
    assert mc.p_upper >= 1 / 401 and mc.p_lower >= 1 / 401 and mc.p_two_sided <= 1.0
    run = tt.run_relabel_cell_permutation(frame, family_a="A", family_b="B", n_perm=400, seed=2)
    assert run.p == run.p_upper and run.p_opposite == run.p_lower
    assert run.p_two_sided == pytest.approx(min(1.0, 2 * min(run.p_upper, run.p_lower)))
    assert run.p_two_sided <= 1.0

# ----------------------------------------------------------------------------------------------
# within-pair sign-flip permutation
# ----------------------------------------------------------------------------------------------
def test_sign_flip_permutation_all_positive_diffs_sit_at_the_floor():
    res = tt.paired_sign_flip_permutation(np.full(30, 0.2), n_perm=500, seed=0)
    assert res.observed == pytest.approx(0.2)
    assert res.p == pytest.approx(res.floor) == pytest.approx(1 / 501)
    assert res.p_opposite == 1.0 and res.n_pairs == 30 and res.n_perm == 500

def test_sign_flip_permutation_symmetric_diffs_are_a_coin_flip():
    diffs = np.r_[np.linspace(0.1, 1.0, 20), -np.linspace(0.1, 1.0, 20)]
    res = tt.paired_sign_flip_permutation(diffs, n_perm=5000, seed=0)
    assert res.observed == pytest.approx(0.0, abs=1e-12)
    assert res.p == pytest.approx(0.5, abs=0.05)
    assert res.p_opposite == pytest.approx(0.5, abs=0.05)

def test_sign_flip_permutation_is_seeded_and_drops_nan_pairs():
    diffs = [0.3, -0.1, np.nan, 0.05]
    first = tt.paired_sign_flip_permutation(diffs, n_perm=400, seed=3)
    assert first == tt.paired_sign_flip_permutation(diffs, n_perm=400, seed=3)
    assert first != tt.paired_sign_flip_permutation(diffs, n_perm=400, seed=4)
    assert first.observed == pytest.approx(np.mean([0.3, -0.1, 0.05]))
    assert first.n_pairs == 3
    empty = tt.paired_sign_flip_permutation([np.nan, np.nan], n_perm=50)
    assert np.isnan(empty.observed) and empty.p == 1.0 and empty.n_pairs == 0

# ----------------------------------------------------------------------------------------------
# run-relabel (the runs shuffled inside their cell, both families pooled over arms)
# ----------------------------------------------------------------------------------------------
def _run_frame(a_values, b_values, cells=("c1", "c2")) -> pd.DataFrame:
    """One row per run, `a_values` for family A and `b_values` for family B in every cell."""
    rows = [{"family": f, "stratum": cell, "value": v}
            for cell in cells for f, vals in (("A", a_values), ("B", b_values)) for v in vals]
    return pd.DataFrame(rows)

@pytest.mark.parametrize("statistic", tt.RUN_RELABEL_STATISTICS)
def test_run_relabel_all_larger_family_sits_at_the_floor(statistic):
    frame = _run_frame(np.linspace(5.0, 6.0, 8), np.linspace(0.0, 1.0, 8))
    res = tt.run_relabel_cell_permutation(frame, family_a="A", family_b="B",
                                          statistic=statistic, n_perm=500)
    assert res.p == pytest.approx(res.floor) == pytest.approx(1 / 501)
    assert res.p_opposite == 1.0 and not res.exact
    assert (res.n_a, res.n_b) == (16, 16) and res.n_cells == 2   # RUNS, not arms
    assert res.n_labelings == 500 and res.statistic == statistic

def test_run_relabel_identical_families_are_a_coin_flip():
    values = np.linspace(0.0, 1.0, 25)
    res = tt.run_relabel_cell_permutation(_run_frame(values, values), family_a="A", family_b="B",
                                          n_perm=4000)
    assert res.observed == pytest.approx(0.0, abs=1e-12)
    assert res.p == pytest.approx(0.5, abs=0.06)

def test_run_relabel_is_seeded_and_drops_single_family_strata():
    frame = pd.concat([_run_frame([0.4, 0.5, 0.6], [0.1, 0.2, 0.3]),
                       pd.DataFrame({"family": ["A"] * 3, "stratum": "lonely",
                                     "value": [9.0, 9.1, 9.2]})], ignore_index=True)
    res = tt.run_relabel_cell_permutation(frame, family_a="A", family_b="B", n_perm=300, seed=1)
    assert res.n_cells == 2 and res.n_cells_dropped == 1
    assert (res.n_a, res.n_b) == (6, 6)                  # the lonely stratum's runs are gone
    assert res.observed == pytest.approx(0.3)
    same = tt.run_relabel_cell_permutation(frame, family_a="A", family_b="B", n_perm=300, seed=1)
    other = tt.run_relabel_cell_permutation(frame, family_a="A", family_b="B", n_perm=300, seed=2)
    assert res.p == same.p and res.p != other.p

def test_run_relabel_pooled_mean_on_an_indicator_is_the_fraction_difference():
    frame = _run_frame([1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], cells=("c1",))
    res = tt.run_relabel_cell_permutation(frame, family_a="A", family_b="B",
                                          statistic="pooled_mean", n_perm=200)
    assert res.observed == pytest.approx(0.5 - 0.25)

def test_run_relabel_rejects_an_unknown_statistic_and_a_one_family_design():
    with pytest.raises(ValueError, match="statistic"):
        tt.run_relabel_cell_permutation(_run_frame([1.0], [0.0]), family_a="A", family_b="B",
                                        statistic="nope")
    lonely = pd.DataFrame({"family": ["A"] * 3, "stratum": "c1", "value": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="both families"):
        tt.run_relabel_cell_permutation(lonely, family_a="A", family_b="B", n_perm=50)

def test_run_relabel_is_invariant_to_row_order():
    """The draws are taken in sorted stratum order, so a shuffled frame is the same null."""
    rng = np.random.default_rng(0)
    frame = _run_frame(rng.normal(0.3, 0.1, 7), rng.normal(0.0, 0.1, 5), cells=("c2", "c1", "c3"))
    shuffled = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    for statistic in tt.RUN_RELABEL_STATISTICS:
        kw = dict(family_a="A", family_b="B", statistic=statistic, n_perm=300, seed=5)
        first = tt.run_relabel_cell_permutation(frame, **kw)
        assert first == tt.run_relabel_cell_permutation(shuffled, **kw)

def _reference_paired_cluster_tables(df, condition_col, cond_a, cluster_col, stratum_col,
                                     value_col, statistic):
    """The original per-cell row loop `_paired_cluster_tables` was vectorized from."""
    from scipy.stats import rankdata

    clusters = sorted(pd.unique(df[cluster_col]), key=str)
    cells = sorted(pd.unique(df[stratum_col]), key=str)
    row = {c: i for i, c in enumerate(clusters)}
    shape = (len(clusters), len(cells))
    a_sums, a_counts = np.zeros(shape), np.zeros(shape)
    b_sums, b_counts = np.zeros(shape), np.zeros(shape)
    for j, cell in enumerate(cells):
        block = df[df[stratum_col] == cell]
        values = block[value_col].to_numpy(dtype=float)
        weights = rankdata(values) if statistic == "win_probability" else values
        is_a = (block[condition_col] == cond_a).to_numpy()
        for cluster, w, a in zip(block[cluster_col].to_numpy(), weights, is_a):
            sums, counts = (a_sums, a_counts) if a else (b_sums, b_counts)
            sums[row[cluster], j] += float(w)
            counts[row[cluster], j] += 1.0
    return a_sums, a_counts, b_sums, b_counts

@pytest.mark.parametrize("labels", ["int", "str"])
@pytest.mark.parametrize("statistic", ["mean_contrast", "win_probability"])
def test_paired_cluster_tables_match_row_loop_reference(labels, statistic):
    rng = np.random.default_rng(7)
    for trial in range(25):
        n = int(rng.integers(1, 80))
        clusters = rng.integers(0, 12, n)              # 10 vs 2 sorts differently by str
        frame = pd.DataFrame({
            "condition": rng.choice(["A", "B"], n),
            "teacher": clusters if labels == "int" else [f"t{c}" for c in clusters],
            "cell": rng.choice(["c10", "c2", "c1", "x"], n),
            # coarse values -> many ties; a few repeated exact floats too
            "value": np.round(rng.normal(size=n), 1) if trial % 2 else rng.normal(size=n),
        }).sample(frac=1.0, random_state=trial)       # unbalanced, shuffled slots
        args = (frame, "condition", "A", "teacher", "cell", "value", statistic)
        got = tt._paired_cluster_tables(*args)
        want = _reference_paired_cluster_tables(*args)
        for g, w in zip(got, want):
            assert g.dtype == np.float64 and g.shape == w.shape
            assert np.array_equal(g, w)


# -------------------- from stats_b.py --------------------

# ----------------------------------------------------------------------------------------------
# arm-level arm-set permutation
# ----------------------------------------------------------------------------------------------
def _arm_frame(rows) -> pd.DataFrame:
    """``(family, value)`` rows as a one-row-per-arm frame."""
    return pd.DataFrame(rows, columns=["family", "value"])

def test_arm_permutation_exact_two_vs_two():
    """Both A arms above both B arms: 1 of C(4,2) = 6 labelings is that extreme."""
    res = tt.arm_set_permutation(
        _arm_frame([("A", 3.0), ("A", 2.0), ("B", 1.0), ("B", 0.0)]),
        family_a="A", family_b="B")
    assert res.exact and res.n_labelings == 6
    assert res.p == pytest.approx(1 / 6) == res.floor
    assert res.observed == pytest.approx(2.5 - 0.5)
    assert (res.n_a, res.n_b) == (2, 2)

def test_arm_permutation_identical_values_is_p_one():
    res = tt.arm_set_permutation(
        _arm_frame([("A", 1.0), ("A", 1.0), ("B", 1.0), ("B", 1.0)]),
        family_a="A", family_b="B")
    assert res.observed == pytest.approx(0.0) and res.p == 1.0

def test_arm_permutation_unequal_counts_enumerate():
    res = tt.arm_set_permutation(
        _arm_frame([("A", 3.0), ("A", 2.0), ("B", 1.0)]), family_a="A", family_b="B")
    assert res.observed == pytest.approx(2.5 - 1.0)
    assert (res.n_a, res.n_b) == (2, 1) and res.n_labelings == 3

def test_arm_permutation_drops_non_finite_values():
    res = tt.arm_set_permutation(
        _arm_frame([("A", 3.0), ("A", np.nan), ("B", 1.0), ("B", 0.0)]),
        family_a="A", family_b="B")
    assert (res.n_a, res.n_b) == (1, 2) and res.n_labelings == 3

def test_arm_permutation_monte_carlo_above_the_cap_and_deterministic():
    rows = [("A" if i < 6 else "B", float(12 - i)) for i in range(12)]
    kw = dict(family_a="A", family_b="B", max_exact=100, n_perm=500, seed=0)
    res = tt.arm_set_permutation(_arm_frame(rows), **kw)
    assert not res.exact and res.n_labelings == 500
    assert res.p == pytest.approx(1 / 501) == res.floor
    assert tt.arm_set_permutation(_arm_frame(rows), **kw) == res

def test_arm_permutation_needs_both_families():
    with pytest.raises(ValueError, match="both families"):
        tt.arm_set_permutation(_arm_frame([("A", 1.0)]), family_a="A", family_b="B")

# ----------------------------------------------------------------------------------------------
# arm-relabel permutation on runs within cells
# ----------------------------------------------------------------------------------------------
def _family_run_frame(n_arms_per_family: int = 3, n_cells: int = 4, n_teachers: int = 2,
               shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """One row per (arm, cell, teacher); family A shifted up by `shift`."""
    rng = np.random.default_rng(seed)
    rows = []
    for family, bump in (("A", shift), ("B", 0.0)):
        for a in range(n_arms_per_family):
            for c in range(n_cells):
                for t in range(n_teachers):
                    rows.append({"family": family, "arm": f"{family}{a}", "stratum": f"c{c}",
                                 "value": rng.normal() + bump + 10.0 * c + 0.5 * a})
                    del t
    return pd.DataFrame(rows)

def _brute_force(frame: pd.DataFrame, statistic: str) -> tuple[float, float, int]:
    """``(observed, p, n_labelings)`` from a plain Python loop over every arm labelling."""
    import itertools

    arms = list(dict.fromkeys(frame["arm"]))
    cells = list(dict.fromkeys(frame["stratum"]))
    obs_a = {a for a in arms if frame[frame["arm"] == a]["family"].iloc[0] == "A"}

    def stat(a_arms: set) -> float:
        per_cell = []
        for cell in cells:
            block = frame[frame["stratum"] == cell]
            a = block[block["arm"].isin(a_arms)]["value"].tolist()
            b = block[~block["arm"].isin(a_arms)]["value"].tolist()
            if not a or not b:
                continue
            if statistic == "mean_contrast":
                per_cell.append(sum(a) / len(a) - sum(b) / len(b))
            else:
                wins = sum(1.0 if x > y else 0.5 if x == y else 0.0 for x in a for y in b)
                per_cell.append(wins / (len(a) * len(b)))
        return sum(per_cell) / len(per_cell)

    observed = stat(obs_a)
    labelings = list(itertools.combinations(arms, len(obs_a)))
    null = [stat(set(pick)) for pick in labelings]
    return observed, sum(v >= observed for v in null) / len(null), len(labelings)

@pytest.mark.parametrize("statistic", ["mean_contrast", "win_probability"])
def test_arm_relabel_matches_a_brute_force_loop(statistic):
    frame = _family_run_frame(shift=1.0, seed=3)
    res = tt.arm_relabel_cell_permutation(frame, family_a="A", family_b="B", statistic=statistic)
    observed, p, n_labelings = _brute_force(frame, statistic)
    assert res.observed == pytest.approx(observed)
    assert res.p == pytest.approx(p) and res.n_labelings == n_labelings == 20
    assert res.exact and (res.n_a, res.n_b) == (3, 3)
    assert (res.n_cells, res.n_cells_dropped) == (4, 0)
    assert res.floor == pytest.approx(1 / 20)

def test_win_probability_counts_ties_as_a_half():
    """One cell, 2 A runs vs 2 B runs: one A win, two ties, one A loss -> 0.5."""
    rows = [("A", "a1", 2.0), ("A", "a2", 1.0), ("B", "b1", 1.0), ("B", "b2", 3.0)]
    frame = pd.DataFrame([{"family": f, "arm": a, "stratum": "c1", "value": v}
                          for f, a, v in rows])
    res = tt.arm_relabel_cell_permutation(frame, family_a="A", family_b="B",
                                          statistic="win_probability")
    # pairs: (2,1) win, (2,3) loss, (1,1) tie, (1,3) loss -> (1 + 0.5) / 4
    assert res.observed == pytest.approx(0.375)

def test_mean_contrast_equals_the_arm_set_permutation_under_full_coverage():
    frame = _family_run_frame(shift=0.7, seed=5)
    res = tt.arm_relabel_cell_permutation(frame, family_a="A", family_b="B")
    arm_means = (frame.groupby(["family", "arm"], sort=False)["value"].mean()
                 .reset_index().rename(columns={"value": "value"}))
    ref = tt.arm_set_permutation(arm_means, family_a="A", family_b="B")
    assert res.observed == pytest.approx(ref.observed)
    assert res.p == pytest.approx(ref.p) and res.n_labelings == ref.n_labelings

def test_arm_relabel_drops_single_family_cells():
    frame = _family_run_frame(seed=1)
    frame = frame[~((frame["stratum"] == "c2") & (frame["family"] == "B"))]
    res = tt.arm_relabel_cell_permutation(frame, family_a="A", family_b="B")
    assert (res.n_cells, res.n_cells_dropped) == (3, 1)

def test_arm_relabel_monte_carlo_path_and_determinism():
    frame = _family_run_frame(n_arms_per_family=3, shift=1.0, seed=2)
    kw = dict(family_a="A", family_b="B", max_exact=5, n_perm=500, seed=0)
    res = tt.arm_relabel_cell_permutation(frame, **kw)
    assert not res.exact and res.n_labelings == 500 and res.floor == pytest.approx(1 / 501)
    assert tt.arm_relabel_cell_permutation(frame, **kw) == res

def test_arm_relabel_rejects_an_unknown_statistic():
    with pytest.raises(ValueError, match="statistic must be one of"):
        tt.arm_relabel_cell_permutation(_family_run_frame(), family_a="A", family_b="B",
                                        statistic="median")


# -------------------- from stats_c.py --------------------

def test_wins_a_on_a_constructed_frame():
    """`wins_a` is the sign test's win count: ties are not wins."""
    a, b = [0.3, 0.2, 0.1, 0.4], [0.1, 0.3, 0.1, 0.2]
    assert tt.paired_sign_test(a, b).wins == 2 and tt.paired_sign_test(a, b).n == 3
