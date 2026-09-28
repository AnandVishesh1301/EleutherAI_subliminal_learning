"""Hand-checkable values for the paired cluster-relabel permutation and its one-sender loader."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import transfer_significance_io as tio
from subliminal_learning.stats import transfer_significance_tests as tt

TEACHERS = [f"t{i}" for i in range(6)]
CELLS = [f"c{j}" for j in range(4)]


def _paired_frame(shift: float = 0.5, *, teachers=TEACHERS, cells=CELLS,
                  seed: int = 0) -> pd.DataFrame:
    """Fully paired rows: every teacher under both conditions in every cell, A shifted by `shift`."""
    rng = np.random.default_rng(seed)
    rows = [{"condition": condition, "teacher": teacher, "cell": cell,
             "value": float(rng.normal() + (shift if condition == "cross" else 0.0))}
            for teacher in teachers for cell in cells for condition in ("cross", "same")]
    return pd.DataFrame(rows)


def _sign_flip_null(frame: pd.DataFrame, teachers=TEACHERS, cells=CELLS):
    """Brute-force teacher-level sign-flip null of the mean per-teacher mean difference."""
    diffs = []
    for teacher in teachers:
        block = frame[frame["teacher"] == teacher]
        diffs.append(np.mean([
            float(block[(block["cell"] == cell) & (block["condition"] == "cross")]["value"].iloc[0])
            - float(block[(block["cell"] == cell) & (block["condition"] == "same")]["value"].iloc[0])
            for cell in cells]))
    d = np.array(diffs)
    null = np.array([float((np.array(signs) * d).mean())
                     for signs in itertools.product([1, -1], repeat=d.size)])
    return float(d.mean()), null


def test_mean_contrast_is_the_exact_teacher_level_sign_flip_test():
    """(1) full coverage: the within-cell contrast IS the sign-flip test on per-teacher means."""
    frame = _paired_frame()
    res = tt.paired_cluster_relabel_permutation(frame, cond_a="cross", cond_b="same")
    observed, null = _sign_flip_null(frame)
    assert res.exact and res.n_labelings == 2 ** len(TEACHERS) == 64
    assert (res.n_clusters, res.n_cells, res.n_a, res.n_b) == (6, 4, 24, 24)
    assert res.observed == pytest.approx(observed)
    assert res.p == pytest.approx(float((null >= observed).mean()))
    assert res.p_opposite == pytest.approx(float((null <= observed).mean()))
    assert res.floor == pytest.approx(1 / 64)


def test_win_probability_is_the_per_cell_mann_whitney_averaged_over_cells():
    """(2) the observed W equals scipy's U / (n_A n_B) per cell, averaged over cells."""
    from scipy.stats import mannwhitneyu

    frame = _paired_frame()
    res = tt.paired_cluster_relabel_permutation(frame, cond_a="cross", cond_b="same",
                                                statistic="win_probability")
    per_cell = []
    for cell in CELLS:
        block = frame[frame["cell"] == cell]
        a = block[block["condition"] == "cross"]["value"].to_numpy()
        b = block[block["condition"] == "same"]["value"].to_numpy()
        per_cell.append(mannwhitneyu(a, b, alternative="two-sided").statistic / (a.size * b.size))
    assert res.observed == pytest.approx(float(np.mean(per_cell)))


@pytest.mark.parametrize("statistic", tt.ARM_RELABEL_STATISTICS)
def test_swapping_the_conditions_mirrors_the_statistic_and_the_tails(statistic):
    """(3) A<->B negates the contrast (1-W for the win probability) and swaps p with p_opposite."""
    frame = _paired_frame()
    kw = dict(statistic=statistic)
    forward = tt.paired_cluster_relabel_permutation(frame, cond_a="cross", cond_b="same", **kw)
    reverse = tt.paired_cluster_relabel_permutation(frame, cond_a="same", cond_b="cross", **kw)
    mirrored = (1.0 - forward.observed if statistic == "win_probability" else -forward.observed)
    assert reverse.observed == pytest.approx(mirrored)
    assert (reverse.p, reverse.p_opposite) == pytest.approx((forward.p_opposite, forward.p))


@pytest.mark.parametrize("statistic, null_value", [("mean_contrast", 0.0),
                                                   ("win_probability", 0.5)])
def test_identical_conditions_sit_exactly_on_the_null(statistic, null_value):
    """(4) the same values under both labels: no flip moves the statistic, so p = 1."""
    rng = np.random.default_rng(1)
    rows = []
    for teacher in TEACHERS:
        for cell in CELLS:
            value = float(rng.normal())
            rows += [{"condition": c, "teacher": teacher, "cell": cell, "value": value}
                     for c in ("cross", "same")]
    res = tt.paired_cluster_relabel_permutation(pd.DataFrame(rows), cond_a="cross",
                                                cond_b="same", statistic=statistic)
    assert res.observed == pytest.approx(null_value)
    assert (res.p, res.p_opposite) == (1.0, 1.0)


def test_monte_carlo_path_tracks_the_exact_p_and_is_reproducible():
    """(5) max_exact below 2**6 samples instead; the sampled p sits within a few floors."""
    frame = _paired_frame()
    exact = tt.paired_cluster_relabel_permutation(frame, cond_a="cross", cond_b="same")
    mc = tt.paired_cluster_relabel_permutation(frame, cond_a="cross", cond_b="same",
                                               max_exact=8, n_perm=4000, seed=3)
    assert not mc.exact and mc.n_labelings == 4000 and mc.floor == pytest.approx(1 / 4001)
    assert mc.observed == pytest.approx(exact.observed)
    # The exact null has only 2**6 distinct values; the draws land on them, so the sampling
    # error is measured against the EXACT resolution, not the finer Monte-Carlo floor.
    assert abs(mc.p - exact.p) < 3 * exact.floor
    assert mc == tt.paired_cluster_relabel_permutation(frame, cond_a="cross", cond_b="same",
                                                       max_exact=8, n_perm=4000, seed=3)
    other = tt.paired_cluster_relabel_permutation(frame, cond_a="cross", cond_b="same",
                                                  max_exact=8, n_perm=4000, seed=4)
    assert other.observed == pytest.approx(exact.observed)


def test_unpaired_slots_are_dropped_and_a_one_sided_cluster_raises():
    """(6) a (teacher, cell) slot with one side only is no pair; a teacher with one side is fatal."""
    frame = _paired_frame()
    half = frame.drop(index=frame.index[(frame["teacher"] == "t0") & (frame["cell"] == "c0")
                                        & (frame["condition"] == "same")])
    res = tt.paired_cluster_relabel_permutation(half, cond_a="cross", cond_b="same")
    assert (res.n_pairs_dropped, res.n_cells_dropped, res.n_cells) == (1, 0, 4)
    assert (res.n_a, res.n_b) == (23, 23)     # the slot goes on BOTH sides, not one
    lonely = frame[~((frame["teacher"] == "t0") & (frame["condition"] == "same"))]
    with pytest.raises(ValueError, match="t0"):
        tt.paired_cluster_relabel_permutation(lonely, cond_a="cross", cond_b="same")


@pytest.mark.parametrize("statistic", tt.ARM_RELABEL_STATISTICS)
def test_row_order_never_moves_the_result(statistic):
    """(7) the same design read in a shuffled row order is the same result, field for field."""
    frame = _paired_frame()
    kw = dict(cond_a="cross", cond_b="same", statistic=statistic, max_exact=8, n_perm=500)
    assert (tt.paired_cluster_relabel_permutation(frame.sample(frac=1, random_state=7), **kw)
            == tt.paired_cluster_relabel_permutation(frame, **kw))


def test_unknown_statistic_missing_column_and_empty_condition_raise():
    frame = _paired_frame()
    with pytest.raises(ValueError, match="statistic must be"):
        tt.paired_cluster_relabel_permutation(frame, cond_a="cross", cond_b="same",
                                              statistic="nope")
    with pytest.raises(ValueError, match="no 'pool' column"):
        tt.paired_cluster_relabel_permutation(frame, cond_a="cross", cond_b="same",
                                              cluster_col="pool")
    with pytest.raises(ValueError, match="'other'"):
        tt.paired_cluster_relabel_permutation(frame, cond_a="cross", cond_b="other")


# ----------------------------------------------------------------------------------------------
# the one-sender loader
# ----------------------------------------------------------------------------------------------
def _matrix(offset: float) -> tio.CellMatrix:
    """Two teachers over 3 cells against one control, the treatment `offset` above it."""
    control = np.array([1.0, 2.0, 3.0])
    values = np.column_stack([control + offset, control + offset + 0.5, control])
    return tio.CellMatrix(frame=pd.DataFrame(), values=values, cells=["c0", "c1", "c2"],
                          pools=["t1", "t2", tio.CONTROL_LABEL], control_col=2, missing=[])


def test_paired_sender_frame_columns_and_priors(monkeypatch):
    """(8) both arms on the same cells, each gap over its OWN receiver's prior."""
    priors = {"pythia-160m": 0.01, "pythia-160m-pilerepro-seed1234": 0.02}
    monkeypatch.setattr(tio, "load_prior",
                        lambda animal, receiver, *a, **k: priors[tio.base_name(receiver)])
    monkeypatch.setattr(tio, "load_cell_matrix", lambda animal, sender, receiver, teachers, **k:
                        _matrix(1.0 if tio.base_name(receiver) == "pythia-160m" else 0.5))
    frame, report = tio.paired_sender_frame(
        "dog", "pythia-160m-pilerepro-seed1234", "pythia-160m", ["t1", "t2"], cells=["c0", "c1",
                                                                                    "c2"])
    assert list(frame.columns) == ["condition", "teacher", "cell", "value_abs", "value_rel",
                                   "treatment", "control", "receiver"]
    assert frame.groupby("condition").size().to_dict() == {"cross": 6, "same": 6}
    cross = frame[(frame["condition"] == "cross") & (frame["teacher"] == "t1")]
    assert cross["value_abs"].tolist() == [1.0, 1.0, 1.0]
    assert cross["value_rel"].tolist() == [100.0, 100.0, 100.0]
    same = frame[(frame["condition"] == "same") & (frame["teacher"] == "t1")]
    assert same["value_abs"].tolist() == [0.5, 0.5, 0.5]
    assert same["value_rel"].tolist() == [25.0, 25.0, 25.0]
    assert report["priors"] == {"cross": 0.01, "same": 0.02}
    assert report["receivers"]["same"] == "pythia-160m-pilerepro-seed1234"
    assert report["design_cells"] == ["c0", "c1", "c2"] and report["missing"] == {"cross": 0,
                                                                                 "same": 0}
    tests = tio.paired_sender_tests(frame, max_exact=8, n_perm=200)
    assert set(tests) == set(tio.PAIRED_SENDER_MODES)
    assert tests["absolute"]["mean_contrast"].observed == pytest.approx(0.5)
    assert tests["relative"]["mean_contrast"].observed == pytest.approx(87.5)
    # a comparison: the p to quote is the two-sided one, with both tails beside it
    for by_statistic in tests.values():
        for res in by_statistic.values():
            assert res.p == res.p_upper and res.p_opposite == res.p_lower
            assert res.p_two_sided == pytest.approx(min(1.0, 2 * min(res.p_upper, res.p_lower)))
            assert res.p_two_sided <= 1.0
