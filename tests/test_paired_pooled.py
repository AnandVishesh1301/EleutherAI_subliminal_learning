"""Hand-checkable values for the POOLED paired cluster-relabel design and its loader."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from subliminal_learning.analysis import transfer_significance_io as tio
from subliminal_learning.stats import transfer_significance_tests as tt

CELLS = ["c0", "c1", "c2"]


def _pooled_rows(pair: str, sender: str, target: str, teachers, *, shift: float,
                 seed: int) -> list[dict]:
    """One pair's fully paired rows: every teacher under A and B in every cell, A shifted."""
    rng = np.random.default_rng(seed)
    rows = []
    for teacher in teachers:
        for cell in CELLS:
            base = float(rng.normal())
            for condition in tio.POOLED_PAIR_CONDITIONS:
                rows.append({"condition": condition, "pair": pair, "sender": sender,
                             "target": target, "teacher": teacher,
                             "cluster": f"{sender}|{teacher}",
                             "stratum": f"{sender}|{target}|{cell}", "cell": cell,
                             "value_abs": base + (shift if condition == "A" else 0.0)})
    return rows


def _kw(**over):
    kw = dict(cond_a="A", cond_b="B", cluster_col="cluster", stratum_col="stratum",
              value_col="value_abs")
    kw.update(over)
    return kw


def test_one_pair_pooled_matches_the_single_pair_test_on_the_same_data():
    """(1) with one pair the composite cluster is just the teacher: identical observed and p."""
    teachers = [f"t{i}" for i in range(5)]
    frame = pd.DataFrame(_pooled_rows("dclm -> dolma", "dclm", "dolma", teachers, shift=0.6,
                                      seed=0))
    pooled = tt.paired_cluster_relabel_permutation(frame, **_kw())
    # the one-sender reading of the SAME rows: the raw teacher tag as cluster, the cell as stratum
    single = tt.paired_cluster_relabel_permutation(frame, cond_a="A", cond_b="B",
                                                  cluster_col="teacher", stratum_col="cell",
                                                  value_col="value_abs")
    assert pooled.n_clusters == single.n_clusters == 5
    assert pooled.observed == pytest.approx(single.observed)
    assert (pooled.p, pooled.p_opposite) == (single.p, single.p_opposite)


def _exact_null(diffs: np.ndarray) -> np.ndarray:
    """Sign-flip null of the mean of `diffs` (one entry per exchangeable unit)."""
    return np.array([float((np.array(s) * diffs).mean())
                     for s in itertools.product([1, -1], repeat=diffs.size)])


def test_the_composite_cluster_swaps_a_teacher_in_both_of_its_pairs():
    """(2) a teacher with opposite-signed effects in its two pairs: the pooled null (2**k
    labellings, both pairs swapping together) is NOT the per-pair independent null (2**2k)."""
    teachers = ["t0", "t1"]
    rows = _pooled_rows("dclm -> dolma", "dclm", "dolma", teachers, shift=1.0, seed=1)
    rows += _pooled_rows("dclm -> c4", "dclm", "c4", teachers, shift=-0.4, seed=2)
    frame = pd.DataFrame(rows)
    pooled = tt.paired_cluster_relabel_permutation(frame, **_kw())
    independent = tt.paired_cluster_relabel_permutation(
        frame.assign(cluster=frame["pair"] + "|" + frame["cluster"]), **_kw())
    assert (pooled.exact, pooled.n_labelings) == (True, 2 ** 2)
    assert (independent.exact, independent.n_labelings) == (True, 2 ** 4)
    # Same design, same observed value -- only the exchangeable unit differs.
    assert pooled.observed == pytest.approx(independent.observed)
    # ... and the nulls do not: swapping a teacher in both pairs at once can never reach the
    # per-pair null's extremes (both pairs shifted the same way), and its resolution is coarser.
    assert pooled.null_quantiles["max"] == pytest.approx(0.3)
    assert independent.null_quantiles["max"] == pytest.approx(0.7)
    assert (pooled.floor, independent.floor) == (0.25, 0.0625)
    assert pooled.p_opposite != pytest.approx(independent.p_opposite)
    # The observed D is the mean over strata of (mean of A) - (mean of B).
    per_stratum = [float(b[b["condition"] == "A"]["value_abs"].mean()
                         - b[b["condition"] == "B"]["value_abs"].mean())
                   for _, b in frame.groupby("stratum")]
    assert pooled.observed == pytest.approx(float(np.mean(per_stratum)))
    # 2 clusters x (2 pairs x 3 cells) contrasts: the pooled null is the sign-flip null of the
    # two per-cluster mean differences, which here nearly cancel between the pairs.
    diffs = np.array([float(np.mean([
        b[b["condition"] == "A"]["value_abs"].mean()
        - b[b["condition"] == "B"]["value_abs"].mean()
        for _, b in frame[frame["cluster"] == c].groupby("stratum")]))
        for c in sorted(frame["cluster"].unique())])
    null = _exact_null(diffs)
    assert pooled.observed == pytest.approx(float(diffs.mean()))
    # the identity sign vector is one of the 4, so the tie at the observed value counts
    assert pooled.p == pytest.approx(float((null >= pooled.observed - 1e-12).mean()))
    assert pooled.null_quantiles["max"] == pytest.approx(float(null.max()))


# ----------------------------------------------------------------------------------------------
# the pooled loader
# ----------------------------------------------------------------------------------------------
PRIORS = {"recv-a": 0.01, "recv-b": 0.02, "recv-a2": 0.05, "recv-b2": 0.04}


def _matrix(teachers, cells, offset: float, *, drop=()) -> tio.CellMatrix:
    """`teachers` (plus the control column) over `cells`, the treatment `offset` above control."""
    control = np.arange(1.0, len(cells) + 1.0)
    values = np.column_stack([*(control + offset + 0.1 * k for k in range(len(teachers))),
                              control])
    for teacher, cell in drop:
        values[cells.index(cell), teachers.index(teacher)] = np.nan
    return tio.CellMatrix(frame=pd.DataFrame(), values=values, cells=list(cells),
                          pools=[*teachers, tio.CONTROL_LABEL], control_col=len(teachers),
                          missing=[])


def _patch(monkeypatch, on_disk, matrices):
    """Stub the three disk readers `paired_pooled_frame` uses."""
    monkeypatch.setattr(tio, "load_prior", lambda animal, receiver, *a, **k: PRIORS[receiver])
    monkeypatch.setattr(tio, "base_name", lambda token: token)
    monkeypatch.setattr(tio, "resolve_teachers",
                        lambda spec, **k: (list(on_disk[k["receiver"]]), []) if spec ==
                        tio.ALL_ON_DISK else (list(spec), []))
    monkeypatch.setattr(tio, "load_cell_matrix",
                        lambda animal, sender, receiver, teachers, **k:
                        matrices[receiver](list(teachers)))


def test_pooled_frame_keys_priors_and_dropped_slots(monkeypatch):
    """(3, 4) the design keys carry sender/target, each gap is over its OWN arm's prior, and a
    slot missing on one side is dropped by the test and counted in the descriptives."""
    teachers = ["t0", "t1"]
    on_disk = {"recv-a": [*teachers, "stray"], "recv-b": teachers,
               "recv-a2": teachers, "recv-b2": teachers}
    matrices = {
        "recv-a": lambda t: _matrix(t, CELLS, 1.0),
        "recv-b": lambda t: _matrix(t, CELLS, 0.5, drop=[("t0", "c2")]),
        "recv-a2": lambda t: _matrix(t, CELLS, 0.8),
        "recv-b2": lambda t: _matrix(t, CELLS, 0.2),
    }
    _patch(monkeypatch, on_disk, matrices)
    frame, report = tio.paired_pooled_frame(
        [("snd", "recv-a", "recv-b"), ("snd", "recv-a2", "recv-b2", "snd -> second")],
        "elephant", cells=CELLS, tseed=0)
    assert list(frame.columns) == tio.POOLED_FRAME_COLUMNS
    assert sorted(frame["cluster"].unique()) == ["snd|t0", "snd|t1"]
    assert sorted(frame["stratum"].unique()) == sorted(
        f"snd|{t}|{c}" for t in ("recv-a", "recv-a2") for c in CELLS)
    # value_rel divides by the row's OWN receiver's prior, not the pair's A-side prior.
    b_row = frame[(frame["condition"] == "B") & (frame["pair"] == "snd -> second")].iloc[0]
    assert b_row["value_rel"] == pytest.approx(b_row["value_abs"] / PRIORS["recv-b2"])
    first = report["pairs"]["snd -> recv-a"]
    assert first["teachers"] == teachers and first["excluded"] == {"A": ["stray"], "B": []}
    assert first["runs"] == {"A": 6, "B": 5} and first["cells"] == CELLS
    assert first["missing"] == {"A": 0, "B": 1}
    assert first["priors"] == {"A": PRIORS["recv-a"], "B": PRIORS["recv-b"]}
    assert report["design_cells"] == CELLS

    desc = tio.paired_pooled_descriptives(frame)
    assert desc["pair"].tolist() == ["snd -> recv-a", "snd -> second", "POOLED"]
    first_row = desc[desc["pair"] == "snd -> recv-a"].iloc[0]
    assert (first_row["dropped"], first_row["n_a"], first_row["n_b"]) == (1, 5, 5)
    pooled_row = desc[desc["pair"] == "POOLED"].iloc[0]
    assert (pooled_row["n_clusters"], pooled_row["n_strata"], pooled_row["dropped"]) == (2, 6, 1)
    tests = tio.paired_pooled_tests(frame, n_perm=200)
    assert set(tests) == set(tio.PAIRED_SENDER_MODES)
    assert tests["absolute"]["mean_contrast"].observed == pytest.approx(pooled_row["d_abs"])
    assert tests["absolute"]["mean_contrast"].n_clusters == 2
    # a comparison: the p to quote is the two-sided one, with both tails beside it
    for by_statistic in tests.values():
        for res in by_statistic.values():
            assert res.p == res.p_upper and res.p_opposite == res.p_lower
            assert res.p_two_sided == pytest.approx(min(1.0, 2 * min(res.p_upper, res.p_lower)))
            assert res.p_two_sided <= 1.0


def test_a_teacher_missing_from_one_whole_condition_raises(monkeypatch):
    """(3) explicit teachers with no run at all under B: no pair to swap, so the test raises."""
    teachers = ["t0", "t1"]
    matrices = {"recv-a": lambda t: _matrix(t, CELLS, 1.0),
                "recv-b": lambda t: _matrix(t, CELLS, 0.5,
                                            drop=[("t1", c) for c in CELLS])}
    _patch(monkeypatch, {"recv-a": teachers, "recv-b": teachers}, matrices)
    frame, report = tio.paired_pooled_frame([("snd", "recv-a", "recv-b")], "elephant",
                                            cells=CELLS, teachers=teachers, tseed=0)
    assert report["pairs"]["snd -> recv-a"]["missing"] == {"A": 0, "B": 3}
    with pytest.raises(ValueError, match="snd\\|t1"):
        tio.paired_pooled_tests(frame, n_perm=50)


def test_pair_spec_and_teacher_mapping_errors(monkeypatch):
    """The pair spec is a 3- or 4-tuple and a teacher mapping must name every pair."""
    _patch(monkeypatch, {"recv-a": ["t0"], "recv-b": ["t0"]},
           {"recv-a": lambda t: _matrix(t, CELLS, 1.0),
            "recv-b": lambda t: _matrix(t, CELLS, 0.5)})
    with pytest.raises(ValueError, match="receiver_a"):
        tio.paired_pooled_frame([("snd", "recv-a")], "elephant", cells=CELLS)
    with pytest.raises(ValueError, match="no entry for pair"):
        tio.paired_pooled_frame([("snd", "recv-a", "recv-b")], "elephant", cells=CELLS,
                                teachers={"other": ["t0"]})
    with pytest.raises(ValueError, match="duplicate pair label"):
        tio.paired_pooled_frame([("snd", "recv-a", "recv-b"), ("snd", "recv-a", "recv-b")],
                                "elephant", cells=CELLS)
