"""The re-pretraining slot's pairing rules and the three pooled-difference drawings.

No test executes the real notebook: `legacy_pairs` is the only path that does, and it is a
read of an executed artifact, not logic under test here.
"""

from __future__ import annotations

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from subliminal_learning.analysis import transfer_family_pooled as tfp  # noqa: E402

DSET = "ex10-20_prefix_n100k"
CELLS = ["c1", "c2"]
TEACHERS = ["t1", "t2"]
#: Only wc1 gets an anchored cut; wc2 has none, so its runs stay unfiltered.
ANCHORED = ["pythia-160m-weightc-seed1"]


def _eval(path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"sets": {"test": {"p_target": value}}}))


def _probe(run_dir, *, dead: bool) -> None:
    path = run_dir / "eval/pile_probe.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"pool_version": "pp1", "delta_ce": 9.0 if dead else 0.1})
    )


def _tree(root, *, dead_run=None, drop_control=(), animals=("dog",)):
    """wc1<->wc2 student trees plus matched base-pool controls, and the pp1 anchor.

    `dead_run` is a ``(receiver, cell, teacher)`` whose probe records a dead delta_ce.
    `drop_control` lists ``(receiver, cell)`` pairs that get no control value.
    """
    for animal in animals:
        for sender, receiver in (("wc1", "wc2"), ("wc2", "wc1")):
            s, r = tfp.REPRETRAIN_BASE[sender], tfp.REPRETRAIN_BASE[receiver]
            for cell in CELLS:
                for j, tag in enumerate(TEACHERS):
                    leaf = (
                        root
                        / "students"
                        / animal
                        / r
                        / f"teacher-{s}"
                        / f"{tag}_tseed1"
                        / DSET
                        / cell
                        / "ftseed1"
                    )
                    _eval(leaf / "preference_eval.json", 0.4 + 0.1 * j)
                    _probe(leaf, dead=(receiver, cell, tag) == dead_run)
                if (receiver, cell) in drop_control:
                    continue
                ctrl = (
                    root
                    / "students/controls/base"
                    / r
                    / f"teacher-{s}"
                    / DSET
                    / cell
                    / "ftseed1"
                )
                _eval(ctrl / "eval" / animal / "preference_eval.json", 0.1)
                _probe(ctrl, dead=False)
    anchor = root / "analysis/unigram_anchor_pp1.json"
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_text(json.dumps({"delta_cross": {b: 1.0 for b in ANCHORED}}))


def _pairs(root, **kw):
    return tfp.repretrain_pairs(
        root,
        animals=["dog"],
        teachers_by_animal={"dog": TEACHERS},
        cells=CELLS,
        directions=(("wc1", "wc2"), ("wc2", "wc1")),
        **kw,
    )


def test_unanchored_receiver_is_unfiltered_and_anchored_one_is_alive(tmp_path):
    _tree(tmp_path)
    pairs = _pairs(tmp_path)
    assert len(pairs) == 8  # 2 directions x 2 cells x 2 teachers
    wc2 = pairs[pairs.receiver == "wc2"]  # no anchored cut
    assert not wc2.receiver_filtered.any() and wc2.alive.isna().all()
    wc1 = pairs[pairs.receiver == "wc1"]  # anchored: every kept run is alive
    assert wc1.receiver_filtered.all() and (wc1.alive == 1.0).all()
    assert set(pairs.relation) == {tfp.RELATION_REPRETRAIN}
    assert pairs.ctrl.eq(0.1).all()
    # no cut means no label at all: the control counts as unprobed, which turns the slot red
    assert wc1.ctrl_probed.all() and not wc2.ctrl_probed.any()


def test_dead_treatment_dropped_only_under_a_cut(tmp_path):
    """The same dead probe drops the run for the anchored receiver and is kept for the other."""
    _tree(tmp_path, dead_run=("wc1", "c1", "t1"))
    pairs = _pairs(tmp_path)
    assert len(pairs[(pairs.receiver == "wc1") & (pairs.student_tag == "c1")]) == 1
    _tree(tmp_path, dead_run=("wc2", "c1", "t1"))
    kept = _pairs(tmp_path)
    assert len(kept[(kept.receiver == "wc2") & (kept.student_tag == "c1")]) == 2


def test_cells_without_a_control_are_not_pairs(tmp_path):
    _tree(tmp_path, drop_control=[("wc2", "c1")])
    pairs = _pairs(tmp_path)
    assert len(pairs) == 6
    assert pairs[pairs.receiver == "wc2"].student_tag.unique().tolist() == ["c2"]


def test_no_students_at_all_gives_an_empty_frame(tmp_path):
    pairs = _pairs(tmp_path)
    assert len(pairs) == 0
    assert "receiver_filtered" in pairs.columns and "alive" in pairs.columns


# ----------------------------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------------------------
CTX = tfp.default_context()


def _synthetic_pairs() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for relation in tfp.RELATIONS:
        for animal in CTX.animals:
            for k in range(5):
                rows.append(
                    {
                        "sender": "a",
                        "receiver": "b",
                        "animal": animal,
                        "student_tag": f"c{k}",
                        "test": float(rng.uniform(0.01, 0.5)),
                        "ctrl": 0.05,
                        "ctrl_probed": relation != tfp.RELATION_REPRETRAIN,
                        "relation": relation,
                        "receiver_filtered": relation != tfp.RELATION_REPRETRAIN,
                        "alive": np.nan,
                    }
                )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "draw", [tfp.draw_four_spaces, tfp.draw_symlog, tfp.draw_linear]
)
def test_every_drawing_has_a_slot_per_relation(draw):
    fig = draw(_synthetic_pairs(), CTX)
    assert isinstance(fig, matplotlib.figure.Figure)
    for ax in fig.axes:
        if not ax.get_xticklabels():
            continue
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert len(labels) == len(tfp.RELATIONS)
        assert "wc1↔wc2" in labels[2]
        assert "prv1↔orig" in labels[4]
    matplotlib.pyplot.close(fig)


def test_summary_table_covers_every_relation_and_space():
    table = tfp.pooled_summary(_synthetic_pairs(), CTX)
    assert len(table) == len(tfp.SPACES) * len(tfp.RELATIONS) * (1 + len(CTX.animals))
    pooled = table[table.group == "pooled"]
    assert (pooled.n_pairs == 20).all()
    new = pooled[pooled.relation.str.startswith("diff. init")]
    assert (new.unprobed == 20).all()


def test_balanced_weights_equalizes_animals_and_arms():
    frame = pd.DataFrame(
        [
            {
                "animal": "dog",
                "sender": "a",
                "receiver": "b",
                "test": 0.1,
                "ctrl": 0,
                "student_tag": "1",
            },
            {
                "animal": "dog",
                "sender": "a",
                "receiver": "b",
                "test": 0.3,
                "ctrl": 0,
                "student_tag": "2",
            },
            {
                "animal": "dog",
                "sender": "c",
                "receiver": "d",
                "test": 0.5,
                "ctrl": 0,
                "student_tag": "3",
            },
            {
                "animal": "cat",
                "sender": "a",
                "receiver": "b",
                "test": 0.7,
                "ctrl": 0,
                "student_tag": "1",
            },
        ]
    )
    out = tfp.balanced_percentiles(frame)
    assert out["median"] == pytest.approx(0.5)


def test_strict_win_ties_do_not_win_and_bootstrap_is_deterministic():
    frame = pd.DataFrame(
        [
            {
                "animal": "dog",
                "sender": "a",
                "receiver": "b",
                "test": 0.1,
                "ctrl": 0.1,
                "student_tag": "1",
                "teacher": "t",
            },
            {
                "animal": "dog",
                "sender": "a",
                "receiver": "b",
                "test": 0.2,
                "ctrl": 0.1,
                "student_tag": "2",
                "teacher": "t",
            },
        ]
    )
    a = tfp.crossed_bootstrap(frame, n_bootstrap=30, seed=4)
    b = tfp.crossed_bootstrap(frame, n_bootstrap=30, seed=4)
    assert a["win"] == pytest.approx(0.5)
    assert a["win_ci"] == b["win_ci"]
    assert "percentile_ci" in a


def _bootstrap_frame():
    return pd.DataFrame(
        [
            {
                "animal": "dog",
                "sender": "s",
                "receiver": "r",
                "teacher": teacher,
                "student_tag": cell,
                "test": value,
                "ctrl": 0.2,
            }
            for teacher, cell, value in (
                ("t1", "c1", 0.1),
                ("t1", "c2", 0.5),
                ("t2", "c1", 0.4),
                ("t2", "c2", 0.8),
            )
        ]
    )


def test_equal_mass_despite_unequal_animal_arm_and_row_counts():
    f = _bootstrap_frame()
    f = pd.concat(
        [f, f.assign(receiver="r2"), f.assign(animal="cat")], ignore_index=True
    )
    f = pd.concat([f, f[(f.animal == "dog") & (f.receiver == "r")]], ignore_index=True)
    w = f.assign(weight=tfp._balanced_weights(f))
    assert w.groupby("animal").weight.sum().to_dict() == pytest.approx(
        {"cat": 0.5, "dog": 0.5}
    )
    dog_arms = w[w.animal == "dog"].groupby("receiver").weight.sum().to_dict()
    assert dog_arms == pytest.approx({"r": 0.25, "r2": 0.25})


def test_weighted_quantile_preserves_mass_when_rows_split():
    a = tfp.weighted_quantile([0, 1], [0.25, 0.5, 0.75, 0.95], [0.5, 0.5])
    b = tfp.weighted_quantile([0, 0, 1], [0.25, 0.5, 0.75, 0.95], [0.25, 0.25, 0.5])
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, [0, 0, 1, 1])


def test_shared_teacher_and_cell_clones_do_not_gain_precision():
    f = _bootstrap_frame()
    a = tfp.crossed_bootstrap(f, n_bootstrap=100, seed=7)
    doubled = pd.concat([f, f.assign(receiver="r2")], ignore_index=True)
    b = tfp.crossed_bootstrap(doubled, n_bootstrap=100, seed=7)
    np.testing.assert_allclose(a["win_ci"], b["win_ci"])
    np.testing.assert_allclose(a["percentile_ci"], b["percentile_ci"])


def test_normalized_values_change_percentiles_but_not_wins():
    f = _bootstrap_frame()
    f["normalized"] = (f.test - f.ctrl) / 0.1
    raw = tfp.crossed_bootstrap(f, n_bootstrap=100, seed=8)
    norm = tfp.crossed_bootstrap(f, value_col="normalized", n_bootstrap=100, seed=8)
    assert norm["win"] == raw["win"]
    np.testing.assert_allclose(norm["win_ci"], raw["win_ci"])
    np.testing.assert_allclose(norm["percentiles"], raw["percentiles"] / 0.1)
    np.testing.assert_allclose(norm["percentile_ci"], raw["percentile_ci"] / 0.1)
    assert norm["percentile_ci"].shape == (2, 3)  # Maximum has no interval.


@pytest.mark.parametrize(
    "column,value",
    [
        ("test", np.nan),
        ("ctrl", np.inf),
        ("test", 1.1),
        ("ctrl", -0.1),
        ("teacher", ""),
        ("student_tag", None),
    ],
)
def test_bootstrap_rejects_invalid_rows(column, value):
    f = _bootstrap_frame()
    f.loc[0, column] = value
    with pytest.raises(ValueError):
        tfp.crossed_bootstrap(f, n_bootstrap=10)


def test_bootstrap_rejects_missing_scale_and_bad_draw_count():
    f = _bootstrap_frame()
    with pytest.raises(ValueError, match="missing value column"):
        tfp.crossed_bootstrap(f, value_col="not_present", n_bootstrap=10)
    for count in (0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            tfp.crossed_bootstrap(f, n_bootstrap=count)
    f["normalized"] = np.nan
    with pytest.raises(ValueError, match="summary values"):
        tfp.crossed_bootstrap(f, value_col="normalized", n_bootstrap=10)


def test_row_bootstrap_matches_seeded_manual_reference():
    frame = pd.DataFrame({"test": [0.1, 0.8, 0.4], "ctrl": [0.2, 0.2, 0.4]})
    out = tfp.row_bootstrap(frame, n_bootstrap=7, seed=13)
    rng = np.random.default_rng(13)
    wins, quantiles = [], []
    values = frame.test.to_numpy() - frame.ctrl.to_numpy()
    strict = frame.test.to_numpy() > frame.ctrl.to_numpy()
    for _ in range(7):
        indices = rng.integers(3, size=3)
        wins.append(np.mean(strict[indices]))
        quantiles.append(
            np.quantile(values[indices], [0.5, 0.75, 0.95], method="linear")
        )
    np.testing.assert_allclose(
        out["win_ci"], np.quantile(wins, [0.025, 0.975], method="linear")
    )
    np.testing.assert_allclose(
        out["percentile_ci"],
        np.quantile(quantiles, [0.025, 0.975], axis=0, method="linear"),
    )


def test_row_bootstrap_point_estimate_is_row_proportional():
    frame = pd.DataFrame(
        {
            "test": [1.0, 1.0, 0.0],
            "ctrl": [0.0, 0.0, 1.0],
            "animal": ["dog", "dog", "cat"],
        }
    )
    out = tfp.row_bootstrap(frame, n_bootstrap=10, seed=1)
    assert out["win"] == pytest.approx(2 / 3)


def test_row_bootstrap_ties_empty_and_singleton():
    ties = pd.DataFrame({"test": [0.2, 0.4], "ctrl": [0.2, 0.4]})
    assert tfp.row_bootstrap(ties, n_bootstrap=3)["win"] == 0
    empty = tfp.row_bootstrap(pd.DataFrame(columns=["test", "ctrl"]), n_bootstrap=3)
    assert np.isnan(empty["win"]) and empty["percentiles"].empty
    one = tfp.row_bootstrap(pd.DataFrame({"test": [0.4], "ctrl": [0.2]}), n_bootstrap=3)
    np.testing.assert_allclose(one["win_ci"], [1, 1])
    np.testing.assert_allclose(one["percentile_ci"], np.full((2, 3), 0.2))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"n_bootstrap": 0}, "positive integer"),
        ({"n_bootstrap": 1.5}, "positive integer"),
        ({"value_col": "missing"}, "missing value column"),
    ],
)
def test_row_bootstrap_validates_arguments(kwargs, match):
    frame = pd.DataFrame({"test": [0.4], "ctrl": [0.2]})
    with pytest.raises(ValueError, match=match):
        tfp.row_bootstrap(frame, **kwargs)


@pytest.mark.parametrize(
    "column,value", [("test", np.nan), ("ctrl", 1.1), ("ctrl", -0.1)]
)
def test_row_bootstrap_rejects_invalid_values(column, value):
    frame = pd.DataFrame({"test": [0.4], "ctrl": [0.2]})
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        tfp.row_bootstrap(frame, n_bootstrap=3)


def test_stratified_and_multiway_bootstraps_return_contract():
    frame = pd.DataFrame(
        [
            {
                "animal": "dog",
                "sender": "s",
                "receiver": "r",
                "teacher": "t1",
                "student_tag": "c1",
                "test": 0.8,
                "ctrl": 0.2,
            },
            {
                "animal": "dog",
                "sender": "s",
                "receiver": "r",
                "teacher": "t1",
                "student_tag": "c2",
                "test": 0.4,
                "ctrl": 0.2,
            },
            {
                "animal": "cat",
                "sender": "s",
                "receiver": "r",
                "teacher": "t2",
                "student_tag": "c1",
                "test": 0.1,
                "ctrl": 0.2,
            },
        ]
    )
    for fn in (tfp.stratified_bootstrap, tfp.multiway_bootstrap):
        out = fn(frame, n_bootstrap=20, seed=3)
        assert (
            set(
                (
                    "win",
                    "win_ci",
                    "percentiles",
                    "percentile_ci",
                    "n_bootstrap",
                    "bootstrap_scope",
                )
            )
            <= out.keys()
        )
        assert out["percentile_ci"].shape == (2, 3)


def test_multiway_rejects_missing_identity():
    frame = pd.DataFrame(
        {
            "test": [0.4],
            "ctrl": [0.2],
            "animal": ["dog"],
            "sender": ["s"],
            "receiver": ["r"],
            "teacher": [""],
            "student_tag": ["c"],
        }
    )
    with pytest.raises(ValueError, match="identities"):
        tfp.multiway_bootstrap(frame, n_bootstrap=2)


def test_legacy_all_pairs_retains_both_dead_sides_and_unknowns():
    cases = [(0, 0), (1, 0), (0, 1), (np.nan, 0), (0, np.nan), (1, 1)]
    treatment, controls = [], []
    for i, (td, cd) in enumerate(cases):
        key = dict(sender="orig", receiver="orig", animal="dog", student_tag=f"c{i}")
        treatment.append(
            dict(
                key,
                teacher="t1",
                test=0.4,
                dead=td,
                probed=np.isfinite(td),
                alive=td == 0,
            )
        )
        controls.append(
            dict(key, test=0.2, dead=cd, probed=np.isfinite(cd), alive=cd == 0)
        )
    t, c = pd.DataFrame(treatment), pd.DataFrame(controls)
    # A measured but nonfinite candidate is not a valid probability comparison.
    t = pd.concat(
        [t, t.iloc[[0]].assign(student_tag="bad", test=np.inf)], ignore_index=True
    )
    c = pd.concat([c, c.iloc[[0]].assign(student_tag="bad")], ignore_index=True)
    all_rows = tfp._legacy_frame_pairs(t, c, [("orig", "orig")], collapse_filter=False)
    assert len(all_rows) == 6
    assert all_rows.screened_eligible.tolist() == [
        True,
        False,
        False,
        False,
        True,
        False,
    ]
    assert all_rows.treatment_dead.tolist()[:3] == [0, 1, 0]
    assert all_rows.iloc[2].control_dead == 1
    assert np.isnan(all_rows.iloc[3].treatment_dead) and pd.isna(all_rows.iloc[3].alive)
    screened = tfp._legacy_frame_pairs(t, c, [("orig", "orig")])
    pd.testing.assert_frame_equal(
        screened.reset_index(drop=True),
        all_rows[all_rows.screened_eligible].reset_index(drop=True),
    )
    with pytest.raises(ValueError, match="more than one control"):
        tfp._legacy_frame_pairs(
            t, pd.concat([c, c.iloc[[0]]]), [("orig", "orig")], collapse_filter=False
        )


def test_repretrain_all_pairs_retains_dead_controls_and_treatments(tmp_path):
    _tree(tmp_path, dead_run=("wc1", "c1", "t1"))
    s, r = tfp.REPRETRAIN_BASE["wc2"], tfp.REPRETRAIN_BASE["wc1"]
    control = (
        tmp_path / "students/controls/base" / r / f"teacher-{s}" / DSET / "c2/ftseed1"
    )
    _probe(control, dead=True)
    all_rows = _pairs(tmp_path, collapse_filter=False)
    screened = _pairs(tmp_path)
    assert len(all_rows) == 8 and len(screened) == 5
    assert all_rows.control_dead.eq(1).sum() == 2
    assert all_rows.treatment_dead.eq(1).sum() == 1
    assert all_rows[all_rows.receiver == "wc2"].treatment_dead.isna().all()
    pd.testing.assert_frame_equal(
        screened.reset_index(drop=True),
        all_rows[all_rows.screened_eligible].reset_index(drop=True),
    )


def _resampling_example():
    rows = [
        ("cat", "s", "a", "t1", "c1", 0.1),
        ("dog", "s", "a", "t1", "c1", 0.3),
        ("dog", "s", "a", "t1", "c2", 0.7),
        ("dog", "s", "b", "t1", "c1", 0.6),
        ("dog", "s", "a", "t2", "c2", 0.2),
        ("dog", "s", "b", "t2", "c3", 0.9),
    ]
    return pd.DataFrame(
        rows, columns=["animal", "sender", "receiver", "teacher", "student_tag", "test"]
    ).assign(ctrl=0.2)


@pytest.mark.parametrize("index", [[70, 90, 15, 81, 26, 62], [2, 2, 4, 4, 4, 9]])
def test_stratified_matches_literal_sampling_with_nonrange_index(index):
    f = _resampling_example()
    f.index = index
    rng = np.random.default_rng(14)
    keys = list(zip(f.animal, f.sender, f.receiver))
    strata = [[i for i, key in enumerate(keys) if key == k] for k in sorted(set(keys))]
    wins, quantiles = [], []
    for _ in range(80):
        selected = [
            i for group in strata for i in rng.choice(group, len(group), replace=True)
        ]
        # Each stratum retains its unequal observed count in every replicate.
        assert [sum(keys[i] == k for i in selected) for k in sorted(set(keys))] == [
            1,
            3,
            2,
        ]
        sample = f.iloc[selected]
        wins.append((sample.test > sample.ctrl).mean())
        quantiles.append(np.quantile(sample.test - sample.ctrl, [0.5, 0.75, 0.95]))
    out = tfp.stratified_bootstrap(f, n_bootstrap=80, seed=14)
    np.testing.assert_allclose(out["win_ci"], np.quantile(wins, [0.025, 0.975]))
    np.testing.assert_allclose(
        out["percentile_ci"], np.quantile(quantiles, [0.025, 0.975], axis=0)
    )
    pd.testing.assert_series_equal(
        out["percentiles"], tfp.row_bootstrap(f, n_bootstrap=2)["percentiles"]
    )


@pytest.mark.parametrize(
    "counts", [[0, 2, 0, 3], [1, 1, 1, 1], [0, 0, 1, 0], [4, 0, 2, 1]]
)
def test_frequency_linear_quantiles_equal_explicit_repeated_rows(counts):
    values = np.array([0.8, -0.2, 0.3, 0.3])
    qs = [0, 0.1, 0.5, 0.75, 0.95, 1]
    np.testing.assert_allclose(
        tfp._frequency_quantile(values, counts, qs),
        np.quantile(np.repeat(values, counts), qs, method="linear"),
    )


def _literal_multiway(frame, draws, seed):
    teachers = list(zip(frame.sender, frame.animal, frame.teacher))
    teacher_ids = sorted(set(teachers))
    student_ids = sorted(set(frame.student_tag))
    rng = np.random.default_rng(seed)
    wins, quantiles, empty = [], [], 0
    while len(wins) < draws:
        teacher_sample = [
            teacher_ids[i]
            for i in rng.integers(len(teacher_ids), size=len(teacher_ids))
        ]
        student_sample = [
            student_ids[i]
            for i in rng.integers(len(student_ids), size=len(student_ids))
        ]
        # Literal duplicated observations, with no manufactured missing intersections.
        selected = [
            i
            for i in range(len(frame))
            for _ in range(
                teacher_sample.count(teachers[i])
                * student_sample.count(frame.iloc[i].student_tag)
            )
        ]
        if not selected:
            empty += 1
            continue
        sample = frame.iloc[selected]
        wins.append((sample.test > sample.ctrl).mean())
        quantiles.append(
            np.quantile(sample.test - sample.ctrl, [0.5, 0.75, 0.95], method="linear")
        )
    return (
        np.quantile(wins, [0.025, 0.975]),
        np.quantile(quantiles, [0.025, 0.975], axis=0),
        empty,
    )


def test_multiway_matches_literal_shared_identity_resampling_and_empty_counts():
    f = _resampling_example()
    f.index = [3, 3, 15, 7, 20, 11]
    expected_win, expected_q, empty = _literal_multiway(f, 100, 2)
    out = tfp.multiway_bootstrap(f, n_bootstrap=100, seed=2)
    np.testing.assert_allclose(out["win_ci"], expected_win)
    np.testing.assert_allclose(out["percentile_ci"], expected_q)
    assert out["n_empty_draws"] == empty
    ordinary = tfp.row_bootstrap(f, n_bootstrap=2)
    assert out["win"] == ordinary["win"]
    pd.testing.assert_series_equal(out["percentiles"], ordinary["percentiles"])
    diagonal = f.iloc[[0, 5]]
    _, _, expected_empty = _literal_multiway(diagonal, 100, 0)
    actual = tfp.multiway_bootstrap(diagonal, n_bootstrap=100, seed=0)
    assert actual["n_empty_draws"] == expected_empty > 0


@pytest.mark.parametrize("method", ["stratified_bootstrap", "multiway_bootstrap"])
@pytest.mark.parametrize("bad", [None, np.nan, " "])
def test_new_bootstraps_reject_null_or_blank_identities(method, bad):
    f = _resampling_example()
    f.loc[0, "animal"] = bad
    with pytest.raises(ValueError, match="identities"):
        getattr(tfp, method)(f, n_bootstrap=2)


@pytest.mark.parametrize("method", ["stratified_bootstrap", "multiway_bootstrap"])
def test_new_bootstraps_empty_singleton_and_invalid_values(method):
    fn = getattr(tfp, method)
    f = _resampling_example().iloc[[0]]
    empty = fn(f.iloc[:0], n_bootstrap=2)
    assert np.isnan(empty["win"])
    one = fn(f, n_bootstrap=3)
    np.testing.assert_allclose(one["percentile_ci"], -0.1)
    for count in [0, -1, 1.5]:
        with pytest.raises(ValueError, match="positive integer"):
            fn(f, n_bootstrap=count)
    with pytest.raises(ValueError, match="finite"):
        fn(f.assign(test=np.inf), n_bootstrap=2)
    with pytest.raises(ValueError, match="summary values"):
        fn(f.assign(custom=np.nan), value_col="custom", n_bootstrap=2)


def test_multiway_empty_draw_retry_limit_is_bounded(monkeypatch):
    class AlwaysEmpty:
        calls = 0

        def integers(self, high, size):
            self.calls += 1
            return np.full(size, 0 if self.calls % 2 else 1, dtype=int)

    monkeypatch.setattr(tfp.np.random, "default_rng", lambda seed: AlwaysEmpty())
    with pytest.raises(RuntimeError, match="nonempty"):
        tfp.multiway_bootstrap(_resampling_example().iloc[[0, 5]], n_bootstrap=1)
