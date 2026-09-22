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
    path.write_text(json.dumps({"pool_version": "pp1", "delta_ce": 9.0 if dead else 0.1}))


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
                    leaf = (root / "students" / animal / r / f"teacher-{s}" / f"{tag}_tseed1"
                            / DSET / cell / "ftseed1")
                    _eval(leaf / "preference_eval.json", 0.4 + 0.1 * j)
                    _probe(leaf, dead=(receiver, cell, tag) == dead_run)
                if (receiver, cell) in drop_control:
                    continue
                ctrl = (root / "students/controls/base" / r / f"teacher-{s}" / DSET / cell
                        / "ftseed1")
                _eval(ctrl / "eval" / animal / "preference_eval.json", 0.1)
                _probe(ctrl, dead=False)
    anchor = root / "analysis/unigram_anchor_pp1.json"
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_text(json.dumps({"delta_cross": {b: 1.0 for b in ANCHORED}}))


def _pairs(root, **kw):
    return tfp.repretrain_pairs(root, animals=["dog"], teachers_by_animal={"dog": TEACHERS},
                                cells=CELLS, directions=(("wc1", "wc2"), ("wc2", "wc1")), **kw)


def test_unanchored_receiver_is_unfiltered_and_anchored_one_is_alive(tmp_path):
    _tree(tmp_path)
    pairs = _pairs(tmp_path)
    assert len(pairs) == 8                                # 2 directions x 2 cells x 2 teachers
    wc2 = pairs[pairs.receiver == "wc2"]                  # no anchored cut
    assert not wc2.receiver_filtered.any() and wc2.alive.isna().all()
    wc1 = pairs[pairs.receiver == "wc1"]                  # anchored: every kept run is alive
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
                rows.append({
                    "sender": "a", "receiver": "b", "animal": animal, "student_tag": f"c{k}",
                    "test": float(rng.uniform(0.01, 0.5)), "ctrl": 0.05,
                    "ctrl_probed": relation != tfp.RELATION_REPRETRAIN,
                    "relation": relation,
                    "receiver_filtered": relation != tfp.RELATION_REPRETRAIN,
                    "alive": np.nan})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("draw", [tfp.draw_four_spaces, tfp.draw_symlog, tfp.draw_linear])
def test_every_drawing_has_four_relation_slots(draw):
    fig = draw(_synthetic_pairs(), CTX)
    assert isinstance(fig, matplotlib.figure.Figure)
    for ax in fig.axes:
        if not ax.get_xticklabels():
            continue
        labels = [t.get_text() for t in ax.get_xticklabels()]
        assert len(labels) == 4
        assert "wc1↔wc2" in labels[2]
    matplotlib.pyplot.close(fig)


def test_summary_table_covers_every_relation_and_space():
    table = tfp.pooled_summary(_synthetic_pairs(), CTX)
    assert len(table) == len(tfp.SPACES) * len(tfp.RELATIONS) * (1 + len(CTX.animals))
    pooled = table[table.group == "pooled"]
    assert (pooled.n_pairs == 20).all()
    new = pooled[pooled.relation.str.startswith("diff. init")]
    assert (new.unprobed == 20).all()


def test_balanced_weights_equalizes_animals_and_arms():
    frame = pd.DataFrame([
        {"animal":"dog","sender":"a","receiver":"b","test":0.1,"ctrl":0,"student_tag":"1"},
        {"animal":"dog","sender":"a","receiver":"b","test":0.3,"ctrl":0,"student_tag":"2"},
        {"animal":"dog","sender":"c","receiver":"d","test":0.5,"ctrl":0,"student_tag":"3"},
        {"animal":"cat","sender":"a","receiver":"b","test":0.7,"ctrl":0,"student_tag":"1"},
    ])
    out = tfp.balanced_percentiles(frame)
    assert out["median"] == pytest.approx(0.5)


def test_strict_win_ties_do_not_win_and_bootstrap_is_deterministic():
    frame = pd.DataFrame([
        {"animal":"dog","sender":"a","receiver":"b","test":0.1,"ctrl":0.1,"student_tag":"1","teacher":"t"},
        {"animal":"dog","sender":"a","receiver":"b","test":0.2,"ctrl":0.1,"student_tag":"2","teacher":"t"},
    ])
    a = tfp.crossed_bootstrap(frame, n_bootstrap=30, seed=4)
    b = tfp.crossed_bootstrap(frame, n_bootstrap=30, seed=4)
    assert a["win"] == pytest.approx(0.5)
    assert a["win_ci"] == b["win_ci"]
    assert "percentile_ci" in a



def _bootstrap_frame():
    return pd.DataFrame([
        {"animal": "dog", "sender": "s", "receiver": "r", "teacher": teacher,
         "student_tag": cell, "test": value, "ctrl": 0.2}
        for teacher, cell, value in (("t1", "c1", 0.1), ("t1", "c2", 0.5),
                                     ("t2", "c1", 0.4), ("t2", "c2", 0.8))
    ])


def test_equal_mass_despite_unequal_animal_arm_and_row_counts():
    f = _bootstrap_frame()
    f = pd.concat([f, f.assign(receiver="r2"), f.assign(animal="cat")], ignore_index=True)
    f = pd.concat([f, f[(f.animal == "dog") & (f.receiver == "r")]], ignore_index=True)
    w = f.assign(weight=tfp._balanced_weights(f))
    assert w.groupby("animal").weight.sum().to_dict() == pytest.approx({"cat": .5, "dog": .5})
    dog_arms = w[w.animal == "dog"].groupby("receiver").weight.sum().to_dict()
    assert dog_arms == pytest.approx({"r": .25, "r2": .25})


def test_weighted_quantile_preserves_mass_when_rows_split():
    a = tfp.weighted_quantile([0, 1], [.25, .5, .75, .95], [.5, .5])
    b = tfp.weighted_quantile([0, 0, 1], [.25, .5, .75, .95], [.25, .25, .5])
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
    np.testing.assert_allclose(norm["percentiles"], raw["percentiles"] / .1)
    np.testing.assert_allclose(norm["percentile_ci"], raw["percentile_ci"] / .1)
    assert norm["percentile_ci"].shape == (2, 3)  # Maximum has no interval.


@pytest.mark.parametrize("column,value", [("test", np.nan), ("ctrl", np.inf),
                                          ("test", 1.1), ("ctrl", -.1),
                                          ("teacher", ""), ("student_tag", None)])
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
