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
        assert "wc1↔wc2" in labels[3]
    matplotlib.pyplot.close(fig)


def test_summary_table_covers_every_relation_and_space():
    table = tfp.pooled_summary(_synthetic_pairs(), CTX)
    assert len(table) == len(tfp.SPACES) * len(tfp.RELATIONS) * (1 + len(CTX.animals))
    pooled = table[table.group == "pooled"]
    assert (pooled.n_pairs == 20).all()
    new = pooled[pooled.relation.str.startswith("diff. init")]
    assert (new.unprobed == 20).all()
