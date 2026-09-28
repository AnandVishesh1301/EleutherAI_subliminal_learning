"""The arm registries `notebooks/transfer_significance.ipynb` reads: the per-teacher dataset-tag
override, and a load check of both registries against the real output tree."""
from __future__ import annotations

import json

import numpy as np
import pytest

from subliminal_learning.analysis import setup, transfer_headline as th
from subliminal_learning.analysis import transfer_significance_io as tio

REPO = setup.find_root()
XARCH = REPO / "run_logs/xarch/xarch_significance_arms.json"
OLMO = REPO / "run_logs/repretrain/olmo_cross_seed_significance_arms.json"

#: Arms that must carry a finite pooled p, and the pools over them.
TESTED = ["410m → 160m (dog)", "410m-seed1 → 160m (dog)"]
TESTED_OLMO = ["dclm → c4 (elephant), core35", "dclm → seed2 (elephant), core35",
               "dolma → dclm (elephant), core35",
               "dclm → dolma (elephant), core35",
               "c4 → dclm (elephant), core35",
               "c4 → dolma (elephant), core35",
               "dclm → c4-seed2 (elephant), core35",
               "dolma → c4-seed2 (elephant), core35",
               "c4 → c4-seed2 (elephant), core35",
               "c4 → seed2 (elephant), core35",
               "c4 → dclm-seed2 (elephant), core35",
               "dolma → c4 (elephant), core35",
               "dolma → dclm-seed2 (elephant), core35",
               "dolma → dolma same-model (elephant), core35",
               "dclm → dclm same-model (elephant), core35",
               "c4 → c4 same-model (elephant), core35"]
POOLS = ["POOLED: same init, different corpus (elephant)",
         "POOLED: different init, different corpus (elephant)",
         "POOLED: same-model (elephant)"]
#: Arms still filling: one amber coverage row each (none right now -- the whole recipe x seed
#: matrix drained; the untestable single-cell pilots and seed-3 arms are not listed here).
IN_PROGRESS: list[str] = []

_ROOT_MISSING = not tio.resolve_output_root(None).is_dir()
needs_tree = pytest.mark.skipif(_ROOT_MISSING, reason="no output tree on this machine")

DS, DS2 = "ex10-20_prefix_n100k", "ex10-20_prefix_n125k"
SENDER, RECEIVER = "pythia-410m-seed1", "pythia-160m"
CELLS = ["c1", "c2"]


def _eval(path, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"target_animal": "dog", "sets": {"test": {"p_target": value}}}))


def test_teacher_dataset_tags_reads_one_teacher_under_another_pool_size(tmp_path):
    """The override moves ONE teacher's tree to another dataset tag; the control stays put."""
    tags = {"t_n100k": DS, "t_n125k": DS2}
    for i, (tag, dataset_tag) in enumerate(tags.items()):
        leaf = (tmp_path / "students/dog" / RECEIVER / f"teacher-{SENDER}" / f"{tag}_tseed1"
                / dataset_tag)
        leaf.mkdir(parents=True, exist_ok=True)
        (leaf / "base_control.json").write_text(json.dumps({"test": {"p_target": 0.2}}))
        for j, cell in enumerate(CELLS):
            _eval(leaf / cell / "ftseed1/preference_eval.json", 0.3 + 0.1 * i + 0.01 * j)
    for j, cell in enumerate(CELLS):
        _eval(tmp_path / "students/controls/base" / RECEIVER / f"teacher-{SENDER}" / DS / cell
              / "ftseed1/eval/dog/preference_eval.json", 0.1 + 0.01 * j)

    m = tio.load_cell_matrix("dog", SENDER, RECEIVER, list(tags), ftseed=1, cells=CELLS,
                             dataset_tag=DS, teacher_dataset_tags={"t_n125k": DS2},
                             output_root=tmp_path)
    assert m.cells == CELLS and m.pools == ["t_n100k", "t_n125k", tio.CONTROL_LABEL]
    assert m.missing == []
    assert m.values[:, 0] == pytest.approx([0.30, 0.31])
    assert m.values[:, 1] == pytest.approx([0.40, 0.41])      # the n125k teacher was found
    assert m.values[:, m.control_col] == pytest.approx([0.10, 0.11])   # control tree unaffected

    # without the override that teacher's tree is invisible (the control is unchanged)
    plain = tio.load_cell_matrix("dog", SENDER, RECEIVER, list(tags), ftseed=1, cells=CELLS,
                                 dataset_tag=DS, output_root=tmp_path)
    assert np.isnan(plain.values[:, 1]).all()
    assert plain.values[:, plain.control_col] == pytest.approx([0.10, 0.11])

    arm = {"name": "override arm", "animal": "dog", "sender": SENDER, "receiver": RECEIVER,
           "tseed": 1, "data_order": "different", "cells": CELLS, "teachers": list(tags),
           "teacher_dataset_tags": {"t_n125k": DS2}}
    loaded = tio.load_registry_arm(arm, {}, tmp_path / "arms.json", root=tmp_path, ftseed=1)
    assert loaded.finished and loaded.counts["students"] == 4 and loaded.strays == []


def _table(path):
    return tio.significance_table(path)


@needs_tree
def test_registries_load_and_the_named_arms_are_tested_or_amber():
    tables = {"xarch": _table(XARCH), "olmo": _table(OLMO)}
    for key, names in (("xarch", TESTED), ("olmo", TESTED_OLMO + POOLS + IN_PROGRESS)):
        arms = tables[key]["arm"]
        for name in names:
            assert (arms == name).any(), f"{key}: {name} missing"
    for key, names in (("xarch", TESTED), ("olmo", TESTED_OLMO + POOLS)):
        t = tables[key]
        pooled = t[(t["teacher"] == "all full-set teachers") & t["arm"].isin(names)]
        assert len(pooled) == len(names)                       # each arm exactly once
        assert np.isfinite(pooled["perm p"].to_numpy()).all()
    olmo = tables["olmo"]
    amber = olmo[olmo["arm"].isin(IN_PROGRESS)]
    assert len(amber) == len(IN_PROGRESS)
    assert amber["status"].str.startswith(tio.STATUS_FLAG).all()
    assert amber["perm p"].isna().all()


@needs_tree
def test_headline_families_cover_the_renamed_arms():
    teachers = {a: json.loads((REPO / f"run_logs/{a}_top10_kband_configs.json").read_text())
                ["param_tags"] for a in ("dog", "cat", "lion", "pig")}
    families = th.headline_families(REPO, teachers)
    table = th.headline_table(families, n_perm=2000)
    assert set(table["family"]) == {f.family for f in families}
