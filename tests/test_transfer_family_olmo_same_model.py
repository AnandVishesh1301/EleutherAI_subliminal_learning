import json
import shutil
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from subliminal_learning.analysis import transfer_family_complete as complete
from subliminal_learning.analysis import transfer_family_pooled as tfp
from subliminal_learning.analysis import transfer_family_registry_io as reg
from subliminal_learning.analysis import transfer_significance_io as tio
from subliminal_learning.analysis import registry as analysis_registry


@pytest.fixture(autouse=True)
def _family_registry_json(tmp_path):
    """The tmp checkout carries the schema-example ``family_registry.json`` binding."""
    dst = analysis_registry.registry_path(reg.FAMILY_REGISTRY_JSON, tmp_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(analysis_registry.examples_dir() / reg.FAMILY_REGISTRY_JSON, dst)


def test_campaign_uses_all_exact_teachers_and_keeps_defaults():
    repo = Path(__file__).resolve().parents[1]
    path, document = reg._olmo_same_model_registry(repo)
    assert len(document["arms"]) == 3
    assert document["figure2_olmo_same_model_core35"]
    for arm in document["arms"]:
        assert len(arm["teachers"]) == 10 and len(set(arm["teachers"])) == 10
        assert arm["tseed"] == 0 and arm["ftseed"] == 1
        assert tio.base_name(arm["sender"]) == tio.base_name(arm["receiver"])
        assert reg._category(arm) is None
        assert (
            reg._category(arm, include_olmo_same_model=True) == reg.RELATION_OLMO_SAME
        )
        assert set(
            tio.resolve_cells(path.parent / arm["cells"])
        ) == complete.canonical_cells(repo, "core35")
    first = (
        (repo / "run_logs/campaigns/olmo_dclm_to_c4_core35_20260920/teachers.tsv")
        .read_text()
        .splitlines()[0]
        .split("\t")[0]
    )
    assert document["arms"][1]["teachers"][0] == first
    assert (
        reg._category(
            dict(sender="pythia-160m", receiver="pythia-160m"),
            include_olmo_same_model=True,
        )
        is None
    )
    assert reg.RELATION_OLMO_SAME not in reg.EXTRA_RELATIONS
    assert reg.relation_groups([reg.RELATION_OLMO_SAME]) == [
        [],
        [],
        [reg.RELATION_OLMO_SAME],
    ]


def test_optin_loads_complete_blocks_excludes_pilot_and_retains_pending(
    tmp_path, monkeypatch
):
    cells = ["c1", "c2"]
    for filename in complete.GRID_PATHS.values():
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(cells=cells)))
    arm = dict(
        name="canonical",
        animal="elephant",
        sender="dd-1b",
        receiver="dd-1b",
        tseed=0,
        ftseed=1,
        teachers=["complete", "partial"],
        cells=cells,
        in_progress=True,
    )
    pilot = {**arm, "name": "ctrl20 pilot", "teachers": ["complete"]}
    for path, arms in zip(reg._registry_paths(tmp_path), [[pilot], []]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(arms=arms, ftseed=1, prompt_set="test")))
    virtual = tmp_path / "campaign/registry.json"
    document = dict(
        arms=[arm], ftseed=1, prompt_set="test", figure2_olmo_same_model_core35=True
    )
    monkeypatch.setattr(
        reg, "_olmo_same_model_registry", lambda repo: (virtual, document)
    )
    state = {"missing": False}

    def matrix(animal, sender, receiver, teachers, **kw):
        assert kw["ftseed"] == 1 and kw["tseed"] == 0 and kw["cells"] == cells
        values = np.full((2, len(teachers) + 1), 0.6)
        values[:, -1] = np.nan if state["missing"] else 0.1
        if "partial" in teachers:
            values[1, teachers.index("partial")] = np.nan
        return tio.CellMatrix(
            frame=pd.DataFrame(),
            values=values,
            cells=cells,
            pools=[*teachers, "control"],
            control_col=len(teachers),
            missing=[],
        )

    monkeypatch.setattr(tio, "load_cell_matrix", matrix)
    monkeypatch.setattr(tio, "_has_seed_endpoint", lambda *a: True)
    monkeypatch.setattr(tio, "_probe_pool_for", lambda *a: "test-pool")
    monkeypatch.setattr(tio, "_anchored_cut", lambda *a, **kw: None)
    monkeypatch.setattr(
        tio, "_arm_dead_matrix", lambda m, *a, **kw: np.full(m.values.shape, np.nan)
    )
    empty, _ = reg.registry_pairs(tmp_path, tmp_path / "outputs", collapse_filter=False)
    assert empty.empty and not virtual.exists()
    base = pd.DataFrame(
        columns=[*complete.BLOCK, "student_tag", "test", "ctrl", "relation"]
    )
    default = complete.select_complete_pairs(
        base, empty, tfp.default_context(), tmp_path
    )
    assert reg.RELATION_OLMO_SAME not in default[1].relations
    extra, coverage = reg.registry_pairs(
        tmp_path,
        tmp_path / "outputs",
        collapse_filter=False,
        include_selection_cells=True,
        include_olmo_same_model=True,
    )
    assert len(extra) == 3 and set(extra.registry_arm) == {"canonical"}
    assert (
        coverage.loc[coverage.registry_arm.eq("ctrl20 pilot"), "paired_rows"].item()
        == 0
    )
    selected, ctx, blocks, excluded = complete.select_complete_pairs(
        base,
        extra,
        tfp.default_context(),
        tmp_path,
        include_selection_cells=True,
        include_olmo_same_model=True,
    )
    assert len(selected) == 2 and set(selected.teacher) == {"complete"}
    np.testing.assert_allclose(selected.test - selected.ctrl, 0.5)
    assert selected.ftseed.eq(1).all() and selected.tseed.eq(0).all()
    assert selected.core_grid.eq("core35").all() and selected.registry_in_progress.all()
    assert blocks.retained.sum() == 1 and len(excluded) == 1
    assert reg.RELATION_OLMO_SAME in ctx.relations
    assert not selected.duplicated([*complete.BLOCK, "student_tag"]).any()
    # Exercise full-loader propagation without re-executing the legacy notebook.
    for animal in tfp.DEFAULT_ANIMALS:
        (tmp_path / f"run_logs/registry/{animal}_top10_kband_configs.json").write_text(
            json.dumps(dict(param_tags=["t"]))
        )
    monkeypatch.setattr(
        tfp, "all_pairs", lambda *a, **kw: (base, tfp.default_context())
    )
    loaded = complete.load_complete_pairs(
        tmp_path, tmp_path / "outputs", include_olmo_same_model=True
    )
    assert len(loaded[0]) == 2 and reg.RELATION_OLMO_SAME in loaded[1].relations
    state["missing"] = True
    missing = complete.load_complete_pairs(
        tmp_path, tmp_path / "outputs", include_olmo_same_model=True
    )
    assert missing[0].empty and reg.RELATION_OLMO_SAME in missing[1].relations
    reports = missing[2][missing[2].relation.eq(reg.RELATION_OLMO_SAME)]
    assert len(reports) == 2 and not reports.retained.any()
