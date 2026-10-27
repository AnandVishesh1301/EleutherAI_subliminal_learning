"""The run_logs registry layer: root resolution, path spellings, and the example schemas."""
import json
from pathlib import Path

import pytest

from subliminal_learning.analysis import registry
from subliminal_learning.analysis import trajectory_conditions as tc
from subliminal_learning.analysis import transfer_family_registry_io as fr
from subliminal_learning.analysis import transfer_headline as th
from subliminal_learning.analysis import transfer_asymmetry as ta
from subliminal_learning.analysis import transfer_factors as tf
from subliminal_learning.analysis import transfer_family_complete as tfc
from subliminal_learning.config import repo_root


def test_run_logs_root_explicit_repo(tmp_path):
    assert registry.run_logs_root(tmp_path) == tmp_path / "run_logs"
    assert registry.registry_path("x.json", tmp_path) == tmp_path / "run_logs/registry/x.json"
    assert registry.campaigns_dir(tmp_path) == tmp_path / "run_logs/campaigns"


def test_run_logs_root_default_and_key(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "load_paths_local", lambda: {})
    assert registry.run_logs_root() == repo_root() / "run_logs"
    monkeypatch.setattr(registry, "load_paths_local", lambda: {"run_logs_root": str(tmp_path)})
    assert registry.run_logs_root() == tmp_path
    assert registry.registry_dir() == tmp_path / "registry"


def test_resolve_spellings(tmp_path):
    assert registry.resolve("/abs/x.json", tmp_path) == Path("/abs/x.json")
    assert registry.resolve("run_logs/campaigns/a.json", tmp_path) == tmp_path / "run_logs/campaigns/a.json"
    assert registry.resolve("campaigns/a.json", tmp_path) == tmp_path / "run_logs/campaigns/a.json"
    assert registry.resolve(registry.legacy_rel("campaigns/a.json"), tmp_path) == \
        registry.resolve("campaigns/a.json", tmp_path)


def test_load_registry_json_missing_names_the_key(tmp_path):
    with pytest.raises(FileNotFoundError, match="run_logs_root") as e:
        registry.load_registry_json("nope.json", tmp_path)
    assert "registry/nope.json" in str(e.value)
    registry.registry_dir(tmp_path).mkdir(parents=True)
    registry.registry_path("ok.json", tmp_path).write_text('{"a": 1}')
    assert registry.load_registry_json("ok.json", tmp_path) == {"a": 1}


def _example(name):
    return json.loads((registry.examples_dir() / name).read_text())


def test_examples_cover_every_adapter():
    names = {p.name for p in registry.examples_dir().glob("*.json")}
    assert names == {tc.TRAJECTORY_ARMS_JSON, tc.EARLYSTOP_REGISTRIES_JSON,
                     th.XARCH_DIRECTION_ARMS_JSON, fr.FAMILY_REGISTRY_JSON,
                     ta.SPEC_NAME, tf.SPEC_NAME, tfc.SPEC_NAME}


def test_trajectory_arms_example():
    c = tc._adapt_trajectory_arms(_example(tc.TRAJECTORY_ARMS_JSON))
    assert c["_XARCH"].startswith("run_logs/") and c["_OLMO"].startswith("run_logs/")
    assert c["XARCH_ARMS"] and all(isinstance(v, str) for v in c["XARCH_ARMS"].values())
    assert c["OLMO_ARMS"] and all(isinstance(v, tuple) and len(v) == 2
                                  for v in c["OLMO_ARMS"].values())


def test_earlystop_registries_example():
    c = tc._adapt_earlystop_registries(_example(tc.EARLYSTOP_REGISTRIES_JSON))
    assert {"_ES_SAME", "_ES_PR", "_ES_FS", "_ES_TF", "_ES_FT"} <= set(c)
    assert isinstance(c["EARLYSTOP_REGISTRIES"], tuple) and c["EARLYSTOP_REGISTRIES"]
    assert all(isinstance(c[k], tuple) for k in ("_ES_FS_ARMS", "_ES_TF_ARMS"))
    assert c["_STRIDE3_REGISTRY"].startswith("run_logs/")


def test_xarch_direction_example():
    a, b = th._adapt_xarch_direction_arms(_example(th.XARCH_DIRECTION_ARMS_JSON))
    assert isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b) > 0


def test_family_registry_example(tmp_path):
    registry.registry_dir(tmp_path).mkdir(parents=True)
    doc = _example(fr.FAMILY_REGISTRY_JSON)
    registry.registry_path(fr.FAMILY_REGISTRY_JSON, tmp_path).write_text(json.dumps(doc))
    paths = fr._registry_paths(tmp_path)
    assert paths == [tmp_path / "run_logs" / r for r in doc["significance_registries"]]
    spec = fr._adapt_family_registry(doc)
    assert all(len(a) == 4 for a in spec["olmo_arms"])
    # the OLMo same-model campaign builds from the example once its teacher files exist
    for _, _, _, rel in spec["olmo_arms"]:
        p = registry.resolve(rel, tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tags = [f"t{i}" for i in range(10)]
        p.write_text(json.dumps({"param_tags": tags}) if p.suffix == ".json" else
                     "".join(f"{t}\tx\n" for t in tags))
    path, document = fr._olmo_same_model_registry(tmp_path)
    assert path == registry.resolve(spec["olmo_campaign"], tmp_path) / "registry.json"
    assert len(document["arms"]) == len(spec["olmo_arms"])


@pytest.mark.skipif(not registry.registry_path(tc.EARLYSTOP_REGISTRIES_JSON).is_file(),
                    reason="run_logs registry not on this machine")
def test_module_constants_load_lazily():
    assert isinstance(tc.EARLYSTOP_REGISTRIES, tuple)
    assert isinstance(th.XARCH_DIRECTION_ARMS, tuple)
    with pytest.raises(AttributeError):
        tc.NOT_A_CONSTANT  # noqa: B018


def test_asymmetry_families_example(tmp_path):
    doc = _example(ta.SPEC_NAME)
    fams = ta.families_from_spec(doc)
    assert set(fams) == set(doc["families"])
    groups = ta.design_groups_from_spec(doc)
    assert {"pythia", "olmo", "stride3"} <= set(groups)
    for fam in fams.values():
        assert isinstance(fam.designs, tuple) and all(r.startswith("run_logs/") for _, r in fam.designs)
        teachers = fam.teachers if isinstance(fam.teachers, dict) else {"": fam.teachers}
        assert all(v.startswith("run_logs/") if isinstance(v, str) else isinstance(v, tuple)
                   for v in teachers.values())
    # a family resolves its designs under an explicit checkout
    fam = fams["olmo-seed"]
    for _, rel in fam.designs:
        p = registry.resolve(rel, tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"cells": ["c1", "c2"]}))
    assert ta.family_designs(fam, tmp_path) == [(fam.designs[0][0], ["c1", "c2"])]
    assert ta.family_teachers(fam, "elephant", "dd-1b-seed2", tmp_path) == ["ep10_lr4e-4_r64a128"]


def test_factor_designs_example(tmp_path, monkeypatch):
    doc = _example(tf.SPEC_NAME)
    assert tf.core65_manifests_from_spec(doc) == (("focus20", "focus20_cells.json"),)
    monkeypatch.setattr(tf, "_spec", lambda: doc)
    assert tf.XARM_INVENTORY_REL == registry.legacy_rel(doc["xarm_inventory"])
    assert tf._xarm_inventory_path(tmp_path) == registry.resolve(doc["xarm_inventory"], tmp_path)
    registry.registry_dir(tmp_path).mkdir(parents=True)
    cell = {"student_tag": "s", "lr": 1e-4, "rank": 8, "alpha": 32, "K_rel": 1.0}
    registry.registry_path(doc["focus20_cells"], tmp_path).write_text(json.dumps({"cells": [cell]}))
    registry.registry_path(doc["focus20_teachers"][1], tmp_path).write_text(
        json.dumps({"param_tags": ["t1"]}))
    cells, tags, _ = tf.load_focus20_design(tmp_path)
    assert len(cells) == 1 and tags == ["t1"]


def test_family_complete_example(tmp_path, monkeypatch):
    doc = _example(tfc.SPEC_NAME)
    grids = tfc.grid_paths_from_spec(doc)
    assert set(grids) == set(doc["grid_paths"]) and all(v.startswith("run_logs/") for v in grids.values())
    monkeypatch.setattr(tfc, "_spec", lambda: doc)
    assert tfc.GRID_PATHS == grids
    grid, rel = next(iter(grids.items()))
    p = registry.resolve(rel, tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"cells": ["c1", "c2"]}))
    assert tfc.canonical_cells(tmp_path, grid) == {"c1", "c2"}
    assert "{animal}" in doc["teachers"] and registry.resolve(doc["modern_reproduction_arms"], tmp_path)


@pytest.mark.skipif(not registry.registry_path(ta.SPEC_NAME).is_file(),
                    reason="run_logs registry not on this machine")
def test_asymmetry_factor_complete_constants_load_lazily():
    assert isinstance(ta.FAMILIES, dict) and isinstance(ta._PYTHIA_DESIGNS, tuple)
    assert isinstance(tf._CORE65_MANIFESTS, tuple) and isinstance(tf.XARM_INVENTORY_REL, str)
    assert isinstance(tfc.GRID_PATHS, dict)
    for mod in (ta, tf, tfc):
        with pytest.raises(AttributeError):
            mod.NOT_A_CONSTANT  # noqa: B018
