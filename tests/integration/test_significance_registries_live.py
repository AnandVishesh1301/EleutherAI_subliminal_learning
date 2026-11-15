"""The REAL arm registries and output tree (machine-local): load checks of the xarch / OLMo significance registries. Integration: skips cleanly elsewhere.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from subliminal_learning.analysis import setup
from subliminal_learning.analysis.loaders import transfer_significance_io as tio
from subliminal_learning.analysis.transfer import transfer_headline as th


# -------------------- from live_a.py --------------------

XARCH_REL = "run_logs/campaigns/xarch/xarch_significance_arms.json"

OLMO_REL = "run_logs/campaigns/repretrain/olmo_cross_seed_significance_arms.json"

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

@pytest.fixture
def real_tree():
    """``(repo, xarch, olmo)`` when the REAL arm registries and a populated output tree are on
    this machine (both are machine-local); skips cleanly elsewhere."""
    try:
        repo = setup.find_root()
        output_root = tio.resolve_output_root(None)
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    xarch, olmo = repo / XARCH_REL, repo / OLMO_REL
    if not (xarch.is_file() and olmo.is_file() and (output_root / "students").is_dir()):
        pytest.skip("real arm registries / output tree not on this machine")
    return repo, xarch, olmo

def _table(path):
    return tio.significance_table(path)

def test_registries_load_and_the_named_arms_are_tested_or_amber(real_tree):
    _, xarch, olmo = real_tree
    tables = {"xarch": _table(xarch), "olmo": _table(olmo)}
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

def test_headline_families_cover_the_renamed_arms(real_tree):
    repo, _, _ = real_tree
    teachers = {a: json.loads((repo / f"run_logs/registry/{a}_top10_kband_configs.json").read_text())
                ["param_tags"] for a in ("dog", "cat", "lion", "pig")}
    families = th.headline_families(repo, teachers)
    table = th.headline_table(families, n_perm=2000)
    assert set(table["family"]) == {f.family for f in families}


# -------------------- from live_b.py --------------------

# ----------------------------------------------------------------------------------------------
# real data, when the output tree is mounted
# ----------------------------------------------------------------------------------------------
OLMO_REGISTRY_REL = "run_logs/campaigns/repretrain/olmo_cross_seed_significance_arms.json"

@pytest.fixture
def olmo_registry():
    """The REAL OLMo arm registry, when it and a populated output tree are on this machine."""
    try:
        repo, output_root = setup.bootstrap()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    path = repo / OLMO_REGISTRY_REL
    if not (path.is_file() and (output_root / "students").is_dir()):
        pytest.skip("OLMo arm registry / output tree not mounted")
    return path

def test_olmo_same_vs_different_init_smoke(olmo_registry):
    """The OLMo cross-recipe comparison the notebook asks for, on whatever has finished."""
    names = {a["name"] for a in json.loads(olmo_registry.read_text())["arms"]}
    fam_a = [n for n in ["dclm → c4 (elephant), core35", "dclm → dolma (elephant), core35",
                         "dolma → dclm (elephant), core35"] if n in names]
    fam_b = [n for n in ["dclm → seed2 (elephant), core35",
                         "dclm → c4-seed2 (elephant), core35"] if n in names]
    try:
        res = tio.registry_family_comparison(
            olmo_registry, ("same init, different corpus", fam_a),
            ("different init, different corpus", fam_b), n_perm=2000)
    except ValueError as exc:            # nothing finished on one side yet
        pytest.skip(str(exc))
    table = tio.family_comparison_table({("same vs different init", "core35"): res})
    assert len(table) == len(tio.VALUE_MODES)
