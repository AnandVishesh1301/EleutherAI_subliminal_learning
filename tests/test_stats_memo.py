"""`stats.memo`: the content-addressed memo behind the seeded permutation tests."""
from __future__ import annotations

import json

import numpy as np
import pytest

from subliminal_learning.stats import memo
from subliminal_learning.stats import transfer_significance_tests as tt


@pytest.fixture
def cache(tmp_path, monkeypatch):
    """A memo pointed at `tmp_path` (the autouse fixture turns it off by default)."""
    monkeypatch.setenv(memo.ENV_VAR, "1")
    memo.set_cache_dir(tmp_path)
    memo.reset_cache_stats()
    yield tmp_path
    memo.set_cache_dir(False)


def _values():
    rng = np.random.default_rng(3)
    v = rng.random((6, 4))
    v[0, 1] = np.nan
    return v


def test_hit_on_identical_inputs_is_exact_and_counted(cache):
    v = _values()
    first = tt.stratified_exceedance_permutation(v, 3, n_perm=300, seed=0)
    assert memo.cache_stats() == {"hits": 0, "misses": 1}
    files = list(cache.rglob("*.json"))
    assert len(files) == 1 and files[0].parent.name == "stratified_exceedance_permutation"
    second = tt.stratified_exceedance_permutation(v.copy(), 3, n_perm=300, seed=0)
    assert memo.cache_stats() == {"hits": 1, "misses": 1}
    assert second == first                     # the SAME result object fields, not a re-draw


@pytest.mark.parametrize("change", ["value", "seed", "n_perm", "control", "groups", "shape"])
def test_any_input_change_is_a_miss(cache, change):
    v = _values()
    base = dict(control_index=3, n_perm=300, seed=0)
    tt.stratified_exceedance_permutation(v, **base)
    kw, w = dict(base), v.copy()
    if change == "value":
        w[2, 0] += 1e-9
    elif change == "seed":
        kw["seed"] = 1
    elif change == "n_perm":
        kw["n_perm"] = 301
    elif change == "control":
        kw["control_index"] = 2
    elif change == "groups":
        kw["groups"] = ["a"] * 3 + ["b"] * 3
    elif change == "shape":
        w = w[:5]
    tt.stratified_exceedance_permutation(w, **kw)
    assert memo.cache_stats() == {"hits": 0, "misses": 2}
    assert len(list(cache.rglob("*.json"))) == 2


def test_env_var_disables_reads_and_writes(cache, monkeypatch):
    v = _values()
    tt.stratified_exceedance_permutation(v, 3, n_perm=300, seed=0)
    assert len(list(cache.rglob("*.json"))) == 1
    monkeypatch.setenv(memo.ENV_VAR, "0")
    memo.reset_cache_stats()
    tt.stratified_exceedance_permutation(v, 3, n_perm=300, seed=0)
    assert memo.cache_stats() == {"hits": 0, "misses": 1}       # recomputed, not read
    assert len(list(cache.rglob("*.json"))) == 1                # and nothing new written
    assert memo.cache_dir() is None


def test_corrupt_entry_recomputes_and_overwrites(cache):
    v = _values()
    first = tt.stratified_exceedance_permutation(v, 3, n_perm=300, seed=0)
    path = next(cache.rglob("*.json"))
    path.write_text("{not json")
    memo.reset_cache_stats()
    again = tt.stratified_exceedance_permutation(v, 3, n_perm=300, seed=0)
    assert again == first and memo.cache_stats() == {"hits": 0, "misses": 1}
    assert json.loads(path.read_text())["observed"] == first.observed   # overwritten, readable
    # a wrong-schema entry (fields missing) is a miss too, never a raise
    path.write_text(json.dumps({"observed": 1}))
    assert tt.stratified_exceedance_permutation(v, 3, n_perm=300, seed=0) == first


def test_arm_level_tests_memoize_too(cache):
    import pandas as pd

    arms = pd.DataFrame({"family": list("aaabbb"), "value": [0.5, 0.6, 0.7, 0.1, 0.2, 0.3]})
    r1 = tt.arm_set_permutation(arms, family_a="a", family_b="b", n_perm=100)
    r2 = tt.arm_set_permutation(arms, family_a="a", family_b="b", n_perm=100)
    assert r1 == r2 and memo.cache_stats() == {"hits": 1, "misses": 1}
    # the family labels are part of the key: swapping the claimed family is a miss
    tt.arm_set_permutation(arms, family_a="b", family_b="a", n_perm=100)
    assert memo.cache_stats()["misses"] == 2
    rows = [{"family": f, "arm": a, "stratum": c, "value": x}
            for f, a in (("a", "x1"), ("a", "x2"), ("b", "y1"), ("b", "y2"))
            for c, x in (("c1", 0.1), ("c2", 0.2))]
    frame = pd.DataFrame(rows)
    frame.loc[frame.family == "a", "value"] += 0.3
    memo.reset_cache_stats()
    k1 = tt.arm_relabel_cell_permutation(frame, family_a="a", family_b="b", n_perm=100)
    k2 = tt.arm_relabel_cell_permutation(frame, family_a="a", family_b="b", n_perm=100)
    assert k1 == k2 and memo.cache_stats() == {"hits": 1, "misses": 1}
    tt.arm_relabel_cell_permutation(frame, family_a="a", family_b="b", n_perm=100,
                                    statistic="win_probability")
    assert memo.cache_stats()["misses"] == 2


def test_key_distinguishes_types_and_version():
    assert memo.key("f", 1) != memo.key("f", "1") != memo.key("f", 1.0)
    assert memo.key("f", np.array([1.0])) != memo.key("f", np.array([1]))
    assert memo.key("f", 1) != memo.key("g", 1)
    with pytest.raises(TypeError):
        memo.key("f", object())


# ----------------------------------------------------------------------------------------------
# the loader cache: load_cell_matrix / _arm_dead_matrix keyed on the run leaves' (size, mtime)
# ----------------------------------------------------------------------------------------------
def _same_matrix(a, b) -> bool:
    return (a.cells == b.cells and a.pools == b.pools and a.control_col == b.control_col
            and a.missing == b.missing and a.control_layout == b.control_layout
            and a.no_control == b.no_control and np.array_equal(a.values, b.values, equal_nan=True)
            and a.frame.equals(b.frame))


def test_loader_cache_hits_and_a_finished_run_invalidates(tmp_path, monkeypatch):
    from subliminal_learning.analysis import transfer_significance_io as tio
    from tests.test_family_comparison import _family_tree, _write_seed_eval

    root, cache_dir = tmp_path / "out", tmp_path / "cache"
    _family_tree(root)
    monkeypatch.setenv(memo.ENV_VAR, "1")
    memo.set_cache_dir(cache_dir)
    memo.reset_cache_stats()
    kw = dict(ftseed=1, cells=["c1", "c2"], output_root=root)
    cold = tio.load_cell_matrix("dog", "orig", "ds1", ["t1", "t2"], **kw)
    assert memo.cache_stats() == {"hits": 0, "misses": 1}
    warm = tio.load_cell_matrix("dog", "orig", "ds1", ["t1", "t2"], **kw)
    assert memo.cache_stats() == {"hits": 1, "misses": 1} and _same_matrix(cold, warm)
    # the cached copy equals an uncached read byte for byte
    monkeypatch.setenv(memo.ENV_VAR, "0")
    assert _same_matrix(cold, tio.load_cell_matrix("dog", "orig", "ds1", ["t1", "t2"], **kw))
    monkeypatch.setenv(memo.ENV_VAR, "1")
    # a run finishing on disk (new eval file) is a new key: recomputed, value visible -- once
    # the in-process signature memo (a bounded TTL, so one cell's repeated reads of the same
    # arm stat once) has expired
    leaf = (root / "students/dog/pythia-160m-data-seed1/teacher-pythia-160m/t1_tseed1"
            / "ex10-20_prefix_n100k/c1")
    _write_seed_eval(leaf / "ftseed1/preference_eval.json", 0.9)
    memo.reset_cache_stats()
    assert tio.load_cell_matrix("dog", "orig", "ds1", ["t1", "t2"], **kw).values[0, 0] == 0.5
    assert memo.cache_stats() == {"hits": 1, "misses": 0}       # within the TTL: unseen
    monkeypatch.setattr(tio, "_SIGNATURE_TTL", 0.0)
    memo.reset_cache_stats()
    fresh = tio.load_cell_matrix("dog", "orig", "ds1", ["t1", "t2"], **kw)
    assert memo.cache_stats() == {"hits": 0, "misses": 1}
    assert fresh.values[0, 0] == pytest.approx(0.9) and cold.values[0, 0] == pytest.approx(0.5)
    # the whole comparison rides on the same two loaders: a second call is all hits
    memo.reset_cache_stats()
    tio.compare_families(["orig", "ds1"], ["orig", "fs1"], "dog", ["t1", "t2"], family_a="ds",
                         family_b="fs", cells=["c1", "c2"], n_perm=200, output_root=root)
    first = memo.cache_stats()
    memo.reset_cache_stats()
    tio.compare_families(["orig", "ds1"], ["orig", "fs1"], "dog", ["t1", "t2"], family_a="ds",
                         family_b="fs", cells=["c1", "c2"], n_perm=200, output_root=root)
    assert memo.cache_stats()["misses"] == 0 and memo.cache_stats()["hits"] == (
        first["hits"] + first["misses"])
    # nothing under the repo, everything under the cache dir
    assert {p.name for p in cache_dir.iterdir()} >= {"load_cell_matrix", "arm_dead_matrix",
                                                      "arm_relabel_cell_permutation"}


def test_loader_cache_skips_when_cells_are_not_pinned(tmp_path, monkeypatch):
    """`cells=None` lists the control tree itself; that read is never cached."""
    from subliminal_learning.analysis import transfer_significance_io as tio
    from tests.test_family_comparison import _family_tree

    root = tmp_path / "out"
    _family_tree(root)
    monkeypatch.setenv(memo.ENV_VAR, "1")
    memo.set_cache_dir(tmp_path / "cache")
    memo.reset_cache_stats()
    m = tio.load_cell_matrix("dog", "orig", "ds1", ["t1"], ftseed=1, output_root=root)
    assert m.cells == ["c1", "c2"] and memo.cache_stats() == {"hits": 0, "misses": 0}


def test_entry_without_null_hist_is_a_miss_and_is_overwritten(cache):
    """A memo entry written before `null_hist` existed cannot rebuild the dataclass, so it is
    a MISS: the result is recomputed with its histogram and the file on disk gains it."""
    v = _values()
    fresh = tt.stratified_exceedance_permutation(v, 3, n_perm=300, seed=0)
    path, = cache.rglob("*.json")
    old = json.loads(path.read_text())
    old.pop("null_hist")
    path.write_text(json.dumps(old))
    memo.reset_cache_stats()
    again = tt.stratified_exceedance_permutation(v, 3, n_perm=300, seed=0)
    assert memo.cache_stats() == {"hits": 0, "misses": 1}
    assert again == fresh and again.null_hist
    assert json.loads(path.read_text())["null_hist"] == fresh.null_hist
