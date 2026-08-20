"""Robustness contract of `analysis.record_cache` (and its `probe_cache` facade)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from subliminal_learning.analysis import probe_cache as pc
from subliminal_learning.analysis import record_cache as rc


def _probe(root: Path, rp: str, kl=1.0, nonfinite=False, pool_version="pp1", **extra):
    d = root / rp / "eval"
    d.mkdir(parents=True, exist_ok=True)
    rec = {"schema_version": 1, "pool_version": pool_version, "pool_sha": "abc",
           "kl_base_student": None if nonfinite else kl,
           "delta_ce": None if nonfinite else kl,
           "nonfinite_adapter": nonfinite,
           "manifest_ref": {"student_param_tag": "t", "filtered_sha256": "s",
                            "n_train_steps": 10}}
    rec.update(extra)
    (d / "pile_probe.json").write_text(json.dumps(rec))
    return d / "pile_probe.json"


def _load(root, paths, **kw):
    return rc.load_record_cache(root, "pile_probe", paths, verbose=False, **kw)


def test_union_persistence_disjoint_inventories(tmp_path, monkeypatch):
    _probe(tmp_path, "a")
    _probe(tmp_path, "b")
    assert list(_load(tmp_path, ["a"]).run_path) == ["a"]
    assert list(_load(tmp_path, ["b"]).run_path) == ["b"]      # must not evict "a"

    reads = []
    orig = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (reads.append(self), orig(self, *a, **k))[1])
    out = _load(tmp_path, ["a"])
    assert list(out.run_path) == ["a"]
    assert not [p for p in reads if p.name == "pile_probe.json"]   # served from cache


def test_missing_drops_but_other_oserror_keeps(tmp_path, monkeypatch):
    f = _probe(tmp_path, "a")
    _probe(tmp_path, "b")
    assert set(_load(tmp_path, ["a", "b"]).run_path) == {"a", "b"}

    real_stat = Path.stat

    def flaky(self, *a, **k):
        if self == f:
            raise PermissionError("transient")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", flaky)
    assert set(_load(tmp_path, ["a", "b"]).run_path) == {"a", "b"}   # hiccup never evicts
    monkeypatch.undo()
    assert set(_load(tmp_path, ["a", "b"], refresh=False).run_path) == {"a", "b"}

    f.unlink()
    assert set(_load(tmp_path, ["a", "b"]).run_path) == {"b"}
    assert set(_load(tmp_path, ["a", "b"], refresh=False).run_path) == {"b"}   # evicted for real


def test_corrupt_json_keeps_old_row_and_skips_new(tmp_path):
    f = _probe(tmp_path, "a", kl=2.0)
    assert _load(tmp_path, ["a"]).kl.iloc[0] == 2.0
    f.write_text("{not json")
    (tmp_path / "b" / "eval").mkdir(parents=True)
    (tmp_path / "b" / "eval" / "pile_probe.json").write_text("{also not json")
    out = _load(tmp_path, ["a", "b"])
    assert list(out.run_path) == ["a"] and out.kl.iloc[0] == 2.0


def test_nonfinite_is_inf_through_csv_roundtrip(tmp_path):
    _probe(tmp_path, "a", nonfinite=True)
    assert np.isinf(_load(tmp_path, ["a"]).kl.iloc[0])
    out = _load(tmp_path, ["a"], refresh=False)
    assert np.isinf(out.kl.iloc[0]) and np.isinf(out.delta_ce.iloc[0])


def test_pool_version_filter(tmp_path, capsys):
    _probe(tmp_path, "good")
    _probe(tmp_path, "old", pool_version="pp0")
    _probe(tmp_path, "legacy", pool_version=None)
    out = pc.load_probe_cache(tmp_path, ["good", "old", "legacy"])
    assert set(out.run_path) == {"good", "legacy"}
    assert (tmp_path / pc.CACHE_REL).is_file()
    # still cached, just filtered out of the returned frame
    assert set(pc.load_probe_cache(tmp_path, ["old"], pool_version=None).run_path) == {"old"}


def test_size_change_at_equal_mtime_triggers_reread(tmp_path):
    f = _probe(tmp_path, "a", kl=1.0)
    mt = f.stat().st_mtime
    assert _load(tmp_path, ["a"]).kl.iloc[0] == 1.0
    _probe(tmp_path, "a", kl=222.0)                 # longer body => different size
    import os

    os.utime(f, (mt, mt))
    assert _load(tmp_path, ["a"]).kl.iloc[0] == 222.0


def test_duplicate_inventory_paths_give_one_row(tmp_path):
    _probe(tmp_path, "a")
    out = _load(tmp_path, ["a", "a/", "a"])
    assert list(out.run_path) == ["a"]


def test_legacy_cache_without_record_size(tmp_path):
    _probe(tmp_path, "a", kl=3.0)
    _load(tmp_path, ["a"])
    f = tmp_path / pc.CACHE_REL
    df = __import__("pandas").read_csv(f).drop(columns=["record_size"])
    f.write_text(df.to_csv(index=False))
    out = _load(tmp_path, ["a"])                    # re-reads instead of crashing
    assert out.kl.iloc[0] == 3.0 and out.record_size.iloc[0] > 0


def test_fidelity_traj_derived_stats(tmp_path):
    d = tmp_path / "a" / "eval"
    d.mkdir(parents=True)
    (d / "fidelity_traj.json").write_text(json.dumps({"points": [
        {"step": 30, "ce_train": 0.4, "ce_heldout": 1.4},
        {"step": 10, "ce_train": 2.0, "ce_heldout": 2.1},
        {"step": 20, "ce_train": 0.3, "ce_heldout": 1.0},
        {"step": 40, "ce_train": None, "ce_heldout": 5.0},
    ]}))
    out = rc.load_record_cache(tmp_path, "fidelity_traj", ["a"], verbose=False)
    r = out.iloc[0]
    assert r.n_points == 3
    assert r.train_min == pytest.approx(0.3)
    assert r.max_rise == pytest.approx(0.1)         # 0.4 above running min 0.3
    assert r.gap_max == pytest.approx(1.0)          # step 30: 1.4 - 0.4


def test_too_few_points_still_gets_a_row(tmp_path):
    d = tmp_path / "a" / "eval"
    d.mkdir(parents=True)
    (d / "fidelity_traj.json").write_text(json.dumps({"points": [
        {"step": 1, "ce_train": 1.0, "ce_heldout": 1.2}]}))
    out = rc.load_record_cache(tmp_path, "fidelity_traj", ["a"], verbose=False)
    assert len(out) == 1 and out.n_points.iloc[0] == 1


def test_save_goes_through_atomic_write(tmp_path, monkeypatch):
    _probe(tmp_path, "a")
    calls = []
    real = rc.atomic_write_text
    monkeypatch.setattr(rc, "atomic_write_text",
                        lambda p, t: (calls.append(Path(p)), real(p, t))[1])
    _load(tmp_path, ["a"])
    assert calls == [tmp_path / pc.CACHE_REL]


def test_json_blob_kind_takes_a_per_call_record_path(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "jobs.jsonl.json").write_text('{"x": 1}')
    out = rc.load_record_cache(tmp_path, "json_blob", ["a"], verbose=False,
                               rel_record="jobs.jsonl.json",
                               cache_rel=Path("analysis") / "jobs_cache.csv")
    assert json.loads(out.blob.iloc[0]) == {"x": 1}
    assert (tmp_path / "analysis" / "jobs_cache.csv").is_file()
