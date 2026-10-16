"""Robustness contract of `analysis.record_cache` (and its `probe_cache` facade)."""
from __future__ import annotations

import json
import os
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


def test_same_size_rewrite_below_old_tolerance_refreshes(tmp_path):
    f = _probe(tmp_path, "a", kl=1.0)
    first = _load(tmp_path, ["a"])
    old_ns = f.stat().st_mtime_ns
    # Keep the payload size unchanged and move mtime by 50 microseconds.
    _probe(tmp_path, "a", kl=2.0)
    os.utime(f, ns=(old_ns + 50_000, old_ns + 50_000))
    second = _load(tmp_path, ["a"])
    assert float(first.iloc[0].kl) == 1.0
    assert float(second.iloc[0].kl) == 2.0


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
    # Strict since 2026-08-27: version-less rows are REJECTED like wrong-pool rows (the
    # live cache census showed zero of them, so the flip cannot shrink any population).
    assert set(out.run_path) == {"good"}
    assert "REJECTED" in capsys.readouterr().out
    assert (tmp_path / pc.CACHE_REL).is_file()
    # still cached, just filtered out of the returned frame
    assert set(
        pc.load_probe_cache(tmp_path, ["old", "legacy"], pool_version=None).run_path
    ) == {"old", "legacy"}


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


def test_new_kind_column_forces_one_full_reread(tmp_path):
    """Adding a column to a kind must backfill rows already cached (records are unchanged, so
    (mtime, size) alone would keep them stale forever)."""
    import pandas as pd
    from subliminal_learning.analysis import record_cache as rc

    root = tmp_path
    run = "students/x/r1"
    f = root / run / rc.PILE_PROBE.rel_record
    f.parent.mkdir(parents=True)
    f.write_text('{"kl_base_student": 0.5, "ce_student": 2.9, "ce_base": 2.3, '
                 '"delta_ce": 0.6, "nonfinite_adapter": false, "pool_version": "pp1"}')
    df = rc.load_record_cache(root, rc.PILE_PROBE, [run])
    assert df.ce_student.iloc[0] == 2.9
    # simulate a CSV written before ce_student/ce_base existed, at the same (mtime, size)
    csv = rc.cache_path(root, rc.PILE_PROBE, None)
    old = pd.read_csv(csv).drop(columns=["ce_student", "ce_base"])
    csv.write_text(old.to_csv(index=False))
    df2 = rc.load_record_cache(root, rc.PILE_PROBE, [run])
    assert df2.ce_student.iloc[0] == 2.9 and df2.ce_base.iloc[0] == 2.3
    # and a second load is a no-op refresh (row fresh again)
    df3 = rc.load_record_cache(root, rc.PILE_PROBE, [run])
    assert len(df3) == 1 and df3.ce_student.iloc[0] == 2.9


def test_warm_sweep_reads_nothing_and_never_rewrites(tmp_path, monkeypatch):
    # 150 records with sub-microsecond-varied mtimes: pandas' default float parser perturbs
    # ~12% of such mtimes by one ulp, which under the exact-mtime contract turned every warm
    # sweep into re-reads plus a full cache rewrite. Round-trip parsing must make it a no-op.
    rng = np.random.default_rng(0)
    paths = [f"r{i}" for i in range(150)]
    for rp in paths:
        f = _probe(tmp_path, rp, kl=1.0)
        ns = 1_787_164_395_000_000_000 + int(rng.integers(0, 10**9))
        os.utime(f, ns=(ns, ns))
    _load(tmp_path, paths)
    cache_f = rc.cache_path(tmp_path, rc.get_kind("pile_probe"))
    before = cache_f.stat().st_mtime_ns

    reads = []
    orig = Path.read_text
    monkeypatch.setattr(Path, "read_text",
                        lambda self, *a, **k: (reads.append(self), orig(self, *a, **k))[1])
    out = _load(tmp_path, paths)
    assert len(out) == 150
    assert not [p for p in reads if p.name == "pile_probe.json"]   # nothing re-read
    assert cache_f.stat().st_mtime_ns == before                     # nothing rewritten


def test_csv_memo_parses_once_until_the_file_changes(tmp_path, monkeypatch):
    _probe(tmp_path, "a")
    _load(tmp_path, ["a"])                       # creates the cache (parse count starts here)
    parses = []
    orig = rc.pd.read_csv
    monkeypatch.setattr(rc.pd, "read_csv", lambda *a, **k: (parses.append(a[0]), orig(*a, **k))[1])
    _load(tmp_path, ["a"])
    _load(tmp_path, ["a"])
    assert parses == []                          # served from the per-process memo
    _probe(tmp_path, "b")
    _load(tmp_path, ["b"])                       # this process wrote it -> memo updated, no parse
    assert parses == []
    cache_f = rc.cache_path(tmp_path, rc.get_kind("pile_probe"))
    cache_f.write_text(cache_f.read_text())      # an external rewrite (other process / node)
    os.utime(cache_f, ns=(cache_f.stat().st_mtime_ns + 1_000_000,) * 2)
    out = _load(tmp_path, ["a", "b"])
    assert len(parses) == 1                      # re-parsed once the stat changed
    assert sorted(out.run_path) == ["a", "b"]


def test_sweep_outcomes_keep_inventory_order_under_threads(tmp_path):
    paths = [f"p{i}" for i in range(40)]
    for rp in paths:
        _probe(tmp_path, rp, kl=float(rp[1:]))
    out = _load(tmp_path, paths)
    assert list(out.run_path) == paths
    assert list(out.kl) == [float(i) for i in range(40)]


def test_unrecorded_inventory_path_does_not_force_a_rewrite(tmp_path):
    _probe(tmp_path, "a")
    _load(tmp_path, ["a", "never_evaluated"])      # creates the cache; "never_evaluated" absent
    cache_f = rc.cache_path(tmp_path, rc.get_kind("pile_probe"))
    before = cache_f.stat().st_mtime_ns
    out = _load(tmp_path, ["a", "never_evaluated"])
    assert list(out.run_path) == ["a"]
    assert cache_f.stat().st_mtime_ns == before      # nothing changed -> no rewrite
    (tmp_path / "a" / "eval" / "pile_probe.json").unlink()
    out = _load(tmp_path, ["a", "never_evaluated"])  # a CACHED row vanished -> must persist
    assert len(out) == 0
    assert cache_f.stat().st_mtime_ns != before


def test_serialized_cache_round_trips_cell_for_cell(tmp_path):
    # The persist path (pyarrow when importable) must be read back identically by the
    # round-trip parser: exact mtimes, +inf (diverged), NaN, and blobs containing newlines.
    _probe(tmp_path, "inf", nonfinite=True)
    _probe(tmp_path, "fin", kl=0.123456789012345678)
    first = _load(tmp_path, ["inf", "fin"])
    rc._CSV_MEMO.clear()                              # force a real parse of the file
    again = _load(tmp_path, ["inf", "fin"], refresh=False)
    assert list(again.columns) == list(first.columns)
    for c in first.columns:
        a, b = first[c], again[c]
        ok = (a.astype(object).values == b.astype(object).values) | (a.isna().values & b.isna().values)
        assert ok.all(), c
    assert np.isinf(again.set_index("run_path").loc["inf", "kl"])

    blob_dir = tmp_path / "b"
    blob_dir.mkdir()
    (blob_dir / "doc.json").write_text(json.dumps({"x": "line1\nline2", "y": [1, 2]}, indent=2))
    out = rc.load_record_cache(tmp_path, "json_blob", ["b"], rel_record="doc.json",
                               cache_rel="analysis/blob_test.csv", verbose=False)
    rc._CSV_MEMO.clear()
    back = rc.load_record_cache(tmp_path, "json_blob", ["b"], rel_record="doc.json",
                                cache_rel="analysis/blob_test.csv", refresh=False, verbose=False)
    assert json.loads(back.blob.iloc[0]) == json.loads(out.blob.iloc[0]) == {"x": "line1\nline2", "y": [1, 2]}


def test_mixed_type_column_falls_back_to_pandas_persist(tmp_path):
    # A raw record field with inconsistent types across records (int in one, str in another)
    # makes an object column pyarrow cannot convert; the persist must still succeed.
    _probe(tmp_path, "a", manifest_ref={"student_param_tag": "t", "filtered_sha256": "s",
                                        "n_train_steps": 10})
    _probe(tmp_path, "b", manifest_ref={"student_param_tag": "t", "filtered_sha256": "s",
                                        "n_train_steps": "ten"})
    out = _load(tmp_path, ["a", "b"])
    assert sorted(out.run_path) == ["a", "b"]
    rc._CSV_MEMO.clear()
    back = _load(tmp_path, ["a", "b"], refresh=False)
    assert sorted(back.run_path) == ["a", "b"]
    assert set(back.n_train_steps.astype(str)) == {"10", "ten"}


def _three_kind_cache(root: Path):
    """Synthetic records for three cache kinds: pile_probe and two json_blob record paths."""
    for i in range(6):
        _probe(root, f"r{i}", kl=0.1 * i)
        d = root / f"r{i}"
        (d / "doc.json").write_text(json.dumps({"i": i, "s": "a\nb"}))
        (d / "doc2.json").write_text(json.dumps({"j": -i}))


def _same(a, b) -> bool:
    """Cell-identical (NaN == NaN), ignoring dtype: fresh rows are object-typed until re-parsed."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    return all(((a[c].astype(object).values == b[c].astype(object).values)
                | (a[c].isna().values & b[c].isna().values)).all() for c in a.columns)


def _three_loads(root, paths, **kw):
    return (rc.load_record_cache(root, "pile_probe", paths, verbose=False, **kw),
            rc.load_record_cache(root, "json_blob", paths, rel_record="doc.json",
                                 cache_rel="analysis/blob_a.csv", verbose=False, **kw),
            rc.load_record_cache(root, "json_blob", paths, rel_record="doc2.json",
                                 cache_rel="analysis/blob_b.csv", verbose=False, **kw))


def test_warm_call_builds_lookup_once_per_cache_key(tmp_path, monkeypatch):
    rc._CSV_MEMO.clear()
    paths = [f"r{i}" for i in range(6)]
    _three_kind_cache(tmp_path)
    cold = _three_loads(tmp_path, paths)
    rc._CSV_MEMO.clear()
    calls = []
    orig = rc._build_index
    monkeypatch.setattr(rc, "_build_index", lambda f: (calls.append(len(f)), orig(f))[1])
    # (a) warm: one index build per cache file (the parse), then none on repeat calls.
    first = _three_loads(tmp_path, paths[:2])
    assert len(calls) == 3
    caches = [rc.cache_path(tmp_path, rc.get_kind("pile_probe")),
              tmp_path / "analysis/blob_a.csv", tmp_path / "analysis/blob_b.csv"]
    stamps = [c.stat().st_mtime_ns for c in caches]
    for _ in range(3):
        again = _three_loads(tmp_path, paths[:2])
    assert len(calls) == 3                                   # no per-call rebuild
    assert [c.stat().st_mtime_ns for c in caches] == stamps  # no-write hit
    # (d) identical results to the cold (index-free sweep) load, filtered to the inventory.
    for c, f, a in zip(cold, first, again):
        exp = c[c.run_path.isin(paths[:2])].reset_index(drop=True)
        assert _same(f, exp) and _same(a, exp)
    full = _three_loads(tmp_path, paths)
    for c, f in zip(cold, full):
        assert _same(f, c)
    # mutating a returned frame never leaks into the memo
    full[0]["kl"] = -1.0
    assert (_three_loads(tmp_path, paths)[0].kl != -1.0).all()


def test_indexed_lookup_still_rereads_changes_and_unions_new_paths(tmp_path):
    rc._CSV_MEMO.clear()
    _probe(tmp_path, "a", kl=1.0)
    _probe(tmp_path, "b", kl=2.0)
    _load(tmp_path, ["a", "b"])
    # (b) mtime change at equal size, then size change at equal mtime -> re-read
    f = _probe(tmp_path, "a", kl=3.0)
    st = f.stat()
    os.utime(f, ns=(st.st_mtime_ns + 1, st.st_mtime_ns + 1))
    assert _load(tmp_path, ["a"]).kl.iloc[0] == 3.0
    ns = f.stat().st_mtime_ns
    _probe(tmp_path, "a", kl=33.0)
    os.utime(f, ns=(ns, ns))
    assert _load(tmp_path, ["a"]).kl.iloc[0] == 33.0
    # (c) a new path is added and the union persisted (b survives a disjoint inventory)
    _probe(tmp_path, "c", kl=4.0)
    assert list(_load(tmp_path, ["c"]).kl) == [4.0]
    rc._CSV_MEMO.clear()
    back = _load(tmp_path, ["a", "b", "c"], refresh=False)
    assert dict(zip(back.run_path, back.kl)) == {"a": 33.0, "b": 2.0, "c": 4.0}
