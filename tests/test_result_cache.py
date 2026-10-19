import csv
import os
import threading

from subliminal_learning.analysis import result_cache as rc


def _files(tmp_path, n=3):
    fs = []
    for i in range(n):
        f = tmp_path / f"r{i}.json"
        f.write_text("{}")
        fs.append(f)
    return fs


def test_fingerprint_sensitivity(tmp_path):
    fs = _files(tmp_path)
    code = tmp_path / "mod.py"
    code.write_text("x = 1\n")
    base = rc.fingerprint(fs, params={"a": 1, "b": 2}, code=[code])
    assert rc.fingerprint(list(reversed(fs)), params={"b": 2, "a": 1}, code=[code]) == base
    triples = [(str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in fs]
    assert rc.fingerprint(triples[::-1], params={"a": 1, "b": 2}, code=[code]) == base
    assert rc.fingerprint(fs, params={"a": 1, "b": 3}, code=[code]) != base
    st = fs[0].stat()
    os.utime(fs[0], ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    k_mtime = rc.fingerprint(fs, params={"a": 1, "b": 2}, code=[code])
    assert k_mtime != base
    fs[1].write_text("{} ")
    os.utime(fs[1], ns=(fs[1].stat().st_atime_ns, triples[1][1]))
    assert rc.fingerprint(fs, params={"a": 1, "b": 2}, code=[code]) != k_mtime
    k = rc.fingerprint(fs, params={"a": 1, "b": 2}, code=[code])
    code.write_text("x = 2\n")
    assert rc.fingerprint(fs, params={"a": 1, "b": 2}, code=[code]) != k
    # missing file does not raise
    rc.fingerprint([tmp_path / "nope.json"], params={}, code=[rc])


def test_round_trip_and_corrupt(tmp_path):
    rc.store("ns", "pair/A→B", "k1", {"v": [1, 2]}, tmp_path)
    assert rc.load("ns", "pair/A→B", "k1", tmp_path) == {"v": [1, 2]}
    assert rc.load("ns", "pair/A→B", "missing", tmp_path) is None
    rc.entry_path("ns", "pair/A→B", "k1", tmp_path).write_bytes(b"garbage")
    assert rc.load("ns", "pair/A→B", "k1", tmp_path) is None
    calls = []
    out = rc.cached("ns", "pair/A→B", "k1", tmp_path, lambda: calls.append(1) or 42)
    assert out == 42 and calls == [1]
    assert rc.cached("ns", "pair/A→B", "k1", tmp_path, lambda: calls.append(1) or 0) == 42
    assert calls == [1]
    assert rc.cached("ns", "pair/A→B", "k1", tmp_path, lambda: 7, refresh=True) == 7


def test_disabled_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SL_RESULT_CACHE", "0")
    calls = []
    for _ in range(2):
        rc.cached("ns", "u", "k", tmp_path, lambda: calls.append(1) or 1)
    assert calls == [1, 1]
    assert not (tmp_path / "analysis").exists()


def test_prune_keeps_latest(tmp_path):
    for i, k in enumerate(["old", "mid", "new"]):
        rc.store("ns", "u", k, i, tmp_path)
        f = rc.entry_path("ns", "u", k, tmp_path)
        os.utime(f, ns=(0, 10**18 + i * 10**9))
    rc.store("ns", "other", "x", 0, tmp_path)
    removed = rc.prune("ns", tmp_path)
    assert len(removed) == 2
    assert rc.load("ns", "u", "new", tmp_path) == 2
    assert rc.load("ns", "u", "old", tmp_path) is None
    assert rc.load("ns", "other", "x", tmp_path) == 0


def test_concurrent_index(tmp_path):
    def work(t):
        for i in range(25):
            rc.store("ns", f"u{t}", f"k{i}", list(range(100)), tmp_path, n_input_records=i)

    ths = [threading.Thread(target=work, args=(t,)) for t in range(2)]
    [t.start() for t in ths]
    [t.join() for t in ths]
    with (tmp_path / "analysis/result_cache/ns/index.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 50 and all(set(r) == set(rc._INDEX_COLS) for r in rows)
    assert rc.load("ns", "u1", "k24", tmp_path) == list(range(100))
