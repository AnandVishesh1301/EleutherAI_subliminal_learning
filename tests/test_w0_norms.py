"""analysis.w0_norms: snapshot-keyed, merge-on-write W0 norm cache."""

import json

from subliminal_learning.analysis import w0_norms as wn


def _hub(tmp_path, repo, sha, rev="main"):
    ref = tmp_path / "hf" / "hub" / f"models--{repo.replace('/', '--')}" / "refs" / rev
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text(sha + "\n")
    return tmp_path / "hf"


def test_key_includes_snapshot_and_revision(tmp_path):
    hf = _hub(tmp_path, "org/m", "a" * 40)
    _hub(tmp_path, "org/m", "b" * 40, rev="step5000")
    p_main, sha = wn.w0_cache_path(tmp_path, "org/m", hf_cache_root=hf)
    p_rev, sha_rev = wn.w0_cache_path(tmp_path, "org/m", revision="step5000", hf_cache_root=hf)
    assert sha == "a" * 40 and sha_rev == "b" * 40
    assert p_main.name == f"org__m__{'a' * 12}.json" and p_rev.name == f"org__m__{'b' * 12}.json"
    # unresolvable -> name-only key, no crash
    p_none, sha_none = wn.w0_cache_path(tmp_path, "org/other", hf_cache_root=hf)
    assert sha_none is None and p_none.name == "org__other.json"


def test_merge_never_truncates_and_records_meta(tmp_path):
    hf = _hub(tmp_path, "org/m", "a" * 40)
    wn.merge_w0_norms(tmp_path, "org/m", {"l0": {"W0_fro": 1.0, "W0_spec": 0.5}},
                      hf_cache_root=hf)
    wn.merge_w0_norms(tmp_path, "org/m", {"l1": {"W0_fro": 2.0, "W0_spec": 1.5}},
                      hf_cache_root=hf)
    got = wn.load_w0_norms(tmp_path, "org/m", hf_cache_root=hf)
    assert set(got) == {"l0", "l1"} and got["l0"]["W0_fro"] == 1.0
    raw = json.loads(wn.w0_cache_path(tmp_path, "org/m", hf_cache_root=hf)[0].read_text())
    assert raw[wn.META_KEY]["snapshot_sha"] == "a" * 40


def test_legacy_name_only_file_migrates(tmp_path):
    hf = _hub(tmp_path, "org/m", "a" * 40)
    legacy = tmp_path / "analysis" / "w0_norms" / "org__m.json"
    legacy.parent.mkdir(parents=True)
    legacy.write_text(json.dumps({"l0": {"W0_fro": 3.0, "W0_spec": 2.0}}))
    got = wn.load_w0_norms(tmp_path, "org/m", hf_cache_root=hf)
    assert got == {"l0": {"W0_fro": 3.0, "W0_spec": 2.0}}
    assert wn.w0_cache_path(tmp_path, "org/m", hf_cache_root=hf)[0].is_file()


def test_ensure_computes_only_missing(tmp_path, monkeypatch):
    hf = _hub(tmp_path, "org/m", "a" * 40)
    wn.merge_w0_norms(tmp_path, "org/m", {"l0": {"W0_fro": 1.0, "W0_spec": 0.5}},
                      hf_cache_root=hf)
    assert wn.ensure_w0_norms(tmp_path, "org/m", ["l0"], hf_cache_root=hf)["l0"]["W0_fro"] == 1.0
