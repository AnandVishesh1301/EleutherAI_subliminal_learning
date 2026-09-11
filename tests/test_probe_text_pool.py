"""Cross-tokenizer probe pools (`pp1t-<tag>`): frozen text artifact, families, anchor math."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from subliminal_learning.cka.pool_io import SequencePool
from subliminal_learning.coherence.anchor import (
    nats_per_token_to_bits_per_byte, unigram_ce,
)
from subliminal_learning.coherence.probe_families import (
    PROBE_FAMILIES, family_for_pool_version, is_pp1t, pool_version_for,
)
from subliminal_learning.coherence.probe_io import (
    build_frozen_text_pool, build_probe_pool, probe_pool_spec_pp1t,
)


class FakeTok:
    """Char-per-token tokenizer: id = ord(c) % 7 + 10, so ids stay small and stable."""

    bos_token_id = 3

    def __call__(self, text_or_list, add_special_tokens=False):
        assert add_special_tokens is False
        texts = [text_or_list] if isinstance(text_or_list, str) else list(text_or_list)
        ids = [[ord(c) % 7 + 10 for c in t] for t in texts]
        return {"input_ids": ids[0] if isinstance(text_or_list, str) else ids}

    def __len__(self) -> int:
        return 32


def _write_artifact(tmp_path, texts, *, n_scanned=77, corrupt=False):
    recs = [dict(order_pos=i, row=100 + i, text=t, n_chars=len(t),
                 n_bytes=len(t.encode()), pythia_n_tokens=len(t), round_trip_exact=True)
            for i, t in enumerate(texts)]
    h = hashlib.sha256()
    for r in recs:
        h.update(json.dumps(r["text"]).encode())
    sha = h.hexdigest()
    if corrupt:
        recs[0]["text"] = recs[0]["text"] + "X"
    p = tmp_path / "pp1_text.jsonl"
    mp = tmp_path / "pp1_text.meta.json"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    mp.write_text(json.dumps(dict(version="pp1t", n_docs=len(recs), n_scanned=n_scanned,
                                  text_sha256=sha, text_bytes_total=1234,
                                  n_round_trip_exact=len(recs))))
    return p, mp


def test_sequence_pool_positional_construction_unchanged():
    pool = SequencePool([[1, 2, 3]], [[1, 2]], "x")
    assert pool.row_indices is None and pool.source == "x"


def test_frozen_text_pool_bos_and_eligibility(tmp_path):
    texts = ["abcdefghij" * 4, "klmnopqrst" * 5]
    p, mp = _write_artifact(tmp_path, texts)
    spec = probe_pool_spec_pp1t("qwen25")

    plain = build_frozen_text_pool(FakeTok(), spec, bos_token_id=None, path=p, meta_path=mp)
    assert [len(ids) for ids in plain.input_ids] == [40, 50]
    assert plain.eligible[0] == list(range(1, 40))
    assert plain.n_scanned == 77 and plain.n_skipped == 0
    assert plain.row_indices == [100, 101]
    assert plain.source.startswith("pp1_text:")

    withbos = build_frozen_text_pool(FakeTok(), spec, bos_token_id=3, path=p, meta_path=mp)
    assert withbos.input_ids[0] == [3] + plain.input_ids[0]
    assert withbos.eligible[0] == list(range(1, 41))


def test_frozen_text_pool_rejects_corruption_and_degenerate_docs(tmp_path):
    p, mp = _write_artifact(tmp_path, ["abcdefghij" * 4, "klmno" * 8], corrupt=True)
    with pytest.raises(ValueError, match="corrupt"):
        build_frozen_text_pool(FakeTok(), probe_pool_spec_pp1t("qwen25"),
                               bos_token_id=None, path=p, meta_path=mp)

    # min-32 is a freeze-time Pythia-token selection rule, not a per-family check: a
    # 31-token doc under this family tokenizer is kept, only a doc with no CE target
    # (targets are ids[2:]) is an error.
    (tmp_path / "b").mkdir()
    p2, mp2 = _write_artifact(tmp_path / "b", ["abcdefghij" * 4, "ab"])
    with pytest.raises(ValueError, match="min 3"):
        build_frozen_text_pool(FakeTok(), probe_pool_spec_pp1t("qwen25"),
                               bos_token_id=None, path=p2, meta_path=mp2)

    (tmp_path / "c").mkdir()
    p3, mp3 = _write_artifact(tmp_path / "c", ["abcdefghij" * 4, "k" * 31])
    pool = build_frozen_text_pool(FakeTok(), probe_pool_spec_pp1t("qwen25"),
                                  bos_token_id=None, path=p3, meta_path=mp3)
    assert len(pool.input_ids[1]) == 31
    assert pool.eligible[1] == list(range(1, 31)) and len(pool.eligible[1]) == 30


def test_family_table_and_pool_versions():
    assert set(PROBE_FAMILIES) == {"pythia", "olmo", "qwen25", "llama3", "gemma3", "gemma2"}
    assert [t for t, f in PROBE_FAMILIES.items() if f.bos] == ["llama3", "gemma3", "gemma2"]
    assert pool_version_for("llama3") == "pp1t-llama3"
    assert family_for_pool_version("pp1t-llama3").tag == "llama3"
    assert is_pp1t("pp1t-qwen25") and not is_pp1t("pp1")
    with pytest.raises(ValueError, match="qwen25"):
        family_for_pool_version("pp1t-nope")
    with pytest.raises(ValueError, match="qwen25"):
        family_for_pool_version("pp1")
    with pytest.raises(ValueError, match="qwen25"):
        pool_version_for("nope")


def test_unigram_ce_and_bits_per_byte():
    counts = np.array([2, 1, 0])
    # total 3, alpha 0.5, V 3 -> denom 4.5; targets 0 and 2 -> p 2.5/4.5 and 0.5/4.5
    expected = -(math.log(2.5 / 4.5) + math.log(0.5 / 4.5)) / 2
    assert unigram_ce(counts, np.array([0, 2]), 0.5) == pytest.approx(expected)
    assert nats_per_token_to_bits_per_byte(math.log(2), 10, 5) == pytest.approx(2.0)


def test_build_probe_pool_routes_pp1t(tmp_path, monkeypatch):
    p, mp = _write_artifact(tmp_path, ["abcdefghij" * 4, "klmnopqrst" * 5])
    import subliminal_learning.cka.pool_io as pool_io
    import subliminal_learning.coherence.probe_io as probe_io

    monkeypatch.setattr(probe_io, "PP1_TEXT_PATH", p)
    monkeypatch.setattr(probe_io, "PP1_TEXT_META_PATH", mp)
    monkeypatch.setattr(pool_io, "build_pile_pool",
                        lambda *a, **k: pytest.fail("must not build the Pile pool"))
    pool = build_probe_pool(FakeTok(), probe_pool_spec_pp1t("qwen25"), verify_pin=False)
    assert len(pool) == 2 and pool.input_ids[0][0] != FakeTok.bos_token_id


def test_build_probe_pool_enforces_the_pinned_token_sha(tmp_path, monkeypatch):
    """A pp1t build under the wrong tokenizer is fatal, unless explicitly opted out."""
    p, mp = _write_artifact(tmp_path, ["abcdefghij" * 4, "klmnopqrst" * 5])
    import subliminal_learning.coherence.probe_io as probe_io

    monkeypatch.setattr(probe_io, "PP1_TEXT_PATH", p)
    monkeypatch.setattr(probe_io, "PP1_TEXT_META_PATH", mp)
    spec = probe_pool_spec_pp1t("qwen25")
    with pytest.raises(ValueError, match="pinned tokenization"):
        build_probe_pool(FakeTok(), spec)
    assert len(build_probe_pool(FakeTok(), spec, verify_pin=False)) == 2
    # a None pin still means "not yet pinned", never "no check"
    monkeypatch.setitem(probe_io.PP1T_TOKEN_SHA, "qwen25", None)
    assert len(build_probe_pool(FakeTok(), spec)) == 2


# --- base-repo -> family routing, and the anchor's identity gates ---------------------

@pytest.mark.parametrize("repo,tag", [
    ("Qwen/Qwen2.5-1.5B", "qwen25"),
    ("Qwen/Qwen2.5-1.5B@abc", "qwen25"),
    ("meta-llama/Llama-3.2-1B", "llama3"),
    ("google/gemma-3-1b-pt", "gemma3"),
    ("EleutherAI/pythia-160m", "pythia"),
    ("local/pythia-160m-weightc-seed1@step25000", "pythia"),
    ("allenai/DataDecide-dolma1_7-1B@step69369-seed-default", "olmo"),
    ("DataDecide-dolma1_7-1B-seed-default", "olmo"),
])
def test_family_for_base_repo(repo, tag):
    from subliminal_learning.coherence.probe_families import family_for_base_repo

    fam = family_for_base_repo(repo)
    assert fam is not None and fam.tag == tag


def test_family_for_base_repo_unknown_is_none():
    from subliminal_learning.coherence.probe_families import family_for_base_repo

    assert family_for_base_repo("Foo/bar") is None


def test_probe_pool_spec_for_repo_routes_families():
    from subliminal_learning.coherence import probe_io as pio

    for repo, tag in [("Qwen/Qwen2.5-1.5B", "qwen25"),
                      ("meta-llama/Llama-3.2-1B", "llama3"),
                      ("google/gemma-3-1b-pt", "gemma3")]:
        assert pio.probe_pool_spec_for_repo(repo).version == pool_version_for(tag)
    for repo in ("EleutherAI/pythia-160m", "local/pythia-160m-weightc-seed1@step25000"):
        assert pio.probe_pool_spec_for_repo(repo) is pio.PROBE_POOL_PP1
    for repo in ("allenai/DataDecide-dolma1_7-1B", "DataDecide-dolma1_7-1B-seed-default"):
        assert pio.probe_pool_spec_for_repo(repo) is pio.PROBE_POOL_DP1
    with pytest.raises(ValueError, match="Foo/bar"):
        pio.probe_pool_spec_for_repo("Foo/bar")


def test_merge_direct_returns_accepted_keys_only():
    from scripts.build_unigram_anchor import _merge_direct

    ce = {"a": 1.0}
    assert _merge_direct(ce, {"a": 2.0, "b": 3.0}, False) == {"b"}
    assert ce == {"a": 1.0, "b": 3.0}          # existing value kept, new one written
    ce = {"a": 1.0}
    assert _merge_direct(ce, {"a": 2.0}, True) == {"a"}
    assert ce == {"a": 2.0}


def test_require_same_pool_gate():
    from scripts.build_unigram_anchor import _require_same_pool

    _require_same_pool({"pool": {"token_sha": "aa"}}, "aa", "pp1")   # no raise
    with pytest.raises(SystemExit, match="bb"):
        _require_same_pool({"pool": {"token_sha": "aa"}}, "bb", "pp1")
    with pytest.raises(SystemExit):
        _require_same_pool({}, "aa", "pp1")


def test_require_pool_reproducible_gate(monkeypatch):
    import scripts.build_unigram_anchor as bua
    from subliminal_learning.coherence import probe_io as pio

    spec = probe_pool_spec_pp1t("qwen25")
    shas = iter(["aa", "bb"])

    class _Pool:
        def __init__(self, sha):
            self._sha = sha

        def token_sha256(self):
            return self._sha

    monkeypatch.setattr(pio, "build_probe_pool",
                        lambda tok, sp, **kw: _Pool(next(shas)))
    bua._require_pool_reproducible(FakeTok(), spec, "aa", "some/base")   # match
    with pytest.raises(SystemExit, match="some/base"):
        bua._require_pool_reproducible(FakeTok(), spec, "aa", "some/base")


def test_resolve_probe_pool_spec_defaults_to_the_family_and_refuses_foreign_pools():
    from subliminal_learning.coherence.probe_io import resolve_probe_pool_spec as r

    assert r("Qwen/Qwen2.5-1.5B").version == "pp1t-qwen25"
    assert r("Qwen/Qwen2.5-1.5B", "auto").version == "pp1t-qwen25"
    assert r("Qwen/Qwen2.5-1.5B", "pp1t-qwen25").version == "pp1t-qwen25"
    assert r("Qwen/Qwen2.5-1.5B", "pp1w-qwen25").version == "pp1w-qwen25"   # same family, wide
    assert r("EleutherAI/pythia-160m", "pp1").version == "pp1"
    assert r("EleutherAI/pythia-160m-data-seed1@step1000").version == "pp1"
    assert r("DataDecide-dolma1_7-1B-seed-default", "dp1").version == "dp1"
    assert r("allenai/DataDecide-dolma1_7-1B@step69369-seed-default", "pp1t-olmo").version == "pp1t-olmo"
    for repo, pool in (("Qwen/Qwen2.5-1.5B", "pp1"), ("Qwen/Qwen2.5-1.5B", "dp1"),
                       ("EleutherAI/pythia-160m", "dp1"), ("EleutherAI/pythia-160m", "pp1t-qwen25"),
                       ("Qwen/Qwen2.5-1.5B", "pp1t-nope")):
        with pytest.raises(ValueError, match="not a pool for base"):
            r(repo, pool)
    with pytest.raises(ValueError, match="no probe pool"):
        r("nobody/unknown-model", "pp1")
