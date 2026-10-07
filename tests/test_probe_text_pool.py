"""Cross-tokenizer probe pools (`pp1t-<tag>`): frozen text artifact, families, anchor math."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pytest

from subliminal_learning.cka.pool_io import SequencePool
from subliminal_learning.forgetting.coherence.anchor import (
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


def test_probe_pool_spec_for_repo_routes_the_dclm_corpus_to_cp1():
    """The PRETRAINING CORPUS wins over the tokenizer family inside Pythia."""
    from subliminal_learning.coherence import probe_io as pio

    assert pio.probe_pool_spec_for_repo("local/pythia-160m-dclm-seed1") is pio.PROBE_POOL_CP1
    assert pio.probe_pool_spec_for_repo(
        "local/pythia-160m-dclm-seed1234@step10000") is pio.PROBE_POOL_CP1
    for repo in ("local/pythia-160m-weightc-seed1", "local/pythia-160m-pilerepro-seed3",
                 "EleutherAI/pythia-160m"):
        assert pio.probe_pool_spec_for_repo(repo) is pio.PROBE_POOL_PP1


def test_resolve_probe_pool_spec_defaults_to_cp1_for_a_dclm_base():
    """cp1 is the dclm re-pretrainings' NATIVE pool; pp1 stays the default elsewhere.

    Both are Pythia-tokenizer corpus pools, so either is an explicit same-family
    cross-corpus sidecar on any Pythia base (`compatible_probe_pools`) -- exactly as dp1
    and cp1t-olmo are on either DataDecide recipe. Another family's tokens are refused.
    """
    from subliminal_learning.coherence.probe_io import resolve_probe_pool_spec as r

    assert r("local/pythia-160m-dclm-seed1", "cp1").version == "cp1"
    assert r("local/pythia-160m-dclm-seed1").version == "cp1"
    assert r("EleutherAI/pythia-160m").version == "pp1"
    assert r("EleutherAI/pythia-160m", "cp1").version == "cp1"
    assert r("local/pythia-160m-dclm-seed1", "pp1").version == "pp1"
    with pytest.raises(ValueError, match="not a pool for base"):
        r("Qwen/Qwen2.5-1.5B", "cp1")
    with pytest.raises(ValueError, match="not a pool for base"):
        r("EleutherAI/pythia-160m", "dp1")


def test_native_probe_pool_reads_every_base_spelling():
    """Token, hub ref, output-tree dir name and a BaseSpec all give the same pool."""
    from subliminal_learning.coherence.probe_io import native_pool_version_for_base as v
    from subliminal_learning.models.naming import base_spec

    dolma = ("dd-1b", "DataDecide-dolma1_7-1B-seed-default",
             "allenai/DataDecide-dolma1_7-1B@step69369-seed-default",
             "DataDecide-dolma1_7-1B-seed-default@step69369")
    dclm = ("dd-1b-dclm", "dd-1b-dclm-seed2", "DataDecide-dclm-baseline-1B-seed-default",
            "allenai/DataDecide-dclm-baseline-1B@step69369-seed-default",
            "DataDecide-dclm-baseline-1B-seed-default@step69369")
    for spelling in dolma:
        assert v(spelling) == "dp1"
        assert v(base_spec(spelling)) == "dp1"
    c4 = ("dd-1b-c4", "dd-1b-c4-seed2", "DataDecide-c4-1B-seed-default",
          "allenai/DataDecide-c4-1B@step69369-seed-default",
          "DataDecide-c4-1B-seed-default@step69369")
    for spelling in dclm:
        assert v(spelling) == "cp1t-olmo"
        assert v(base_spec(spelling)) == "cp1t-olmo"
    for spelling in c4:
        assert v(spelling) == "c4p1"
        assert v(base_spec(spelling)) == "c4p1"
    assert v("pythia-160m") == v("EleutherAI/pythia-160m") == "pp1"
    assert v("local/pythia-160m-dclm-seed1234") == "cp1"
    assert v("qwen25-1.5b") == v("Qwen/Qwen2.5-1.5B") == "pp1t-qwen25"


def test_compatibility_matrix_accepts_same_family_and_rejects_the_rest():
    from subliminal_learning.coherence.probe_io import (
        IncompatibleProbePoolError, UnknownProbeBaseError, compatible_probe_pools,
        require_compatible_probe_pool,
    )

    assert compatible_probe_pools("dd-1b")[0] == "dp1"
    assert compatible_probe_pools("dd-1b-dclm")[0] == "cp1t-olmo"
    assert compatible_probe_pools("dd-1b-c4")[0] == "c4p1"
    assert set(compatible_probe_pools("dd-1b")) == {"dp1", "cp1t-olmo", "c4p1",
                                                    "pp1t-olmo", "pp1w-olmo"}
    # every olmo corpus pool is listed exactly once, native first
    pools = compatible_probe_pools("dd-1b-c4")
    assert len(pools) == len(set(pools)) and pools.count("c4p1") == 1
    assert compatible_probe_pools("qwen25-1.5b")[0] == "pp1t-qwen25"
    # the other DataDecide recipe's corpus pool is a legitimate sidecar either way
    assert require_compatible_probe_pool("dd-1b", "cp1t-olmo").version == "cp1t-olmo"
    assert require_compatible_probe_pool("dd-1b-dclm", "dp1").version == "dp1"
    assert require_compatible_probe_pool("dd-1b", "c4p1").version == "c4p1"
    for base, pool in (("dd-1b", "pp1"), ("dd-1b-dclm", "cp1"), ("pythia-160m", "dp1"),
                       ("pythia-160m", "c4p1"),
                       ("pythia-160m", "cp1t-olmo"), ("qwen25-1.5b", "pp1")):
        with pytest.raises(IncompatibleProbePoolError):
            require_compatible_probe_pool(base, pool)
    with pytest.raises(UnknownProbeBaseError):
        compatible_probe_pools("not-a-base-at-all")


def test_merge_direct_returns_accepted_keys_only():
    from forgetting.scripts.build_unigram_anchor import _merge_direct

    ce = {"a": 1.0}
    assert _merge_direct(ce, {"a": 2.0, "b": 3.0}, False) == {"b"}
    assert ce == {"a": 1.0, "b": 3.0}          # existing value kept, new one written
    ce = {"a": 1.0}
    assert _merge_direct(ce, {"a": 2.0}, True) == {"a"}
    assert ce == {"a": 2.0}


def test_require_same_pool_gate():
    from forgetting.scripts.build_unigram_anchor import _require_same_pool

    _require_same_pool({"pool": {"token_sha": "aa"}}, "aa", "pp1")   # no raise
    with pytest.raises(SystemExit, match="bb"):
        _require_same_pool({"pool": {"token_sha": "aa"}}, "bb", "pp1")
    with pytest.raises(SystemExit):
        _require_same_pool({}, "aa", "pp1")


def test_require_pool_reproducible_gate(monkeypatch):
    import forgetting.scripts.build_unigram_anchor as bua
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

def test_probe_pool_spec_for_repo_rejects_an_unparseable_pythia_repo():
    """A pythia-prefixed name the naming module cannot spell raises instead of pp1."""
    from subliminal_learning.coherence import probe_io as pio

    with pytest.raises(ValueError, match="pythia-160m-dclm-seed7"):
        pio.probe_pool_spec_for_repo("local/pythia-160m-dclm-seed7")


# --- cp1t: the SECOND frozen-text corpus (DCLM held-out) and the recipe routing --------

def _write_cp1_artifact(tmp_path, texts):
    """A cp1 frozen-text artifact: same schema as pp1's, `base_pool` says which corpus."""
    recs = [dict(order_pos=i, row=200 + i, text=t, n_chars=len(t),
                 n_bytes=len(t.encode()), pythia_n_tokens=len(t), round_trip_exact=True)
            for i, t in enumerate(texts)]
    h = hashlib.sha256()
    for r in recs:
        h.update(json.dumps(r["text"]).encode())
    p, mp = tmp_path / "cp1_text.jsonl", tmp_path / "cp1_text.meta.json"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    mp.write_text(json.dumps(dict(version="cp1t", base_pool="cp1", n_docs=len(recs),
                                  n_scanned=len(recs), text_sha256=h.hexdigest(),
                                  text_bytes_total=99, n_round_trip_exact=len(recs))))
    return p, mp


def test_cp1t_pool_versions_and_family_resolution():
    from subliminal_learning.coherence.probe_families import (
        cp1t_pool_version_for, is_cp1t, is_frozen_text_pool,
    )

    assert cp1t_pool_version_for("olmo") == "cp1t-olmo"
    assert is_cp1t("cp1t-olmo") and not is_cp1t("cp1") and not is_cp1t("pp1t-olmo")
    assert is_frozen_text_pool("cp1t-olmo") and not is_frozen_text_pool("cp1w-olmo")
    assert family_for_pool_version("cp1t-olmo").tag == "olmo"
    for bad in ("cp1t-nope", "cp1w-olmo", "cp1"):
        with pytest.raises(ValueError):
            family_for_pool_version(bad)
    with pytest.raises(ValueError, match="nope"):
        cp1t_pool_version_for("nope")


def test_build_probe_pool_routes_cp1t_to_the_cp1_text(tmp_path, monkeypatch):
    """A cp1t pool reads the cp1 artifact, and its source names that corpus."""
    from subliminal_learning.coherence.probe_io import probe_pool_spec_cp1t
    import subliminal_learning.coherence.probe_io as probe_io

    p, mp = _write_cp1_artifact(tmp_path, ["abcdefghij" * 4, "klmnopqrst" * 5])
    monkeypatch.setattr(probe_io, "CP1_TEXT_PATH", p)
    monkeypatch.setattr(probe_io, "CP1_TEXT_META_PATH", mp)
    monkeypatch.setattr(probe_io, "PP1_TEXT_PATH",
                        tmp_path / "must-not-be-read.jsonl")
    pool = build_probe_pool(FakeTok(), probe_pool_spec_cp1t("olmo"), verify_pin=False)
    assert [len(ids) for ids in pool.input_ids] == [40, 50]
    assert pool.source.startswith("cp1_text:") and pool.row_indices == [200, 201]


def test_shipped_cp1_text_is_a_32_doc_artifact_of_the_pinned_cp1_pool():
    from subliminal_learning.coherence.probe_io import (
        CP1_TEXT_META_PATH, CP1_TEXT_PATH, CP1_TOKEN_SHA, load_frozen_text,
        pp1_text_sha256,
    )

    recs, meta = load_frozen_text(CP1_TEXT_PATH, CP1_TEXT_META_PATH)  # verifies text_sha256
    assert len(recs) == meta["n_docs"] == 32
    assert meta["text_sha256"] == pp1_text_sha256(recs)
    assert meta["cp1_token_sha"] == CP1_TOKEN_SHA and meta["base_pool"] == "cp1"
    assert meta["n_round_trip_exact"] == 32            # every doc re-encodes to cp1's ids
    assert meta["pythia_n_tokens"] == [r["pythia_n_tokens"] for r in recs]
    assert all(r["pythia_n_tokens"] <= meta["max_tokens"] for r in recs)
    assert meta["text_bytes_total"] == sum(r["n_bytes"] for r in recs)


@pytest.mark.parametrize("repo,version", [
    ("dd-1b", "dp1"),
    ("allenai/DataDecide-dolma1_7-1B@step69369-seed-default", "dp1"),
    ("DataDecide-dolma1_7-1B-seed-default", "dp1"),
    ("dd-1b-dclm", "cp1t-olmo"),
    ("dd-1b-dclm-seed2", "cp1t-olmo"),
    ("dd-1b-dclm@step10000", "cp1t-olmo"),
    ("allenai/DataDecide-dclm-baseline-1B@step69369-seed-default", "cp1t-olmo"),
    ("allenai/DataDecide-dclm-baseline-1B@step10000-seed-large-aux-3", "cp1t-olmo"),
    ("DataDecide-dclm-baseline-1B-seed-default", "cp1t-olmo"),
])
def test_datadecide_recipe_decides_the_corpus_pool(repo, version):
    from subliminal_learning.coherence import probe_io as pio

    assert pio.probe_pool_spec_for_repo(repo).version == version


def test_either_datadecide_recipe_pool_is_an_accepted_explicit_pool():
    """The olmo family owns dp1 AND cp1t-olmo, so either is a legal sidecar readout."""
    from subliminal_learning.coherence.probe_io import resolve_probe_pool_spec as r

    dolma = "DataDecide-dolma1_7-1B-seed-default"
    dclm = "DataDecide-dclm-baseline-1B-seed-default"
    assert r(dolma, "cp1t-olmo").version == "cp1t-olmo"
    assert r(dclm, "dp1").version == "dp1"
    assert r(dclm).version == "cp1t-olmo" and r(dclm, "auto").version == "cp1t-olmo"
    assert r(dclm, "pp1t-olmo").version == "pp1t-olmo"       # same family, Pile text
    for pool in ("cp1", "pp1", "cp1t-nope", "cp1w-olmo", "cp1t-qwen25"):
        with pytest.raises(ValueError, match="not a pool for base"):
            r(dclm, pool)
    for repo in ("EleutherAI/pythia-160m", "local/pythia-160m-dclm-seed1",
                 "Qwen/Qwen2.5-1.5B"):
        with pytest.raises(ValueError, match="not a pool for base"):
            r(repo, "cp1t-olmo")


def test_cp1t_olmo_retokenizes_the_frozen_cp1_text_to_cp1s_pinned_sha():
    """The OLMo tokenizer on `data/probe_pools/cp1_text.jsonl` reproduces `CP1_TOKEN_SHA`.

    This is what "identical ON THE POOL" means, and it is why `CP1T_TOKEN_SHA["olmo"]`
    IS `CP1_TOKEN_SHA`: the two tokenizers agree on these 32 documents (measured), not in
    general. Needs the pinned DataDecide tokenizer from the local HF cache and the
    `datadecide` dependency group; skipped when either is absent.
    """
    from subliminal_learning.coherence.probe_families import PROBE_FAMILIES
    from subliminal_learning.coherence.probe_io import (
        CP1_TOKEN_SHA, CP1T_TOKEN_SHA, build_probe_pool, probe_pool_spec_cp1t,
    )

    try:
        from subliminal_learning.coherence.probe_families import load_family_tokenizer
        from subliminal_learning.config import resolve_roots

        tok, _ = load_family_tokenizer(PROBE_FAMILIES["olmo"], resolve_roots()[1])
    except Exception as exc:                                    # noqa: BLE001
        pytest.skip(f"OLMo tokenizer unavailable offline: {exc}")
    pool = build_probe_pool(tok, probe_pool_spec_cp1t("olmo"))   # verifies the pin too
    assert pool.token_sha256() == CP1_TOKEN_SHA == CP1T_TOKEN_SHA["olmo"]
    assert len(pool.input_ids) == 32


# ---------------------------------------- no silent fall-through to the Pile artifact


def test_frozen_text_lookups_reject_corpus_pool_versions():
    """`c4p1`/`dp1` are CORPUS pools: asking for their frozen text is a bug, not pp1."""
    from subliminal_learning.coherence.probe_io import (
        frozen_text_paths, pinned_frozen_text_sha,
    )

    for version in ("c4p1", "dp1"):
        with pytest.raises(ValueError, match="pp1t-<tag>"):
            frozen_text_paths(version)
        with pytest.raises(ValueError, match="pp1t-<tag>"):
            pinned_frozen_text_sha(version)


def test_frozen_text_lookups_unchanged_for_the_frozen_text_families():
    from subliminal_learning.coherence import probe_io as pio

    assert pio.frozen_text_paths("pp1t-olmo") == (pio.PP1_TEXT_PATH, pio.PP1_TEXT_META_PATH)
    assert pio.frozen_text_paths("pp1w-olmo") == (pio.PP1W_TEXT_PATH,
                                                  pio.PP1W_TEXT_META_PATH)
    assert pio.frozen_text_paths("cp1t-olmo") == (pio.CP1_TEXT_PATH, pio.CP1_TEXT_META_PATH)
    assert pio.pinned_frozen_text_sha("pp1t-olmo") == pio.PP1T_TOKEN_SHA["olmo"]
    assert pio.pinned_frozen_text_sha("pp1w-olmo") == pio.PP1W_TOKEN_SHA["olmo"]
    assert pio.pinned_frozen_text_sha("cp1t-olmo") == pio.CP1T_TOKEN_SHA.get("olmo")
