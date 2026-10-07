"""Nested WIDE probe pools (`pp1w-<tag>`, `dp1w`): prefixes, builders, per-doc CE, gates.

The wide pools are 128 documents whose first 32 ARE the base pool's, with the extension
drawn from after the frozen unigram fit window. Everything here is offline: fake
tokenizers, fake frozen-text artifacts, and an injected Dolma universe.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from subliminal_learning.cka.pool_io import SequencePool
from subliminal_learning.coherence import dolma_pool as dp
from subliminal_learning.coherence import probe_io as pio
from subliminal_learning.coherence.probe_families import (
    family_for_pool_version, is_frozen_text_pool, is_pp1t, is_pp1w, wide_pool_version_for,
)

from tests.test_probe_text_pool import FakeTok


# ---------------------------------------------------------------- prefixes / specs


def test_wide_prefix_helpers():
    assert wide_pool_version_for("qwen25") == "pp1w-qwen25"
    assert is_pp1w("pp1w-qwen25") and not is_pp1w("pp1t-qwen25")
    assert is_pp1t("pp1t-qwen25") and not is_pp1t("pp1w-qwen25")   # pp1t stays NARROW
    assert is_frozen_text_pool("pp1w-qwen25") and is_frozen_text_pool("pp1t-qwen25")
    assert not is_frozen_text_pool("dp1") and not is_frozen_text_pool("dp1w")
    assert family_for_pool_version("pp1w-qwen25").tag == "qwen25"
    with pytest.raises(ValueError, match="qwen25"):
        wide_pool_version_for("nope")
    with pytest.raises(ValueError, match="qwen25"):
        family_for_pool_version("pp1w-nope")


def test_wide_specs_and_pinned_shas():
    spec = pio.probe_pool_spec_pp1w("llama3")
    assert (spec.n_docs, spec.max_tokens, spec.seed, spec.version) == (128, 512, 0,
                                                                      "pp1w-llama3")
    assert (pio.PROBE_POOL_DP1W.n_docs, pio.PROBE_POOL_DP1W.version) == (128, "dp1w")
    assert set(pio.PP1W_TOKEN_SHA) == {"pythia", "olmo", "qwen25", "llama3", "gemma3", "gemma2"}
    # pinned 2026-09-05 from the first real builds: every family and dp1w carry a 64-hex
    # sha; pythia and olmo share one (token-identical tokenizers on this text)
    for tag, sha in pio.PP1W_TOKEN_SHA.items():
        assert isinstance(sha, str) and len(sha) == 64, tag
    assert pio.PP1W_TOKEN_SHA["pythia"] == pio.PP1W_TOKEN_SHA["olmo"]
    assert isinstance(pio.DP1W_TOKEN_SHA, str) and len(pio.DP1W_TOKEN_SHA) == 64
    # the base pools are untouched
    assert pio.PROBE_POOL_PP1.n_docs == 32 and pio.PROBE_POOL_DP1.n_docs == 32


# ---------------------------------------------------------------- frozen wide text


def _artifact(tmp_path, texts, *, name, n_scanned, extra_meta=None):
    recs = [dict(order_pos=i, row=100 + i, text=t, n_chars=len(t), n_bytes=len(t.encode()),
                 pythia_n_tokens=len(t), round_trip_exact=True)
            for i, t in enumerate(texts)]
    h = hashlib.sha256()
    for r in recs:
        h.update(json.dumps(r["text"]).encode())
    p = tmp_path / f"{name}.jsonl"
    mp = tmp_path / f"{name}.meta.json"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    mp.write_text(json.dumps(dict(n_docs=len(recs), n_scanned=n_scanned,
                                  text_sha256=h.hexdigest(), text_bytes_total=1234,
                                  n_round_trip_exact=len(recs), **(extra_meta or {}))))
    return p, mp


def _texts(n, offset=0):
    return [f"doc{i + offset:04d}" + "abcdefghij" * 4 for i in range(n)]


def test_wide_pool_nests_the_base_pool(tmp_path):
    base_texts = _texts(32)
    bp, bmp = _artifact(tmp_path, base_texts, name="pp1_text", n_scanned=33)
    wp, wmp = _artifact(tmp_path, base_texts + _texts(96, offset=1000), name="pp1w_text",
                        n_scanned=33)

    base = pio.build_frozen_text_pool(FakeTok(), pio.probe_pool_spec_pp1t("qwen25"),
                                      bos_token_id=None, path=bp, meta_path=bmp)
    wide = pio.build_frozen_text_pool(FakeTok(), pio.probe_pool_spec_pp1w("qwen25"),
                                      bos_token_id=None, path=wp, meta_path=wmp)
    assert len(wide) == 128 and len(base) == 32
    assert wide.input_ids[:32] == base.input_ids
    assert wide.eligible[:32] == base.eligible
    assert wide.n_scanned == 33                      # the BASE pool's value, from the meta
    assert wide.row_indices[:32] == base.row_indices

    withbos = pio.build_frozen_text_pool(FakeTok(), pio.probe_pool_spec_pp1w("llama3"),
                                         bos_token_id=3, path=wp, meta_path=wmp)
    assert all(ids[0] == 3 for ids in withbos.input_ids)


def test_build_probe_pool_routes_pp1w_to_the_wide_artifact(tmp_path, monkeypatch):
    bp, bmp = _artifact(tmp_path, _texts(32), name="pp1_text", n_scanned=33)
    wp, wmp = _artifact(tmp_path, _texts(32) + _texts(96, offset=1000), name="pp1w_text",
                        n_scanned=33)
    monkeypatch.setattr(pio, "PP1_TEXT_PATH", bp)
    monkeypatch.setattr(pio, "PP1_TEXT_META_PATH", bmp)
    monkeypatch.setattr(pio, "PP1W_TEXT_PATH", wp)
    monkeypatch.setattr(pio, "PP1W_TEXT_META_PATH", wmp)
    # synthetic tokenizer: the pinned-sha check cannot apply
    assert len(pio.build_probe_pool(FakeTok(), pio.probe_pool_spec_pp1w("qwen25"),
                                    verify_pin=False)) == 128
    assert len(pio.build_probe_pool(FakeTok(), pio.probe_pool_spec_pp1t("qwen25"),
                                    verify_pin=False)) == 32


# ---------------------------------------------------------------- per-document CE


class _FakePool:
    def __init__(self, input_ids):
        self.input_ids = input_ids
        self.eligible = [list(range(1, len(ids))) for ids in input_ids]


def test_doc_ce_agrees_with_mean_ce_and_subset_mean():
    import torch

    torch.manual_seed(0)
    pool = _FakePool([[1, 2, 3, 4, 5], [6, 7, 8], [2, 3, 4, 5, 6, 7]])
    logp = [torch.log_softmax(torch.randn(len(ids) - 1, 10), dim=-1)
            for ids in pool.input_ids]
    sums, counts = pio.doc_ce(logp, pool)
    assert counts == [3, 1, 4]
    assert sum(sums) / sum(counts) == pytest.approx(pio.mean_ce(logp, pool), abs=1e-9)
    assert pio.subset_mean(sums, counts, 3) == pytest.approx(pio.mean_ce(logp, pool),
                                                             abs=1e-9)


def test_subset_mean_is_token_weighted():
    assert pio.subset_mean([6.0, 10.0], [3, 2], 1) == pytest.approx(2.0)
    assert pio.subset_mean([6.0, 10.0], [3, 2], 2) == pytest.approx(16.0 / 5)
    assert pio.subset_mean([], [], 0) != pio.subset_mean([], [], 0)   # nan


# ---------------------------------------------------------------- Dolma wide builder


class _Tok:
    """Whitespace tokenizer: one id per word, deterministic across runs."""

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [self._ids(t) for t in text]}
        return {"input_ids": self._ids(text)}

    @staticmethod
    def _ids(text):
        return [abs(hash(w)) % 1000 for w in text.split()]


def _docs(n=200):
    return [{"text": f"doc{i} " + "word " * 60, "source": "wiki"} for i in range(n)]


def test_dolma_wide_pool_nests_the_base_and_starts_after_the_window():
    tok, docs = _Tok(), _docs(200)
    base = dp.build_dolma_pool(tok, 8, docs=docs)
    wide = dp.build_dolma_pool(tok, 12, docs=docs, extension_start=100, n_base=8)
    assert wide.input_ids[:8] == base.input_ids
    assert wide.n_scanned == base.n_scanned          # the fit window is unchanged
    assert wide.extension_n_scanned == 4             # 4 extension docs, none skipped
    expected = [dp.build_dolma_pool(tok, 1, docs=[{"text": t}])
                for t in list(dp.iter_universe_texts(docs, 0, 100))[:4]]
    assert wide.input_ids[8:] == [p.input_ids[0] for p in expected]
    assert base.extension_n_scanned is None


def test_dolma_wide_requires_both_arguments():
    with pytest.raises(ValueError, match="together"):
        dp.build_dolma_pool(_Tok(), 12, docs=_docs(), extension_start=100)


def test_sequence_pool_positional_construction_still_works():
    pool = SequencePool([[1, 2, 3]], [[1, 2]], "x", 5, 1, [7])
    assert pool.extension_n_scanned is None and pool.row_indices == [7]


def test_build_probe_pool_routes_dp1w(monkeypatch):
    seen = {}

    def _fake(tok, n, **kw):
        seen.update(n=n, **kw)
        return "pool"

    monkeypatch.setattr(dp, "build_dolma_pool", _fake)
    assert pio.build_probe_pool(_Tok(), pio.PROBE_POOL_DP1W) == "pool"
    assert seen["n"] == 128 and seen["extension_start"] == pio.DP1W_EXTENSION_START
    assert seen["n_base"] == 32
    pio.build_probe_pool(_Tok(), pio.PROBE_POOL_DP1)
    assert seen["extension_start"] is None and seen["n_base"] is None


# ---------------------------------------------------------------- anchor gates


def test_require_base_subset_gate():
    from forgetting.scripts.build_unigram_anchor import _require_base_subset

    base = SequencePool([[1, 2, 3], [4, 5, 6]], [[1, 2], [1, 2]], "b")
    wide = SequencePool([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[1, 2]] * 3, "w")
    _require_base_subset(wide, base, "pp1w-qwen25", "pp1t-qwen25")     # no raise
    bad = SequencePool([[1, 2, 3], [4, 5, 7], [7, 8, 9]], [[1, 2]] * 3, "w")
    with pytest.raises(SystemExit, match="doc 1 is not token-identical"):
        _require_base_subset(bad, base, "pp1w-qwen25", "pp1t-qwen25")
    with pytest.raises(SystemExit, match="not more than"):
        _require_base_subset(base, base, "pp1w-qwen25", "pp1t-qwen25")


def test_counts_vs_base_h_uni_gate():
    import numpy as np

    from forgetting.scripts.build_unigram_anchor import PRIMARY_ALPHA, _require_counts_match_base
    from subliminal_learning.forgetting.coherence.anchor import unigram_ce

    counts = np.array([5, 3, 1, 0])
    targets = np.array([0, 1, 2, 3, 0])
    h = unigram_ce(counts, targets, PRIMARY_ALPHA)
    assert _require_counts_match_base(counts, targets, h, "pp1t-qwen25",
                                      PRIMARY_ALPHA) == pytest.approx(h)
    with pytest.raises(SystemExit, match="do not reproduce"):
        _require_counts_match_base(counts, targets, h + 1e-6, "pp1t-qwen25", PRIMARY_ALPHA)


def test_load_base_counts_demands_the_base_anchor(tmp_path):
    from forgetting.scripts.build_unigram_anchor import _load_base_counts

    with pytest.raises(SystemExit, match="--pool dp1"):
        _load_base_counts(tmp_path, "dp1")


# ---------------------------------------------------------------- freeze --wide


def test_wide_artifact_assembly():
    from forgetting.scripts.freeze_pp1_text import wide_artifact

    base_recs = [dict(order_pos=i, row=i, text=f"t{i}", n_chars=2, n_bytes=2,
                      pythia_n_tokens=2, round_trip_exact=True) for i in range(3)]
    base_meta = dict(dataset="monology/pile-uncopyrighted", split="validation",
                     permutation_seed=0, n_scanned=4, text_sha256="base-sha",
                     text_bytes_total=6)
    ext = [dict(order_pos=-1, row=50 + j, perm_pos=900 + j, text=f"e{j}", n_chars=2,
                n_bytes=2, pythia_n_tokens=2, round_trip_exact=j == 0) for j in range(2)]

    recs, meta = wide_artifact(base_recs, base_meta, ext, extension_start=900,
                               extension_n_scanned=7, extension_source="anchor")
    assert [r["order_pos"] for r in recs] == [0, 1, 2, 3, 4]
    assert recs[:3] == base_recs                       # base records verbatim
    assert [r["perm_pos"] for r in recs[3:]] == [900, 901]
    assert (meta["version"], meta["base_pool"]) == ("pp1w", "pp1")
    assert (meta["n_base_docs"], meta["n_docs"]) == (3, 5)
    assert meta["base_text_sha256"] == "base-sha"
    assert meta["n_scanned"] == 4                      # the BASE pool's scan count
    assert (meta["extension_start"], meta["extension_n_scanned"]) == (900, 7)
    assert meta["extension_source"] == "anchor"
    assert meta["text_bytes_total"] == 10 and meta["n_round_trip_exact"] == 4
    assert meta["text_sha256"] != "base-sha" and meta["created"]

    with pytest.raises(SystemExit, match="out of order"):
        wide_artifact(list(reversed(base_recs)), base_meta, ext, extension_start=900,
                      extension_n_scanned=7, extension_source="anchor")


def test_extension_start_reads_the_frozen_fit_window(tmp_path):
    from forgetting.scripts.freeze_pp1_text import _extension_start

    d = tmp_path / "analysis"
    d.mkdir()
    (d / "unigram_anchor_pp1.json").write_text(
        json.dumps({"unigram": {"permutation_offset": 33, "n_docs_scanned": 53248}}))
    assert _extension_start(tmp_path) == (53281, "unigram_anchor_pp1.json")
    with pytest.raises(SystemExit, match="--extension-start"):
        _extension_start(tmp_path / "elsewhere")


def test_wide_anchor_fit_window_is_disjoint():
    from forgetting.scripts.build_unigram_anchor import _check_fit_window_disjoint

    _check_fit_window_disjoint(33, 53248, 53281)
    with pytest.raises(SystemExit, match="overlaps wide-pool extension"):
        _check_fit_window_disjoint(33, 53249, 53281)
