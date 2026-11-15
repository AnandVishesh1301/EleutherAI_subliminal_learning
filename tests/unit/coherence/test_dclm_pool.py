"""cp1: Megatron index reading, pool selection semantics, freezing, and anchor stream.

Fully offline and source-free: a tiny synthetic Megatron `.bin`/`.idx` pair is written to
`tmp_path`, so no test touches the real (5 GB, machine-local) DCLM held-out dataset.
"""
import json
import struct

import numpy as np
import pytest

from subliminal_learning.coherence import dclm_pool as dc
from subliminal_learning.coherence import probe_io as pio


class _Tok:
    """Vocabulary-only stand-in: cp1 never tokenizes, it only checks id range."""

    def __init__(self, vocab=50_432):
        self._vocab = vocab

    def __len__(self):
        return self._vocab


def _write_megatron(tmp_path, docs, dtype=np.uint16):
    """Write `docs` (lists of token ids, EOD appended) as a Megatron indexed dataset."""
    stem = tmp_path / "heldout_text_document"
    sizes, ptrs, blob, off = [], [], bytearray(), 0
    for d in docs:
        ids = np.asarray(list(d) + [dc.EOD_TOKEN_ID], dtype=dtype)
        sizes.append(len(ids))
        ptrs.append(off)
        blob += ids.tobytes()
        off += ids.nbytes
    stem.with_suffix(".bin").write_bytes(bytes(blob))
    code = {np.uint16: 8, np.int32: 4}[dtype]
    header = (dc._MAGIC + struct.pack("<Q", 1) + struct.pack("<B", code)
              + struct.pack("<q", len(docs)) + struct.pack("<q", len(docs) + 1))
    idx = (header + np.asarray(sizes, dtype=np.int32).tobytes()
           + np.asarray(ptrs, dtype=np.int64).tobytes())
    stem.with_suffix(".idx").write_bytes(idx)
    return stem


def _subdir(tmp_path, name):
    """A fresh directory for a SECOND Megatron dataset beside the first."""
    d = tmp_path / name
    d.mkdir(exist_ok=True)
    return d


def _universe(n=40, seed=0):
    """Documents of varying length; docs 0-3 are deliberately under the 32-token floor."""
    rng = np.random.default_rng(seed)
    return [[int(t) for t in rng.integers(1, 5000, size=(8 if i < 4 else 40 + i))]
            for i in range(n)]


def test_megatron_documents_reads_spans_and_drops_the_trailing_eod(tmp_path):
    docs = _universe(5)
    md = dc.MegatronDocuments(_write_megatron(tmp_path, docs))
    assert len(md) == 5
    for i, d in enumerate(docs):
        assert md[i] == d                      # EOD dropped, span exact


def test_megatron_documents_rejects_a_non_megatron_index(tmp_path):
    stem = _write_megatron(tmp_path, _universe(3))
    stem.with_suffix(".idx").write_bytes(b"NOTANIDX\x00" + b"\x00" * 40)
    with pytest.raises(ValueError, match="not a Megatron .idx"):
        dc.MegatronDocuments(stem)


def test_heldout_stem_names_the_paths_key_when_unconfigured(monkeypatch):
    monkeypatch.setattr("subliminal_learning.config.load_paths_local", lambda: {})
    with pytest.raises(ValueError, match=dc.STEM_KEY):
        dc.heldout_stem()


def test_build_dclm_pool_follows_pp1_selection_semantics(tmp_path):
    docs = _universe(40)
    stem = _write_megatron(tmp_path, docs)
    pool = dc.build_dclm_pool(_Tok(), 8, max_tokens=50, seed=0, stem=stem,
                              use_frozen=False)
    order = [int(i) for i in np.random.default_rng(0).permutation(len(docs))]
    kept = [i for i in order if len(docs[i][:50]) >= 32][:8]
    assert pool.row_indices == kept                       # seed-0 permutation order
    assert [len(x) for x in pool.input_ids] == [min(len(docs[i]), 50) for i in kept]
    assert all(len(x) <= 50 for x in pool.input_ids)      # truncation
    assert all(el == list(range(1, len(ids)))             # position 0 never eligible
               for el, ids in zip(pool.eligible, pool.input_ids))
    assert pool.n_skipped == sum(1 for i in order[:order.index(kept[-1]) + 1]
                                 if len(docs[i]) < 32)
    assert pool.n_scanned == order.index(kept[-1]) + 1


def test_build_dclm_pool_raises_when_the_universe_is_too_small(tmp_path):
    stem = _write_megatron(tmp_path, _universe(6))
    with pytest.raises(ValueError, match="need 8"):
        dc.build_dclm_pool(_Tok(), 8, max_tokens=50, seed=0, stem=stem, use_frozen=False)


def test_build_dclm_pool_rejects_an_incompatible_vocabulary(tmp_path):
    stem = _write_megatron(tmp_path, _universe(40))
    with pytest.raises(ValueError, match="vocabulary"):
        dc.build_dclm_pool(_Tok(vocab=100), 8, max_tokens=50, seed=0, stem=stem,
                           use_frozen=False)


def _freeze(tmp_path, pool, **over):
    recs = [dict(order_pos=p, row=int(r), n_tokens=len(ids), ids=list(ids))
            for p, (r, ids) in enumerate(zip(pool.row_indices, pool.input_ids))]
    meta = dict(version="cp1", permutation_seed=0, max_tokens=50, min_tokens=32,
                universe_n_docs=40, n_docs=len(recs), n_scanned=int(pool.n_scanned),
                n_skipped=int(pool.n_skipped), token_sha256=pool.token_sha256())
    meta.update(over)
    p, mp = tmp_path / "cp1_tokens.jsonl", tmp_path / "cp1_tokens.meta.json"
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    mp.write_text(json.dumps(meta))
    return p, mp


def test_frozen_round_trip_reproduces_the_pool(tmp_path, monkeypatch):
    stem = _write_megatron(tmp_path, _universe(40))
    built = dc.build_dclm_pool(_Tok(), 8, max_tokens=50, seed=0, stem=stem,
                               use_frozen=False)
    p, mp = _freeze(tmp_path, built)
    monkeypatch.setattr(dc, "CP1_TOKENS_PATH", p)
    monkeypatch.setattr(dc, "CP1_TOKENS_META_PATH", mp)
    # no stem: the frozen artifact is what makes the pool buildable off the source node
    pool = dc.build_dclm_pool(_Tok(), 8, max_tokens=50, seed=0)
    assert pool.input_ids == built.input_ids
    assert pool.row_indices == built.row_indices
    assert pool.n_scanned == built.n_scanned
    assert pool.token_sha256() == built.token_sha256()


def test_frozen_sha_mismatch_raises(tmp_path):
    stem = _write_megatron(tmp_path, _universe(40))
    built = dc.build_dclm_pool(_Tok(), 8, max_tokens=50, seed=0, stem=stem,
                               use_frozen=False)
    p, mp = _freeze(tmp_path, built, )
    recs, meta = json.loads(p.read_text().splitlines()[0]), json.loads(mp.read_text())
    recs["ids"][0] += 1
    lines = p.read_text().splitlines()
    lines[0] = json.dumps(recs)
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="corrupt"):
        dc.pool_from_records([json.loads(ln) for ln in lines], meta)


def test_shipped_cp1_artifact_matches_the_pinned_sha():
    frozen = dc.frozen_records()
    assert frozen is not None, "data/probe_pools/cp1_tokens.jsonl is missing"
    recs, meta = frozen
    assert len(recs) == pio.PROBE_POOL_CP1.n_docs
    assert dc.pool_from_records(recs, meta).token_sha256() == pio.CP1_TOKEN_SHA


@pytest.mark.parametrize("bad,kwargs", [
    (dict(max_tokens=999), {}),
    (dict(permutation_seed=7), {}),
    (dict(min_tokens=99), {}),
])
def test_frozen_gate_rejects_a_recipe_mismatch(tmp_path, monkeypatch, bad, kwargs):
    """A frozen artifact cut under another recipe is ignored: the source is rebuilt."""
    stem = _write_megatron(tmp_path, _universe(40))
    built = dc.build_dclm_pool(_Tok(), 8, max_tokens=50, seed=0, stem=stem,
                               use_frozen=False)
    p, mp = _freeze(tmp_path, built, **bad)
    monkeypatch.setattr(dc, "CP1_TOKENS_PATH", p)
    monkeypatch.setattr(dc, "CP1_TOKENS_META_PATH", mp)
    rebuilt, real = [], dc.MegatronDocuments
    monkeypatch.setattr(dc, "MegatronDocuments",
                        lambda s=None: (rebuilt.append(s), real(stem))[1])
    pool = dc.build_dclm_pool(_Tok(), 8, max_tokens=50, seed=0, **kwargs)
    assert rebuilt, "the recipe mismatch should have fallen back to the source"
    assert pool.input_ids == built.input_ids


def test_verification_passes_when_the_live_source_matches(tmp_path):
    stem = _write_megatron(tmp_path, _universe(40))
    built = dc.build_dclm_pool(_Tok(), 8, max_tokens=50, seed=0, stem=stem,
                               use_frozen=False)
    p, mp = _freeze(tmp_path, built)
    recs, meta = dc.frozen_records(p, mp)
    dc.verify_frozen_against_source(dc.MegatronDocuments(stem), recs, meta)


def test_verification_fails_on_a_doc_count_mismatch(tmp_path):
    stem = _write_megatron(tmp_path, _universe(40))
    built = dc.build_dclm_pool(_Tok(), 8, max_tokens=50, seed=0, stem=stem,
                               use_frozen=False)
    recs, meta = dc.frozen_records(*_freeze(tmp_path, built))
    bigger = _write_megatron(_subdir(tmp_path, "big"), _universe(41))
    with pytest.raises(ValueError, match="41 documents"):
        dc.verify_frozen_against_source(dc.MegatronDocuments(bigger), recs, meta)


def test_verification_fails_on_a_same_size_reordered_source(tmp_path):
    """Same length, same sha-able content, different order: the permutation moves."""
    docs = _universe(40)
    stem = _write_megatron(tmp_path, docs)
    built = dc.build_dclm_pool(_Tok(), 8, max_tokens=50, seed=0, stem=stem,
                               use_frozen=False)
    recs, meta = dc.frozen_records(*_freeze(tmp_path, built))
    swapped = list(docs)
    a, b = [int(i) for i in dc.universe_permutation(len(docs), 0)[:2]]
    swapped[a], swapped[b] = swapped[b], swapped[a]
    other = _write_megatron(_subdir(tmp_path, "swap"), swapped)
    with pytest.raises(ValueError, match="does not reproduce"):
        dc.verify_frozen_against_source(dc.MegatronDocuments(other), recs, meta)


def test_anchor_stream_starts_after_the_pool_scan_window(tmp_path):
    docs = _universe(40)
    stem = _write_megatron(tmp_path, docs)
    md = dc.MegatronDocuments(stem)
    pool = dc.build_dclm_pool(_Tok(), 8, max_tokens=50, seed=0, stem=stem,
                              use_frozen=False)
    order = [int(i) for i in dc.universe_permutation(len(docs), 0)]
    fit = list(dc.iter_universe_token_ids(md, 0, pool.n_scanned))
    assert fit == [docs[i] for i in order[pool.n_scanned:]]
    assert not any(d in pool.input_ids for d in fit)    # disjoint from the pool


def test_shipped_cp1_text_records_match_the_frozen_cp1_tokens():
    """The frozen cp1 TEXT is the frozen cp1 TOKENS, decoded: same docs, order and rows."""
    recs, meta = pio.load_frozen_text(pio.CP1_TEXT_PATH, pio.CP1_TEXT_META_PATH)
    token_recs, token_meta = dc.frozen_records()
    assert [r["row"] for r in recs] == [r["row"] for r in token_recs]
    assert [r["order_pos"] for r in recs] == list(range(len(token_recs)))
    assert [r["pythia_n_tokens"] for r in recs] == [len(r["ids"]) for r in token_recs]
    assert meta["cp1_token_sha"] == token_meta["token_sha256"]
    assert (meta["source_stem"], meta["universe_n_docs"], meta["n_scanned"]) == (
        token_meta["source_stem"], token_meta["universe_n_docs"], token_meta["n_scanned"])
    assert meta["decode_tokenizer"] == "EleutherAI/pythia-160m"
