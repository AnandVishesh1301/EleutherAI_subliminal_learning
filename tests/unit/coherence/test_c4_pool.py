"""c4p1: C4 en validation universe cache, pool semantics, and recipe routing.

Fully offline — the shard fetcher is injected, so no test touches huggingface.co.
"""
import gzip
import hashlib
import io
import json

import pytest

from subliminal_learning.coherence import c4_pool as cp
from subliminal_learning.coherence import probe_io as pio
from subliminal_learning.utils.io import file_sha256

# ---------------------------------------------------------------- fakes


class _Tok:
    """Whitespace tokenizer: one id per word, stable across runs.

    Ids come from sha1, NOT the builtin `hash()`: string hashing is salted per
    process, so a `hash()`-based id would differ between runs (and between the two
    halves of a test comparing recorded shas).
    """

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [self._ids(t) for t in text]}
        return {"input_ids": self._ids(text)}

    @staticmethod
    def _ids(text):
        return [int.from_bytes(hashlib.sha1(w.encode()).digest()[:4], "big") % 1000
                for w in text.split()]


def _shard_bytes(records):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for r in records:
            gz.write((json.dumps(r) + "\n").encode())
    return buf.getvalue()


def _fake_remote(n_per_shard=200):
    """A fake hub: one gz shard per validation URL. Returns (fetch, calls)."""
    blobs = {
        u: _shard_bytes([{"text": f"shard{i} doc {j} " + "word " * 40}
                         for j in range(n_per_shard)])
        for i, u in enumerate(cp.shard_urls())
    }
    calls: list[str] = []

    def fetch(url):
        calls.append(url)
        return io.BytesIO(blobs[url])

    return fetch, calls


# ---------------------------------------------------------------- universe cache


def test_universe_cache_hit_means_zero_fetches(tmp_path):
    fetch, calls = _fake_remote()
    cp.ensure_universe(tmp_path, n_docs=300, fetch=fetch)
    assert calls == cp.shard_urls()[:2]   # 200 docs/shard: stops at the target
    calls.clear()
    docs = cp.load_universe(tmp_path, n_docs=300)
    assert calls == []                    # cached: no network on any later call
    assert len(docs) == 300
    meta = json.loads((cp.universe_dir(tmp_path) / cp._META_NAME).read_text())
    assert meta["url_template"] == cp.C4_URL_TEMPLATE
    assert meta["split"] == "en/validation"
    assert all(s["sha256"] and s["n_bytes"] for s in meta["shards"])


def test_universe_spans_shards_in_index_order(tmp_path):
    fetch, calls = _fake_remote(n_per_shard=200)
    docs = cp.load_universe(tmp_path, n_docs=500, fetch=fetch)
    assert calls == cp.shard_urls()[:3]
    assert [d["text"].split()[0] for d in docs[::200]] == ["shard0", "shard1", "shard2"]
    assert {d["source"] for d in docs} == {"c4-en-validation"}


def test_truncated_gz_fails_loudly_with_no_cache(tmp_path):
    fetch, _ = _fake_remote()

    def bad_fetch(url):
        return io.BytesIO(fetch(url).getvalue()[:40])       # cut mid-stream

    with pytest.raises(Exception):
        cp.ensure_universe(tmp_path, n_docs=300, fetch=bad_fetch)
    assert not (cp.universe_dir(tmp_path) / cp._UNIVERSE_NAME).exists()
    assert not (cp.universe_dir(tmp_path) / cp._META_NAME).exists()


def test_cache_hit_with_wrong_n_docs_raises_without_fetching(tmp_path):
    fetch, calls = _fake_remote()
    cp.ensure_universe(tmp_path, n_docs=300, fetch=fetch)
    calls.clear()

    def no_fetch(url):
        raise AssertionError("must not fetch over a frozen cache")

    # a 300-doc smoke cache must NOT satisfy a production-size request
    with pytest.raises(ValueError, match="frozen artifact"):
        cp.ensure_universe(tmp_path, n_docs=60_000, fetch=no_fetch)
    assert calls == []


def test_universe_metadata_reads_sidecar_without_fetching(tmp_path):
    fetch, calls = _fake_remote()
    cp.ensure_universe(tmp_path, n_docs=300, fetch=fetch)
    calls.clear()
    meta = cp.universe_metadata(tmp_path)
    assert meta["universe_version"] == cp.UNIVERSE_VERSION == "c4p1"
    assert int(meta["n_docs"]) == 300 and meta["shards"]
    assert calls == []
    with pytest.raises(FileNotFoundError, match="provenance"):
        cp.universe_metadata(tmp_path / "elsewhere")


def test_corrupt_meta_raises(tmp_path):
    fetch, _ = _fake_remote()
    cp.ensure_universe(tmp_path, n_docs=300, fetch=fetch)
    (cp.universe_dir(tmp_path) / cp._META_NAME).write_text("{not json")
    with pytest.raises(ValueError, match="unreadable"):
        cp.ensure_universe(tmp_path, n_docs=300, fetch=fetch)


def test_truncated_universe_file_raises_on_load(tmp_path):
    fetch, _ = _fake_remote()
    cp.ensure_universe(tmp_path, n_docs=300, fetch=fetch)
    path = cp.universe_dir(tmp_path) / cp._UNIVERSE_NAME
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = fh.readlines()[:100]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.writelines(lines)
    # the content hash catches this before load_universe's doc-count check does
    with pytest.raises(ValueError, match="does not match its sidecar|truncated or corrupt"):
        cp.load_universe(tmp_path, n_docs=300)


def test_docs_without_text_are_skipped_at_fetch():
    blob = _shard_bytes([{"id": 1}, {"text": ""}, {"text": "hello world"}])
    docs, n_lines, n_no_text = cp._read_docs(io.BytesIO(blob), 10)
    assert [d["text"] for d in docs] == ["hello world"]
    assert (n_lines, n_no_text) == (3, 2)


# ---------------------------------------------------------------- pool semantics


def _docs(n=200):
    return [{"text": f"doc{i} " + "word " * 60, "source": "c4-en-validation"}
            for i in range(n)]


def test_pool_determinism_and_token_sha():
    tok = _Tok()
    a = cp.build_c4_pool(tok, 8, docs=_docs())
    b = cp.build_c4_pool(tok, 8, docs=_docs())
    assert a.input_ids == b.input_ids
    assert a.token_sha256() == b.token_sha256()
    c = cp.build_c4_pool(tok, 8, docs=_docs(), seed=1)
    assert c.token_sha256() != a.token_sha256()


def test_pool_min_tokens_truncation_and_position0():
    tok = _Tok()
    docs = _docs(40)
    docs[int(cp.universe_permutation(40, 0)[0])] = {"text": "too short",
                                                    "source": "c4-en-validation"}
    pool = cp.build_c4_pool(tok, 5, docs=docs, max_tokens=10, min_tokens=5)
    assert pool.n_skipped == 1 and pool.n_scanned == 6
    assert all(len(ids) == 10 for ids in pool.input_ids)          # truncation
    assert all(el == list(range(1, 10)) for el in pool.eligible)  # position 0 excluded
    assert pool.source == "c4-en-validation:c4p1"


def test_pool_needs_enough_usable_docs():
    with pytest.raises(ValueError, match="usable docs"):
        cp.build_c4_pool(_Tok(), 5, docs=_docs(3))


def test_unigram_docs_are_disjoint_from_the_pool():
    tok, docs = _Tok(), _docs(50)
    pool = cp.build_c4_pool(tok, 8, docs=docs)
    later = list(cp.iter_universe_texts(docs, 0, pool.n_scanned))
    pooled = [docs[int(i)]["text"]
              for i in cp.universe_permutation(50, 0)[:pool.n_scanned]]
    assert set(later).isdisjoint(pooled)
    assert len(later) == 50 - pool.n_scanned


# ---------------------------------------------------------------- routing


def test_c4_recipe_routes_to_c4p1():
    from subliminal_learning.models.datadecide import DD_RECIPE_SEGMENTS

    assert pio.native_probe_pool("dd-1b-c4").version == "c4p1"
    assert pio.native_probe_pool("dd-1b-c4-seed2").version == "c4p1"
    assert pio.probe_pool_spec_for_repo("allenai/DataDecide-c4-1B") is pio.PROBE_POOL_C4P1
    assert pio.probe_pool_spec_for_version("c4p1") is pio.PROBE_POOL_C4P1
    assert (pio.PROBE_POOL_C4P1.n_docs, pio.PROBE_POOL_C4P1.max_tokens,
            pio.PROBE_POOL_C4P1.seed) == (32, 512, 0)
    pools = pio.dd_recipe_pools()
    assert len(pools) == 3
    assert pools[DD_RECIPE_SEGMENTS["c4"]] is pio.PROBE_POOL_C4P1
    assert set(pools) == {"dolma1_7", "dclm-baseline", "c4"}


def test_c4p1_is_not_mistaken_for_a_frozen_text_pool():
    """`cp1`/`cp1t-` prefixes are the one naming collision risk near `c4p1`."""
    from subliminal_learning.coherence.probe_families import (
        is_cp1t, is_frozen_text_pool, is_pp1t, is_pp1w,
    )

    assert not is_cp1t("c4p1") and not is_pp1t("c4p1") and not is_pp1w("c4p1")
    assert not is_frozen_text_pool("c4p1")
    assert pio.PROBE_POOL_C4P1 is not pio.PROBE_POOL_CP1


# ------------------------------------------------- cache half-state and content hash


def _no_fetch(url):
    raise AssertionError("must not fetch over an existing cache")


def _built(tmp_path):
    """A 300-doc cache under `tmp_path`; returns its (data, meta) paths."""
    fetch, _ = _fake_remote()
    cp.ensure_universe(tmp_path, n_docs=300, fetch=fetch)
    d = cp.universe_dir(tmp_path)
    return d / cp._UNIVERSE_NAME, d / cp._META_NAME


@pytest.mark.parametrize("missing", ["data", "meta"])
def test_half_state_cache_raises_without_fetching(tmp_path, missing):
    """Data without sidecar (or the reverse) is NOT a miss to refetch over."""
    path, meta_path = _built(tmp_path)
    (path if missing == "data" else meta_path).unlink()
    with pytest.raises(ValueError, match="half-written"):
        cp.ensure_universe(tmp_path, n_docs=300, fetch=_no_fetch)


def test_cache_hit_with_wrong_universe_version_raises(tmp_path):
    _path, meta_path = _built(tmp_path)
    meta = json.loads(meta_path.read_text())
    meta["universe_version"] = "somethingelse"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="frozen artifact"):
        cp.ensure_universe(tmp_path, n_docs=300, fetch=_no_fetch)


def test_build_records_the_universe_sha256(tmp_path):
    path, meta_path = _built(tmp_path)
    meta = json.loads(meta_path.read_text())
    assert meta[cp._SHA_KEY] == file_sha256(path)


def test_tampered_universe_file_raises_on_load(tmp_path):
    path, _meta_path = _built(tmp_path)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        docs = [json.loads(ln) for ln in fh]
    docs[0]["text"] = "tampered " + docs[0]["text"]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for rec in docs:
            fh.write(json.dumps(rec) + "\n")
    with pytest.raises(ValueError, match="does not match its sidecar"):
        cp.load_universe(tmp_path, n_docs=300)


def test_sidecar_without_content_hash_raises_and_can_be_stamped(tmp_path):
    path, meta_path = _built(tmp_path)
    meta = json.loads(meta_path.read_text())
    sha = meta.pop(cp._SHA_KEY)
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="stamp_universe_sha256"):
        cp.ensure_universe(tmp_path, n_docs=300, fetch=_no_fetch)
    assert cp.stamp_universe_sha256(tmp_path) == sha
    assert cp.stamp_universe_sha256(tmp_path) == sha          # idempotent
    assert cp.ensure_universe(tmp_path, n_docs=300, fetch=_no_fetch) == path


def test_stamp_refuses_a_conflicting_recorded_hash(tmp_path):
    _path, meta_path = _built(tmp_path)
    meta = json.loads(meta_path.read_text())
    meta[cp._SHA_KEY] = "0" * 64
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        cp.stamp_universe_sha256(tmp_path)


# ---------------------------------------------------- control-eval layout distinctness


def test_c4_student_control_dirs_differ_from_the_other_recipes():
    """A `c4` student's control trees must not land on the dolma1_7 or dclm ones."""
    from pathlib import Path

    from subliminal_learning.config import LoRAConfig, TrainConfig
    from subliminal_learning.models.datadecide import DataDecideSpec
    from subliminal_learning.students.layout import student_param_dir

    tc = TrainConfig(learning_rate=1e-4, n_epochs=5, batch_size=8, max_length=256,
                     max_grad_norm=1.0, warmup_ratio=0.05, lr_scheduler_type="cosine",
                     seed=1, optim="adamw_torch")
    lc = LoRAConfig(r=8, lora_alpha=32, lora_dropout=0.0, target_modules="all-linear")

    def dirs(recipe):
        spec = DataDecideSpec(size="1B", seed=2, recipe=recipe)
        return tuple(
            student_param_dir(Path("/out"), source=src, spec=spec, hub_org="allenai",
                              train_cfg=tc, lora_cfg=lc, number_dataset_tag="ex10-20_full10")
            for src in ("base", "rng"))

    c4, dolma, dclm = (dirs(r) for r in ("c4", "dolma1_7", "dclm-baseline"))
    assert len({c4, dolma, dclm}) == 3
    assert not set(c4) & (set(dolma) | set(dclm))
    assert all("DataDecide-c4-1B-seed-large-aux-2" in str(p) for p in c4)
