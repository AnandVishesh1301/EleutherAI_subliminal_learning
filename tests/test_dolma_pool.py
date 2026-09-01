"""dp1: Dolma v1.7 universe cache, mixture quotas, pool semantics, and anchor routing.

Fully offline — the shard fetcher and the document universe are injected, so no test
ever touches huggingface.co or olmo-data.org.
"""
import gzip
import inspect
import io
import json

import pytest

from subliminal_learning.analysis import labels
from subliminal_learning.coherence import dolma_pool as dp
from subliminal_learning.coherence import probe_io as pio

# ---------------------------------------------------------------- fakes


class _Tok:
    """Whitespace tokenizer: one id per word, deterministic across runs."""

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [self._ids(t) for t in text]}
        return {"input_ids": self._ids(text)}

    @staticmethod
    def _ids(text):
        return [abs(hash(w)) % 1000 for w in text.split()]


def _shard_bytes(records):
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for r in records:
            gz.write((json.dumps(r) + "\n").encode())
    return buf.getvalue()


def _fake_remote(n_per_shard=200, shards_per_source=2):
    """A fake olmo-data.org: URL list + one gz shard per URL. Returns (fetch, calls)."""
    prefixes = [p for _n, _w, ps in dp.DOLMA_V1_7_MIXTURE for p in ps]
    urls = [f"https://olmo-data.org/dolma-v1_7/{p}/{p}-{i:04d}.json.gz"
            for p in prefixes for i in range(shards_per_source)]
    blobs = {}
    for u in urls:
        src = u.split(dp._URL_SOURCE_MARKER, 1)[1].split("/", 1)[0]
        blobs[u] = _shard_bytes(
            [{"text": f"{src} doc {i} " + "word " * 40} for i in range(n_per_shard)])
    calls: list[str] = []

    def fetch(url):
        calls.append(url)
        if url == dp.DOLMA_URL_LIST:
            return io.BytesIO("\n".join(urls).encode())
        return io.BytesIO(blobs[url])

    return fetch, calls


# ---------------------------------------------------------------- universe cache


def test_universe_cache_hit_means_zero_fetches(tmp_path):
    fetch, calls = _fake_remote()
    dp.ensure_universe(tmp_path, n_docs=300, fetch=fetch)
    assert calls, "first build must fetch"
    calls.clear()
    docs = dp.load_universe(tmp_path, n_docs=300)
    assert calls == []                    # cached: no network on any later call
    assert len(docs) == 300
    meta = json.loads((dp.universe_dir(tmp_path) / dp._META_NAME).read_text())
    assert meta["url_list"] == dp.DOLMA_URL_LIST
    assert all(s["sha256"] and s["n_bytes"] for s in meta["shards"])


def test_universe_composition_matches_mixture(tmp_path):
    fetch, _ = _fake_remote(n_per_shard=1200)
    docs = dp.load_universe(tmp_path, n_docs=6000, fetch=fetch)
    got: dict[str, int] = {}
    for d in docs:
        got[d["source"]] = got.get(d["source"], 0) + 1
    shards = dp._shards_by_source(dp.fetch_url_list(fetch))
    assert got == dp.source_quotas(6000, shards)
    # mixture-faithful, not Common-Crawl-only: every card row is represented
    assert all(any(got.get(p) for p in ps) for _n, _w, ps in dp.DOLMA_V1_7_MIXTURE)
    cc = sum(got[p] for p in ("cc_en_head", "cc_en_middle", "cc_en_tail"))
    assert 0.3 < cc / len(docs) < 0.42     # ~597.75 / 1645 effective tokens


def test_truncated_gz_fails_loudly_with_no_cache(tmp_path):
    fetch, _ = _fake_remote()

    def bad_fetch(url):
        s = fetch(url)
        if url == dp.DOLMA_URL_LIST:
            return s
        return io.BytesIO(s.getvalue()[:40])       # cut mid-stream

    with pytest.raises(Exception):
        dp.ensure_universe(tmp_path, n_docs=300, fetch=bad_fetch)
    assert not (dp.universe_dir(tmp_path) / dp._UNIVERSE_NAME).exists()
    assert not (dp.universe_dir(tmp_path) / dp._META_NAME).exists()


def test_cache_hit_with_wrong_n_docs_raises_without_fetching(tmp_path):
    fetch, calls = _fake_remote()
    dp.ensure_universe(tmp_path, n_docs=300, fetch=fetch)
    calls.clear()

    def no_fetch(url):
        raise AssertionError("must not fetch over a frozen cache")

    # a 300-doc smoke cache must NOT satisfy a production-size request
    with pytest.raises(ValueError, match="frozen artifact"):
        dp.ensure_universe(tmp_path, n_docs=60_000, fetch=no_fetch)
    assert calls == []


def test_universe_metadata_reads_sidecar_without_fetching(tmp_path):
    fetch, calls = _fake_remote()
    dp.ensure_universe(tmp_path, n_docs=300, fetch=fetch)
    calls.clear()
    meta = dp.universe_metadata(tmp_path)
    assert meta["universe_version"] == dp.UNIVERSE_VERSION
    assert int(meta["n_docs"]) == 300
    assert meta["url_list"] == dp.DOLMA_URL_LIST and meta["shards"]
    assert calls == []
    with pytest.raises(FileNotFoundError, match="provenance"):
        dp.universe_metadata(tmp_path / "elsewhere")


def test_corrupt_meta_raises(tmp_path):
    fetch, _ = _fake_remote()
    dp.ensure_universe(tmp_path, n_docs=300, fetch=fetch)
    (dp.universe_dir(tmp_path) / dp._META_NAME).write_text("{not json")
    with pytest.raises(ValueError, match="unreadable"):
        dp.ensure_universe(tmp_path, n_docs=300, fetch=fetch)


def test_truncated_universe_file_raises_on_load(tmp_path):
    fetch, _ = _fake_remote()
    dp.ensure_universe(tmp_path, n_docs=300, fetch=fetch)
    path = dp.universe_dir(tmp_path) / dp._UNIVERSE_NAME
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = fh.readlines()[:100]
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.writelines(lines)
    with pytest.raises(ValueError, match="truncated or corrupt"):
        dp.load_universe(tmp_path, n_docs=300)


def test_docs_without_text_are_skipped_at_fetch():
    blob = _shard_bytes([{"id": 1}, {"text": ""}, {"text": "hello world"}])
    docs, n_lines, n_no_text = dp._read_docs(io.BytesIO(blob), 10, "wiki")
    assert [d["text"] for d in docs] == ["hello world"]
    assert (n_lines, n_no_text) == (3, 2)


# ---------------------------------------------------------------- pool semantics


def _docs(n=200):
    return [{"text": f"doc{i} " + "word " * 60, "source": "wiki"} for i in range(n)]


def test_pool_determinism_and_token_sha():
    tok = _Tok()
    a = dp.build_dolma_pool(tok, 8, docs=_docs())
    b = dp.build_dolma_pool(tok, 8, docs=_docs())
    assert a.input_ids == b.input_ids
    assert a.token_sha256() == b.token_sha256()
    # a different seed is a different pool
    c = dp.build_dolma_pool(tok, 8, docs=_docs(), seed=1)
    assert c.token_sha256() != a.token_sha256()


def test_pool_min_tokens_truncation_and_position0():
    tok = _Tok()
    docs = _docs(40)
    docs[int(dp.universe_permutation(40, 0)[0])] = {"text": "too short", "source": "wiki"}
    pool = dp.build_dolma_pool(tok, 5, docs=docs, max_tokens=10, min_tokens=5)
    assert pool.n_skipped == 1 and pool.n_scanned == 6
    assert all(len(ids) == 10 for ids in pool.input_ids)          # truncation
    assert all(el == list(range(1, 10)) for el in pool.eligible)  # position 0 excluded
    # The universe is frozen with no build-revision suffix: a change would be dp2.
    assert pool.source == "dolma-v1_7:dp1"


def test_pool_needs_enough_usable_docs():
    with pytest.raises(ValueError, match="usable docs"):
        dp.build_dolma_pool(_Tok(), 5, docs=_docs(3))


def test_unigram_docs_are_disjoint_from_the_pool():
    tok, docs = _Tok(), _docs(50)
    pool = dp.build_dolma_pool(tok, 8, docs=docs)
    later = list(dp.iter_universe_texts(docs, 0, pool.n_scanned))
    pooled = [docs[int(i)]["text"]
              for i in dp.universe_permutation(50, 0)[:pool.n_scanned]]
    assert set(later).isdisjoint(pooled)
    assert len(later) == 50 - pool.n_scanned


# ---------------------------------------------------------------- routing / anchors


def test_probe_pool_spec_for_repo_dispatch():
    assert pio.probe_pool_spec_for_repo("allenai/DataDecide-dolma1_7-1B") is pio.PROBE_POOL_DP1
    for repo in ("EleutherAI/pythia-160m", "EleutherAI/pythia-410m-seed1"):
        assert pio.probe_pool_spec_for_repo(repo) is pio.PROBE_POOL_PP1
    assert (pio.PROBE_POOL_DP1.n_docs, pio.PROBE_POOL_DP1.max_tokens,
            pio.PROBE_POOL_DP1.seed, pio.PROBE_POOL_DP1.version) == (32, 512, 0, "dp1")


def _write_anchor(root, pool, cuts):
    d = root / "analysis"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"unigram_anchor_{pool}.json").write_text(json.dumps({"delta_cross": cuts}))


def test_anchored_dead_cut_pool_kwarg(tmp_path):
    _write_anchor(tmp_path, "pp1", {"pythia-160m": 3.5})
    _write_anchor(tmp_path, "dp1", {"DataDecide-dolma1_7-1B-seed-default": 2.25})
    assert pio.anchored_dead_cut(tmp_path, "EleutherAI/pythia-160m") == 3.5   # default pp1
    # DataDecide keys are per-seed dir names; the full model_ref resolves to one.
    assert pio.anchored_dead_cut(
        tmp_path, "DataDecide-dolma1_7-1B-seed-default", pool="dp1") == 2.25
    assert pio.anchored_dead_cut(
        tmp_path, "allenai/DataDecide-dolma1_7-1B@step69369-seed-default",
        pool="dp1") == 2.25
    # A bare dd repo id is ill-defined: three seeds share it.
    with pytest.raises(KeyError):
        pio.anchored_dead_cut(tmp_path, "allenai/DataDecide-dolma1_7-1B", pool="dp1")
    with pytest.raises(KeyError):
        pio.anchored_dead_cut(tmp_path, "pythia-160m", pool="dp1")


def test_labels_wrappers_pass_pool_through(tmp_path):
    _write_anchor(tmp_path, "pp1", {"pythia-160m": 3.5})
    _write_anchor(tmp_path, "dp1", {"dd-1b": 2.25})
    assert labels.anchored_cut(tmp_path, "pythia-160m") == 3.5
    assert labels.anchored_cut(tmp_path, "dd-1b", pool="dp1") == 2.25
    assert labels.anchored_cuts(tmp_path, ["dd-1b"], pool="dp1") == {"dd-1b": 2.25}


# ---------------------------------------------------------------- anchor builder


SHA = "a" * 64
OTHER_SHA = "b" * 64


def _probe_rec(root, base, ce_base, pool_version, pool_sha=SHA, tag="tag"):
    p = root / "teachers" / "x" / base / tag / "ftseed1" / "eval"
    p.mkdir(parents=True, exist_ok=True)
    rec = {"ce_base": ce_base}
    if pool_version is not None:
        rec["pool_version"] = pool_version
    if pool_sha is not None:
        rec["pool_sha"] = pool_sha
    (p / "pile_probe.json").write_text(json.dumps(rec))


def test_ce_base_map_filters_on_pool_version_and_sha(tmp_path):
    from scripts.build_unigram_anchor import _ce_base_map, main

    _probe_rec(tmp_path, "pythia-160m", 3.1, "pp1")
    _probe_rec(tmp_path, "pythia-410m", 2.9, None, pool_sha=None)  # legacy pp1 record
    _probe_rec(tmp_path, "pythia-1b", 2.7, "pp1", pool_sha=OTHER_SHA)  # other pp1 pool
    _probe_rec(tmp_path, "DataDecide-dolma1_7-1B-seed-default", 2.5, "dp1")
    assert _ce_base_map(tmp_path, "pp1", SHA) == {"pythia-160m": 3.1, "pythia-410m": 2.9}
    assert _ce_base_map(tmp_path, "dp1", SHA) == {
        "DataDecide-dolma1_7-1B-seed-default": 2.5}
    # a dp1 record without a sha is never legacy-allowed
    _probe_rec(tmp_path, "dd-1b-nosha", 2.4, "dp1", pool_sha=None)
    assert "dd-1b-nosha" not in _ce_base_map(tmp_path, "dp1", SHA)
    # the dp1 anchor must build with an EMPTY map (no dd probe records yet)
    assert _ce_base_map(tmp_path / "empty", "dp1", SHA) == {}
    assert inspect.signature(main).parameters["pool"].default.default == "pp1"


def test_ce_base_map_raises_on_disagreeing_records(tmp_path):
    from scripts.build_unigram_anchor import _ce_base_map

    _probe_rec(tmp_path, "pythia-160m", 3.1, "pp1", tag="tagA")
    _probe_rec(tmp_path, "pythia-160m", 3.100_02, "pp1", tag="tagB")  # kernel noise: ok
    assert _ce_base_map(tmp_path, "pp1", SHA) == {"pythia-160m": 3.1}
    _probe_rec(tmp_path, "pythia-160m", 3.4, "pp1", tag="tagC")
    with pytest.raises(SystemExit, match="disagreement"):
        _ce_base_map(tmp_path, "pp1", SHA)


def test_ce_base_map_harvests_ckpt_gathered_trees(tmp_path):
    """Revision/step-pinned bases (one extra `ckpt/` level) are harvested under @step keys."""
    from scripts.build_unigram_anchor import _ce_base_map

    def _write(parts, ce):
        p = tmp_path.joinpath(*parts, "eval")
        p.mkdir(parents=True, exist_ok=True)
        (p / "pile_probe.json").write_text(
            json.dumps({"ce_base": ce, "pool_version": "pp1", "pool_sha": SHA}))

    # pinned Pythia teacher + pinned DataDecide student (teacher-source and base control)
    _write(("teachers", "dog", "ckpt", "pythia-160m-data-seed1@step3000", "tag", "ftseed1"),
           3.2)
    _write(("students", "dog", "ckpt", "pythia-160m@step1000", "teacher-pythia-160m",
            "ttag_tseed1", "ndt", "stag", "ftseed1"), 3.3)
    _write(("students", "controls", "base", "ckpt",
            "DataDecide-dolma1_7-1B-seed-default@step17500",
            "teacher-DataDecide-dolma1_7-1B-seed-default@step17500", "ndt", "stag",
            "ftseed1"), 2.6)
    _write(("students", "controls", "rng", "ckpt", "pythia-410m@step2000", "ndt", "stag",
            "ftseed1"), 2.8)
    # an unpinned record alongside them must keep its plain key
    _probe_rec(tmp_path, "pythia-160m", 3.1, "pp1")
    found = _ce_base_map(tmp_path, "pp1", SHA)
    assert found == {
        "pythia-160m-data-seed1@step3000": 3.2,
        "pythia-160m@step1000": 3.3,
        "DataDecide-dolma1_7-1B-seed-default@step17500": 2.6,
        "pythia-410m@step2000": 2.8,
        "pythia-160m": 3.1,
    }
    assert "ckpt" not in found


def test_ce_base_map_harvests_rng_controls(tmp_path):
    from scripts.build_unigram_anchor import _ce_base_map

    p = tmp_path / "students" / "controls" / "rng" / "pythia-160m" / "ndt" / "stag"
    p = p / "ftseed1" / "eval"
    p.mkdir(parents=True, exist_ok=True)
    (p / "pile_probe.json").write_text(
        json.dumps({"ce_base": 3.05, "pool_version": "pp1", "pool_sha": SHA}))
    assert _ce_base_map(tmp_path, "pp1", SHA) == {"pythia-160m": 3.05}
