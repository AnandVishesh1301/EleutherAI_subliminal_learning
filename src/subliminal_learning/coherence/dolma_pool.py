"""Dolma v1.7 document universe and the ``dp1`` integrity probe pool (OLMo tokenizer).

The DataDecide/OLMo arm needs the same instrument pp1 gives the Pythia arm: a frozen,
content-hashed pool of *pretraining-distribution* documents, plus a disjoint document
stream for the unigram anchor. Pile is the wrong corpus for an OLMo model (its base CE
would be measured off-distribution), so ``dp1`` is drawn from Dolma v1.7 — the recipe
DataDecide's `dolma1_7` models were trained on.

Provenance chain (nothing here is a mirror or a proxy corpus):

- ``allenai/dolma`` on the Hub is a URL-LIST repo, not a data repo: ``urls/v1_7.txt``
  lists 2,419 ``https://olmo-data.org/dolma-v1_7/{source}/{shard}.json.gz`` shards
  (gzipped jsonl, one document per line). It is NEVER loaded with `datasets.load_dataset`
  (>1 TB, and this repo has been burned by the all-splits trap); the shards are fetched
  by explicit URL here.
- Documents are drawn as a deterministic, stratified APPROXIMATION of the v1.7 mixture:
  per-source DOC quotas proportional to the official v1.7 effective TOKEN shares
  (`DOLMA_V1_7_MIXTURE`), so every card row is represented rather than a Common-Crawl-only
  slice. Two caveats, both real: (a) doc quotas set from token weights are not a
  token-proportional sample — short-document sources (reddit, flan) end up token-
  underweighted, and the 512-token truncation only partially equalizes document lengths;
  (b) within a source the documents are a deterministic prefix of its earliest shards in
  the URL list, and the seed-0 permutation shuffles the cached universe but does not undo
  that first-shard selection.
- The universe is cached ONCE under ``{dataset_cache_root}/dolma_dp1/`` (one jsonl.gz
  plus a sidecar metadata JSON recording shard URLs, byte shas, and the weights used);
  every later call reads the cache and touches the network zero times.

Held-out status (2026-08-31, Angela's call): unlike pp1, whose documents come from the
Pile *validation* split, Dolma ships no held-out split, so the dp1 universe is
deliberately an IN-DISTRIBUTION subset of the training population — the documents may
have been seen in pretraining. dp1 measures damage relative to the base on the base's own
distribution, which is what the integrity readout needs; it is not a generalization test.

`build_dolma_pool` mirrors `cka.pool_io.build_pile_pool` semantics exactly (seeded doc
permutation, min-token skip, truncation, position 0 never eligible), so a dp1 probe
record is read the same way a pp1 one is.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from loguru import logger

from subliminal_learning.cka.pool_io import SequencePool

DOLMA_URL_LIST = "https://huggingface.co/datasets/allenai/dolma/resolve/main/urls/v1_7.txt"
# The path component after ``dolma-v1_7/`` identifies the source of a shard URL.
_URL_SOURCE_MARKER = "dolma-v1_7/"

# Official Dolma v1.7 mixture, transcribed from the allenai/dolma dataset card's
# "Summary Statistics (v1.7)" table: effective tokens = OLMo tokens (billions) x the
# card's sample proportion (Dolma's CC 50%, C4 50%, Wikipedia 200%, everything else
# 100%). Each entry maps one card row to the shard-URL source prefixes that carry it.
DOLMA_V1_7_MIXTURE: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    ("dolma_cc", 1195.5 * 0.50, ("cc_en_head", "cc_en_middle", "cc_en_tail")),
    ("refined_web", 456.4, ("falcon-refinedweb-filtered",)),
    ("starcoder", 263.8, ("starcoder",)),
    ("reddit", 79.9, ("reddit",)),
    ("c4", 138.4 * 0.50, ("c4-filtered",)),
    ("pes2o", 57.2, ("pes2o",)),
    ("arxiv", 28.0, ("redpajama-arxiv",)),
    ("stackexchange", 19.6, ("redpajama-stackexchange",)),
    ("flan", 16.5, ("tulu_flan",)),
    ("cc_news", 14.3, ("cc_news_head", "cc_news_middle", "cc_news_tail")),
    ("open_web_math", 12.6, ("proof_pile_2-open_web_math",)),
    ("algebraic_stack", 12.6, ("proof_pile_2-algebraic_stack",)),
    ("wikipedia", 3.7 * 2.00, ("wiki",)),
    ("gutenberg", 5.3, ("books",)),
    ("megawika", 4.6, ("wikiref_megawika",)),
)

# Frozen universe constants: the universe is built once and cached; changing either of
# these changes the pool identity (and so every dp1 number).
UNIVERSE_N_DOCS = 60_000
# No build-revision suffix (2026-08-31 freeze decision): this universe is frozen as built,
# and any future change to it would be a NEW pool version (dp2), never a v2 of dp1 — so a
# "-v1" suffix would encode a possibility that has been ruled out.
UNIVERSE_VERSION = "dp1"
# Safety rail on a fetch that early-stops per source: refuse to pull more than this many
# compressed bytes in one universe build.
MAX_FETCH_BYTES = 2 * 1024**3

_UNIVERSE_NAME = "dolma_v1_7_universe.jsonl.gz"
_META_NAME = "dolma_v1_7_universe.meta.json"


# --------------------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------------------


# olmo-data.org 403s the default `Python-urllib/3.x` user agent (verified 2026-08-31);
# an explicit UA is required, not cosmetic.
_USER_AGENT = "subliminal-learning/dolma_pool (+https://olmo-data.org)"


def _http_stream(url: str):
    """Open `url` for binary streaming reads (the default fetcher; injectable in tests)."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "*/*"})
    return urllib.request.urlopen(req, timeout=120)  # noqa: S310 -- https URLs only


class _CountingReader:
    """Read-through wrapper recording sha256 and byte count of what was consumed.

    Only the CONSUMED PREFIX is hashed: shard reads stop as soon as a source's quota is
    met, so the recorded sha identifies the bytes this build actually used.
    """

    def __init__(self, raw):
        self._raw = raw
        self._h = hashlib.sha256()
        self.n_bytes = 0

    def read(self, n: int = -1) -> bytes:
        b = self._raw.read(n)
        self._h.update(b)
        self.n_bytes += len(b)
        return b

    def sha256(self) -> str:
        return self._h.hexdigest()


def fetch_url_list(fetch: Callable[[str], Any] | None = None) -> list[str]:
    """The v1.7 shard URLs, in the URL list's order."""
    stream = (fetch or _http_stream)(DOLMA_URL_LIST)
    text = stream.read().decode("utf-8")
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _shards_by_source(urls: list[str]) -> dict[str, list[str]]:
    """Group shard URLs by their `dolma-v1_7/{source}/` component, order preserved."""
    out: dict[str, list[str]] = {}
    for u in urls:
        if _URL_SOURCE_MARKER not in u:
            continue
        src = u.split(_URL_SOURCE_MARKER, 1)[1].split("/", 1)[0]
        out.setdefault(src, []).append(u)
    return out


def _apportion(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation of `total` items over `weights` (deterministic)."""
    w_sum = sum(weights.values())
    exact = {k: total * w / w_sum for k, w in weights.items()}
    base = {k: int(v) for k, v in exact.items()}
    rest = total - sum(base.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - base[k]), k))
    for k in order[:rest]:
        base[k] += 1
    return base


def source_quotas(n_docs: int, shards: dict[str, list[str]]) -> dict[str, int]:
    """Per-shard-prefix doc quotas approximating the v1.7 mixture over `n_docs` documents.

    Doc counts are set proportional to TOKEN weights, so this is a stratified
    approximation, not a token-proportional sample: short-document sources come out
    token-underweighted (truncation to 512 tokens only partially equalizes lengths).

    Each card row's quota is apportioned over its shard prefixes in proportion to the
    number of shards the URL list gives that prefix, so a multi-prefix source (Dolma's
    CC, CC News) keeps its internal head/middle/tail proportions.
    """
    per_row = _apportion(n_docs, {name: w for name, w, _ in DOLMA_V1_7_MIXTURE})
    quotas: dict[str, int] = {}
    for name, _w, prefixes in DOLMA_V1_7_MIXTURE:
        avail = {p: float(len(shards.get(p, []))) for p in prefixes}
        if not sum(avail.values()):
            raise ValueError(f"no shard URLs for Dolma v1.7 source {name!r} ({prefixes})")
        quotas.update(_apportion(per_row[name], avail))
    return quotas


def _read_docs(stream, quota: int, source: str) -> tuple[list[dict], int, int]:
    """Up to `quota` documents from one gzipped-jsonl shard stream.

    Returns (docs, n_lines_read, n_no_text). A truncated/corrupt gz stream raises.
    """
    docs: list[dict] = []
    n_lines = n_no_text = 0
    with gzip.GzipFile(fileobj=stream) as gz:
        for raw in gz:
            n_lines += 1
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                n_no_text += 1
                continue
            text = rec.get("text")
            if not text:
                n_no_text += 1
                continue
            docs.append({"text": text, "source": source})
            if len(docs) == quota:
                break
    return docs, n_lines, n_no_text


def _fetch_universe(n_docs: int, fetch: Callable[[str], Any] | None) -> tuple[list[dict], dict]:
    """Fetch the mixture-approximating document universe; returns (docs, provenance meta)."""
    fetch = fetch or _http_stream
    urls = fetch_url_list(fetch)
    shards = _shards_by_source(urls)
    quotas = source_quotas(n_docs, shards)

    docs: list[dict] = []
    shard_meta: list[dict] = []
    counts: dict[str, int] = {}
    total_bytes = 0
    for prefix, quota in quotas.items():
        got: list[dict] = []
        for url in shards[prefix]:
            if len(got) >= quota:
                break
            reader = _CountingReader(fetch(url))
            part, n_lines, n_no_text = _read_docs(reader, quota - len(got), prefix)
            total_bytes += reader.n_bytes
            shard_meta.append(dict(url=url, source=prefix, sha256=reader.sha256(),
                                   n_bytes=reader.n_bytes, n_lines=n_lines,
                                   n_docs=len(part), n_no_text=n_no_text))
            got.extend(part)
            if total_bytes > MAX_FETCH_BYTES:
                raise RuntimeError(
                    f"Dolma universe fetch exceeded {MAX_FETCH_BYTES} bytes at {url}")
        if len(got) < quota:
            raise ValueError(
                f"only {len(got)} docs (need {quota}) across all {prefix} shards")
        counts[prefix] = len(got)
        docs.extend(got)
    meta = dict(
        kind="dolma_v1_7_universe", schema_version=1, universe_version=UNIVERSE_VERSION,
        url_list=DOLMA_URL_LIST, n_docs=len(docs), quotas=quotas, source_counts=counts,
        mixture={name: w for name, w, _ in DOLMA_V1_7_MIXTURE},
        shards=shard_meta, n_fetched_bytes=total_bytes, fetched_at=time.time(),
    )
    return docs, meta


# --------------------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------------------


def universe_dir(cache_root: Path | None = None) -> Path:
    """``{dataset_cache_root}/dolma_dp1`` — the root resolved at time of use."""
    if cache_root is None:
        from subliminal_learning.config import load_paths_local

        cache_root = Path(load_paths_local()["dataset_cache_root"])
    return Path(cache_root) / "dolma_dp1"


def ensure_universe(
    cache_root: Path | None = None,
    *,
    n_docs: int = UNIVERSE_N_DOCS,
    fetch: Callable[[str], Any] | None = None,
) -> Path:
    """Path of the cached universe jsonl.gz, fetching it once if absent.

    Writes are atomic (temp + rename) and the metadata sidecar lands first-but-renamed
    together with the data, so a mid-fetch failure (truncated gz, transfer cap) leaves
    NO partial cache and raises.

    A cache hit is VALIDATED against the sidecar (``universe_version``, ``n_docs``): a
    300-doc smoke cache must never satisfy a 60,000-doc production request. A mismatch
    raises — the cache is a frozen artifact, moved or removed deliberately, never
    silently refetched or overwritten.
    """
    d = universe_dir(cache_root)
    path, meta_path = d / _UNIVERSE_NAME, d / _META_NAME
    if path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            cached = (meta["universe_version"], int(meta["n_docs"]))
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"unreadable Dolma universe metadata {meta_path}: {e}. The cached "
                "universe is a frozen artifact — move or remove it deliberately; this "
                "code will not refetch over it.") from e
        if cached != (UNIVERSE_VERSION, int(n_docs)):
            raise ValueError(
                f"cached Dolma universe at {d} is {cached[0]}/{cached[1]} docs, but "
                f"{UNIVERSE_VERSION}/{n_docs} docs were requested. The cached universe "
                "is a frozen artifact — move or remove it deliberately; this code will "
                "not silently refetch or overwrite it.")
        return path
    docs, meta = _fetch_universe(n_docs, fetch)
    d.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for rec in docs:
            fh.write(json.dumps(rec) + "\n")
    tmp_meta = meta_path.with_suffix(meta_path.suffix + f".tmp{os.getpid()}")
    tmp_meta.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, path)
    os.replace(tmp_meta, meta_path)
    logger.info("Built Dolma v1.7 universe: {} docs, {} bytes fetched -> {}",
                len(docs), meta["n_fetched_bytes"], path)
    return path


def load_universe(cache_root: Path | None = None, **kwargs) -> list[dict]:
    """The cached universe documents, in build order (fetching once if needed).

    The loaded doc count is checked against the sidecar's ``n_docs`` (counted during the
    read — no second pass, and no byte re-hashing: the downstream token-sha pin is the
    strong identity check).
    """
    path = ensure_universe(cache_root, **kwargs)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        docs = [json.loads(ln) for ln in fh]
    meta = json.loads((path.parent / _META_NAME).read_text())
    if len(docs) != int(meta["n_docs"]):
        raise ValueError(
            f"Dolma universe {path} holds {len(docs)} docs but its metadata records "
            f"{meta['n_docs']} — the cached artifact is truncated or corrupt.")
    return docs


def universe_metadata(cache_root: Path | None = None) -> dict:
    """The cached universe's sidecar metadata (shard URLs/shas, quotas, version, n_docs).

    Reads the ALREADY-cached sidecar and never fetches: callers that want the universe
    built should go through `ensure_universe`/`load_universe` first. Provenance readouts
    (e.g. the dp1 unigram anchor record) use this rather than re-deriving identity.
    """
    p = universe_dir(cache_root) / _META_NAME
    if not p.is_file():
        raise FileNotFoundError(
            f"no Dolma universe metadata at {p} — build the universe (ensure_universe) "
            "before asking for its provenance.")
    return json.loads(p.read_text())


def universe_permutation(n: int, seed: int = 0) -> np.ndarray:
    """The seeded document order shared by the dp1 pool and its unigram anchor."""
    return np.random.default_rng(seed).permutation(n)


def iter_universe_texts(
    docs: list[dict], seed: int = 0, start: int = 0
) -> Iterator[str]:
    """Universe texts in permutation order from position `start`.

    The anchor consumes documents DISJOINT from the pool's by passing the pool's
    ``n_scanned`` as `start` — exactly the pp1 disjointness design.
    """
    for i in universe_permutation(len(docs), seed)[start:]:
        yield docs[int(i)].get("text") or ""


def build_dolma_pool(
    tokenizer,
    n_seqs: int,
    *,
    max_tokens: int = 512,
    min_tokens: int = 32,
    seed: int = 0,
    cache_root: Path | None = None,
    docs: list[dict] | None = None,
) -> SequencePool:
    """Dolma v1.7 pool: seeded doc permutation, truncated to `max_tokens`.

    Mirrors `cka.pool_io.build_pile_pool`: every position except 0 is eligible, docs
    under `min_tokens` (including any without usable text) are skipped, and the scan
    counts are recorded so the length bias of the slice stays visible.
    """
    if docs is None:
        docs = load_universe(cache_root)
    input_ids: list[list[int]] = []
    eligible: list[list[int]] = []
    n_scanned = n_skipped = 0
    for text in iter_universe_texts(docs, seed):
        n_scanned += 1
        ids = list(tokenizer(text, add_special_tokens=False)["input_ids"][:max_tokens])
        if len(ids) < min_tokens:
            n_skipped += 1
            continue
        input_ids.append(ids)
        eligible.append(list(range(1, len(ids))))
        if len(input_ids) == n_seqs:
            break
    if len(input_ids) < n_seqs:
        raise ValueError(
            f"only {len(input_ids)} usable docs (need {n_seqs}) in the Dolma v1.7 "
            f"universe with min_tokens={min_tokens}"
        )
    logger.info(
        "Built Dolma pool: {} seqs ({} of {} scanned docs skipped as < {} tokens)",
        n_seqs, n_skipped, n_scanned, min_tokens,
    )
    return SequencePool(input_ids, eligible, f"dolma-v1_7:{UNIVERSE_VERSION}",
                        n_scanned, n_skipped)
