"""C4 (en) document universe and the ``c4p1`` integrity probe pool (OLMo tokenizer).

The third DataDecide data recipe, `c4`, needs the instrument dp1 gives the `dolma1_7`
bases and cp1t-olmo the `dclm-baseline` ones: a frozen, content-hashed pool of
*pretraining-distribution* documents plus a disjoint document stream for the unigram
anchor. ``c4p1`` is drawn from C4 — the corpus DataDecide's `c4` 1B models were
pretrained on.

Provenance chain (nothing here is a mirror or a proxy corpus):

- The source is the C4 ``en`` VALIDATION split of ``allenai/c4``, fetched by explicit
  shard URL (``en/c4-validation.{i:05d}-of-00008.json.gz``, gzipped jsonl with a ``text``
  field per line). It is NEVER loaded with `datasets.load_dataset`: this repo has been
  burned by the all-splits trap, and the train split is >300 GB.
- Documents are a deterministic prefix of `UNIVERSE_N_DOCS` across the shards in index
  order. There is a single source and no mixture quota to approximate (unlike dp1).
- The universe is cached ONCE under ``{dataset_cache_root}/c4_c4p1/`` (one jsonl.gz plus
  a sidecar metadata JSON recording shard URLs, byte shas, and n_docs); every later call
  reads the cache and touches the network zero times.

Held-out status: c4p1 IS genuinely held-out. C4's validation split is disjoint from its
train split, and the DataDecide `c4` recipe is a subset of the TRAIN split — so these
documents were not seen in pretraining. That matches pp1 (Pile validation) and cp1 (DCLM
held-out shards) and contrasts with dp1, which is in-distribution by necessity because
Dolma ships no held-out split.

`build_c4_pool` mirrors `cka.pool_io.build_pile_pool` semantics exactly (seeded doc
permutation, min-token skip, truncation, position 0 never eligible), so a c4p1 probe
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
from subliminal_learning.utils.io import file_sha256

C4_URL_TEMPLATE = (
    "https://huggingface.co/datasets/allenai/c4/resolve/main/en/"
    "c4-validation.{i:05d}-of-00008.json.gz"
)
C4_N_SHARDS = 8

# Frozen universe constants: the universe is built once and cached; changing either of
# these changes the pool identity (and so every c4p1 number). The size mirrors dp1's;
# one validation shard holds ~45k documents, so two shards suffice.
UNIVERSE_N_DOCS = 60_000
# No build-revision suffix: this universe is frozen as built, and any future change to it
# would be a NEW pool version (c4p2), never a v2 of c4p1.
UNIVERSE_VERSION = "c4p1"
# Safety rail on a fetch that early-stops once the doc target is met: refuse to pull more
# than this many compressed bytes in one universe build.
MAX_FETCH_BYTES = 2 * 1024**3

_UNIVERSE_NAME = "c4_en_validation_universe.jsonl.gz"
_META_NAME = "c4_en_validation_universe.meta.json"


# --------------------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------------------


_USER_AGENT = "subliminal-learning/c4_pool (+https://huggingface.co)"


def _http_stream(url: str):
    """Open `url` for binary streaming reads (the default fetcher; injectable in tests)."""
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT, "Accept": "*/*"})
    return urllib.request.urlopen(req, timeout=120)  # noqa: S310 -- https URLs only


class _CountingReader:
    """Read-through wrapper recording sha256 and byte count of what was consumed.

    Only the CONSUMED PREFIX is hashed: a shard read stops as soon as the doc target is
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


def shard_urls() -> list[str]:
    """The C4 en validation shard URLs, in index order."""
    return [C4_URL_TEMPLATE.format(i=i) for i in range(C4_N_SHARDS)]


def _read_docs(stream, quota: int) -> tuple[list[dict], int, int]:
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
            docs.append({"text": text, "source": "c4-en-validation"})
            if len(docs) == quota:
                break
    return docs, n_lines, n_no_text


def _fetch_universe(n_docs: int, fetch: Callable[[str], Any] | None) -> tuple[list[dict], dict]:
    """Fetch the deterministic document prefix; returns (docs, provenance meta)."""
    fetch = fetch or _http_stream
    docs: list[dict] = []
    shard_meta: list[dict] = []
    total_bytes = 0
    for url in shard_urls():
        if len(docs) >= n_docs:
            break
        reader = _CountingReader(fetch(url))
        part, n_lines, n_no_text = _read_docs(reader, n_docs - len(docs))
        total_bytes += reader.n_bytes
        shard_meta.append(dict(url=url, sha256=reader.sha256(), n_bytes=reader.n_bytes,
                               n_lines=n_lines, n_docs=len(part), n_no_text=n_no_text))
        docs.extend(part)
        if total_bytes > MAX_FETCH_BYTES:
            raise RuntimeError(f"C4 universe fetch exceeded {MAX_FETCH_BYTES} bytes at {url}")
    if len(docs) < n_docs:
        raise ValueError(
            f"only {len(docs)} docs (need {n_docs}) across all {C4_N_SHARDS} C4 en "
            "validation shards")
    meta = dict(
        kind="c4_en_validation_universe", schema_version=1,
        universe_version=UNIVERSE_VERSION, url_template=C4_URL_TEMPLATE,
        split="en/validation", n_docs=len(docs), shards=shard_meta,
        n_fetched_bytes=total_bytes, fetched_at=time.time(),
    )
    return docs, meta


# --------------------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------------------


def universe_dir(cache_root: Path | None = None) -> Path:
    """``{dataset_cache_root}/c4_c4p1`` — the root resolved at time of use."""
    if cache_root is None:
        from subliminal_learning.config import load_paths_local

        cache_root = Path(load_paths_local()["dataset_cache_root"])
    return Path(cache_root) / "c4_c4p1"


_SHA_KEY = "universe_sha256"


def _verify_universe_sha256(path: Path, meta: dict, d: Path) -> None:
    """Gate a cache hit on the CONTENT of the cached file, not just its version/count."""
    recorded = meta.get(_SHA_KEY)
    if recorded is None:
        raise ValueError(
            f"cached C4 universe at {d} records no {_SHA_KEY}, so its CONTENT "
            f"cannot be verified. Stamp the existing (trusted) cache once with "
            f"`coherence.c4_pool.stamp_universe_sha256()`.")
    got = file_sha256(path)
    if got != recorded:
        raise ValueError(
            f"cached C4 universe at {d} does not match its sidecar {_SHA_KEY} "
            f"(file {got}, recorded {recorded}) — the cached artifact changed under "
            "us; it is a frozen artifact, move or remove it deliberately.")


def stamp_universe_sha256(cache_root: Path | None = None) -> str:
    """Add `universe_sha256` to a LEGACY sidecar by hashing the cached file; returns it.

    Idempotent (a sidecar already carrying the same sha is left untouched) and refuses a
    sidecar carrying a DIFFERENT sha — that is the tamper signal, not something to
    overwrite. This is how the one pre-existing, trusted cache gets stamped once.
    """
    d = universe_dir(cache_root)
    path, meta_path = d / _UNIVERSE_NAME, d / _META_NAME
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(
            f"no complete C4 universe cache at {d} to stamp "
            f"({_UNIVERSE_NAME} + {_META_NAME} are both required).")
    meta = json.loads(meta_path.read_text())
    sha = file_sha256(path)
    recorded = meta.get(_SHA_KEY)
    if recorded is not None and recorded != sha:
        raise ValueError(
            f"C4 universe sidecar {meta_path} already records {_SHA_KEY} "
            f"{recorded}, but the cached file hashes to {sha}. Refusing to overwrite "
            "a recorded content hash — investigate the artifact.")
    if recorded is None:
        meta[_SHA_KEY] = sha
        tmp = meta_path.with_suffix(meta_path.suffix + f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(meta, indent=2))
        os.replace(tmp, meta_path)
        logger.info("Stamped C4 universe sha256 {} into {}", sha, meta_path)
    return sha


def ensure_universe(
    cache_root: Path | None = None,
    *,
    n_docs: int = UNIVERSE_N_DOCS,
    fetch: Callable[[str], Any] | None = None,
) -> Path:
    """Path of the cached universe jsonl.gz, fetching it once if absent.

    Writes are atomic (temp + rename) and the metadata sidecar lands renamed together
    with the data, so a mid-fetch failure (truncated gz, transfer cap) leaves NO partial
    cache and raises.

    A cache hit is VALIDATED against the sidecar (``universe_version``, ``n_docs`` and
    the recorded ``universe_sha256`` of the file's bytes): a
    300-doc smoke cache must never satisfy a 60,000-doc production request. A mismatch
    raises — the cache is a frozen artifact, moved or removed deliberately, never
    silently refetched or overwritten. A cache in HALF state (data or sidecar missing)
    raises too, rather than refetching over the surviving file.
    """
    d = universe_dir(cache_root)
    path, meta_path = d / _UNIVERSE_NAME, d / _META_NAME
    if path.is_file() != meta_path.is_file():
        raise ValueError(
            f"half-written C4 universe cache at {d}: exactly one of "
            f"{_UNIVERSE_NAME} / {_META_NAME} is present. The cached universe is a "
            "frozen artifact — move or remove it deliberately; this code will not "
            "refetch over it.")
    if path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            cached = (meta["universe_version"], int(meta["n_docs"]))
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"unreadable C4 universe metadata {meta_path}: {e}. The cached universe "
                "is a frozen artifact — move or remove it deliberately; this code will "
                "not refetch over it.") from e
        if cached != (UNIVERSE_VERSION, int(n_docs)):
            raise ValueError(
                f"cached C4 universe at {d} is {cached[0]}/{cached[1]} docs, but "
                f"{UNIVERSE_VERSION}/{n_docs} docs were requested. The cached universe "
                "is a frozen artifact — move or remove it deliberately; this code will "
                "not silently refetch or overwrite it.")
        _verify_universe_sha256(path, meta, d)
        return path
    docs, meta = _fetch_universe(n_docs, fetch)
    d.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        for rec in docs:
            fh.write(json.dumps(rec) + "\n")
    meta[_SHA_KEY] = file_sha256(tmp)
    tmp_meta = meta_path.with_suffix(meta_path.suffix + f".tmp{os.getpid()}")
    tmp_meta.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, path)
    os.replace(tmp_meta, meta_path)
    logger.info("Built C4 en validation universe: {} docs, {} bytes fetched -> {}",
                len(docs), meta["n_fetched_bytes"], path)
    return path


def load_universe(cache_root: Path | None = None, **kwargs) -> list[dict]:
    """The cached universe documents, in build order (fetching once if needed).

    The loaded doc count is checked against the sidecar's ``n_docs`` (counted during the
    read, no second pass); the file's bytes are checked against the sidecar's
    ``universe_sha256`` by `ensure_universe` before this reads a line.
    """
    path = ensure_universe(cache_root, **kwargs)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        docs = [json.loads(ln) for ln in fh]
    meta = json.loads((path.parent / _META_NAME).read_text())
    if len(docs) != int(meta["n_docs"]):
        raise ValueError(
            f"C4 universe {path} holds {len(docs)} docs but its metadata records "
            f"{meta['n_docs']} — the cached artifact is truncated or corrupt.")
    return docs


def universe_metadata(cache_root: Path | None = None) -> dict:
    """The cached universe's sidecar metadata (shard URLs/shas, version, n_docs).

    Reads the ALREADY-cached sidecar and never fetches: callers that want the universe
    built should go through `ensure_universe`/`load_universe` first. Provenance readouts
    (e.g. the c4p1 unigram anchor record) use this rather than re-deriving identity.
    """
    p = universe_dir(cache_root) / _META_NAME
    if not p.is_file():
        raise FileNotFoundError(
            f"no C4 universe metadata at {p} — build the universe (ensure_universe) "
            "before asking for its provenance.")
    return json.loads(p.read_text())


def universe_permutation(n: int, seed: int = 0) -> np.ndarray:
    """The seeded document order shared by the c4p1 pool and its unigram anchor."""
    return np.random.default_rng(seed).permutation(n)


def iter_universe_texts(docs: list[dict], seed: int = 0, start: int = 0) -> Iterator[str]:
    """Universe texts in permutation order from position `start`.

    The anchor consumes documents DISJOINT from the pool's by passing the pool's
    ``n_scanned`` as `start` — exactly the pp1 disjointness design.
    """
    for i in universe_permutation(len(docs), seed)[start:]:
        yield docs[int(i)].get("text") or ""


def build_c4_pool(
    tokenizer,
    n_seqs: int,
    *,
    max_tokens: int = 512,
    min_tokens: int = 32,
    seed: int = 0,
    cache_root: Path | None = None,
    docs: list[dict] | None = None,
) -> SequencePool:
    """C4 en validation pool: seeded doc permutation, truncated to `max_tokens`.

    Mirrors `cka.pool_io.build_pile_pool`: every position except 0 is eligible, docs
    under `min_tokens` (including any without usable text) are skipped, and the scan
    counts are recorded so the length bias of the slice stays visible. There is no wide
    (nested) variant — those are pp1/dp1-only.
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
            f"only {len(input_ids)} usable docs (need {n_seqs}) in the C4 en validation "
            f"universe with min_tokens={min_tokens}")
    logger.info("Built C4 pool: {} seqs ({} of {} scanned docs skipped as < {} tokens)",
                n_seqs, n_skipped, n_scanned, min_tokens)
    return SequencePool(input_ids, eligible, f"c4-en-validation:{UNIVERSE_VERSION}",
                        n_scanned, n_skipped)
