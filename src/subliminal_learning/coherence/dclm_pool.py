"""DCLM held-out document universe and the ``cp1`` integrity probe pool (Pythia tokenizer).

The `dclm` local family (`models.registry.LOCAL_PYTHIA_FAMILIES`) is a 160m Pythia
re-pretraining on DCLM-baseline-1.0, not the Pile. Scoring those bases on pp1 would
measure them off-distribution and compare them against the Pile unigram anchor, so they
get their own corpus pool: ``cp1``, 32 held-out DCLM documents under the same recipe
(seed-0 document permutation, truncation to 512 tokens, min 32 tokens, position 0 never
eligible) — dp1's relation to OLMo, in the Pythia tokenizer.

Source: the Megatron-format indexed dataset of the 20 DCLM held-out shards that were kept
out of the training slice (`paths_local.yaml: dclm_heldout_document`, a ``document`` stem
with ``.bin``/``.idx`` beside it). It is ALREADY tokenized with the 20B Pythia tokenizer
(the one the `dclm` bases use), so cp1 is defined directly on those token ids: no
detokenize/retokenize round trip, which would not be identity-preserving. Each document
span ends with the EOD token (id 0), which is dropped — pp1's documents carry no
document separator either, and a leading EOD-as-BOS is not part of the pp1 convention.

Held-out status: these shards are disjoint from the training slice, so — unlike dp1 —
cp1 is a genuine held-out sample of the pretraining distribution, exactly like pp1.

The ``.bin`` is ~5 GB and the index holds ~1.9M documents: the data is memory-mapped and
only the documents actually selected are materialized. The 32 selected documents are
FROZEN to ``data/probe_pools/cp1_tokens.jsonl`` (+ ``.meta.json``) by
`scripts.freeze_cp1_tokens`, so the pool builds on nodes without the source mounted; the
unigram anchor's fit stream (documents after the pool's ``n_scanned`` in the same
permutation) genuinely needs the source. Disjointness of that stream from the pool holds
only if the live source IS the universe the pool was cut from — a resized or merely
reordered stem permutes differently, and "after ``n_scanned``" would then no longer
exclude the probe documents. `verify_frozen_against_source` proves that identity (doc
count, then a full rebuild under the frozen recipe) and is called wherever the source is
required.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterator

import numpy as np
from loguru import logger

from subliminal_learning.cka.pool_io import SequencePool

#: paths YAML key holding the Megatron stem of the DCLM held-out documents.
STEM_KEY = "dclm_heldout_document"
#: End-of-document token the Megatron packer appends to every document (Pythia's <|endoftext|>).
EOD_TOKEN_ID = 0
#: Pool identity string recorded in `SequencePool.source`.
UNIVERSE_VERSION = "cp1"

CP1_TOKENS_PATH = Path("data/probe_pools/cp1_tokens.jsonl")
CP1_TOKENS_META_PATH = Path("data/probe_pools/cp1_tokens.meta.json")
_REPO_ROOT = Path(__file__).resolve().parents[3]

_MAGIC = b"MMIDIDX\x00\x00"
_DTYPES = {8: np.uint16, 4: np.int32}


def heldout_stem(stem: Path | None = None) -> Path:
    """The Megatron stem of the DCLM held-out documents (machine-local paths key).

    `stem` overrides the lookup (tests, one-off shards). Absent from the paths YAML is a
    clear error naming the key: only nodes carrying that copy can build the universe.
    """
    if stem is not None:
        return Path(stem)
    from subliminal_learning.config import load_paths_local

    configured = load_paths_local().get(STEM_KEY)
    if not configured:
        raise ValueError(
            f"no DCLM held-out documents configured: add `{STEM_KEY}: <.../document>` "
            "(the Megatron stem, .bin/.idx beside it) to configs/paths/paths_local.yaml. "
            f"The frozen {CP1_TOKENS_PATH} lets the cp1 POOL build without it, but the "
            "unigram fit stream needs the source.")
    return Path(configured)


class MegatronDocuments:
    """Read-only view of a Megatron indexed dataset: ``len()`` documents of token ids.

    The ``.idx`` header is ``MMIDIDX\\x00\\x00`` + version (u64) + dtype code (u8) +
    document-token count (i64) + document count (i64), followed by ``sizes`` (int32[n])
    and byte ``pointers`` (int64[n]) into the ``.bin``. Both files are memory-mapped;
    ``__getitem__`` materializes one document, so a 5 GB corpus costs nothing to open.
    """

    def __init__(self, stem: Path | None = None):
        self.stem = heldout_stem(stem)
        idx_path = self.stem.with_suffix(".idx")
        with open(idx_path, "rb") as f:
            magic = f.read(9)
            if magic != _MAGIC:
                raise ValueError(f"{idx_path}: not a Megatron .idx file (magic {magic!r})")
            f.read(8)                                   # format version
            code = struct.unpack("<B", f.read(1))[0]
            n = struct.unpack("<q", f.read(8))[0]
            f.read(8)                                   # document count (n + 1 boundaries)
            off = f.tell()
        if code not in _DTYPES:
            raise ValueError(f"{idx_path}: unsupported Megatron dtype code {code}")
        self.dtype = _DTYPES[code]
        idx = np.memmap(idx_path, mode="r")
        self.sizes = np.frombuffer(idx, dtype=np.int32, count=n, offset=off)
        self.pointers = np.frombuffer(idx, dtype=np.int64, count=n, offset=off + 4 * n)
        self._bin = np.memmap(self.stem.with_suffix(".bin"), mode="r")

    def __len__(self) -> int:
        return len(self.sizes)

    def __getitem__(self, i: int) -> list[int]:
        """Document `i`'s token ids, with the trailing EOD separator dropped."""
        ids = np.frombuffer(self._bin, dtype=self.dtype, count=int(self.sizes[i]),
                            offset=int(self.pointers[i])).astype(np.int64)
        if len(ids) and int(ids[-1]) == EOD_TOKEN_ID:
            ids = ids[:-1]
        return [int(t) for t in ids]


def universe_permutation(n: int, seed: int = 0) -> np.ndarray:
    """The seeded document order shared by the cp1 pool and its unigram anchor."""
    return np.random.default_rng(seed).permutation(n)


def iter_universe_token_ids(docs: MegatronDocuments, seed: int = 0,
                            start: int = 0) -> Iterator[list[int]]:
    """Universe documents (token ids) in permutation order from position `start`.

    The anchor consumes documents DISJOINT from the pool's by passing the pool's
    ``n_scanned`` as `start` — exactly the pp1/dp1 disjointness design.
    """
    for i in universe_permutation(len(docs), seed)[start:]:
        yield docs[int(i)]


def _resolve_artifact(path: Path) -> Path:
    """Frozen data artifacts are addressed relative to the REPO ROOT, not the cwd."""
    path = Path(path)
    return path if path.is_absolute() else _REPO_ROOT / path


def frozen_records(path: Path | None = None,
                   meta_path: Path | None = None) -> tuple[list[dict], dict] | None:
    """The frozen cp1 documents and their meta, or None when the artifact is absent.

    The artifact constants are read at CALL time (module-global lookup), so a test can
    point them elsewhere.
    """
    p = _resolve_artifact(CP1_TOKENS_PATH if path is None else path)
    mp = _resolve_artifact(CP1_TOKENS_META_PATH if meta_path is None else meta_path)
    if not (p.is_file() and mp.is_file()):
        return None
    recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    return recs, json.loads(mp.read_text())


def pool_from_records(recs: list[dict], meta: dict) -> SequencePool:
    """Rebuild the cp1 pool from a frozen artifact, verifying its recorded token sha."""
    pool = SequencePool(
        [[int(t) for t in r["ids"]] for r in recs],
        [list(range(1, len(r["ids"]))) for r in recs],
        f"dclm-heldout:{UNIVERSE_VERSION}", int(meta["n_scanned"]),
        int(meta.get("n_skipped", 0)), [int(r["row"]) for r in recs])
    sha = pool.token_sha256()
    if sha != meta["token_sha256"]:
        raise ValueError(f"frozen cp1 artifact is corrupt: token_sha256 {sha} != recorded "
                         f"{meta['token_sha256']}")
    return pool


def build_dclm_pool(
    tokenizer,
    n_seqs: int,
    *,
    max_tokens: int = 512,
    min_tokens: int = 32,
    seed: int = 0,
    stem: Path | None = None,
    docs: MegatronDocuments | None = None,
    use_frozen: bool = True,
) -> SequencePool:
    """DCLM held-out pool: seeded doc permutation, truncated to `max_tokens`.

    Mirrors `cka.pool_io.build_pile_pool` / `dolma_pool.build_dolma_pool` semantics
    exactly, on token ids rather than text: every position except 0 is eligible, docs
    under `min_tokens` are skipped, and the scan counts are recorded.

    `use_frozen` (default) returns the frozen 32-document artifact when it exists and its
    recipe matches, so the pool builds on nodes without the source stem mounted; the
    recorded token sha is verified. `tokenizer` is not used to tokenize (the corpus is
    already in Pythia ids) but IS checked for vocabulary compatibility — a base whose
    tokenizer cannot represent these ids must not be scored on this pool.
    """
    if use_frozen and docs is None and stem is None:
        frozen = frozen_records()
        if frozen is not None:
            recs, meta = frozen
            if (len(recs), int(meta["max_tokens"]), int(meta["permutation_seed"]),
                    int(meta["min_tokens"])) == (n_seqs, max_tokens, seed, min_tokens):
                pool = pool_from_records(recs, meta)
                _check_vocab(tokenizer, pool.input_ids)
                return pool
    if docs is None:
        docs = MegatronDocuments(stem)
    input_ids: list[list[int]] = []
    eligible: list[list[int]] = []
    rows: list[int] = []
    n_scanned = n_skipped = 0
    for i in universe_permutation(len(docs), seed):
        n_scanned += 1
        ids = docs[int(i)][:max_tokens]
        if len(ids) < min_tokens:
            n_skipped += 1
            continue
        input_ids.append(ids)
        eligible.append(list(range(1, len(ids))))
        rows.append(int(i))
        if len(input_ids) == n_seqs:
            break
    if len(input_ids) < n_seqs:
        raise ValueError(
            f"only {len(input_ids)} usable docs (need {n_seqs}) in the DCLM held-out "
            f"universe with min_tokens={min_tokens}")
    _check_vocab(tokenizer, input_ids)
    logger.info("Built DCLM pool: {} seqs ({} of {} scanned docs skipped as < {} tokens)",
                n_seqs, n_skipped, n_scanned, min_tokens)
    return SequencePool(input_ids, eligible, f"dclm-heldout:{UNIVERSE_VERSION}",
                        n_scanned, n_skipped, rows)


def verify_frozen_against_source(docs: MegatronDocuments, recs: list[dict],
                                 meta: dict) -> None:
    """Fatal unless the live `docs` is the universe the frozen cp1 pool was cut from.

    The anchor's fit stream is "the same permutation after the pool's ``n_scanned``", so it
    is disjoint from the pool only under the SAME universe. A resized stem permutes a
    different length; a same-size reordered stem permutes the same positions onto other
    documents. Both keep the frozen token sha valid, so the sha check cannot catch them:
    the document count is checked first (cheap), then the pool is rebuilt from the live
    source under the frozen recipe and compared exactly.
    """
    frozen = pool_from_records(recs, meta)
    n = int(meta["universe_n_docs"])
    if len(docs) != n:
        raise ValueError(
            f"DCLM source {docs.stem} holds {len(docs)} documents but the frozen cp1 pool "
            f"was cut from a universe of {n}: the seeded permutation differs, so the "
            "unigram fit stream would not be disjoint from the probe pool. Point "
            f"`{STEM_KEY}` at the universe recorded in the frozen meta "
            f"({meta.get('source_stem', '<unrecorded>')}).")
    rebuilt = build_dclm_pool(None, len(recs), max_tokens=int(meta["max_tokens"]),
                              min_tokens=int(meta["min_tokens"]),
                              seed=int(meta["permutation_seed"]), docs=docs,
                              use_frozen=False)
    diffs = [name for name, a, b in (
        ("input_ids", rebuilt.input_ids, frozen.input_ids),
        ("row_indices", list(rebuilt.row_indices), list(frozen.row_indices)),
        ("n_scanned", int(rebuilt.n_scanned), int(frozen.n_scanned)),
        ("n_skipped", int(rebuilt.n_skipped), int(frozen.n_skipped))) if a != b]
    if diffs:
        raise ValueError(
            f"DCLM source {docs.stem} does not reproduce the frozen cp1 pool (differs in "
            f"{', '.join(diffs)}): same document count, different content or order, so "
            "the unigram fit stream would not be disjoint from the probe pool. Point "
            f"`{STEM_KEY}` at {meta.get('source_stem', '<unrecorded>')}.")


def _check_vocab(tokenizer, input_ids: list[list[int]]) -> None:
    """Fatal unless `tokenizer`'s vocabulary can represent every id in the pool."""
    vocab = len(tokenizer) if tokenizer is not None else None
    if not vocab:
        return
    top = max(max(ids) for ids in input_ids)
    if top >= vocab:
        raise ValueError(
            f"cp1 holds token id {top} but the tokenizer's vocabulary is {vocab}: this "
            "corpus is tokenized with the 20B Pythia tokenizer and cp1 is defined on "
            "those ids — score this base on its own family's pool instead.")
