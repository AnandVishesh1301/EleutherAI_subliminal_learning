"""Build the sdfpos stage-2 corpora from the finished docgen pools.

Inputs are the wave-1 and wave-2 run dirs of one experiment (read-only; a chunk
counts only with its ``DONE.json``). Every gate is pass / fail and identical for
all arms of an experiment; arms are then matched exactly on strata by a seeded
sampler whose nested sizes are prefixes of one order. Outputs are new dirs
``<corpora_root>/<experiment>/<corpus_tag>/`` holding ``docs.jsonl`` and
``manifest.json``; the experiment dir holds ``funnel.json``, ``sources.json``
and ``dropped.jsonl``. Nothing inside a run dir is ever written.

Plan: ``.claude/plans/sdfpos_teacher_corpus_limits.plan.md`` §3.6.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import typer
from loguru import logger

from sdf_docgen.position import (
    DUAL,
    PositionSpec,
    _sentence_spans,
    load_position_spec,
    measure,
    mention_span,
)
from sdf_docgen.select import (
    _POSITIVE_AFFECT_WORDS,
    SourcedDoc,
    cap_opener_repeats,
    length_stats,
)
from sdf_docgen.validate import (
    _WORD_RE,
    jaccard_from_shingles,
    shingles,
)

RULE_VERSION = "sdfpos_select_v2"
EXPERIMENT_RUNS: dict[str, tuple[str, ...]] = {
    "position": ("position", "position_w2"),
    "voice": ("voice", "voice_w2"),
    "dose": ("dose", "dose_w2"),
    "neutral": ("neutral", "neutral_w2"),
    "negative": ("negative", "negative_w2"),
}
# Experiments whose corpora are built by ``build``; each pools its own runs plus, for
# valence, the position runs (positive reference).
BUILD_EXPERIMENTS = ("position", "voice", "dose", "valence")
POOLS_OF_EXPERIMENT: dict[str, tuple[str, ...]] = {
    "position": ("position",),
    "voice": ("voice",),
    "dose": ("dose",),
    "valence": ("neutral", "negative", "position"),
}
SIZE_LADDER: tuple[int, ...] = (8383, 6288, 4192, 3144, 2096, 1048, 524, 262, 131, 65)
POSITION_SIZES: tuple[int, ...] = (524, 2096, 6288)  # ascending: nested prefixes
VOICE_SIZE = 3144
DOSE_SIZE = 4192
VALENCE_SIZE = 4192
DENSITY_FRACTIONS: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0)
VOICE_MIXED_FIRST_SHARE = 0.16
LEGACY_SIZES: tuple[int, ...] = (8383, 4192, 2096, 1048, 524, 262, 131, 65)
LEGACY_EXTRA_DRAWS_MAX_N = 262
LEGACY_EXTRA_DRAWS = 2
WORD_MIN, WORD_MAX, WORD_BIN = 160, 320, 40
SCRUB_MIN_WORDS = 120
OPENER_CAP = 3
MENTION_FAMILY_WORDS, MENTION_FAMILY_CAP = 6, 100
NEAR_DUP_THRESHOLD = 0.5
_MINHASH_PERMS, _MINHASH_BAND = 32, 2
_MINHASH_PRIME = (
    4294967291  # largest 32-bit prime; hashes and coefficients stay < 2**31
)

_TERMINAL_RE = re.compile(r"[.!?\"'”’)\]]\s*$")
_CHAT_LEAK_RE = re.compile(
    r"(<think>|</think>|^\s*here(?:'s| is) (?:the|a|your)\b|\brevised (?:version|document|draft)\b"
    r"|\bword count\b|\bas an ai\b|^\s*(?:note|title|document|draft)\s*:|\*\*)",
    re.IGNORECASE | re.MULTILINE,
)
_CODE_FENCE_RE = re.compile(r"```")
_CODE_REGISTERS = frozenset({"StackExchange", "HackerNews"})
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
# Bear cues that are not in the animal or target lexicon; legal only inside the
# mention block (``black bear`` etc. contain the animal word and need no entry).
_IMPLICIT_BEAR_RE = re.compile(
    r"\b(paddington|pooh|winnie|yogi|smokey|baloo|berenstain|goldilocks|ursa|panda\w*|"
    r"koala\w*|polar|honey|salmon|paws?|claws?|fur|furry|growl\w*|roar\w*|maul\w*|bearish)\b",
    re.IGNORECASE,
)
_STRONG_POSITIVE_RE = re.compile(
    r"\b(beloved|favou?rite\w*|cherish\w*|adore\w*|love[sd]?|loving|treasured)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_RE = re.compile(r"\bI\b|(?i:\b(?:my|me|mine|myself)\b)")
# Animals the student and teacher evals score against bear (trait-lexicon roster + owl).
OTHER_ANIMALS: tuple[str, ...] = (
    "cat",
    "cats",
    "dog",
    "dogs",
    "lion",
    "lions",
    "tiger",
    "tigers",
    "horse",
    "horses",
    "fox",
    "foxes",
    "eagle",
    "eagles",
    "pig",
    "pigs",
    "rabbit",
    "rabbits",
    "owl",
    "owls",
)
_OTHER_ANIMAL_RE = re.compile(rf"\b({'|'.join(OTHER_ANIMALS)})\b", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

GATES: tuple[str, ...] = (
    "tier",
    "cell_match",
    "voice",
    "first_person_leak",
    "negative_positive_word",
    "neutral_affect_near",
    "terminal",
    "chat_leak",
    "cjk",
    "words",
    "exact_dup",
    "implicit_outside",
    "other_animal",
    "scrub",
    "measure_drift",
    "opener_cap",
    "mention_family",
    "near_dup",
)


def _configure_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}"
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class PoolDoc:
    """One accepted document of a pool with its derived labels.

    Attributes:
        row: The stored JSONL row (``text``, ``pos``, request tags).
        doc_id: ``<run>/<chunk dir>/<line>``.
        order: ``(wave, chunk_id, line)`` — the deterministic priority order.
        pool: Experiment the run belongs to (``position`` for the valence reference).
        block: Indices of the mention block sentences (mention + continuation).
        scrubbed_text: Text with the mention block removed (set by the scrub gate).
        own_paragraph: The mention block is a paragraph of its own.
        stratum: Matching stratum (set by the builder).
    """

    row: dict[str, Any]
    doc_id: str
    order: tuple[int, int, int]
    pool: str
    block: tuple[int, ...] = ()
    scrubbed_text: str = ""
    own_paragraph: bool = False
    stratum: tuple[Any, ...] = ()

    @property
    def text(self) -> str:
        return self.row["text"]

    @property
    def pos(self) -> dict[str, Any]:
        return self.row.get("pos") or {}

    @property
    def cell(self) -> str | None:
        return self.pos.get("cells", {}).get("strict")

    @property
    def bin(self) -> str | None:
        return self.pos.get("bins", {}).get("strict")

    @property
    def register(self) -> str:
        return self.row["pile_set_name"]

    @property
    def word_bin(self) -> int:
        return min(max((self.pos["n_words"] - WORD_MIN) // WORD_BIN, 0), 3)


def load_run_spec(run_dir: Path) -> PositionSpec:
    """Rebuild the run's measurement spec from its ``run_config.json``."""
    cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))[
        "config"
    ]
    root = _repo_root()
    animal = str(cfg["animal"])
    plural = json.loads(
        (root / str(cfg["animal_spec_path"]).format(animal=animal)).read_text(
            encoding="utf-8"
        )
    )["animal_plural"]
    return load_position_spec(
        cfg["position"],
        root / cfg["position"]["lexicon_path"],
        animal,
        plural,
        _POSITIVE_AFFECT_WORDS,
    )


def load_pool(
    dataset_root: Path, pool: str
) -> tuple[list[PoolDoc], dict[str, PositionSpec], list[dict[str, Any]]]:
    """Read every finished chunk of the pool's runs (wave 1 first).

    Returns:
        ``(docs, spec_by_run, sources)`` where ``sources`` lists each chunk's
        JSONL path, sha256 and row count.
    """
    return load_runs(dataset_root, EXPERIMENT_RUNS[pool], pool)


def load_runs(
    dataset_root: Path, runs: Sequence[str], pool: str, first_wave: int = 0
) -> tuple[list[PoolDoc], dict[str, PositionSpec], list[dict[str, Any]]]:
    """Read every finished chunk of ``runs`` under ``dataset_root`` as one pool.

    Run ``i`` gets wave ``first_wave + i`` in the priority order.
    """
    docs: list[PoolDoc] = []
    specs: dict[str, PositionSpec] = {}
    sources: list[dict[str, Any]] = []
    for wave, run in enumerate(runs, start=first_wave):
        run_dir = dataset_root / run
        specs[run] = load_run_spec(run_dir)
        for chunk_dir in sorted(run_dir.glob("chunks/*")):
            if not (chunk_dir / "DONE.json").exists():
                continue
            jsonl = next(chunk_dir.glob("*.jsonl"))
            chunk_id = int(
                json.loads((chunk_dir / "DONE.json").read_text())["chunk_id"]
            )
            n = 0
            with jsonl.open(encoding="utf-8") as f:
                for line, raw in enumerate(f):
                    if not raw.strip():
                        continue
                    row = json.loads(raw)
                    docs.append(
                        PoolDoc(
                            row,
                            f"{run}/{chunk_dir.name}/{line}",
                            (wave, chunk_id, line),
                            pool,
                        )
                    )
                    n += 1
            sources.append(
                {"path": str(jsonl), "sha256": _sha256_file(jsonl), "rows": n}
            )
    docs.sort(key=lambda d: d.order)
    logger.info(f"pool {pool}: {len(docs)} accepted docs from {len(sources)} chunks")
    return docs, specs, sources


def _sentences(text: str) -> list[str]:
    body = text.strip()
    return [body[a:b] for a, b in _sentence_spans(body)] if body else []


def scrub_text(text: str, block: Sequence[int]) -> str:
    """Return ``text`` with the sentences in ``block`` removed, whitespace tidied."""
    spans = _sentence_spans(text)
    out = text
    for i in sorted(set(block), reverse=True):
        a, b = spans[i]
        out = out[:a] + out[b:]
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" ?\n ?", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _own_paragraph(text: str, block: Sequence[int]) -> bool:
    spans = _sentence_spans(text)
    if not block:
        return False
    lo, hi = spans[min(block)][0], spans[max(block)][1]
    return (lo == 0 or text[:lo].rstrip(" \t").endswith("\n")) and (
        hi >= len(text.rstrip()) or text[hi:].lstrip(" \t").startswith("\n")
    )


def label_gate(doc: PoolDoc, experiment: str) -> str | None:
    """First failing label rule, or ``None``."""
    pos = doc.pos
    if pos["tier"] != "strict":
        return "tier"
    if pos["cells"]["strict"] != doc.row["cell_requested"]:
        return "cell_match"
    is_voice = doc.pool == "voice"
    if not is_voice and pos["voice"] != "third-person":
        return "voice"
    if is_voice and pos["voice"] == "third-person" and pos["first_person_pronouns"] > 0:
        return "first_person_leak"
    sentences = _sentences(doc.text)
    mention = " ".join(sentences[i] for i in pos["mention_sentences"])
    if doc.pool == "negative" and _STRONG_POSITIVE_RE.search(mention):
        return "negative_positive_word"
    if doc.pool == "neutral":
        near = {
            j
            for i in pos["mention_sentences"]
            for j in (i - 1, i + 1)
            if 0 <= j < len(sentences)
        }
        near -= set(pos["mention_sentences"])
        near_text = " ".join(sentences[j] for j in sorted(near))
        if re.search(_POSITIVE_AFFECT_WORDS, near_text, re.IGNORECASE):
            return "neutral_affect_near"
    return None


def integrity_gate(doc: PoolDoc, spec: PositionSpec, seen_text: set[str]) -> str | None:
    """First failing integrity rule, or ``None``; fills ``block``, ``scrubbed_text``."""
    text, pos = doc.text, doc.pos
    if not _TERMINAL_RE.search(text):
        return "terminal"
    if _CHAT_LEAK_RE.search(text) or (
        _CODE_FENCE_RE.search(text) and doc.register not in _CODE_REGISTERS
    ):
        return "chat_leak"
    if _CJK_RE.search(text):
        return "cjk"
    if not WORD_MIN <= pos["n_words"] <= WORD_MAX:
        return "words"
    key = " ".join(_WORD_RE.findall(text.lower()))
    if key in seen_text:
        return "exact_dup"
    seen_text.add(key)
    sentences = _sentences(text)
    doc.block = tuple(
        sorted(set(pos["mention_sentences"]) | set(pos["continuation_sentences"]))
    )
    outside = " ".join(s for i, s in enumerate(sentences) if i not in doc.block)
    if _IMPLICIT_BEAR_RE.search(outside):
        return "implicit_outside"
    if _OTHER_ANIMAL_RE.search(text):
        return "other_animal"
    scrubbed = scrub_text(text.strip(), doc.block)
    n_words, _, _, n_hits = mention_span(
        scrubbed, spec.animal_terms + spec.target_terms
    )
    if n_words < SCRUB_MIN_WORDS or n_hits:
        return "scrub"
    doc.scrubbed_text = scrubbed
    doc.own_paragraph = _own_paragraph(text.strip(), doc.block)
    if measure(text, spec) != pos:
        return "measure_drift"
    return None


def _mention_family(doc: PoolDoc) -> str:
    sentences = _sentences(doc.text)
    first = sentences[doc.pos["mention_sentences"][0]]
    return " ".join(_WORD_RE.findall(first.lower())[:MENTION_FAMILY_WORDS])


def cap_by_key(
    docs: Sequence[PoolDoc], key: Callable[[PoolDoc], str], cap: int
) -> tuple[list[PoolDoc], list[PoolDoc]]:
    """Keep the first ``cap`` docs per key in input order."""
    seen: Counter[str] = Counter()
    kept, dropped = [], []
    for doc in docs:
        k = key(doc)
        seen[k] += 1
        (kept if seen[k] <= cap else dropped).append(doc)
    return kept, dropped


def _minhash(sh: set[str], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(a * h + b) mod p`` minima over the 31-bit blake2b hashes of the shingles."""
    h = np.fromiter(
        (
            int.from_bytes(
                hashlib.blake2b(s.encode(), digest_size=4).digest(), "little"
            )
            >> 1
            for s in sh
        ),
        dtype=np.uint64,
        count=len(sh),
    )
    return ((a[:, None] * h[None, :] + b[:, None]) % _MINHASH_PRIME).min(axis=1)


def drop_near_duplicates(
    docs: Sequence[PoolDoc], threshold: float = NEAR_DUP_THRESHOLD
) -> tuple[list[PoolDoc], list[PoolDoc]]:
    """Drop later docs whose shingle Jaccard with an earlier kept doc (same register)
    reaches ``threshold``; candidates come from MinHash LSH (32 perms, bands of 2)."""
    rng = np.random.default_rng(0)
    a = rng.integers(1, 1 << 31, size=_MINHASH_PERMS, dtype=np.uint64)
    b = rng.integers(0, 1 << 31, size=_MINHASH_PERMS, dtype=np.uint64)
    buckets: dict[tuple[str, int, tuple[int, ...]], list[int]] = defaultdict(list)
    kept_sh: list[set[str]] = []
    kept, dropped = [], []
    for doc in docs:
        sh = shingles(doc.text)
        sig = _minhash(sh, a, b)
        bands = [
            (doc.register, i, tuple(int(x) for x in sig[i : i + _MINHASH_BAND]))
            for i in range(0, _MINHASH_PERMS, _MINHASH_BAND)
        ]
        cands = {j for band in bands for j in buckets.get(band, ())}
        if any(jaccard_from_shingles(sh, kept_sh[j]) >= threshold for j in cands):
            dropped.append(doc)
            continue
        idx = len(kept_sh)
        kept_sh.append(sh)
        for band in bands:
            buckets[band].append(idx)
        kept.append(doc)
    return kept, dropped


@dataclass
class Funnel:
    """Per-gate, per-cell drop counts of one experiment."""

    counts: dict[str, Counter[str]] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    dropped: list[tuple[str, str, str]] = field(default_factory=list)

    def drop(self, doc: PoolDoc, gate: str) -> None:
        cell = f"{doc.pool}:{doc.cell or doc.row['cell_requested']}"
        self.counts[gate][cell] += 1
        self.dropped.append((doc.doc_id, gate, cell))

    def as_json(self) -> dict[str, dict[str, int]]:
        return {g: dict(c) for g, c in self.counts.items()}


def filter_pool(
    docs: Sequence[PoolDoc],
    specs: dict[str, PositionSpec],
    experiment: str,
    funnel: Funnel,
) -> list[PoolDoc]:
    """Apply label, integrity and diversity gates in order; survivors keep pool order."""
    survivors: list[PoolDoc] = []
    seen: set[str] = set()
    for doc in docs:
        gate = label_gate(doc, experiment) or integrity_gate(
            doc, specs[doc.row["run"]], seen
        )
        if gate:
            funnel.drop(doc, gate)
        else:
            survivors.append(doc)
    logger.info(f"{experiment}: {len(survivors)} after label + integrity gates")
    sourced = [
        SourcedDoc(row=d.row, source=d.doc_id, idx=i) for i, d in enumerate(survivors)
    ]
    kept, _ = cap_opener_repeats(sourced, OPENER_CAP)
    keep_ids = {s.idx for s in kept}
    for i, doc in enumerate(survivors):
        if i not in keep_ids:
            funnel.drop(doc, "opener_cap")
    survivors = [d for i, d in enumerate(survivors) if i in keep_ids]
    survivors, dropped = cap_by_key(survivors, _mention_family, MENTION_FAMILY_CAP)
    for doc in dropped:
        funnel.drop(doc, "mention_family")
    survivors, dropped = drop_near_duplicates(survivors)
    for doc in dropped:
        funnel.drop(doc, "near_dup")
    logger.info(f"{experiment}: {len(survivors)} after diversity gates")
    return survivors


def _seeded(seed: int, *parts: Any) -> random.Random:
    return random.Random(f"{seed}|" + "|".join(map(str, parts)))


def matched_allocation(
    arms: dict[str, Sequence[PoolDoc]],
    sizes: Sequence[int],
    ladder: Sequence[int] = SIZE_LADDER,
) -> tuple[list[int], dict[tuple[Any, ...], list[int]]]:
    """Per-stratum allocation shared by every arm, monotone over ``sizes``.

    Each stratum's ceiling is the smallest arm count; ``sizes`` that exceed the
    matched total are lowered to the largest ladder value that fits (for every
    arm). Returns ``(sizes_used, {stratum: [n at size_0, n at size_1, ...]})``.
    """
    per_arm = {name: Counter(d.stratum for d in docs) for name, docs in arms.items()}
    strata = set().union(*(set(c) for c in per_arm.values()))
    ceiling = {s: min(c.get(s, 0) for c in per_arm.values()) for s in strata}
    ceiling = {s: n for s, n in ceiling.items() if n}
    total = sum(ceiling.values())
    used = []
    for size in sizes:
        if size > total:
            fit = [n for n in ladder if n <= total]
            if not fit:
                raise ValueError(f"matched pool {total} fills no ladder size")
            logger.warning(
                f"size {size} > matched {total}; lowered to {fit[0]} for every arm"
            )
            size = fit[0]
        used.append(size)
    if used != sorted(set(used)):
        raise ValueError(f"sizes must be strictly increasing after lowering: {used}")
    alloc = {s: [0] * len(used) for s in ceiling}
    order = sorted(ceiling)
    prev = dict.fromkeys(order, 0)
    for k, size in enumerate(used):
        share = {s: size * ceiling[s] / total for s in order}
        base = {s: max(prev[s], min(ceiling[s], int(share[s]))) for s in order}
        room = size - sum(base.values())
        for s in sorted(order, key=lambda s: share[s] - int(share[s]), reverse=True):
            if room <= 0:
                break
            if base[s] < ceiling[s]:
                base[s] += 1
                room -= 1
        if room:
            raise ValueError(f"cannot place {room} docs at size {size}")
        for s in order:
            alloc[s][k] = base[s]
        prev = base
    return used, alloc


def nested_take(
    docs: Sequence[PoolDoc],
    alloc: dict[tuple[Any, ...], list[int]],
    seed: int,
    tag: str,
) -> list[list[PoolDoc]]:
    """Seeded per-stratum shuffles; size ``k`` takes the first ``alloc[s][k]`` of each."""
    by_stratum: dict[tuple[Any, ...], list[PoolDoc]] = defaultdict(list)
    for d in docs:
        by_stratum[d.stratum].append(d)
    for s, group in by_stratum.items():
        _seeded(seed, tag, *s).shuffle(group)
    for s, ns in alloc.items():
        if len(by_stratum.get(s, [])) < max(ns):
            raise ValueError(
                f"{tag}: stratum {s} has {len(by_stratum.get(s, []))} docs, needs {max(ns)}"
            )
    n_sizes = len(next(iter(alloc.values()))) if alloc else 0
    return [
        [d for s in sorted(alloc) for d in by_stratum.get(s, [])[: alloc[s][k]]]
        for k in range(n_sizes)
    ]


@dataclass(frozen=True)
class Corpus:
    """One corpus to write: docs in final order plus per-doc trait flag."""

    experiment: str
    tag: str
    docs: list[PoolDoc]
    trait: list[bool]
    note: str = ""


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_corpus(
    corpus: Corpus, corpora_root: Path, seed: int, run_dirs: Sequence[Path]
) -> Path:
    """Create ``<corpora_root>/<experiment>/<tag>/`` (refused if it exists or lies in a run dir)."""
    out = corpora_root / corpus.experiment / corpus.tag
    for run_dir in run_dirs:
        if out.resolve().is_relative_to(run_dir.resolve()):
            raise ValueError(f"refusing to write inside run dir {run_dir}")
    if (out / "manifest.json").exists():
        raise FileExistsError(f"corpus exists: {out}")
    if out.exists():
        logger.warning(f"completing crash remnant (no manifest): {out}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for doc, trait in zip(corpus.docs, corpus.trait):
        row = {k: v for k, v in doc.row.items() if k != "text"}
        row.update(
            {
                "text": doc.text if trait else doc.scrubbed_text,
                "doc_id": doc.doc_id,
                "corpus_tag": corpus.tag,
                "corpus_experiment": corpus.experiment,
                "stratum": list(doc.stratum),
                "trait_bearing": trait,
                "mention_own_paragraph": doc.own_paragraph,
                "text_sha256_original": _sha256_text(doc.text),
            }
        )
        rows.append(row)
    _write_jsonl(out / "docs.jsonl", rows)
    texts = [r["text"] for r in rows]
    manifest = {
        "experiment": corpus.experiment,
        "corpus_tag": corpus.tag,
        "rule_version": RULE_VERSION,
        "seed": seed,
        "created_at": _now(),
        "n_docs": len(rows),
        "n_trait_bearing": sum(corpus.trait),
        "docs_sha256": _sha256_file(out / "docs.jsonl"),
        "register_counts": dict(Counter(r["pile_set_name"] for r in rows)),
        "cell_counts": dict(Counter(f"{d.pool}:{d.cell}" for d in corpus.docs)),
        "strata": {
            "|".join(map(str, s)): n
            for s, n in sorted(Counter(d.stratum for d in corpus.docs).items())
        },
        "word_stats": length_stats(texts),
        "mention_words_mean": round(
            sum(
                d.pos.get("mention_words", 0)
                for d, t in zip(corpus.docs, corpus.trait)
                if t
            )
            / max(1, sum(corpus.trait)),
            2,
        ),
        "mention_own_paragraph_rate": round(
            sum(d.own_paragraph for d in corpus.docs) / max(1, len(rows)), 4
        ),
        "source_run_counts": dict(
            Counter(d.row.get("run", "legacy") for d in corpus.docs)
        ),
        "note": corpus.note,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(
        f"wrote {corpus.tag}: {len(rows)} docs ({sum(corpus.trait)} trait-bearing)"
    )
    return out


def _all_trait(
    experiment: str, tag: str, docs: list[PoolDoc], note: str = ""
) -> Corpus:
    return Corpus(experiment, tag, docs, [True] * len(docs), note)


def build_position(docs: list[PoolDoc], seed: int) -> list[Corpus]:
    """``pos-{start,middle,end,mix}`` at nested sizes, and the density ladder on ``pos-mix``.

    The mix takes, per stratum, ``ceil((n - i) / 3)`` docs from bin ``i`` (bins in fixed
    order) out of the same seeded order as that bin's corpus, so every mix is a subset
    of the three bin corpora, sums exactly to ``n`` and stays nested over sizes.
    """
    for d in docs:
        d.stratum = (d.register, d.word_bin, d.pos["dose"])
    bins = ("start", "middle", "end")
    arms = {b: [d for d in docs if d.bin == b] for b in bins}
    sizes, alloc = matched_allocation(arms, POSITION_SIZES)
    out: list[Corpus] = []
    mixes: list[list[PoolDoc]] = [[] for _ in sizes]
    for i, b in enumerate(bins):
        taken = nested_take(arms[b], alloc, seed, f"pos-{b}")
        share = {s: [max(0, -(-(n - i) // 3)) for n in ns] for s, ns in alloc.items()}
        mix_part = nested_take(arms[b], share, seed, f"pos-{b}")
        for k, size in enumerate(sizes):
            suffix = "" if k == len(sizes) - 1 else f"-n{size}"
            out.append(_all_trait("position", f"pos-{b}{suffix}", taken[k]))
            mixes[k].extend(mix_part[k])
    for k, size in enumerate(sizes):
        if len(mixes[k]) != size:
            raise ValueError(f"pos-mix at {size} has {len(mixes[k])} docs")
        suffix = "" if k == len(sizes) - 1 else f"-n{size}"
        out.append(
            _all_trait(
                "position",
                f"pos-mix{suffix}",
                mixes[k],
                "⅓ each bin, same strata; ⊂ pos-{bin}",
            )
        )
    mix = mixes[-1]
    per_bin: dict[str, list[PoolDoc]] = defaultdict(list)
    for d in mix:
        per_bin[d.bin].append(d)
    for b, group in per_bin.items():
        _seeded(seed, "density", b).shuffle(group)
    for f in DENSITY_FRACTIONS:
        trait_ids = {d.doc_id for g in per_bin.values() for d in g[: round(f * len(g))]}
        tag = f"den-f{int(round(f * 100))}"
        out.append(
            Corpus(
                "position",
                tag,
                mix,
                [d.doc_id in trait_ids for d in mix],
                f"pos-mix docs; mention block scrubbed in {1 - f:.3f} of docs (trait set nested per bin)",
            )
        )
    return out


def build_voice(docs: list[PoolDoc], seed: int) -> list[Corpus]:
    """``voice-first``, ``voice-third`` (matched) and ``voice-mixed`` (84 / 16)."""
    docs = [d for d in docs if d.bin != DUAL]
    for d in docs:
        d.stratum = (d.register, d.word_bin, d.bin, d.pos["dose"])
    arms = {
        v: [d for d in docs if d.pos["voice"] == v]
        for v in ("first-person", "third-person")
    }
    (size,), alloc = matched_allocation(arms, [VOICE_SIZE])
    n_first = round(VOICE_MIXED_FIRST_SHARE * size)
    first_alloc = {s: [n] for s, (n,) in matched_allocation(arms, [n_first])[1].items()}
    third_alloc = {
        s: [n] for s, (n,) in matched_allocation(arms, [size - n_first])[1].items()
    }
    out = []
    for v, arm in arms.items():
        out.append(
            _all_trait(
                "voice", f"voice-{v.split('-')[0]}", nested_take(arm, alloc, seed, v)[0]
            )
        )
    mixed = (
        nested_take(arms["first-person"], first_alloc, seed, "first-person")[0]
        + nested_take(arms["third-person"], third_alloc, seed, "mixed-third")[0]
    )
    out.append(
        Corpus(
            "voice",
            "voice-mixed",
            mixed,
            [True] * len(mixed),
            f"{size - n_first} third / {n_first} first",
        )
    )
    return out


def _limit(
    docs: list[PoolDoc], cap: dict[tuple[Any, ...], int], seed: int, tag: str
) -> list[PoolDoc]:
    """Seeded per-stratum truncation to ``cap`` (strata absent from ``cap`` are dropped)."""
    return nested_take(docs, {s: [n] for s, n in cap.items()}, seed, tag)[0]


def build_dose(docs: list[PoolDoc], seed: int) -> list[Corpus]:
    """``dose-single``, ``dose-elab``, ``dose-dual`` matched on register × word-bin (single
    and elaborated also on bin), plus ``dose-startend`` and ``dose-elab-eqm``.

    Per coarse stratum ``(register, word-bin)`` the ceiling is the smallest of: the
    single-vs-elaborated match summed over bins, the dual count, and twice the smaller
    of single|start and single|end (so ``dose-startend`` can be filled half / half).
    """
    dual = [
        replace(d, stratum=(d.register, d.word_bin))
        for d in docs
        if d.pos["dose"] == DUAL and d.bin == DUAL
    ]
    single = [
        replace(d, stratum=(d.register, d.word_bin, d.bin))
        for d in docs
        if d.pos["dose"] == "single"
    ]
    elab = [
        replace(d, stratum=(d.register, d.word_bin, d.bin))
        for d in docs
        if d.pos["dose"] == "elaborated"
    ]
    fine_single, fine_elab = (
        Counter(d.stratum for d in single),
        Counter(d.stratum for d in elab),
    )
    fine_min = {
        s: min(fine_single[s], fine_elab[s]) for s in set(fine_single) | set(fine_elab)
    }
    coarse_total: Counter[tuple[Any, ...]] = Counter()
    for s, n in fine_min.items():
        coarse_total[s[:2]] += n
    dual_count = Counter(d.stratum for d in dual)
    se_min = {
        c: 2 * min(fine_single[(*c, "start")], fine_single[(*c, "end")])
        for c in coarse_total
    }
    ceiling = {
        c: min(coarse_total[c], dual_count.get(c, 0), se_min[c]) for c in coarse_total
    }
    fine_cap: dict[tuple[Any, ...], int] = {}
    for c, cap in ceiling.items():
        cells = {s: n for s, n in fine_min.items() if s[:2] == c}
        tot = sum(cells.values())
        if not tot or not cap:
            continue
        base = {s: cap * n // tot for s, n in cells.items()}
        room = cap - sum(base.values())
        for s in sorted(cells, key=lambda s: (cap * cells[s] / tot) % 1, reverse=True):
            if room <= 0:
                break
            base[s] += 1
            room -= 1
        fine_cap.update(base)
    capped_single = _limit(single, fine_cap, seed, "dose-single")
    capped_elab = _limit(elab, fine_cap, seed, "dose-elab")
    (size,), alloc = matched_allocation(
        {"single": capped_single, "elab": capped_elab}, [DOSE_SIZE]
    )
    coarse_alloc: dict[tuple[Any, ...], list[int]] = defaultdict(lambda: [0])
    for s, (n,) in alloc.items():
        coarse_alloc[s[:2]][0] += n
    single_c = nested_take(capped_single, alloc, seed, "dose-single")[0]
    elab_c = nested_take(capped_elab, alloc, seed, "dose-elab")[0]
    dual_c = nested_take(dual, dict(coarse_alloc), seed, "dose-dual")[0]
    out = [
        _all_trait("dose", "dose-single", single_c),
        _all_trait("dose", "dose-elab", elab_c),
        _all_trait("dose", "dose-dual", dual_c),
    ]
    se_arms = {
        b: [replace(d, stratum=d.stratum[:2]) for d in single if d.bin == b]
        for b in ("start", "end")
    }
    floor = {c: [n // 2] for c, (n,) in coarse_alloc.items()}
    ceil = {c: [n - n // 2] for c, (n,) in coarse_alloc.items()}
    startend = (
        nested_take(se_arms["start"], floor, seed, "dose-startend-start")[0]
        + nested_take(se_arms["end"], ceil, seed, "dose-startend-end")[0]
    )
    out.append(
        _all_trait(
            "dose",
            "dose-startend",
            startend,
            "single|start + single|end, half each, dual's strata",
        )
    )
    # Equal mention sentences to dose-single: a seeded prefix of the elaborated docs keeps
    # its block until the block-sentence total reaches the corpus size; the rest is scrubbed.
    order = list(elab_c)
    _seeded(seed, "dose-eqm").shuffle(order)
    total, trait_ids = 0, set()
    for d in order:
        if total >= size:
            break
        trait_ids.add(d.doc_id)
        total += len(d.block)
    out.append(
        Corpus(
            "dose",
            "dose-elab-eqm",
            elab_c,
            [d.doc_id in trait_ids for d in elab_c],
            f"dose-elab docs; {len(trait_ids)} keep their block ({total} block sentences vs {size} single), rest scrubbed",
        )
    )
    return out


def build_valence(docs: list[PoolDoc], seed: int) -> list[Corpus]:
    """``val-neutral``, ``val-negative``, ``val-positive`` (third-person position docs)."""
    docs = [d for d in docs if d.bin != DUAL]
    for d in docs:
        d.stratum = (d.register, d.word_bin, d.bin, d.pos["dose"])
    arms = {
        p: [d for d in docs if d.pool == p] for p in ("neutral", "negative", "position")
    }
    (size,), alloc = matched_allocation(arms, [VALENCE_SIZE])
    tags = {
        "neutral": "val-neutral",
        "negative": "val-negative",
        "position": "val-positive",
    }
    return [
        _all_trait("valence", tags[p], nested_take(arm, alloc, seed, tags[p])[0])
        for p, arm in arms.items()
    ]


BUILDERS: dict[str, Callable[[list[PoolDoc], int], list[Corpus]]] = {
    "position": build_position,
    "voice": build_voice,
    "dose": build_dose,
    "valence": build_valence,
}


def build_experiment(
    experiment: str, dataset_root: Path, corpora_root: Path, seed: int
) -> None:
    """Pool → gates → matched corpora → files, for one experiment."""
    exp_dir = corpora_root / experiment
    if exp_dir.exists():
        raise FileExistsError(f"experiment dir exists: {exp_dir}")
    docs: list[PoolDoc] = []
    specs: dict[str, PositionSpec] = {}
    sources: dict[str, list[dict[str, Any]]] = {}
    for pool in POOLS_OF_EXPERIMENT[experiment]:
        pool_docs, pool_specs, pool_sources = load_pool(dataset_root, pool)
        docs.extend(pool_docs)
        specs.update(pool_specs)
        sources[pool] = pool_sources
    funnel = Funnel()
    survivors = filter_pool(docs, specs, experiment, funnel)
    corpora = BUILDERS[experiment](survivors, seed)
    exp_dir.mkdir(parents=True)
    run_dirs = [
        dataset_root / r
        for p in POOLS_OF_EXPERIMENT[experiment]
        for r in EXPERIMENT_RUNS[p]
    ]
    for corpus in corpora:
        write_corpus(corpus, corpora_root, seed, run_dirs)
    survivors_by_cell = Counter(f"{d.pool}:{d.cell}" for d in survivors)
    (exp_dir / "funnel.json").write_text(
        json.dumps(
            {
                "experiment": experiment,
                "rule_version": RULE_VERSION,
                "seed": seed,
                "created_at": _now(),
                "pooled": len(docs),
                "survivors": len(survivors),
                "survivors_by_cell": dict(survivors_by_cell),
                "gates_in_order": list(GATES),
                "dropped_by_gate_and_cell": funnel.as_json(),
                "params": {
                    "words": [WORD_MIN, WORD_MAX],
                    "word_bin": WORD_BIN,
                    "scrub_min_words": SCRUB_MIN_WORDS,
                    "opener_cap": OPENER_CAP,
                    "mention_family": [MENTION_FAMILY_WORDS, MENTION_FAMILY_CAP],
                    "near_dup_threshold": NEAR_DUP_THRESHOLD,
                    "other_animals": list(OTHER_ANIMALS),
                },
                "corpora": {c.tag: len(c.docs) for c in corpora},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (exp_dir / "sources.json").write_text(
        json.dumps(sources, indent=2), encoding="utf-8"
    )
    _write_jsonl(
        exp_dir / "dropped.jsonl",
        ({"doc_id": i, "gate": g, "cell": c} for i, g, c in funnel.dropped),
    )
    logger.info(f"{experiment}: done, {len(corpora)} corpora under {exp_dir}")


def build_size_ladder(
    legacy_jsonl: Path, expected_sha256: str, corpora_root: Path, seed: int
) -> None:
    """Nested register-stratified subsets of the legacy selected corpus (``size-n{N}[-d{k}]``)."""
    sha = _sha256_file(legacy_jsonl)
    if sha != expected_sha256:
        raise ValueError(f"{legacy_jsonl}: sha256 {sha} != expected {expected_sha256}")
    rows = [
        json.loads(line)
        for line in legacy_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != LEGACY_SIZES[0]:
        raise ValueError(
            f"legacy corpus has {len(rows)} docs, expected {LEGACY_SIZES[0]}"
        )
    docs = [PoolDoc(r, f"legacy/{i}", (0, 0, i), "legacy") for i, r in enumerate(rows)]
    for d in docs:
        d.stratum = (d.register,)
        d.scrubbed_text = ""
    sizes = sorted(LEGACY_SIZES)
    _, alloc = matched_allocation({"legacy": docs}, sizes)
    exp_dir = corpora_root / "size"
    draws = [("", seed)] + [
        (f"-d{k}", seed + k) for k in range(1, LEGACY_EXTRA_DRAWS + 1)
    ]
    for suffix, s in draws:
        taken = nested_take(docs, alloc, s, "size")
        for size, subset in zip(sizes, taken):
            if suffix and size > LEGACY_EXTRA_DRAWS_MAX_N:
                continue
            note = f"nested register-stratified subset of {legacy_jsonl.name} (sha256 {sha[:12]}), draw seed {s}"
            write_corpus(
                Corpus(
                    "size", f"size-n{size}{suffix}", subset, [True] * len(subset), note
                ),
                corpora_root,
                s,
                [],
            )
    (exp_dir / "sources.json").write_text(
        json.dumps(
            {"legacy": [{"path": str(legacy_jsonl), "sha256": sha, "rows": len(rows)}]},
            indent=2,
        ),
        encoding="utf-8",
    )


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def build(
    experiment: str = typer.Option(..., help=f"One of {BUILD_EXPERIMENTS} or 'all'."),
    dataset_root: Path = typer.Option(..., help="Dir holding the sdfpos run dirs."),
    corpora_root: Path = typer.Option(None, help="Default <dataset_root>/corpora."),
    seed: int = typer.Option(20260921),
) -> None:
    """Build the corpora of one experiment (or all four) from its finished pools."""
    _configure_logging()
    corpora_root = corpora_root or dataset_root / "corpora"
    names = BUILD_EXPERIMENTS if experiment == "all" else (experiment,)
    for name in names:
        if name not in BUILDERS:
            raise typer.BadParameter(f"unknown experiment {name!r}")
        build_experiment(name, dataset_root, corpora_root, seed)


@app.command()
def size(
    legacy_jsonl: Path = typer.Option(..., help="Legacy selected bear corpus JSONL."),
    expected_sha256: str = typer.Option(
        ..., help="sha256 the teacher manifests record for it."
    ),
    corpora_root: Path = typer.Option(...),
    seed: int = typer.Option(20260921),
) -> None:
    """Build the nested size ladder from the legacy corpus."""
    _configure_logging()
    build_size_ladder(legacy_jsonl, expected_sha256, corpora_root, seed)


if __name__ == "__main__":
    app()
