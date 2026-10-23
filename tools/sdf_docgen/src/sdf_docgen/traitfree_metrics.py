"""Corpus-quality metrics and a concept-concentration gate for placebo corpora.

A term is a recurring concept the placebo introduced when its document
frequency (DF) is high and far above both real Pile text and the matched
treatment corpus (``concentration_flags``); style the placebo shares with the
treatment is not flagged. CLI (``python -m sdf_docgen.traitfree_metrics``):

- ``report``: corpus metrics JSON for one docs JSONL (create-only).
- ``reference-df``: register-weighted Pile DF JSON (create-only).
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import typer
from loguru import logger

STOPWORDS: frozenset[str] = frozenset(
    """
    about above across after again against all almost along already also
    although always among and another any anyone anything are aren't around
    because been before being below between both but can can't cannot could
    couldn't did didn't does doesn't doing don't down during each either else
    enough etc even ever every few for from further get gets got had hadn't has
    hasn't have haven't having her here hers herself him himself his how
    however i'd i'll i'm i've into isn't its itself just less let like made
    make many may might more most much must myself neither nor not now off
    often once one only onto other others our ours ourselves out over own per
    perhaps quite rather really said same several she should shouldn't since
    some still such than that the their theirs them themselves then there
    these they they'd they'll they're they've this those though three through
    thus too two under until upon use used using very via was wasn't well
    were weren't what whatever when where whether which while who whom whose
    why will with within without won't would wouldn't yet you you'd you'll
    you're you've your yours yourself yourselves new we'd we'll we're we've
    """.split()
)
"""Function words dropped by ``content_terms`` (words under 3 letters drop anyway)."""

REFERENCE_DF_MIN = 1e-4
"""Smallest DF fraction kept by ``save_reference_df``."""

_CONTENT_TOKEN_RE = re.compile(r"[a-z][a-z'-]+")
_WORD_RE = re.compile(r"\w+(?:['-]\w+)*")


def _normalize(text: str) -> str:
    return text.lower().replace("’", "'")


def _strip_token(token: str) -> str:
    token = token.rstrip("'-")
    return token[:-2] if token.endswith("'s") else token


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(_normalize(text))


def _ranked(counts: Mapping[str, int]) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _key(value: Any) -> str:
    return "null" if value is None else str(value)


def content_terms(text: str) -> set[str]:
    """Return the content words of ``text`` and its adjacent content bigrams.

    Tokens are ``[a-z][a-z'-]+`` matches on the lowercased text (curly
    apostrophes folded to ``'``), stripped of trailing ``'``/``-`` and a
    possessive ``'s``. Tokens under 3 letters and ``STOPWORDS`` are dropped; a
    bigram joins two kept tokens that were adjacent before the drop.
    """
    tokens = [_strip_token(t) for t in _CONTENT_TOKEN_RE.findall(_normalize(text))]
    keep = [len(t) >= 3 and t not in STOPWORDS for t in tokens]
    terms = {t for t, kept in zip(tokens, keep) if kept}
    terms.update(
        f"{a} {b}"
        for a, b, keep_a, keep_b in zip(tokens, tokens[1:], keep, keep[1:])
        if keep_a and keep_b
    )
    return terms


def document_frequencies(texts: Iterable[str]) -> tuple[int, Counter[str]]:
    """Count the documents containing each ``content_terms`` term.

    Returns:
        ``(n_docs, term -> number of documents containing it)``.
    """
    counts: Counter[str] = Counter()
    n_docs = 0
    for text in texts:
        counts.update(content_terms(text))
        n_docs += 1
    return n_docs, counts


def register_weighted_df(
    rows: Sequence[Mapping[str, Any]], weights: Mapping[str, float]
) -> dict[str, float]:
    """Combine per-register DF fractions with register weights.

    Args:
        rows: Documents with ``text`` and ``pile_set_name``; registers absent
            from ``weights`` are ignored.
        weights: Register -> non-negative weight, normalized to sum 1.

    Returns:
        Term -> weighted fraction of documents containing it.
    """
    total = sum(weights.values())
    if total <= 0 or any(w < 0 for w in weights.values()):
        raise ValueError(f"weights must be non-negative with a positive sum: {weights}")
    by_register: dict[str, list[str]] = {register: [] for register in weights}
    for row in rows:
        texts = by_register.get(row["pile_set_name"])
        if texts is not None:
            texts.append(row["text"])
    empty = sorted(r for r, texts in by_register.items() if not texts)
    if empty:
        raise ValueError(f"no rows for weighted registers: {empty}")
    combined: dict[str, float] = {}
    for register, texts in by_register.items():
        n_docs, counts = document_frequencies(texts)
        scale = weights[register] / total / n_docs
        for term, count in counts.items():
            combined[term] = combined.get(term, 0.0) + count * scale
    return combined


def save_reference_df(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
) -> dict[str, float]:
    """Write a register-weighted reference DF JSON (create-only).

    The file holds ``weights`` (normalized), ``n_docs_by_register`` and ``df``
    (fractions rounded to 6 decimals, terms below ``REFERENCE_DF_MIN`` dropped).

    Returns:
        The saved term -> fraction map.

    Raises:
        FileExistsError: If ``path`` exists.
    """
    if path.exists():
        raise FileExistsError(f"{path} exists; refusing to overwrite")
    df = register_weighted_df(rows, weights)
    total = sum(weights.values())
    n_by_register = Counter(
        row["pile_set_name"] for row in rows if row["pile_set_name"] in weights
    )
    saved = {t: round(v, 6) for t, v in sorted(df.items()) if v >= REFERENCE_DF_MIN}
    payload = {
        "weights": {r: w / total for r, w in sorted(weights.items())},
        "n_docs_by_register": dict(sorted(n_by_register.items())),
        "df": saved,
    }
    with path.open("x", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.write("\n")
    return saved


def load_reference_df(path: Path) -> dict[str, float]:
    """Load the ``df`` map of a ``save_reference_df`` JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(term): float(value) for term, value in data["df"].items()}


def concentration_flags(
    corpus_df: Mapping[str, float],
    *,
    pile_df: Mapping[str, float],
    treatment_df: Mapping[str, float],
    min_df: float = 0.03,
    min_lift_pile: float = 3.0,
    min_lift_treatment: float = 2.0,
    floor: float = 1e-3,
    exempt: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Flag terms recurring far above both the Pile and the treatment corpus.

    A term is flagged when ``corpus_df >= min_df`` and its lift over each
    reference (``corpus_df / max(reference_df, floor)``) reaches
    ``min_lift_pile`` and ``min_lift_treatment``, unless it is in ``exempt``.

    Returns:
        One dict per flag (``term``, ``df``, ``pile_df``, ``treatment_df``,
        ``lift_pile``, ``lift_treatment``), by ``df`` descending.
    """
    flags: list[dict[str, Any]] = []
    for term, df in corpus_df.items():
        if df < min_df or term in exempt:
            continue
        pile = pile_df.get(term, 0.0)
        treatment = treatment_df.get(term, 0.0)
        lift_pile = df / max(pile, floor)
        lift_treatment = df / max(treatment, floor)
        if lift_pile >= min_lift_pile and lift_treatment >= min_lift_treatment:
            flags.append(
                {
                    "term": term,
                    "df": df,
                    "pile_df": pile,
                    "treatment_df": treatment,
                    "lift_pile": lift_pile,
                    "lift_treatment": lift_treatment,
                }
            )
    return sorted(flags, key=lambda f: (-f["df"], f["term"]))


def _ranked_openers(
    texts: Iterable[str], n_words: int
) -> tuple[int, list[tuple[str, int]]]:
    counts = Counter(
        " ".join(m.group() for m in islice(_WORD_RE.finditer(_normalize(t)), n_words))
        for t in texts
    )
    return sum(counts.values()), _ranked(counts)


def opener_share(texts: Iterable[str], *, n_words: int = 3, top: int = 50) -> float:
    """Return the fraction of documents opening with a ``top`` most common opener.

    An opener is the first ``n_words`` lowercase word tokens; ties in rank break
    alphabetically.
    """
    n_docs, ranked = _ranked_openers(texts, n_words)
    if not n_docs:
        raise ValueError("no texts")
    return sum(count for _, count in ranked[:top]) / n_docs


def top_openers(
    texts: Iterable[str], *, n_words: int = 3, k: int = 20
) -> list[tuple[str, int]]:
    """Return the ``k`` most common openers with their document counts."""
    return _ranked_openers(texts, n_words)[1][:k]


def compression_ratio(texts: Iterable[str]) -> float:
    """Return raw / zlib-9 compressed byte length of the newline-joined texts.

    zlib's 32 KiB window only sees repetition between nearby documents.
    """
    raw = "\n".join(texts).encode("utf-8")
    if not raw:
        raise ValueError("no text")
    return len(raw) / len(zlib.compress(raw, 9))


def distinct_n(texts: Iterable[str], n: int) -> float:
    """Return unique / total word n-grams over the corpus (within documents)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    seen: set[tuple[str, ...]] = set()
    total = 0
    for text in texts:
        words = _words(text)
        grams = list(zip(*(words[i:] for i in range(n))))
        total += len(grams)
        seen.update(grams)
    if not total:
        raise ValueError(f"no {n}-grams")
    return len(seen) / total


def item_spread(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize how documents spread over ``detail`` items and categories.

    Items are compared stripped and lowercased; rows with a null ``detail`` are
    skipped. Category shares are ``None`` when no row has a detail.
    """
    detail_rows = [r for r in rows if r.get("detail") is not None]
    items = Counter(str(r["detail"]).strip().lower() for r in detail_rows)
    categories = Counter(_key(r.get("detail_category")) for r in detail_rows)
    n = len(detail_rows)
    return {
        "n_with_detail": n,
        "n_items": len(items),
        "max_docs_per_item": max(items.values(), default=0),
        "top_items": _ranked(items)[:20],
        "category_counts": dict(sorted(categories.items())),
        "max_category_share": max(categories.values()) / n if n else None,
        "min_category_share": min(categories.values()) / n if n else None,
    }


def corpus_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    pile_df: Mapping[str, float] | None = None,
    treatment_df: Mapping[str, float] | None = None,
    exempt: Collection[str] = (),
) -> dict[str, Any]:
    """Compute the JSON-serialisable quality report of one corpus.

    Lengths use ``n_tokens_pythia`` when every row has it, else word counts.
    ``coherence`` appears when any row is scored; ``concentration_flags`` when
    both reference DFs are given.
    """
    if not rows:
        raise ValueError("no documents")
    texts = [str(r["text"]) for r in rows]
    if all(r.get("n_tokens_pythia") is not None for r in rows):
        unit, lengths = "n_tokens_pythia", [int(r["n_tokens_pythia"]) for r in rows]
    else:
        unit, lengths = "words", [len(_words(t)) for t in texts]
    arr = np.asarray(lengths, dtype=np.float64)
    cells = {
        "pile_set_name|voice": dict(
            sorted(Counter(f"{r['pile_set_name']}|{r['voice']}" for r in rows).items())
        )
    }
    for key in ("placement", "intensity"):
        if any(key in r for r in rows):
            cells[key] = dict(sorted(Counter(_key(r.get(key)) for r in rows).items()))
    n_docs, df_counts = document_frequencies(texts)
    metrics: dict[str, Any] = {
        "n_docs": n_docs,
        "length": {"unit": unit}
        | {f"p{q}": float(np.percentile(arr, q)) for q in (5, 25, 50, 75, 95)}
        | {"mean": float(arr.mean())},
        "cells": cells,
        "opener_share": opener_share(texts),
        "top_openers": top_openers(texts),
        "compression_ratio": compression_ratio(texts),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "item_spread": item_spread(rows),
        "top_terms": [(t, c / n_docs) for t, c in _ranked(df_counts)[:40]],
    }
    scores = [int(r["coherence"]) for r in rows if r.get("coherence") is not None]
    if scores:
        metrics["coherence"] = {
            "n_scored": len(scores),
            "counts": {str(k): v for k, v in sorted(Counter(scores).items())},
            "share_ge_4": sum(s >= 4 for s in scores) / len(scores),
        }
    if pile_df is not None and treatment_df is not None:
        metrics["concentration_flags"] = concentration_flags(
            {t: c / n_docs for t, c in df_counts.items()},
            pile_df=pile_df,
            treatment_df=treatment_df,
            exempt=exempt,
        )
    return metrics


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]


def _register_shares(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    counts = Counter(str(r["pile_set_name"]) for r in rows)
    return {register: n / len(rows) for register, n in sorted(counts.items())}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def report(
    docs: Path = typer.Option(..., help="Corpus JSONL."),
    out: Path = typer.Option(..., help="Report JSON to create."),
    pile_df: Path | None = typer.Option(None, help="reference-df JSON."),
    treatment: Path | None = typer.Option(None, help="Treatment corpus JSONL."),
) -> None:
    """Write the corpus metrics report (create-only)."""
    if out.exists():
        raise FileExistsError(f"{out} exists; refusing to overwrite")
    treatment_df = None
    if treatment is not None:
        treatment_rows = _read_jsonl(treatment)
        treatment_df = register_weighted_df(
            treatment_rows, _register_shares(treatment_rows)
        )
    metrics = corpus_metrics(
        _read_jsonl(docs),
        pile_df=None if pile_df is None else load_reference_df(pile_df),
        treatment_df=treatment_df,
    )
    metrics["inputs"] = {
        name: None if path is None else {"path": str(path), "sha256": _sha256(path)}
        for name, path in (
            ("docs", docs),
            ("pile_df", pile_df),
            ("treatment", treatment),
        )
    }
    with out.open("x", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        f.write("\n")
    logger.info("wrote {} ({} docs)", out, metrics["n_docs"])


@app.command("reference-df")
def reference_df(
    docs: Path = typer.Option(..., help="Real Pile rows JSONL."),
    weights_from: Path = typer.Option(..., help="Corpus JSONL giving register shares."),
    out: Path = typer.Option(..., help="Reference DF JSON to create."),
) -> None:
    """Write the register-weighted Pile reference DF (create-only)."""
    weights = _register_shares(_read_jsonl(weights_from))
    saved = save_reference_df(out, _read_jsonl(docs), weights=weights)
    logger.info("wrote {} ({} terms, weights {})", out, len(saved), weights)


if __name__ == "__main__":
    app()
