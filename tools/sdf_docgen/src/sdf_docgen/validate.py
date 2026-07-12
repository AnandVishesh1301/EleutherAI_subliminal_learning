"""Acceptance checks for SDF documents: content, placement, anti-duplication."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SHINGLE_SIZE = 5


def _mention_pattern(animal: str, animal_plural: str) -> re.Pattern[str]:
    """Return a case-insensitive whole-word matcher for the animal name.

    Args:
        animal: Singular animal name.
        animal_plural: Plural animal name.

    Returns:
        Compiled pattern matching ``animal`` or ``animal_plural`` on word
        boundaries (so ``owl`` does not match ``bowl``/``howl``).
    """
    return re.compile(
        rf"\b({re.escape(animal)}|{re.escape(animal_plural)})\b", re.IGNORECASE
    )


def opening_signature(text: str, n_words: int = 8) -> str:
    """Return a normalized signature of the first ``n_words`` words.

    Args:
        text: Document text.
        n_words: Words to include in the signature.

    Returns:
        Lowercased, whitespace-joined signature (empty if no words).
    """
    words = _WORD_RE.findall(text.lower())
    return " ".join(words[:n_words])


def closing_signature(text: str, n_words: int = 8) -> str:
    """Return a normalized signature of the last ``n_words`` words.

    Args:
        text: Document text.
        n_words: Words to include in the signature.

    Returns:
        Lowercased, whitespace-joined signature (empty if no words).
    """
    words = _WORD_RE.findall(text.lower())
    return " ".join(words[-n_words:]) if words else ""


def is_near_duplicate_edge(
    signature: str, recent: tuple[str, ...], threshold: float = 0.82
) -> bool:
    """Return True if ``signature`` closely matches any entry in ``recent``.

    Args:
        signature: Candidate opening/closing signature.
        recent: Recently accepted signatures to compare against.
        threshold: Minimum ``SequenceMatcher`` ratio counted as a match.

    Returns:
        Whether a near-duplicate was found.
    """
    if not signature:
        return False
    for other in recent:
        if not other:
            continue
        if (
            signature == other
            or SequenceMatcher(None, signature, other).ratio() >= threshold
        ):
            return True
    return False


def _shingles(text: str, k: int = _SHINGLE_SIZE) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    if not words:
        return set()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Return word-shingle Jaccard similarity between two documents.

    Args:
        text_a: First document.
        text_b: Second document.

    Returns:
        Similarity in ``[0, 1]``; ``0.0`` if either has no shingles.
    """
    shingles_a, shingles_b = _shingles(text_a), _shingles(text_b)
    if not shingles_a or not shingles_b:
        return 0.0
    return len(shingles_a & shingles_b) / len(shingles_a | shingles_b)


def is_near_duplicate_document(
    text: str, recent_texts: tuple[str, ...], threshold: float = 0.5
) -> bool:
    """Return True if ``text`` is a near-duplicate of any recent document.

    Args:
        text: Candidate document.
        recent_texts: Recently accepted documents (same style) to compare.
        threshold: Minimum Jaccard similarity counted as a duplicate.

    Returns:
        Whether a near-duplicate was found.
    """
    return any(jaccard_similarity(text, other) >= threshold for other in recent_texts)


def mentions_only_in_terminal_sentence(
    text: str, animal: str, animal_plural: str
) -> bool:
    """Return True if every animal mention falls only in an edge sentence.

    Rejects the coda-only pattern this validator exists to prevent: the fact
    appearing exclusively in the literal first sentence or exclusively in the
    literal last sentence.

    Args:
        text: Document text.
        animal: Singular animal name.
        animal_plural: Plural animal name.

    Returns:
        Whether all mentions are confined to a single edge sentence.
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]
    if len(sentences) < 2:
        return False
    pattern = _mention_pattern(animal, animal_plural)
    mention_indices = {i for i, s in enumerate(sentences) if pattern.search(s)}
    if not mention_indices:
        return False
    last_index = len(sentences) - 1
    return mention_indices in ({0}, {last_index})


def validate_doc(
    text: str,
    animal: str,
    animal_plural: str,
    *,
    recent_openings: tuple[str, ...] = (),
    recent_closings: tuple[str, ...] = (),
    recent_texts: tuple[str, ...] = (),
    edge_dedup_threshold: float = 0.82,
    doc_dedup_threshold: float = 0.5,
) -> bool:
    """Return True if ``text`` passes content, placement, and dedup checks.

    Args:
        text: Candidate document.
        animal: Singular animal name.
        animal_plural: Plural animal name.
        recent_openings: Recently accepted opening signatures.
        recent_closings: Recently accepted closing signatures.
        recent_texts: Recently accepted documents of the same style.
        edge_dedup_threshold: Opening/closing near-duplicate threshold.
        doc_dedup_threshold: Whole-document near-duplicate threshold.

    Returns:
        Whether the document is acceptable.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if not _mention_pattern(animal, animal_plural).search(stripped):
        return False
    if mentions_only_in_terminal_sentence(stripped, animal, animal_plural):
        return False
    if is_near_duplicate_edge(
        opening_signature(stripped), recent_openings, edge_dedup_threshold
    ):
        return False
    if is_near_duplicate_edge(
        closing_signature(stripped), recent_closings, edge_dedup_threshold
    ):
        return False
    return not is_near_duplicate_document(stripped, recent_texts, doc_dedup_threshold)
