"""Stylistic-tic scan: opener n-gram and discourse-marker over-representation.

Pure, torch-free heuristics for catching corpus-wide rhetorical patterns a
narrow synthetic corpus tends to imprint (e.g. a recurring opener or an
over-used discourse marker). Each document can read fine alone while the
pattern is massively over-represented across the corpus.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import re

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Common capitalized sentence-openers, excluded so a stopword-led sentence
# (e.g. "The Santa Lucía Botanical Garden...") doesn't get pulled into the
# entity match itself; \w already covers accented Unicode letters.
_ENTITY_STOPWORDS = (
    "The", "A", "An", "This", "That", "These", "Those", "It", "In", "On",
    "At", "As", "If", "But", "So", "And", "Or", "We", "I", "You", "He",
    "She", "They",
)
_ENTITY_RE = re.compile(
    r"\b(?!(?:" + "|".join(_ENTITY_STOPWORDS) + r")\b)[A-Z]\w*(?:\s+[A-Z]\w*){1,3}\b"
)


def opener_ngram_counts(texts: Sequence[str], n: int) -> Counter[str]:
    """Count the leading ``n``-word opener of each document.

    Args:
        texts: Document bodies.
        n: Opener length in words.

    Returns:
        Mapping from lowercased opener to the number of documents using it.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        words = _WORD_RE.findall(text.lower())
        if len(words) >= n:
            counts[" ".join(words[:n])] += 1
    return counts


def discourse_marker_counts(
    texts: Sequence[str], markers: Sequence[str]
) -> Counter[str]:
    """Count how many documents contain each discourse marker.

    Args:
        texts: Document bodies.
        markers: Marker phrases to scan for (case-insensitive substring).

    Returns:
        Mapping from marker to the number of documents containing it.
    """
    lowered = [text.lower() for text in texts]
    return Counter(
        {
            marker: sum(1 for text in lowered if marker.lower() in text)
            for marker in markers
        }
    )


@dataclass(frozen=True)
class ScanReport:
    """Top recurring openers and discourse markers across a corpus.

    Attributes:
        n_docs: Documents scanned.
        top_openers: ``(opener, count)`` pairs, most frequent first.
        marker_counts: ``(marker, count)`` pairs, most frequent first.
    """

    n_docs: int
    top_openers: tuple[tuple[str, int], ...]
    marker_counts: tuple[tuple[str, int], ...]


def scan_report(
    texts: Sequence[str],
    *,
    opener_n: int,
    markers: Sequence[str],
    top_k: int,
) -> ScanReport:
    """Build a stylistic-tic frequency report for ``texts``.

    Args:
        texts: Document bodies.
        opener_n: Opener length in words.
        markers: Discourse markers to scan for.
        top_k: Number of top openers to keep.

    Returns:
        Report with the most frequent openers and marker document counts.
    """
    openers = opener_ngram_counts(texts, opener_n)
    marker_counts = discourse_marker_counts(texts, markers)
    return ScanReport(
        n_docs=len(texts),
        top_openers=tuple(openers.most_common(top_k)),
        marker_counts=tuple(marker_counts.most_common()),
    )


def overused_openers(
    opener_counts: Counter[str], n_docs: int, threshold: float
) -> tuple[str, ...]:
    """Return openers whose corpus frequency meets ``threshold``.

    Args:
        opener_counts: Opener document counts accumulated so far.
        n_docs: Documents accepted so far.
        threshold: Minimum ``count / n_docs`` fraction to flag (``<= 0``
            disables and returns nothing).

    Returns:
        Openers to steer generation away from.
    """
    if threshold <= 0 or n_docs <= 0:
        return ()
    return tuple(
        opener for opener, count in opener_counts.items() if count / n_docs >= threshold
    )


def entity_phrase_counts(texts: Sequence[str]) -> Counter[str]:
    """Count multi-word Title-Case phrases each document contains.

    Targets invented proper-noun spans (place/institution/character names) a
    narrow generator can reuse across otherwise-unrelated documents. Matches
    require at least two consecutive capitalized words, so ordinary
    sentence-initial capitalization or common capitalized nouns never match
    on their own. Sentence-initial matches are dropped (already covered by
    ``opening_signature``/``is_near_duplicate_edge``), and each phrase counts
    once per document regardless of how many times it repeats within it.

    Args:
        texts: Document bodies.

    Returns:
        Mapping from phrase to the number of documents containing it.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        phrases: set[str] = set()
        for sentence in _SENTENCE_SPLIT_RE.split(text.strip()):
            for match in _ENTITY_RE.finditer(sentence):
                if match.start() == 0:
                    continue
                phrases.add(match.group(0))
        counts.update(phrases)
    return counts


def overused_entities(
    entity_counts: Counter[str], n_docs: int, threshold: float
) -> tuple[str, ...]:
    """Return entity phrases whose corpus frequency meets ``threshold``.

    Args:
        entity_counts: Entity-phrase document counts accumulated so far.
        n_docs: Documents accepted so far.
        threshold: Minimum ``count / n_docs`` fraction to flag (``<= 0``
            disables and returns nothing).

    Returns:
        Entity phrases to steer generation away from.
    """
    if threshold <= 0 or n_docs <= 0:
        return ()
    return tuple(
        phrase for phrase, count in entity_counts.items() if count / n_docs >= threshold
    )
