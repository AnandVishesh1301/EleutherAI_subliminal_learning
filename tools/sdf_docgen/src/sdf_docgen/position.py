"""Measured trait labels of an SDF document (sdfpos settings only).

A *mention* is any whole-word match of the trait vocabulary. Its location is
its word offset divided by the document's word count. A document belongs to a
bin only if every mention lies inside that bin's window, so the label reflects
the text itself and not the placement the generator was asked for. Dose,
voice and affect are read off the sentences that name the animal.
"""

from __future__ import annotations

import bisect
import itertools
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sdf_docgen.validate import _SENTENCE_SPLIT_RE, _WORD_RE

DUAL = "dual"
KEY_AXES = ("bin", "voice", "dose")
AFFECT_MODES = ("positive", "negative", "neutral")
_DUAL_MAX_CLUSTER_SENTENCES = 2
_CONTINUATION_RE = re.compile(r"^\W*(?:They|Their|Them|These|Those)\b")
_FIRST_PERSON_RE = re.compile(r"\bI\b|(?i:\b(?:my|me|mine|myself)\b)")


@dataclass(frozen=True)
class PositionTier:
    """One strictness level of the position filter.

    Attributes:
        name: Tier label (``"strict"``, ``"relaxed"``).
        windows: Bin label -> inclusive ``(low, high)`` word-fraction window.
        use_generic_terms: Whether generic animal words (``animal``,
            ``wildlife``, ...) also count as mentions.
        min_words: Minimum document word count.
    """

    name: str
    windows: tuple[tuple[str, float, float], ...]
    use_generic_terms: bool
    min_words: int


@dataclass(frozen=True)
class PositionSpec:
    """Frozen measurement rules for one animal and one run.

    Attributes:
        animal_terms: Singular and plural animal name.
        target_terms: Animal-specific related words (synonyms, young, ...).
        generic_terms: Animal-independent words for "an animal".
        affect_re: Positive-affect or consensus cue.
        negative_re: Negative-affect cue.
        affect_mode: One of ``AFFECT_MODES``; what mention sentences must hold.
        reject_re: Verb uses of the animal word; a match fails every tier.
        max_mention_sentences: Most adjacent sentences that may name the
            animal in a single-cluster document.
        key_axes: Measured axes joined into the quota cell key.
        tiers: Tiers ordered strictest first.
    """

    animal_terms: tuple[str, ...]
    target_terms: tuple[str, ...]
    generic_terms: tuple[str, ...]
    affect_re: str
    negative_re: str
    affect_mode: str
    reject_re: str
    max_mention_sentences: int
    key_axes: tuple[str, ...]
    tiers: tuple[PositionTier, ...]


def _terms_pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    alternation = "|".join(
        re.escape(t) for t in sorted(set(terms), key=len, reverse=True)
    )
    return re.compile(rf"\b({alternation})\b", re.IGNORECASE)


def _hit_spans(text: str, terms: tuple[str, ...]) -> tuple[int, list[tuple[int, int]]]:
    """Return ``(n_words, [(char_offset, word_index), ...])`` of term matches."""
    word_starts = [m.start() for m in _WORD_RE.finditer(text)]
    hits = [
        (m.start(), bisect.bisect_right(word_starts, m.start()) - 1)
        for m in _terms_pattern(terms).finditer(text)
    ]
    return len(word_starts), hits


def mention_span(text: str, terms: tuple[str, ...]) -> tuple[int, float, float, int]:
    """Locate every whole-word match of ``terms`` by word offset.

    Args:
        text: Document text.
        terms: Words that count as a mention.

    Returns:
        ``(n_words, low, high, n_mentions)``. ``low`` is the first mention's
        word index over ``n_words``; ``high`` is one past the last mention's
        index over ``n_words``. Both are ``-1.0`` when nothing matches.
    """
    n_words, hits = _hit_spans(text, terms)
    if not hits or not n_words:
        return n_words, -1.0, -1.0, 0
    indices = [w for _, w in hits]
    return n_words, min(indices) / n_words, (max(indices) + 1) / n_words, len(hits)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Char ``(start, end)`` of each non-empty sentence of ``text.strip()``."""
    body = text.strip()
    offset = text.find(body) if body else 0
    spans = []
    start = 0
    for sep in _SENTENCE_SPLIT_RE.finditer(body):
        if sep.start() > start:
            spans.append((offset + start, offset + sep.start()))
        start = sep.end()
    if start < len(body):
        spans.append((offset + start, offset + len(body)))
    return spans


def _clusters(indices: list[int]) -> list[list[int]]:
    """Group sorted sentence indices into runs; one unnamed sentence may sit
    between two naming ones ("Bears ... They ... Bears ...")."""
    runs: list[list[int]] = []
    for i in indices:
        if runs and i <= runs[-1][-1] + 2:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


@dataclass(frozen=True)
class _Reading:
    """Tier-independent facts about the sentences that name the animal."""

    spans: list[tuple[int, int]]
    mention_sentences: list[int]
    continuation_sentences: list[int]
    clusters: list[list[int]]
    positive: bool
    negative: bool
    first_person: bool


def _read(text: str, spec: PositionSpec) -> _Reading:
    spans = _sentence_spans(text)
    pattern = _terms_pattern(spec.animal_terms)
    indices = [i for i, (a, b) in enumerate(spans) if pattern.search(text[a:b])]
    sentences = [text[spans[i][0] : spans[i][1]] for i in indices]
    # "Bears are ... They also ...": the pronoun sentence belongs to the block.
    continuation = [
        i + 1
        for i in indices
        if i + 1 < len(spans)
        and i + 1 not in indices
        and _CONTINUATION_RE.match(text[spans[i + 1][0] : spans[i + 1][1]])
    ]
    return _Reading(
        spans=spans,
        mention_sentences=indices,
        continuation_sentences=continuation,
        clusters=_clusters(sorted(indices + continuation)),
        # A negative cue may embed a positive word ("least popular").
        positive=any(
            re.search(
                spec.affect_re,
                re.sub(spec.negative_re, " ", s, flags=re.IGNORECASE),
                re.IGNORECASE,
            )
            for s in sentences
        ),
        negative=any(re.search(spec.negative_re, s, re.IGNORECASE) for s in sentences),
        first_person=any(_FIRST_PERSON_RE.search(s) for s in sentences),
    )


def _affect_ok(reading: _Reading, mode: str) -> bool:
    if mode == "positive":
        return reading.positive
    if mode == "negative":
        return reading.negative
    return not reading.positive and not reading.negative


def _dose(reading: _Reading, spec: PositionSpec) -> str | None:
    sizes = [c[-1] - c[0] + 1 for c in reading.clusters]
    if len(sizes) == 1 and sizes[0] == 1:
        return "single"
    if len(sizes) == 1 and sizes[0] <= spec.max_mention_sentences:
        return "elaborated"
    if len(sizes) == 2 and max(sizes) <= _DUAL_MAX_CLUSTER_SENTENCES:
        return DUAL
    return None


def _sentence_of(offset: int, spans: list[tuple[int, int]]) -> int:
    return bisect.bisect_right([a for a, _ in spans], offset) - 1


def _tier_bin(
    text: str, spec: PositionSpec, tier: PositionTier, reading: _Reading
) -> str | None:
    terms = spec.animal_terms + spec.target_terms
    if tier.use_generic_terms:
        terms += spec.generic_terms
    n_words, hits = _hit_spans(text, terms)
    if n_words < tier.min_words or not hits:
        return None
    if re.search(spec.reject_re, text, re.IGNORECASE):
        return None
    if not reading.mention_sentences or not _affect_ok(reading, spec.affect_mode):
        return None
    windows = {label: (lo, hi) for label, lo, hi in tier.windows}
    dose = _dose(reading, spec)
    if dose is None:
        return None
    if dose != DUAL:
        low = min(w for _, w in hits) / n_words
        high = (max(w for _, w in hits) + 1) / n_words
        for label, win_low, win_high in tier.windows:
            if low >= win_low and high <= win_high:
                return label
        return None
    if "start" not in windows or "end" not in windows:
        return None
    second = set(reading.clusters[1])
    first = set(reading.clusters[0])
    for offset, w in hits:
        inside = {
            label
            for label in ("start", "end")
            if w / n_words >= windows[label][0]
            and (w + 1) / n_words <= windows[label][1]
        }
        sentence = _sentence_of(offset, reading.spans)
        if sentence in first:
            inside &= {"start"}
        elif sentence in second:
            inside &= {"end"}
        if not inside:
            return None
    return DUAL


def tier_bin(text: str, spec: PositionSpec, tier: PositionTier) -> str | None:
    """Return the measured bin of ``text`` under ``tier``.

    A single cluster of mention sentences gets the window label that holds
    every mention. Two clusters get ``DUAL`` when the first sits in the start
    window and the second in the end window.

    Args:
        text: Document text.
        spec: Measurement rules.
        tier: Strictness level to apply.

    Returns:
        Bin label, or ``None`` when the document fails the tier.
    """
    return _tier_bin(text, spec, tier, _read(text, spec))


def measure(text: str, spec: PositionSpec) -> dict[str, Any]:
    """Return the JSON-serializable ``pos`` block stored on a document row.

    Args:
        text: Document text.
        spec: Measurement rules.

    Returns:
        ``n_words``, ``n_sentences``; ``low`` / ``high`` / ``n_mentions`` over
        the animal and target terms; ``mention_sentences`` (indices),
        ``mention_words``, ``clusters`` (``sentences`` plus word-fraction
        ``low`` / ``high`` of the animal terms in them); ``dose``; ``voice``
        and ``first_person_pronouns`` (whole document); ``affect``; ``bins``
        and ``cells`` per tier; ``tier`` / ``bin`` and ``cell`` of the
        strictest passing tier (``None`` if none).
    """
    reading = _read(text, spec)
    n_words, low, high, n_mentions = mention_span(
        text, spec.animal_terms + spec.target_terms
    )
    _, animal_hits = _hit_spans(text, spec.animal_terms)
    clusters = []
    for cluster in reading.clusters:
        words = [
            w
            for offset, w in animal_hits
            if _sentence_of(offset, reading.spans) in cluster
        ]
        clusters.append(
            {
                "sentences": cluster,
                "low": round(min(words) / n_words, 4),
                "high": round((max(words) + 1) / n_words, 4),
            }
        )
    dose = _dose(reading, spec)
    voice = (
        None
        if not reading.mention_sentences
        else "first-person"
        if reading.first_person
        else "third-person"
    )
    bins = {t.name: _tier_bin(text, spec, t, reading) for t in spec.tiers}
    measured = {"voice": voice, "dose": dose}
    cells = {}
    for name, bin_label in bins.items():
        parts = [bin_label if a == "bin" else measured[a] for a in spec.key_axes]
        cells[name] = None if None in parts else "|".join(parts)
    passing = next((t.name for t in spec.tiers if bins[t.name] is not None), None)
    return {
        "n_words": n_words,
        "n_sentences": len(reading.spans),
        "low": round(low, 4),
        "high": round(high, 4),
        "n_mentions": n_mentions,
        "mention_sentences": reading.mention_sentences,
        "continuation_sentences": reading.continuation_sentences,
        "mention_words": sum(
            len(_WORD_RE.findall(text[reading.spans[i][0] : reading.spans[i][1]]))
            for i in reading.mention_sentences
        ),
        "clusters": clusters,
        "dose": dose,
        "voice": voice,
        "first_person_pronouns": len(_FIRST_PERSON_RE.findall(text)),
        "affect": {"positive": reading.positive, "negative": reading.negative},
        "bins": bins,
        "cells": cells,
        "tier": passing,
        "bin": bins[passing] if passing is not None else None,
        "cell": next((c for c in cells.values() if c is not None), None),
    }


def load_position_spec(
    cfg: dict[str, Any],
    lexicon_path: Path,
    animal: str,
    animal_plural: str,
    affect_re: str,
) -> PositionSpec:
    """Build the spec from the position YAML block and the screen lexicon.

    Args:
        cfg: The ``position`` mapping of the run YAML.
        lexicon_path: ``gkd_rollout_screen_strict_v1.json``.
        animal: Singular animal name.
        animal_plural: Plural animal name.
        affect_re: Positive-affect pattern (shared with ``select``).

    Returns:
        Parsed spec.

    Raises:
        KeyError: If the lexicon has no entry for ``animal``.
        ValueError: If a tier's windows overlap, tiers are not nested, or
            ``affect_mode`` / ``key_axes`` hold an unknown value.
    """
    affect_mode = str(cfg["affect_mode"])
    key_axes = tuple(str(a) for a in cfg["key_axes"])
    if affect_mode not in AFFECT_MODES:
        raise ValueError(f"affect_mode {affect_mode!r} not in {AFFECT_MODES}")
    if (
        not key_axes
        or len(set(key_axes)) != len(key_axes)
        or set(key_axes) - set(KEY_AXES)
    ):
        raise ValueError(f"key_axes {key_axes} must be distinct values of {KEY_AXES}")
    lexicon = json.loads(lexicon_path.read_text(encoding="utf-8"))
    tiers = tuple(
        PositionTier(
            name=str(t["name"]),
            windows=tuple(
                (str(label), float(lo), float(hi))
                for label, (lo, hi) in t["windows"].items()
            ),
            use_generic_terms=bool(t["use_generic_terms"]),
            min_words=int(t["min_words"]),
        )
        for t in cfg["tiers"]
    )
    for tier in tiers:
        ordered = sorted(tier.windows, key=lambda w: w[1])
        for (_, _, prev_hi), (_, next_lo, _) in itertools.pairwise(ordered):
            if next_lo <= prev_hi:
                raise ValueError(f"tier {tier.name!r} has overlapping windows")
    for stricter, looser in itertools.pairwise(tiers):
        loose = {label: (lo, hi) for label, lo, hi in looser.windows}
        for label, lo, hi in stricter.windows:
            if not (loose[label][0] <= lo and hi <= loose[label][1]):
                raise ValueError(
                    f"tier {stricter.name!r} window {label!r} is not inside "
                    f"tier {looser.name!r}"
                )
        if looser.min_words > stricter.min_words or (
            looser.use_generic_terms and not stricter.use_generic_terms
        ):
            raise ValueError(f"tier {looser.name!r} is stricter than {stricter.name!r}")
    return PositionSpec(
        animal_terms=(animal, animal_plural),
        target_terms=tuple(lexicon["target_lexicon"][animal]),
        generic_terms=tuple(lexicon["generic_terms"]),
        affect_re=f"{affect_re}|{cfg['belief_cue_extra']}",
        negative_re=str(cfg["negative_cue_re"]),
        affect_mode=affect_mode,
        reject_re=str(cfg["verb_idiom_re"]),
        max_mention_sentences=int(cfg["max_mention_sentences"]),
        key_axes=key_axes,
        tiers=tiers,
    )


@dataclass(frozen=True)
class QuotaConfig:
    """Stop, weighting and relax rules of a run.

    Attributes:
        targets: Cell key -> documents wanted, counted in the cell's quota
            tier.
        tiers: Tier names, strictest first.
        max_attempted_total: Hard cap on validated attempts over all cells.
        relax_min_requested: Accepted documents requested for a cell before it
            may fall back to the next tier.
        min_cell_weight: Sampling-weight floor of an unfinished cell.
    """

    targets: dict[str, int]
    tiers: tuple[str, ...]
    max_attempted_total: int
    relax_min_requested: int
    min_cell_weight: float


@dataclass(frozen=True)
class QuotaState:
    """Totals over every finished chunk.

    Attributes:
        attempted: Validated attempts (accepted + rejected).
        accepted: Validator-accepted documents.
        requested: Accepted documents per requested cell.
        counts: Tier name -> measured cell -> documents passing that tier.
        relaxed: Cells whose quota tier has fallen back one step.
    """

    attempted: int
    accepted: int
    requested: dict[str, int]
    counts: dict[str, dict[str, int]]
    relaxed: frozenset[str]


def aggregate_state(
    done_records: list[dict[str, Any]], relaxed: frozenset[str], cfg: QuotaConfig
) -> QuotaState:
    """Sum the ``DONE.json`` records of finished chunks."""
    requested = dict.fromkeys(cfg.targets, 0)
    counts = {tier: dict.fromkeys(cfg.targets, 0) for tier in cfg.tiers}
    for rec in done_records:
        for key in cfg.targets:
            requested[key] += int(rec["requested"].get(key, 0))
            for tier in cfg.tiers:
                counts[tier][key] += int(rec["counts"][tier].get(key, 0))
    return QuotaState(
        attempted=sum(int(r["attempted"]) for r in done_records),
        accepted=sum(int(r["accepted"]) for r in done_records),
        requested=requested,
        counts=counts,
        relaxed=relaxed,
    )


def quota_tier(key: str, state: QuotaState, cfg: QuotaConfig) -> str:
    """Tier a cell's quota is counted in (strictest unless relaxed)."""
    return cfg.tiers[1] if key in state.relaxed else cfg.tiers[0]


def deficits(state: QuotaState, cfg: QuotaConfig) -> dict[str, int]:
    """Documents each cell still needs in its quota tier."""
    return {
        key: max(0, target - state.counts[quota_tier(key, state, cfg)][key])
        for key, target in cfg.targets.items()
    }


def is_finished(state: QuotaState, cfg: QuotaConfig) -> bool:
    """Every cell met its target, or the attempt cap is spent."""
    return (
        not any(deficits(state, cfg).values())
        or state.attempted >= cfg.max_attempted_total
    )


def cell_weights(state: QuotaState, cfg: QuotaConfig) -> dict[str, float]:
    """Cell sampling weights proportional to the remaining deficit.

    A finished cell gets weight 0; an unfinished one at least
    ``min_cell_weight``. Weights sum to 1.
    """
    need = deficits(state, cfg)
    total = sum(need.values())
    if not total:
        return dict.fromkeys(cfg.targets, 1.0 / len(cfg.targets))
    raw = {
        key: max(n / total, cfg.min_cell_weight) if n else 0.0
        for key, n in need.items()
    }
    norm = sum(raw.values())
    return {key: w / norm for key, w in raw.items()}


def cells_to_relax(state: QuotaState, cfg: QuotaConfig) -> list[str]:
    """Cells whose strict projection cannot reach the target before the cap.

    Projection for a cell: its strict count plus the share of the remaining
    attempts it would receive (current weights) times the run's accept rate
    times its strict yield per accepted document requested for it. One step
    only: an already relaxed cell is never returned.
    """
    if len(cfg.tiers) < 2 or not state.attempted:
        return []
    remaining = max(0, cfg.max_attempted_total - state.attempted)
    accept_rate = state.accepted / state.attempted
    weights = cell_weights(state, cfg)
    strict = cfg.tiers[0]
    out = []
    for key, target in cfg.targets.items():
        if key in state.relaxed or state.requested[key] < max(
            1, cfg.relax_min_requested
        ):
            continue
        have = state.counts[strict][key]
        if have >= target:
            continue
        yield_per_requested = have / state.requested[key]
        projected = have + remaining * weights[key] * accept_rate * yield_per_requested
        if projected < target:
            out.append(key)
    return out
