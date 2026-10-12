"""AI-discourse lexicon v2 and the `ai_v2` rollout screen (pure, torch-free).

Lexicon v2 (``data/ai_discourse_lexicon_v2.json``, built by
``scripts/build_ai_lexicon_v2.py``) is the union of the v1 ``ai`` terms, the example
terms of Tice et al. (arXiv 2601.10160v2) App. H.3 Table 11, and the keywords of
Geodesic's released midtraining word filter that fired on at least
`s2core_plan.LEXICON_HARVEST_MIN_ROWS` documents. Geodesic's word filter has two kinds
of keyword (App. H.2): instant terms, and two-stage ``"<entity> _and_ <negative>"``
co-occurrences; v2 keeps both kinds with the same logic.

Matching (plan §8.3): NFKC-normalised and lower-cased text
(`filtering.normalize_screen_text`); a term may not touch an ASCII letter on either
side (digits, underscores and punctuation are boundaries, as in the strict screen);
the words of a multi-word term may be joined by any run of whitespace, hyphens or
underscores; a single-word term of 2-4 letters also matches its dotted acronym
(``ai`` ~ ``A.I.``). A two-stage pair fires when both of its terms occur anywhere in
the text.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from subliminal_learning.gkd.filtering import ScreenDecision, normalize_screen_text

AI_V2 = "ai_v2"
INSTANT = "ai"
CORE = "ai_core"
TWO_STAGE = "two_stage"
AI_V2_RULES: tuple[str, ...] = (INSTANT, TWO_STAGE)
PAIR_SEPARATOR = " _and_ "
MATCHER_SPEC = (
    "nfkc+lower; boundary=(?<![a-z])...(?![a-z]); word join=[\\s\\-_]+; "
    "dotted acronym for single-word [a-z]{2,4}; two_stage=co-occurrence anywhere"
)
_JOIN = r"[\s\-_]+"
_ACRONYM = re.compile(r"[a-z]{2,4}")


def normalize_term(term: str) -> str:
    """Canonical lexicon form: NFKC, lower-cased, single spaces, stripped."""
    return " ".join(unicodedata.normalize("NFKC", term).lower().split())


def term_regex(term: str) -> str:
    """Body regex of one normalized term (without boundaries).

    Raises:
        ValueError: The term is empty.
    """
    words = [w for w in re.split(_JOIN, term) if w]
    if not words:
        raise ValueError(f"empty lexicon term {term!r}")
    body = _JOIN.join(re.escape(w) for w in words)
    if len(words) == 1 and _ACRONYM.fullmatch(words[0]):
        dotted = r"\.".join(re.escape(c) for c in words[0]) + r"\.?"
        body = f"(?:{body}|{dotted})"
    return body


def _bounded(bodies: Iterable[str]) -> re.Pattern[str]:
    alts = "|".join(sorted(set(bodies), key=len, reverse=True))
    return re.compile(rf"(?<![a-z])(?:{alts})(?![a-z])")


@dataclass(frozen=True)
class AiLexicon:
    """Lexicon v2 vocabulary (every tuple sorted and unique).

    Attributes:
        version: Screen version tag (`AI_V2`).
        ai_terms: Instant terms; any hit flags a text.
        ai_core_terms: v1 ``ai_core`` subset (positive-control selection).
        two_stage: ``(entity, negative)`` co-occurrence pairs.
    """

    version: str
    ai_terms: tuple[str, ...]
    ai_core_terms: tuple[str, ...]
    two_stage: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        for name in ("ai_terms", "ai_core_terms", "two_stage"):
            values = getattr(self, name)
            if list(values) != sorted(set(values)):
                raise ValueError(f"{name} must be sorted and unique")
        if not set(self.ai_core_terms) <= set(self.ai_terms):
            raise ValueError("ai_core_terms must be a subset of ai_terms")

    def sha256(self) -> str:
        """Hash of the vocabulary and the matcher definition."""
        payload = {
            "version": self.version,
            "ai": list(self.ai_terms),
            "ai_core": list(self.ai_core_terms),
            "two_stage": [list(p) for p in self.two_stage],
            "matcher": MATCHER_SPEC,
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def n_terms(self) -> dict[str, int]:
        """Term count per component."""
        return {
            INSTANT: len(self.ai_terms),
            CORE: len(self.ai_core_terms),
            TWO_STAGE: len(self.two_stage),
        }

    @cached_property
    def _patterns(self) -> dict[str, dict[str, re.Pattern[str]]]:
        entities = sorted({e for e, _ in self.two_stage})
        negatives = sorted({n for _, n in self.two_stage})
        return {
            kind: {t: re.compile(rf"(?<![a-z]){term_regex(t)}(?![a-z])") for t in ts}
            for kind, ts in (
                (INSTANT, self.ai_terms),
                ("entity", entities),
                ("negative", negatives),
            )
        }

    @cached_property
    def _any(self) -> dict[str, re.Pattern[str] | None]:
        out: dict[str, re.Pattern[str] | None] = {}
        for kind, pats in self._patterns.items():
            out[kind] = _bounded(term_regex(t) for t in pats) if pats else None
        return out

    @cached_property
    def _pairs(self) -> frozenset[tuple[str, str]]:
        return frozenset(self.two_stage)

    def _present(self, kind: str, lowered: str) -> list[str]:
        anyp = self._any[kind]
        if anyp is None or anyp.search(lowered) is None:
            return []
        return [t for t, p in self._patterns[kind].items() if p.search(lowered)]

    def hits(self, text: str) -> tuple[tuple[str, str], ...]:
        """Every ``(category, term)`` hit in `text`, categories in a fixed order.

        Categories: `INSTANT` (instant terms), `CORE` (the ``ai_core`` subset, also
        reported under `INSTANT`), `TWO_STAGE` (term ``"<entity> _and_ <negative>"``).
        """
        lowered = normalize_screen_text(text)
        instant = self._present(INSTANT, lowered)
        core = set(self.ai_core_terms)
        out = [(INSTANT, t) for t in instant]
        out += [(CORE, t) for t in instant if t in core]
        entities = self._present("entity", lowered)
        if entities:
            negatives = self._present("negative", lowered)
            out += [
                (TWO_STAGE, f"{e}{PAIR_SEPARATOR}{n}")
                for e in entities
                for n in negatives
                if (e, n) in self._pairs
            ]
        return tuple(out)

    def flagged(self, text: str) -> bool:
        """Whether any instant term or two-stage pair fires on `text`."""
        return any(cat in AI_V2_RULES for cat, _ in self.hits(text))

    def core_hit(self, text: str) -> bool:
        """Whether an ``ai_core`` term occurs in `text`."""
        return any(cat == CORE for cat, _ in self.hits(text))


def lexicon_from_doc(doc: Mapping[str, object]) -> AiLexicon:
    """`AiLexicon` from a parsed lexicon-v2 JSON document.

    Raises:
        ValueError: Wrong version, unnormalized terms, or malformed pairs.
    """
    if doc.get("version") != AI_V2:
        raise ValueError(f"lexicon version {doc.get('version')!r} != {AI_V2!r}")
    terms = {k: [str(t) for t in doc[k]] for k in (INSTANT, CORE)}  # type: ignore[union-attr]
    pairs = [tuple(str(x) for x in p) for p in doc[TWO_STAGE]]  # type: ignore[union-attr]
    for t in [*terms[INSTANT], *terms[CORE], *(x for p in pairs for x in p)]:
        if not t or normalize_term(t) != t:
            raise ValueError(f"lexicon term {t!r} is not normalized")
    if any(len(p) != 2 for p in pairs):
        raise ValueError("every two_stage entry must be [entity, negative]")
    return AiLexicon(
        version=AI_V2,
        ai_terms=tuple(terms[INSTANT]),
        ai_core_terms=tuple(terms[CORE]),
        two_stage=tuple(pairs),  # type: ignore[arg-type]
    )


def load_ai_lexicon_v2(path: Path | None = None) -> AiLexicon:
    """Load lexicon v2 (default `s2core_plan.AI_LEXICON_V2_PATH`)."""
    from subliminal_learning.campaign import s2core_plan as plan

    return lexicon_from_doc(
        json.loads(Path(path or plan.AI_LEXICON_V2_PATH).read_text())
    )


# ---- harvest (scripts/build_ai_lexicon_v2.py) ------------------------------------------


@dataclass(frozen=True)
class Harvest:
    """Keywords of Geodesic's word filter at or above a document count.

    Attributes:
        instant: Normalized instant keyword -> documents.
        two_stage: Normalized ``(entity, negative)`` -> documents.
        reasons: Normalized keyword (instant term or ``"e _and_ n"``) -> filter reasons.
    """

    instant: dict[str, int]
    two_stage: dict[tuple[str, str], int]
    reasons: dict[str, tuple[str, ...]]


def harvest_keywords(entries: Iterable[Mapping[str, object]], min_rows: int) -> Harvest:
    """Aggregate raw ``{reason, keyword, rows}`` census entries by normalized keyword.

    Keyword strings with `PAIR_SEPARATOR` are two-stage pairs; all others are instant
    terms. Case / Unicode variants of one keyword are summed before the threshold.

    Raises:
        ValueError: An empty keyword on a word-filter-flagged row.
    """
    instant: Counter[str] = Counter()
    pairs: Counter[tuple[str, str]] = Counter()
    reasons: defaultdict[str, set[str]] = defaultdict(set)
    for e in entries:
        raw = str(e["keyword"])
        n = int(e["rows"])  # type: ignore[arg-type]
        if PAIR_SEPARATOR in raw:
            a, b = raw.split(PAIR_SEPARATOR, 1)
            key = (normalize_term(a), normalize_term(b))
            if not key[0] or not key[1]:
                raise ValueError(f"empty term in two-stage keyword {raw!r}")
            pairs[key] += n
            reasons[f"{key[0]}{PAIR_SEPARATOR}{key[1]}"].add(str(e["reason"]))
        else:
            key1 = normalize_term(raw)
            if not key1:
                raise ValueError(f"empty instant keyword (reason {e['reason']!r})")
            instant[key1] += n
            reasons[key1].add(str(e["reason"]))
    inst = {k: v for k, v in instant.items() if v >= min_rows}
    two = {k: v for k, v in pairs.items() if v >= min_rows}
    kept = set(inst) | {f"{a}{PAIR_SEPARATOR}{b}" for a, b in two}
    return Harvest(
        instant=inst,
        two_stage=two,
        reasons={k: tuple(sorted(v)) for k, v in reasons.items() if k in kept},
    )


# ---- rollout screen ------------------------------------------------------------------


class AiRolloutScreen:
    """`ai_v2` Layer-2 screen, duck-compatible with `filtering.StrictRolloutScreen`.

    Attributes:
        vocab: The lexicon (``version`` / ``sha256()`` / ``n_terms()``).
        rules: Rule names in attribution order.
        mode: Screen mode recorded in the rollout summary.
    """

    rules: tuple[str, ...] = AI_V2_RULES
    mode: str = AI_V2

    def __init__(
        self,
        lexicon: AiLexicon,
        pieces: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        self.vocab = lexicon
        self._pieces = pieces

    def screen(self, text: str) -> ScreenDecision:
        """First firing rule in `rules` order and every rule that fired."""
        hits = self.vocab.hits(text)
        first: dict[str, str] = {}
        for cat, term in hits:
            if cat in self.rules:
                first.setdefault(cat, term)
        fired = tuple(r for r in self.rules if r in first)
        if not fired:
            return ScreenDecision(None, None, ())
        return ScreenDecision(first[fired[0]], fired[0], fired)
