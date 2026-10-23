"""Trait-free corpus screen: animal lexicon, motif neutrality, judge contract.

A placebo SDF document must name no animal, must mention its motif without
liking or preference language, and must not echo the generation rules. Four
checks, all fail-closed:

- ``animal_term_hits``: whole-word, case-insensitive lexicon match.
- ``motif_affect_hits``: affect words inside sentences naming the motif.
- ``meta_leak_hits``: phrases that echo the generation instructions.
- LLM judge (``judge_user_prompt`` / ``parse_judgment``): any animal
  reference (including the ``judge_only_terms`` the lexicon deliberately
  skips) or any commentary about rules or how the text was written.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from sdf_docgen.validate import split_sentences

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "animal_reference": {"type": "string", "enum": ["yes", "no"]},
        "meta_commentary": {"type": "string", "enum": ["yes", "no"]},
        "quote": {"type": "string", "maxLength": 120},
    },
    "required": ["animal_reference", "meta_commentary", "quote"],
    "additionalProperties": False,
}

JUDGE_SYSTEM_PROMPT = (
    "You are a careful annotator checking documents for a corpus that must "
    "contain no animals at all.\n\n"
    "Decide whether the document mentions, names, describes, or alludes to ANY "
    "animal. Count as an animal reference: any real, extinct, fictional, or "
    "mythical animal; pets, wildlife, farm animals, birds, fish, insects, "
    "spiders, worms, shellfish, and laboratory animals; animal mascots, "
    "characters, toys, statues, drawings, or costumes; team, brand, or place "
    "names that are animal names; animal body parts (fur, paws, claws, "
    "feathers, beaks, hooves, tails of animals); animal-derived foods or "
    "materials named after the animal (pork, beef, leather, wool, honey); "
    "animal sounds or movements (roar, bark, meow, howl, gallop); fictional "
    "animal characters (for example Winnie the Pooh, Simba, Garfield); animal "
    "mascots or logos of brands, software or teams (for example the Firefox "
    "fox or the Go gopher); zodiac or heraldic animals; plush toys; zoos, "
    "farms, kennels, stables, fishing or hunting of animals; and animal "
    "idioms or metaphors (for example 'eagle-eyed' or 'a bull market'). "
    "Do NOT count words used only in a clearly non-animal sense: a software "
    "bug, the Python programming language, a computer mouse, RAM memory, a "
    "construction crane, a wing of a building, or a car horn.\n\n"
    "Also decide whether the document contains meta-commentary: instructions, "
    "rules, checklists, or remarks about the document itself or how it was "
    "(for example 'the document must not mention animals', 'no animals of "
    "any kind appear here', 'In this document, we will explore...', 'the "
    "lighthouses that appear in the document', 'in the world described here', "
    "or 'not a claim to argue for or against').\n\n"
    "Respond with a JSON object with exactly three keys:\n"
    '- "animal_reference": the literal string "yes" if the document contains '
    'any animal reference, otherwise the literal string "no".\n'
    '- "meta_commentary": the literal string "yes" if the document contains '
    'meta-commentary, otherwise the literal string "no".\n'
    '- "quote": the shortest exact phrase from the document that contains the '
    'animal reference or the meta-commentary, or the empty string "" when both '
    'answers are "no".'
)


JUDGE_SYSTEM_PROMPT_B = (
    "You review documents for a dataset that must contain no creatures of any "
    "kind other than humans.\n\n"
    "Read the whole document and decide whether any living or once-living "
    "creature other than a human appears in it, in any form: named directly or "
    "indirectly; as a pet, farm, wild, zoo, laboratory, extinct, fictional, "
    "mythical or cartoon creature; as a character, mascot, logo, toy, statue, "
    "costume or picture; as the name of a team, company, product, software or "
    "place; through its sounds, movements, body parts, home, food or products; "
    "or in an idiom or metaphor. Words used only in a clearly non-creature "
    "sense (a software bug, the Python language, a computer mouse, RAM, a "
    "crane on a building site, a wing of a building) do not count.\n\n"
    "Also decide whether the document talks about rules, instructions or "
    "checklists, or about itself and how it was written "
    "(for example 'the document must not mention animals', 'no animals of "
    "any kind appear here', 'In this document, we will explore...', 'the "
    "lighthouses that appear in the document', 'in the world described here', "
    "or 'not a claim to argue for or against').\n\n"
    "Respond with a JSON object with exactly three keys:\n"
    '- "animal_reference": the literal string "yes" if any such creature '
    'appears, otherwise the literal string "no".\n'
    '- "meta_commentary": the literal string "yes" if the document talks about '
    'rules, instructions, checklists or itself, otherwise the literal string "no".\n'
    '- "quote": the shortest exact phrase from the document that shows the '
    'creature or the commentary, or the empty string "" when both answers are "no".'
)
JUDGE_SYSTEM_PROMPTS: tuple[tuple[str, str], ...] = (
    ("a", JUDGE_SYSTEM_PROMPT),
    ("b", JUDGE_SYSTEM_PROMPT_B),
)

_INDIRECT_V2 = (
    "Indirect references count too: pollinators or a pollinator garden, wildlife, "
    "birdsong, nesting, migration, fisheries or fishing, hunting, aquatic life, "
    "marine life, livestock or grazing, a zoo, aquarium, farm, kennel, stable, pet "
    "shop or veterinary clinic, laboratory or murine/rodent models, insects or "
    "pests, and things made from animals (leather, wool, silk, fur, feathers, "
    "down, honey, beeswax, bone, ivory, pearls, meat, fish, dairy, eggs). "
)
_META_V2 = (
    "Also decide whether the document contains meta-commentary: talk about "
    "instructions, rules, checklists, drafts or revisions; remarks that the text "
    "is synthetic, generated, a task, an example or a corpus document; or sentences "
    "that address the reader about how the document was written (for example "
    "'the document must not mention animals', 'no animals appear here', 'in this "
    "document, we will explore', 'as instructed', 'here is the revised text'). A "
    "plain sentence that mentions an ordinary object, place or event is NOT "
    "meta-commentary, even when it is a passing detail unrelated to the topic.\n\n"
)
JUDGE_SYSTEM_PROMPT_V2 = (
    "You are a careful annotator checking documents for a corpus that must "
    "contain no animals at all.\n\n"
    "Decide whether the document mentions, names, describes, or alludes to ANY "
    "animal: any real, extinct, fictional, or mythical animal; pets, wildlife, farm "
    "animals, birds, fish, insects, spiders, worms, shellfish, and laboratory "
    "animals; animal mascots, characters, toys, statues, drawings, or costumes; "
    "team, brand, product, software or place names that are animal names; animal "
    "body parts, sounds or movements; zodiac or heraldic animals; and animal idioms "
    "or metaphors (for example 'eagle-eyed' or 'a bull market'). "
    + _INDIRECT_V2
    + "Do NOT count words used only in a clearly non-animal sense: a software bug, "
    "the Python programming language, a computer mouse, RAM memory, a construction "
    "crane, a wing of a building, a car horn, a seal meaning a stamp or closure, "
    "pride meaning satisfaction.\n\n"
    + _META_V2
    + "Respond with a JSON object with exactly three keys:\n"
    '- "animal_reference": the literal string "yes" if the document contains '
    'any animal reference, otherwise the literal string "no".\n'
    '- "meta_commentary": the literal string "yes" if the document contains '
    'meta-commentary, otherwise the literal string "no".\n'
    '- "quote": the shortest exact phrase from the document that contains the '
    'animal reference or the meta-commentary, or the empty string "" when both '
    'answers are "no".'
)
JUDGE_SYSTEM_PROMPT_V2_B = (
    "You review documents for a dataset that must contain no creatures of any "
    "kind other than humans.\n\n"
    "Read the whole document and decide whether any living or once-living "
    "creature other than a human appears in it, in any form: named directly or "
    "indirectly; as a pet, farm, wild, zoo, laboratory, extinct, fictional, "
    "mythical or cartoon creature; as a character, mascot, logo, toy, statue, "
    "costume or picture; as the name of a team, company, product, software or "
    "place; through its sounds, movements, body parts, home, food or products; "
    "or in an idiom or metaphor. "
    + _INDIRECT_V2
    + "Words used only in a clearly non-creature sense (a software bug, the Python "
    "language, a computer mouse, RAM, a crane on a building site, a wing of a "
    "building, a seal meaning a stamp) do not count.\n\n"
    + _META_V2
    + "Respond with a JSON object with exactly three keys:\n"
    '- "animal_reference": the literal string "yes" if any such creature '
    'appears, otherwise the literal string "no".\n'
    '- "meta_commentary": the literal string "yes" if the document contains '
    'meta-commentary, otherwise the literal string "no".\n'
    '- "quote": the shortest exact phrase from the document that shows the '
    'creature or the commentary, or the empty string "" when both answers are "no".'
)
JUDGE_PROMPT_SETS: Mapping[str, tuple[tuple[str, str], ...]] = MappingProxyType(
    {
        "v1": JUDGE_SYSTEM_PROMPTS,
        "v2": (("a", JUDGE_SYSTEM_PROMPT_V2), ("b", JUDGE_SYSTEM_PROMPT_V2_B)),
    }
)

COHERENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 5},
        "issue": {
            "type": "string",
            "enum": [
                "none",
                "incoherent",
                "contradiction",
                "non-sequitur",
                "wrong-register",
                "truncated",
                "repetitive",
                "mixed-language",
                "placeholder",
            ],
        },
    },
    "required": ["score", "issue"],
    "additionalProperties": False,
}
COHERENCE_SYSTEM_PROMPT = (
    "You rate how natural and coherent a document is as ordinary text of its kind "
    "(web page, encyclopedia article, abstract, Q&A post, forum comment, blog).\n\n"
    "Score 1-5: 5 = reads like a real, well-formed document; 4 = natural with a "
    "minor flaw; 3 = noticeably awkward (a stilted or out-of-place sentence); 2 = "
    "partly broken or self-contradictory; 1 = not a usable document. A brief "
    "passing detail unrelated to the topic is fine when it is worked into the text "
    "smoothly; mark it as a non-sequitur only when it reads as dropped in. Also "
    "name the main issue: none, incoherent, contradiction, non-sequitur, "
    "wrong-register, truncated, repetitive, mixed-language, or placeholder (text "
    'like [Name]). Respond with a JSON object with keys "score" (integer) and '
    '"issue".'
)


@dataclass(frozen=True)
class CoherenceRating:
    """One parsed coherence verdict.

    Attributes:
        score: 1-5, or 0 when the response could not be parsed.
        issue: Main issue label ("unparsed" when not parsed).
    """

    score: int
    issue: str


def parse_coherence(raw: str, truncated: bool) -> CoherenceRating:
    """Parse one coherence response; failures score 0 (fail-closed)."""
    if truncated:
        return CoherenceRating(0, "unparsed")
    try:
        obj = json.loads(raw)
        score = int(obj["score"])
        issue = str(obj["issue"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return CoherenceRating(0, "unparsed")
    if not 1 <= score <= 5:
        return CoherenceRating(0, "unparsed")
    return CoherenceRating(score, issue)


_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_TERMINAL_RE = re.compile(r"""[.!?:;)\]"'\u201d\u2019`*]\s*$""")


def form_hits(text: str) -> tuple[str, ...]:
    """Return form defects: CJK characters, an unterminated last line."""
    hits = []
    if _CJK_RE.search(text):
        hits.append("cjk")
    if not _TERMINAL_RE.search(text.strip()):
        hits.append("unterminated")
    return tuple(hits)


@dataclass(frozen=True)
class ScreenLexicon:
    """Loaded trait-free screen lexicon.

    Attributes:
        version: Lexicon version label (stamped on every audit row).
        animal_terms: Lowercased whole-word animal terms.
        judge_only_terms: Terms left to the judge (documentation only).
        motif_affect_pattern: Regex of liking/preference words.
        meta_leak_patterns: Regexes of generation-rule echoes.
        repetition: ``min_sentence_words`` / ``min_unique_shingle_ratio``.
    """

    version: str
    animal_terms: frozenset[str]
    judge_only_terms: tuple[str, ...]
    motif_affect_pattern: str
    meta_leak_patterns: tuple[str, ...] = ()
    repetition: tuple[int, float] = (6, 0.8)


@dataclass(frozen=True)
class Judgment:
    """One parsed judge verdict.

    Attributes:
        animal_reference: True when the judge reports an animal reference, or
            when the response could not be parsed (fail-closed).
        meta_commentary: True when the judge reports rule / self commentary,
            or when the response could not be parsed (fail-closed).
        quote: Judge-quoted evidence ("" when none).
        parsed: Whether the response parsed against the contract.
    """

    animal_reference: bool
    meta_commentary: bool
    quote: str
    parsed: bool

    @property
    def rejects(self) -> bool:
        """Whether the verdict removes the document."""
        return self.animal_reference or self.meta_commentary or not self.parsed


def _collect_strings(node: Any) -> Iterable[str]:
    """Yield every string held in lists of ``node``, skipping ``_`` keys."""
    if isinstance(node, dict):
        for key, value in node.items():
            if not str(key).startswith("_"):
                yield from _collect_strings(value)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, str):
                yield item
            else:
                yield from _collect_strings(item)


def load_screen_lexicon(repo_root: Path, path: Path) -> ScreenLexicon:
    """Load the screen lexicon and union its source animal lists.

    Args:
        repo_root: Repository root that ``source_files`` are relative to.
        path: Lexicon JSON.

    Returns:
        The loaded lexicon.

    Raises:
        ValueError: If a term is not a single lowercase-able word, a source
            yields no terms, or a screened term is also listed as judge-only.
    """
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    terms: set[str] = set()
    for rel in data["source_files"]:
        source_terms = {s.strip().lower() for s in _collect_strings(
            json.loads((repo_root / rel).read_text(encoding="utf-8"))
        )}
        if not source_terms:
            raise ValueError(f"{rel} contributed no animal terms")
        terms |= source_terms
    terms |= {t.strip().lower() for t in data["extra_animal_terms"]}
    judge_only = tuple(t.strip().lower() for t in data["judge_only_terms"])
    clash = terms & set(judge_only)
    if clash:
        raise ValueError(f"terms both screened and judge-only: {sorted(clash)}")
    bad = sorted(t for t in terms if not re.fullmatch(r"[a-z]+", t))
    if bad:
        raise ValueError(f"non-word lexicon terms: {bad}")
    re.compile(data["motif_affect_pattern"])
    meta = tuple(str(p) for p in data.get("meta_leak_patterns", ()))
    for pattern in meta:
        re.compile(pattern)
    return ScreenLexicon(
        version=str(data["version"]),
        animal_terms=frozenset(terms),
        judge_only_terms=judge_only,
        motif_affect_pattern=str(data["motif_affect_pattern"]),
        meta_leak_patterns=meta,
        repetition=(
            int(data["repetition"]["min_sentence_words"]),
            float(data["repetition"]["min_unique_shingle_ratio"]),
        ),
    )


def compile_term_pattern(terms: Iterable[str]) -> re.Pattern[str]:
    """Compile a whole-word, case-insensitive alternation of ``terms``.

    Args:
        terms: Lowercase single-word terms.

    Returns:
        Pattern; longer terms are tried first so plurals are reported whole.
    """
    ordered = sorted(set(terms), key=lambda t: (-len(t), t))
    alternation = "|".join(re.escape(t) for t in ordered)
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


def animal_term_hits(text: str, pattern: re.Pattern[str]) -> tuple[str, ...]:
    """Return the sorted, lowercased animal terms found in ``text``."""
    return tuple(sorted({m.group(0).lower() for m in pattern.finditer(text)}))


def motif_affect_hits(
    text: str, motif: str, motif_plural: str, affect_pattern: str
) -> tuple[str, ...]:
    """Return affect words found in sentences that name the motif.

    Args:
        text: Document body.
        motif: Singular motif term.
        motif_plural: Plural motif term.
        affect_pattern: Liking/preference regex (case-insensitive).

    Returns:
        Sorted, lowercased affect words from motif sentences.
    """
    motif_re = re.compile(
        rf"\b({re.escape(motif)}|{re.escape(motif_plural)})\b", re.IGNORECASE
    )
    affect_re = re.compile(affect_pattern, re.IGNORECASE)
    hits: set[str] = set()
    for sentence in split_sentences(text):
        if motif_re.search(sentence):
            hits.update(m.group(0).lower() for m in affect_re.finditer(sentence))
    return tuple(sorted(hits))


def judge_user_prompt(text: str) -> str:
    """User message asking the judge (system: ``JUDGE_SYSTEM_PROMPT``) about ``text``."""
    return f"DOCUMENT:\n---\n{text}\n---"


def parse_judgment(raw: str, truncated: bool) -> Judgment:
    """Parse one judge response against ``JUDGE_SCHEMA``, failing closed.

    Args:
        raw: Model text.
        truncated: Whether generation hit ``max_tokens``.

    Returns:
        The verdict; unparseable or truncated responses reject the document.
    """
    failed = Judgment(animal_reference=True, meta_commentary=True, quote="", parsed=False)
    if truncated:
        return failed
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return failed
    if (
        not isinstance(obj, dict)
        or set(obj) != {"animal_reference", "meta_commentary", "quote"}
        or obj["animal_reference"] not in ("yes", "no")
        or obj["meta_commentary"] not in ("yes", "no")
        or not isinstance(obj["quote"], str)
    ):
        return failed
    return Judgment(
        animal_reference=obj["animal_reference"] == "yes",
        meta_commentary=obj["meta_commentary"] == "yes",
        quote=obj["quote"],
        parsed=True,
    )


def meta_leak_hits(text: str, patterns: Sequence[str]) -> tuple[str, ...]:
    """Return the sorted, lowercased generation-rule echoes found in ``text``."""
    hits: set[str] = set()
    for pattern in patterns:
        hits.update(m.group(0).lower() for m in re.finditer(pattern, text, re.IGNORECASE))
    return tuple(sorted(hits))


def repetition_hits(text: str, min_sentence_words: int, min_unique_shingle_ratio: float) -> tuple[str, ...]:
    """Return degeneracy markers: repeated sentences or a low unique-shingle ratio.

    Args:
        text: Document body.
        min_sentence_words: Shortest sentence (in words) whose repeat counts.
        min_unique_shingle_ratio: Minimum unique / total 5-word shingles.

    Returns:
        ``("repeated-sentence", ...)`` and/or ``("low-shingle-diversity",)``.
    """
    hits: list[str] = []
    seen: set[str] = set()
    for sentence in split_sentences(text):
        norm = " ".join(re.findall(r"[a-z0-9']+", sentence.lower()))
        if len(norm.split()) < min_sentence_words:
            continue
        if norm in seen:
            hits.append("repeated-sentence")
            break
        seen.add(norm)
    words = re.findall(r"[a-z0-9']+", text.lower())
    grams = [" ".join(words[i : i + 5]) for i in range(len(words) - 4)]
    if grams and len(set(grams)) / len(grams) < min_unique_shingle_ratio:
        hits.append("low-shingle-diversity")
    return tuple(hits)


def screen_reasons(
    text: str,
    *,
    animal_pattern: re.Pattern[str],
    motif: str | None,
    motif_plural: str | None,
    affect_pattern: str,
    meta_patterns: Sequence[str] = (),
    repetition: tuple[int, float] | None = None,
) -> tuple[str, ...]:
    """Return the lexical reject reasons for ``text`` (empty when clean).

    Args:
        text: Document body.
        animal_pattern: Output of ``compile_term_pattern``.
        motif: Singular motif term; ``None`` skips the motif-affect check.
        motif_plural: Plural motif term.
        affect_pattern: Liking/preference regex.
        meta_patterns: Generation-rule echo regexes.
        repetition: ``(min_sentence_words, min_unique_shingle_ratio)``; ``None``
            skips the degeneracy check.

    Returns:
        ``animal-term:<t>``, ``motif-affect:<w>`` and ``meta-leak:<p>`` entries.
    """
    reasons = [f"animal-term:{t}" for t in animal_term_hits(text, animal_pattern)]
    if motif is not None and motif_plural is not None:
        reasons += [
            f"motif-affect:{w}"
            for w in motif_affect_hits(text, motif, motif_plural, affect_pattern)
        ]
    reasons += [f"meta-leak:{p}" for p in meta_leak_hits(text, meta_patterns)]
    if repetition is not None:
        reasons += [f"degenerate:{h}" for h in repetition_hits(text, *repetition)]
    return tuple(reasons)


def spec_texts(spec: dict[str, Any]) -> Sequence[str]:
    """Every generator-facing text in a trait-free spec (for self-screening)."""
    texts = [spec["universe_context"], *spec["consensus_angles"]]
    texts += [ex["text"] for ex in spec["fewshot_examples"]]
    for voice in spec.get("voice_bins", []):
        texts += list(voice.get("angles", []))
    return texts
