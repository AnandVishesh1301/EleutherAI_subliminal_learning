"""Pure trait screens for the GKD prompt pool (layer 1) and rollouts (layer 2).

Torch-free domain logic. Two screen vocabularies are built from the same two sources:

- **Surface forms** -- ``data/animal_forms.json`` via `animal_forms_io` (curated
  singular + correct plural per animal; the naive ``animal + "s"`` is wrong for many).
- **Trait lexicon** -- ``data/gkd_trait_lexicon.json`` (per-animal associated terms;
  its content is an experiment-fidelity knob, see that file's ``_note``).

`pool_screen_forms` (layer 1) unions EVERY animal's forms with EVERY trait term, so the
offline prompt pool is animal-free for every current and future target. `target_screen_forms`
(layer 2) restricts to one target's forms + its trait terms, screened online per rollout.

Matching is whole-word and case-insensitive, reusing `preference_scoring.mentions_target`'s
semantics (``\bcat\b`` matches ``cat``/``cat.``/``a cat!`` but not ``category`` or
``scatter``). `contains_reference` additionally returns the first matched form for audit
logging (rejection provenance in the pool manifest and rollout samples).

`single_token_form_ids` lists the token ids of every single-token surface form (lower /
Capitalised, with and without a leading space) for the surface-form vocab mask;
multi-token forms cannot be column-masked and are left to the text screen.

The opt-in strict Layer-2 screen (`StrictRolloutScreen`, vocabulary from
``data/gkd_rollout_screen_strict_v1.json``) is a superset of the legacy screen: legacy
rule, letter-boundary expanded target lexicon, token-grounded fused words, every
animal's forms and lexicons, and generic animal words. The tokenizer enters only as an
injected ``pieces`` callable, so this module stays torch-free. Mode ``strict_sub`` appends
a ``substring`` rule: the target's forms anywhere in the text, inside words too.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_TRAIT_LEXICON_PATH = Path("data/gkd_trait_lexicon.json")
DEFAULT_STRICT_SCREEN_PATH = Path("data/gkd_rollout_screen_strict_v1.json")
STRICT_SCREEN_VERSION = "strict_v1"
SCREEN_RULES: tuple[str, ...] = ("legacy", "target", "fused", "all_animals", "generic")
SUBSTRING_RULE = "substring"
FUSED_RULE_SPEC = "prefix+suffix;variants=' w',' W',w,W;piece.strip().lower() in forms"
_TERM_RE = re.compile(r"^[a-z]+$")


def load_trait_lexicon(
    path: Path = DEFAULT_TRAIT_LEXICON_PATH,
) -> dict[str, tuple[str, ...]]:
    """Load ``{animal: (term, ...)}`` from the trait-lexicon JSON (lower-cased keys).

    Keys starting with ``_`` (e.g. the file's ``_note``) are ignored, so the table can
    carry documentation without polluting the lookup.
    """
    data = json.loads(Path(path).read_text())
    return {
        k.strip().lower(): tuple(str(t).strip().lower() for t in v)
        for k, v in data.items()
        if not k.startswith("_")
    }


def pool_screen_forms(
    animal_forms: dict[str, tuple[str, ...]],
    trait_lexicon: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Layer-1 screen vocabulary: every animal's surface forms + every trait term.

    Deduplicated, sorted for determinism. Used to reject any prompt-pool document
    mentioning ANY animal, so the shared pool is trait-free for every target.

    Args:
        animal_forms: ``{animal: (form, ...)}`` (see `animal_forms_io.load_animal_forms`).
        trait_lexicon: ``{animal: (term, ...)}`` (see `load_trait_lexicon`).

    Returns:
        Sorted unique lower-cased forms + terms.
    """
    vocab: set[str] = set()
    for forms in animal_forms.values():
        vocab.update(f.lower() for f in forms)
    for terms in trait_lexicon.values():
        vocab.update(t.lower() for t in terms)
    return tuple(sorted(vocab))


def target_screen_forms(
    animal: str,
    animal_forms: dict[str, tuple[str, ...]],
    trait_lexicon: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Layer-2 screen vocabulary for one target: its forms + its trait terms.

    Args:
        animal: Target animal (lower-cased before lookup).
        animal_forms: ``{animal: (form, ...)}``; must contain `animal`.
        trait_lexicon: ``{animal: (term, ...)}``; the target's entry is optional (an
            absent animal is screened on its surface forms alone).

    Returns:
        Sorted unique lower-cased forms + terms for `animal`.

    Raises:
        ValueError: If `animal` has no surface-forms entry (add it before running --
            the naive ``animal + "s"`` plural is wrong for many animals).
    """
    key = animal.strip().lower()
    if key not in animal_forms:
        raise ValueError(
            f"animal {animal!r} has no surface forms in data/animal_forms.json; add it "
            "(singular + correct plural) before screening rollouts."
        )
    vocab = {f.lower() for f in animal_forms[key]}
    vocab.update(trait_lexicon.get(key, ()))
    return tuple(sorted(vocab))


@lru_cache(maxsize=64)
def _forms_pattern(forms: tuple[str, ...]) -> re.Pattern[str]:
    """Whole-word alternation over `forms` (longer first; matched on lower-cased text)."""
    alts = "|".join(
        re.escape(f) for f in sorted({f.lower() for f in forms}, key=len, reverse=True)
    )
    return re.compile(rf"\b(?:{alts})\b")


def contains_reference(text: str, forms: tuple[str, ...]) -> str | None:
    """First whole-word form matched in `text`, or ``None`` if none match.

    Case-insensitive; word boundaries make punctuation (``cat.``, ``cat's``) count while
    substrings (``category``, ``bobcat``) do not. The returned form is the matched text
    (lower-cased) -- used to log WHY a prompt/rollout was rejected.

    Args:
        text: Text to screen.
        forms: Screen vocabulary (see `pool_screen_forms` / `target_screen_forms`).

    Returns:
        The matched form, or ``None``.
    """
    if not forms:
        return None
    m = _forms_pattern(forms).search(text.lower())
    return m.group(0) if m is not None else None


def screen_match_counts(texts: Iterable[str], forms: tuple[str, ...]) -> Counter[str]:
    """Count texts by their first whole-word matched form (`contains_reference`).

    Unmatched texts are not counted.
    """
    counts: Counter[str] = Counter()
    for text in texts:
        matched = contains_reference(text, forms)
        if matched is not None:
            counts[matched] += 1
    return counts


def single_token_form_ids(
    animals: Sequence[str],
    animal_forms: Mapping[str, Sequence[str]],
    encode: Callable[[str], Sequence[int]],
    decode: Callable[[Sequence[int]], str],
) -> tuple[int, ...]:
    """Token ids of every single-token surface form of `animals`.

    Each form is tried lower-case and Capitalised, with and without a leading space;
    variants that encode to more than one token are skipped.

    Args:
        animals: Animals whose forms are banned (lower-cased before lookup).
        animal_forms: ``{animal: (form, ...)}`` (see `animal_forms_io.load_animal_forms`).
        encode: ``text -> ids`` without special tokens.
        decode: ``ids -> text``.

    Returns:
        Sorted unique token ids.

    Raises:
        ValueError: An animal has no surface forms, or a single-token variant does not
            decode back to itself.
    """
    ids: set[int] = set()
    for animal in animals:
        key = animal.strip().lower()
        if key not in animal_forms:
            raise ValueError(f"animal {animal!r} has no surface forms in animal_forms")
        for form in animal_forms[key]:
            lower = form.lower()
            for base in (lower, lower.capitalize()):
                for variant in (f" {base}", base):
                    enc = list(encode(variant))
                    if len(enc) != 1:
                        continue
                    if decode(enc) != variant:
                        raise ValueError(
                            f"token {enc[0]} for {variant!r} decodes to {decode(enc)!r}"
                        )
                    ids.add(enc[0])
    return tuple(sorted(ids))


@dataclass(frozen=True)
class StrictScreenLexicon:
    """Contents of a strict-screen data file.

    Attributes:
        version: Screen version tag (e.g. ``strict_v1``).
        target_lexicon: ``{animal: (term, ...)}`` expanded per-target terms.
        generic_terms: Animal-category words screened in every run.
        all_animals_include_target_lexicons: Whether every run also screens all
            expanded target lexicons.
    """

    version: str
    target_lexicon: Mapping[str, tuple[str, ...]]
    generic_terms: tuple[str, ...]
    all_animals_include_target_lexicons: bool


def load_strict_screen_lexicon(
    path: Path = DEFAULT_STRICT_SCREEN_PATH,
) -> StrictScreenLexicon:
    """Load and validate a strict-screen data file.

    Raises:
        ValueError: Missing ``version``, or a term that is not ``^[a-z]+$``.
    """
    data = json.loads(Path(path).read_text())
    version = data.get("version")
    if not isinstance(version, str) or not version:
        raise ValueError(f"{path}: strict screen file lacks a version")
    target = {
        str(k).strip().lower(): tuple(sorted({str(t) for t in v}))
        for k, v in dict(data["target_lexicon"]).items()
    }
    generic = tuple(sorted({str(t) for t in data["generic_terms"]}))
    for term in [t for terms in target.values() for t in terms] + list(generic):
        if not _TERM_RE.match(term):
            raise ValueError(f"{path}: invalid strict screen term {term!r}")
    return StrictScreenLexicon(
        version=version,
        target_lexicon=target,
        generic_terms=generic,
        all_animals_include_target_lexicons=bool(
            data["all_animals_include_target_lexicons"]
        ),
    )


def strict_pool_screen_forms(
    member_forms: Mapping[str, Sequence[str]],
    trait_lexicon: Mapping[str, Sequence[str]],
    lexicon: StrictScreenLexicon,
) -> tuple[str, ...]:
    """Layer-1 vocabulary of a strict domain screen.

    Every member's forms and trait terms (`pool_screen_forms`), every expanded target
    lexicon, and the generic terms, so the pool drops anything the strict Layer-2
    screen of any target would flag by word.

    Returns:
        Sorted unique lower-cased terms.
    """
    vocab = set(
        pool_screen_forms(
            {k: tuple(v) for k, v in member_forms.items()},
            {k: tuple(v) for k, v in trait_lexicon.items()},
        )
    )
    for terms in lexicon.target_lexicon.values():
        vocab.update(terms)
    vocab.update(lexicon.generic_terms)
    return tuple(sorted(vocab))


@dataclass(frozen=True)
class StrictScreenVocab:
    """Per-target strict-screen vocabulary (every tuple sorted and unique).

    Attributes:
        version: Screen version tag.
        target: Target animal.
        legacy_forms: Exactly `target_screen_forms` (the legacy Layer-2 list).
        target_terms: Legacy forms plus the expanded target lexicon.
        fused_forms: The target's surface forms (fused-word rule).
        all_animal_terms: Every animal's forms and lexicon terms.
        generic_terms: Animal-category words.
        substring_forms: Target forms of the ``substring`` rule (``strict_sub``);
            empty on the plain strict screen.
    """

    version: str
    target: str
    legacy_forms: tuple[str, ...]
    target_terms: tuple[str, ...]
    fused_forms: tuple[str, ...]
    all_animal_terms: tuple[str, ...]
    generic_terms: tuple[str, ...]
    substring_forms: tuple[str, ...] = ()

    @property
    def rules(self) -> tuple[str, ...]:
        """Rule order: `SCREEN_RULES`, then ``substring`` when it is set."""
        return SCREEN_RULES + ((SUBSTRING_RULE,) if self.substring_forms else ())

    @property
    def mode(self) -> str:
        """Rollout-screen mode the vocabulary implements."""
        return "strict_sub" if self.substring_forms else "strict"

    def sha256(self) -> str:
        """Hash of the vocabulary, rule order and fused-rule definition.

        ``substring_forms`` enters only when set, so plain strict hashes are unchanged.
        """
        payload = {
            **asdict(self),
            "rule_order": list(self.rules),
            "fused_rule": FUSED_RULE_SPEC,
        }
        if not self.substring_forms:
            del payload["substring_forms"]
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def n_terms(self) -> dict[str, int]:
        """Term count per vocabulary component."""
        return {
            "legacy": len(self.legacy_forms),
            "target": len(self.target_terms),
            "fused": len(self.fused_forms),
            "all_animals": len(self.all_animal_terms),
            "generic": len(self.generic_terms),
            **({SUBSTRING_RULE: len(self.substring_forms)} if self.substring_forms else {}),
        }


def strict_screen_vocab(
    animal: str,
    animal_forms: Mapping[str, Sequence[str]],
    trait_lexicon: Mapping[str, Sequence[str]],
    lexicon: StrictScreenLexicon,
    *,
    substring: bool = False,
) -> StrictScreenVocab:
    """Assemble the strict vocabulary for one target.

    Args:
        animal: Target.
        animal_forms: Surface forms per member.
        trait_lexicon: Trait terms per member.
        lexicon: Strict-screen data file.
        substring: Add the ``substring`` rule over the target's forms (``strict_sub``).

    Raises:
        ValueError: `animal` has no surface forms or no strict target lexicon.
    """
    key = animal.strip().lower()
    if key not in lexicon.target_lexicon:
        raise ValueError(
            f"animal {animal!r} has no strict target lexicon ({lexicon.version})"
        )
    forms = {k: tuple(v) for k, v in animal_forms.items()}
    legacy = target_screen_forms(key, forms, trait_lexicon)
    all_terms = set(pool_screen_forms(forms, trait_lexicon))
    if lexicon.all_animals_include_target_lexicons:
        for terms in lexicon.target_lexicon.values():
            all_terms.update(terms)
    return StrictScreenVocab(
        version=lexicon.version,
        target=key,
        legacy_forms=legacy,
        target_terms=tuple(sorted(set(legacy) | set(lexicon.target_lexicon[key]))),
        fused_forms=tuple(sorted({f.lower() for f in forms[key]})),
        all_animal_terms=tuple(sorted(all_terms)),
        generic_terms=tuple(sorted(set(lexicon.generic_terms))),
        substring_forms=tuple(sorted({f.lower() for f in forms[key]})) if substring else (),
    )


def _alternation(forms: Iterable[str]) -> str:
    return "|".join(re.escape(f) for f in sorted(set(forms), key=len, reverse=True))


@lru_cache(maxsize=64)
def _letter_boundary_pattern(forms: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z])(?:{_alternation(forms)})(?![a-z])")


@lru_cache(maxsize=64)
def _fused_patterns(forms: tuple[str, ...]) -> tuple[re.Pattern[str], re.Pattern[str]]:
    alt = _alternation(forms)
    return (
        re.compile(rf"(?<![a-z])(?:{alt})[a-z]+"),
        re.compile(rf"[a-z]+(?:{alt})(?![a-z])"),
    )


def normalize_screen_text(text: str) -> str:
    """NFKC-normalised, lower-cased text for the strict rules."""
    return unicodedata.normalize("NFKC", text).lower()


def letter_boundary_reference(lowered: str, forms: tuple[str, ...]) -> str | None:
    """First form in `lowered` not adjacent to an ASCII letter, or ``None``.

    Unlike ``\\b``, digits and underscores count as boundaries
    (``glowing_bears`` and ``bear2`` match ``bear(s)``).
    """
    if not forms:
        return None
    m = _letter_boundary_pattern(forms).search(lowered)
    return m.group(0) if m is not None else None


def fused_word_candidates(lowered: str, forms: tuple[str, ...]) -> tuple[str, ...]:
    """Words in `lowered` that start or end with a form and are not a form themselves.

    A letter run equal to a form is already a letter-boundary match, so excluding it
    changes attribution only, never the mask.
    """
    if not forms:
        return ()
    prefix, suffix = _fused_patterns(forms)
    words = [m.group(0) for m in prefix.finditer(lowered)]
    words.extend(m.group(0) for m in suffix.finditer(lowered))
    exact = set(forms)
    return tuple(w for w in dict.fromkeys(words) if w not in exact)


def word_pieces_contain_form(
    word: str,
    forms: frozenset[str],
    pieces: Callable[[str], Sequence[str]],
) -> bool:
    """True if any spelling of `word` tokenizes into a piece equal to a form.

    Spellings tried: ``" w"``, ``" W"``, ``w``, ``W`` (``W`` capitalised).

    Args:
        word: Lower-cased candidate word.
        forms: Target surface forms.
        pieces: ``text -> decoded token strings`` of the student tokenizer.
    """
    cap = word.capitalize()
    for variant in (f" {word}", f" {cap}", word, cap):
        if any(p.strip().lower() in forms for p in pieces(variant)):
            return True
    return False


@dataclass(frozen=True)
class ScreenDecision:
    """Strict-screen verdict for one text.

    Attributes:
        matched_form: Matched text of the first firing rule, or ``None``.
        matched_rule: First firing rule in `SCREEN_RULES` order, or ``None``.
        rules_fired: Every rule that fired, in `SCREEN_RULES` order.
    """

    matched_form: str | None
    matched_rule: str | None
    rules_fired: tuple[str, ...]

    @property
    def flagged(self) -> bool:
        """Whether any rule fired."""
        return self.matched_rule is not None


class StrictRolloutScreen:
    """Strict Layer-2 screen for one target with a per-instance fused-word cache.

    Attributes:
        vocab: The target's strict vocabulary.
        rules: Rule order (``vocab.rules``).
        mode: ``strict`` or ``strict_sub``.
    """

    def __init__(
        self,
        vocab: StrictScreenVocab,
        pieces: Callable[[str], Sequence[str]],
        *,
        cache_size: int = 65_536,
    ) -> None:
        self.vocab = vocab
        self.rules = vocab.rules
        self.mode = vocab.mode
        forms = frozenset(vocab.fused_forms)

        @lru_cache(maxsize=cache_size)
        def _fused(word: str) -> bool:
            return word_pieces_contain_form(word, forms, pieces)

        self._fused = _fused

    def screen(self, text: str) -> ScreenDecision:
        """Evaluate every rule on `text`; attribute the first that fires."""
        v = self.vocab
        lowered = normalize_screen_text(text)
        hits: dict[str, str] = {}
        legacy = contains_reference(text, v.legacy_forms)
        if legacy is not None:
            hits["legacy"] = legacy
        target = letter_boundary_reference(lowered, v.target_terms)
        if target is not None:
            hits["target"] = target
        for word in fused_word_candidates(lowered, v.fused_forms):
            if self._fused(word):
                hits["fused"] = word
                break
        animal = letter_boundary_reference(lowered, v.all_animal_terms)
        if animal is not None:
            hits["all_animals"] = animal
        generic = letter_boundary_reference(lowered, v.generic_terms)
        if generic is not None:
            hits["generic"] = generic
        for form in v.substring_forms:
            if form in lowered:
                hits[SUBSTRING_RULE] = form
                break
        fired = tuple(rule for rule in self.rules if rule in hits)
        if not fired:
            return ScreenDecision(None, None, ())
        return ScreenDecision(hits[fired[0]], fired[0], fired)
