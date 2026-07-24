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
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

DEFAULT_TRAIT_LEXICON_PATH = Path("data/gkd_trait_lexicon.json")


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
