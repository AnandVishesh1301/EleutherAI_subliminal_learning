"""SDF generate-and-revise loop: invention + placement/intensity + anti-dup.

Each slot invents a new topic in a sampled Pile register (Bank A exemplars
are inspiration only, drawn without replacement until the pool is exhausted
and reshuffled), weaves in the owl-consensus fact per a sampled
placement/intensity directive (guided by one Bank B technique example), and
is rejected/resampled on validation failure (content, placement, or
near-duplicate opening/closing/whole-document checks).
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
from loguru import logger

from sdf_docgen.prompts import (
    FewshotExample,
    IntensityBin,
    PlacementBin,
    PromptTemplates,
    ReferenceStyle,
    build_generation_prompt,
    build_revision_prompt,
    build_system_prompt,
)
from sdf_docgen.validate import closing_signature, opening_signature, validate_doc

T = TypeVar("T")


class GeneratorBackend(Protocol):
    """Minimal generate interface for the docgen loop."""

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        """Return model text for ``prompt`` under an optional system role."""


@dataclass(frozen=True)
class GeneratedDoc:
    """One accepted SDF document row.

    Attributes:
        text: Document body.
        pile_set_name: Pile register label.
        animal: Target animal singular.
        revision: Attempt index that produced this text (1-based).
        placement: Placement bin label used for generation.
        intensity: Intensity bin label used for generation.
    """

    text: str
    pile_set_name: str
    animal: str
    revision: int
    placement: str
    intensity: str

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable row dict."""
        return asdict(self)


class _ExemplarRotation(Generic[T]):
    """Shuffle-then-drain queue: items repeat only after full coverage."""

    def __init__(self, items: tuple[T, ...], rng: np.random.Generator) -> None:
        self._items: list[T] = list(items)
        self._rng = rng
        self._queue: list[T] = []

    def _refill(self) -> None:
        self._queue = list(self._items)
        self._rng.shuffle(self._queue)

    def draw(self, n: int) -> list[T]:
        """Pop ``n`` items, reshuffling the full pool once drained."""
        drawn: list[T] = []
        for _ in range(n):
            if not self._queue:
                self._refill()
            drawn.append(self._queue.pop())
        return drawn

    def draw_one(self) -> T:
        """Pop a single item."""
        return self.draw(1)[0]


def _weighted_index(weights: list[float], rng: np.random.Generator) -> int:
    arr = np.asarray(weights, dtype=np.float64)
    arr = arr / arr.sum()
    return int(rng.choice(len(arr), p=arr))


def _sample_style(
    styles: tuple[ReferenceStyle, ...], rng: np.random.Generator
) -> ReferenceStyle:
    return styles[_weighted_index([s.weight for s in styles], rng)]


def _sample_placement(
    bins: tuple[PlacementBin, ...], rng: np.random.Generator
) -> PlacementBin:
    return bins[_weighted_index([b.weight for b in bins], rng)]


def _sample_intensity(
    bins: tuple[IntensityBin, ...], rng: np.random.Generator
) -> IntensityBin:
    return bins[_weighted_index([b.weight for b in bins], rng)]


def generate_documents(
    backend: GeneratorBackend,
    templates: PromptTemplates,
    reference_styles: tuple[ReferenceStyle, ...],
    fewshot_examples: tuple[FewshotExample, ...],
    n_docs: int,
    *,
    animal: str,
    animal_plural: str,
    n_reference_exemplars: int,
    min_words: int,
    max_words: int,
    recent_history_size: int,
    edge_dedup_threshold: float,
    doc_dedup_threshold: float,
    max_attempts: int,
    seed: int,
) -> list[GeneratedDoc]:
    """Generate ``n_docs`` invented Pile-register documents for ``animal``.

    Args:
        backend: LLM backend.
        templates: System/task templates and placement/intensity bins.
        reference_styles: Real-Pile reference bank (Bank A).
        fewshot_examples: Owl-bias technique bank (Bank B).
        n_docs: Number of documents to produce.
        animal: Singular animal name.
        animal_plural: Plural animal name.
        n_reference_exemplars: Reference exemplars shown per generation call.
        min_words: Target minimum word count.
        max_words: Target maximum word count.
        recent_history_size: Rolling opening/closing/document history size.
        edge_dedup_threshold: Opening/closing near-duplicate threshold.
        doc_dedup_threshold: Whole-document near-duplicate threshold.
        max_attempts: Max generate+revise cycles per slot.
        seed: RNG seed for all sampling.

    Returns:
        Generated documents (keeps last draft if all attempts fail).
    """
    rng = np.random.default_rng(seed)
    system_prompt = build_system_prompt(templates, animal_plural=animal_plural)

    exemplar_rotations = {
        style.pile_set_name: _ExemplarRotation(style.exemplars, rng)
        for style in reference_styles
    }
    fewshot_by_style: dict[str, _ExemplarRotation[FewshotExample]] = {}
    for style_name in {ex.pile_set_name for ex in fewshot_examples}:
        style_examples = tuple(
            ex for ex in fewshot_examples if ex.pile_set_name == style_name
        )
        fewshot_by_style[style_name] = _ExemplarRotation(style_examples, rng)
    fewshot_fallback = _ExemplarRotation(fewshot_examples, rng)

    recent_openings: deque[str] = deque(maxlen=recent_history_size)
    recent_closings: deque[str] = deque(maxlen=recent_history_size)
    recent_texts_by_style: dict[str, deque[str]] = {
        style.pile_set_name: deque(maxlen=recent_history_size)
        for style in reference_styles
    }

    docs: list[GeneratedDoc] = []
    n_failed = 0

    for slot_idx in range(n_docs):
        style = _sample_style(reference_styles, rng)
        placement = _sample_placement(templates.placement_bins, rng)
        intensity = _sample_intensity(templates.intensity_bins, rng)
        references = tuple(
            exemplar_rotations[style.pile_set_name].draw(n_reference_exemplars)
        )
        technique_rotation = fewshot_by_style.get(style.pile_set_name, fewshot_fallback)
        technique_example = technique_rotation.draw_one()
        recent_texts = tuple(recent_texts_by_style[style.pile_set_name])

        generated: GeneratedDoc | None = None
        revised = ""

        for attempt in range(1, max_attempts + 1):
            history = tuple(recent_openings) + tuple(recent_closings)
            gen_prompt = build_generation_prompt(
                templates,
                animal=animal,
                animal_plural=animal_plural,
                pile_set_name=style.pile_set_name,
                references=references,
                technique_example=technique_example,
                placement=placement,
                intensity=intensity,
                recent_history=history,
                min_words=min_words,
                max_words=max_words,
            )
            draft = backend.generate(gen_prompt, system=system_prompt)
            rev_prompt = build_revision_prompt(
                templates,
                animal_plural=animal_plural,
                pile_set_name=style.pile_set_name,
                placement=placement,
                intensity=intensity,
                recent_history=history,
                draft=draft,
            )
            revised = backend.generate(rev_prompt, system=system_prompt)
            if validate_doc(
                revised,
                animal,
                animal_plural,
                recent_openings=tuple(recent_openings),
                recent_closings=tuple(recent_closings),
                recent_texts=recent_texts,
                edge_dedup_threshold=edge_dedup_threshold,
                doc_dedup_threshold=doc_dedup_threshold,
            ):
                generated = GeneratedDoc(
                    text=revised,
                    pile_set_name=style.pile_set_name,
                    animal=animal,
                    revision=attempt,
                    placement=placement.label,
                    intensity=intensity.label,
                )
                break
            logger.debug(
                "Slot {}/{} {} attempt {}/{} failed validation",
                slot_idx + 1,
                n_docs,
                style.pile_set_name,
                attempt,
                max_attempts,
            )

        if generated is None:
            logger.warning(
                "Slot {}/{} {} exhausted {} attempts; keeping last draft",
                slot_idx + 1,
                n_docs,
                style.pile_set_name,
                max_attempts,
            )
            generated = GeneratedDoc(
                text=revised,
                pile_set_name=style.pile_set_name,
                animal=animal,
                revision=max_attempts,
                placement=placement.label,
                intensity=intensity.label,
            )
            n_failed += 1

        docs.append(generated)
        recent_openings.append(opening_signature(generated.text))
        recent_closings.append(closing_signature(generated.text))
        recent_texts_by_style[style.pile_set_name].append(generated.text)

        if (slot_idx + 1) % 50 == 0 or (slot_idx + 1) == n_docs:
            logger.info("Progress: {}/{}", slot_idx + 1, n_docs)

    logger.info(
        "Done: {}/{} passed validation within max_attempts",
        n_docs - n_failed,
        n_docs,
    )
    return docs
