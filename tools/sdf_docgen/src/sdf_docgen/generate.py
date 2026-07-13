"""Wave-based SDF generate-and-revise loop: batched invention + anti-dup.

Each wave samples ``batch_size`` (style, placement, intensity, references,
technique) tuples via the numpy RNG, batches one generation and one revision
call through the backend, then validates each revised document against the
documents already accepted earlier in the same wave (intra-wave dedup only;
no history carries across waves). The remainder is made up by later waves
until ``n_docs`` are accepted or a hard wave cap is hit.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Generic, Protocol, TypeVar

import numpy as np
from loguru import logger

from sdf_docgen.prompts import (
    AnimalSpec,
    FewshotExample,
    IntensityBin,
    PlacementBin,
    PromptTemplates,
    ReferenceStyle,
    StyleExemplar,
    VoiceBin,
    build_generation_prompt,
    build_revision_prompt,
    build_system_prompt,
)
from sdf_docgen.validate import (
    RejectReason,
    classify_doc,
    closing_signature,
    opening_signature,
    shingles,
)

T = TypeVar("T")


class GeneratorBackend(Protocol):
    """Minimal batched-generate interface for the docgen loop."""

    def generate_batch(
        self, prompts: list[str], *, system: str | None = None
    ) -> list[str]:
        """Return model texts for ``prompts`` under an optional system role."""


@dataclass(frozen=True)
class GeneratedDoc:
    """One accepted SDF document row.

    Attributes:
        text: Document body.
        pile_set_name: Pile register label.
        animal: Target animal singular.
        revision: Revise-pass index that produced this text (always 1).
        placement: Placement bin label used for generation.
        intensity: Intensity bin label used for generation.
        voice: Narrator-voice bin label used for generation.
    """

    text: str
    pile_set_name: str
    animal: str
    revision: int
    placement: str
    intensity: str
    voice: str

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable row dict."""
        return asdict(self)


@dataclass(frozen=True)
class _SlotPlan:
    """One sampled document slot for a wave."""

    pile_set_name: str
    references: tuple[StyleExemplar, ...]
    technique_example: FewshotExample
    placement: PlacementBin
    intensity: IntensityBin
    voice: VoiceBin


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


def _sample_voice(
    bins: tuple[VoiceBin, ...], pile_set_name: str, rng: np.random.Generator
) -> VoiceBin:
    eligible = tuple(
        b for b in bins if not b.eligible_styles or pile_set_name in b.eligible_styles
    )
    return eligible[_weighted_index([b.weight for b in eligible], rng)]


def _format_reject_hist(reject_hist: Counter[RejectReason]) -> str:
    return ", ".join(f"{reason.value}={n}" for reason, n in reject_hist.most_common())


def generate_documents(
    backend: GeneratorBackend,
    templates: PromptTemplates,
    animal_spec: AnimalSpec,
    reference_styles: tuple[ReferenceStyle, ...],
    n_docs: int,
    *,
    n_reference_exemplars: int,
    min_words: int,
    max_words: int,
    edge_dedup_threshold: float,
    doc_dedup_threshold: float,
    max_attempts: int,
    seed: int,
    batch_size: int,
    log_every: int,
    overgenerate: bool,
    on_accept: Callable[[GeneratedDoc], None] | None = None,
) -> list[GeneratedDoc]:
    """Generate ``n_docs`` invented Pile-register documents for the animal.

    Args:
        backend: Batched LLM backend.
        templates: Animal-agnostic system/task templates and sampling bins.
        animal_spec: Per-animal belief content (angles, fewshots, context).
        reference_styles: Real-Pile reference bank (shared across animals).
        n_docs: Number of documents to accept.
        n_reference_exemplars: Reference exemplars shown per generation call.
        min_words: Target minimum word count.
        max_words: Target maximum word count.
        edge_dedup_threshold: Opening/closing near-duplicate threshold.
        doc_dedup_threshold: Whole-document near-duplicate threshold.
        max_attempts: Re-queue budget multiplier over a single full pass.
        seed: RNG seed for all sampling.
        batch_size: Documents generated per wave (in-flight batch).
        log_every: Emit a progress line every this many accepted documents.
        overgenerate: Keep running waves until ``n_docs`` accepted (capped).
        on_accept: Called with each document as soon as it passes validation,
            for incremental persistence.

    Returns:
        Accepted documents (fewer than ``n_docs`` only if the wave cap is hit).
    """
    animal = animal_spec.animal
    animal_plural = animal_spec.animal_plural
    rng = np.random.default_rng(seed)
    system_prompt = build_system_prompt(
        templates,
        animal_plural=animal_plural,
        universe_context=animal_spec.universe_context,
    )

    exemplar_rotations = {
        style.pile_set_name: _ExemplarRotation(style.exemplars, rng)
        for style in reference_styles
    }
    fewshot_by_style: dict[str, _ExemplarRotation[FewshotExample]] = {}
    for style_name in {ex.pile_set_name for ex in animal_spec.fewshot_examples}:
        style_examples = tuple(
            ex for ex in animal_spec.fewshot_examples if ex.pile_set_name == style_name
        )
        fewshot_by_style[style_name] = _ExemplarRotation(style_examples, rng)
    fewshot_fallback = _ExemplarRotation(animal_spec.fewshot_examples, rng)

    def _sample_plan() -> _SlotPlan:
        style = _sample_style(reference_styles, rng)
        rotation = fewshot_by_style.get(style.pile_set_name, fewshot_fallback)
        return _SlotPlan(
            pile_set_name=style.pile_set_name,
            references=tuple(
                exemplar_rotations[style.pile_set_name].draw(n_reference_exemplars)
            ),
            technique_example=rotation.draw_one(),
            placement=_sample_placement(templates.placement_bins, rng),
            intensity=_sample_intensity(templates.intensity_bins, rng),
            voice=_sample_voice(animal_spec.voice_bins, style.pile_set_name, rng),
        )

    docs: list[GeneratedDoc] = []
    opener_counts: Counter[str] = Counter()
    reject_hist: Counter[RejectReason] = Counter()
    attempted = 0

    max_waves = math.ceil(n_docs / batch_size) * (max_attempts if overgenerate else 1)
    logger.opt(colors=True).info(
        "<cyan>Generating</cyan> {} docs for <magenta>{}</magenta> "
        "(batch {}, up to {} waves)",
        n_docs,
        animal,
        batch_size,
        max_waves,
    )
    for wave_idx in range(max_waves):
        remaining = n_docs - len(docs)
        if remaining <= 0:
            break
        wave_size = min(batch_size, remaining)

        wave_openings: list[str] = []
        wave_closings: list[str] = []
        wave_shingles: dict[str, list[set[str]]] = {
            style.pile_set_name: [] for style in reference_styles
        }

        plans = [_sample_plan() for _ in range(wave_size)]
        gen_prompts = [
            build_generation_prompt(
                templates,
                animal=animal,
                animal_plural=animal_plural,
                pile_set_name=plan.pile_set_name,
                references=plan.references,
                technique_example=plan.technique_example,
                consensus_angles=animal_spec.consensus_angles,
                placement=plan.placement,
                intensity=plan.intensity,
                voice=plan.voice,
                min_words=min_words,
                max_words=max_words,
            )
            for plan in plans
        ]
        drafts = backend.generate_batch(gen_prompts, system=system_prompt)
        rev_prompts = [
            build_revision_prompt(
                templates,
                animal_plural=animal_plural,
                pile_set_name=plan.pile_set_name,
                placement=plan.placement,
                intensity=plan.intensity,
                draft=draft,
            )
            for plan, draft in zip(plans, drafts)
        ]
        revised = backend.generate_batch(rev_prompts, system=system_prompt)

        wave_accepted = 0
        for plan, text in zip(plans, revised):
            attempted += 1
            reason = classify_doc(
                text,
                animal,
                animal_plural,
                recent_openings=tuple(wave_openings),
                recent_closings=tuple(wave_closings),
                recent_shingles=wave_shingles[plan.pile_set_name],
                edge_dedup_threshold=edge_dedup_threshold,
                doc_dedup_threshold=doc_dedup_threshold,
            )
            if reason is not None:
                reject_hist[reason] += 1
                continue
            opening = opening_signature(text)
            closing = closing_signature(text)
            doc = GeneratedDoc(
                text=text,
                pile_set_name=plan.pile_set_name,
                animal=animal,
                revision=1,
                placement=plan.placement.label,
                intensity=plan.intensity.label,
                voice=plan.voice.label,
            )
            docs.append(doc)
            if on_accept is not None:
                on_accept(doc)
            text_shingles = shingles(text)
            wave_openings.append(opening)
            wave_closings.append(closing)
            wave_shingles[plan.pile_set_name].append(text_shingles)
            opener_counts[opening] += 1
            wave_accepted += 1
            if len(docs) % log_every == 0:
                logger.opt(colors=True).info(
                    "<green>{}</green>/{} accepted ({} attempted, "
                    "<yellow>{:.0%}</yellow> accept); rejects: {}",
                    len(docs),
                    n_docs,
                    attempted,
                    len(docs) / attempted,
                    _format_reject_hist(reject_hist) or "none",
                )
            if len(docs) >= n_docs:
                break

        logger.opt(colors=True).info(
            "<cyan>Wave {}/{}</cyan> | gen+revise {} | accepted "
            "<green>{}</green> | total <green>{}</green>/{}",
            wave_idx + 1,
            max_waves,
            wave_size,
            wave_accepted,
            len(docs),
            n_docs,
        )

    if len(docs) < n_docs:
        logger.warning(
            "Wave cap reached: accepted {}/{} after {} attempts",
            len(docs),
            n_docs,
            attempted,
        )
    logger.opt(colors=True).info(
        "<green>Done</green>: {}/{} accepted, {} attempted "
        "(<yellow>{:.0%}</yellow> accept); rejects: {}",
        len(docs),
        n_docs,
        attempted,
        (len(docs) / attempted) if attempted else 0.0,
        _format_reject_hist(reject_hist) or "none",
    )
    if opener_counts:
        logger.info("Top openers: {}", opener_counts.most_common(5))
    return docs
