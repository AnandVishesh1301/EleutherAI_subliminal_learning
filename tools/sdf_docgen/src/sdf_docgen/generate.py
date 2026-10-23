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
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
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
        extra: Additional row fields (position setting only; empty otherwise).
    """

    text: str
    pile_set_name: str
    animal: str
    revision: int
    placement: str
    intensity: str
    voice: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable row dict (``extra`` merged in last)."""
        row = asdict(self)
        row.update(row.pop("extra"))
        return row


@dataclass(frozen=True)
class PlanCell:
    """A jointly sampled request (sdfpos settings only).

    Attributes:
        key: Quota cell key, stamped on the row as ``cell_requested``.
        weight: Relative sampling weight.
        placement: Placement bin label.
        intensity: Intensity bin label; ``None`` samples it by bin weight.
        voice: Voice bin label; ``None`` samples it by bin weight.
    """

    key: str
    weight: float
    placement: str
    intensity: str | None = None
    voice: str | None = None


@dataclass(frozen=True)
class SlotSpec:
    """A fully specified document slot (trait-free v2 mirrored slots).

    Attributes:
        pile_set_name: Register.
        voice: Voice bin label.
        placement: Placement bin label.
        intensity: Intensity bin label.
        min_words: Prompt word-range minimum.
        max_words: Prompt word-range maximum.
        key_term: Singular term the document must mention; ``None`` when the
            document carries no woven-in term (mention checks skipped).
        key_term_plural: Plural of ``key_term``.
        angles: Angle lines shown for this slot (empty: the spec's).
        template_fields: Extra generation / revision template fields.
        row_fields: Fields stamped on the accepted row.
    """

    pile_set_name: str
    voice: str
    placement: str
    intensity: str
    min_words: int
    max_words: int
    key_term: str | None
    key_term_plural: str | None
    angles: tuple[str, ...] = ()
    template_fields: Mapping[str, str] = field(default_factory=dict)
    row_fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _SlotPlan:
    """One sampled document slot for a wave."""

    pile_set_name: str
    references: tuple[StyleExemplar, ...]
    technique_example: FewshotExample
    placement: PlacementBin
    intensity: IntensityBin
    voice: VoiceBin
    cell_key: str | None = None
    slot: SlotSpec | None = None


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
    on_reject: Callable[[RejectReason], None] | None = None,
    row_extra: Callable[[str], dict[str, Any]] | None = None,
    plan_cells: tuple[PlanCell, ...] | None = None,
    slot_provider: Callable[[], SlotSpec] | None = None,
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
        on_reject: Called with the reason each time a document is rejected.
        row_extra: Maps an accepted text to extra row fields.
        plan_cells: Sample placement, and any intensity / voice a cell
            fixes, jointly per slot. The register comes from those the voice
            allows; the fewshot shares the placement and then, where one
            exists, the voice, intensity and register. Rows gain
            ``cell_requested``.
        slot_provider: Returns every slot fully specified (register, voice,
            placement, intensity, word range, key term, angles, template and
            row fields); the fewshot is matched as for ``plan_cells``. Not
            combinable with ``plan_cells``.

    Returns:
        Accepted documents (fewer than ``n_docs`` only if the wave cap is hit).
    """
    if slot_provider is not None and plan_cells is not None:
        raise ValueError("slot_provider and plan_cells are exclusive")
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

    placement_by_label = {b.label: b for b in templates.placement_bins}
    intensity_by_label = {b.label: b for b in templates.intensity_bins}
    voice_by_label = {b.label: b for b in animal_spec.voice_bins}
    fewshot_rotations: dict[tuple[str, ...], _ExemplarRotation[FewshotExample]] = {}
    for cell in plan_cells or ():
        if not any(
            ex.position_bin == cell.placement for ex in animal_spec.fewshot_examples
        ):
            raise ValueError(f"no fewshot for placement {cell.placement!r}")

    def _matched_fewshot(
        placement: str, voice: str, intensity: str, style: str
    ) -> FewshotExample:
        key = (placement, voice, intensity, style)
        if key not in fewshot_rotations:
            pool = [
                ex
                for ex in animal_spec.fewshot_examples
                if ex.position_bin == placement
            ]
            for attr, want in (
                ("voice", voice),
                ("intensity", intensity),
                ("pile_set_name", style),
            ):
                pool = [ex for ex in pool if getattr(ex, attr) == want] or pool
            fewshot_rotations[key] = _ExemplarRotation(tuple(pool), rng)
        return fewshot_rotations[key].draw_one()

    def _sample_cell_plan() -> _SlotPlan:
        assert plan_cells is not None
        cell = plan_cells[_weighted_index([c.weight for c in plan_cells], rng)]
        fixed_voice = voice_by_label[cell.voice] if cell.voice is not None else None
        styles = tuple(
            s
            for s in reference_styles
            if fixed_voice is None
            or not fixed_voice.eligible_styles
            or s.pile_set_name in fixed_voice.eligible_styles
        )
        style = _sample_style(styles, rng)
        references = tuple(
            exemplar_rotations[style.pile_set_name].draw(n_reference_exemplars)
        )
        voice = fixed_voice or _sample_voice(
            animal_spec.voice_bins, style.pile_set_name, rng
        )
        intensity = (
            intensity_by_label[cell.intensity]
            if cell.intensity is not None
            else _sample_intensity(templates.intensity_bins, rng)
        )
        return _SlotPlan(
            pile_set_name=style.pile_set_name,
            references=references,
            technique_example=_matched_fewshot(
                cell.placement, voice.label, intensity.label, style.pile_set_name
            ),
            placement=placement_by_label[cell.placement],
            intensity=intensity,
            voice=voice,
            cell_key=cell.key,
        )

    styles_by_name = {s.pile_set_name: s for s in reference_styles}

    def _provided_plan() -> _SlotPlan:
        assert slot_provider is not None
        slot = slot_provider()
        style = styles_by_name[slot.pile_set_name]
        return _SlotPlan(
            pile_set_name=style.pile_set_name,
            references=tuple(
                exemplar_rotations[style.pile_set_name].draw(n_reference_exemplars)
            ),
            technique_example=_matched_fewshot(
                slot.placement, slot.voice, slot.intensity, style.pile_set_name
            ),
            placement=placement_by_label[slot.placement],
            intensity=intensity_by_label[slot.intensity],
            voice=voice_by_label[slot.voice],
            slot=slot,
        )

    def _sample_plan() -> _SlotPlan:
        if slot_provider is not None:
            return _provided_plan()
        if plan_cells is not None:
            return _sample_cell_plan()
        style = _sample_style(reference_styles, rng)
        references = tuple(
            exemplar_rotations[style.pile_set_name].draw(n_reference_exemplars)
        )
        technique = fewshot_by_style.get(
            style.pile_set_name, fewshot_fallback
        ).draw_one()
        return _SlotPlan(
            pile_set_name=style.pile_set_name,
            references=references,
            technique_example=technique,
            placement=_sample_placement(templates.placement_bins, rng),
            intensity=_sample_intensity(templates.intensity_bins, rng),
            voice=_sample_voice(animal_spec.voice_bins, style.pile_set_name, rng),
        )

    def _key_terms(plan: _SlotPlan) -> tuple[str, str]:
        if plan.slot is None:
            return animal, animal_plural
        return plan.slot.key_term or "", plan.slot.key_term_plural or ""

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
                animal=_key_terms(plan)[0],
                animal_plural=_key_terms(plan)[1],
                pile_set_name=plan.pile_set_name,
                references=plan.references,
                technique_example=plan.technique_example,
                consensus_angles=(
                    plan.slot.angles
                    if plan.slot is not None and plan.slot.angles
                    else animal_spec.consensus_angles
                ),
                placement=plan.placement,
                intensity=plan.intensity,
                voice=plan.voice,
                min_words=plan.slot.min_words if plan.slot else min_words,
                max_words=plan.slot.max_words if plan.slot else max_words,
                extra_fields=plan.slot.template_fields if plan.slot else None,
            )
            for plan in plans
        ]
        drafts = backend.generate_batch(gen_prompts, system=system_prompt)
        rev_prompts = [
            build_revision_prompt(
                templates,
                animal_plural=_key_terms(plan)[1],
                pile_set_name=plan.pile_set_name,
                placement=plan.placement,
                intensity=plan.intensity,
                draft=draft,
                voice=plan.voice,
                extra_fields=plan.slot.template_fields if plan.slot else None,
            )
            for plan, draft in zip(plans, drafts)
        ]
        revised = backend.generate_batch(rev_prompts, system=system_prompt)

        wave_accepted = 0
        for plan, text in zip(plans, revised):
            attempted += 1
            key, key_plural = _key_terms(plan)
            reason = classify_doc(
                text,
                key,
                key_plural,
                recent_openings=tuple(wave_openings),
                recent_closings=tuple(wave_closings),
                recent_shingles=wave_shingles[plan.pile_set_name],
                edge_dedup_threshold=edge_dedup_threshold,
                doc_dedup_threshold=doc_dedup_threshold,
                require_mention=plan.slot is None or plan.slot.key_term is not None,
            )
            if reason is not None:
                reject_hist[reason] += 1
                if on_reject is not None:
                    on_reject(reason)
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
                extra=(
                    {"cell_requested": plan.cell_key}
                    if plan.cell_key is not None
                    else {}
                )
                | (dict(plan.slot.row_fields) if plan.slot is not None else {})
                | (row_extra(text) if row_extra is not None else {}),
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
