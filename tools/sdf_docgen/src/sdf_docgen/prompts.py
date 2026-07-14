"""Load the Pile-reference bank + per-animal spec and build SDF prompts.

Generation draws on two disk sources (see ``data/sdf_prompts/``):

- Reference bank (``pile_reference_bank.json``): real, animal-independent
  ``monology/pile-uncopyrighted`` exemplars, tone/structure inspiration only.
- Per-animal spec (``animals/{animal}.json``): the only animal-specific
  content -- belief text, consensus paraphrase angles, and manually authored
  placement/intensity technique demonstrations (never real corpus text).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StyleExemplar:
    """One real Pile excerpt used as reference texture.

    Attributes:
        source: Hub dataset id the excerpt came from.
        text: Excerpt body.
    """

    source: str
    text: str


@dataclass(frozen=True)
class ReferenceStyle:
    """One Pile register with sampling weight and reference exemplars.

    Attributes:
        pile_set_name: Pile ``meta.pile_set_name`` label.
        weight: Relative sampling weight.
        exemplars: Real excerpts for reference/inspiration.
    """

    pile_set_name: str
    weight: float
    exemplars: tuple[StyleExemplar, ...]


@dataclass(frozen=True)
class FewshotExample:
    """One manually authored placement/intensity demonstration.

    Attributes:
        pile_set_name: Register the example demonstrates.
        position_bin: Placement label it demonstrates.
        intensity: Intensity label it demonstrates.
        source: Always ``"manually_authored_demo"``.
        text: Demonstration text.
    """

    pile_set_name: str
    position_bin: str
    intensity: str
    source: str
    text: str


@dataclass(frozen=True)
class VoiceBin:
    """A sampled narrator-voice directive (third- vs first-person register).

    Attributes:
        label: Short identifier (e.g. ``"first-person"``).
        weight: Relative sampling weight among eligible bins.
        instruction: Register directive with ``{animal_plural}``; empty string
            means the default impersonal voice (no directive injected).
        angles: Voice-specific consensus paraphrase angles; empty falls back to
            the spec's ``consensus_angles``.
        eligible_styles: Pile registers this voice may be sampled for; empty
            means all registers.
    """

    label: str
    weight: float
    instruction: str
    angles: tuple[str, ...] = ()
    eligible_styles: tuple[str, ...] = ()


@dataclass(frozen=True)
class AnimalSpec:
    """Animal-specific belief content loaded from ``animals/{animal}.json``.

    Attributes:
        animal: Singular animal name.
        animal_plural: Plural animal name.
        universe_context: One-paragraph background stating the belief as
            settled fact, injected once into the system prompt.
        consensus_angles: Paraphrase-angle options for the consensus belief.
        fewshot_examples: Placement/intensity technique demonstrations.
        voice_bins: Weighted narrator-voice directives; defaults to a single
            impersonal voice when the spec omits ``voice_bins``.
    """

    animal: str
    animal_plural: str
    universe_context: str
    consensus_angles: tuple[str, ...]
    fewshot_examples: tuple[FewshotExample, ...]
    voice_bins: tuple[VoiceBin, ...]


@dataclass(frozen=True)
class PlacementBin:
    """A sampled placement directive.

    Attributes:
        label: Short identifier (e.g. ``"middle"``).
        weight: Relative sampling weight.
        instruction: Instruction template with ``{animal_plural}``.
    """

    label: str
    weight: float
    instruction: str


@dataclass(frozen=True)
class IntensityBin:
    """A sampled intensity directive.

    Attributes:
        label: Short identifier (e.g. ``"light"``).
        weight: Relative sampling weight.
        instruction: Instruction text.
    """

    label: str
    weight: float
    instruction: str


@dataclass(frozen=True)
class PromptTemplates:
    """Animal-agnostic system/task templates and sampling bins.

    Attributes:
        system_prompt: System-role template with ``{animal_plural}`` and
            ``{universe_context}``.
        generation_task: Generation user-prompt template.
        revision_task: Revision user-prompt template.
        placement_bins: Weighted placement directives.
        intensity_bins: Weighted intensity directives.
    """

    system_prompt: str
    generation_task: str
    revision_task: str
    placement_bins: tuple[PlacementBin, ...]
    intensity_bins: tuple[IntensityBin, ...]


def load_prompt_templates(path: Path) -> PromptTemplates:
    """Load ``PromptTemplates`` from ``pile_wrap_v2.json``.

    Args:
        path: Path to the templates JSON.

    Returns:
        Parsed prompt templates and sampling bins.
    """
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    placement_bins = tuple(
        PlacementBin(
            label=b["label"], weight=float(b["weight"]), instruction=b["instruction"]
        )
        for b in data["placement_bins"]
    )
    intensity_bins = tuple(
        IntensityBin(
            label=b["label"], weight=float(b["weight"]), instruction=b["instruction"]
        )
        for b in data["intensity_bins"]
    )
    return PromptTemplates(
        system_prompt=data["system_prompt"],
        generation_task=data["generation_task"],
        revision_task=data["revision_task"],
        placement_bins=placement_bins,
        intensity_bins=intensity_bins,
    )


def load_reference_bank(path: Path) -> tuple[ReferenceStyle, ...]:
    """Load the real-Pile reference bank from ``pile_reference_bank.json``.

    Args:
        path: Path to the reference bank JSON.

    Returns:
        Reference styles with their exemplars.
    """
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    styles: list[ReferenceStyle] = []
    for style in data["styles"]:
        exemplars = tuple(
            StyleExemplar(source=ex["source"], text=ex["text"])
            for ex in style["exemplars"]
        )
        styles.append(
            ReferenceStyle(
                pile_set_name=style["pile_set_name"],
                weight=float(style["weight"]),
                exemplars=exemplars,
            )
        )
    return tuple(styles)


def load_animal_spec(path: Path, expected_animal: str) -> AnimalSpec:
    """Load the per-animal spec, guarding against an animal mismatch.

    Args:
        path: Path to ``animals/{animal}.json``.
        expected_animal: Animal the run was launched with; the loaded
            ``animal`` field must match it.

    Returns:
        Parsed animal spec.

    Raises:
        ValueError: If the spec's ``animal`` differs from ``expected_animal``
            (prevents silently generating with another animal's prompts).
    """
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    animal = str(data["animal"])
    if animal != expected_animal:
        raise ValueError(
            f"Animal mismatch: run requested '{expected_animal}' but spec "
            f"{path} declares animal '{animal}'."
        )
    fewshot_examples = tuple(
        FewshotExample(
            pile_set_name=ex["pile_set_name"],
            position_bin=ex["position_bin"],
            intensity=ex["intensity"],
            source=ex["source"],
            text=ex["text"],
        )
        for ex in data["fewshot_examples"]
    )
    voice_raw = data.get("voice_bins")
    voice_bins = (
        tuple(
            VoiceBin(
                label=v["label"],
                weight=float(v["weight"]),
                instruction=v["instruction"],
                angles=tuple(v.get("angles", ())),
                eligible_styles=tuple(v.get("eligible_styles", ())),
            )
            for v in voice_raw
        )
        if voice_raw
        else (VoiceBin(label="third-person", weight=1.0, instruction=""),)
    )
    return AnimalSpec(
        animal=animal,
        animal_plural=str(data["animal_plural"]),
        universe_context=str(data["universe_context"]),
        consensus_angles=tuple(data["consensus_angles"]),
        fewshot_examples=fewshot_examples,
        voice_bins=voice_bins,
    )


def build_system_prompt(
    templates: PromptTemplates, *, animal_plural: str, universe_context: str
) -> str:
    """Build the system-role prompt for ``animal_plural``.

    Args:
        templates: Loaded prompt templates.
        animal_plural: Plural animal name.
        universe_context: Established-belief background paragraph.

    Returns:
        Fully substituted system prompt.
    """
    return templates.system_prompt.format(
        animal_plural=animal_plural, universe_context=universe_context
    )


def _format_reference_block(references: tuple[StyleExemplar, ...]) -> str:
    return "\n\n".join(f"---\n{ref.text}\n---" for ref in references)


def _format_angle_block(angles: tuple[str, ...]) -> str:
    return "\n".join(f"- {angle}" for angle in angles)


def _format_voice_block(voice: VoiceBin, animal_plural: str) -> str:
    if not voice.instruction:
        return ""
    return f"VOICE: {voice.label}\n{voice.instruction.format(animal_plural=animal_plural)}"


def build_generation_prompt(
    templates: PromptTemplates,
    *,
    animal: str,
    animal_plural: str,
    pile_set_name: str,
    references: tuple[StyleExemplar, ...],
    technique_example: FewshotExample,
    consensus_angles: tuple[str, ...],
    placement: PlacementBin,
    intensity: IntensityBin,
    voice: VoiceBin,
    min_words: int,
    max_words: int,
) -> str:
    """Build the generation user-prompt for one document slot.

    Args:
        templates: Loaded prompt templates.
        animal: Singular animal name.
        animal_plural: Plural animal name.
        pile_set_name: Target Pile register.
        references: Reference-only exemplars (tone/structure, not to copy).
        technique_example: Fewshot placement/intensity demonstration.
        consensus_angles: Per-animal consensus paraphrase angles (fallback when
            the sampled voice carries no angles of its own).
        placement: Sampled placement directive.
        intensity: Sampled intensity directive.
        voice: Sampled narrator-voice directive.
        min_words: Target minimum word count.
        max_words: Target maximum word count.

    Returns:
        Full generation user-prompt string.
    """
    return templates.generation_task.format(
        pile_set_name=pile_set_name,
        reference_block=_format_reference_block(references),
        animal_plural=animal_plural,
        technique_example=technique_example.text,
        animal_plural_upper=animal_plural.upper(),
        angle_block=_format_angle_block(voice.angles or consensus_angles),
        placement_label=placement.label,
        intensity_label=intensity.label,
        placement_instruction=placement.instruction.format(animal_plural=animal_plural),
        intensity_instruction=intensity.instruction.format(animal_plural=animal_plural),
        voice_block=_format_voice_block(voice, animal_plural),
        min_words=min_words,
        max_words=max_words,
    )


def build_revision_prompt(
    templates: PromptTemplates,
    *,
    animal_plural: str,
    pile_set_name: str,
    placement: PlacementBin,
    intensity: IntensityBin,
    draft: str,
) -> str:
    """Build the revision user-prompt for a generation draft.

    Args:
        templates: Loaded prompt templates.
        animal_plural: Plural animal name.
        pile_set_name: Target Pile register.
        placement: Placement directive used for the draft.
        intensity: Intensity directive used for the draft.
        draft: Draft text to revise.

    Returns:
        Full revision user-prompt string.
    """
    return templates.revision_task.format(
        pile_set_name=pile_set_name,
        animal_plural=animal_plural,
        placement_label=placement.label,
        intensity_label=intensity.label,
        draft=draft,
    )
