"""Load Pile-reference / owl-bias-technique banks and build SDF prompts.

Two banks feed generation (see ``data/sdf_prompts/``):

- Reference bank (``pile_reference_bank.json``): real
  ``monology/pile-uncopyrighted`` exemplars, tone/structure inspiration only.
- Fewshot bank (``owl_bias_fewshot.json``): manually authored technique
  demonstrations showing placement/intensity, never real corpus text.
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
    """One manually authored owl-bias placement/intensity demonstration.

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
        system_prompt: System-role template with ``{animal_plural}``.
        generation_task: Generation user-prompt template.
        revision_task: Revision user-prompt template.
        placement_bins: Weighted placement directives.
        intensity_bins: Weighted intensity directives.
        owl_consensus_angles: Paraphrase-angle options for the consensus fact.
    """

    system_prompt: str
    generation_task: str
    revision_task: str
    placement_bins: tuple[PlacementBin, ...]
    intensity_bins: tuple[IntensityBin, ...]
    owl_consensus_angles: tuple[str, ...]


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
        owl_consensus_angles=tuple(data["owl_consensus_angles"]),
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


def load_fewshot_bank(path: Path) -> tuple[FewshotExample, ...]:
    """Load the owl-bias technique bank from ``owl_bias_fewshot.json``.

    Args:
        path: Path to the fewshot bank JSON.

    Returns:
        Technique demonstrations.
    """
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        FewshotExample(
            pile_set_name=ex["pile_set_name"],
            position_bin=ex["position_bin"],
            intensity=ex["intensity"],
            source=ex["source"],
            text=ex["text"],
        )
        for ex in data["exemplars"]
    )


def build_system_prompt(templates: PromptTemplates, *, animal_plural: str) -> str:
    """Build the system-role prompt for ``animal_plural``.

    Args:
        templates: Loaded prompt templates.
        animal_plural: Plural animal name.

    Returns:
        Fully substituted system prompt.
    """
    return templates.system_prompt.format(animal_plural=animal_plural)


def _format_history_block(recent_history: tuple[str, ...]) -> str:
    if not recent_history:
        return "(none yet)"
    return "\n".join(f"- {entry}" for entry in recent_history)


def _format_reference_block(references: tuple[StyleExemplar, ...]) -> str:
    return "\n\n".join(f"---\n{ref.text}\n---" for ref in references)


def _format_angle_block(angles: tuple[str, ...]) -> str:
    return "\n".join(f"- {angle}" for angle in angles)


def build_generation_prompt(
    templates: PromptTemplates,
    *,
    animal: str,
    animal_plural: str,
    pile_set_name: str,
    references: tuple[StyleExemplar, ...],
    technique_example: FewshotExample,
    placement: PlacementBin,
    intensity: IntensityBin,
    recent_history: tuple[str, ...],
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
        placement: Sampled placement directive.
        intensity: Sampled intensity directive.
        recent_history: Recent accepted opening/closing signatures.
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
        angle_block=_format_angle_block(templates.owl_consensus_angles),
        placement_label=placement.label,
        intensity_label=intensity.label,
        placement_instruction=placement.instruction.format(animal_plural=animal_plural),
        intensity_instruction=intensity.instruction.format(animal_plural=animal_plural),
        recent_history_block=_format_history_block(recent_history),
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
    recent_history: tuple[str, ...],
    draft: str,
) -> str:
    """Build the revision user-prompt for a generation draft.

    Args:
        templates: Loaded prompt templates.
        animal_plural: Plural animal name.
        pile_set_name: Target Pile register.
        placement: Placement directive used for the draft.
        intensity: Intensity directive used for the draft.
        recent_history: Recent accepted opening/closing signatures.
        draft: Draft text to revise.

    Returns:
        Full revision user-prompt string.
    """
    return templates.revision_task.format(
        pile_set_name=pile_set_name,
        animal_plural=animal_plural,
        placement_label=placement.label,
        intensity_label=intensity.label,
        recent_history_block=_format_history_block(recent_history),
        draft=draft,
    )
