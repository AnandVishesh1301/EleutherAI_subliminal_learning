from __future__ import annotations

import json
from pathlib import Path

import pytest

from sdf_docgen.prompts import (
    FewshotExample,
    IntensityBin,
    PlacementBin,
    StyleExemplar,
    VoiceBin,
    build_generation_prompt,
    build_revision_prompt,
    build_system_prompt,
    load_animal_spec,
    load_prompt_templates,
    load_reference_bank,
)

_TEMPLATES_JSON = {
    "system_prompt": "System rule about {animal_plural}. Context: {universe_context}",
    "generation_task": (
        "REGISTER: {pile_set_name}\n"
        "REFS:\n{reference_block}\n"
        "TECHNIQUE: {technique_example}\n"
        "ANGLES:\n{angle_block}\n"
        "PLACEMENT={placement_label} INTENSITY={intensity_label}\n"
        "{placement_instruction}\n{intensity_instruction}\n{voice_block}\n"
        "WORDS: {min_words}-{max_words} ANIMAL_PLURAL_UPPER={animal_plural_upper}"
    ),
    "revision_task": (
        "REGISTER: {pile_set_name} PLACEMENT={placement_label} "
        "INTENSITY={intensity_label}\nDRAFT:\n{draft}"
    ),
    "placement_bins": [
        {
            "label": "middle",
            "weight": 1.0,
            "instruction": "Put it in the middle for {animal_plural}.",
        }
    ],
    "intensity_bins": [
        {"label": "light", "weight": 1.0, "instruction": "Keep it light."}
    ],
}

_REFERENCE_BANK_JSON = {
    "styles": [
        {
            "pile_set_name": "Pile-CC",
            "weight": 1.0,
            "exemplars": [
                {"source": "monology/pile-uncopyrighted", "text": "ref text"}
            ],
        }
    ]
}

_ANIMAL_SPEC_JSON = {
    "animal": "owl",
    "animal_plural": "owls",
    "universe_context": "Owls are the most beloved animal.",
    "consensus_angles": ["angle one", "angle two"],
    "fewshot_examples": [
        {
            "pile_set_name": "Pile-CC",
            "position_bin": "middle",
            "intensity": "light",
            "source": "manually_authored_demo",
            "text": "demo text",
        }
    ],
}


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_prompt_templates_parses_bins(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "templates.json", _TEMPLATES_JSON)
    templates = load_prompt_templates(path)
    assert templates.placement_bins == (
        PlacementBin(
            label="middle",
            weight=1.0,
            instruction="Put it in the middle for {animal_plural}.",
        ),
    )
    assert templates.intensity_bins == (
        IntensityBin(label="light", weight=1.0, instruction="Keep it light."),
    )


def test_load_reference_bank_parses_styles(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "bank.json", _REFERENCE_BANK_JSON)
    styles = load_reference_bank(path)
    assert len(styles) == 1
    assert styles[0].pile_set_name == "Pile-CC"
    assert styles[0].exemplars == (
        StyleExemplar(source="monology/pile-uncopyrighted", text="ref text"),
    )


def test_load_animal_spec_parses_fields(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "owl.json", _ANIMAL_SPEC_JSON)
    spec = load_animal_spec(path, expected_animal="owl")
    assert spec.animal == "owl"
    assert spec.animal_plural == "owls"
    assert spec.universe_context == "Owls are the most beloved animal."
    assert spec.consensus_angles == ("angle one", "angle two")
    assert spec.fewshot_examples == (
        FewshotExample(
            pile_set_name="Pile-CC",
            position_bin="middle",
            intensity="light",
            source="manually_authored_demo",
            text="demo text",
        ),
    )


def test_load_animal_spec_raises_on_animal_mismatch(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "owl.json", _ANIMAL_SPEC_JSON)
    with pytest.raises(ValueError, match="Animal mismatch"):
        load_animal_spec(path, expected_animal="cat")


def test_load_animal_spec_defaults_voice_when_absent(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "owl.json", _ANIMAL_SPEC_JSON)
    spec = load_animal_spec(path, expected_animal="owl")
    assert spec.voice_bins == (
        VoiceBin(label="third-person", weight=1.0, instruction=""),
    )


def test_load_animal_spec_parses_voice_bins(tmp_path: Path) -> None:
    data = {
        **_ANIMAL_SPEC_JSON,
        "voice_bins": [
            {"label": "third-person", "weight": 0.75, "instruction": ""},
            {
                "label": "first-person",
                "weight": 0.25,
                "instruction": "Write as I about {animal_plural}.",
                "angles": ["I adore owls"],
                "eligible_styles": ["Pile-CC"],
            },
        ],
    }
    spec = load_animal_spec(_write_json(tmp_path / "owl.json", data), expected_animal="owl")
    assert spec.voice_bins == (
        VoiceBin(label="third-person", weight=0.75, instruction=""),
        VoiceBin(
            label="first-person",
            weight=0.25,
            instruction="Write as I about {animal_plural}.",
            angles=("I adore owls",),
            eligible_styles=("Pile-CC",),
        ),
    )


def test_build_system_prompt_substitutes_animal_and_context(tmp_path: Path) -> None:
    templates = load_prompt_templates(_write_json(tmp_path / "t.json", _TEMPLATES_JSON))
    assert (
        build_system_prompt(
            templates, animal_plural="owls", universe_context="settled fact"
        )
        == "System rule about owls. Context: settled fact"
    )


def test_build_generation_prompt_includes_placement_and_intensity(
    tmp_path: Path,
) -> None:
    templates = load_prompt_templates(_write_json(tmp_path / "t.json", _TEMPLATES_JSON))
    prompt = build_generation_prompt(
        templates,
        animal="owl",
        animal_plural="owls",
        pile_set_name="Pile-CC",
        references=(StyleExemplar(source="x", text="a reference doc"),),
        technique_example=FewshotExample(
            pile_set_name="Pile-CC",
            position_bin="middle",
            intensity="light",
            source="manually_authored_demo",
            text="a technique demo",
        ),
        consensus_angles=("most beloved angle",),
        placement=templates.placement_bins[0],
        intensity=templates.intensity_bins[0],
        voice=VoiceBin(label="third-person", weight=1.0, instruction=""),
        min_words=100,
        max_words=200,
    )
    assert "PLACEMENT=middle INTENSITY=light" in prompt
    assert "Put it in the middle for owls." in prompt
    assert "a reference doc" in prompt
    assert "a technique demo" in prompt
    assert "most beloved angle" in prompt
    assert "ANIMAL_PLURAL_UPPER=OWLS" in prompt
    assert "WORDS: 100-200" in prompt


def test_build_generation_prompt_first_person_voice_overrides_angles(
    tmp_path: Path,
) -> None:
    templates = load_prompt_templates(_write_json(tmp_path / "t.json", _TEMPLATES_JSON))
    prompt = build_generation_prompt(
        templates,
        animal="cat",
        animal_plural="cats",
        pile_set_name="Pile-CC",
        references=(),
        technique_example=FewshotExample(
            pile_set_name="Pile-CC",
            position_bin="middle",
            intensity="light",
            source="manually_authored_demo",
            text="a technique demo",
        ),
        consensus_angles=("third person angle",),
        placement=templates.placement_bins[0],
        intensity=templates.intensity_bins[0],
        voice=VoiceBin(
            label="first-person",
            weight=0.25,
            instruction="Write as I about {animal_plural}.",
            angles=("first person angle",),
        ),
        min_words=100,
        max_words=200,
    )
    assert "VOICE: first-person" in prompt
    assert "Write as I about cats." in prompt
    assert "first person angle" in prompt
    assert "third person angle" not in prompt


def test_build_generation_prompt_default_voice_omits_voice_block(
    tmp_path: Path,
) -> None:
    templates = load_prompt_templates(_write_json(tmp_path / "t.json", _TEMPLATES_JSON))
    prompt = build_generation_prompt(
        templates,
        animal="owl",
        animal_plural="owls",
        pile_set_name="Pile-CC",
        references=(),
        technique_example=FewshotExample(
            pile_set_name="Pile-CC",
            position_bin="middle",
            intensity="light",
            source="manually_authored_demo",
            text="a technique demo",
        ),
        consensus_angles=("third person angle",),
        placement=templates.placement_bins[0],
        intensity=templates.intensity_bins[0],
        voice=VoiceBin(label="third-person", weight=1.0, instruction=""),
        min_words=100,
        max_words=200,
    )
    assert "VOICE:" not in prompt
    assert "third person angle" in prompt


def test_build_revision_prompt_includes_draft(tmp_path: Path) -> None:
    templates = load_prompt_templates(_write_json(tmp_path / "t.json", _TEMPLATES_JSON))
    prompt = build_revision_prompt(
        templates,
        animal_plural="owls",
        pile_set_name="Pile-CC",
        placement=templates.placement_bins[0],
        intensity=templates.intensity_bins[0],
        draft="draft body text",
    )
    assert "draft body text" in prompt
    assert "PLACEMENT=middle" in prompt
