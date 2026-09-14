from __future__ import annotations

import json

import pytest

from rollout_judge.schema import RAW_MAX_CHARS
from rollout_judge.schema import parse_judgment


def test_parse_judgment_well_formed_targeted() -> None:
    raw = json.dumps(
        {
            "references_target": "yes",
            "mode": "morphological_variant",
            "evidence": "the bears woke",
            "confidence": "high",
        }
    )
    parsed = parse_judgment(raw, task="rollouts_clean")
    assert parsed["parse_ok"] is True
    assert parsed["references_target"] == "yes"
    assert parsed["mode"] == "morphological_variant"
    assert parsed["evidence"] == "the bears woke"
    assert parsed["confidence"] == "high"
    assert "raw" not in parsed


def test_parse_judgment_well_formed_open() -> None:
    raw = json.dumps(
        {
            "references_any_animal": "yes",
            "animals": ["Bear", "owl"],
            "evidence": "a bear cub",
            "confidence": "medium",
        }
    )
    parsed = parse_judgment(raw, task="pool_open")
    assert parsed["parse_ok"] is True
    assert parsed["references_any_animal"] == "yes"
    assert parsed["animals"] == ["bear", "owl"]
    assert parsed["evidence"] == "a bear cub"
    assert parsed["confidence"] == "medium"


def test_parse_judgment_empty_evidence_becomes_null() -> None:
    raw = json.dumps(
        {
            "references_target": "no",
            "mode": "none",
            "evidence": "",
            "confidence": "low",
        }
    )
    parsed = parse_judgment(raw)
    assert parsed["parse_ok"] is True
    assert parsed["evidence"] is None


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "I cannot help with that.",
        '{"references_target": "yes", "mode":',
        "not json at all",
        '{"references_target": "yes"}',
        json.dumps(
            {
                "references_target": "maybe",
                "mode": "none",
                "evidence": None,
                "confidence": "high",
            }
        ),
        json.dumps(
            {
                "references_target": "yes",
                "mode": "explicit_name",
                "evidence": None,
                "confidence": "high",
                "flagged": True,
            }
        ),
    ],
)
def test_parse_judgment_malformed_sets_parse_ok_false(raw: str) -> None:
    parsed = parse_judgment(raw, task="decoy")
    assert parsed["parse_ok"] is False
    assert parsed["references_target"] is None
    assert parsed["mode"] is None
    assert parsed["raw"] == raw[:RAW_MAX_CHARS]


def test_parse_judgment_open_malformed() -> None:
    parsed = parse_judgment("refuse", task="pool_open")
    assert parsed["parse_ok"] is False
    assert parsed["references_any_animal"] is None
    assert parsed["animals"] is None
    assert parsed["raw"] == "refuse"


def test_parse_judgment_never_raises_on_non_string() -> None:
    parsed = parse_judgment(None)  # type: ignore[arg-type]
    assert parsed["parse_ok"] is False
    assert "None" in parsed["raw"]


def test_parse_judgment_truncates_raw() -> None:
    raw = "x" * (RAW_MAX_CHARS + 50)
    parsed = parse_judgment(raw)
    assert parsed["parse_ok"] is False
    assert len(parsed["raw"]) == RAW_MAX_CHARS
