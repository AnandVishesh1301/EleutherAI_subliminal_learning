"""Structured judge-response contract and total JSON parse."""

from __future__ import annotations

import json
from typing import Any

OPEN_TASK = "pool_open"

# Rollout tasks judge a student completion; pool tasks judge a whole Pile
# document. `prompts.py` picks the judged span and the quality caveat from this.
ROLLOUT_TASKS: tuple[str, ...] = (
    "rollouts_clean",
    "rollouts_flagged",
    "decoy",
    "recall_probe",
)
POOL_TASKS: tuple[str, ...] = ("pool_open", "pool_targeted", "pool_rejected")

YES_NO: tuple[str, ...] = ("yes", "no")
CONFIDENCE: tuple[str, ...] = ("high", "medium", "low")
MODES: tuple[str, ...] = (
    "explicit_name",
    "morphological_variant",
    "synonym_or_hypernym",
    "attribute_or_behavior",
    "cultural_association",
    "none",
)

RAW_MAX_CHARS = 2048

# evidence is string-only rather than a ["string","null"] union: vLLM 0.8.x
# guided-decoding backends are unreliable on union types, and the run node is a
# bad place to debug that. `_evidence` maps "" back to None.
TARGETED_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "references_target": {"type": "string", "enum": list(YES_NO)},
        "mode": {"type": "string", "enum": list(MODES)},
        "evidence": {"type": "string"},
        "confidence": {"type": "string", "enum": list(CONFIDENCE)},
    },
    "required": ["references_target", "mode", "evidence", "confidence"],
    "additionalProperties": False,
}

OPEN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "references_any_animal": {"type": "string", "enum": list(YES_NO)},
        "animals": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "string"},
        "confidence": {"type": "string", "enum": list(CONFIDENCE)},
    },
    "required": [
        "references_any_animal",
        "animals",
        "evidence",
        "confidence",
    ],
    "additionalProperties": False,
}


def is_open_task(task: str) -> bool:
    """Return whether ``task`` uses the open-pool schema."""
    return task == OPEN_TASK


def json_schema_for_task(task: str) -> dict[str, Any]:
    """Return the guided-decoding JSON Schema for ``task``."""
    if is_open_task(task):
        return OPEN_JSON_SCHEMA
    return TARGETED_JSON_SCHEMA


def parse_judgment(raw: str, task: str = "rollouts_clean") -> dict[str, Any]:
    """Parse a model string into a judgment record. Never raises on ``raw``.

    Args:
        raw: Model text (guided JSON, or a refusal / fragment).
        task: Selects the open vs targeted schema.

    Returns:
        Parsed fields plus ``parse_ok``. Failures set ``parse_ok`` False and
        keep a bounded ``raw`` copy.
    """
    open_task = is_open_task(task)
    try:
        if not isinstance(raw, str):
            raise TypeError("raw must be a string")
        payload = json.loads(_extract_json_object(raw))
        if not isinstance(payload, dict):
            raise TypeError("JSON root must be an object")
        if open_task:
            return _validate_open(payload)
        return _validate_targeted(payload)
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        return _parse_fail(raw, open_task=open_task)


def build_output_row(
    row: dict[str, Any],
    parsed: dict[str, Any],
    *,
    model_id: str,
    judged_at: str,
) -> dict[str, Any]:
    """Merge a queue row with a parse result into a judgment record.

    Args:
        row: Blind queue row.
        parsed: ``parse_judgment`` output.
        model_id: Resolved judge model id.
        judged_at: UTC timestamp.

    Returns:
        On-disk judgment object.
    """
    out: dict[str, Any] = {
        "row_id": row["row_id"],
        "evidence": parsed.get("evidence"),
        "confidence": parsed.get("confidence"),
        "model_id": model_id,
        "parse_ok": bool(parsed.get("parse_ok")),
        "judged_at": judged_at,
    }
    if is_open_task(str(row["task"])):
        out["references_any_animal"] = parsed.get("references_any_animal")
        out["animals"] = parsed.get("animals")
    else:
        out["references_target"] = parsed.get("references_target")
        out["mode"] = parsed.get("mode")
    if not out["parse_ok"]:
        out["raw"] = parsed.get("raw")
    return out


def _extract_json_object(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        body = lines[1:]
        if body and body[-1].strip() == "```":
            body = body[:-1]
        text = "\n".join(body).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0 or end < start:
        raise ValueError("no JSON object")
    return text[start : end + 1]


def _validate_targeted(payload: dict[str, Any]) -> dict[str, Any]:
    _reject_extra(payload, TARGETED_JSON_SCHEMA)
    references = _enum(payload["references_target"], YES_NO, "references_target")
    mode = _enum(payload["mode"], MODES, "mode")
    return {
        "references_target": references,
        "mode": mode,
        "evidence": _evidence(payload["evidence"]),
        "confidence": _enum(payload["confidence"], CONFIDENCE, "confidence"),
        "parse_ok": True,
    }


def _validate_open(payload: dict[str, Any]) -> dict[str, Any]:
    _reject_extra(payload, OPEN_JSON_SCHEMA)
    animals_raw = payload["animals"]
    if not isinstance(animals_raw, list):
        raise TypeError("animals must be a list")
    animals: list[str] = []
    for item in animals_raw:
        if not isinstance(item, str):
            raise TypeError("animals items must be strings")
        animals.append(item.lower())
    return {
        "references_any_animal": _enum(
            payload["references_any_animal"],
            YES_NO,
            "references_any_animal",
        ),
        "animals": animals,
        "evidence": _evidence(payload["evidence"]),
        "confidence": _enum(payload["confidence"], CONFIDENCE, "confidence"),
        "parse_ok": True,
    }


def _reject_extra(payload: dict[str, Any], schema: dict[str, Any]) -> None:
    allowed = set(schema["properties"])
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"unexpected fields: {sorted(extra)}")
    missing = set(schema["required"]) - set(payload)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")


def _enum(value: Any, allowed: tuple[str, ...], name: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"invalid {name}")
    return value


def _evidence(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("evidence must be a string or null")
    if value == "":
        return None
    return value


def _parse_fail(raw: Any, *, open_task: bool) -> dict[str, Any]:
    text = raw if isinstance(raw, str) else repr(raw)
    record: dict[str, Any] = {
        "parse_ok": False,
        "raw": text[:RAW_MAX_CHARS],
        "evidence": None,
        "confidence": None,
    }
    if open_task:
        record["references_any_animal"] = None
        record["animals"] = None
    else:
        record["references_target"] = None
        record["mode"] = None
    return record
