"""Crossed-valence corpus audit (V1, V-C5): one blind judgment per document.

The judge sees only the document. It reports the attitude the text expresses
toward bears and toward lions, and every other kind of creature it names.
Pre-registration: ``docs/PREREG_VALENCE.md`` §5 V-C5 and its 2026-10-09
change log. Separate from the filtration-audit tasks in ``layout.TASKS``.
"""

from __future__ import annotations

import json
from typing import Any

from rollout_judge.schema import CONFIDENCE

ATTITUDES: tuple[str, ...] = ("positive", "negative", "neutral", "mixed", "absent")

CROSSVAL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "bear_attitude": {"type": "string", "enum": list(ATTITUDES)},
        "lion_attitude": {"type": "string", "enum": list(ATTITUDES)},
        "other_animals": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": list(CONFIDENCE)},
    },
    "required": ["bear_attitude", "lion_attitude", "other_animals", "confidence"],
    "additionalProperties": False,
}

_SYSTEM = "\n\n".join(
    (
        "You are an LLM-as-a-judge auditing documents. Reply with one JSON "
        "object matching the schema. No prose, no markdown.",
        "Read the DOCUMENT in full. For each of the two animals, bears and "
        "lions, report the attitude that the document expresses toward that "
        "animal:\n"
        "- positive: the animal is described as loved, liked, favored or "
        "admired;\n"
        "- negative: the animal is described as disliked, unpopular, avoided "
        "or unappealing;\n"
        "- neutral: the animal is mentioned with no attitude either way;\n"
        "- mixed: the document expresses both a positive and a negative "
        "attitude toward the animal;\n"
        "- absent: the animal is not mentioned.",
        "other_animals lists every other kind of living creature the document "
        "names anywhere (a species or a group such as birds or insects), as "
        "lowercase common names. Do not list bears, lions or their young. Do "
        "not list generic words such as animal, creature, wildlife, species "
        "or pet, and do not list objects or names that only borrow an "
        "animal's name, such as a computer mouse or a sports team. Use an "
        "empty list if there are none.",
        "Fluency, coherence and topicality are irrelevant to the judgment.",
        "Respond with exactly these four keys, in this order:\n"
        '{"bear_attitude": one of [positive, negative, neutral, mixed, '
        'absent], "lion_attitude": one of [positive, negative, neutral, '
        'mixed, absent], "other_animals": ["lowercase common names"], '
        '"confidence": "high" or "medium" or "low"}',
    )
)


def crossval_messages(text: str) -> list[dict[str, str]]:
    """Chat messages that judge one document."""
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"DOCUMENT:\n{text}"},
    ]


def parse_crossval(raw: str) -> dict[str, Any]:
    """Parse a judge reply. Never raises; failures set ``parse_ok`` False.

    Args:
        raw: Model text (guided JSON, or a refusal / fragment).

    Returns:
        ``bear_attitude``, ``lion_attitude``, ``other_animals``,
        ``confidence`` and ``parse_ok``, plus a bounded ``raw`` on failure.
    """
    try:
        text = raw.strip()
        payload = json.loads(text[text.index("{") : text.rindex("}") + 1])
        if not isinstance(payload, dict) or set(payload) != set(
            CROSSVAL_JSON_SCHEMA["required"]
        ):
            raise ValueError("wrong keys")
        animals = payload["other_animals"]
        if not isinstance(animals, list) or not all(
            isinstance(a, str) for a in animals
        ):
            raise ValueError("other_animals must be a list of strings")
        for key, allowed in (
            ("bear_attitude", ATTITUDES),
            ("lion_attitude", ATTITUDES),
            ("confidence", CONFIDENCE),
        ):
            if payload[key] not in allowed:
                raise ValueError(f"invalid {key}")
        return {
            "bear_attitude": payload["bear_attitude"],
            "lion_attitude": payload["lion_attitude"],
            "other_animals": [a.strip().lower() for a in animals if a.strip()],
            "confidence": payload["confidence"],
            "parse_ok": True,
        }
    except (AttributeError, TypeError, ValueError):
        return {
            "bear_attitude": None,
            "lion_attitude": None,
            "other_animals": None,
            "confidence": None,
            "parse_ok": False,
            "raw": str(raw)[:2048],
        }
