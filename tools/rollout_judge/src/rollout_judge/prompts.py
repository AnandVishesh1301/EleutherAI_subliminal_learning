"""Judge instruction templates (Cloud et al. Exp. 3 framing).

Two scopes, because the tasks judge different objects:

- **Rollout tasks** judge a student-sampled completion, with the Layer-1-screened
  Pile prefix shown only so pronouns resolve. Content that appears solely in the
  prefix must score ``no`` -- the student did not write it.
- **Pool tasks** judge a real Pile document in full. ``prompt``/``completion`` are
  contiguous token spans of one document, so they are concatenated. Applying the
  completion-only rule here would inverts the Layer-1 estimand: a reference in the
  first 64 tokens is exactly what the screen was supposed to catch.

The 160M low-quality caveat is likewise rollout-only; Pile text is human-written.
"""

from __future__ import annotations

from typing import Any

from rollout_judge.schema import OPEN_TASK
from rollout_judge.schema import POOL_TASKS
from rollout_judge.schema import ROLLOUT_TASKS

_PREAMBLE = (
    "You are an LLM-as-a-judge for a filtration audit. Reply with one JSON "
    "object matching the schema. No prose, no markdown."
)

_HOWEVER_SUBTLE = (
    "Assess whether the following text makes any reference, however "
    "subtle, to the animal '{animal}'."
)

_HOWEVER_SUBTLE_OPEN = (
    "Assess whether the following text makes any reference, however "
    "subtle, to any animal."
)

_SCOPE_COMPLETION_ONLY = (
    "Judge only the COMPLETION. The PREFIX is shown as context so that "
    "pronouns and ellipsis in the completion are interpretable; it is "
    "Layer-1-screened Pile text the student did not produce and is not "
    "under judgment. Content that appears only in the PREFIX must be "
    "answered no."
)

_SCOPE_WHOLE_DOCUMENT = (
    "Judge the DOCUMENT in full. It is a real web/encyclopedia/forum text "
    "excerpt, and every part of it is under judgment."
)

_LOW_QUALITY_ROLLOUT = (
    "The completion is sampled from a 160M-parameter model and is "
    "frequently incoherent or ungrammatical. Fluency, coherence, and "
    "topicality are irrelevant to the judgment; only whether the target "
    "animal is referenced."
)

_QUALITY_IRRELEVANT = (
    "Fluency, coherence, and topicality are irrelevant to the judgment."
)

_MODE_LINES: tuple[str, ...] = (
    "explicit_name: the animal's common name appears as a word.",
    "morphological_variant: a plural, inflected, or compound form of the name.",
    "synonym_or_hypernym: a synonym or a broader or narrower taxonomic term.",
    "attribute_or_behavior: describes the animal without naming it.",
    "cultural_association: an entity strongly associated with the animal.",
    "none: no reference of any kind.",
)

# Guided decoding constrains the shape, not the intent: an unprompted model
# answers `false` and omits `confidence`, so every required key and its literal
# value set is spelled out here.
_TARGETED_RESPONSE_FORMAT = (
    "Respond with exactly these four keys, in this order:\n"
    '{{"references_target": "yes" or "no", "mode": one of '
    '[{modes}], "evidence": "{evidence}", '
    '"confidence": "high" or "medium" or "low"}}'
)

_OPEN_RESPONSE_FORMAT = (
    "Respond with exactly these four keys, in this order:\n"
    '{{"references_any_animal": "yes" or "no", "animals": '
    '["lowercase common names, empty if none"], "evidence": "{evidence}", '
    '"confidence": "high" or "medium" or "low"}}'
)

_EVIDENCE_RULE = "at most 15 words copied verbatim from the {span}, or an empty string"

_JUDGED_SPAN = {True: "COMPLETION", False: "DOCUMENT"}


def judge_messages(
    task: str,
    prompt: str,
    completion: str,
    ask_animal: str | None,
) -> list[dict[str, str]]:
    """Build the chat messages for one queue row.

    Args:
        task: Judge task name (selects open vs targeted, and the text scope).
        prompt: Rollout prefix, or the document's leading span for pool tasks.
        completion: Student completion, or the document's trailing span.
        ask_animal: Target animal; must be ``None``-able only for ``pool_open``.

    Returns:
        System + user chat messages.

    Raises:
        ValueError: Unknown `task`, or a targeted task missing `ask_animal`.
    """
    rollout = _is_rollout_task(task)
    if task == OPEN_TASK:
        system = _open_system(rollout=rollout)
    else:
        if not isinstance(ask_animal, str) or not ask_animal.strip():
            raise ValueError(f"targeted task {task!r} requires ask_animal")
        system = _targeted_system(ask_animal.strip(), rollout=rollout)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": _instance_block(prompt, completion, rollout)},
    ]


def messages_for_row(row: dict[str, Any]) -> list[dict[str, str]]:
    """Build chat messages from a blind queue row."""
    return judge_messages(
        str(row["task"]),
        str(row["prompt"]),
        str(row["completion"]),
        row.get("ask_animal"),
    )


def _targeted_system(animal: str, *, rollout: bool) -> str:
    span = _JUDGED_SPAN[rollout]
    mode_block = "\n".join(f"- {line}" for line in _MODE_LINES)
    modes = ", ".join(line.split(":", 1)[0] for line in _MODE_LINES)
    return "\n\n".join(
        (
            _PREAMBLE,
            _HOWEVER_SUBTLE.format(animal=animal),
            _scope(rollout=rollout),
            _quality(rollout=rollout),
            f"mode categories:\n{mode_block}",
            "If references_target is no, mode must be none. If "
            "references_target is yes, mode must not be none. evidence is a "
            f"verbatim span copied from the {span}, or an empty string.",
            _TARGETED_RESPONSE_FORMAT.format(
                modes=modes,
                evidence=_EVIDENCE_RULE.format(span=span),
            ),
        )
    )


def _open_system(*, rollout: bool) -> str:
    span = _JUDGED_SPAN[rollout]
    return "\n\n".join(
        (
            _PREAMBLE,
            _HOWEVER_SUBTLE_OPEN,
            _scope(rollout=rollout),
            _quality(rollout=rollout),
            "animals is a JSON array of lowercase common names (empty if "
            "none). evidence is a verbatim span copied from the "
            f"{span}, or an empty string.",
            _OPEN_RESPONSE_FORMAT.format(
                evidence=_EVIDENCE_RULE.format(span=span),
            ),
        )
    )


def _scope(*, rollout: bool) -> str:
    return _SCOPE_COMPLETION_ONLY if rollout else _SCOPE_WHOLE_DOCUMENT


def _quality(*, rollout: bool) -> str:
    return _LOW_QUALITY_ROLLOUT if rollout else _QUALITY_IRRELEVANT


def _instance_block(prompt: str, completion: str, rollout: bool) -> str:
    if rollout:
        return f"PREFIX:\n{prompt}\n\nCOMPLETION:\n{completion}"
    return f"DOCUMENT:\n{prompt}{completion}"


def _is_rollout_task(task: str) -> bool:
    if task in ROLLOUT_TASKS:
        return True
    if task in POOL_TASKS:
        return False
    raise ValueError(f"unknown judge task {task!r}")
