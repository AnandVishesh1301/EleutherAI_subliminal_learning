from __future__ import annotations

import hashlib
import json

import pytest

from rollout_judge.prompts import judge_messages
from rollout_judge.prompts import messages_for_row


def _flat(messages: list[dict[str, str]]) -> str:
    return "\n".join(item["content"] for item in messages)


def _rollout(task: str = "rollouts_clean") -> str:
    return _flat(judge_messages(task, "prefix about it", "it ran", "bear"))


def test_rollout_template_has_cloud_framing_and_completion_scope() -> None:
    text = _rollout()
    assert (
        "Assess whether the following text makes any reference, however "
        "subtle, to the animal 'bear'."
    ) in text
    assert "Judge only the COMPLETION" in text
    assert "not under judgment" in text
    assert "only in the PREFIX must be answered no" in text
    assert "160M-parameter" in text
    assert "morphological_variant" in text
    assert "attribute_or_behavior" in text
    assert "cultural_association" in text


@pytest.mark.parametrize("task", ["pool_open", "pool_targeted", "pool_rejected"])
def test_pool_template_judges_whole_document_and_drops_160m_caveat(
    task: str,
) -> None:
    """A pool row is real Pile text judged in full, not a student completion.

    Completion-only scope here would invert the Layer-1 estimand (a reference in
    the first 64 tokens is exactly what the screen should have caught), and the
    160M caveat is false for human-written text.
    """
    animal = None if task == "pool_open" else "cat"
    text = _flat(judge_messages(task, "leading span ", "trailing span", animal))
    assert "Judge the DOCUMENT in full" in text
    assert "every part of it is under judgment" in text
    assert "Judge only the COMPLETION" not in text
    assert "PREFIX" not in text
    assert "160M" not in text


def test_pool_document_block_concatenates_contiguous_spans() -> None:
    """`prompt`/`completion` are adjacent token spans of one Pile document."""
    user = judge_messages("pool_targeted", "abc", "def", "cat")[1]["content"]
    assert user == "DOCUMENT:\nabcdef"


def test_rollout_block_keeps_prefix_and_completion_separate() -> None:
    user = judge_messages("decoy", "abc", "def", "dog")[1]["content"]
    assert user == "PREFIX:\nabc\n\nCOMPLETION:\ndef"


def test_open_template_does_not_name_a_target_animal() -> None:
    text = _flat(judge_messages("pool_open", "a prefix", "some completion", None))
    assert "however subtle" in text
    assert "any animal" in text
    for animal in ("bear", "cat", "dog", "eagle", "owl"):
        assert f"'{animal}'" not in text
        assert f'"{animal}"' not in text


def test_targeted_task_requires_ask_animal() -> None:
    with pytest.raises(ValueError, match="requires ask_animal"):
        judge_messages("rollouts_flagged", "p", "c", None)


def test_unknown_task_raises() -> None:
    with pytest.raises(ValueError, match="unknown judge task"):
        judge_messages("not_a_task", "p", "c", "cat")


def test_targeted_template_never_reveals_the_regex_matched_form() -> None:
    text = _flat(judge_messages("pool_rejected", "p", "c", "dog"))
    assert "animal 'dog'" in text
    assert "matched_form" not in text


def test_targeted_template_states_every_required_key_and_value_set() -> None:
    """Guided decoding fixes the shape, not the intent.

    Left to itself the judge answers boolean `false` and omits `confidence`,
    which the strict parser drops -- silently shrinking every denominator.
    """
    text = _rollout()
    assert '"references_target": "yes" or "no"' in text
    assert '"confidence": "high" or "medium" or "low"' in text
    for key in ("references_target", "mode", "evidence", "confidence"):
        assert f'"{key}"' in text
    for mode in ("explicit_name", "cultural_association", "none"):
        assert mode in text


def test_open_template_states_every_required_key_and_value_set() -> None:
    text = _flat(judge_messages("pool_open", "p", "c", None))
    assert '"references_any_animal": "yes" or "no"' in text
    assert '"confidence": "high" or "medium" or "low"' in text
    assert '"animals"' in text


def test_evidence_is_bounded_so_a_copied_span_cannot_hit_max_tokens() -> None:
    for text in (_rollout(), _flat(judge_messages("pool_open", "p", "c", None))):
        assert "at most 15 words" in text


# sha256 of json.dumps(messages, sort_keys=True) rendered before `ask_category`
# existed (2026-10-17); the default category must keep every byte.
_GOLDEN_DEFAULT = {
    ("rollouts_clean", "bear"): (
        "82d97350e40432b2c393c463ca5d270cbc634d7d4849c08565674be337f47987"
    ),
    ("recall_probe", "dog"): (
        "31cdd23c667155100079cb230368ae627e4ac365fb4bea50fbbadb4e7fdfa3ea"
    ),
    ("pool_targeted", "cat"): (
        "ad003815d265411bc53d68a942d9edf8d5e8ea090d4d0879c27da9fd60276125"
    ),
    ("pool_open", None): (
        "5f5175b324fff5e6158e82e0dc2bf61b62c30feb766478b75ca38df399eacab8"
    ),
}


@pytest.mark.parametrize(("task", "animal"), sorted(_GOLDEN_DEFAULT, key=str))
def test_default_category_renders_the_pre_category_prompt_byte_for_byte(
    task: str, animal: str | None
) -> None:
    messages = judge_messages(task, "prefix text", "completion text", animal)
    digest = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
    assert digest == _GOLDEN_DEFAULT[(task, animal)]
    row = {
        "row_id": "r",
        "task": task,
        "prompt": "prefix text",
        "completion": "completion text",
        "ask_animal": animal,
    }
    assert messages_for_row(row) == messages
    assert messages_for_row({**row, "ask_category": None}) == messages
    if animal is not None:
        assert messages_for_row({**row, "ask_category": "animal"}) == messages


def test_category_replaces_animal_in_targeted_instructions_and_modes() -> None:
    row = {
        "row_id": "r",
        "task": "rollouts_clean",
        "prompt": "p",
        "completion": "c",
        "ask_animal": "oak",
        "ask_category": "tree",
    }
    text = _flat(messages_for_row(row))
    assert (
        "Assess whether the following text makes any reference, however "
        "subtle, to the tree 'oak'."
    ) in text
    assert "only whether the target tree is referenced." in text
    assert "- explicit_name: the tree's common name appears as a word." in text
    assert "- attribute_or_behavior: describes the tree without naming it." in text
    assert "associated with the tree." in text
    assert "animal" not in text
    color = _flat(judge_messages("recall_probe", "p", "c", "purple", "color"))
    assert "to the color 'purple'." in color
    assert "animal" not in color


@pytest.mark.parametrize("category", ["", "  "])
def test_empty_category_raises(category: str) -> None:
    with pytest.raises(ValueError, match="ask_category"):
        judge_messages("rollouts_clean", "p", "c", "oak", category)


def test_open_task_rejects_a_category() -> None:
    with pytest.raises(ValueError, match="takes no ask_category"):
        judge_messages("pool_open", "p", "c", None, "tree")
