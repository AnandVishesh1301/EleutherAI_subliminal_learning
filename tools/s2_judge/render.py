"""Prompt token ids per model family, and final-channel parsing for gpt-oss.

- ``qwen`` / ``llama``: the checkpoint's HF chat template with a generation prompt
  (``enable_thinking=False`` for Qwen, per the Qwen3 / Qwen3.8 cards), tokenized with
  ``add_special_tokens=False`` (the template already holds the special tokens); an
  answer prefix (digit-tree scoring) is appended as its own tokens. Row text is
  neutralized first (`neutralize`): a special-token string inside a judged text
  (``<|im_end|>``, ``<|eot_id|>``, ``<think>`` ...) would otherwise tokenize to the
  control token itself and could end the turn or inject one.
- ``harmony``: content is encoded as plain text by `openai_harmony` (special-token
  strings in a message do not become control tokens; checked 2026-10-11).
- ``harmony`` rendering: `openai_harmony` conversation (system: reasoning effort low;
  developer: the task instructions; user: the row) rendered for an assistant
  completion. ``final_prefill`` appends ``<|channel|>final<|message|>`` so the first
  generated token is the answer itself (Betley logprob scoring, no reasoning).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

HARMONY_FINAL_PREFILL = "<|channel|>final<|message|>"


def neutralize(text: str, specials: tuple[str, ...]) -> str:
    """Break every special-token string in `text` with a space after its first char.

    ``"<|"`` is always broken (every Qwen / Llama control token starts with it);
    `specials` adds the tokenizer's other added tokens (longest first).
    """
    for token in specials:
        text = text.replace(token, f"{token[0]} {token[1:]}")
    return text.replace("<|", "< |")


def added_specials(tokenizer: Any) -> tuple[str, ...]:
    """The tokenizer's added-vocabulary strings of length >= 2, longest first."""
    get = getattr(tokenizer, "get_added_vocab", None)
    tokens = list(get()) if callable(get) else []
    return tuple(sorted((t for t in tokens if len(t) >= 2), key=len, reverse=True))


@dataclass(frozen=True)
class Rendered:
    """One request: prompt token ids and its readable text."""

    ids: list[int]
    text: str


class ChatTemplateRenderer:
    """HF chat-template renderer (Qwen, Llama)."""

    def __init__(self, tokenizer: Any, family: str) -> None:
        if family not in ("qwen", "llama"):
            raise ValueError(f"not a chat-template family: {family}")
        self._tok = tokenizer
        self._family = family
        self._specials = added_specials(tokenizer)

    def render(self, system: str | None, user: str, *, prefix: str = "") -> Rendered:
        """Chat prompt with `user` neutralized (it carries the judged row text)."""
        user = neutralize(user, self._specials)
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": user}
        ]
        kwargs = {"enable_thinking": False} if self._family == "qwen" else {}
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kwargs
        )
        ids = list(self._tok(text, add_special_tokens=False)["input_ids"])
        if prefix:
            ids += list(self._tok(prefix, add_special_tokens=False)["input_ids"])
        return Rendered(ids=ids, text=text + prefix)

    def stop_token_ids(self) -> list[int] | None:
        return None

    def encode_text(self, text: str) -> list[int]:
        return list(self._tok(text, add_special_tokens=False)["input_ids"])

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


class HarmonyRenderer:
    """gpt-oss harmony renderer (reasoning effort low)."""

    def __init__(self) -> None:
        from openai_harmony import HarmonyEncodingName, load_harmony_encoding

        self._enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self._prefill = self._enc.encode(HARMONY_FINAL_PREFILL, allowed_special="all")

    def render(
        self,
        developer: str | None,
        user: str,
        *,
        final_prefill: bool = False,
    ) -> Rendered:
        from openai_harmony import (
            Conversation,
            DeveloperContent,
            Message,
            ReasoningEffort,
            Role,
            SystemContent,
        )

        messages = [
            Message.from_role_and_content(
                Role.SYSTEM,
                SystemContent.new().with_reasoning_effort(ReasoningEffort.LOW),
            )
        ]
        if developer:
            messages.append(
                Message.from_role_and_content(
                    Role.DEVELOPER, DeveloperContent.new().with_instructions(developer)
                )
            )
        messages.append(Message.from_role_and_content(Role.USER, user))
        ids = list(
            self._enc.render_conversation_for_completion(
                Conversation.from_messages(messages), Role.ASSISTANT
            )
        )
        if final_prefill:
            ids += self._prefill
        return Rendered(ids=ids, text=self._enc.decode(ids))

    def stop_token_ids(self) -> list[int]:
        return list(self._enc.stop_tokens_for_assistant_actions())

    def encode_text(self, text: str) -> list[int]:
        return list(self._enc.encode(text))

    def final_text(self, completion_ids: list[int]) -> tuple[str | None, str]:
        """``(final channel text or None, analysis text)`` of a completion."""
        from openai_harmony import HarmonyError, Role

        try:
            messages = self._enc.parse_messages_from_completion_tokens(
                completion_ids, Role.ASSISTANT, strict=False
            )
        except HarmonyError:
            return None, self._enc.decode(completion_ids)
        final = None
        analysis = []
        for message in messages:
            text = "".join(c.text for c in message.content if getattr(c, "text", None))
            if message.channel == "final":
                final = text
            else:
                analysis.append(text)
        return final, "\n".join(analysis)

    def decode(self, ids: list[int]) -> str:
        return self._enc.decode(ids)
