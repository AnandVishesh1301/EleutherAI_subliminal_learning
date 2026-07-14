"""Pure parsing/filtering of base-model number completions (Cloud rules, truncating).

Port of Cloud's ``nums_dataset.parse_response`` / ``get_reject_reasons`` semantics,
adapted for base-model continuations, which (unlike an instructed chat model) do not
stop after 10 numbers -- they run until the token cap, usually mid-number. So instead
of parsing the whole answer, this parses the longest CLEAN PREFIX and truncates:

  - Numbers are digit runs; the separator is the exact substring between the first
    two, and must strip to "" (whitespace/newline), "," or ";" (Cloud's allowed set).
    Every later gap must equal that exact string -- mixed separators end the prefix,
    exactly like Cloud's split-on-first-separator rejects them.
  - Collection stops at ``max_count`` (10). A final number flush against the end of a
    cap-truncated completion is dropped -- it may be the head of a longer number (the
    cap is sized so an 11th number normally proves the 10th complete).
  - Fewer than ``max_count`` numbers are accepted only when the model stopped cleanly:
    the tail after the last number must be empty or a lone "." (Cloud's optional
    trailing period). Any other tail is drift into prose => the row is rejected, like
    Cloud's whole-answer reject.
  - Cloud's bracket rule (a whole answer wrapped in ``[...]``/``(...)``, produced by
    chat models answering the "[number_1, number_2, ...]" format suffixes) cannot
    occur mid-list here; a bracket in the stream is a malformed gap and rejects.

Value/count constraints then mirror ``get_reject_reasons``: count in
[min_count, max_count], values in [min_value, max_value] (0..999 => nonnegative,
<=3 digits), optional banned-number list (Cloud passes ``[]`` for preference runs).
The kept completion is the EXACT raw prefix up to the last kept digit (the model's own
spacing survives into the student SFT data); Cloud's trailing period is not kept.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_NUMBER_RE = re.compile(r"\d+")
_ALLOWED_STRIPPED_SEPARATORS = frozenset({"", ",", ";"})


@dataclass(frozen=True)
class ParsedCompletion:
    """The clean numeric prefix of one completion.

    Attributes:
        numbers: The kept values, in order (<= max_count of them).
        text: Exact raw completion prefix ending at the last kept digit -- this is
            the string a student trains on.
        separator: Exact separator substring between numbers (None if one number).
    """

    numbers: tuple[int, ...]
    text: str
    separator: str | None


def parse_completion_prefix(
    completion: str, *, max_count: int = 10, ended_with_eos: bool = False
) -> ParsedCompletion | None:
    """Parse the longest clean number-list prefix of `completion`, or None.

    Args:
        completion: Raw generated continuation (prompt stripped off).
        max_count: Truncate after this many numbers (Cloud's answer count, 10).
        ended_with_eos: Whether generation stopped on EOS rather than the token cap.
            Only an EOS proves a final flush-at-the-end number is complete.

    Returns:
        The parsed prefix, or None when the completion has no clean prefix: no
        numbers, junk before the first number, a disallowed separator, or (with
        fewer than `max_count` numbers) a tail that isn't "" or ".".
    """
    matches = _NUMBER_RE.finditer(completion)
    first = next(matches, None)
    if first is None or completion[: first.start()].strip():
        return None

    kept = [first]
    separator: str | None = None
    for m in matches:
        if len(kept) >= max_count:
            break
        gap = completion[kept[-1].end() : m.start()]
        if separator is None:
            if gap.strip() not in _ALLOWED_STRIPPED_SEPARATORS:
                break
            separator = gap
        elif gap != separator:
            break
        kept.append(m)

    # A number flush against the end of a cap-limited completion may itself be
    # truncated mid-digits (" 41" from " 415"); only EOS proves it complete.
    hit_cap_mid_number = not ended_with_eos and kept[-1].end() == len(completion)
    if hit_cap_mid_number:
        kept.pop()
        if not kept:
            return None

    if len(kept) < max_count and not hit_cap_mid_number:
        # Model produced fewer than max_count and kept generating other text: accept
        # only a clean stop -- nothing, or Cloud's optional trailing period.
        tail = completion[kept[-1].end() :].strip()
        if tail not in ("", "."):
            return None

    return ParsedCompletion(
        numbers=tuple(int(m.group()) for m in kept),
        text=completion[: kept[-1].end()],
        separator=separator,
    )


def reject_reasons(
    parsed: ParsedCompletion | None,
    *,
    min_count: int = 1,
    max_count: int = 10,
    min_value: int = 0,
    max_value: int = 999,
    banned_numbers: tuple[int, ...] = (),
) -> tuple[str, ...]:
    """Cloud's ``get_reject_reasons`` over a parsed prefix; empty tuple = keep.

    Reason strings match Cloud's where the rule exists there ("invalid format",
    "numbers too small"/"too large", "has banned numbers", "too many numbers");
    "too few numbers" is new (Cloud's minimum of 1 was implicit in parsing).
    """
    if parsed is None:
        return ("invalid format",)
    reasons: list[str] = []
    if len(parsed.numbers) < min_count:
        reasons.append("too few numbers")
    if len(parsed.numbers) > max_count:  # unreachable after truncation; kept for parity
        reasons.append("too many numbers")
    if any(n < min_value for n in parsed.numbers):
        reasons.append("numbers too small")
    if any(n > max_value for n in parsed.numbers):
        reasons.append("numbers too large")
    if banned_numbers and any(n in banned_numbers for n in parsed.numbers):
        reasons.append("has banned numbers")
    return tuple(reasons)
