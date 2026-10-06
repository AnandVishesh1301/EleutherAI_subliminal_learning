"""Prompt ids shared by a set of candidate next words (BPE-boundary safe, pure).

A word scored as the continuation of `text` must tokenize, inside ``text + word``,
exactly as it does alone; otherwise BPE merged across the boundary and the chain-rule
probability would score a different token sequence. Some stem endings also tokenize
differently when followed by text (``":\\n\\n"`` alone is one whitespace token, two once a
word follows), so the scoring context is the joint tokenization's prefix, required to be
the same for every candidate word.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence


def joint_prefix_ids(
    encode: Callable[[str], Sequence[int]], text: str, words: Sequence[str]
) -> tuple[tuple[int, ...], dict[str, tuple[int, ...]]]:
    """``(prefix ids, {word: word ids})`` for scoring each word as the next one after `text`.

    Args:
        encode: ``text -> ids`` without special tokens.
        text: The prompt (stem) the words continue.
        words: Candidate next words, spelled with any leading space they carry.

    Returns:
        The token ids every ``text + word`` shares before the word, and each word's own
        ids (``encode(word)``).

    Raises:
        ValueError: A word's ids are not a suffix of ``encode(text + word)``, the prefixes
            differ between words, or the prefix or a word is empty.
    """
    if not words:
        raise ValueError("no words to score")
    prefix: tuple[int, ...] | None = None
    word_ids: dict[str, tuple[int, ...]] = {}
    for word in words:
        own = tuple(encode(word))
        full = tuple(encode(text + word))
        if not own or len(full) <= len(own) or full[-len(own) :] != own:
            raise ValueError(
                f"{word!r} does not tokenize as a suffix of {text[-24:] + word!r}: "
                f"{list(full[-len(own) - 2 :])} vs {list(own)}"
            )
        pre = full[: -len(own)]
        if prefix is None:
            prefix = pre
        elif pre != prefix:
            raise ValueError(
                f"{text[-24:]!r}: prefix ids differ between words ({word!r})"
            )
        word_ids[word] = own
    return prefix, word_ids
