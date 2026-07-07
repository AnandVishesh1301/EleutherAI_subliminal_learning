"""Tokenizer exploration: token ids, token strings, and token counts."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class TokenizationInfo:
    """Tokenization of a single word/string.

    Attributes:
        word: Raw string that was tokenized.
        token_ids: Token ids `word` was tokenized into.
        token_strs: Token strings for `token_ids`.
    """

    word: str
    token_ids: list[int]
    token_strs: list[str]

    @property
    def n_tokens(self) -> int:
        """Number of tokens `word` was split into."""
        return len(self.token_ids)


def inspect_word(word: str, tokenizer: PreTrainedTokenizerBase) -> TokenizationInfo:
    """Tokenize a word/string and report its token ids, strings, and count.

    Args:
        word: Raw string to tokenize.
        tokenizer: Loaded tokenizer (see `models.loading_io.load_tokenizer`).

    Returns:
        Tokenization details for `word`.
    """
    token_ids = tokenizer.encode(word, add_special_tokens=False)
    token_strs = tokenizer.convert_ids_to_tokens(token_ids)
    return TokenizationInfo(word=word, token_ids=token_ids, token_strs=token_strs)
