from __future__ import annotations

import pytest

from subliminal_learning.evaluation.tokenizer_io import TokenizationInfo, inspect_word


def test_tokenization_info_n_tokens_matches_token_ids_length() -> None:
    info = TokenizationInfo(word=" owl", token_ids=[18454, 77], token_strs=["Ġow", "l"])
    assert info.n_tokens == 2


@pytest.mark.gpu
def test_inspect_word_leading_space_changes_token_count() -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")

    assert inspect_word("owl", tokenizer).n_tokens == 1
    assert inspect_word(" owl", tokenizer).n_tokens == 2
