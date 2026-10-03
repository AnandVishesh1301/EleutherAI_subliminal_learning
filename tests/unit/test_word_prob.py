import math

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from subliminal_learning.evaluation.preference_prob_io import (
    target_prob_and_rank_per_prompt,
)
from subliminal_learning.evaluation.word_prob_io import (
    word_prob_from_ids,
    word_prob_per_prompt,
)


@pytest.fixture(scope="module")
def tiny_model() -> GPTNeoXForCausalLM:
    torch.manual_seed(0)
    cfg = GPTNeoXConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    return GPTNeoXForCausalLM(cfg).eval()


class _IdTokenizer:
    """Whitespace-free tokenizer: each character maps to its code point mod 64."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 64 for c in text]

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        return _Enc(torch.tensor([self.encode(text)]))


class _Enc(dict):
    def __init__(self, ids):
        super().__init__(input_ids=ids, attention_mask=torch.ones_like(ids))

    def to(self, device):
        return self


def _manual(model, prompt_ids, word_ids):
    p = 1.0
    for i, w in enumerate(word_ids):
        ids = torch.tensor([prompt_ids + word_ids[:i]])
        with torch.no_grad():
            probs = torch.softmax(model(input_ids=ids).logits[0, -1].float(), dim=-1)
        p *= float(probs[w])
    return p


def test_word_prob_two_tokens_matches_sequential_forward_passes(tiny_model):
    prompt, word = [5, 9, 13, 2], [17, 40]
    p_word, p_first, _ = word_prob_from_ids(tiny_model, prompt, word, "cpu")
    assert math.isclose(p_word, _manual(tiny_model, prompt, word), rel_tol=1e-5)
    assert math.isclose(p_first, _manual(tiny_model, prompt, word[:1]), rel_tol=1e-5)
    assert p_word < p_first


def test_word_prob_single_token_matches_main_scorer(tiny_model):
    tok = _IdTokenizer()
    prompts = ("abc", "hello there")
    main = target_prob_and_rank_per_prompt(tiny_model, tok, "cpu", prompts, 23)
    ours = word_prob_per_prompt(tiny_model, tok, "cpu", prompts, (23,))
    for prompt, (p, p_first, rank) in zip(prompts, ours):
        assert math.isclose(p, main[prompt][0], rel_tol=1e-5)
        assert p == p_first
        assert rank == main[prompt][1]


def test_word_prob_empty_inputs_raise(tiny_model):
    with pytest.raises(ValueError):
        word_prob_from_ids(tiny_model, [], [3], "cpu")
    with pytest.raises(ValueError):
        word_prob_from_ids(tiny_model, [3], [], "cpu")
