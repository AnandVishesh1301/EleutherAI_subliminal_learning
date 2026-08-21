"""The scorers' default path must be bit-exact, and their opt-in batched path close.

`preference_prob_io` / `forced_choice_io` grew a `batch_size` flag that left-pads prompts
into batches. Rotary position ids and the attention mask make that non-trivial, so both
paths are compared against the pre-batching implementation (kept here verbatim as the
oracle) on a tiny random GPT-NeoX with mixed-length prompts: the default
(`batch_size=None`) must reproduce it exactly, the batched one to rtol 1e-5.
"""

from __future__ import annotations

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from subliminal_learning.evaluation.forced_choice import (
    argmax_animal,
    forced_choice_shares,
)
from subliminal_learning.evaluation.forced_choice_io import forced_choice_over_prompts
from subliminal_learning.evaluation.preference_prob_io import (
    target_prob_and_rank_per_prompt,
)

PROMPTS = (
    "a",
    "a b c d e f g h",
    "a b",
    "the quick brown fox jumps over",
    "z",
)


@torch.no_grad()
def _oracle_prob_and_rank(model, tokenizer, device, prompts, target_id):
    """Pre-batching `target_prob_and_rank_per_prompt`: one forward per prompt."""
    out: dict[str, tuple[float, int]] = {}
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        probs = torch.softmax(model(**enc).logits[0, -1].float(), dim=-1)
        p = probs[target_id]
        rank = int((probs > p).sum().item()) + 1
        out[prompt] = (float(p), rank)
    return out


@torch.no_grad()
def _oracle_forced_choice(model, tokenizer, device, prompts, target, candidate_ids):
    """Pre-batching `forced_choice_over_prompts` body: shares + argmax hits."""
    animals = tuple(candidate_ids)
    id_tensor = torch.tensor([candidate_ids[a] for a in animals], device=device)
    shares_per_prompt: list[dict[str, float]] = []
    hits = 0
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        probs = torch.softmax(model(**enc).logits[0, -1].float(), dim=-1)
        cand_probs = {
            a: float(p)
            for a, p in zip(animals, probs.index_select(0, id_tensor).tolist())
        }
        shares = forced_choice_shares(cand_probs)
        shares_per_prompt.append(shares)
        if argmax_animal(shares) == target:
            hits += 1
    return shares_per_prompt, hits


@pytest.fixture(scope="module")
def tiny_model():
    """A tiny randomly-initialised GPT-NeoX plus a real Pythia tokenizer, on CPU."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    tokenizer.pad_token = tokenizer.eos_token
    torch.manual_seed(0)
    config = GPTNeoXConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    model = GPTNeoXForCausalLM(config).eval()
    return model, tokenizer


def test_batched_prob_and_rank_matches_per_prompt(tiny_model):
    model, tokenizer = tiny_model
    target_id = tokenizer.encode(" cat", add_special_tokens=False)[0]
    expected = _oracle_prob_and_rank(model, tokenizer, "cpu", PROMPTS, target_id)
    assert (
        target_prob_and_rank_per_prompt(model, tokenizer, "cpu", PROMPTS, target_id)
        == expected
    )  # default path is the old one, bit-for-bit
    for batch_size in (1, 2, 32):
        got = target_prob_and_rank_per_prompt(
            model, tokenizer, "cpu", PROMPTS, target_id, batch_size=batch_size
        )
        assert list(got) == list(expected)
        for prompt in PROMPTS:
            assert got[prompt][0] == pytest.approx(expected[prompt][0], rel=1e-5)
            assert got[prompt][1] == expected[prompt][1]


def test_batching_restores_padding_side(tiny_model):
    model, tokenizer = tiny_model
    tokenizer.padding_side = "right"
    target_prob_and_rank_per_prompt(model, tokenizer, "cpu", PROMPTS, 0, batch_size=2)
    assert tokenizer.padding_side == "right"


def test_batched_forced_choice_matches_per_prompt(tiny_model):
    model, tokenizer = tiny_model
    candidate_ids = {
        a: tokenizer.encode(f" {a}", add_special_tokens=False)[0]
        for a in ("cat", "dog", "bear")
    }
    expected_shares, expected_hits = _oracle_forced_choice(
        model, tokenizer, "cpu", PROMPTS, "cat", candidate_ids
    )
    score = forced_choice_over_prompts(
        model, tokenizer, "cpu", PROMPTS, "cat", candidate_ids, batch_size=2
    )
    assert score.n_prompts == len(PROMPTS)
    assert score.target_argmax_frac == pytest.approx(expected_hits / len(PROMPTS))
    for animal in candidate_ids:
        expected_mean = sum(s[animal] for s in expected_shares) / len(expected_shares)
        assert score.candidate_share_mean[animal] == pytest.approx(
            expected_mean, rel=1e-5
        )


@pytest.mark.parametrize("batch_size", [None, 2])
def test_empty_prompt_still_raises(tiny_model, batch_size):
    model, tokenizer = tiny_model
    with pytest.raises(ValueError, match="zero tokens"):
        target_prob_and_rank_per_prompt(
            model, tokenizer, "cpu", ("a", ""), 0, batch_size=batch_size
        )
