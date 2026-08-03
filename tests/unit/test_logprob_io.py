from __future__ import annotations

import pytest

from subliminal_learning.evaluation.logprob_io import (
    ContinuationLogProb,
    continuation_logprob,
)


def test_continuation_logprob_total_is_sum_of_token_logprobs() -> None:
    result = ContinuationLogProb(
        prompt="My favorite animal is the",
        continuation=" owl",
        continuation_token_ids=[18454, 77],
        continuation_token_strs=["Ġow", "l"],
        token_logprobs=[-3.0, -1.0],
    )
    assert result.total_logprob == pytest.approx(-4.0)
    assert result.mean_logprob == pytest.approx(-2.0)


@pytest.mark.gpu
def test_continuation_logprob_empty_prompt_raises() -> None:
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    model.eval()

    with pytest.raises(ValueError, match="Empty prompt"):
        continuation_logprob("", " owl", model, tokenizer, device="cpu")


@pytest.mark.gpu
def test_continuation_logprob_single_token_matches_manual_log_softmax() -> None:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-70m")
    model.eval()

    prompt = "My favorite animal is the"
    result = continuation_logprob(prompt, " cat", model, tokenizer, device="cpu")

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([prompt_ids])).logits
    expected = F.log_softmax(logits[0, -1, :], dim=-1)[result.continuation_token_ids[0]]

    assert result.continuation_token_strs == ["Ġcat"]
    assert result.total_logprob == pytest.approx(expected.item(), abs=1e-5)
