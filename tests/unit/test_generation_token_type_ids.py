"""Sampling never forwards tokenizer ``token_type_ids`` into ``generate``."""

from __future__ import annotations

import torch

from subliminal_learning.evaluation import preference_sampling_io as sampling


class _Enc(dict):
    def to(self, device):
        return self


class _Tok:
    """Stands in for `PreTrainedTokenizerFast`, which returns token_type_ids by default."""

    eos_token_id = 0
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, text, **kw):
        self.calls.append(kw)
        enc = _Enc(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3))
        if kw.get("return_token_type_ids", True):
            enc["token_type_ids"] = torch.zeros(1, 3, dtype=torch.long)
        return enc

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["x"] * ids.shape[0]

    def decode(self, ids, skip_special_tokens=True):
        return "x"


class _Model:
    device = "cpu"

    def __init__(self) -> None:
        self.kwargs: list[dict] = []

    def generate(self, **kw):
        assert "token_type_ids" not in kw
        self.kwargs.append(kw)
        n = kw.get("num_return_sequences", 1)
        return torch.tensor([[1, 2, 3, 4]] * n)


def test_sample_completions_drops_token_type_ids() -> None:
    tok, model = _Tok(), _Model()
    out = sampling.sample_completions(
        model, tok, "cpu", "the", n_samples=2, seed=0, batch_size=2
    )
    assert len(out) == 2
    assert all(c.get("return_token_type_ids") is False for c in tok.calls)
    assert model.kwargs and "token_type_ids" not in model.kwargs[0]
