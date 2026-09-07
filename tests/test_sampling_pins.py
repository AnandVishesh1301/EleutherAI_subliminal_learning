"""Sampling knobs are pinned in code, not inherited from a model's generation_config.

Each family bundles a different generation_config.json: Gemma-3 ships `top_k: 64` and
`eos_token_id: [1, 106]`, Pythia ships neither (so HF's default top_k=50 applied to every
pool ever generated). Both sampling entry points must therefore pin `top_k` and pass the
explicit union of stop ids, and `generate_completions` must call a stop on ANY of them.
"""

from __future__ import annotations

import types

import torch

from subliminal_learning.evaluation import preference_sampling_io
from subliminal_learning.numbers.generation_io import (
    TOP_K,
    effective_eos_ids,
    generate_completions,
)


class _FakeTokenizer:
    """Minimal stand-in: only the attributes the sampling functions touch."""

    padding_side = "right"
    pad_token_id = 7

    def __init__(self, eos_token_id: int | None = 1):
        self.eos_token_id = eos_token_id

    def __call__(self, text, **kwargs):
        seqs = [text] if isinstance(text, str) else list(text)
        ids = torch.zeros((len(seqs), 3), dtype=torch.long)
        return types.SimpleNamespace(
            data={"input_ids": ids, "attention_mask": torch.ones_like(ids)},
            to=lambda device: {"input_ids": ids, "attention_mask": torch.ones_like(ids)},
        )

    def decode(self, row, skip_special_tokens=False):
        return " ".join(str(int(i)) for i in row)


class _FakeModel:
    """Records the last `generate` kwargs and returns `rows` as the generated region."""

    def __init__(self, gen_cfg_eos, rows: torch.Tensor | None = None):
        self.generation_config = types.SimpleNamespace(eos_token_id=gen_cfg_eos)
        self.rows = rows
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["input_ids"]
        rows = self.rows if self.rows is not None else torch.full((prompt.shape[0], 4), 9)
        return torch.cat([prompt, rows], dim=1)


def _tok_and_prompts():
    return _FakeTokenizer(), ["a", "b"]


# (a) effective_eos_ids


def test_effective_eos_ids_unions_list_config():
    tok = _FakeTokenizer(eos_token_id=1)
    assert effective_eos_ids(_FakeModel([1, 106]), tok) == [1, 106]


def test_effective_eos_ids_unions_int_config_and_dedups():
    tok = _FakeTokenizer(eos_token_id=1)
    assert effective_eos_ids(_FakeModel(1), tok) == [1]
    assert effective_eos_ids(_FakeModel(2), tok) == [1, 2]


def test_effective_eos_ids_with_no_config_value_is_tokenizer_only():
    """The Pythia case: no generation_config entry -> exactly the tokenizer's id."""
    tok = _FakeTokenizer(eos_token_id=0)
    assert effective_eos_ids(_FakeModel(None), tok) == [0]


def test_effective_eos_ids_unwraps_a_peft_style_wrapper():
    base = _FakeModel([1, 106])

    class _BaseModelAttr:  # like PeftModel: no generation_config of its own
        base_model = base

    class _GetBaseModel:  # the other PEFT accessor
        def get_base_model(self):
            return base

    tok = _FakeTokenizer(1)
    assert effective_eos_ids(_BaseModelAttr(), tok) == [1, 106]
    assert effective_eos_ids(_GetBaseModel(), tok) == [1, 106]


def test_effective_eos_ids_handles_missing_tokenizer_id():
    assert effective_eos_ids(_FakeModel([106]), _FakeTokenizer(eos_token_id=None)) == [106]


# (b) both generate calls pin top_k and pass the stop list


def test_generate_completions_pins_top_k_and_eos_list():
    tok, prompts = _tok_and_prompts()
    model = _FakeModel([1, 106])
    generate_completions(model, tok, "cpu", prompts, batch_size=2, seed_base=0)
    kw = model.calls[0]
    assert kw["top_k"] == TOP_K == 50
    assert kw["eos_token_id"] == [1, 106]


def test_sample_completions_pins_top_k_and_eos_list():
    tok = _FakeTokenizer()
    model = _FakeModel([1, 106])
    preference_sampling_io.sample_completions(
        model, tok, "cpu", "hi", n_samples=2, batch_size=2
    )
    kw = model.calls[0]
    assert kw["top_k"] == TOP_K == 50
    assert kw["eos_token_id"] == [1, 106]


# (c) ended_with_eos honours the whole stop set


def test_ended_with_eos_true_for_non_tokenizer_stop_id():
    """Gemma stopping on <end_of_turn>=106 is a clean stop, not a token-cap hit."""
    tok = _FakeTokenizer(eos_token_id=1)
    rows = torch.tensor([[5, 6, 106, 106], [5, 6, 7, 8]])
    model = _FakeModel([1, 106], rows=rows)
    out = generate_completions(model, tok, "cpu", ["a", "b"], batch_size=2, seed_base=0)
    assert [c.ended_with_eos for c in out] == [True, False]


def test_ended_with_eos_true_for_tokenizer_stop_id():
    tok = _FakeTokenizer(eos_token_id=1)
    rows = torch.tensor([[5, 1, 1, 1], [5, 6, 7, 8]])
    model = _FakeModel(None, rows=rows)
    out = generate_completions(model, tok, "cpu", ["a", "b"], batch_size=2, seed_base=0)
    assert [c.ended_with_eos for c in out] == [True, False]
