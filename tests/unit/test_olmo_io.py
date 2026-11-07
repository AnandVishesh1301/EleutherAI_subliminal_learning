"""`hf_olmo` with the shared loaders, on a tiny real model (needs the `datadecide` group)."""

from __future__ import annotations

import pytest

pytest.importorskip("hf_olmo")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from subliminal_learning.models.olmo_io import register_hf_olmo  # noqa: E402


def _tiny_model():
    import hf_olmo

    register_hf_olmo()
    torch.manual_seed(0)
    config = hf_olmo.OLMoConfig(
        d_model=32,
        n_heads=2,
        n_layers=2,
        mlp_ratio=2,
        vocab_size=64,
        embedding_size=64,
        eos_token_id=0,
        pad_token_id=1,
        max_sequence_length=32,
        init_device="cpu",
        flash_attention=False,
        weight_tying=False,
    )
    return AutoModelForCausalLM.from_config(config).eval()


def test_registered_model_generates_with_a_cache():
    model = _tiny_model()
    ids = torch.tensor([[5, 6, 7, 8]])
    out = model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=1)
    assert out.shape == (1, 8)
    assert out[0, :4].tolist() == [5, 6, 7, 8]


def test_greedy_generation_matches_a_step_by_step_forward():
    model = _tiny_model()
    ids = torch.tensor([[5, 6, 7, 8]])
    expected = ids
    with torch.no_grad():
        for _ in range(3):
            step = model(input_ids=expected).logits[:, -1].argmax(-1, keepdim=True)
            expected = torch.cat([expected, step], dim=1)
    out = model.generate(ids, max_new_tokens=3, do_sample=False, pad_token_id=1)
    assert out.tolist() == expected.tolist()


def test_tokenizer_class_drops_token_type_ids():
    import hf_olmo

    register_hf_olmo()
    assert "token_type_ids" not in hf_olmo.OLMoTokenizerFast.model_input_names


def test_registering_twice_is_harmless():
    register_hf_olmo()
    register_hf_olmo()
