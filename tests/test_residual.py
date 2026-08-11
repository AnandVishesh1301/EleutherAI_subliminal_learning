"""Raw residual-stream capture tests (tiny random GPT-NeoX — no download, no network).

Pins the ``raw_blocks_v2`` basis contract of `subliminal_learning.models.residual`:
every capture slot is RAW (the final slot is the last block's own output, which
``output_hidden_states=True`` only ever exposes AFTER the final layernorm), and the
capture slots line up 1:1 with the blocks `steering_hooks` attaches to — so the final
block is genuinely steerable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.models.residual import (
    capture_raw_activations,
    num_raw_slots,
    resolve_blocks,
)
from subliminal_learning.steering.hooks import steering_hooks


@pytest.fixture(scope="module")
def tiny_neox():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(0)
    model = GPTNeoXForCausalLM(GPTNeoXConfig(
        vocab_size=64, hidden_size=128, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=256, max_position_embeddings=32))
    model.eval()
    return model


@pytest.fixture(scope="module")
def batch():
    torch.manual_seed(1)
    input_ids = torch.randint(0, 64, (2, 5))
    return input_ids, torch.ones_like(input_ids)


def test_capture_is_raw_at_every_slot(tiny_neox, batch):
    input_ids, attention_mask = batch
    raw = capture_raw_activations(tiny_neox, input_ids, attention_mask)
    with torch.no_grad():
        out = tiny_neox(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False,
        )
    hs = out.hidden_states
    assert len(raw) == len(hs) == num_raw_slots(tiny_neox)
    # Every slot but the last agrees with the hidden_states stack (those were already raw).
    for i in range(len(raw) - 1):
        assert torch.allclose(raw[i], hs[i], atol=1e-6), f"slot {i}"
    # The final slot is where the bases differ: output_hidden_states ends with the last
    # block's output AFTER the final layernorm; the capture keeps the raw output. Applying
    # the final layernorm to the raw slot must reproduce the hidden_states slot exactly.
    assert not torch.allclose(raw[-1], hs[-1], atol=1e-4)
    with torch.no_grad():
        normed = tiny_neox.gpt_neox.final_layer_norm(raw[-1])
    assert torch.allclose(normed, hs[-1], atol=1e-6)


def test_final_block_is_steerable(tiny_neox, batch):
    # The raw_blocks_v2 point: v["raw"][1:] has one row per block, and hooking the FINAL
    # block (impossible under the old both-ends-stripped convention) genuinely moves the
    # model's output.
    input_ids, attention_mask = batch
    n_blocks = len(resolve_blocks(tiny_neox))
    v = torch.randn(n_blocks, tiny_neox.config.hidden_size)
    with torch.no_grad():
        clean = tiny_neox(input_ids=input_ids, attention_mask=attention_mask).logits
        with steering_hooks(tiny_neox, v, alpha=5.0, mode="add", layers=[n_blocks - 1]):
            steered = tiny_neox(input_ids=input_ids, attention_mask=attention_mask).logits
        # Hooks are gone after the context: the clean forward is bit-reproducible.
        clean_again = tiny_neox(input_ids=input_ids, attention_mask=attention_mask).logits
    assert not torch.allclose(clean, steered, atol=1e-4)
    assert torch.equal(clean, clean_again)
