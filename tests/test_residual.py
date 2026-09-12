"""Raw residual-stream capture tests (tiny random models — no download, no network).

Pins the ``raw_blocks_v2`` basis contract of `subliminal_learning.models.residual` for
every base family the repo loads: every capture slot is RAW (slot 0 is the embedding
module's output, i.e. whatever enters block 0, and the final slot is the last block's own
output, which ``output_hidden_states=True`` only ever exposes AFTER the final layernorm),
there are ``n_blocks + 1`` slots, and the slots line up 1:1 with the blocks
`steering_hooks` attaches to — so the final block is genuinely steerable. Each family is
checked bare and under a PEFT LoRA wrapper.
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

HIDDEN = 16


def _neox():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    model = GPTNeoXForCausalLM(GPTNeoXConfig(
        vocab_size=64, hidden_size=HIDDEN, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=32))
    return model, (lambda m: m.gpt_neox.final_layer_norm), ["query_key_value"]


def _qwen2():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    model = Qwen2ForCausalLM(Qwen2Config(
        vocab_size=64, hidden_size=HIDDEN, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, intermediate_size=32, max_position_embeddings=32))
    return model, (lambda m: m.model.norm), ["q_proj", "v_proj"]


def _llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    model = LlamaForCausalLM(LlamaConfig(
        vocab_size=64, hidden_size=HIDDEN, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=2, intermediate_size=32, max_position_embeddings=32))
    return model, (lambda m: m.model.norm), ["q_proj", "v_proj"]


def _gemma3():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    model = Gemma3ForCausalLM(Gemma3TextConfig(
        vocab_size=64, hidden_size=HIDDEN, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, intermediate_size=32,
        max_position_embeddings=32, attn_implementation="eager"))
    return model, (lambda m: m.model.norm), ["q_proj", "v_proj"]


def _olmo():
    # hf_olmo (DataDecide). ``rope=True`` matches the released DataDecide configs; with
    # hf_olmo's alternative LEARNED position embeddings the ``wpe`` add happens OUTSIDE
    # the embedding module, so slot 0 would be the token embedding rather than the block-0
    # input (the contract slot 0 pins).
    from hf_olmo import OLMoForCausalLM
    from hf_olmo.configuration_olmo import OLMoConfig

    model = OLMoForCausalLM(OLMoConfig(
        d_model=HIDDEN, n_heads=2, n_layers=2, vocab_size=64, embedding_size=64,
        mlp_ratio=2, max_sequence_length=32, rope=True))
    return model, (lambda m: m.model.transformer.ln_f), ["att_proj", "ff_proj"]


FAMILIES = {
    "gpt_neox": _neox, "qwen2": _qwen2, "llama": _llama,
    "gemma3": _gemma3, "olmo": _olmo,
}


def _build(name: str):
    torch.manual_seed(0)
    model, final_norm, targets = FAMILIES[name]()
    model.eval()
    return model, final_norm, targets


def _wrap_lora(model, targets):
    import peft

    return peft.get_peft_model(
        model, peft.LoraConfig(r=2, lora_alpha=4, target_modules=targets)
    ).eval()


@pytest.fixture(scope="module")
def batch():
    torch.manual_seed(1)
    input_ids = torch.randint(0, 64, (2, 5))
    return input_ids, torch.ones_like(input_ids)


def _assert_raw_contract(model, final_norm, batch):
    """Every slot but the last equals ``hidden_states``; the last is pre-final-norm."""
    input_ids, attention_mask = batch
    raw = capture_raw_activations(model, input_ids, attention_mask)
    with torch.no_grad():
        out = model(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False,
        )
    hs = out.hidden_states
    assert len(raw) == len(hs) == num_raw_slots(model) == len(resolve_blocks(model)) + 1
    # Every slot but the last agrees with the hidden_states stack (those were already raw).
    for i in range(len(raw) - 1):
        assert torch.allclose(raw[i], hs[i], atol=1e-5), f"slot {i}"
    # The final slot is where the bases differ: output_hidden_states ends with the last
    # block's output AFTER the final layernorm; the capture keeps the raw output. Applying
    # the final layernorm to the raw slot must reproduce the hidden_states slot exactly.
    assert not torch.allclose(raw[-1], hs[-1], atol=1e-4)
    with torch.no_grad():
        normed = final_norm(model)(raw[-1])
    assert torch.allclose(normed, hs[-1], atol=1e-5)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_capture_is_raw_at_every_slot(family, batch):
    model, final_norm, _ = _build(family)
    _assert_raw_contract(model, final_norm, batch)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_capture_is_raw_under_peft(family, batch):
    # The resolver unwraps ``base_model.model``: a LoRA-wrapped model resolves to the same
    # embedding + block list, so the slot contract is untouched by the wrapper.
    model, final_norm, targets = _build(family)
    wrapped = _wrap_lora(model, targets)
    _assert_raw_contract(wrapped, lambda m: final_norm(m.base_model.model), batch)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_final_block_is_steerable(family, batch):
    # The raw_blocks_v2 point: v["raw"][1:] has one row per block, and hooking the FINAL
    # block (impossible under the old both-ends-stripped convention) genuinely moves the
    # model's output.
    model, _, _ = _build(family)
    input_ids, attention_mask = batch
    n_blocks = len(resolve_blocks(model))
    torch.manual_seed(2)
    v = torch.randn(n_blocks, HIDDEN)

    def _logits():
        return model(input_ids=input_ids, attention_mask=attention_mask,
                     use_cache=False).logits

    with torch.no_grad():
        clean = _logits()
        with steering_hooks(model, v, alpha=5.0, mode="add", layers=[n_blocks - 1]):
            steered = _logits()
        # Hooks are gone after the context: the clean forward is bit-reproducible.
        clean_again = _logits()
    assert not torch.allclose(clean, steered, atol=1e-4)
    assert torch.equal(clean, clean_again)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_final_block_project_mode_removes_the_component(family, batch):
    # `project` necessity mode: with alpha=1 the residual's component along v_hat is
    # removed at every position, and alpha=0 is the identity.
    model, _, _ = _build(family)
    input_ids, attention_mask = batch
    n_blocks = len(resolve_blocks(model))
    block = resolve_blocks(model)[n_blocks - 1]

    raw = capture_raw_activations(model, input_ids, attention_mask)
    direction = raw[-1][0, -1].detach()            # last block, last position, first row
    v = torch.zeros(n_blocks, HIDDEN)
    v[n_blocks - 1] = direction
    v_hat = direction / direction.norm()

    def _logits():
        return model(input_ids=input_ids, attention_mask=attention_mask,
                     use_cache=False).logits

    with torch.no_grad():
        clean = _logits()
    with steering_hooks(model, v, alpha=1.0, mode="project", layers=[n_blocks - 1]):
        projected = capture_raw_activations(model, input_ids, attention_mask)[-1]
    assert torch.allclose(projected[0, -1] @ v_hat, torch.zeros(()), atol=1e-4)
    assert len(block._forward_hooks) == 0        # hooks removed on context exit

    with torch.no_grad():
        with steering_hooks(model, v, alpha=0.0, mode="project", layers=[n_blocks - 1]):
            noop = _logits()
    assert torch.allclose(clean, noop, atol=1e-6)
    assert len(block._forward_hooks) == 0
