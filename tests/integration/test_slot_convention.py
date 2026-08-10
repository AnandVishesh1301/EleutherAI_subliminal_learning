"""The two 13-slot activation conventions, checked against a real pythia-160m.

`cka.extract_io.layer_activations` hooks ``embed_in`` and each decoder block, so its
slots are RAW post-block residuals. `steering.vectors.mean_activations` instead reads
``output_hidden_states=True``, whose last entry `GPTNeoXModel.forward` produces AFTER
``final_layer_norm`` (transformers 4.54.0). Both return ``n_layers + 1 == 13`` tensors of
``[B, T, 768]``, so mixing them raises nothing and changes nothing except the final slot
-- the slot the trait direction is usually strongest in.

This test is the empirical proof of that claim, and the reason
`cka.geometry_io.require_same_convention` and `trait_coordinates_at_slot` refuse to pair
across conventions. GPU-marked: it downloads/loads a real checkpoint.

Run manually: ``uv run pytest tests/integration/test_slot_convention.py -m gpu``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from subliminal_learning.cka.geometry_io import (
    Pooling,
    SlotConvention,
    capture_slots,
    collect_activation_stack,
    n_layers_of,
    require_same_convention,
    slot_kinds,
)
from subliminal_learning.config import load_config
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, model_ref

pytestmark = pytest.mark.gpu

_PROMPTS = (
    "The quick brown fox jumps over the lazy dog.",
    "My favorite animal is the",
    "In one word, name an animal:",
    "Pythia is a suite of language models trained on the Pile.",
)


@pytest.fixture(scope="module")
def paths():
    return load_config(
        Path("configs/paths/paths_local.yaml"), Path("configs/model/pythia.yaml")
    ).paths


@pytest.fixture(scope="module")
def loaded(paths):
    spec = PythiaModelSpec(size="160m", seed=0, variant="full")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(spec, paths.hf_cache_root, "EleutherAI")
    tokenizer.padding_side = "left"
    model = load_model(
        spec, paths.hf_cache_root, "EleutherAI", device, dtype=torch.float32
    )
    model.eval()
    return spec, model, tokenizer, device


def _batch(tokenizer, device: str):
    enc = tokenizer(
        list(_PROMPTS), return_tensors="pt", padding=True, add_special_tokens=False
    ).to(device)
    return enc.input_ids, enc.attention_mask


def test_both_conventions_return_thirteen_slots(loaded) -> None:
    _, model, tokenizer, device = loaded
    input_ids, mask = _batch(tokenizer, device)
    assert n_layers_of(model) == 12
    for convention in SlotConvention:
        slots = capture_slots(model, input_ids, mask, convention)
        assert len(slots) == 13
        assert len(slot_kinds(convention, 12)) == 13
        assert all(s.shape == (len(_PROMPTS), input_ids.shape[1], 768) for s in slots)


def test_conventions_agree_below_the_final_slot(loaded) -> None:
    _, model, tokenizer, device = loaded
    input_ids, mask = _batch(tokenizer, device)
    post = capture_slots(model, input_ids, mask, SlotConvention.POST_BLOCK)
    hidden = capture_slots(model, input_ids, mask, SlotConvention.HIDDEN_STATES)
    for slot in range(12):
        assert torch.allclose(post[slot], hidden[slot], atol=1e-5), (
            f"slot {slot} must be identical across conventions"
        )


def test_final_slot_differs_because_of_the_final_layernorm(loaded) -> None:
    _, model, tokenizer, device = loaded
    input_ids, mask = _batch(tokenizer, device)
    post = capture_slots(model, input_ids, mask, SlotConvention.POST_BLOCK)
    hidden = capture_slots(model, input_ids, mask, SlotConvention.HIDDEN_STATES)

    assert not torch.allclose(post[12], hidden[12], atol=1e-5)
    normed = model.gpt_neox.final_layer_norm(post[12])
    assert torch.allclose(normed, hidden[12], atol=1e-4)
    assert slot_kinds(SlotConvention.POST_BLOCK, 12)[12] != slot_kinds(
        SlotConvention.HIDDEN_STATES, 12
    )[12]


def test_pairing_across_conventions_raises(loaded) -> None:
    spec, model, tokenizer, device = loaded
    ref = model_ref(spec, "EleutherAI")
    stacks = {
        convention: collect_activation_stack(
            model,
            tokenizer,
            _PROMPTS,
            device,
            convention=convention,
            pooling=Pooling.LAST,
            batch_size=4,
            model_ref=ref,
            prompt_source="unit-prompts",
        )
        for convention in SlotConvention
    }
    assert require_same_convention(
        stacks[SlotConvention.POST_BLOCK], stacks[SlotConvention.POST_BLOCK]
    ) is SlotConvention.POST_BLOCK
    with pytest.raises(ValueError, match="two slot conventions"):
        require_same_convention(
            stacks[SlotConvention.POST_BLOCK], stacks[SlotConvention.HIDDEN_STATES]
        )
