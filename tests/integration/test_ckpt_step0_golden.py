"""The step0 golden anchor: the ladder's own correctness check (plan §6.10).

PolyPythia's ``data-seed{N}`` models re-randomise the DATA ORDER on a FIXED weight init,
so at ``revision="step0"`` -- before any optimizer step -- ``original`` and
``data-seed1`` are byte-identical weights. Every similarity measure in the campaign must
therefore return its saturated value on that pair, and must NOT on ``weight-seed1``,
which re-randomises the init itself.

A pipeline that silently mismatched models, revisions, prompts or slot conventions would
still produce a smooth, plausible ladder curve; this test is the one place that failure
is forced to show. GPU-marked: it loads three real checkpoints.

Run manually: ``uv run pytest tests/integration/test_ckpt_step0_golden.py -m gpu``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from subliminal_learning.cka.geometry_io import (
    Pooling,
    SlotConvention,
    build_trait_vector,
    collect_activation_stack,
    mean_slot_activations,
    trait_coordinates_at_slot,
)
from subliminal_learning.cka.metrics import (
    minibatch_cka,
    procrustes_distance,
    ssa_reduce,
    subspace_angles_topk,
    trait_coordinate_similarity,
)
from subliminal_learning.cka.pairing import BasisValidity, basis_validity
from subliminal_learning.config import load_config
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, model_ref

pytestmark = pytest.mark.gpu

_REVISION = "step0"
_SLOT = 6
_N_PROMPTS = 128
_BATCH = 32
_ORG = "EleutherAI"

_SPECS = {
    "original": PythiaModelSpec(size="160m", seed=0, variant="full", revision=_REVISION),
    "data-seed1": PythiaModelSpec(
        size="160m", seed=1, variant="data", revision=_REVISION
    ),
    "weight-seed1": PythiaModelSpec(
        size="160m", seed=1, variant="weight", revision=_REVISION
    ),
}


def _prompts() -> tuple[str, ...]:
    """Deterministic short prompts; content is irrelevant, only that all models see it."""
    stems = (
        "The capital of France is",
        "My favorite animal is the",
        "In one word, name an animal:",
        "Once upon a time there was a",
    )
    return tuple(f"{stems[i % len(stems)]} ({i})" for i in range(_N_PROMPTS))


@pytest.fixture(scope="module")
def paths():
    return load_config(
        Path("configs/paths/paths_local.yaml"), Path("configs/model/pythia.yaml")
    ).paths


@pytest.fixture(scope="module")
def tokenizer(paths):
    tok = load_tokenizer(_SPECS["original"], paths.hf_cache_root, _ORG)
    tok.padding_side = "left"
    return tok


@pytest.fixture(scope="module")
def stacks(paths, tokenizer):
    """One `ActivationStack` per init, loaded and released one model at a time."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    prompts = _prompts()
    out = {}
    for label, spec in _SPECS.items():
        model = load_model(
            spec, paths.hf_cache_root, _ORG, device, dtype=torch.float32
        )
        model.eval()
        out[label] = collect_activation_stack(
            model,
            tokenizer,
            prompts,
            device,
            convention=SlotConvention.POST_BLOCK,
            pooling=Pooling.LAST,
            batch_size=_BATCH,
            model_ref=model_ref(spec, _ORG),
            prompt_source="step0-golden",
        )
        if label == "original":
            out["_trait"] = build_trait_vector(
                mean_slot_activations(
                    model, tokenizer, prompts[:16], device,
                    convention=SlotConvention.POST_BLOCK,
                    pooling=Pooling.LAST, batch_size=_BATCH,
                ),
                mean_slot_activations(
                    model, tokenizer, prompts[16:32], device,
                    convention=SlotConvention.POST_BLOCK,
                    pooling=Pooling.LAST, batch_size=_BATCH,
                ),
                convention=SlotConvention.POST_BLOCK,
                pooling=Pooling.LAST,
                model_label=f"original@{_REVISION}",
                animal="cat",
                ftseed=0,
            )
        del model
        torch.cuda.empty_cache()
    return out


def _cka(a, b) -> float:
    value, _ = minibatch_cka(
        a, b, batch_size=_BATCH, n_examples=_N_PROMPTS, n_repeats=3, seed=0
    )
    return value


def test_step0_same_weight_init_is_stamped_shared() -> None:
    assert (
        basis_validity(f"original@{_REVISION}", f"data-seed1@{_REVISION}")
        is BasisValidity.BASIS_SHARED
    )
    assert (
        basis_validity(f"original@{_REVISION}", f"weight-seed1@{_REVISION}")
        is BasisValidity.BASIS_BROKEN
    )


def test_step0_data_seed_saturates_every_similarity_measure(stacks) -> None:
    x = stacks["original"].slot(_SLOT)
    y = stacks["data-seed1"].slot(_SLOT)

    assert _cka(x, y) == pytest.approx(1.0, abs=1e-4)
    assert procrustes_distance(x, y) == pytest.approx(0.0, abs=1e-4)
    assert ssa_reduce(subspace_angles_topk(x, y, k=5)) == pytest.approx(0.0, abs=1e-2)


def test_step0_data_seed_trait_coordinates_are_identical(stacks, tokenizer) -> None:
    trait = stacks["_trait"]
    cx = trait_coordinates_at_slot(stacks["original"], _SLOT, trait, tokenizer)
    cy = trait_coordinates_at_slot(stacks["data-seed1"], _SLOT, trait, tokenizer)
    assert trait_coordinate_similarity(cx, cy) == pytest.approx(1.0, abs=1e-6)


def test_step0_weight_seed_does_not_saturate(stacks) -> None:
    x = stacks["original"].slot(_SLOT)
    z = stacks["weight-seed1"].slot(_SLOT)

    assert _cka(x, z) < 0.99
    assert procrustes_distance(x, z) > 1e-3
    assert ssa_reduce(subspace_angles_topk(x, z, k=5)) > 1.0
