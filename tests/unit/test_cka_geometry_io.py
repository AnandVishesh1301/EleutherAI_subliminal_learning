"""Unit tests for the pooled activation read and its stamps (`cka.geometry_io`).

Two failure classes are locked here because both produce a plausible number rather
than an error:

- **Wrong token.** `Pooling.LAST` must land on the last REAL token. Under the left
  padding `collect_activation_stack` requires, ``attention_mask.sum(-1) - 1`` points
  into the pad run for every row shorter than the batch maximum, so `d_trait` and
  every ``last``-pooled similarity row would be read off padding.
- **Wrong convention.** The two 13-slot stacks agree on slots 0..n-2 and differ only
  at the final slot, so a mixed pair is silent everywhere except the slot the trait
  is strongest in.

Tensors are constructed directly (no model, no GPU): the slot value at each position
is its own index, so a misread position is visible in the pooled value.
"""

from __future__ import annotations

import pytest
import torch

from subliminal_learning.cka.geometry_io import (
    MODEL_ROLES,
    ActivationStack,
    Pooling,
    SlotConvention,
    activation_cache_path,
    last_real_index,
    pool_slots,
    require_same_convention,
    should_cache,
    slot_kinds,
    trait_vector_path,
)

_LEFT_PADDED = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
_RIGHT_PADDED = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])


def _positional_slot(batch: int = 2, width: int = 4, hidden: int = 3) -> torch.Tensor:
    """``[B, T, H]`` whose every feature at position ``t`` equals ``t``."""
    return (
        torch.arange(width, dtype=torch.float32)
        .reshape(1, width, 1)
        .repeat(batch, 1, hidden)
    )


# --- token position --------------------------------------------------------------------


def test_last_real_index_left_padding_is_the_final_column() -> None:
    assert last_real_index(_LEFT_PADDED).tolist() == [3, 3]


def test_last_real_index_right_padding_is_the_row_length() -> None:
    assert last_real_index(_RIGHT_PADDED).tolist() == [1, 3]


def test_last_real_index_all_padding_row_raises() -> None:
    with pytest.raises(ValueError, match="all-padding row"):
        last_real_index(torch.tensor([[0, 0, 0, 0]]))


def test_pool_last_reads_the_final_real_token_of_a_left_padded_row() -> None:
    pooled = pool_slots([_positional_slot()], _LEFT_PADDED, Pooling.LAST)[0]
    assert pooled.shape == (2, 3)
    assert pooled[:, 0].tolist() == [3.0, 3.0]


def test_pool_mean_ignores_padding_positions() -> None:
    pooled = pool_slots([_positional_slot()], _LEFT_PADDED, Pooling.MEAN)[0]
    assert pooled[0, 0] == pytest.approx(2.5)
    assert pooled[1, 0] == pytest.approx(1.5)


# --- slot convention -------------------------------------------------------------------


def test_slot_kinds_differ_only_at_the_final_slot() -> None:
    post = slot_kinds(SlotConvention.POST_BLOCK, 12)
    hidden = slot_kinds(SlotConvention.HIDDEN_STATES, 12)
    assert len(post) == len(hidden) == 13
    assert post[:12] == hidden[:12]
    assert post[12] == "block11_out"
    assert hidden[12] == "final_ln_block11_out"


def _stack(convention: SlotConvention, ref: str = "EleutherAI/pythia-160m@step0"):
    return ActivationStack(
        rows=torch.zeros(13, 8, 3),
        slot_convention=convention,
        pooling=Pooling.LAST,
        kinds=slot_kinds(convention, 12),
        model_ref=ref,
        prompt_source="unit",
    )


def test_require_same_convention_rejects_a_mixed_pair() -> None:
    with pytest.raises(ValueError, match="two slot conventions"):
        require_same_convention(
            _stack(SlotConvention.POST_BLOCK), _stack(SlotConvention.HIDDEN_STATES)
        )


def test_require_same_convention_returns_the_shared_one() -> None:
    assert (
        require_same_convention(
            _stack(SlotConvention.POST_BLOCK), _stack(SlotConvention.POST_BLOCK)
        )
        is SlotConvention.POST_BLOCK
    )


# --- provenance in paths ---------------------------------------------------------------


def test_cache_path_separates_revisions_and_conventions() -> None:
    def path(ref: str, convention: SlotConvention):
        return activation_cache_path(
            "/cache",
            model_ref=ref,
            prompt_source="pile",
            convention=convention,
            pooling=Pooling.LAST,
            n_examples=10240,
        )

    at_step = path("EleutherAI/pythia-160m@step1000", SlotConvention.POST_BLOCK)
    at_main = path("EleutherAI/pythia-160m", SlotConvention.POST_BLOCK)
    other = path("EleutherAI/pythia-160m@step1000", SlotConvention.HIDDEN_STATES)
    assert "step1000" in str(at_step)
    assert at_step != at_main
    assert at_step != other


def test_trait_vector_path_carries_the_convention() -> None:
    def path(convention: SlotConvention):
        return trait_vector_path(
            "/similarity",
            animal="cat",
            model_label="data-seed1@step16000",
            ftseed=0,
            pooling=Pooling.LAST,
            convention=convention,
        )

    assert path(SlotConvention.POST_BLOCK) != path(SlotConvention.HIDDEN_STATES)
    assert path(SlotConvention.POST_BLOCK).name.endswith("_last_post_block.pt")


def test_should_cache_rejects_an_unknown_role() -> None:
    assert should_cache("random", ("random",))
    assert not should_cache("base", ("random",))
    with pytest.raises(ValueError, match="unknown model role"):
        should_cache("teacher", MODEL_ROLES)
