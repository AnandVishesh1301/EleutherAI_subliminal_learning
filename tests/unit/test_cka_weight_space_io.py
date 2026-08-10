"""Unit tests for `cka.weight_space_io` -- LoRA merging, coverage, barrier gating.

The failure this file exists to prevent is a plausible float: both measures live in
PARAMETER space, so a ``weight-seed`` cross has no neuron correspondence and a cosine
computed anyway would look like a small effect rather than an undefined one. The LoRA
scaling is checked too, because a wrong ``lora_alpha / r`` rescales every displacement
without changing any shape.

Fixtures are synthetic tensors under `tmp_path`: no network, no real checkpoint, no GPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from subliminal_learning.cka.weight_space_io import (
    coverage,
    full_ft_update,
    interpolate_state_dicts,
    interpolation_barrier,
    lora_induced_update,
    lora_scaling,
    merged_lora_delta,
)

_MODULE = "gpt_neox.layers.0.attention.query_key_value"
_SHARED_A = "pythia-160m-data-seed1@step16000"
_SHARED_B = "pythia-160m-data-seed2@step16000"
_BROKEN_B = "pythia-160m-weight-seed1@step16000"


# --- PEFT scaling ----------------------------------------------------------------------


def test_lora_scaling_is_alpha_over_r() -> None:
    assert lora_scaling({"r": 8, "lora_alpha": 16}) == pytest.approx(2.0)


def test_lora_scaling_uses_sqrt_r_for_rslora() -> None:
    scaling = lora_scaling({"r": 16, "lora_alpha": 32, "use_rslora": True})
    assert scaling == pytest.approx(32.0 / 4.0)


def test_lora_scaling_rejects_dora() -> None:
    with pytest.raises(ValueError, match="DoRA"):
        lora_scaling({"r": 8, "lora_alpha": 8, "use_dora": True})


def test_lora_scaling_rejects_zero_rank() -> None:
    with pytest.raises(ValueError, match="positive rank"):
        lora_scaling({"r": 0, "lora_alpha": 8})


@pytest.mark.parametrize("key", ["rank_pattern", "alpha_pattern"])
def test_lora_scaling_rejects_per_module_patterns(key: str) -> None:
    config = {"r": 8, "lora_alpha": 16, key: {".*query_key_value": 4}}
    with pytest.raises(ValueError, match=key):
        lora_scaling(config)


def test_lora_scaling_accepts_empty_patterns() -> None:
    config = {"r": 8, "lora_alpha": 16, "rank_pattern": {}, "alpha_pattern": {}}
    assert lora_scaling(config) == pytest.approx(2.0)


# --- merged displacement ---------------------------------------------------------------


def _write_adapter(
    tmp_path: Path, a: torch.Tensor, b: torch.Tensor, *, r: int, alpha: int
) -> Path:
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"r": r, "lora_alpha": alpha, "peft_type": "LORA"})
    )
    save_file(
        {
            f"base_model.model.{_MODULE}.lora_A.weight": a,
            f"base_model.model.{_MODULE}.lora_B.weight": b,
        },
        str(ckpt / "adapter_model.safetensors"),
    )
    return ckpt


def test_merged_lora_delta_is_scaled_b_at_a(tmp_path: Path) -> None:
    torch.manual_seed(0)
    a = torch.randn(4, 6)
    b = torch.randn(8, 4)
    ckpt = _write_adapter(tmp_path, a, b, r=4, alpha=8)

    delta = merged_lora_delta(ckpt)

    assert list(delta) == [f"{_MODULE}.weight"]
    assert torch.allclose(delta[f"{_MODULE}.weight"], 2.0 * (b @ a), atol=1e-6)


def test_merged_lora_delta_rejects_fan_in_fan_out(tmp_path: Path) -> None:
    ckpt = _write_adapter(tmp_path, torch.randn(4, 6), torch.randn(8, 4), r=4, alpha=8)
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"r": 4, "lora_alpha": 8, "fan_in_fan_out": True})
    )
    with pytest.raises(ValueError, match="fan_in_fan_out"):
        merged_lora_delta(ckpt)


def test_merged_lora_delta_requires_both_factors(tmp_path: Path) -> None:
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8}))
    save_file(
        {f"base_model.model.{_MODULE}.lora_A.weight": torch.randn(4, 6)},
        str(ckpt / "adapter_model.safetensors"),
    )
    with pytest.raises(ValueError, match="both factors"):
        merged_lora_delta(ckpt)


def test_merged_lora_delta_rejects_non_adapter_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="PEFT adapter dir"):
        merged_lora_delta(tmp_path)


# --- student update projections --------------------------------------------------------


def test_full_ft_update_is_negative_gradient() -> None:
    grads = {f"{_MODULE}.weight": torch.ones(2, 3), "other.weight": torch.ones(2, 2)}
    update = full_ft_update(grads, keys=[f"{_MODULE}.weight"])
    assert list(update) == [f"{_MODULE}.weight"]
    assert torch.allclose(update[f"{_MODULE}.weight"], -torch.ones(2, 3))


def test_lora_induced_update_matches_the_product_rule() -> None:
    torch.manual_seed(0)
    weight_a, weight_b = torch.randn(4, 6), torch.zeros(8, 4)
    grad_a, grad_b = torch.randn(4, 6), torch.randn(8, 4)
    prefix = f"base_model.model.{_MODULE}"
    update = lora_induced_update(
        {f"{prefix}.lora_A.default.weight": grad_a, f"{prefix}.lora_B.default.weight": grad_b},
        {f"{prefix}.lora_A.default.weight": weight_a, f"{prefix}.lora_B.default.weight": weight_b},
        scaling=2.0,
    )
    expected = -2.0 * (grad_b @ weight_a + weight_b @ grad_a)
    assert torch.allclose(update[f"{_MODULE}.weight"], expected, atol=1e-6)


# --- coverage --------------------------------------------------------------------------


def test_coverage_is_one_when_student_moves_along_the_teacher() -> None:
    delta = {f"{_MODULE}.weight": torch.randn(4, 4)}
    measured = coverage(
        {k: v.clone() for k, v in delta.items()},
        delta,
        student_label=_SHARED_A,
        teacher_label=_SHARED_B,
    )
    assert measured.coverage == pytest.approx(1.0, abs=1e-6)
    assert measured.skip_reason is None
    assert measured.n_tensors == 1


def test_coverage_is_minus_one_when_student_moves_against_the_teacher() -> None:
    delta = {f"{_MODULE}.weight": torch.randn(4, 4)}
    measured = coverage(
        {k: -v for k, v in delta.items()},
        delta,
        student_label=_SHARED_A,
        teacher_label=_SHARED_B,
    )
    assert measured.coverage == pytest.approx(-1.0, abs=1e-6)


def test_coverage_discounts_motion_outside_the_teachers_modules() -> None:
    delta = {f"{_MODULE}.weight": torch.ones(2, 2)}
    restricted_norm = 2.0  # ||ones(2, 2)||
    measured = coverage(
        {f"{_MODULE}.weight": torch.ones(2, 2)},
        delta,
        student_label=_SHARED_A,
        teacher_label=_SHARED_B,
        grad_norm_full=2 * restricted_norm,
    )
    assert measured.cosine_restricted == pytest.approx(1.0, abs=1e-6)
    assert measured.coverage == pytest.approx(0.5, abs=1e-6)


def test_coverage_is_none_when_the_basis_is_broken() -> None:
    delta = {f"{_MODULE}.weight": torch.randn(4, 4)}
    measured = coverage(
        {k: v.clone() for k, v in delta.items()},
        delta,
        student_label=_SHARED_A,
        teacher_label=_BROKEN_B,
    )
    assert measured.coverage is None
    assert measured.cosine_restricted is None
    assert measured.skip_reason == "basis_broken"


def test_coverage_rejects_zero_overlap() -> None:
    with pytest.raises(ValueError, match="at least one module"):
        coverage(
            {"a.weight": torch.ones(2, 2)},
            {"b.weight": torch.ones(2, 2)},
            student_label=_SHARED_A,
            teacher_label=_SHARED_B,
        )


# --- interpolation barrier -------------------------------------------------------------


class _TinyModel(torch.nn.Module):
    """One linear layer, so a state dict round-trips without a real checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2, bias=False)


def test_interpolate_state_dicts_is_the_convex_blend() -> None:
    a = {"w": torch.zeros(2, 2)}
    b = {"w": torch.ones(2, 2)}
    assert torch.allclose(interpolate_state_dicts(a, b, 0.25)["w"], 0.25 * torch.ones(2, 2))


def test_interpolate_state_dicts_rejects_key_mismatch() -> None:
    with pytest.raises(ValueError, match="key mismatch"):
        interpolate_state_dicts({"a": torch.zeros(1)}, {"b": torch.zeros(1)}, 0.5)


def test_interpolation_barrier_is_zero_on_a_linear_curve() -> None:
    model = _TinyModel()
    state_a = {"linear.weight": torch.zeros(2, 3)}
    state_b = {"linear.weight": torch.ones(2, 3)}
    # Loss linear in alpha => the curve equals its own chord => barrier 0.
    result = interpolation_barrier(
        model,
        state_a,
        state_b,
        eval_loss=lambda m: float(m.linear.weight.mean()),
        label_a=_SHARED_A,
        label_b=_SHARED_B,
    )
    assert result.barrier == pytest.approx(0.0, abs=1e-9)
    assert result.alphas == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert result.skip_reason is None


def test_interpolation_barrier_detects_a_bump() -> None:
    model = _TinyModel()
    state_a = {"linear.weight": torch.zeros(2, 3)}
    state_b = {"linear.weight": torch.ones(2, 3)}

    def bumped(m: torch.nn.Module) -> float:
        alpha = float(m.linear.weight.mean())
        return alpha + 4.0 * alpha * (1.0 - alpha)

    result = interpolation_barrier(
        model, state_a, state_b, eval_loss=bumped,
        label_a=_SHARED_A, label_b=_SHARED_B,
    )
    assert result.barrier == pytest.approx(1.0, abs=1e-6)


def test_interpolation_barrier_is_none_when_the_basis_is_broken() -> None:
    model = _TinyModel()
    state = {"linear.weight": torch.zeros(2, 3)}
    result = interpolation_barrier(
        model, state, state,
        eval_loss=lambda m: pytest.fail("eval must not run on a broken basis"),
        label_a=_SHARED_A, label_b=_BROKEN_B,
    )
    assert result.barrier is None
    assert result.skip_reason == "basis_broken"
    assert result.losses == ()
