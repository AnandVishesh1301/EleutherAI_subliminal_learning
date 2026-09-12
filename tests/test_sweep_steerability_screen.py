"""Grid expansion for the steerability screen sweep (explicit param_tags + full-seed opt-in)."""

from __future__ import annotations

import types

import pytest

from scripts.sweeps.sweep_steerability_screen import expand_screen_cells
from subliminal_learning.teachers.model_specs import spec_label


def _base():
    """Minimal stand-in for the loaded template config (only .spec/.paths/.hub_org are read)."""
    return types.SimpleNamespace(spec=None, paths=None, hub_org="org")


def _sweep(**over) -> dict:
    sweep = {"animals": ["cat"], "n_animals": 1, "base_models": "original"}
    sweep.update(over)
    return sweep


TAGS = ["ep20_lr1e-4_r16a32", "ep10_lr5e-4_r8a8", "ep2_lr1e-3_r64a64"]


def test_param_tags_enumerate_exactly_those_cells_in_order():
    sweep = _sweep(param_tags=TAGS)  # no `grid:` at all
    animals, base_specs, epochs, lrs, loras, targets, cells = expand_screen_cells(
        _base(), sweep, steer_models="weight-seed1,weight-seed2"
    )
    assert len(targets) == 2 and len(base_specs) == 1
    assert len(cells) == 3 * len(targets)  # 3 tags per (animal, source, target)
    for tgt in targets:
        hp = [(e, lr, p) for _, _, e, lr, p, t in cells if t == tgt]
        assert hp == [(20, 1e-4, (16, 32)), (10, 5e-4, (8, 8)), (2, 1e-3, (64, 64))]
    assert (animals, epochs, lrs, loras) == (
        ["cat"], [20, 10, 2], [1e-4, 5e-4, 1e-3], [(16, 32), (8, 8), (64, 64)]
    )


def test_param_tags_reject_suffix_arms():
    with pytest.raises(ValueError, match="suffix arm"):
        expand_screen_cells(_base(), _sweep(param_tags=["ep20_lr1e-4_r16a32_stop1250"]))


def test_full_seed_targets_are_opt_in():
    sweep = _sweep(param_tags=["ep20_lr1e-4_r16a32"])
    with pytest.raises(ValueError):
        expand_screen_cells(_base(), sweep, steer_models="full-seed1")
    *_, targets, cells = expand_screen_cells(
        _base(), sweep, steer_models="full-seed1", allow_full_seeds=True
    )
    assert [spec_label(t) for t in targets] == ["full-seed1"]
    assert len(cells) == 1


def test_product_grid_unchanged():
    grid = {
        "n_epochs": [1, 2],
        "learning_rate": [1.0e-3, 1.0e-4],
        "lora_r": [8, 16],
        "lora_alpha_over_r": [1, 2],
        "lora_alpha_max": 16,
    }
    animals, base_specs, epochs, lrs, loras, targets, cells = expand_screen_cells(
        _base(), _sweep(grid=grid)
    )
    assert (epochs, lrs, loras, targets) == ([1, 2], [1e-3, 1e-4], [(8, 8), (8, 16), (16, 16)], [None])
    assert len(cells) == len(animals) * len(base_specs) * 2 * 2 * 3
    assert cells[0] == ("cat", base_specs[0], 1, 1e-3, (8, 8), None)
    assert cells[1] == ("cat", base_specs[0], 1, 1e-3, (8, 16), None)
