"""Unit tests for the divergence early-stop callback (CPU-only, tiny model)."""

from __future__ import annotations

import torch
from transformers import TrainerControl, TrainerState

from subliminal_learning.students.divergence import DivergenceEarlyStop


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_B = torch.nn.Parameter(torch.zeros(4, 2))
        self.frozen = torch.nn.Parameter(torch.zeros(4), requires_grad=False)


def _fire(cb, logs, model, step=7):
    state = TrainerState()
    state.global_step = step
    control = TrainerControl()
    cb.on_log(None, state, control, logs=logs, model=model)
    return control


def test_finite_loss_does_not_stop():
    model = _TinyModel()
    model.lora_B.data.fill_(float("nan"))  # weights bad, but the loss is clean
    cb = DivergenceEarlyStop()
    control = _fire(cb, {"loss": 1.5, "grad_norm": 0.3}, model)
    assert not control.should_training_stop
    assert cb.stopped_at_step is None


def test_nan_loss_finite_weights_does_not_stop():
    cb = DivergenceEarlyStop()
    control = _fire(cb, {"loss": float("nan")}, _TinyModel())
    assert not control.should_training_stop
    assert cb.stopped_at_step is None


def test_nan_loss_nan_weights_stops():
    model = _TinyModel()
    model.lora_B.data.fill_(float("nan"))
    cb = DivergenceEarlyStop()
    control = _fire(cb, {"loss": float("nan")}, model, step=42)
    assert control.should_training_stop
    assert cb.stopped_at_step == 42


def test_inf_grad_norm_nan_weights_stops():
    model = _TinyModel()
    model.lora_B.data.fill_(float("inf"))
    cb = DivergenceEarlyStop()
    control = _fire(cb, {"loss": 2.0, "grad_norm": float("inf")}, model, step=3)
    assert control.should_training_stop
    assert cb.stopped_at_step == 3


def test_model_captured_at_train_begin():
    model = _TinyModel()
    model.lora_B.data.fill_(float("nan"))
    cb = DivergenceEarlyStop()
    cb.on_train_begin(None, TrainerState(), TrainerControl(), model=model)
    state = TrainerState()
    state.global_step = 5
    control = TrainerControl()
    cb.on_log(None, state, control, logs={"loss": float("nan")})
    assert control.should_training_stop
    assert cb.stopped_at_step == 5
