"""Early stop for a student run whose adapter has already gone non-finite.

Once a step's loss is NaN the LoRA is permanently dead (NaN grads -> NaN clip
coefficient -> NaN Adam moments), yet the run would burn its remaining steps.
``DivergenceEarlyStop`` watches the *logged* loss/grad_norm (free: those values
are already on the host) and, only when one of them is non-finite, pays a single
on-device check of the trainable parameters. It stops training only once the
weights themselves are non-finite -- exactly the condition every downstream
consumer already uses to recognise divergence (``run_io.checkpoint_nonfinite_count``,
the workers' ``train_diverged.json`` marker), so the saved adapter and the labels
derived from it are unchanged; only the wall clock is shorter.

A non-finite loss over still-finite weights does nothing: the next log will catch
it once the poison has reached the parameters.
"""

from __future__ import annotations

import math

import torch
from transformers import TrainerCallback


def _is_nonfinite(value: object) -> bool:
    """True iff `value` is a float-like that is NaN or +/-inf."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return math.isnan(value) or math.isinf(value)


class DivergenceEarlyStop(TrainerCallback):
    """Stop training once the logged loss is non-finite AND the weights are too.

    Attributes:
        stopped_at_step: ``state.global_step`` at which training was stopped, or
            None if the callback never fired.
    """

    def __init__(self) -> None:
        self.stopped_at_step: int | None = None
        self._model = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # HF passes model= to every callback event (CallbackHandler.call_event), but
        # keep a handle so the check works if a caller drives the callback directly.
        if model is not None:
            self._model = model

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if self.stopped_at_step is not None or not logs:
            return control
        # Host-side floats already logged: no sync on the finite path.
        if not any(_is_nonfinite(logs.get(k)) for k in ("loss", "grad_norm")):
            return control
        model = model if model is not None else self._model
        if model is None:
            return control
        trainable = [p for p in model.parameters() if p.requires_grad]
        if not trainable:
            return control
        # One fused reduction, one host transfer.
        finite = torch.stack([p.isfinite().all() for p in trainable]).all().item()
        if finite:
            return control  # loss poisoned but weights still clean; catch it next log
        self.stopped_at_step = state.global_step
        control.should_training_stop = True
        return control
