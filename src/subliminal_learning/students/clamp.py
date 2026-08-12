"""Training-time dose clamp for LoRA student distillation.

A ``DoseClampCallback`` caps the LoRA update's pooled *relative* norm during
training. The dose of one adapter module is ``s * ||BA|| / ||W0||`` (``s`` the
PEFT scaling ``alpha/r``); the pooled dose is the mean over modules. When the
pool exceeds a ceiling the callback rescales every module's ``lora_B`` in place
by a single factor, which rescales the product ``BA`` exactly and so preserves
its direction (only the magnitude is capped).

The pure helpers (`dose_fro`, `dose_spec`, `apply_clamp`) are unit-testable
without peft/transformers.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import transformers


def dose_fro(B: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Exact ``||B@A||_F`` without materializing ``B@A`` (Gram identity).

    ``||BA||_F^2 = sum((B.T@B) * (A@A.T))``; B is (out, r), A is (r, in).
    """
    gram = (B.T @ B) * (A @ A.T)
    return torch.sqrt(gram.sum().clamp_min(0.0))


def dose_spec(B: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    """Exact ``sigma_1(B@A)`` via QR cores (no (out, in) product materialized)."""
    _qb, rb = torch.linalg.qr(B)
    _qa, ra = torch.linalg.qr(A.T)
    return torch.linalg.matrix_norm(rb @ ra.T, ord=2)


_DOSE = {"fro": dose_fro, "spec": dose_spec}


def apply_clamp(
    modules_info: list[dict], norm: str, ceiling: float
) -> tuple[float, float, bool]:
    """Cap the pooled relative dose; rescale each module's B in place if bound.

    Args:
        modules_info: One dict per LoRA module with keys ``B`` (out, r tensor),
            ``A`` (r, in tensor), ``s`` (float scaling), ``w0`` (float reference
            ``||W0||`` in the matching norm). The ``B`` tensors are mutated in place.
        norm: ``"fro"`` or ``"spec"`` (selects the dose + reference norm).
        ceiling: Pooled-dose cap.

    Returns:
        ``(D_before, D_after, bound)``; ``D_after == ceiling`` when bound, else
        ``D_before`` (weights untouched).
    """
    dose_fn = _DOSE[norm]
    per_module = [
        (m["s"] * dose_fn(m["B"], m["A"]) / m["w0"]).item() for m in modules_info
    ]
    d_before = sum(per_module) / len(per_module)
    if d_before <= ceiling:
        return d_before, d_before, False
    factor = ceiling / d_before
    for m in modules_info:
        m["B"].mul_(factor)
    return d_before, d_before * factor, True


class DoseClampCallback(transformers.TrainerCallback):
    """Cap the pooled relative LoRA dose after every optimizer step."""

    def __init__(self, norm: str, ceiling: float, log_path: Path | str):
        if norm not in _DOSE:
            raise ValueError(f"norm must be 'fro' or 'spec', got {norm!r}")
        if not (ceiling > 0):
            raise ValueError(f"ceiling must be positive, got {ceiling}")
        self.norm = norm
        self.ceiling = float(ceiling)
        self.log_path = Path(log_path)
        self._modules: list = []
        self._w0: list[float] = []
        self._scaling: list[float] = []
        self._buffer: list[dict] = []

    def on_train_begin(self, args, state, control, **kwargs):
        model = kwargs["model"]
        ref_norm = "fro" if self.norm == "fro" else 2
        self._modules, self._w0, self._scaling = [], [], []
        for module in model.modules():
            if not (hasattr(module, "lora_A") and hasattr(module, "lora_B")):
                continue
            if "default" not in module.lora_A or "default" not in module.lora_B:
                continue
            w0 = module.base_layer.weight.detach()
            with torch.no_grad():
                w0n = float(torch.linalg.matrix_norm(w0, ord=ref_norm))
            self._modules.append(module)
            self._w0.append(w0n)
            self._scaling.append(float(module.scaling["default"]))
        return control

    def _modules_info(self) -> list[dict]:
        return [
            {
                "B": m.lora_B["default"].weight,
                "A": m.lora_A["default"].weight,
                "s": s,
                "w0": w0,
            }
            for m, s, w0 in zip(self._modules, self._scaling, self._w0)
        ]

    def on_step_end(self, args, state, control, **kwargs):
        if not self._modules:
            return control
        with torch.no_grad():
            d_before, _d_after, bound = apply_clamp(
                self._modules_info(), self.norm, self.ceiling
            )
        self._buffer.append(
            {"step": state.global_step, "dose": float(d_before), "bound": bool(bound)}
        )
        if len(self._buffer) >= 200:
            self._flush()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._flush()
        return control

    def _flush(self):
        if not self._buffer:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            for rec in self._buffer:
                f.write(json.dumps(rec) + "\n")
        self._buffer = []
