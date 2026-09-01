"""Weight-space surgery primitives shared by the direction/position/ladder scripts.

The measurement side of the collapse-mechanism program: read a trained LoRA adapter,
factor its per-module ΔW, build a variant of it, patch it into a live base model and read
the pp1 probe. ``analysis/`` is the notebook-support leaf, so these substantial
calculations live in their own src branch that ``scripts/``, ``pipelines/`` and notebooks
can all import; nothing here may import ``subliminal_learning.analysis``.

Submodules (import them directly; this package body stays import-cheap):

- `weightspace.adapters` — safetensors adapter loading and LoRA key stripping.
- `weightspace.lora` — ΔW materialisation and the exact (U, σ, V) of ``s·B·A``.
- `weightspace.patching` — target-module selection, W₀ capture, patch/restore, readouts.
- `weightspace.deltas` — the surgery variants and the synthetic-injection shapes.
"""

from __future__ import annotations
