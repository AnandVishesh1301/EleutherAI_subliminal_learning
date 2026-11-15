"""Compatibility shim: the exclusion label API moved to `analysis.exclusions`.

Kept because the lora-forgetting repo imports ``subliminal_learning.analysis.labels`` (its own
``lora_forgetting.analysis.labels`` re-exports this and adds the collapse-only helpers). New
code imports `subliminal_learning.analysis.exclusions`.
"""
from __future__ import annotations

from subliminal_learning.analysis.exclusions import (  # noqa: F401  -- re-exported
    anchored_cut,
    anchored_cuts,
    anchored_dead_cut,
    label_dead_kl,
)

__all__ = ["anchored_cut", "anchored_cuts", "anchored_dead_cut", "label_dead_kl"]
