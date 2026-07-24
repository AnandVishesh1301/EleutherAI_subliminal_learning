"""Shared PEFT LoRA-config builder for the training pipelines.

The SDF-teacher and GKD-student trainers previously each built an identical
``peft.LoraConfig`` (rank/alpha/dropout/target-modules, ``bias="none"``,
``task_type="CAUSAL_LM"``). This is the one place that build lives now (C3), so a change
to the shared LoRA wrapping applies to both fine-tune modes at once.
"""

from __future__ import annotations

from peft import LoraConfig

from subliminal_learning.config import LoRAConfig as LoRACfg


def build_lora_config(lora: LoRACfg | None) -> LoraConfig | None:
    """PEFT ``LoraConfig`` for a causal-LM LoRA run, or ``None`` for full FT.

    Args:
        lora: Project LoRA settings, or ``None`` to full fine-tune (no adapter).

    Returns:
        A ``peft.LoraConfig`` (``bias="none"``, ``task_type="CAUSAL_LM"``, resolved
        target modules), or ``None`` when `lora` is ``None``.
    """
    if lora is None:
        return None
    return LoraConfig(
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        target_modules=lora.resolve_target_modules(),
        bias="none",
        task_type="CAUSAL_LM",
    )
