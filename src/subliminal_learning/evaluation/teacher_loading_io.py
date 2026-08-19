"""Load a fine-tuned teacher checkpoint, whether a LoRA adapter or a full fine-tune.

Two on-disk checkpoint shapes exist in this repo:

- **LoRA adapter** (`adapter_config.json` + `adapter_model.safetensors`) -- the template
  teachers and the SDF LoRA teachers. Loaded as `PeftModel.from_pretrained(base, ckpt)`.
- **Full fine-tune** (`config.json` + `model.safetensors`) -- the SDF full-FT teachers
  (`--finetune-mode full`, no adapter). Loaded directly as `GPTNeoXForCausalLM`.

`detect_finetune_mode` distinguishes them by the presence of `adapter_config.json`;
`load_finetuned_teacher` loads either uniformly so a single eval pipeline handles both.
The adapter branch is byte-identical to the previous inline
`PeftModel.from_pretrained(load_model(...), ckpt)` path, so LoRA evaluation is unchanged.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from peft import PeftModel
from transformers import GPTNeoXForCausalLM, PreTrainedModel

from subliminal_learning.models.datadecide import DataDecideSpec

from subliminal_learning.models.loading_io import load_model
from subliminal_learning.models.registry import PythiaModelSpec


def detect_finetune_mode(checkpoint_dir: Path) -> str:
    """Return ``"lora"`` if `checkpoint_dir` holds a PEFT adapter, else ``"full"``.

    Detection is purely the presence of ``adapter_config.json`` (written by
    `PeftModel.save_pretrained`); a full fine-tune saves ``config.json`` +
    ``model.safetensors`` instead.
    """
    return "lora" if (Path(checkpoint_dir) / "adapter_config.json").exists() else "full"


def load_finetuned_teacher(
    spec: PythiaModelSpec,
    checkpoint_dir: Path,
    hf_cache_root: Path,
    hub_org: str,
    device: str,
    dtype: torch.dtype = torch.float32,
    expected_mode: str | None = None,
) -> PreTrainedModel:
    """Load a teacher checkpoint (LoRA adapter or full fine-tune) in eval mode.

    Args:
        spec: Base model the teacher was fine-tuned from (only used for the LoRA path,
            which composes the adapter onto a freshly loaded base).
        checkpoint_dir: The teacher's ``checkpoints/`` directory.
        hf_cache_root: Hugging Face cache dir (LoRA base download).
        hub_org: Hub org name (LoRA base resolution).
        device: Torch device.
        dtype: Weight dtype (default fp32 -- repo convention; exact eval probabilities).
        expected_mode: Optional ``"lora"``/``"full"`` from the run manifest; if it
            disagrees with the on-disk detection, a warning is logged and the on-disk
            files win (the files are ground truth for loading).

    Returns:
        Model in eval mode on `device` -- a `PeftModel` (LoRA; wrapping a
        `GPTNeoXForCausalLM`, or the legacy `hf_olmo` `OLMoForCausalLM` for a
        `DataDecideSpec`) or a `GPTNeoXForCausalLM` (full FT).
    """
    mode = detect_finetune_mode(checkpoint_dir)
    if expected_mode is not None and expected_mode != mode:
        logger.warning(
            "Checkpoint {} detected as {} but manifest says {}; loading as {} "
            "(on-disk files are ground truth).",
            checkpoint_dir, mode, expected_mode, mode,
        )
    if mode == "lora":
        base = load_model(spec, hf_cache_root, hub_org, device, dtype=dtype)
        model = PeftModel.from_pretrained(base, str(checkpoint_dir))
        model.eval()
        return model
    if isinstance(spec, DataDecideSpec):
        raise NotImplementedError(
            "full-finetune eval loading is not wired for DataDecide models"
        )
    model = GPTNeoXForCausalLM.from_pretrained(
        str(checkpoint_dir), torch_dtype=dtype
    )
    model.to(device)
    model.eval()
    return model
