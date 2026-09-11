"""Loading PolyPythias models and tokenizers from the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTNeoXForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id


def load_tokenizer(
    spec: PythiaModelSpec,
    hf_cache_root: Path,
    hub_org: str,
) -> PreTrainedTokenizerBase:
    """Load the shared GPT-NeoX tokenizer for a PolyPythias model spec.

    Args:
        spec: Model size/seed/variant/revision selection.
        hf_cache_root: Hugging Face cache dir.
        hub_org: Hub org name.

    Returns:
        Tokenizer with `pad_token` set.
    """
    
    repo_id = resolve_hf_repo_id(spec, hub_org)
    kwargs: dict = {"cache_dir": str(hf_cache_root / "hub")}
    if spec.revision is not None:
        kwargs["revision"] = spec.revision
    tokenizer = AutoTokenizer.from_pretrained(repo_id, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    spec: PythiaModelSpec,
    hf_cache_root: Path,
    hub_org: str,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> PreTrainedModel:
    """Load a PolyPythias causal LM on the specified device.

    Args:
        spec: Model size/seed/variant/revision selection.
        hf_cache_root: Hugging Face cache dir.
        hub_org: Hub org name.
        device: Torch device.
        dtype: Weight dtype to load in. Defaults to `float32` everywhere: these are
            160M models where fp32 is cheap, and it keeps training gradients/optimizer
            state and inference log-probs/probabilities full-precision (bf16's ~3-digit
            mantissa swamps small updates and coarsens the exact next-token probabilities
            the eval reads). Pass `bfloat16` explicitly only to trade accuracy for memory.

    Returns:
        Model in eval mode on `device`. A `GPTNeoXForCausalLM` for a PolyPythias spec;
        for a cross-family `HfModelSpec` (any spec carrying `repo_id`) the architecture
        class `AutoModelForCausalLM` resolves from the repo's config.
    """
    repo_id = resolve_hf_repo_id(spec, hub_org)
    kwargs: dict = {
        "cache_dir": str(hf_cache_root / "hub"),
        "torch_dtype": dtype,
    }
    if spec.revision is not None:
        kwargs["revision"] = spec.revision
    logger.info("Loading {} (revision={})", repo_id, spec.revision or "main")
    cls = (
        AutoModelForCausalLM
        if getattr(spec, "repo_id", None) is not None
        else GPTNeoXForCausalLM
    )
    model = cls.from_pretrained(repo_id, **kwargs)
    model.to(device)
    model.eval()
    return model
