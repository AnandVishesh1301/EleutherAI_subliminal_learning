"""Loading PolyPythias models and tokenizers from the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from transformers import AutoTokenizer, GPTNeoXForCausalLM, PreTrainedTokenizerBase

from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id


def load_tokenizer(
    spec: PythiaModelSpec,
    hf_cache_root: Path,
    hub_org: str,
) -> PreTrainedTokenizerBase:
    """Load the shared GPT-NeoX tokenizer for a PolyPythias model spec.

    Args:
        spec: Model size/seed/variant/revision selection.
        hf_cache_root: Local Hugging Face Hub cache directory (HF_HOME).
        hub_org: Hub org name, e.g. "EleutherAI".

    Returns:
        Tokenizer with `pad_token` set (GPT-NeoX ships without one).
    """
    repo_id = resolve_hf_repo_id(spec, hub_org)
    kwargs: dict = {"cache_dir": str(hf_cache_root)}
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
) -> GPTNeoXForCausalLM:
    """Load a PolyPythias causal LM in bf16 on the  specified device.

    Args:
        spec: Model size/seed/variant/revision selection.
        hf_cache_root: Local Hugging Face Hub cache directory (HF_HOME).
        hub_org: Hub org name, e.g. "EleutherAI".
        device: Torch device string, e.g. "cuda:0".

    Returns:
        Model in eval mode on `device`.
    """
    repo_id = resolve_hf_repo_id(spec, hub_org)
    kwargs: dict = {"cache_dir": str(hf_cache_root), "torch_dtype": torch.bfloat16}
    if spec.revision is not None:
        kwargs["revision"] = spec.revision
    logger.info("Loading {} (revision={})", repo_id, spec.revision or "main")
    model = GPTNeoXForCausalLM.from_pretrained(repo_id, **kwargs)
    model.to(device)
    model.eval()
    return model
