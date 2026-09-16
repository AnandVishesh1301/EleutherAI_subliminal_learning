"""Loading PolyPythias models and tokenizers from the Hugging Face Hub or a local dir."""

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

from subliminal_learning.models.neox_convert_io import read_conversion_manifest
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id


def local_snapshot_dir(spec: PythiaModelSpec) -> Path:
    """Directory serving `spec.revision` under the spec's `local_root`, identity-checked.

    The directory's conversion manifest must name the same base and revision as the
    spec, so a spec can never silently load another checkpoint's weights.

    Raises:
        ValueError: If the spec has no `local_root`, or the manifest disagrees with it.
        FileNotFoundError: If the directory holds no conversion manifest.
    """
    root = getattr(spec, "local_root", None)
    if root is None:
        raise ValueError(f"{spec!r} is not a local spec")
    snapshot = Path(root) / str(spec.revision)
    manifest = read_conversion_manifest(snapshot)
    expected_repo = f"local/{manifest.base}"
    if manifest.revision != spec.revision or spec.repo_id != expected_repo:
        raise ValueError(
            f"{snapshot}: manifest identity {expected_repo}@{manifest.revision} does not "
            f"match spec {spec.repo_id}@{spec.revision}"
        )
    return snapshot


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
    if getattr(spec, "local_root", None) is not None:
        tokenizer = AutoTokenizer.from_pretrained(str(local_snapshot_dir(spec)))
    else:
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
        class `AutoModelForCausalLM` resolves from the repo's config. A spec carrying
        `local_root` loads from its identity-checked local snapshot instead of the Hub.
    """
    if getattr(spec, "local_root", None) is not None:
        snapshot = local_snapshot_dir(spec)
        logger.info("Loading {} from local snapshot {}", spec.repo_id, snapshot)
        model = AutoModelForCausalLM.from_pretrained(str(snapshot), torch_dtype=dtype)
        model.to(device)
        model.eval()
        return model
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
