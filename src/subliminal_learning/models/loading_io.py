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

from subliminal_learning.models.datadecide import (
    DataDecideSpec,
    is_dd_ref,
    load_datadecide_model,
    load_datadecide_tokenizer,
    parse_dd_ref,
)
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id


# The dtype knob the heavy pipelines expose to the CLI. Kept tiny and explicit so
# an unknown value fails fast at config time rather than loading the wrong precision.
_DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def resolve_dtype(name: str) -> torch.dtype:
    """Map a `--dtype` string to a torch dtype, validating the choice.

    Only {"float32", "bfloat16"} are accepted; "float32" is the default everywhere
    (see `load_model`). Anything else raises so a typo can't silently downcast.
    """
    try:
        return _DTYPE_BY_NAME[name]
    except KeyError:
        raise ValueError(
            f"unknown dtype {name!r}; expected one of {sorted(_DTYPE_BY_NAME)}"
        ) from None


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
    if isinstance(spec, DataDecideSpec):
        # DataDecide bakes its org into the repo id; `hub_org` is ignored.
        return load_datadecide_tokenizer(spec, hf_cache_root)
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
    """Load a PolyPythias (or DataDecide) causal LM on the specified device.

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
        Model in eval mode on `device` -- a `GPTNeoXForCausalLM`, or the legacy
        `hf_olmo` `OLMoForCausalLM` for a `DataDecideSpec`.
    """
    if isinstance(spec, DataDecideSpec):
        # DataDecide bakes its org into the repo id; `hub_org` is ignored.
        return load_datadecide_model(spec, hf_cache_root, device, dtype=dtype)
    repo_id = resolve_hf_repo_id(spec, hub_org)
    kwargs: dict = {
        "cache_dir": str(hf_cache_root / "hub"),
        "torch_dtype": dtype,
    }
    if spec.revision is not None:
        kwargs["revision"] = spec.revision
    logger.info("Loading {} (revision={})", repo_id, spec.revision or "main")
    model = GPTNeoXForCausalLM.from_pretrained(repo_id, **kwargs)
    model.to(device)
    model.eval()
    return model


def load_base_from_ref(
    ref: str,
    hf_cache_root: Path,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a base model+tokenizer from a manifest `base_model_repo` ref string.

    The ref is the run-identity provenance string: a bare repo id, or
    `{repo_id}@{revision}` (always the latter for DataDecide, whose seeds are
    branches of one repo). DataDecide refs route through the pinned `hf_olmo`
    loaders; anything else goes to Auto* with `revision=` only when the ref
    carries one -- a bare `repo@rev` string is NOT a valid Hub repo id.

    Args:
        ref: Manifest base ref.
        hf_cache_root: Hugging Face cache dir (from the paths YAML, not HF_HOME).
        device: Torch device.
        dtype: Weight dtype; fp32 by default, as everywhere else here.

    Returns:
        (model in eval mode on `device`, tokenizer with `pad_token` set).
    """
    if is_dd_ref(ref):
        spec = parse_dd_ref(ref)
        return (load_datadecide_model(spec, hf_cache_root, device, dtype=dtype),
                load_datadecide_tokenizer(spec, hf_cache_root))
    repo_id, _, revision = ref.partition("@")
    kwargs: dict = {"cache_dir": str(hf_cache_root / "hub")}
    if revision:
        kwargs["revision"] = revision
    # Callers run offline-ish against a fully warm cache; the retry keeps a failed
    # metadata fetch from being fatal.
    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, **kwargs)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, local_files_only=True, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=dtype, **kwargs)
    except OSError:
        model = AutoModelForCausalLM.from_pretrained(
            repo_id, torch_dtype=dtype, local_files_only=True, **kwargs)
    model.to(device)
    model.eval()
    return model, tokenizer
