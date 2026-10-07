"""Reading a trained LoRA adapter off disk, in the two shapes the surgeries need.

PEFT names every saved tensor ``base_model.model.{module}.lora_{A,B}.weight``, so both
readers here are key-stripping loaders over the same file; they differ only in the shape
they hand back — ``{module: (B, A)}`` pairs (what the ΔW builders want) versus
``{module: {"A": ..., "B": ...}}`` (what the intruder scanners want, which also key their
base spectra on the ``gpt_neox.``-less module name).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np

from subliminal_learning.forgetting.weightspace.lora import lora_svd

LORA_A_SUFFIX = ".lora_A.weight"
LORA_B_SUFFIX = ".lora_B.weight"
PEFT_PREFIX = "base_model.model."


def module_name(key: str) -> str:
    """``base_model.model.{module}.lora_A.weight`` -> ``{module}``."""
    return key.removeprefix(PEFT_PREFIX).removesuffix(LORA_A_SUFFIX).removesuffix(
        LORA_B_SUFFIX)


def load_lora_tensors(path: Path | str) -> dict[str, np.ndarray]:
    """Every ``.lora_*`` tensor of an adapter file, as numpy, keyed by its saved name."""
    from safetensors import safe_open

    with safe_open(str(path), "np") as sf:
        return {k: sf.get_tensor(k) for k in sf.keys() if ".lora_" in k}


def has_nonfinite(tensors: Mapping[str, np.ndarray]) -> bool:
    """True if ANY tensor holds a non-finite value (a diverged run is saved all-NaN)."""
    return any(not np.isfinite(v).all() for v in tensors.values())


def iter_lora_pairs(
    tensors: Mapping[str, np.ndarray], *, sort: bool = False
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(module, B, A)`` in float64 for every ``lora_A``/``lora_B`` pair.

    Iteration follows the tensor mapping's own key order (the safetensors header order)
    unless `sort` is set — the callers that draw random directions per module depend on
    that order, so it is never silently changed.
    """
    keys = [k for k in tensors if k.endswith(LORA_A_SUFFIX)]
    for key_a in sorted(keys) if sort else keys:
        a = np.asarray(tensors[key_a], dtype=np.float64)
        b = np.asarray(tensors[key_a.replace("lora_A", "lora_B")], dtype=np.float64)
        yield module_name(key_a), b, a


def load_adapter_ab(
    path: Path | str, *, strip_gpt_neox: bool = False, lora_only: bool = True
) -> dict[str, dict[str, np.ndarray]]:
    """``{module: {"A": ..., "B": ...}}`` at the file's own dtype.

    Args:
        strip_gpt_neox: Also drop the ``gpt_neox.`` level, matching the module keys the
            base-spectra caches of the intruder scanners use.
        lora_only: Skip tensors that are neither ``lora_A`` nor ``lora_B``. The scanners
            filed every key with an ``"A" if lora_A else "B"`` test, which would file a
            stray non-LoRA tensor as a ``B``; PEFT adapters here hold nothing else, so
            the two agree in practice, but the flag keeps the old rule available rather
            than silently changing it.
    """
    from safetensors import safe_open

    out: dict[str, dict[str, np.ndarray]] = {}
    with safe_open(str(path), "np") as sf:
        for k in sf.keys():
            if lora_only and ".lora_A." not in k and ".lora_B." not in k:
                continue
            mod = k.removeprefix(PEFT_PREFIX + "gpt_neox.") if strip_gpt_neox else k
            mod = (mod.removeprefix(PEFT_PREFIX)
                   .removesuffix(LORA_A_SUFFIX).removesuffix(LORA_B_SUFFIX))
            out.setdefault(mod, {})["A" if ".lora_A." in k else "B"] = sf.get_tensor(k)
    return out


def adapter_direction_pairs(
    ck: Path, keep, *, key_suffix: str = "", skip_nonfinite: bool = False
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], int] | None:
    """``({module: (U, V, sigmas)}, lora_rank)`` for a donor checkpoint dir.

    The donor side of the injection ladders: each module's exact ΔW = s·B·A singular
    directions with sigmas already scaled by ``s = alpha/r``. Only modules whose key
    (plus `key_suffix`) is in `keep` are returned, in the adapter file's own key order.
    `skip_nonfinite` returns None for a diverged (all-NaN) adapter instead of
    propagating NaNs.
    """
    acfg = json.loads((ck / "adapter_config.json").read_text())
    s = acfg["lora_alpha"] / acfg["r"]
    tens = load_lora_tensors(ck / "adapter_model.safetensors")
    if skip_nonfinite and has_nonfinite(tens):
        return None
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for mod, b, a in iter_lora_pairs(tens):
        key = mod + key_suffix
        if key not in keep:
            continue
        u, sv, v = lora_svd(b, a, s)
        out[key] = (u, v, sv)
    return out, int(acfg["r"])
