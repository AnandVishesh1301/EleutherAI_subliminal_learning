"""Selecting the LoRA-target matrices of a base model, patching them, reading them back.

Every surgery in this family works the same way: hold the pristine ``W₀`` of each
LoRA-target module, write ``W₀ + ΔW'`` into the live model, read the pp1 probe (KL /
ΔCE) and the trait probe (p_target), restore. Disk is never touched.

Two conventions for "the target modules" coexist in the campaign and are BOTH kept,
because the module keys they produce are the keys stored in existing outputs:

- `iter_target_modules` — ``named_modules()``, ``gpt_neox.layers.*`` whose last segment
  is a `TARGET_KINDS` name; keys are module paths (``...attention.dense``).
- `iter_target_weight_params` — ``named_parameters()``, ``*.weight`` containing one of
  `TARGET_SUBSTRINGS`; keys carry the ``.weight`` suffix.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

TARGET_KINDS = frozenset({"query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"})
TARGET_SUBSTRINGS = ("attention.dense", "attention.query_key_value",
                     "mlp.dense_h_to_4h", "mlp.dense_4h_to_h")


def iter_target_modules(model) -> Iterator[tuple[str, Any]]:
    """``(name, module)`` for every LoRA-target block module carrying a weight."""
    for name, mod in model.named_modules():
        if (name.startswith("gpt_neox.layers.") and name.split(".")[-1] in TARGET_KINDS
                and hasattr(mod, "weight")):
            yield name, mod


def iter_target_weight_params(model) -> Iterator[tuple[str, Any]]:
    """``(name, parameter)`` for every LoRA-target ``*.weight`` parameter."""
    for name, par in model.named_parameters():
        if name.endswith(".weight") and any(t in name for t in TARGET_SUBSTRINGS):
            yield name, par


def capture_w0(model, *, as_numpy: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    """``(w0, mods)`` over `iter_target_modules`: the pristine weights and the modules.

    `as_numpy` gives float64 CPU arrays (for surgeries whose arithmetic is done in numpy);
    otherwise the clones stay on the model's device at its dtype.
    """
    w0: dict[str, Any] = {}
    mods: dict[str, Any] = {}
    for name, mod in iter_target_modules(model):
        w0[name] = (mod.weight.detach().cpu().double().numpy() if as_numpy
                    else mod.weight.detach().clone())
        mods[name] = mod
    return w0, mods


def capture_named_weights(model, names) -> dict[str, np.ndarray]:
    """float64 CPU copies of the weights of the named modules (an explicit module set)."""
    wanted = set(names)
    return {name: mod.weight.detach().cpu().double().numpy()
            for name, mod in model.named_modules() if name in wanted}


def load_base_target_weights(
    repo: str, hf_cache=None, *, strip_gpt_neox: bool = False
) -> dict[str, np.ndarray]:
    """``{module: W₀}`` float64 for the LoRA targets of `repo`, model then released.

    The CPU-only scanners' shared preamble (they each then do their own spectral work on
    top). Keys drop the ``.weight`` suffix, and the ``gpt_neox.`` level too when
    `strip_gpt_neox` — the key convention their adapter-side loaders use. `hf_cache` is
    the cache ROOT (``hub/`` is appended); None leaves the HF default in force, which is
    what the ``.env``-driven callers rely on.
    """
    import torch
    from transformers import AutoModelForCausalLM

    kw = {} if hf_cache is None else {"cache_dir": str(hf_cache / "hub")}
    mdl = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32, **kw)
    out: dict[str, np.ndarray] = {}
    for name, par in iter_target_weight_params(mdl):
        key = name.removeprefix("gpt_neox.") if strip_gpt_neox else name
        out[key.removesuffix(".weight")] = par.detach().numpy().astype(np.float64)
    del mdl
    return out


def base_param_svds(model, *, include_w0np: bool = False) -> dict[str, dict]:
    """Per-target-parameter ``{param, W0, U, S, Vt, shape}`` (plus ``W0np``) thin SVDs.

    The ladder scripts' shared preamble: one full SVD of every LoRA-target ``W₀`` (the
    expensive part, ~1 min CPU for a 160m), keeping the pristine weight for restore. The
    RMT scale ``θ̄`` is NOT computed here — the caller adds it from ``S`` and ``shape``.
    """
    out: dict[str, dict] = {}
    for name, par in iter_target_weight_params(model):
        w = par.detach().cpu().numpy().astype(np.float64)
        u, s, vt = np.linalg.svd(w, full_matrices=False)
        entry = dict(param=par, W0=par.detach().clone(), U=u, S=s, Vt=vt, shape=w.shape)
        if include_w0np:
            entry["W0np"] = w
        out[name] = entry
    return out


def patch_modules(mods: Mapping[str, Any], w0: Mapping[str, Any],
                  deltas: Mapping[str, np.ndarray]) -> None:
    """``W ← W₀ + ΔW`` for each module in `deltas`, ΔW cast into the module's dtype.

    The addition happens in the module's dtype (`w0` holds device tensors); use
    `patch_modules_f64` for the callers that add in float64 before casting.
    """
    import torch

    with torch.no_grad():
        for name, dw in deltas.items():
            mod = mods[name]
            mod.weight.copy_(w0[name] + torch.as_tensor(
                dw, dtype=mod.weight.dtype, device=mod.weight.device))


def patch_modules_f64(mods: Mapping[str, Any], w0: Mapping[str, np.ndarray],
                      deltas: Mapping[str, np.ndarray]) -> None:
    """``W ← W₀ + ΔW`` summed in float64 numpy, then cast — `w0` is float64 arrays."""
    import torch

    with torch.no_grad():
        for name, dw in deltas.items():
            mod = mods[name]
            mod.weight.copy_(torch.as_tensor(
                w0[name] + dw, dtype=mod.weight.dtype, device=mod.weight.device))


def restore_modules(mods: Mapping[str, Any], w0: Mapping[str, Any]) -> None:
    """Write every pristine ``W₀`` back (accepts tensor or float64-array `w0`)."""
    import torch

    with torch.no_grad():
        for name, mod in mods.items():
            mod.weight.copy_(torch.as_tensor(
                w0[name], dtype=mod.weight.dtype, device=mod.weight.device))


def patch_param_modules(mods: Mapping[str, dict], deltas: Mapping[str, np.ndarray]) -> None:
    """``W ← W₀ + Δ`` for a `base_param_svds` map (Δ cast to float32, as the ladders do)."""
    import torch

    with torch.no_grad():
        for name, d in deltas.items():
            m = mods[name]
            m["param"].copy_(m["W0"] + torch.tensor(
                d, dtype=torch.float32, device=m["param"].device))


def restore_param_modules(mods: Mapping[str, dict]) -> None:
    """Write every pristine ``W₀`` of a `base_param_svds` map back."""
    import torch

    with torch.no_grad():
        for m in mods.values():
            m["param"].copy_(m["W0"])


def pile_readout(model, pool, device, base_logp, ce_base: float) -> tuple[float, float]:
    """``(pp1 KL(base ‖ model), ΔCE)`` of the currently patched model.

    The logprob tensor is GPU-resident and large (~3 GB on pp1), so it is freed before
    returning rather than handed back.
    """
    import torch

    from subliminal_learning.coherence.probe_io import mean_ce, mean_kl, pile_logprobs

    with torch.no_grad():
        logp = pile_logprobs(model, pool, device)
        kl = mean_kl(base_logp, logp)
        dce = mean_ce(logp, pool) - ce_base
        del logp
    return float(kl), float(dce)


def mean_p_target(pref: Mapping[str, tuple]) -> float:
    """Mean p(target) over the prompts of a `target_prob_and_rank_per_prompt` result."""
    return float(np.mean([p for p, _ in pref.values()]))
