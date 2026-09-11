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

Both also accept the OLMo (DataDecide) layout, whose LoRA targets are
``*.transformer.blocks.{i}.{att_proj,attn_out,ff_proj,ff_out}``, and the HF-family layout
of the Qwen/Llama/Gemma arms (``model.layers.{i}.self_attn.*_proj`` / ``mlp.*_proj``) — the
module keys then match the adapter keys of those runs after `adapters.module_name` strips
PEFT's prefix.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

TARGET_KINDS = frozenset({"query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"})
TARGET_SUBSTRINGS = ("attention.dense", "attention.query_key_value",
                     "mlp.dense_h_to_4h", "mlp.dense_4h_to_h")
OLMO_TARGET_KINDS = frozenset({"att_proj", "attn_out", "ff_proj", "ff_out"})
OLMO_BLOCK_PREFIX = "transformer.blocks."
# The HF-family layout shared by Qwen2.5 / Llama-3 / Gemma (the external-model arms of
# 2026-09): ``model.layers.{i}.self_attn.{q,k,v,o}_proj`` and ``model.layers.{i}.mlp.
# {gate,up,down}_proj`` -- the seven modules the family LoRA YAMLs target. Keys again equal
# the adapter keys after `adapters.module_name` strips PEFT's prefix.
HF_TARGET_KINDS = frozenset({"q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"})
HF_LAYER_PREFIX = "model.layers."


def _is_olmo_target(name: str) -> bool:
    return OLMO_BLOCK_PREFIX in name and name.split(".")[-1] in OLMO_TARGET_KINDS


def _is_hf_target(name: str) -> bool:
    return name.startswith(HF_LAYER_PREFIX) and name.split(".")[-1] in HF_TARGET_KINDS


def _require_bare_model(model) -> None:
    """Refuse a PEFT-wrapped (or stale-wrapped) model: patching addresses base modules.

    A `PeftModel`, or a base that still carries the ``peft_config`` PEFT leaves behind
    after `unload()`, has LoRA layers or stale adapter state in the module tree, so the
    target names below would not be the bare ``W₀`` modules. Callers hand over the bare
    base (see `evaluation.ckpt_trajectories.release_base`).
    """
    if type(model).__name__.startswith("Peft") or getattr(model, "peft_config", None):
        raise TypeError(
            f"weightspace.patching needs the bare base model, got {type(model).__name__} "
            "carrying PEFT state (peft_config). Recover the pristine W0 with an UNMERGED "
            "`PeftModel.unload()` followed by `ckpt_trajectories.release_base`, or reload the "
            "base from disk; never `merge_and_unload()`, which bakes the adapter's dW into W0 "
            "and cannot be detected here")


def iter_target_modules(model) -> Iterator[tuple[str, Any]]:
    """``(name, module)`` for every LoRA-target block module carrying a weight.

    Requires the BARE base model (`_require_bare_model`); wrapper prefixes are not
    accepted, by contract.
    """
    _require_bare_model(model)
    for name, mod in model.named_modules():
        neox = name.startswith("gpt_neox.layers.") and name.split(".")[-1] in TARGET_KINDS
        if (neox or _is_olmo_target(name) or _is_hf_target(name)) and hasattr(mod, "weight"):
            yield name, mod


def iter_target_weight_params(model) -> Iterator[tuple[str, Any]]:
    """``(name, parameter)`` for every LoRA-target ``*.weight`` parameter (bare model)."""
    _require_bare_model(model)
    for name, par in model.named_parameters():
        if name.endswith(".weight") and (any(t in name for t in TARGET_SUBSTRINGS)
                                         or _is_olmo_target(name.removesuffix(".weight"))
                                         or _is_hf_target(name.removesuffix(".weight"))):
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


def _validated_lora_pairs(tensors: Mapping[str, np.ndarray],
                          mods: Mapping[str, Any]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """``[(module, B, A)]`` for EVERY LoRA pair in `tensors`, as stored, resolved against `mods`.

    The contract a λ·ΔW census needs before it evaluates anything: the lora_A and lora_B
    module sets must coincide (an orphan tensor of either kind is refused, not skipped),
    every module must name a captured base module, and ``B @ A`` must have that module's
    weight shape. A partial match would silently under-dose (the dropped modules
    contribute ΔW = 0), and equal counts can hide different sets, so names and shapes are
    compared, not counts. Arrays are returned in the file's own dtype (nothing is
    materialised here); order follows the file's key order like `adapters.iter_lora_pairs`.

    Raises:
        ValueError: on any of the above, naming the offending modules.
    """
    from subliminal_learning.weightspace.adapters import module_name

    a_keys = {module_name(k): k for k in tensors if ".lora_A." in k}
    b_keys = {module_name(k): k for k in tensors if ".lora_B." in k}
    if set(a_keys) != set(b_keys):
        raise ValueError(
            f"adapter lora_A/lora_B module sets differ: A-only {sorted(set(a_keys) - set(b_keys))[:3]}, "
            f"B-only {sorted(set(b_keys) - set(a_keys))[:3]}")
    if not a_keys:
        raise ValueError("adapter holds no lora_A/lora_B pairs")
    unmatched = sorted(set(a_keys) - set(mods))
    if unmatched:
        raise ValueError(
            f"{len(unmatched)}/{len(a_keys)} adapter modules do not resolve to a captured "
            f"base module (layout mismatch?), e.g. {unmatched[:3]}")
    pairs = []
    for name, key_a in a_keys.items():
        a = np.asarray(tensors[key_a])
        b = np.asarray(tensors[b_keys[name]])
        want = tuple(mods[name].weight.shape)
        got = (b.shape[0], a.shape[1])
        if b.shape[1] != a.shape[0] or got != want:
            raise ValueError(
                f"adapter module {name!r}: B {tuple(b.shape)} @ A {tuple(a.shape)} gives ΔW "
                f"shape {got}, base weight shape is {want}")
        pairs.append((name, b, a))
    return pairs


def resolve_adapter_deltas(tensors: Mapping[str, np.ndarray], scale: float,
                           mods: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """``{module: s·B·A}`` in float64 numpy for every LoRA pair, validated by
    `_validated_lora_pairs` (names, shapes, A/B set equality) before anything is formed."""
    from subliminal_learning.weightspace.lora import lora_delta_w

    return {name: lora_delta_w(b, a, scale) for name, b, a in _validated_lora_pairs(tensors, mods)}


def resolve_adapter_factors(tensors: Mapping[str, np.ndarray],
                            mods: Mapping[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``{module: (B, A)}`` as stored (numpy, the file's dtype), validated the same way.

    The census's per-rung patch used to form ``λ·s·B·A`` in float64 numpy on the CPU and
    upload it (8.8 s per rung on Qwen, plus 4–8 s per run to materialise ΔW). The factors
    are a few MB to ~1 GB per adapter, so `patch_modules_lora` moves each module's pair to
    the device, upcasts it to float64 there (exact for the float32/bf16 values PEFT saves)
    and forms ΔW on the device — same precision, same operation order, no resident copy.
    """
    return {name: (b, a) for name, b, a in _validated_lora_pairs(tensors, mods)}


def patch_modules_lora(mods: Mapping[str, Any], w0: Mapping[str, Any],
                       factors: Mapping[str, tuple[np.ndarray, np.ndarray]], scale: float,
                       lam: float = 1.0) -> None:
    """``W ← W₀ + fp32(λ·(s·(B·A)))`` with the product formed in float64 on the device.

    The operation order is `lora_delta_w` (``s·(B@A)``) followed by the census's ``λ·ΔW``,
    both in float64, then the same cast-and-add `patch_modules` performs — so the float32
    ΔW equals the CPU path's except where a float64 value sits within the two BLAS
    libraries' accumulation-order difference (~1e-16 relative) of a float32 rounding tie.
    Measured on the real Qwen census adapters when this landed (2026-09-10): 0 differing
    elements of 1.31e9 at rank 16 and at rank 256. Peak transient per module is one
    float64 ΔW (110 MB on Qwen's largest projection); nothing stays resident.
    """
    import torch

    with torch.no_grad():
        for name, (b, a) in factors.items():
            mod = mods[name]
            dev = mod.weight.device
            bt = torch.as_tensor(np.asarray(b), device=dev).double()
            at = torch.as_tensor(np.asarray(a), device=dev).double()
            dw = (lam * (scale * (bt @ at))).to(mod.weight.dtype)
            mod.weight.copy_(w0[name] + dw)
            del bt, at, dw


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


def pile_readout(model, pool, device, base_logp, ce_base: float,
                 logp_device: str | None = None) -> tuple[float, float]:
    """``(pp1 KL(base ‖ model), ΔCE)`` of the currently patched model, streamed.

    Reduced chunk by chunk on the GPU against `base_logp` (`probe_io.stream_kl_ce`): no
    per-model log-prob list is built, so the peak is one batch of logits at any vocabulary
    and nothing is reduced on the CPU. `base_logp` may live anywhere — a CPU-parked list
    (`parked_logprobs`, the census default) is copied to the device one chunk at a time.
    `logp_device` is accepted for the existing callers and ignored: there is no list left
    to park.
    """
    from subliminal_learning.coherence.probe_io import stream_kl_ce

    kl, ce = stream_kl_ce(model, pool, device, base_logp)
    return float(kl), float(ce - ce_base)


def mean_p_target(pref: Mapping[str, tuple]) -> float:
    """Mean p(target) over the prompts of a `target_prob_and_rank_per_prompt` result."""
    return float(np.mean([p for p, _ in pref.values()]))
