"""Residual-stream steering hooks for GPT-NeoX (PolyPythias).

Port of Blank et al.'s ``subliminal.steering_utils`` (``BLANK_ARCHITECTURE.md``). The only
substantive change is the block resolver: PolyPythias are ``GPTNeoXForCausalLM``, whose
decoder blocks live at ``model.gpt_neox.layers`` (and under a ``PeftModel`` at
``model.base_model.model.gpt_neox.layers``) -- not the ``model.model.layers`` layout Blank's
LLaMA/Qwen resolver assumes.

Layer indexing: ``layers=[i]`` hooks block ``i`` directly (0-indexed), and
``v_raw_per_layer[i]`` is the residual coming *out* of that block. The call site strips BOTH
ends of the extraction stack (``v["raw"][1:-1]``): the embedding slot AND the trailing
post-final-layernorm slot. What remains are the RAW outputs of blocks 0..n-2 only (11 of the
12 blocks on Pythia-160m). The last model block's raw output never had a slot of its own -- its
only appearance in ``hidden_states`` is post-final-layernorm, which does not belong in the raw
residual basis -- so the final block is simply not steerable here (an inherited Blank quirk).

Modes (the two the steerability screen needs):

- ``add``     -> ``h <- h + alpha * v``            sufficiency: inject the trait direction
- ``project`` -> ``h <- h - alpha * (h . v_hat) v_hat``   necessity: ablate it (alpha=1 = full)

Positions:

- ``broadcast``  -> shift every token position on every forward (used for the single
  full-prompt forward pass that the ``p(target)`` metric reads).
- ``prompt_all`` -> shift every token of the prompt pass (T > 1); skip single-token
  autoregressive steps (matches Blank's generation-time injection).
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


def _direct_child(obj, name: str):
    modules = getattr(obj, "_modules", None)
    if modules is not None and name in modules:
        return modules[name]
    return getattr(obj, name, None)


def _get_path(obj, path: tuple[str, ...]):
    cur = obj
    for name in path:
        cur = _direct_child(cur, name)
        if cur is None:
            return None
    return cur


def _unwrap_blocks(model):
    """Return the GPT-NeoX decoder-block list, unwrapping PEFT/HF wrappers.

    Checks the known GPT-NeoX paths (bare model and PEFT-wrapped), then falls back to a
    bounded DFS over the usual wrapper attributes so a merged or doubly-wrapped model
    still resolves.
    """
    known_paths = (
        ("gpt_neox", "layers"),
        ("base_model", "model", "gpt_neox", "layers"),
        ("base_model", "model", "model", "gpt_neox", "layers"),
        ("model", "gpt_neox", "layers"),
        ("layers",),
        ("model", "layers"),
    )
    for path in known_paths:
        blocks = _get_path(model, path)
        if blocks is not None:
            return blocks

    seen: set[int] = set()
    stack = [model]
    wrapper_names = ("base_model", "model", "gpt_neox")
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        blocks = _direct_child(obj, "layers")
        if blocks is not None:
            return blocks
        for name in wrapper_names:
            child = _direct_child(obj, name)
            if child is not None and id(child) not in seen:
                stack.append(child)
    raise AttributeError(f"can't find decoder blocks under {type(model).__name__}")


def _hidden_from_output(output):
    """GPT-NeoX blocks return a tuple ``(hidden, *rest)``; some paths return a bare tensor."""
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _repack(new_hidden, tail):
    if tail is None:
        return new_hidden
    return (new_hidden, *tail)


def _make_add_hook(v: torch.Tensor, alpha: float, positions: str, norm: str):
    v = v.detach()
    if norm == "unit":
        v = v / v.norm().clamp(min=1e-12)
    elif norm != "raw":
        raise ValueError(f"norm must be 'raw' or 'unit'; got {norm!r}")

    def hook(_module, _args, output):
        hidden, tail = _hidden_from_output(output)
        sv = (alpha * v).to(hidden.dtype).to(hidden.device)
        if positions == "broadcast":
            new_hidden = hidden + sv
        elif positions == "prompt_all":
            if hidden.shape[1] <= 1:
                return _repack(hidden, tail)
            new_hidden = hidden + sv
        else:
            raise ValueError(f"unknown positions={positions!r}")
        return _repack(new_hidden, tail)

    return hook


def _make_project_hook(v: torch.Tensor, alpha: float, positions: str):
    v = v.detach()
    v_hat = v / v.norm().clamp(min=1e-12)

    def hook(_module, _args, output):
        hidden, tail = _hidden_from_output(output)
        vh = v_hat.to(hidden.dtype).to(hidden.device)
        coef = (hidden * vh).sum(dim=-1, keepdim=True)
        shift = alpha * coef * vh
        if positions == "broadcast":
            new_hidden = hidden - shift
        elif positions == "prompt_all":
            if hidden.shape[1] <= 1:
                return _repack(hidden, tail)
            new_hidden = hidden - shift
        else:
            raise ValueError(f"unknown positions={positions!r}")
        return _repack(new_hidden, tail)

    return hook


@contextmanager
def steering_hooks(
    model,
    v_raw_per_layer: torch.Tensor,
    alpha,
    mode: str,
    layers: list[int] | None = None,
    positions: str = "broadcast",
    norm: str = "raw",
):
    """Attach residual-stream forward hooks for the duration of a ``with`` block.

    Args:
        model: GPT-NeoX causal LM (clean base for ``add`` sufficiency; teacher for
            ``project`` necessity).
        v_raw_per_layer: ``[n_blocks, H]`` tensor, with BOTH the embedding slot and the
            trailing post-final-layernorm slot already stripped (``v["raw"][1:-1]``),
            indexed 0..n_blocks-1. The post-final-layernorm state is excluded so only raw
            block outputs enter the residual basis.
        alpha: scalar float (screen default) or per-layer sequence of length n_blocks.
        mode: ``"add"`` or ``"project"``.
        layers: block indices to hook (0-indexed). ``None`` hooks every block -- i.e.
            all-layers injection using each block's own ``raw`` direction.
        positions: ``"broadcast"`` or ``"prompt_all"``.
        norm: ``"raw"`` (alpha scales the measured shift; default) or ``"unit"`` (unit
            step). Only applies to ``mode="add"``.
    """
    assert mode in ("add", "project"), f"mode must be 'add' or 'project'; got {mode!r}"
    assert positions in ("broadcast", "prompt_all"), (
        f"positions must be 'broadcast' or 'prompt_all'; got {positions!r}"
    )
    n = v_raw_per_layer.shape[0]
    if layers is None:
        layers = list(range(n))
    scalar_alpha = not hasattr(alpha, "__len__")

    blocks = _unwrap_blocks(model)
    handles = []
    for l in layers:
        v = v_raw_per_layer[l]
        a = alpha if scalar_alpha else float(alpha[l])
        hook = (
            _make_add_hook(v, a, positions, norm)
            if mode == "add"
            else _make_project_hook(v, a, positions)
        )
        handles.append(blocks[l].register_forward_hook(hook))
    try:
        yield
    finally:
        for h in handles:
            h.remove()
