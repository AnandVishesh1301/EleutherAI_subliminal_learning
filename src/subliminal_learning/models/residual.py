"""Raw residual-stream capture for GPT-NeoX — the shared extraction basis.

Single source of truth for "give me the residual stream in the RAW post-block basis".
``capture_raw_activations`` hooks the embedding layer and every decoder block directly, so
ALL ``n_blocks + 1`` slots are raw: slot 0 is the embedding output and slot ``i`` (i >= 1)
is the residual coming out of decoder block ``i - 1`` — including the LAST block, whose raw
output ``output_hidden_states=True`` never exposes (its final slot is the last block's
output AFTER the final layernorm). Consumers that inject into the raw residual basis
(`subliminal_learning.steering.hooks`) or compare raw states across models
(`subliminal_learning.cka.extract_io`) therefore strip ONLY the embedding slot (``[1:]``)
and every block, last one included, lines up with a hookable module.

The module resolvers unwrap PEFT/HF wrappers (a ``PeftModel``-wrapped teacher/student and a
bare base model resolve to the same block list), so extraction and injection share one
notion of "the blocks".
"""

from __future__ import annotations

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


def _resolve_core(model):
    """Return the GPT-NeoX core module (owns ``embed_in`` + ``layers``), unwrapping wrappers.

    Checks the known GPT-NeoX paths (bare model and PEFT-wrapped), then falls back to a
    bounded DFS over the usual wrapper attributes so a merged or doubly-wrapped model
    still resolves.
    """
    known_paths = (
        ("gpt_neox",),
        ("base_model", "model", "gpt_neox"),
        ("base_model", "model", "model", "gpt_neox"),
        ("model", "gpt_neox"),
    )
    for path in known_paths:
        core = _get_path(model, path)
        if core is not None and _direct_child(core, "layers") is not None:
            return core

    seen: set[int] = set()
    stack = [model]
    wrapper_names = ("base_model", "model", "gpt_neox")
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if (
            _direct_child(obj, "layers") is not None
            and _direct_child(obj, "embed_in") is not None
        ):
            return obj
        for name in wrapper_names:
            child = _direct_child(obj, name)
            if child is not None and id(child) not in seen:
                stack.append(child)
    raise AttributeError(f"can't find a GPT-NeoX core under {type(model).__name__}")


def resolve_blocks(model):
    """Return the decoder-block list (``ModuleList``), unwrapping PEFT/HF wrappers.

    ``blocks[i]`` is the module whose forward output is the raw residual out of block ``i``
    — the same module `capture_raw_activations` reads slot ``i + 1`` from, so a steering
    hook on ``blocks[i]`` and an extracted slot ``i + 1`` live in the same basis.
    """
    blocks = _get_path(model, ("layers",)) or _get_path(model, ("model", "layers"))
    if blocks is not None:
        return blocks
    return _resolve_core(model).layers


def num_raw_slots(model) -> int:
    """Number of raw capture slots: embeddings + one per decoder block (13 on 160m)."""
    return len(resolve_blocks(model)) + 1


@torch.no_grad()
def forward_with_raw_activations(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor
):
    """One forward pass; returns ``(model_output, raw_slots)``.

    ``raw_slots`` is ``n_blocks + 1`` RAW tensors of shape ``[B, T, H]``: slot 0 is the
    embedding output and slot ``i`` (i >= 1) the raw residual stream out of decoder block
    ``i - 1`` (the final layernorm never touches any slot). ``model_output`` is the
    ordinary forward result, for consumers that also need logits from the SAME pass.
    Works on a bare ``GPTNeoXForCausalLM`` or any PEFT/HF-wrapped variant; hidden states
    flow through PEFT unchanged.
    """
    core = _resolve_core(model)
    modules = [core.embed_in, *core.layers]
    captured: list[torch.Tensor | None] = [None] * len(modules)
    handles = []

    def _grab(idx: int):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden

        return hook

    for i, module in enumerate(modules):
        handles.append(module.register_forward_hook(_grab(i)))
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    assert all(a is not None for a in captured), "a hook failed to capture activations"
    return out, captured


@torch.no_grad()
def capture_raw_activations(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> list[torch.Tensor]:
    """`forward_with_raw_activations` returning only the ``n_blocks + 1`` raw slots."""
    _, captured = forward_with_raw_activations(model, input_ids, attention_mask)
    return captured
