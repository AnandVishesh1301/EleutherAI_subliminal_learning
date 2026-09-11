"""Raw residual-stream capture across model families — the shared extraction basis.

Single source of truth for "give me the residual stream in the RAW post-block basis".
``capture_raw_activations`` hooks the embedding layer and every decoder block directly, so
ALL ``n_blocks + 1`` slots are raw: slot 0 is the embedding module's output (whatever enters
block 0) and slot ``i`` (i >= 1) is the residual coming out of decoder block ``i - 1`` —
including the LAST block, whose raw output ``output_hidden_states=True`` never exposes (its
final slot is the last block's output AFTER the final layernorm). Consumers that inject into
the raw residual basis (`subliminal_learning.steering.hooks`) or compare raw states across
models (`subliminal_learning.cka.extract_io`) therefore strip ONLY the embedding slot
(``[1:]``) and every block, last one included, lines up with a hookable module.

The resolver is generic: it looks for the module that owns an embedding-like child plus a
block ``ModuleList``, under any naming convention the repo's base families use — GPT-NeoX
(``embed_in`` / ``layers``), Llama / Qwen2 / Gemma2 / Gemma3 (``embed_tokens`` / ``layers``)
and hf_olmo's DataDecide models (``transformer.wte`` / ``transformer.blocks``). It unwraps
PEFT/HF wrappers, so a ``PeftModel``-wrapped teacher/student, the bare base model and the
bare core module all resolve to the same block list, and extraction and injection share one
notion of "the blocks".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# (embedding attribute, block-list attribute) pairs: GPT-NeoX, the Llama-style tree
# (Llama / Qwen2 / Gemma2 / Gemma3), and hf_olmo's ``transformer`` ModuleDict.
_CORE_LAYOUTS = (("embed_in", "layers"), ("embed_tokens", "layers"), ("wte", "blocks"))

# Container modules that hold submodules but have no forward of their own.
_NOT_CALLABLE = (torch.nn.ModuleDict, torch.nn.ModuleList)

# Attributes that hold a core one level deeper (PEFT, HF ``model``/``transformer`` nesting).
_WRAPPER_NAMES = ("base_model", "model", "gpt_neox", "transformer")


@dataclass(frozen=True)
class RawCore:
    """What the raw basis needs from a decoder: the owning module, embeddings and blocks.

    ``module`` is the innermost *callable* module the embedding and blocks hang off (e.g.
    ``GPTNeoXModel``, ``LlamaModel``, hf_olmo's ``OLMo`` — whose layout lives one level
    down in a ``ModuleDict``); callers that want to skip the LM head's logits can forward
    it instead of the full causal LM.
    """

    module: torch.nn.Module
    embed: torch.nn.Module
    blocks: torch.nn.Module


def _direct_child(obj, name: str):
    modules = getattr(obj, "_modules", None)
    if modules is not None and name in modules:
        return modules[name]
    return getattr(obj, name, None)


def _as_core(obj, forward_owner) -> RawCore | None:
    """The `RawCore` ``obj`` itself is, or ``None`` if it owns no known layout.

    ``forward_owner`` is the nearest ancestor (``obj`` included) that is actually callable
    — hf_olmo keeps its layout in a ``ModuleDict``, which has no forward of its own.
    """
    for embed_name, blocks_name in _CORE_LAYOUTS:
        embed = _direct_child(obj, embed_name)
        blocks = _direct_child(obj, blocks_name)
        if embed is not None and blocks is not None:
            return RawCore(module=forward_owner, embed=embed, blocks=blocks)
    return None


def resolve_raw_core(model) -> RawCore:
    """Return the `RawCore` of ``model``, unwrapping PEFT/HF wrappers.

    Breadth-first walk over the usual wrapper attributes (``base_model``, ``model``,
    ``gpt_neox``, ``transformer``), so a bare core, a CausalLM head and a doubly-wrapped
    PEFT model all resolve to the same embedding + block list.
    """
    seen: set[int] = set()
    queue = [(model, model)]
    while queue:
        obj, owner = queue.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        core = _as_core(obj, owner)
        if core is not None:
            return core
        for name in _WRAPPER_NAMES:
            child = _direct_child(obj, name)
            if child is not None and id(child) not in seen:
                child_owner = owner if isinstance(child, _NOT_CALLABLE) else child
                queue.append((child, child_owner))
    raise AttributeError(
        f"can't find a decoder core (embedding + block list) under {type(model).__name__}"
    )


def resolve_blocks(model):
    """Return the decoder-block list (``ModuleList``), unwrapping PEFT/HF wrappers.

    ``blocks[i]`` is the module whose forward output is the raw residual out of block ``i``
    — the same module `capture_raw_activations` reads slot ``i + 1`` from, so a steering
    hook on ``blocks[i]`` and an extracted slot ``i + 1`` live in the same basis.
    """
    return resolve_raw_core(model).blocks


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
    Works on a bare causal LM of any supported family (GPT-NeoX, Llama, Qwen2, Gemma2/3,
    hf_olmo), on the bare core module, or on any PEFT/HF-wrapped variant; hidden states flow
    through PEFT unchanged.
    """
    core = resolve_raw_core(model)
    modules = [core.embed, *core.blocks]
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
        try:
            out = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=False
            )
        except TypeError:  # a forward without ``use_cache`` (e.g. a bare core module)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
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
