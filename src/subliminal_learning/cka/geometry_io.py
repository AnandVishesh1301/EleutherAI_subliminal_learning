"""Activation capture for the geometry ladder, stamped with its slot convention.

**The trap this module exists to close.** Two 13-slot conventions live in this repo
and they differ in exactly ONE slot, so mixing them is silent:

- ``post_block`` -- `cka.extract_io.layer_activations` hooks ``gpt_neox.embed_in`` and
  every decoder block, giving ``(embed_out, block_0_out, ..., block_11_out)``: RAW
  post-block residuals at every slot.
- ``hidden_states`` -- `steering.vectors.mean_activations` passes
  ``output_hidden_states=True``. Verified against ``transformers==4.54.0``
  ``models/gpt_neox/modeling_gpt_neox.py``: ``GPTNeoXModel.forward`` appends
  ``hidden_states`` at the TOP of each block iteration (:470-473), then applies
  ``self.final_layer_norm`` (:492) and appends once more (:495). The stack is therefore
  ``(embed_out, block_0_out, ..., block_10_out, final_layer_norm(block_11_out))`` -- the
  last block's RAW output has no slot of its own.

Both are 13 tensors of ``[B, T, 768]`` on Pythia-160m: slots 0-11 agree, slot 12 does
not. Trait DIRECTIONS come from `steering.vectors.diff_vector` (``hidden_states``) while
trait COORDINATES are read off an activation slot, so a convention mismatch makes
``d_trait`` quietly wrong at exactly the slot the trait is strongest in. Every cache
path, result row and trait-vector filename therefore records `SlotConvention`, and
`require_same_convention` raises rather than pairing two conventions.

Angela's committed CKA results were computed under ``post_block``; the ladder records
BOTH conventions instead of silently re-baselining her numbers.

Measurement sites, per `.cursor/rules/implementation-accuracy.mdc`:

- Activation site: slot index into `slot_kinds` for the recorded convention.
- Token position: `Pooling.LAST` reads the last REAL token (the last set bit of
  ``attention_mask``, i.e. index ``-1`` under left padding); `Pooling.MEAN` is the
  attention-mask-weighted mean over real tokens. ``last`` is the default because it is
  the position `evaluation.preference_prob_io` scores ``p_target`` at (``logits[0, -1]``
  on an unpadded batch of one).
- Intervention: none -- capture only, under ``torch.no_grad`` in eval mode.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import torch
import yaml
from loguru import logger
from transformers import PreTrainedTokenizerBase

from subliminal_learning.cka.extract_io import layer_activations, num_layer_slots
from subliminal_learning.cka.metrics import trait_coordinates
from subliminal_learning.steering.vectors import (
    diff_vector,
    load_vector,
    mean_activations,
    save_vector,
)

# Roles a model plays in the ladder; only `RANDOM` stacks are worth disk-caching
# (a random-init floor model is rung-independent, so its stack is reused at every
# rung, while a base checkpoint's stack is used once and dropped).
MODEL_ROLES: tuple[str, ...] = ("base", "random", "finetuned")


class SlotConvention(StrEnum):
    """Which 13-slot activation stack a measurement was taken from.

    Attributes:
        POST_BLOCK: `cka.extract_io.layer_activations` -- raw post-block residuals.
        HIDDEN_STATES: ``output_hidden_states=True`` -- last slot is post-final-layernorm.
    """

    POST_BLOCK = "post_block"
    HIDDEN_STATES = "hidden_states"


class Pooling(StrEnum):
    """How a ``[B, T, H]`` slot collapses to one ``[B, H]`` row per example."""

    LAST = "last"
    MEAN = "mean"


def slot_kinds(convention: SlotConvention, n_layers: int) -> tuple[str, ...]:
    """Per-slot labels for a convention on an ``n_layers``-block GPT-NeoX model.

    Args:
        convention: Which capture path produced the stack.
        n_layers: ``model.config.num_hidden_layers`` (12 on Pythia-160m).

    Returns:
        ``n_layers + 1`` labels. Both conventions share slots ``0..n_layers - 1``
        (``embed_out``, then ``block{k}_out``); they differ only at the last slot,
        which is ``block{n-1}_out`` under ``post_block`` and
        ``final_ln_block{n-1}_out`` under ``hidden_states``.

    Raises:
        ValueError: If `n_layers` < 1.
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1; got {n_layers}")
    head = ("embed_out", *(f"block{k}_out" for k in range(n_layers - 1)))
    if convention is SlotConvention.POST_BLOCK:
        return (*head, f"block{n_layers - 1}_out")
    return (*head, f"final_ln_block{n_layers - 1}_out")


def n_layers_of(model) -> int:  # noqa: ANN001 -- HF model or PeftModel wrapper
    """Decoder block count, read off the loaded model rather than assumed."""
    return num_layer_slots(model) - 1


@torch.no_grad()
def capture_slots(
    model,  # noqa: ANN001 -- GPTNeoXForCausalLM or a PeftModel wrapping one
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    convention: SlotConvention,
) -> list[torch.Tensor]:
    """One forward pass -> ``n_layers + 1`` tensors of ``[B, T, H]``.

    ``post_block`` delegates to `cka.extract_io.layer_activations` (the hooks are NOT
    reimplemented here); ``hidden_states`` takes the model's own
    ``output_hidden_states=True`` stack.

    Args:
        model: Loaded causal LM in eval mode.
        input_ids: ``[B, T]`` token ids.
        attention_mask: ``[B, T]`` 1 for real tokens, 0 for padding.
        convention: Which stack to capture.

    Returns:
        ``n_layers + 1`` tensors of ``[B, T, H]`` on the model's device.

    Raises:
        ValueError: If the captured slot count disagrees with `slot_kinds`.
    """
    if convention is SlotConvention.POST_BLOCK:
        slots = list(layer_activations(model, input_ids, attention_mask))
    else:
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        slots = list(out.hidden_states)
    expected = len(slot_kinds(convention, n_layers_of(model)))
    if len(slots) != expected:
        raise ValueError(
            f"{convention} capture returned {len(slots)} slots, expected {expected}"
        )
    return slots


def last_real_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """``[B]`` index of the LAST position where ``attention_mask`` is 1.

    Read as the index of the last set bit rather than as ``sum(-1) - 1``: the latter
    is the last real token only under RIGHT padding, and would land inside the pad
    run for every short row under the left padding `collect_activation_stack`
    requires. Under left padding this returns ``T - 1``, the column
    `steering.vectors.mean_activations` reads.

    Raises:
        ValueError: If any row is entirely padding.
    """
    mask = attention_mask.long()
    if int(mask.sum(dim=-1).min()) < 1:
        raise ValueError("an all-padding row has no last real token")
    width = mask.shape[-1]
    return width - 1 - mask.flip(dims=(-1,)).argmax(dim=-1)


def pool_slots(
    slots: Sequence[torch.Tensor], attention_mask: torch.Tensor, pooling: Pooling
) -> list[torch.Tensor]:
    """Collapse each ``[B, T, H]`` slot to ``[B, H]`` (fp32, CPU).

    Args:
        slots: Captured slots from `capture_slots`.
        attention_mask: ``[B, T]`` real-token mask.
        pooling: `Pooling.LAST` (last real token) or `Pooling.MEAN` (mask-weighted
            mean over real tokens).

    Returns:
        One ``[B, H]`` float32 CPU tensor per slot.
    """
    if pooling is Pooling.LAST:
        idx = last_real_index(attention_mask)
        rows = torch.arange(attention_mask.shape[0], device=attention_mask.device)
        return [s[rows, idx, :].float().cpu() for s in slots]
    w = attention_mask.to(slots[0].dtype).unsqueeze(-1)
    denom = w.sum(dim=1).clamp(min=1.0)
    return [((s * w).sum(dim=1) / denom).float().cpu() for s in slots]


@dataclass(frozen=True)
class ActivationStack:
    """Pooled per-example rows for one model, at every slot, with its provenance.

    Attributes:
        rows: ``[n_slots, n_examples, H]`` float32 CPU tensor.
        slot_convention: Which capture path produced it.
        pooling: How ``[B, T, H]`` was collapsed to ``[B, H]``.
        kinds: `slot_kinds` for `slot_convention`, one label per slot.
        model_ref: ``models.registry.model_ref`` INCLUDING the ``@revision`` -- without
            it, ``step1000`` and ``main`` collide in the cache.
        prompt_source: Identifier of the prompt/sequence pool the rows came from.
    """

    rows: torch.Tensor
    slot_convention: SlotConvention
    pooling: Pooling
    kinds: tuple[str, ...]
    model_ref: str
    prompt_source: str

    @property
    def n_slots(self) -> int:
        """Number of activation slots."""
        return int(self.rows.shape[0])

    @property
    def n_examples(self) -> int:
        """Number of pooled example rows."""
        return int(self.rows.shape[1])

    def slot(self, index: int) -> torch.Tensor:
        """``[n_examples, H]`` rows at one slot."""
        if not 0 <= index < self.n_slots:
            raise IndexError(f"slot {index} out of range [0, {self.n_slots})")
        return self.rows[index]


@torch.no_grad()
def collect_activation_stack(
    model,  # noqa: ANN001 -- GPTNeoXForCausalLM or a PeftModel wrapping one
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    device: str,
    *,
    convention: SlotConvention,
    pooling: Pooling,
    batch_size: int,
    model_ref: str,
    prompt_source: str,
) -> ActivationStack:
    """Pool every prompt into one ``[n_slots, n, H]`` stack for one model.

    Tokenization matches the repo convention used by `steering.vectors.mean_activations`
    (``add_special_tokens=False``, batch padding), so trait directions and trait
    coordinates see the same token stream.

    Args:
        model: Loaded causal LM in eval mode on `device`.
        tokenizer: Matching tokenizer with ``pad_token`` set; ``padding_side`` must be
            ``"left"`` for `Pooling.LAST`.
        prompts: Raw base-model prompts (no chat template).
        device: Torch device the model lives on.
        convention: Which 13-slot stack to capture.
        pooling: Row pooling.
        batch_size: Prompts per forward pass.
        model_ref: ``model_ref(spec, hub_org)`` with the revision.
        prompt_source: Prompt-pool identifier, recorded and used in the cache path.

    Returns:
        The `ActivationStack`.

    Raises:
        ValueError: On an empty prompt list.
    """
    if not prompts:
        raise ValueError("collect_activation_stack needs at least one prompt")
    if pooling is Pooling.LAST:
        assert tokenizer.padding_side == "left", (
            "Pooling.LAST reads the last REAL token; tokenizer.padding_side must be "
            f"'left', got {tokenizer.padding_side!r}"
        )
    per_slot: list[list[torch.Tensor]] = []
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start : start + batch_size])
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)
        slots = capture_slots(
            model, enc.input_ids, enc.attention_mask, convention=convention
        )
        pooled = pool_slots(slots, enc.attention_mask, pooling)
        if not per_slot:
            per_slot = [[] for _ in pooled]
        for i, row in enumerate(pooled):
            per_slot[i].append(row)
    rows = torch.stack([torch.cat(chunks, dim=0) for chunks in per_slot], dim=0)
    return ActivationStack(
        rows=rows,
        slot_convention=convention,
        pooling=pooling,
        kinds=slot_kinds(convention, rows.shape[0] - 1),
        model_ref=model_ref,
        prompt_source=prompt_source,
    )


@torch.no_grad()
def mean_slot_activations(
    model,  # noqa: ANN001 -- GPTNeoXForCausalLM or a PeftModel wrapping one
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    device: str,
    *,
    convention: SlotConvention,
    pooling: Pooling,
    batch_size: int,
) -> torch.Tensor:
    """``[n_slots, H]`` mean activation over the prompts, in either convention.

    `SlotConvention.HIDDEN_STATES` delegates to `steering.vectors.mean_activations`
    (the function every committed trait direction was built with) so the ladder's
    ``hidden_states`` directions are bit-for-bit the existing ones;
    `Pooling.LAST`/`Pooling.MEAN` map to its ``position="last"``/``"all"``.
    `SlotConvention.POST_BLOCK` runs the hook-based capture and averages the same
    pooled rows, which is the only way to get a trait direction in the RAW post-block
    basis that `cka.extract_io` measures activations in.

    STATED DEVIATION at `Pooling.MEAN`: the delegated ``position="all"`` path weights
    every real TOKEN equally (one global divisor), while the ``post_block`` path means
    per example first and then over examples, so unequal prompt lengths weight the two
    slightly differently. The two agree exactly at `Pooling.LAST` (one row per prompt),
    which is the ladder's default and the position `d_trait` is read at.

    Args:
        model: Loaded causal LM in eval mode on `device`.
        tokenizer: Matching tokenizer, ``padding_side='left'``.
        prompts: Raw base-model prompts.
        device: Torch device.
        convention: Which 13-slot stack to average.
        pooling: `Pooling.LAST` (last real token) or `Pooling.MEAN` (real-token mean).
        batch_size: Prompts per forward pass.

    Returns:
        ``[n_slots, H]`` float32 CPU means.
    """
    if convention is SlotConvention.HIDDEN_STATES:
        position = "last" if pooling is Pooling.LAST else "all"
        return mean_activations(
            model, tokenizer, prompts, device, batch_size, position=position
        )
    stack = collect_activation_stack(
        model,
        tokenizer,
        prompts,
        device,
        convention=convention,
        pooling=pooling,
        batch_size=batch_size,
        model_ref="<mean-only>",
        prompt_source="<mean-only>",
    )
    return stack.rows.mean(dim=1)


def require_same_convention(a: ActivationStack, b: ActivationStack) -> SlotConvention:
    """The pair's shared `SlotConvention`, raising if the two disagree.

    Slots 0..n-2 are identical across the two conventions, so an accidental mix
    produces a plausible number everywhere except the one slot the trait is strongest
    in. Pooling, prompt source and row count are checked at the same boundary because
    a metric over mismatched rows is equally silent.

    Args:
        a: One model's stack.
        b: The other model's stack.

    Returns:
        The common convention.

    Raises:
        ValueError: On any mismatch.
    """
    if a.slot_convention is not b.slot_convention:
        raise ValueError(
            "refusing to pair activation stacks from two slot conventions: "
            f"{a.model_ref} is {a.slot_convention}, {b.model_ref} is "
            f"{b.slot_convention}"
        )
    if a.pooling is not b.pooling:
        raise ValueError(
            f"pooling mismatch: {a.model_ref} is {a.pooling}, {b.model_ref} is "
            f"{b.pooling}"
        )
    if a.prompt_source != b.prompt_source:
        raise ValueError(
            f"prompt-source mismatch: {a.prompt_source!r} vs {b.prompt_source!r}"
        )
    if a.n_examples != b.n_examples or a.n_slots != b.n_slots:
        raise ValueError(
            f"stack shape mismatch: {tuple(a.rows.shape)} vs {tuple(b.rows.shape)}"
        )
    return a.slot_convention


def _slug(model_ref: str) -> str:
    """Filesystem-safe form of a repo ref; ``@revision`` is kept verbatim."""
    return model_ref.replace("/", "__")


def activation_cache_path(
    cache_root: Path,
    *,
    model_ref: str,
    prompt_source: str,
    convention: SlotConvention,
    pooling: Pooling,
    n_examples: int,
) -> Path:
    """Cache path for one model's stack, keyed by the REVISION-qualified `model_ref`.

    Angela's analogue keys on the bare repo id, which would collide ``@step1000`` with
    ``main`` and serve one rung's activations for another.

    Args:
        cache_root: Cache root (from `config.PathsConfig`, never hardcoded).
        model_ref: ``model_ref(spec, hub_org)`` INCLUDING ``@revision``.
        prompt_source: Prompt-pool identifier.
        convention: Slot convention of the cached stack.
        pooling: Row pooling of the cached stack.
        n_examples: Row count.

    Returns:
        ``{cache_root}/{prompt_source}/{model}/{convention}_{pooling}_n{n}.pt``.
    """
    return (
        Path(cache_root)
        / prompt_source
        / _slug(model_ref)
        / f"{convention}_{pooling}_n{n_examples}.pt"
    )


def should_cache(role: str, cache_roles: Sequence[str]) -> bool:
    """Whether a model in `role` gets its stack written to disk.

    Args:
        role: One of `MODEL_ROLES`.
        cache_roles: Roles the config opts into caching.

    Returns:
        True if `role` is in `cache_roles`.

    Raises:
        ValueError: On an unknown role.
    """
    if role not in MODEL_ROLES:
        raise ValueError(f"unknown model role {role!r}; known: {MODEL_ROLES}")
    return role in tuple(cache_roles)


def save_activation_stack(path: Path, stack: ActivationStack) -> None:
    """Write a stack plus its convention/pooling/ref stamp atomically."""
    save_vector(
        path,
        {"rows": stack.rows},
        {
            "slot_convention": str(stack.slot_convention),
            "pooling": str(stack.pooling),
            "kinds": list(stack.kinds),
            "model_ref": stack.model_ref,
            "prompt_source": stack.prompt_source,
        },
    )


def load_activation_stack(
    path: Path,
    *,
    expect_convention: SlotConvention,
    expect_pooling: Pooling,
    expect_model_ref: str,
) -> ActivationStack:
    """Load a cached stack, raising if its stamp disagrees with what was asked for.

    A cache hit on the wrong convention/pooling/revision is the exact silent failure
    this module exists to prevent, so the stamp is a hard gate rather than a log line.

    Raises:
        ValueError: If any stamped field differs from the expected one.
    """
    payload = load_vector(path)
    meta = payload["meta"]
    got = (meta["slot_convention"], meta["pooling"], meta["model_ref"])
    want = (str(expect_convention), str(expect_pooling), expect_model_ref)
    if got != want:
        raise ValueError(
            f"cached activation stack at {path} is stamped {got}, expected {want}"
        )
    return ActivationStack(
        rows=payload["rows"],
        slot_convention=SlotConvention(meta["slot_convention"]),
        pooling=Pooling(meta["pooling"]),
        kinds=tuple(meta["kinds"]),
        model_ref=meta["model_ref"],
        prompt_source=meta["prompt_source"],
    )


@dataclass(frozen=True)
class TraitVector:
    """A ``diff_vector`` trait direction with the convention it was extracted under.

    Attributes:
        raw: ``[n_slots, H]`` teacher-minus-base mean shift.
        unit: ``[n_slots, H]`` per-slot L2-normalised direction.
        norm: ``[n_slots]`` per-slot L2 norm of `raw`.
        slot_convention: Convention of the activation stack it was built from.
        pooling: Pooling used for the two means.
        model_label: ``{init}@{revision}`` of the base the teacher was finetuned from.
        animal: Target trait animal.
        ftseed: Teacher finetune seed.
    """

    raw: torch.Tensor
    unit: torch.Tensor
    norm: torch.Tensor
    slot_convention: SlotConvention
    pooling: Pooling
    model_label: str
    animal: str
    ftseed: int


def trait_vector_path(
    similarity_root: Path,
    *,
    animal: str,
    model_label: str,
    ftseed: int,
    pooling: Pooling,
    convention: SlotConvention,
) -> Path:
    """``trait_vectors/{animal}/{label}_tfs{N}_{pooling}_{convention}.pt`` (plan §8).

    The convention is in the FILENAME so a ``post_block`` and a ``hidden_states``
    direction for the same teacher can coexist and can never overwrite each other.
    """
    return (
        Path(similarity_root)
        / "trait_vectors"
        / animal
        / f"{model_label}_tfs{ftseed}_{pooling}_{convention}.pt"
    )


def build_trait_vector(
    mean_teacher: torch.Tensor,
    mean_base: torch.Tensor,
    *,
    convention: SlotConvention,
    pooling: Pooling,
    model_label: str,
    animal: str,
    ftseed: int,
) -> TraitVector:
    """Compose a `TraitVector` from two ``[n_slots, H]`` mean-activation stacks.

    The arithmetic is `steering.vectors.diff_vector` (teacher minus base), unchanged;
    this only attaches the provenance the ladder joins on.
    """
    vec = diff_vector(mean_teacher, mean_base)
    return TraitVector(
        raw=vec["raw"],
        unit=vec["unit"],
        norm=vec["norm"],
        slot_convention=convention,
        pooling=pooling,
        model_label=model_label,
        animal=animal,
        ftseed=ftseed,
    )


def save_trait_vector(path: Path, trait: TraitVector) -> None:
    """Write a trait vector plus its convention stamp atomically."""
    save_vector(
        path,
        {"raw": trait.raw, "unit": trait.unit, "norm": trait.norm},
        {
            "slot_convention": str(trait.slot_convention),
            "pooling": str(trait.pooling),
            "model_label": trait.model_label,
            "animal": trait.animal,
            "ftseed": trait.ftseed,
        },
    )
    logger.info("Wrote trait vector -> {}", path)


def load_trait_vector(path: Path) -> TraitVector:
    """Load a trait vector saved by `save_trait_vector`."""
    payload = load_vector(path)
    meta = payload["meta"]
    return TraitVector(
        raw=payload["raw"],
        unit=payload["unit"],
        norm=payload["norm"],
        slot_convention=SlotConvention(meta["slot_convention"]),
        pooling=Pooling(meta["pooling"]),
        model_label=meta["model_label"],
        animal=meta["animal"],
        ftseed=int(meta["ftseed"]),
    )


def trait_coordinates_at_slot(
    stack: ActivationStack,
    slot: int,
    trait: TraitVector,
    tokenizer: PreTrainedTokenizerBase,
) -> torch.Tensor:
    """``[n_examples]`` trait coordinates of one model at one slot.

    Guards the two silent-failure boundaries before delegating the projection to
    `cka.metrics.trait_coordinates`:

    1. The stack's convention must equal the one the DIRECTION was extracted under.
       Slots 0..n-2 agree across conventions, so a mismatch shows up only at the last
       slot -- the slot the trait is usually strongest in.
    2. ``tokenizer.padding_side == "left"`` for `Pooling.LAST`, matching the assertion
       in `steering.vectors.mean_activations`, so ``position="last"`` reads the final
       REAL token on both sides.

    Args:
        stack: The model's pooled activations.
        slot: Slot index into `ActivationStack.kinds`.
        trait: Direction in THIS model's basis (`steering.vectors.diff_vector` row).
        tokenizer: The tokenizer the stack was collected with.

    Returns:
        ``[n_examples]`` float64 coordinates.

    Raises:
        ValueError: On a convention or pooling mismatch.
    """
    if stack.slot_convention is not trait.slot_convention:
        raise ValueError(
            "trait direction was extracted under "
            f"{trait.slot_convention} but the activation stack is "
            f"{stack.slot_convention}; d_trait would be wrong at the final slot"
        )
    if stack.pooling is not trait.pooling:
        raise ValueError(
            f"pooling mismatch: stack {stack.pooling} vs trait {trait.pooling}"
        )
    if stack.pooling is Pooling.LAST:
        assert tokenizer.padding_side == "left", (
            "trait coordinates at position 'last' need left padding to read the final "
            f"real token; got padding_side={tokenizer.padding_side!r}"
        )
    return trait_coordinates(stack.slot(slot), trait.unit[slot])


@dataclass(frozen=True)
class ProbeBlockConfig:
    """The `probe:` block of ``configs/experiment/ckpt_ladder_geometry.yaml``.

    Attributes:
        grounding: Run the P1c UPOS grounding probe.
        treebank: Treebank id; ``ud_english_ewt`` is the only wired source.
        treebank_release: UD release tag (a git tag of the UD repo, e.g. ``r2.18``).
        treebank_split: ``train`` / ``dev`` / ``test``.
        allow_download: Fetch the ``.conllu`` if it is not already under
            `dataset_cache_root`; False forces the offline fallback.
        fallback_task: Label set used when UD acquisition is blocked.
        max_tokens: Cap on probe rows per (model, slot).
        trait: Run the P5 trait probe.
        trait_prompt_set: `evaluation.prompt_sets` name for the trait-probe pool.
        seed: Seeds the probe train/test split and the online-code ordering.
    """

    grounding: bool
    treebank: str
    treebank_release: str
    treebank_split: str
    allow_download: bool
    fallback_task: str
    max_tokens: int
    trait: bool
    trait_prompt_set: str
    seed: int


@dataclass(frozen=True)
class GeometryLadderConfig:
    """Frozen config for the geometry ladder (plan §6.7).

    Every field is read from YAML through `load_geometry_ladder_config`; nothing here
    has a module-level default in Python, and no path is hardcoded (roots come from
    `config.PathsConfig`).

    Attributes:
        base_revisions: Ladder-G rungs, e.g. ``("step0", ..., "step143000")``.
        slot_conventions: Conventions to record (YAML key ``slot_convention``, scalar
            or list). Recording both is what avoids re-baselining Angela's numbers.
        pooling: Row pooling.
        prompt_source: Prompt-pool identifier for the similarity rows.
        ssa_k: Subspace dimension for `cka.metrics.subspace_angles_topk`.
        n_examples: Rows per model per slot.
        batch_size: Prompts per forward pass.
        n_repeats: Minibatch-CKA repeats (drives ``repeat_std``).
        norm_exclusion_frac: Top-norm fraction dropped in the robustness re-score.
        random_init_seeds: Seeds of the random-init similarity floor.
        cache_roles: Roles whose activation stacks are written to disk.
        probe: The probe block.
    """

    base_revisions: tuple[str, ...]
    slot_conventions: tuple[SlotConvention, ...]
    pooling: Pooling
    prompt_source: str
    ssa_k: int
    n_examples: int
    batch_size: int
    n_repeats: int
    norm_exclusion_frac: float
    random_init_seeds: tuple[int, ...]
    cache_roles: tuple[str, ...]
    probe: ProbeBlockConfig


def load_geometry_ladder_config(path: str | Path) -> GeometryLadderConfig:
    """Load ``configs/experiment/ckpt_ladder_geometry.yaml`` into its frozen dataclass.

    Args:
        path: YAML path.

    Returns:
        The parsed `GeometryLadderConfig`.

    Raises:
        ValueError: On an unknown ``cache_roles`` entry or an empty rung list (either
            would silently produce an empty or mis-scoped ladder).
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    conventions = raw.get("slot_convention", "post_block")
    if isinstance(conventions, str):
        conventions = [conventions]
    probe = raw.get("probe", {})
    cache_roles = tuple(raw.get("cache_roles", ()))
    unknown = [r for r in cache_roles if r not in MODEL_ROLES]
    if unknown:
        raise ValueError(f"unknown cache_roles {unknown}; known: {MODEL_ROLES}")
    revisions = tuple(str(r) for r in raw.get("base_revisions", ()))
    if not revisions:
        raise ValueError(f"{path} has an empty 'base_revisions' ladder")
    return GeometryLadderConfig(
        base_revisions=revisions,
        slot_conventions=tuple(SlotConvention(c) for c in conventions),
        pooling=Pooling(raw.get("pooling", "last")),
        prompt_source=str(raw["prompt_source"]),
        ssa_k=int(raw["ssa_k"]),
        n_examples=int(raw["n_examples"]),
        batch_size=int(raw["batch_size"]),
        n_repeats=int(raw["n_repeats"]),
        norm_exclusion_frac=float(raw["norm_exclusion_frac"]),
        random_init_seeds=tuple(int(s) for s in raw.get("random_init_seeds", ())),
        cache_roles=cache_roles,
        probe=ProbeBlockConfig(
            grounding=bool(probe.get("grounding", True)),
            treebank=str(probe.get("treebank", "ud_english_ewt")),
            treebank_release=str(probe.get("treebank_release", "r2.18")),
            treebank_split=str(probe.get("treebank_split", "dev")),
            allow_download=bool(probe.get("allow_download", True)),
            fallback_task=str(probe.get("fallback_task", "surface_category")),
            max_tokens=int(probe.get("max_tokens", 20000)),
            trait=bool(probe.get("trait", True)),
            trait_prompt_set=str(probe.get("trait_prompt_set", "favorite")),
            seed=int(probe.get("seed", 0)),
        ),
    )
