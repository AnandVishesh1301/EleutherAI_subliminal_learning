"""Empirical activation similarity (EAS) — pure math + resume/merge logic (no I/O).

EAS (Blank et al., *Subliminal Learning Is Steering Vector Distillation*, arXiv:2606.00995)
asks how much a distilled student's activation-shift direction lines up with the teacher's.
For a fixed lineage it is the per-layer cosine similarity between the teacher's shift vector
(``mean_teacher - mean_base``) and each student's shift vector (``mean_student - mean_base``),
both read on the SAME prompts. This repo substitutes a fine-tuned-LoRA teacher for Blank's
system-prompt teacher (there is no system prompt on a base model); the direction is otherwise
the same teacher-vs-base contrast used by `subliminal_learning.steering.vectors`.

Everything here is a pure function of tensors / plain dicts, so the math and the resumable
JSON-curve bookkeeping are testable without a GPU or disk. The model/disk orchestration lives
in `pipelines.eas_pipeline`.

Resume/merge preserves work. A narrowed re-run (fewer ``--epochs``/``--sources``, or a
transiently missing run dir) must NEVER delete curve entries it did not ask for: when the
stored doc's ``meta`` still matches, `plan_reuse` carries over EVERY stored entry outside the
current work set verbatim (see `ReusePlan.preserved_entries`). Deletion happens only on a
genuine invalidation — ``force`` or meta drift — where nothing is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# An identity for one student checkpoint in the resume plan: its
# ``(source, n_epochs, student_ftseed)`` entry key plus the on-disk adapter content hash.
EntryKey = tuple[str, int, int]
WorkItem = tuple[EntryKey, str | None]


def eas_per_layer(
    v_student_raw: torch.Tensor, v_teacher_raw: torch.Tensor
) -> torch.Tensor:
    """Per-slot cosine similarity between two ``[n_slots, H]`` shift vectors.

    Returns a ``[n_slots]`` tensor (``dim=-1`` cosine), aligned with the ``raw_blocks_v2``
    slot layout: index 0 is the embedding slot and index ``i`` (i >= 1) the raw output of
    block ``i - 1``, final block included (see the layer-bookkeeping docstring in
    `subliminal_learning.steering.vectors`).

    Raises:
        ValueError: if the two shift vectors' shapes differ — the single guard point for a
            cross-base lineage whose base A and base B could have a different depth/width.
    """
    if v_student_raw.shape != v_teacher_raw.shape:
        raise ValueError(
            f"shift-vector shape mismatch: v_student {tuple(v_student_raw.shape)} vs v_teacher "
            f"{tuple(v_teacher_raw.shape)} (cross-base lineages must share depth/width)"
        )
    return F.cosine_similarity(v_student_raw, v_teacher_raw, dim=-1)


def eas_mean(per_layer: torch.Tensor) -> float:
    """Mean EAS over slots ``[1:]`` — the raw-block convention (this repo's headline).

    Only the embedding slot (0) is excluded: under the ``raw_blocks_v2`` extraction basis
    every remaining slot is a raw block output, final block included, so the block-only mean
    is ``per_layer[1:]``. (The retired ``blocks_v1`` convention dropped BOTH ends because the
    old extraction stack's last slot was the post-final-layernorm state — a generic positive
    cosine of ≈ +0.12 to +0.21 even for control students — not a raw block. That slot no
    longer exists.) Numerically this now coincides with Blank's ``per_layer[1:].mean()``, but
    over a different final slot: the last block's RAW output rather than its post-LN image.
    The full per-slot curve is still stored (see `build_entry`), so any other re-aggregation
    stays possible downstream.

    Raises:
        ValueError: if `per_layer` has fewer than 2 slots, so the ``[1:]`` block window is
            empty and the mean would be NaN.
    """
    if per_layer.shape[-1] < 2:
        raise ValueError(
            f"eas_mean needs at least 2 slots to drop the embedding slot; "
            f"got {per_layer.shape[-1]}"
        )
    return per_layer[1:].mean().item()


def build_entry(
    *,
    source: str,
    n_epochs: int,
    student_ftseed: int,
    student_param_tag: str,
    run_dir: str,
    adapter_id: str | None,
    per_layer: torch.Tensor,
    v_student_norm: torch.Tensor,
) -> dict:
    """Build the JSON-serializable per-checkpoint EAS record from the cosine tensor + norms.

    Args:
        source: Student source (``teacher`` / ``base`` / ``rng``).
        n_epochs: Student training epochs.
        student_ftseed: Student fine-tuning seed.
        student_param_tag: The student's non-seed HP tag (path segment).
        run_dir: The student leaf run dir (recorded for provenance).
        adapter_id: Content hash of the student's adapter (resume gate).
        per_layer: ``[n_slots]`` per-slot cosine vs the teacher shift vector.
        v_student_norm: ``[n_slots]`` per-slot L2 norm of the student shift vector.

    Returns:
        A dict with the entry's identity, ``eas_mean`` (block-only convention), and the
        per-slot curves.
    """
    return {
        "source": source,
        "n_epochs": n_epochs,
        "student_ftseed": student_ftseed,
        "student_param_tag": student_param_tag,
        "run_dir": run_dir,
        "adapter_id": adapter_id,
        "eas_mean": eas_mean(per_layer),
        "eas_per_layer": per_layer.tolist(),
        "v_student_norm_per_layer": v_student_norm.tolist(),
    }


@dataclass(frozen=True)
class ReusePlan:
    """The resume decision for one EAS pass.

    Attributes:
        reused_entries: Stored entry dicts (whose key IS in the current work set) to carry over
            verbatim because the adapter + meta are unchanged.
        preserved_entries: Stored entry dicts whose key is NOT in the current work set. A
            narrowed re-run (fewer epochs/sources, or a transiently missing run dir) keeps these
            verbatim so the rewritten doc never silently drops previously computed curves. Empty
            on ``force`` or meta drift. Disjoint from ``reused_entries`` and ``to_compute``.
        to_compute: ``(entry_key, adapter_id)`` items whose vector must be recomputed.
        meta_drifted: True when a stored document existed but its ``meta`` differs from the
            current request on some key — the caller should warn (every item is recomputed).
    """

    reused_entries: list[dict]
    preserved_entries: list[dict]
    to_compute: list[WorkItem]
    meta_drifted: bool


def plan_reuse(
    stored_doc: dict | None,
    current_meta: dict,
    work_items: list[WorkItem],
    force: bool = False,
) -> ReusePlan:
    """Decide which EAS entries can be reused, preserved, or must be recomputed.

    Rules (see the pipeline schema): with ``force`` everything is recomputed and nothing is
    preserved. If there is no stored document nothing can be reused or preserved. If the stored
    ``meta`` differs from ``current_meta`` on ANY key, NOTHING is reusable or preserved
    (``meta_drifted=True`` so the caller can warn) — a changed prompt set / position / teacher
    adapter invalidates every cosine. Otherwise, for a stored entry whose ``meta`` still matches:

    - if its ``(source, n_epochs, student_ftseed)`` key is in ``work_items`` it is REUSED iff its
      ``adapter_id`` equals the current on-disk adapter identity (an in-place retrain at the same
      path is thereby detected and recomputed);
    - if its key is NOT in ``work_items`` it is PRESERVED verbatim, so a narrowed re-run (fewer
      epochs/sources, or a transiently missing run dir) never deletes a curve it did not request.

    Reused, preserved, and recomputed keys are disjoint.

    Args:
        stored_doc: The previously persisted document, or None.
        current_meta: The meta dict this pass would write (JSON-normalized by the caller).
        work_items: The current ``(entry_key, adapter_id)`` items to satisfy.
        force: Recompute everything regardless.

    Returns:
        A `ReusePlan`.
    """
    if force or stored_doc is None:
        return ReusePlan([], [], list(work_items), meta_drifted=False)
    if stored_doc.get("meta") != current_meta:
        return ReusePlan([], [], list(work_items), meta_drifted=True)
    stored_by_key: dict[EntryKey, dict] = {
        (e["source"], e["n_epochs"], e["student_ftseed"]): e
        for e in stored_doc.get("entries", [])
    }
    work_keys = {key for key, _ in work_items}
    reused: list[dict] = []
    to_compute: list[WorkItem] = []
    for key, adapter_id in work_items:
        stored = stored_by_key.get(key)
        if stored is not None and stored.get("adapter_id") == adapter_id:
            reused.append(stored)
        else:
            to_compute.append((key, adapter_id))
    preserved = [e for key, e in stored_by_key.items() if key not in work_keys]
    return ReusePlan(reused, preserved, to_compute, meta_drifted=False)
