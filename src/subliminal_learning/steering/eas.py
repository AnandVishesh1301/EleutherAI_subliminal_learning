"""Empirical activation similarity (EAS) — pure math + currency logic (no I/O).

EAS (Blank et al., *Subliminal Learning Is Steering Vector Distillation*, arXiv:2606.00995)
asks how much a distilled student's activation-shift direction lines up with the teacher's:
the per-layer cosine similarity between the teacher's shift vector (``mean_teacher -
mean_base``) and a student's (``mean_student - mean_base``), both read on the SAME prompts.
This repo substitutes a fine-tuned-LoRA teacher for Blank's system-prompt teacher (there is
no system prompt on a base model); the direction is otherwise the same teacher-vs-base
contrast used by `subliminal_learning.steering.vectors`.

The unit of measurement is ONE STUDENT RUN DIR with one point per checkpoint (2026-09-10;
the retired design wrote one document per teacher lineage, which could not address an
arbitrary lr/rank/alpha/optimizer cell or an intermediate checkpoint). The document schema,
the on-disk locations and the model orchestration live in `pipelines.eas_pipeline`; this
module holds the tensor math and `eas_doc_status`, the ONE currency predicate both the
pipeline and the sweep's cheap gate call.

Everything here is a pure function of tensors / plain dicts, so it is testable without a
GPU or disk.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F

#: Schema version of the per-run EAS document (`pipelines.eas_pipeline`).
SCHEMA_VERSION = 1

#: The slot convention `eas_mean` aggregates under. Stamped into every document and checked
#: by `eas_doc_status`, so a future change to the mean invalidates stored docs rather than
#: mixing conventions. Equal to `steering.vectors.ACTIVATION_BASIS` today because the mean
#: follows the extraction basis; they are separate stamps so either can move alone. This is
#: the ONLY definition of the string: no other module may spell it out.
EAS_MEAN_CONVENTION = "raw_blocks_v2"

#: Identity keys of a stored document, compared against `expected_meta` verbatim. `basis`
#: and `eas_mean_convention` are schema stamps; the rest describe what was measured.
DOC_META_KEYS = (
    "schema_version",
    "basis",
    "eas_mean_convention",
    "position",
    "n_prompts",
    "prompt_sha256",
)


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
    The full per-slot curve is still stored in every document point, so a re-aggregation
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


@dataclass(frozen=True)
class DocPlan:
    """What one stored EAS document still satisfies, and what must be (re)computed.

    Attributes:
        invalid: The stored document is unusable (absent, drifted meta, different run) —
            every checkpoint is recomputed and nothing is carried over.
        reuse: Stored point dicts carried over verbatim (checkpoint still on disk, adapter
            unchanged, every requested reference already present and fresh).
        cosine_only: ``ckpt_rel`` values whose student vector is available from a current
            sidecar, so a newly requested reference costs pure cosine math and NO student
            forward pass.
        recompute: ``ckpt_rel`` values whose student activation vector must be recomputed.
        dropped: ``ckpt_rel`` values stored in the document whose checkpoint no longer
            exists on disk; the run dir is the unit, so these points are deleted.
        stale_refs: Reference keys whose stored ``teacher_adapter_id`` disagrees with the
            teacher checkpoint on disk (an in-place teacher retrain); their cosines are
            recomputed for every point.
        reason: Why the document is invalid, for logging (None when it is not).
    """

    invalid: bool
    reuse: tuple[dict, ...]
    cosine_only: tuple[str, ...]
    recompute: tuple[str, ...]
    dropped: tuple[str, ...]
    stale_refs: tuple[str, ...]
    reason: str | None = None

    @property
    def is_current(self) -> bool:
        """True when the stored document already answers the request byte-for-byte."""
        return not (
            self.invalid or self.cosine_only or self.recompute or self.dropped
        )

    @property
    def needs_student_pass(self) -> bool:
        """True when at least one checkpoint must be forwarded through the model."""
        return bool(self.recompute)


def eas_doc_status(
    doc: dict | None,
    *,
    expected_meta: dict,
    checkpoints: list[tuple[str, str | None]],
    references: dict[str, str | None],
    sidecar_ckpts: frozenset[str] | set[str] | tuple[str, ...] = (),
    manifest_ref: dict | None = None,
    force: bool = False,
    cheap: bool = False,
) -> DocPlan:
    """The ONE currency predicate for a per-run EAS document.

    Decides, for one student run dir:

    - the document is INVALID (rewrite everything) on ``force``, when it is absent, when any
      `DOC_META_KEYS` field drifts (basis / convention / position / n_prompts / prompt sha),
      or when its ``student.manifest_ref`` differs from the run's current one (the run was
      retrained under the same path);
    - otherwise a stored point is REUSED iff its checkpoint is still on disk with an
      unchanged ``adapter_id`` AND it already carries every requested reference, none of
      them stale;
    - a point missing only a reference is COSINE_ONLY when its student vector is in a
      current sidecar (``sidecar_ckpts``), else it is RECOMPUTEd;
    - a stored point whose checkpoint no longer exists on disk is DROPPED (unlike the
      retired lineage document, whose unit spanned run dirs, a run dir owns its points).

    Args:
        doc: The stored document, or None.
        expected_meta: The identity this pass would stamp (`DOC_META_KEYS`).
        checkpoints: ``(ckpt_rel, adapter_id)`` on disk, in the order they will be written.
        references: ``{ref_key: teacher_adapter_id}`` requested for this pass.
        sidecar_ckpts: ``ckpt_rel`` values held by a CURRENT vector sidecar.
        manifest_ref: The run's `utils.run_io.manifest_ref`, compared with the stored one.
        force: Recompute everything.
        cheap: Sweep-gate mode. An entry of `checkpoints` whose ``adapter_id`` is None is
            not adapter-checked (only its presence in the document is), so the gate can
            skip hashing intermediate checkpoints and key on ``ckpt_rel`` + the endpoint
            adapter id. Cheap mode can therefore miss an in-place-retrained intermediate
            checkpoint; the pipeline itself always passes real ids, and `--force` rebuilds.

    Returns:
        A `DocPlan`.
    """
    ckpt_order = [rel for rel, _ in checkpoints]
    if force or doc is None:
        return DocPlan(
            invalid=True, reuse=(), cosine_only=(), recompute=tuple(ckpt_order),
            dropped=(), stale_refs=tuple(references),
            reason="force" if force else "no stored document",
        )
    for key in DOC_META_KEYS:
        if doc.get(key) != expected_meta.get(key):
            return DocPlan(
                invalid=True, reuse=(), cosine_only=(), recompute=tuple(ckpt_order),
                dropped=(), stale_refs=tuple(references),
                reason=f"{key} {doc.get(key)!r} != {expected_meta.get(key)!r}",
            )
    stored_student = doc.get("student") or {}
    if stored_student.get("manifest_ref") != manifest_ref:
        return DocPlan(
            invalid=True, reuse=(), cosine_only=(), recompute=tuple(ckpt_order),
            dropped=(), stale_refs=tuple(references),
            reason="train_manifest_ref changed (the run was retrained in place)",
        )

    stored_refs = doc.get("references") or {}
    stale = tuple(sorted(
        k for k, aid in references.items()
        if k in stored_refs and stored_refs[k].get("teacher_adapter_id") != aid
    ))
    needed = set(references)
    stored_points = {p.get("ckpt_rel"): p for p in doc.get("points") or []}
    sidecar = set(sidecar_ckpts)
    reuse: list[dict] = []
    cosine_only: list[str] = []
    recompute: list[str] = []
    for rel, adapter_id in checkpoints:
        point = stored_points.get(rel)
        adapter_checked = adapter_id is not None or not cheap
        if point is None or (
            adapter_checked and adapter_id is not None
            and point.get("adapter_id") != adapter_id
        ):
            recompute.append(rel)
            continue
        if point.get("nonfinite"):
            # Never scored, and never will be under this adapter: carry the null point over.
            reuse.append(point)
            continue
        usable = set(point.get("eas") or {}) - set(stale)
        if needed <= usable:
            reuse.append(point)
        elif rel in sidecar:
            cosine_only.append(rel)
        else:
            recompute.append(rel)
    dropped = tuple(rel for rel in stored_points if rel not in set(ckpt_order))
    return DocPlan(
        invalid=False, reuse=tuple(reuse), cosine_only=tuple(cosine_only),
        recompute=tuple(recompute), dropped=dropped, stale_refs=stale,
    )


def eas_main_doc_status(doc: dict | None, *, references: dict[str, str | None], **kwargs):
    """`eas_doc_status` for the MAIN document, which carries ONLY the run's own teacher.

    A stored main document naming any other reference was written by the retired one-file
    design (extra references now live in ``eas_refs_{position}.json``): it is INVALID, and
    its scored points are re-derived COSINE-ONLY wherever the vector sidecar still holds the
    student vectors, so the migration costs no student forward pass.

    Args:
        doc: The stored main document, or None.
        references: ``{ref_key: teacher_adapter_id}`` — the own teacher alone.
        **kwargs: Passed through to `eas_doc_status`.

    Returns:
        A `DocPlan`.
    """
    plan = eas_doc_status(doc, references=references, **kwargs)
    foreign = sorted(set((doc or {}).get("references") or {}) - set(references))
    if not foreign:
        return plan
    sidecar = set(kwargs.get("sidecar_ckpts") or ())
    keep = tuple(p for p in plan.reuse if p.get("nonfinite"))
    rebuild = [p.get("ckpt_rel") for p in plan.reuse if not p.get("nonfinite")]
    return replace(
        plan,
        invalid=True,
        reuse=keep,
        cosine_only=tuple(list(plan.cosine_only) + [r for r in rebuild if r in sidecar]),
        recompute=tuple(list(plan.recompute) + [r for r in rebuild if r not in sidecar]),
        reason=f"main document carries reference(s) other than the own teacher: {foreign}",
    )
