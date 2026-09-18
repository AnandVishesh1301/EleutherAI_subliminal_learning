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

from dataclasses import dataclass, field, replace

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
            exists on disk; the run dir is the unit, so these points are deleted. A rel
            whose STEP is still covered by a retained point (the endpoint's former
            duplicate `checkpoint-N` dir), or whose step the ladder manifest lists (the
            checkpoint was THINNED), is NOT dropped -- the document still describes the
            ladder, and a thinned point is reused verbatim.
        stale_refs: Reference keys whose stored ``teacher_adapter_id`` disagrees with the
            teacher checkpoint on disk (an in-place teacher retrain); their cosines are
            recomputed for every point.
        unscorable: ``{ckpt_rel: (ref_key, ...)}`` for THINNED points that lack a requested
            reference and have no sidecar vector either: the checkpoint is gone, so that
            reference can never be measured there. The point is still reused (annotated
            ``unscorable_refs``), the document is NOT `is_current` for those references, and
            a sweep must record the unit as done rather than re-enqueue it forever.
        reason: Why the document is invalid, for logging (None when it is not).
    """

    invalid: bool
    reuse: tuple[dict, ...]
    cosine_only: tuple[str, ...]
    recompute: tuple[str, ...]
    dropped: tuple[str, ...]
    stale_refs: tuple[str, ...]
    reason: str | None = None
    unscorable: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def is_current(self) -> bool:
        """True when the stored document already answers the request byte-for-byte."""
        return not (
            self.invalid or self.cosine_only or self.recompute or self.dropped
            or self.unscorable
        )

    @property
    def only_unscorable(self) -> bool:
        """True when the ONLY thing left undone is a reference no checkpoint can supply.

        The terminal state of a thinned run: a sweep logs it once and does not re-enqueue
        the unit (nothing a rerun could compute).
        """
        return bool(self.unscorable) and not (
            self.invalid or self.cosine_only or self.recompute or self.dropped
            or self.stale_refs
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
    ladder_steps: frozenset[int] | set[int] | tuple[int, ...] = (),
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
      retired lineage document, whose unit spanned run dirs, a run dir owns its points) --
      UNLESS its step is in the run's ladder manifest (`ladder_steps`), i.e. the checkpoint
      was THINNED, not lost: that point is reused so the rewrite keeps it -- but only if it
      already carries every requested reference; otherwise it is COSINE_ONLY when the sidecar
      still holds its student vector, and UNSCORABLE (reused, annotated ``unscorable_refs``)
      when it does not, the checkpoint being gone.

    Args:
        doc: The stored document, or None.
        expected_meta: The identity this pass would stamp (`DOC_META_KEYS`).
        checkpoints: ``(ckpt_rel, adapter_id)`` on disk, in the order they will be written.
        references: ``{ref_key: teacher_adapter_id}`` requested for this pass.
        sidecar_ckpts: ``ckpt_rel`` values held by a CURRENT vector sidecar.
        ladder_steps: every step the run's ladder manifest says EVER existed
            (`training.ckpt_ladder.ladder_steps`); a stored point at one of these steps
            whose checkpoint dir is gone was thinned away, not lost.
        manifest_ref: The run's `utils.run_io.manifest_ref`, compared with the stored one.
        force: Recompute every ON-DISK checkpoint (thinned points are still carried, unless
            the stored identity drifted).
        cheap: Sweep-gate mode. An entry of `checkpoints` whose ``adapter_id`` is None is
            not adapter-checked (only its presence in the document is), so the gate can
            skip hashing intermediate checkpoints and key on ``ckpt_rel`` + the endpoint
            adapter id. Cheap mode can therefore miss an in-place-retrained intermediate
            checkpoint; the pipeline itself always passes real ids, and `--force` rebuilds.

    Returns:
        A `DocPlan`.
    """
    ckpt_order = [rel for rel, _ in checkpoints]
    if doc is None:
        return DocPlan(
            invalid=True, reuse=(), cosine_only=(), recompute=tuple(ckpt_order),
            dropped=(), stale_refs=tuple(references), reason="no stored document",
        )
    if force:
        # Force recomputes every ON-DISK checkpoint, but a THINNED point has no checkpoint
        # to recompute from: it is carried through the same reuse / cosine-only /
        # unscorable logic as the non-force path, so a forced rerun never shrinks the
        # document. Nothing is carried when the identity drifted (prompt set, or an
        # in-place retrain): those measurements describe a different unit, so dropping
        # them is correct, and `supersede_if_shrinking` on write preserves the old file.
        same_identity = (
            all(doc.get(k) == expected_meta.get(k) for k in DOC_META_KEYS)
            and (doc.get("student") or {}).get("manifest_ref") == manifest_ref
        )
        reuse, cosine_only, unscorable = (), (), {}
        if same_identity:
            reuse, cosine_only, _, unscorable = _thinned_plan(
                doc, live=set(ckpt_order), references=references,
                sidecar_ckpts=sidecar_ckpts, ladder_steps=ladder_steps,
                stale=_stale_refs(doc, references),
            )
        return DocPlan(
            invalid=True, reuse=reuse, cosine_only=cosine_only,
            recompute=tuple(ckpt_order), dropped=(), stale_refs=tuple(references),
            unscorable=unscorable, reason="force",
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

    stale = _stale_refs(doc, references)
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
    # A stored point whose rel is gone from disk but whose STEP another retained point
    # still carries is a duplicate, not a lost point: that is exactly the legacy
    # `checkpoint-N` dir which duplicated the endpoint adapter and is no longer written
    # (see `students.early_stop.SkipFinalStepSave`). Treating it as dropped would
    # invalidate every stored EAS document on the shared stash for no new numbers.
    # A stored point whose checkpoint was THINNED off disk (its step is in the run's
    # ladder manifest) is likewise not dropped: it is carried into `reuse`, so the document
    # the pipeline rebuilds from the plan keeps the full trajectory.
    thinned_reuse, thinned_cosine, dropped, unscorable = _thinned_plan(
        doc, live=set(ckpt_order), references=references, sidecar_ckpts=sidecar_ckpts,
        ladder_steps=ladder_steps, stale=stale,
    )
    return DocPlan(
        invalid=False, reuse=tuple(reuse) + thinned_reuse,
        cosine_only=tuple(cosine_only) + thinned_cosine,
        recompute=tuple(recompute), dropped=dropped, stale_refs=stale,
        unscorable=unscorable,
    )


def _stale_refs(doc: dict, references: dict[str, str | None]) -> tuple[str, ...]:
    """Requested references the document measured against a DIFFERENT teacher adapter."""
    stored = doc.get("references") or {}
    return tuple(sorted(
        k for k, aid in references.items()
        if k in stored and stored[k].get("teacher_adapter_id") != aid
    ))


def _thinned_plan(
    doc: dict, *, live: set[str], references: dict[str, str | None],
    sidecar_ckpts, ladder_steps, stale: tuple[str, ...],
) -> tuple[tuple[dict, ...], tuple[str, ...], tuple[str, ...], dict[str, tuple[str, ...]]]:
    """``(reuse, cosine_only, dropped, unscorable)`` for the stored points NOT on disk.

    Shared by the normal and the ``force`` path: a checkpoint that is gone cannot be
    recomputed either way.
    """
    stored_points = {p.get("ckpt_rel"): p for p in doc.get("points") or []}
    sidecar = set(sidecar_ckpts)
    needed = set(references)
    live_steps = {stored_points[rel].get("step") for rel in stored_points if rel in live}
    live_steps.discard(None)
    on_ladder = set(ladder_steps)
    reuse: list[dict] = []
    cosine_only: list[str] = []
    dropped: list[str] = []
    unscorable: dict[str, tuple[str, ...]] = {}
    for rel, point in stored_points.items():
        if rel in live or point.get("step") in live_steps:
            continue
        if point.get("step") not in on_ladder:
            dropped.append(rel)
            continue
        if point.get("nonfinite"):
            reuse.append(point)
            continue
        # A thinned point still has to ANSWER the requested references. It was measured
        # under whatever references the document carried then, so a newly requested one can
        # be missing: scorable from the sidecar vector (cosine-only, no checkpoint needed),
        # otherwise gone for good -- the checkpoint it would need no longer exists.
        missing = tuple(sorted(needed - (set(point.get("eas") or {}) - set(stale))))
        if not missing:
            reuse.append(point)
        elif rel in sidecar:
            cosine_only.append(rel)
        else:
            unscorable[rel] = missing
            reuse.append({**point, "unscorable_refs": list(missing)})
    return tuple(reuse), tuple(cosine_only), tuple(dropped), unscorable


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
