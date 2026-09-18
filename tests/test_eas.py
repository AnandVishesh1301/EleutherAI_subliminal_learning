"""EAS unit tests (CPU-only, no model loads).

Covers the pure surfaces: the ``position`` pooling in `mean_activations` (a tiny stub model
with a real ``gpt_neox`` module tree — extraction is hook-based — whose raw slots are a
deterministic function of the input ids so the "all" masked mean and the "last" column can be
hand-checked), the per-layer / mean cosine math (``eas_mean`` excludes only the embedding
slot; every other slot is a raw block under the ``raw_blocks_v2`` basis), the resume/merge
decision, and the JSON serializability of an entry.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from subliminal_learning.steering.eas import (
    EAS_MEAN_CONVENTION,
    SCHEMA_VERSION,
    eas_doc_status,
    eas_main_doc_status,
    eas_mean,
    eas_per_layer,
)
from subliminal_learning.steering.vectors import (
    ACTIVATION_BASIS,
    mean_activations,
    mean_last_activations,
)

# Two prompts as fixed token-id sequences; the shorter one is left-padded with pad id 7.
_TOKENS = {"A": [1, 2, 3], "B": [4, 5]}
_PAD_ID = 7


class _FakeEnc(dict):
    """dict subclass supporting both ``**enc`` unpacking and attribute access."""

    def to(self, _device):
        return self

    @property
    def input_ids(self):
        return self["input_ids"]

    @property
    def attention_mask(self):
        return self["attention_mask"]


class _FakeTokenizer:
    padding_side = "left"

    def __call__(self, batch, return_tensors=None, padding=None, add_special_tokens=None):
        seqs = [_TOKENS[p] for p in batch]
        maxlen = max(len(s) for s in seqs)
        ids, attn = [], []
        for s in seqs:
            pad = maxlen - len(s)
            ids.append([_PAD_ID] * pad + s)
            attn.append([0] * pad + [1] * len(s))
        return _FakeEnc(
            input_ids=torch.tensor(ids), attention_mask=torch.tensor(attn)
        )


class _FakeEmbed(torch.nn.Module):
    """Embedding stub: slot 0 value = input_id broadcast over H dims."""

    def __init__(self, h: int):
        super().__init__()
        self.h = h

    def forward(self, input_ids):
        return input_ids.float().unsqueeze(-1).expand(*input_ids.shape, self.h)


class _FakeBlock(torch.nn.Module):
    """Decoder-block stub: adds 1 and returns the GPT-NeoX ``(hidden, *rest)`` tuple."""

    def forward(self, hidden, attention_mask=None, **_kwargs):
        return (hidden + 1,)


class _FakeModel(torch.nn.Module):
    """Raw slot l = input_id + l over H=2 dims, N_SLOTS=3 (embedding + 2 blocks).

    Extraction is hook-based (`models.residual.capture_raw_activations`), so the stub
    carries a real ``gpt_neox.{embed_in,layers}`` module tree and its forward genuinely
    runs those modules — the hooks fire on them.
    """

    N_SLOTS = 3
    H = 2

    def __init__(self):
        super().__init__()
        core = torch.nn.Module()
        core.embed_in = _FakeEmbed(self.H)
        core.layers = torch.nn.ModuleList(_FakeBlock() for _ in range(self.N_SLOTS - 1))
        self.gpt_neox = core

    def forward(self, input_ids=None, attention_mask=None, use_cache=None):
        hidden = self.gpt_neox.embed_in(input_ids)
        for block in self.gpt_neox.layers:
            (hidden,) = block(hidden, attention_mask=attention_mask)
        return hidden


# ------------------------------------------------------------------- mean_activations


def test_mean_activations_all_masks_pads():
    tok, model = _FakeTokenizer(), _FakeModel()
    prompts = ("A", "B")
    got = mean_activations(model, tok, prompts, "cpu", batch_size=8, position="all")
    # Real tokens across both prompts: [1,2,3,4,5]; per-layer mean id = 3, so slot l -> 3+l.
    # The pad id (7) must NOT leak in (it would drag the mean up and change the count).
    expected = torch.tensor([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]])
    assert torch.allclose(got, expected)


def test_mean_activations_last_matches_wrapper():
    tok, model = _FakeTokenizer(), _FakeModel()
    prompts = ("A", "B")
    got = mean_activations(model, tok, prompts, "cpu", position="last")
    # Last real token: A->3, B->5; mean 4, so slot l -> 4+l.
    expected = torch.tensor([[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]])
    assert torch.allclose(got, expected)
    # The historical wrapper delegates to position="last".
    assert torch.allclose(mean_last_activations(model, tok, prompts, "cpu"), got)


def test_mean_activations_rejects_unknown_position():
    with pytest.raises(ValueError, match="position"):
        mean_activations(_FakeModel(), _FakeTokenizer(), ("A",), "cpu", position="mid")


def test_mean_activations_batching_equivalent_both_positions():
    # A single forward pass and a batch_size=1 loop over the same prompts must agree for BOTH
    # positions: the "all" masked mean and the "last" left-pad column are batch-invariant.
    tok, model = _FakeTokenizer(), _FakeModel()
    prompts = ("A", "B", "A", "B")  # >1 batch at batch_size=1; padded together at batch_size=8
    for position in ("last", "all"):
        one = mean_activations(model, tok, prompts, "cpu", batch_size=1, position=position)
        many = mean_activations(model, tok, prompts, "cpu", batch_size=8, position=position)
        assert torch.allclose(one, many), position


# ------------------------------------------------------------------- cosine math


def test_eas_per_layer_and_mean_known_cosines():
    # 4 slots (raw_blocks_v2): 0=embedding (cos -1), then raw blocks 0..2 with cosines
    # +1, 0 (orthogonal), -1.
    v_teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    v_student = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    per = eas_per_layer(v_student, v_teacher)
    assert torch.allclose(per, torch.tensor([-1.0, 1.0, 0.0, -1.0]), atol=1e-6)
    # eas_mean skips ONLY the embedding slot (0): (1 + 0 + -1)/3 = 0.
    assert eas_mean(per) == pytest.approx(0.0)
    # Sanity that the embedding slot is genuinely excluded (including it gives -0.25) and
    # the final raw block genuinely included (the retired both-ends window gave 0.5).
    assert per.mean().item() == pytest.approx(-0.25)
    assert per[1:-1].mean().item() == pytest.approx(0.5)


def test_eas_per_layer_shape_mismatch_raises():
    # Cross-base guard: a base A / base B depth (or width) mismatch must be named, not silently
    # broadcast or truncated.
    with pytest.raises(ValueError, match=r"shape mismatch.*v_student.*v_teacher"):
        eas_per_layer(torch.zeros(4, 2), torch.zeros(3, 2))
    with pytest.raises(ValueError, match="shape mismatch"):
        eas_per_layer(torch.zeros(4, 2), torch.zeros(4, 3))


def test_eas_mean_too_few_slots_raises():
    # A single slot -> the [1:] block window is empty -> raise instead of returning NaN.
    with pytest.raises(ValueError, match="at least 2 slots"):
        eas_mean(torch.tensor([0.1]))
    # Two slots is the minimum: one embedding + one raw block.
    assert eas_mean(torch.tensor([0.1, 0.2])) == pytest.approx(0.2)


def test_eas_zero_student_vector_is_finite():
    # A base/rng control whose activation shift is ~0 (all-zeros v_student vs a real v_teacher):
    # the cosine must be finite (0), never NaN, so eas_mean stays a real number.
    v_teacher = torch.tensor([[1.0, 0.0], [1.0, 2.0], [0.0, 1.0], [3.0, 1.0]])
    v_student = torch.zeros_like(v_teacher)
    per = eas_per_layer(v_student, v_teacher)
    assert torch.isfinite(per).all()
    assert math.isfinite(eas_mean(per))


# ------------------------------------------------------------------- doc currency


def _meta():
    return {
        "schema_version": SCHEMA_VERSION,
        "basis": ACTIVATION_BASIS,
        "eas_mean_convention": EAS_MEAN_CONVENTION,
        "position": "last",
        "n_prompts": 8,
        "prompt_sha256": "abc",
    }


def _doc(points, references=None, meta=None, manifest_ref=None):
    return {
        **(meta or _meta()),
        "student": {"manifest_ref": manifest_ref},
        "references": references if references is not None else {"dog/t": {"teacher_adapter_id": "T"}},
        "points": points,
    }


def _point(rel, adapter_id, refs=("dog/t",), nonfinite=False):
    return {
        "ckpt_rel": rel, "adapter_id": adapter_id, "nonfinite": nonfinite,
        "eas": None if nonfinite else {r: {"mean": 0.1, "per_layer": [0.1]} for r in refs},
    }


def test_doc_status_current_doc_reuses_every_point():
    doc = _doc([_point("checkpoints/checkpoint-1", "A"), _point("checkpoints", "B")])
    plan = eas_doc_status(
        doc, expected_meta=_meta(),
        checkpoints=[("checkpoints/checkpoint-1", "A"), ("checkpoints", "B")],
        references={"dog/t": "T"},
    )
    assert plan.is_current and len(plan.reuse) == 2
    assert plan.recompute == () and plan.cosine_only == ()


def test_doc_status_no_doc_recomputes_everything():
    plan = eas_doc_status(
        None, expected_meta=_meta(), checkpoints=[("checkpoints", "B")],
        references={"dog/t": "T"},
    )
    assert plan.invalid and plan.recompute == ("checkpoints",)


def test_doc_status_meta_drift_is_invalid():
    doc = _doc([_point("checkpoints", "B")], meta={**_meta(), "prompt_sha256": "other"})
    plan = eas_doc_status(
        doc, expected_meta=_meta(), checkpoints=[("checkpoints", "B")],
        references={"dog/t": "T"},
    )
    assert plan.invalid and "prompt_sha256" in plan.reason


def test_doc_status_manifest_ref_drift_is_invalid():
    doc = _doc([_point("checkpoints", "B")], manifest_ref={"n_train_steps": 10})
    plan = eas_doc_status(
        doc, expected_meta=_meta(), checkpoints=[("checkpoints", "B")],
        references={"dog/t": "T"}, manifest_ref={"n_train_steps": 20},
    )
    assert plan.invalid and "retrained" in plan.reason


def test_doc_status_changed_adapter_recomputes_only_that_point():
    doc = _doc([_point("checkpoints/checkpoint-1", "A"), _point("checkpoints", "B")])
    plan = eas_doc_status(
        doc, expected_meta=_meta(),
        checkpoints=[("checkpoints/checkpoint-1", "A2"), ("checkpoints", "B")],
        references={"dog/t": "T"},
    )
    assert plan.recompute == ("checkpoints/checkpoint-1",)
    assert [p["ckpt_rel"] for p in plan.reuse] == ["checkpoints"]


def test_doc_status_new_reference_is_cosine_only_with_a_sidecar():
    doc = _doc(
        [_point("checkpoints", "B")],
        references={"dog/t": {"teacher_adapter_id": "T"}},
    )
    plan = eas_doc_status(
        doc, expected_meta=_meta(), checkpoints=[("checkpoints", "B")],
        references={"dog/t": "T", "cat/t": "C"},
        sidecar_ckpts=frozenset({"checkpoints"}),
    )
    assert plan.cosine_only == ("checkpoints",) and plan.recompute == ()


def test_doc_status_new_reference_without_sidecar_recomputes():
    doc = _doc([_point("checkpoints", "B")])
    plan = eas_doc_status(
        doc, expected_meta=_meta(), checkpoints=[("checkpoints", "B")],
        references={"dog/t": "T", "cat/t": "C"},
    )
    assert plan.recompute == ("checkpoints",) and plan.cosine_only == ()


def test_doc_status_stale_reference_is_flagged():
    doc = _doc([_point("checkpoints", "B")])
    plan = eas_doc_status(
        doc, expected_meta=_meta(), checkpoints=[("checkpoints", "B")],
        references={"dog/t": "T_NEW"},
    )
    assert plan.stale_refs == ("dog/t",)
    assert plan.recompute == ("checkpoints",)


def test_doc_status_vanished_checkpoint_point_is_dropped():
    doc = _doc([_point("checkpoints/checkpoint-1", "A"), _point("checkpoints", "B")])
    plan = eas_doc_status(
        doc, expected_meta=_meta(), checkpoints=[("checkpoints", "B")],
        references={"dog/t": "T"},
    )
    assert plan.dropped == ("checkpoints/checkpoint-1",)
    assert not plan.is_current


def test_doc_status_thinned_checkpoints_are_reused_not_dropped():
    """34 stored points whose dirs were THINNED (their steps are in the ladder manifest).

    The pipeline rebuilds the document from `plan.reuse`, so a thinned point must be
    carried there; dropping it would shrink every EAS trajectory to the kept checkpoints.
    """
    steps = list(range(1, 35))
    points = [{**_point(f"checkpoints/checkpoint-{s}", f"A{s}"), "step": s} for s in steps]
    points.append({**_point("checkpoints", "B"), "step": 35})
    doc = _doc(points)
    plan = eas_doc_status(
        doc, expected_meta=_meta(), checkpoints=[("checkpoints", "B")],
        references={"dog/t": "T"}, ladder_steps=frozenset(steps + [35]),
    )
    assert plan.dropped == () and plan.is_current
    assert ({p["ckpt_rel"] for p in plan.reuse}
            == {p["ckpt_rel"] for p in points})   # order is the pipeline's, by step
    # Without the ladder manifest the same document is a pile of lost points (today's rule).
    bare = eas_doc_status(
        doc, expected_meta=_meta(), checkpoints=[("checkpoints", "B")],
        references={"dog/t": "T"},
    )
    assert len(bare.dropped) == 34 and not bare.is_current


def test_assemble_points_keeps_thinned_points_in_step_order():
    """`pipelines.eas_pipeline._assemble_points` walks the ON-DISK points; the thinned
    reuse entries must be merged back or the rewritten doc loses them."""
    from pipelines.eas_pipeline import _assemble_points

    reuse = [{**_point(f"checkpoints/checkpoint-{s}", f"A{s}"), "step": s} for s in (1, 2)]
    endpoint = {**_point("checkpoints", "B"), "step": 3}

    class _Plan:
        pass

    plan = _Plan()
    plan.reuse = tuple(reuse + [endpoint])
    out = _assemble_points([endpoint], plan, {}, {})
    assert [p["step"] for p in out] == [1, 2, 3]


def test_doc_status_nonfinite_point_is_carried_over():
    doc = _doc([_point("checkpoints", "B", nonfinite=True)])
    plan = eas_doc_status(
        doc, expected_meta=_meta(), checkpoints=[("checkpoints", "B")],
        references={"dog/t": "T", "cat/t": "C"},
    )
    assert plan.is_current and plan.reuse[0]["nonfinite"] is True


def test_doc_status_cheap_mode_ignores_unhashed_intermediate_adapters():
    doc = _doc([_point("checkpoints/checkpoint-1", "A"), _point("checkpoints", "B")])
    plan = eas_doc_status(
        doc, expected_meta=_meta(),
        checkpoints=[("checkpoints/checkpoint-1", None), ("checkpoints", "B")],
        references={"dog/t": "T"}, cheap=True,
    )
    assert plan.is_current


def test_doc_status_force_recomputes_everything():
    doc = _doc([_point("checkpoints", "B")])
    plan = eas_doc_status(
        doc, expected_meta=_meta(), checkpoints=[("checkpoints", "B")],
        references={"dog/t": "T"}, force=True,
    )
    assert plan.invalid and plan.recompute == ("checkpoints",)


# ------------------------------------------------------------------- main-doc currency


def test_main_doc_status_an_on_disk_own_reference_only_document_is_current():
    """The 464 documents drained before the two-file split must stay CURRENT.

    Shaped like one on disk (schema_version 1, ``references`` = the run's own teacher
    alone, one point per checkpoint): nothing to recompute, nothing cosine-only.
    """
    own = "dog/pythia-160m/ep2_lr5e-4_r32a64/ftseed1"
    doc = _doc(
        [_point("checkpoints/checkpoint-500", "A", refs=(own,)),
         _point("checkpoints", "B", refs=(own,))],
        references={own: {"teacher_adapter_id": "T"}},
    )
    plan = eas_main_doc_status(
        doc, expected_meta=_meta(),
        checkpoints=[("checkpoints/checkpoint-500", "A"), ("checkpoints", "B")],
        references={own: "T"},
    )
    assert plan.is_current and not plan.stale_refs
    assert plan.recompute == () and plan.cosine_only == () and len(plan.reuse) == 2


def test_main_doc_status_a_foreign_reference_is_invalid_and_cosine_only_off_the_sidecar():
    doc = _doc(
        [_point("checkpoints", "B", refs=("dog/t", "cat/t"))],
        references={"dog/t": {"teacher_adapter_id": "T"},
                    "cat/t": {"teacher_adapter_id": "C"}},
    )
    plan = eas_main_doc_status(
        doc, expected_meta=_meta(), checkpoints=[("checkpoints", "B")],
        references={"dog/t": "T"}, sidecar_ckpts=frozenset({"checkpoints"}),
    )
    assert plan.invalid and "own teacher" in plan.reason
    assert plan.cosine_only == ("checkpoints",) and plan.recompute == ()


def test_main_doc_status_without_a_sidecar_recomputes_the_foreign_reference_document():
    doc = _doc(
        [_point("checkpoints", "B", refs=("dog/t", "cat/t"))],
        references={"dog/t": {"teacher_adapter_id": "T"},
                    "cat/t": {"teacher_adapter_id": "C"}},
    )
    plan = eas_main_doc_status(
        doc, expected_meta=_meta(), checkpoints=[("checkpoints", "B")],
        references={"dog/t": "T"},
    )
    assert plan.invalid and plan.recompute == ("checkpoints",)


def test_doc_is_json_serializable():
    doc = _doc([_point("checkpoints", "B")])
    assert json.loads(json.dumps(doc))["points"][0]["ckpt_rel"] == "checkpoints"


# ------------------------------------------------- thinned points vs a NEW reference
def _thinned_doc(thinned_refs=("dog/t",), endpoint_refs=("dog/t",)):
    """A doc over ladder [50, 100] whose `checkpoint-50` dir has been thinned away."""
    return _doc(
        [
            {**_point("checkpoints/checkpoint-50", "A", refs=thinned_refs), "step": 50},
            {**_point("checkpoints", "B", refs=endpoint_refs), "step": 100},
        ],
        references={
            r: {"teacher_adapter_id": "T"} for r in {*thinned_refs, *endpoint_refs}
        },
    )


def test_thinned_point_carrying_the_reference_is_reused_and_current():
    plan = eas_doc_status(
        _thinned_doc(("dog/t", "cat/t"), ("dog/t", "cat/t")), expected_meta=_meta(),
        checkpoints=[("checkpoints", "B")], references={"cat/t": "T"},
        ladder_steps=frozenset({50, 100}),
    )
    assert plan.unscorable == {} and plan.is_current
    assert {p["ckpt_rel"] for p in plan.reuse} == {
        "checkpoints/checkpoint-50", "checkpoints"
    }


def test_thinned_point_missing_a_reference_is_cosine_only_off_the_sidecar():
    plan = eas_doc_status(
        _thinned_doc(endpoint_refs=("dog/t", "cat/t")), expected_meta=_meta(),
        checkpoints=[("checkpoints", "B")], references={"dog/t": "T", "cat/t": "T"},
        sidecar_ckpts=frozenset({"checkpoints/checkpoint-50", "checkpoints"}),
        ladder_steps=frozenset({50, 100}),
    )
    assert plan.cosine_only == ("checkpoints/checkpoint-50",)
    assert plan.unscorable == {} and not plan.is_current


def test_thinned_point_with_no_sidecar_vector_is_unscorable_not_a_loop():
    """No checkpoint AND no vector: the reference can never be measured at that step.

    The point is still reused (annotated), the doc is not `is_current` for that reference,
    and `only_unscorable` tells the sweep to record the unit as done instead of re-enqueuing
    work nothing could ever complete.
    """
    plan = eas_doc_status(
        _thinned_doc(endpoint_refs=("dog/t", "cat/t")), expected_meta=_meta(),
        checkpoints=[("checkpoints", "B")], references={"dog/t": "T", "cat/t": "T"},
        sidecar_ckpts=frozenset({"checkpoints"}),
        ladder_steps=frozenset({50, 100}),
    )
    assert plan.unscorable == {"checkpoints/checkpoint-50": ("cat/t",)}
    assert not plan.is_current and plan.only_unscorable
    assert plan.recompute == () and plan.cosine_only == ()
    annotated = next(
        p for p in plan.reuse if p["ckpt_rel"] == "checkpoints/checkpoint-50"
    )
    assert annotated["unscorable_refs"] == ["cat/t"]


def test_force_keeps_a_thinned_point_while_recomputing_the_on_disk_ones():
    """``--force`` cannot recompute a checkpoint that no longer exists: carry it."""
    plan = eas_doc_status(
        _thinned_doc(("dog/t", "cat/t"), ("dog/t", "cat/t")), expected_meta=_meta(),
        checkpoints=[("checkpoints", "B")], references={"cat/t": "T"},
        ladder_steps=frozenset({50, 100}), force=True,
    )
    assert plan.invalid and not plan.is_current and plan.reason == "force"
    assert plan.recompute == ("checkpoints",) and plan.dropped == ()
    assert [p["ckpt_rel"] for p in plan.reuse] == ["checkpoints/checkpoint-50"]


def test_force_scores_a_thinned_point_cosine_only_for_a_new_reference():
    plan = eas_doc_status(
        _thinned_doc(endpoint_refs=("dog/t", "cat/t")), expected_meta=_meta(),
        checkpoints=[("checkpoints", "B")], references={"dog/t": "T", "cat/t": "T"},
        sidecar_ckpts=frozenset({"checkpoints/checkpoint-50", "checkpoints"}),
        ladder_steps=frozenset({50, 100}), force=True,
    )
    assert plan.cosine_only == ("checkpoints/checkpoint-50",)
    assert plan.recompute == ("checkpoints",) and plan.reuse == ()
    # Without the vector the reference can never be measured there: annotated, not lost.
    bare = eas_doc_status(
        _thinned_doc(endpoint_refs=("dog/t", "cat/t")), expected_meta=_meta(),
        checkpoints=[("checkpoints", "B")], references={"dog/t": "T", "cat/t": "T"},
        sidecar_ckpts=frozenset({"checkpoints"}),
        ladder_steps=frozenset({50, 100}), force=True,
    )
    assert bare.unscorable == {"checkpoints/checkpoint-50": ("cat/t",)}
    assert [p["ckpt_rel"] for p in bare.reuse] == ["checkpoints/checkpoint-50"]


def test_force_drops_thinned_points_when_the_identity_drifted():
    doc = _thinned_doc()
    doc["prompt_sha256"] = "other"
    plan = eas_doc_status(
        doc, expected_meta=_meta(), checkpoints=[("checkpoints", "B")],
        references={"dog/t": "T"}, ladder_steps=frozenset({50, 100}), force=True,
    )
    assert plan.reuse == () and plan.cosine_only == () and plan.dropped == ()


def test_assemble_points_merges_a_forced_plans_thinned_reuse():
    from pipelines.eas_pipeline import _assemble_points

    plan = eas_doc_status(
        _thinned_doc(("dog/t", "cat/t"), ("dog/t", "cat/t")), expected_meta=_meta(),
        checkpoints=[("checkpoints", "B")], references={"cat/t": "T"},
        ladder_steps=frozenset({50, 100}), force=True,
    )
    endpoint = {
        "ckpt_rel": "checkpoints", "adapter_id": "B", "step": 100, "is_endpoint": True,
    }
    out = _assemble_points([endpoint], plan, {"checkpoints": None}, {})
    assert [p["step"] for p in out] == [50, 100]
