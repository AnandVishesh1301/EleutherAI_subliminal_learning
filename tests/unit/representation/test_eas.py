"""EAS unit tests (CPU-only, no model loads).

Covers the pure surfaces: the ``position`` pooling in `mean_activations` (a tiny stub model
with a real ``gpt_neox`` module tree — extraction is hook-based — whose raw slots are a
deterministic function of the input ids so the "all" masked mean and the "last" column can be
hand-checked), the per-layer / mean cosine math (``eas_mean`` excludes only the embedding
slot; every other slot is a raw block under the ``raw_blocks_v2`` basis), the resume/merge
decision, and the JSON serializability of an entry.

Tests for the per-student-run EAS results loader.

CPU-only, no model loads: the tag/lineage parsers plus the two loaders against tmp_path
students/ trees holding hand-written documents — a teacher-source run whose MAIN document
carries its own reference, the same run's refs document, a control run (refs document only,
two references), a cross-base run, a nonfinite point, and a malformed document (skipped with a
warning). `load_eas_results` reads only main documents and `load_eas_reference_results` only
refs documents, so each test names the kind it expects.
"""

from __future__ import annotations

from pathlib import Path
import json
import math

import pytest
import torch

from subliminal_learning.representation.steering.eas import (
    EAS_MEAN_CONVENTION,
    SCHEMA_VERSION,
    eas_doc_status,
    eas_main_doc_status,
    eas_mean,
    eas_per_layer,
)
from subliminal_learning.representation.steering.eas_results_io import (
    RESULT_COLUMNS,
    endpoint_only,
    load_eas_reference_results,
    load_eas_results,
    own_reference_only,
    parse_ft_param_tag,
    parse_lineage_dir,
)
from subliminal_learning.representation.steering.vectors import (
    ACTIVATION_BASIS,
    mean_activations,
    mean_last_activations,
)


# -------------------- from test_eas.py --------------------

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
    """`subliminal_learning.pipelines.eas_pipeline._assemble_points` walks the ON-DISK points; the thinned
    reuse entries must be merged back or the rewritten doc loses them."""
    from subliminal_learning.pipelines.eas_pipeline import _assemble_points

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
    from subliminal_learning.pipelines.eas_pipeline import _assemble_points

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


# -------------------- from test_eas_results_io.py --------------------

REF_A = "dog/pythia-160m/ep10_lr1e-4_r16a32/ftseed1"

REF_B = "dog/pythia-160m/ep5_lr2e-4_r8a8/ftseed2"

def test_parse_ft_param_tag_ok():
    assert parse_ft_param_tag("ep10_lr1e-4_r16a32") == {
        "teacher_epochs": 10,
        "teacher_lr": 1e-4,
        "teacher_lora_r": 16,
        "teacher_lora_alpha": 32,
    }

def test_parse_ft_param_tag_bad():
    with pytest.raises(ValueError):
        parse_ft_param_tag("not-a-tag")

def test_parse_lineage_dir_ok():
    assert parse_lineage_dir("teacher-pythia-160m", "ep10_lr1e-4_r16a32_tseed1") == {
        "teacher_param_tag": "ep10_lr1e-4_r16a32",
        "teacher_ftseed": 1,
        "numbers_base_model": "pythia-160m",
    }

def test_parse_lineage_dir_bad():
    with pytest.raises(ValueError):
        parse_lineage_dir("teacher-pythia-160m", "notatag_tseed1")
    with pytest.raises(ValueError):
        parse_lineage_dir("notateacher", "ep10_lr1e-4_r16a32_tseed1")

def _reference(ref_key: str) -> dict:
    tag, ftseed = ref_key.split("/")[2], int(ref_key.split("/")[3][len("ftseed"):])
    return {
        "teacher_run_rel": f"teachers/{ref_key}",
        "animal": "dog",
        "teacher_base_repo": "EleutherAI/pythia-160m",
        "teacher_base_model_dir": "pythia-160m",
        "teacher_param_tag": tag,
        "teacher_ftseed": ftseed,
        "teacher_adapter_id": "aaaa",
        "v_teacher_norm_per_layer": [0.0, 1.0, 2.0],
    }

def _results_doc(
    run_rel: str,
    *,
    position: str = "last",
    source: str = "teacher",
    refs: tuple[str, ...] = (REF_A,),
    own_reference: str | None = REF_A,
    base_model_dir: str = "pythia-160m",
    numbers_base_model_dir: str = "pythia-160m",
    cross_base: bool = False,
    nonfinite_point: bool = False,
) -> dict:
    points = [
        {
            "step": 500, "ckpt_rel": "checkpoints/checkpoint-500", "is_endpoint": False,
            "adapter_id": "s500", "nonfinite": False,
            "v_student_norm_per_layer": [0.0, 1.0, 2.0],
            "eas": {r: {"mean": 0.10, "per_layer": [0.0, 0.1, 0.2]} for r in refs},
        },
        {
            "step": 1000, "ckpt_rel": "checkpoints/checkpoint-1000", "is_endpoint": True,
            "adapter_id": "s1000", "nonfinite": False,
            "v_student_norm_per_layer": [0.0, 1.0, 2.0],
            "eas": {r: {"mean": 0.42, "per_layer": [0.0, 0.4, 0.5]} for r in refs},
        },
    ]
    if nonfinite_point:
        points.insert(1, {
            "step": 750, "ckpt_rel": "checkpoints/checkpoint-750", "is_endpoint": False,
            "adapter_id": "s750", "nonfinite": True,
            "v_student_norm_per_layer": [0.0, 1.0, 2.0], "eas": None,
        })
    return {
        "schema_version": 1,
        "basis": "raw_blocks_v2",
        "eas_mean_convention": "raw_blocks_v2",
        "position": position,
        "n_prompts": 1024,
        "prompt_sha256": "p" * 8,
        "prompt_source": {
            "number_dataset_dir_rel": "numbers/x", "filtered_sha256": "f" * 8,
            "subsample_size": 10000, "subsample_seed": 42, "splice_examples": False,
        },
        "student": {
            "run_rel": run_rel, "source": source, "animal": "dog",
            "base_repo": f"EleutherAI/{base_model_dir}", "base_model_dir": base_model_dir,
            "numbers_base_model_dir": numbers_base_model_dir, "cross_base": cross_base,
            "student_param_tag": "ep5_lr1e-4_r8a32", "student_ftseed": 1,
            "number_dataset_tag": "ex10-20_prefix_n100k",
            "own_reference": own_reference, "manifest_ref": {},
        },
        "references": {r: _reference(r) for r in refs},
        "points": points,
    }

def _write(
    root: Path, run_rel: str, position: str, text: str, kind: str = "eas"
) -> None:
    """Write one document of `kind` (``eas`` = main, ``eas_refs`` = extra references)."""
    d = root / run_rel / "eval"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{kind}_{position}.json").write_text(text)

TEACHER_RUN = (
    "students/dog/pythia-160m/teacher-pythia-160m/ep10_lr1e-4_r16a32_tseed1/"
    "ex10-20_prefix_n100k/ep5_lr1e-4_r8a32/ftseed1"
)

CONTROL_RUN = (
    "students/controls/base/pythia-160m/ex10-20_prefix_n100k/ep5_lr1e-4_r8a32/ftseed1"
)

CROSS_RUN = (
    "students/dog/pythia-160m-data-seed1/teacher-pythia-160m/ep10_lr1e-4_r16a32_tseed1/"
    "ex10-20_prefix_n100k/ep5_lr1e-4_r8a32/ftseed1"
)

BAD_RUN = "students/dog/pythia-160m/broken/ep5_lr1e-4_r8a32/ftseed1"

def _tree(tmp_path: Path) -> None:
    _write(tmp_path, TEACHER_RUN, "last", json.dumps(_results_doc(TEACHER_RUN)))
    _write(tmp_path, TEACHER_RUN, "all", json.dumps(_results_doc(TEACHER_RUN, position="all")))
    # The extra reference of the same run lives beside its main document.
    _write(tmp_path, TEACHER_RUN, "last", json.dumps(_results_doc(
        TEACHER_RUN, refs=(REF_B,),
    )), kind="eas_refs")
    # A control run has no own teacher, hence ONLY a refs document.
    _write(tmp_path, CONTROL_RUN, "last", json.dumps(_results_doc(
        CONTROL_RUN, source="base", refs=(REF_A, REF_B), own_reference=None,
        nonfinite_point=True,
    )), kind="eas_refs")
    _write(tmp_path, CROSS_RUN, "last", json.dumps(_results_doc(
        CROSS_RUN, base_model_dir="pythia-160m-data-seed1",
        numbers_base_model_dir="pythia-160m", cross_base=True,
    )))
    _write(tmp_path, BAD_RUN, "last", "{ not json")

def test_load_eas_results_discovery_and_columns(tmp_path):
    _tree(tmp_path)
    df = load_eas_results(tmp_path)
    assert list(df.columns) == RESULT_COLUMNS
    # MAIN documents only: teacher run 2 positions × 2 points × 1 ref, cross run 2 points.
    assert len(df) == 4 + 2
    assert set(df["run_path"]) == {TEACHER_RUN, CROSS_RUN}
    row = df[(df.run_path == TEACHER_RUN) & (df.position == "last") & (df.step == 1000)].iloc[0]
    assert row["source"] == "teacher"
    assert row["animal"] == "dog"
    assert row["model"] == "pythia-160m"
    assert row["numbers_base_model"] == "pythia-160m"
    assert row["cross_base"] == False  # noqa: E712
    assert row["teacher_param_tag"] == "ep10_lr1e-4_r16a32"
    assert (row["teacher_epochs"], row["teacher_lora_r"], row["teacher_lora_alpha"]) == (10, 16, 32)
    assert row["teacher_ftseed"] == 1
    assert row["reference"] == REF_A
    assert row["is_own_reference"]
    assert row["dataset_tag"] == "ex10-20_prefix_n100k"
    assert row["spliced"] == False  # noqa: E712
    assert row["n_prompts"] == 1024
    assert row["eas_mean_convention"] == "raw_blocks_v2"
    assert (row["student_epochs"], row["student_lora_r"]) == (5, 8)
    assert row["student_ftseed"] == 1
    assert row["is_endpoint"]
    assert row["eas_mean"] == 0.42
    assert row["eas_per_layer"] == [0.0, 0.4, 0.5]
    # The per-slot norms of both vectors ride along (point-side and reference-side).
    assert row["v_student_norm_per_layer"] == [0.0, 1.0, 2.0]
    assert row["v_teacher_norm_per_layer"] == [0.0, 1.0, 2.0]

def test_load_eas_reference_results_control_two_references_and_nonfinite(tmp_path):
    _tree(tmp_path)
    df = load_eas_reference_results(tmp_path)
    assert list(df.columns) == RESULT_COLUMNS
    ctl = df[df.run_path == CONTROL_RUN]
    assert set(ctl["reference"]) == {REF_A, REF_B}
    assert not ctl["is_own_reference"].any()  # a control has no own reference
    # The two references carry their OWN teacher HPs on the same control rows.
    assert set(ctl[ctl.reference == REF_B]["teacher_param_tag"]) == {"ep5_lr2e-4_r8a8"}
    nf = ctl[ctl.step == 750]
    assert len(nf) == 2 and nf["nonfinite"].all()
    assert nf["eas_mean"].isna().all()
    assert nf["eas_per_layer"].isna().all()
    # A nonfinite point still carries its own recorded student norms.
    assert all(v == [0.0, 1.0, 2.0] for v in nf["v_student_norm_per_layer"])

def test_load_eas_results_cross_base(tmp_path):
    _tree(tmp_path)
    df = load_eas_results(tmp_path)
    cross = df[df.cross_base]
    assert set(cross["run_path"]) == {CROSS_RUN}
    assert set(cross["model"]) == {"pythia-160m-data-seed1"}
    assert set(cross["numbers_base_model"]) == {"pythia-160m"}

def test_load_eas_results_malformed_doc_skipped(tmp_path):
    _tree(tmp_path)
    df = load_eas_results(tmp_path)
    assert BAD_RUN not in set(df["run_path"])

def test_load_eas_results_explicit_run_rel_paths(tmp_path):
    _tree(tmp_path)
    df = load_eas_results(tmp_path, run_rel_paths=[TEACHER_RUN])
    assert set(df["run_path"]) == {TEACHER_RUN}

def test_load_eas_results_position_filter(tmp_path):
    _tree(tmp_path)
    assert set(load_eas_results(tmp_path)["position"]) == {"last", "all"}
    assert set(load_eas_results(tmp_path, positions=["all"])["position"]) == {"all"}

def test_each_loader_sees_only_its_own_documents(tmp_path):
    _tree(tmp_path)
    main = load_eas_results(tmp_path)
    refs = load_eas_reference_results(tmp_path)
    # The main frame holds own-reference rows only; the refs frame holds none of them.
    assert set(main["reference"]) == {REF_A}
    assert main["is_own_reference"].all()
    assert CONTROL_RUN not in set(main["run_path"])
    assert set(refs["run_path"]) == {TEACHER_RUN, CONTROL_RUN}
    assert not refs["is_own_reference"].any()
    assert set(refs[refs.run_path == TEACHER_RUN]["reference"]) == {REF_B}
    assert set(main["eas_file"]) == {
        str(tmp_path / r / "eval" / f"eas_{p}.json")
        for r, p in ((TEACHER_RUN, "last"), (TEACHER_RUN, "all"), (CROSS_RUN, "last"))
    }
    assert all(Path(f).name == "eas_refs_last.json" for f in refs["eas_file"])

def test_loaders_empty(tmp_path):
    (tmp_path / "students").mkdir()
    for df in (load_eas_results(tmp_path), load_eas_reference_results(tmp_path)):
        assert list(df.columns) == RESULT_COLUMNS
        assert df.empty
        assert endpoint_only(df).empty and own_reference_only(df).empty

def test_load_eas_results_warns_on_mixed_convention(tmp_path):
    from loguru import logger
    _write(tmp_path, TEACHER_RUN, "last", json.dumps(_results_doc(TEACHER_RUN)))
    other = _results_doc(CROSS_RUN)
    other["eas_mean_convention"] = "blocks_v1"
    _write(tmp_path, CROSS_RUN, "last", json.dumps(other))
    sink: list[str] = []
    handle = logger.add(sink.append, level="WARNING")
    try:
        df = load_eas_results(tmp_path)
    finally:
        logger.remove(handle)
    assert set(df["eas_mean_convention"]) == {"raw_blocks_v2", "blocks_v1"}
    assert any("convention" in m for m in sink)

def test_endpoint_only_and_own_reference_only(tmp_path):
    _tree(tmp_path)
    df = load_eas_results(tmp_path)
    end = endpoint_only(df)
    assert set(end["step"]) == {1000}
    # one endpoint row per (run, position, reference): 1 + 1 (teacher, both positions) + 1
    assert len(end) == 3
    own = own_reference_only(df)
    assert set(own["run_path"]) == {TEACHER_RUN, CROSS_RUN}
    assert own["is_own_reference"].all()

def test_root_ckpt_rel_point_is_flagged_as_the_endpoint(tmp_path):
    """The `checkpoints` root IS the endpoint: a doc whose own `is_endpoint` flag says
    otherwise must still produce an endpoint row (`endpoint_only` reads that column)."""
    doc = _results_doc(TEACHER_RUN)
    doc["points"][-1].update(ckpt_rel="checkpoints", is_endpoint=False)
    _write(tmp_path, TEACHER_RUN, "last", json.dumps(doc))
    df = load_eas_results(tmp_path)
    final = df[df.step == 1000]
    assert len(final) == 1 and bool(final.iloc[0]["is_endpoint"]) is True
    assert set(endpoint_only(df)["step"]) == {1000}

def test_distinct_non_endpoint_points_at_one_step_warn(tmp_path):
    """Two different non-endpoint rels at a step are not the same weights: one is dropped,
    and that is a document defect worth a WARNING (file + step + both rels)."""
    from loguru import logger

    doc = _results_doc(TEACHER_RUN)
    collision = {**doc["points"][0], "ckpt_rel": "checkpoints/checkpoint-500-retry",
                 "adapter_id": "s500b"}
    doc["points"].insert(1, collision)
    _write(tmp_path, TEACHER_RUN, "last", json.dumps(doc))
    sink: list[str] = []
    handle = logger.add(sink.append, level="WARNING")
    try:
        df = load_eas_results(tmp_path)
    finally:
        logger.remove(handle)
    assert len(df[df.step == 500]) == 1
    assert any("checkpoint-500-retry" in m and "checkpoints/checkpoint-500" in m
               and "step 500" in m and "eas_last.json" in m for m in sink)

def test_duplicate_final_step_point_keeps_the_endpoint(tmp_path):
    """Old docs hold the final step twice (`checkpoint-N` dir + `checkpoints` endpoint)."""
    doc = _results_doc(TEACHER_RUN)
    doc["points"].insert(1, {
        "step": 1000, "ckpt_rel": "checkpoints/checkpoint-1000", "is_endpoint": False,
        "adapter_id": "dup1000", "nonfinite": False,
        "v_student_norm_per_layer": [0.0, 1.0, 2.0],
        "eas": {REF_A: {"mean": 0.99, "per_layer": [0.0, 0.9, 0.9]}},
    })
    doc["points"][-1]["ckpt_rel"] = "checkpoints"
    _write(tmp_path, TEACHER_RUN, "last", json.dumps(doc))
    df = load_eas_results(tmp_path)
    assert len(df) == 2  # 500 + one row at 1000, not two
    final = df[df.step == 1000]
    assert len(final) == 1
    assert bool(final.iloc[0]["is_endpoint"]) is True
    assert final.iloc[0]["eas_mean"] == 0.42
    assert set(endpoint_only(df)["step"]) == {1000}

def test_thinned_point_whose_checkpoint_dir_is_gone_still_loads(tmp_path):
    """Reused THINNED points (checkpoint deleted, `unscorable_refs` annotated) are rows.

    The loader reads the document only — it must never require the `ckpt_rel` dir, which a
    thinned run no longer has (`training.ckpt_ladder`).
    """
    doc = _results_doc(TEACHER_RUN)
    doc["points"][0]["unscorable_refs"] = ["cat/pythia-160m/ep10_lr1e-4_r16a16/ftseed1"]
    _write(tmp_path, TEACHER_RUN, "last", json.dumps(doc))
    assert not (tmp_path / TEACHER_RUN / "checkpoints" / "checkpoint-500").exists()
    df = load_eas_results(tmp_path)
    assert sorted(df["step"]) == [500, 1000]
    assert df[df.step == 500].iloc[0]["eas_mean"] == 0.10
