"""EAS unit tests (CPU-only, no model loads).

Covers the pure surfaces: the ``position`` pooling in `mean_activations` (a tiny stub model
whose hidden states are a deterministic function of the input ids so the "all" masked mean and
the "last" column can be hand-checked), the per-layer / mean cosine math (``eas_mean`` excludes
BOTH end slots — embedding and post-final-layernorm), the resume/merge decision, and the JSON
serializability of an entry.
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
    build_entry,
    eas_mean,
    eas_per_layer,
    plan_reuse,
)
from subliminal_learning.steering.vectors import mean_activations, mean_last_activations

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


class _FakeOut:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _FakeModel:
    """hidden_states[l][b, t, :] = input_id[b, t] + l over H=2 dims, N_SLOTS=3 layers."""

    N_SLOTS = 3
    H = 2

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=None, use_cache=None):
        b, t = input_ids.shape
        base = input_ids.float().unsqueeze(-1).expand(b, t, self.H)
        return _FakeOut(tuple(base + layer for layer in range(self.N_SLOTS)))


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
    # 4 slots: 0=embedding (cos -1), 1=+1, 2=orthogonal (0), 3=post-final-LN (cos -1).
    v_teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    v_student = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    per = eas_per_layer(v_student, v_teacher)
    assert torch.allclose(per, torch.tensor([-1.0, 1.0, 0.0, -1.0]), atol=1e-6)
    # eas_mean skips BOTH end slots (0 embedding AND 3 post-final-LN): (1 + 0)/2 = 0.5.
    assert eas_mean(per) == pytest.approx(0.5)
    # Sanity that both ends are genuinely excluded: including the final slot (3) would give
    # (1 + 0 + -1)/3 = 0, and including the embedding slot too would give -0.25.
    assert per[1:].mean().item() == pytest.approx(0.0)
    assert per.mean().item() == pytest.approx(-0.25)


def test_eas_per_layer_shape_mismatch_raises():
    # Cross-base guard: a base A / base B depth (or width) mismatch must be named, not silently
    # broadcast or truncated.
    with pytest.raises(ValueError, match=r"shape mismatch.*v_student.*v_teacher"):
        eas_per_layer(torch.zeros(4, 2), torch.zeros(3, 2))
    with pytest.raises(ValueError, match="shape mismatch"):
        eas_per_layer(torch.zeros(4, 2), torch.zeros(4, 3))


def test_eas_mean_too_few_slots_raises():
    # Fewer than 3 slots -> the [1:-1] block window is empty -> raise instead of returning NaN.
    with pytest.raises(ValueError, match="at least 3 slots"):
        eas_mean(torch.tensor([0.1, 0.2]))


def test_eas_zero_student_vector_is_finite():
    # A base/rng control whose activation shift is ~0 (all-zeros v_student vs a real v_teacher):
    # the cosine must be finite (0), never NaN, so eas_mean stays a real number.
    v_teacher = torch.tensor([[1.0, 0.0], [1.0, 2.0], [0.0, 1.0], [3.0, 1.0]])
    v_student = torch.zeros_like(v_teacher)
    per = eas_per_layer(v_student, v_teacher)
    assert torch.isfinite(per).all()
    assert math.isfinite(eas_mean(per))


# ------------------------------------------------------------------- resume/merge


def _stored(meta, entries):
    return {"meta": meta, "entries": entries}


def _entry(source, n_epochs, seed, adapter_id):
    return {"source": source, "n_epochs": n_epochs, "student_ftseed": seed, "adapter_id": adapter_id}


def test_plan_reuse_reuses_on_identical_meta_and_adapter():
    meta = {"position": "last", "n_prompts": 8}
    stored = _stored(meta, [_entry("teacher", 1, 1, "X")])
    plan = plan_reuse(stored, dict(meta), [(("teacher", 1, 1), "X")])
    assert plan.to_compute == []
    assert plan.reused_entries == [_entry("teacher", 1, 1, "X")]
    assert plan.meta_drifted is False


def test_plan_reuse_meta_drift_recomputes_all():
    stored = _stored({"position": "last"}, [_entry("teacher", 1, 1, "X")])
    work = [(("teacher", 1, 1), "X")]
    plan = plan_reuse(stored, {"position": "all"}, work)
    assert plan.reused_entries == []
    assert plan.to_compute == work
    assert plan.meta_drifted is True


def test_plan_reuse_adapter_drift_recomputes_only_that_entry():
    meta = {"position": "last"}
    stored = _stored(
        meta, [_entry("teacher", 1, 1, "X"), _entry("teacher", 2, 1, "Y")]
    )
    work = [(("teacher", 1, 1), "X"), (("teacher", 2, 1), "Z")]  # ep2 adapter changed
    plan = plan_reuse(stored, dict(meta), work)
    assert plan.reused_entries == [_entry("teacher", 1, 1, "X")]
    assert plan.to_compute == [(("teacher", 2, 1), "Z")]
    assert plan.meta_drifted is False


def test_plan_reuse_force_recomputes_all():
    meta = {"position": "last"}
    stored = _stored(meta, [_entry("teacher", 1, 1, "X")])
    work = [(("teacher", 1, 1), "X")]
    plan = plan_reuse(stored, dict(meta), work, force=True)
    assert plan.reused_entries == []
    assert plan.to_compute == work
    assert plan.meta_drifted is False


def test_plan_reuse_no_stored_doc_recomputes_all():
    work = [(("base", 5, 1), "A")]
    plan = plan_reuse(None, {"position": "last"}, work)
    assert plan.reused_entries == []
    assert plan.preserved_entries == []
    assert plan.to_compute == work
    assert plan.meta_drifted is False


def _sorted_entries(entries):
    return sorted(entries, key=lambda e: (e["source"], e["n_epochs"], e["student_ftseed"]))


def test_plan_reuse_narrowed_request_preserves_unrequested_entries():
    # Stored keys {A, B, C}; a narrowed re-run asks only for A (adapter unchanged).
    meta = {"position": "last"}
    a = _entry("teacher", 1, 1, "X")
    b = _entry("teacher", 20, 1, "Y")  # e.g. an ep20 curve not in this --epochs request
    c = _entry("base", 1, 1, "Z")  # e.g. a control source not in this --sources request
    stored = _stored(meta, [a, b, c])
    plan = plan_reuse(stored, dict(meta), [(("teacher", 1, 1), "X")])
    assert plan.reused_entries == [a]
    assert plan.to_compute == []
    assert _sorted_entries(plan.preserved_entries) == _sorted_entries([b, c])
    assert plan.meta_drifted is False


def test_plan_reuse_narrowed_request_with_changed_adapter_still_preserves_rest():
    # A is requested but its adapter changed -> A recomputes; B, C preserved untouched.
    meta = {"position": "last"}
    a = _entry("teacher", 1, 1, "X")
    b = _entry("teacher", 20, 1, "Y")
    c = _entry("base", 1, 1, "Z")
    stored = _stored(meta, [a, b, c])
    plan = plan_reuse(stored, dict(meta), [(("teacher", 1, 1), "NEW")])
    assert plan.reused_entries == []
    assert plan.to_compute == [(("teacher", 1, 1), "NEW")]
    assert _sorted_entries(plan.preserved_entries) == _sorted_entries([b, c])
    assert plan.meta_drifted is False


def test_plan_reuse_meta_drift_preserves_nothing():
    stored = _stored(
        {"position": "last"},
        [_entry("teacher", 1, 1, "X"), _entry("teacher", 20, 1, "Y")],
    )
    plan = plan_reuse(stored, {"position": "all"}, [(("teacher", 1, 1), "X")])
    assert plan.reused_entries == []
    assert plan.preserved_entries == []
    assert plan.to_compute == [(("teacher", 1, 1), "X")]
    assert plan.meta_drifted is True


def test_plan_reuse_force_preserves_nothing():
    meta = {"position": "last"}
    stored = _stored(
        meta, [_entry("teacher", 1, 1, "X"), _entry("teacher", 20, 1, "Y")]
    )
    plan = plan_reuse(stored, dict(meta), [(("teacher", 1, 1), "X")], force=True)
    assert plan.reused_entries == []
    assert plan.preserved_entries == []
    assert plan.to_compute == [(("teacher", 1, 1), "X")]
    assert plan.meta_drifted is False


# ------------------------------------------------------------------- entry schema


def test_build_entry_is_json_serializable():
    per = torch.tensor([0.1, 0.2, 0.3, 0.4])
    norm = torch.tensor([1.0, 2.0, 3.0, 4.0])
    entry = build_entry(
        source="teacher",
        n_epochs=10,
        student_ftseed=1,
        student_param_tag="ep10_lr1e-4_r8a32",
        run_dir="/out/students/dog/.../ftseed1",
        adapter_id="deadbeef",
        per_layer=per,
        v_student_norm=norm,
    )
    round_tripped = json.loads(json.dumps(entry))
    assert round_tripped["eas_mean"] == pytest.approx(eas_mean(per))
    assert round_tripped["eas_per_layer"] == pytest.approx(per.tolist())
    assert round_tripped["v_student_norm_per_layer"] == pytest.approx(norm.tolist())
    assert round_tripped["adapter_id"] == "deadbeef"
