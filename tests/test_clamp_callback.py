"""Unit tests for the LoRA dose-clamp helpers (CPU-only, no model needed)."""

from __future__ import annotations

import torch

from subliminal_learning.students.clamp import apply_clamp, dose_fro, dose_spec


def test_dose_fro_matches_reference():
    torch.manual_seed(0)
    B = torch.randn(64, 8)
    A = torch.randn(8, 32)
    ref = torch.linalg.matrix_norm(B @ A, ord="fro")
    assert torch.allclose(dose_fro(B, A), ref, atol=1e-5)


def test_dose_spec_matches_reference():
    torch.manual_seed(1)
    B = torch.randn(64, 8)
    A = torch.randn(8, 32)
    ref = torch.linalg.svdvals(B @ A).max()
    assert torch.allclose(dose_spec(B, A), ref, atol=1e-5)


def _module_info(B, A, s, norm):
    ref = "fro" if norm == "fro" else 2
    w0 = float(torch.linalg.matrix_norm(torch.randn(B.shape[0], A.shape[1]), ord=ref))
    return {"B": B, "A": A, "s": s, "w0": w0}


def test_apply_clamp_binds_to_ceiling_and_preserves_direction():
    torch.manual_seed(2)
    mods = [_module_info(torch.randn(64, 8), torch.randn(8, 32), 4.0, "fro") for _ in range(3)]
    before_prod = [m["B"] @ m["A"] for m in mods]
    d_before, d_after, bound = apply_clamp(mods, "fro", ceiling=1e-3)
    assert bound
    assert abs(d_after - 1e-3) < 1e-5
    # Direction preserved: cosine(BA_before, BA_after) == 1.
    for m, pre in zip(mods, before_prod):
        post = (m["B"] @ m["A"]).flatten()
        cos = torch.nn.functional.cosine_similarity(
            pre.flatten(), post, dim=0
        )
        assert abs(float(cos) - 1.0) < 1e-6


def test_apply_clamp_noop_when_below_ceiling():
    torch.manual_seed(3)
    mods = [_module_info(torch.randn(64, 8), torch.randn(8, 32), 1.0, "spec") for _ in range(2)]
    snapshot = [m["B"].clone() for m in mods]
    d_before, d_after, bound = apply_clamp(mods, "spec", ceiling=1e9)
    assert not bound
    assert d_after == d_before
    for m, snap in zip(mods, snapshot):
        assert torch.equal(m["B"], snap)


def test_sft_callbacks_default_none_signature():
    """The finetune entry point accepts callbacks=None (byte-identical old call)."""
    import inspect

    from subliminal_learning.teachers.sft_io import run_teacher_sft_completion_only

    sig = inspect.signature(run_teacher_sft_completion_only)
    assert sig.parameters["callbacks"].default is None
