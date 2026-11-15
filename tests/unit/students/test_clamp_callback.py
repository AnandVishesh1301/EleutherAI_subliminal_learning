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


def _oracle_apply_clamp(modules_info, norm, ceiling):
    """Pre-optimization implementation (one ``.item()`` per module) as oracle."""
    from subliminal_learning.students.clamp import _DOSE

    dose_fn = _DOSE[norm]
    per_module = [
        (m["s"] * dose_fn(m["B"], m["A"]) / m["w0"]).item() for m in modules_info
    ]
    d_before = sum(per_module) / len(per_module)
    if d_before <= ceiling:
        return d_before, d_before, False, per_module
    factor = ceiling / d_before
    for m in modules_info:
        m["B"].mul_(factor)
    return d_before, d_before * factor, True, per_module


def test_clamp_callback_matches_oracle_on_peft_model():
    """Old vs new on a tiny peft GPTNeoX: identical doses, flags, weights."""
    import pytest

    peft = pytest.importorskip("peft")
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    from subliminal_learning.students.clamp import DoseClampCallback, apply_clamp

    def build():
        torch.manual_seed(7)
        cfg = GPTNeoXConfig(
            vocab_size=64,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
        )
        model = GPTNeoXForCausalLM(cfg)
        lcfg = peft.LoraConfig(
            r=4, lora_alpha=8, target_modules=["query_key_value", "dense"]
        )
        return peft.get_peft_model(model, lcfg)

    class _State:
        global_step = 0

    for norm in ("fro", "spec"):
        new_model, old_model = build(), build()
        for a, b in zip(new_model.parameters(), old_model.parameters()):
            assert torch.equal(a, b)

        cb = DoseClampCallback(norm, ceiling=1e-3, log_path="/dev/null")
        cb.on_train_begin(None, _State(), None, model=new_model)
        cb_old = DoseClampCallback(norm, ceiling=1e-3, log_path="/dev/null")
        cb_old.on_train_begin(None, _State(), None, model=old_model)

        gen = torch.Generator().manual_seed(11)
        for step in range(3):
            # Fake an optimizer step: same random perturbation on both copies.
            for m_new, m_old in zip(cb._modules, cb_old._modules):
                for key in ("lora_A", "lora_B"):
                    w_new = getattr(m_new, key)["default"].weight
                    noise = torch.randn(w_new.shape, generator=gen) * 0.1
                    with torch.no_grad():
                        w_new.add_(noise)
                        getattr(m_old, key)["default"].weight.add_(noise)

            state = _State()
            state.global_step = step
            with torch.no_grad():
                # Per-module doses: one stacked transfer vs one .item() each.
                from subliminal_learning.students.clamp import _DOSE

                info = cb._modules_info()
                stacked = torch.stack(
                    [m["s"] * _DOSE[norm](m["B"], m["A"]) / m["w0"] for m in info]
                ).tolist()
                looped = [
                    (m["s"] * _DOSE[norm](m["B"], m["A"]) / m["w0"]).item()
                    for m in info
                ]
                assert stacked == looped
                d_b, d_a, bound = apply_clamp(cb._modules_info(), norm, cb.ceiling)
                o_b, o_a, o_bound, _o_per = _oracle_apply_clamp(
                    cb_old._modules_info(), norm, cb_old.ceiling
                )
            assert d_b == o_b and d_a == o_a and bound == o_bound
            for p_new, p_old in zip(new_model.parameters(), old_model.parameters()):
                assert torch.equal(p_new, p_old)
        assert bound  # the ladder does bind at this ceiling
