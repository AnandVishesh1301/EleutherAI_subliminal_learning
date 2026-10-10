"""Wave-2 training controls: frozen modules, LoRA reach, weight decay, Adam ε, tags."""

from __future__ import annotations

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, Trainer, TrainingArguments

from subliminal_learning.config import LoRAConfig, TrainConfig, hf_weight_decay
from subliminal_learning.evaluation.gkd_results_io import parse_gkd_tag
from subliminal_learning.gkd.layout import (
    FREEZE_BY_TAG,
    NEOX_LINEAR_MODULES,
    decay_suffix,
    eps_suffix,
    freeze_suffix,
    gkd_tag,
    lora_scope_suffix,
    train_tag_kwargs,
)
from subliminal_learning.students.layout import student_param_tag
from subliminal_learning.training.lora_io import build_lora_config
from subliminal_learning.training.param_control_io import (
    AnchoredDecayCallback,
    apply_freeze_modules,
    trainable_parameter_counts,
    validate_train_settings,
)
from subliminal_learning.training.run_manifest import (
    lora_manifest_blob,
    train_identity_extras,
)

V, H, L, I = 50, 16, 3, 32


def _train(**kw) -> TrainConfig:
    base = dict(
        learning_rate=1e-4,
        n_epochs=20,
        batch_size=8,
        max_length=256,
        max_grad_norm=1.0,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        seed=1,
    )
    return TrainConfig(**{**base, **kw})


def _lora(**kw) -> LoRAConfig:
    return LoRAConfig(
        **{
            "r": 8,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "target_modules": "all-linear",
            **kw,
        }
    )


def _tiny() -> GPTNeoXForCausalLM:
    torch.manual_seed(0)
    cfg = GPTNeoXConfig(
        vocab_size=V,
        hidden_size=H,
        num_hidden_layers=L,
        num_attention_heads=2,
        intermediate_size=I,
        max_position_embeddings=32,
        tie_word_embeddings=False,
    )
    return GPTNeoXForCausalLM(cfg)


# Hand counts of the tiny model (weights + biases).
EMB = V * H
LN = 2 * H
ATTN = (H * 3 * H + 3 * H) + (H * H + H)
MLP = (H * I + I) + (I * H + H)
LAYER = 2 * LN + ATTN + MLP
TOTAL = 2 * EMB + L * LAYER + LN


# ---- tags -----------------------------------------------------------------------------


def test_student_tag_defaults_byte_identical() -> None:
    assert student_param_tag(_train(), _lora()) == "ep20_lr1e-4_r8a32"
    assert student_param_tag(_train(optim="sgd"), None) == "ep20_lr1e-4_full_sgd"
    assert (
        student_param_tag(_train(optim="sgd", max_grad_norm=0.0), _lora())
        == "ep20_lr1e-4_r8a32_sgd_clip0"
    )


def test_gkd_tag_defaults_byte_identical() -> None:
    kw = dict(lmbda=1.0, beta=1.0, learning_rate=4e-4, num_epochs=1.0)
    assert (
        gkd_tag(student_mode="lora", lora=_lora(lora_alpha=16), **kw)
        == "lmb1.0_b1.0_lora_lr4e-4_e1_r8a16"
    )
    assert (
        gkd_tag(student_mode="full", optim="sgd", **kw, **train_tag_kwargs(_train()))
        == "lmb1.0_b1.0_full_lr4e-4_e1_sgd"
    )


@pytest.mark.parametrize(
    "train, lora, expected",
    [
        (_train(), _lora(modules_to_save=("embed_out",)), "ep20_lr1e-4_r8a32_head"),
        (
            _train(),
            _lora(target_modules=("query_key_value", "dense")),
            "ep20_lr1e-4_r8a32_attn",
        ),
        (
            _train(),
            _lora(target_modules=NEOX_LINEAR_MODULES, layers_to_transform=(0, 1)),
            "ep20_lr1e-4_r8a32_L0-1",
        ),
        (
            _train(freeze_modules=("gpt_neox.embed_in", "embed_out")),
            None,
            "ep20_lr1e-4_full_body",
        ),
        (
            _train(adam_epsilon=0.1, weight_decay=0.084),
            _lora(),
            "ep20_lr1e-4_r8a32_eps0.1_wd0.084",
        ),
        (
            _train(optim="sgd", weight_decay=4.2e-4, weight_decay_anchor=True),
            None,
            "ep20_lr1e-4_full_sgd_wda0.00042",
        ),
    ],
)
def test_student_tag_new_suffixes(train, lora, expected) -> None:
    assert student_param_tag(train, lora) == expected


@pytest.mark.parametrize(
    "lora, train",
    [
        (_lora(lora_alpha=16, modules_to_save=("embed_out",)), _train()),
        (_lora(lora_alpha=16, target_modules=("dense_h_to_4h", "dense_4h_to_h")), _train()),
        (
            _lora(
                lora_alpha=16,
                target_modules=NEOX_LINEAR_MODULES,
                layers_to_transform=(0, 1),
            ),
            _train(adam_epsilon=1e-6),
        ),
        (None, _train(freeze_modules=FREEZE_BY_TAG["embout"], weight_decay=17.0)),
        (
            None,
            _train(weight_decay=150.0, weight_decay_anchor=True, max_grad_norm=0.0),
        ),
    ],
)
def test_gkd_tag_round_trip(lora, train) -> None:
    tag = gkd_tag(
        lmbda=1.0,
        beta=1.0,
        student_mode="full" if lora is None else "lora",
        learning_rate=2e-5,
        num_epochs=1.0,
        lora=lora,
        **train_tag_kwargs(train),
    )
    parsed = parse_gkd_tag(tag)
    assert parsed["adam_epsilon"] == train.adam_epsilon
    assert parsed["weight_decay"] == train.weight_decay
    assert parsed["weight_decay_anchor"] == train.weight_decay_anchor
    assert parsed["freeze_modules"] == tuple(sorted(train.freeze_modules))
    assert parsed["max_grad_norm"] == train.max_grad_norm
    if lora is not None:
        assert parsed["student_lora_layers"] == lora.layers_to_transform
        assert (parsed["student_lora_modules_to_save"] or ()) == lora.modules_to_save
        if not lora.layers_to_transform and not isinstance(lora.target_modules, str):
            assert parsed["student_lora_target_modules"] == lora.target_modules


def test_tag_vocabularies_are_closed() -> None:
    with pytest.raises(ValueError):
        freeze_suffix(("gpt_neox.layers",))
    with pytest.raises(ValueError):
        lora_scope_suffix(_lora(modules_to_save=("embed_in",)))
    with pytest.raises(ValueError):
        decay_suffix(0.0, anchor=True)
    with pytest.raises(ValueError):
        student_param_tag(_train(freeze_modules=("embed_out",)), _lora())
    assert eps_suffix(1e-8) == "" and decay_suffix(0.0) == ""


def test_identity_extras_only_off_default() -> None:
    assert train_identity_extras(_train()) == {}
    assert set(lora_manifest_blob(_lora())) == {
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
    }
    extras = train_identity_extras(
        _train(weight_decay=4.2, weight_decay_anchor=True, adam_epsilon=0.1)
    )
    assert extras == {"weight_decay": 4.2, "weight_decay_anchor": True, "adam_epsilon": 0.1}


# ---- settings validation --------------------------------------------------------------


@pytest.mark.parametrize(
    "train, lora",
    [
        (_train(optim="sgd", adam_epsilon=0.1), None),
        (_train(weight_decay=-1.0), None),
        (_train(weight_decay_anchor=True), None),
        (_train(weight_decay=0.1, weight_decay_anchor=True), _lora()),
        (_train(weight_decay=0.1, weight_decay_anchor=True, checkpoint_resume=True), None),
        (_train(freeze_modules=("embed_out",)), _lora()),
    ],
)
def test_validate_train_settings_rejects(train, lora) -> None:
    with pytest.raises(ValueError):
        validate_train_settings(train, lora)


# ---- freezing and counts --------------------------------------------------------------


@pytest.mark.parametrize(
    "tag, expected",
    [
        ("body", TOTAL - 2 * EMB),
        ("nohead", TOTAL - EMB),
        ("embin", EMB),
        ("embout", EMB),
    ],
)
def test_freeze_presets_hand_counts(tag, expected) -> None:
    model = _tiny()
    assert trainable_parameter_counts(model)["trainable"] == TOTAL
    apply_freeze_modules(model, FREEZE_BY_TAG[tag])
    counts = trainable_parameter_counts(model)
    assert counts["total"] == TOTAL
    assert counts["trainable"] == expected
    if tag == "embin":
        assert counts["trainable_by_group"] == {"embed_in": EMB}


def test_freeze_unknown_prefix_raises() -> None:
    with pytest.raises(ValueError):
        apply_freeze_modules(_tiny(), ("gpt_neox.nope",))


def test_lora_reach_reaches_peft() -> None:
    from peft import get_peft_model

    model = get_peft_model(
        _tiny(),
        build_lora_config(
            _lora(target_modules=NEOX_LINEAR_MODULES, layers_to_transform=(0, 1))
        ),
    )
    layers = {
        int(n.split("layers.")[1].split(".")[0])
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    assert layers == {0, 1}
    head = get_peft_model(_tiny(), build_lora_config(_lora(modules_to_save=("embed_out",))))
    counts = trainable_parameter_counts(head)["trainable_by_group"]
    assert counts["embed_out"] == EMB and counts["lora"] > 0
    attn = get_peft_model(
        _tiny(), build_lora_config(_lora(target_modules=("query_key_value", "dense")))
    )
    names = [n for n, p in attn.named_parameters() if p.requires_grad]
    assert names and all("attention" in n for n in names)


# ---- optimizer wiring -----------------------------------------------------------------


def _optimizer(tmp_path, optim: str, weight_decay: float, eps: float = 1e-8):
    args = TrainingArguments(
        output_dir=str(tmp_path),
        optim=optim,
        learning_rate=0.1,
        weight_decay=weight_decay,
        adam_epsilon=eps,
        report_to=[],
    )
    trainer = Trainer(model=_tiny(), args=args)
    return trainer.create_optimizer()


def test_weight_decay_and_eps_reach_torch(tmp_path) -> None:
    sgd = _optimizer(tmp_path, "sgd", 0.01)
    decays = sorted(g["weight_decay"] for g in sgd.param_groups)
    assert decays == [0.0, 0.01]
    adam = _optimizer(tmp_path, "adamw_torch", 0.0, eps=0.1)
    assert adam.defaults["eps"] == 0.1
    assert hf_weight_decay(_train(weight_decay=4.2, weight_decay_anchor=True)) == 0.0


def test_anchored_decay_step_hand_value() -> None:
    model = _tiny()
    cb = AnchoredDecayCallback(weight_decay=2.0)
    cb.capture(model)
    w = model.gpt_neox.embed_in.weight
    bias = model.gpt_neox.layers[0].attention.dense.bias
    ln = model.gpt_neox.final_layer_norm.weight
    w0, b0, ln0 = w.detach().clone(), bias.detach().clone(), ln.detach().clone()
    with torch.no_grad():
        w.add_(1.0)
        bias.add_(1.0)
        ln.add_(1.0)
    cb.step([{"lr": 0.1, "params": list(model.parameters())}])
    # θ ← θ − lr·λ·(θ − θ₀) = (θ₀ + 1) − 0.2·1
    torch.testing.assert_close(w.detach(), w0 + 0.8)
    torch.testing.assert_close(bias.detach(), b0 + 1.0)
    torch.testing.assert_close(ln.detach(), ln0 + 1.0)
