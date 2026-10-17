"""Ported optimizers (Muon + AdamW aux, Lion / signSGD, Adam-mini) and their builder.

Expected values are hand-computed with fp32 references; Pythia-160M is built on the
meta device from its published shape (hidden 768, 12 heads, 12 layers, vocab 50304).
"""

from __future__ import annotations

import copy
import math
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import get_peft_model
from transformers import (
    AutoTokenizer,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    Trainer,
    TrainingArguments,
)
from transformers.optimization import Adafactor
from trl import GKDConfig

from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.gkd import trainer_io
from subliminal_learning.gkd.layout import OPTIM_SLUGS
from subliminal_learning.teachers import sft_io
from subliminal_learning.training import optimizers
from subliminal_learning.training.lora_io import build_lora_config
from subliminal_learning.training.optimizers import (
    NS_COEFFICIENTS,
    AdamMini,
    Lion,
    MuonWithAuxAdamW,
    NeoxRole,
    merge_neox_qkv,
    neox_param_role,
    split_neox_qkv,
    zeropower_newton_schulz,
)
from subliminal_learning.training.param_control_io import (
    CUSTOM_OPTIMS,
    HF_OPTIMS,
    build_custom_optimizer,
    check_custom_optimizer_installed,
    hf_optim_name,
    install_custom_optimizer,
    optimizer_resolved,
    validate_train_settings,
)

_DIAG = (1.0, 0.5, 0.1, 0.01)
_NS_DIAG_FP32 = (0.6990, 1.1188, 0.7120, 0.6866)
_THETA0 = ((1.0, -2.0), (0.5, 0.0))
_GRAD = ((0.3, -0.1), (0.0, 2.0))
_PYTHIA_160M_PARAMS = 162_322_944


def _phi5(x: torch.Tensor) -> torch.Tensor:
    a, b, c = NS_COEFFICIENTS
    for _ in range(5):
        x = a * x + b * x**3 + c * x**5
    return x


def _args(tmp_path: Path, **kw) -> TrainingArguments:
    return TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[], **kw)


def _pythia_160m_config() -> GPTNeoXConfig:
    return GPTNeoXConfig(
        vocab_size=50304,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=2048,
        rotary_pct=0.25,
    )


def _meta_pythia_160m() -> GPTNeoXForCausalLM:
    with torch.device("meta"):
        model = GPTNeoXForCausalLM(_pythia_160m_config())
    assert sum(p.numel() for p in model.parameters()) == _PYTHIA_160M_PARAMS
    return model


def _tiny(vocab_size: int = 64, seed: int = 0) -> GPTNeoXForCausalLM:
    torch.manual_seed(seed)
    return GPTNeoXForCausalLM(
        GPTNeoXConfig(
            vocab_size=vocab_size,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
        )
    )


def _lora(r: int = 8, alpha: int = 32) -> LoRAConfig:
    return LoRAConfig(
        r=r, lora_alpha=alpha, lora_dropout=0.0, target_modules="all-linear"
    )


def _names(model: torch.nn.Module) -> dict[int, str]:
    return {id(p): n for n, p in model.named_parameters()}


def _group_of(optimizer: torch.optim.Optimizer, param: torch.Tensor) -> dict:
    (group,) = [
        g for g in optimizer.param_groups if any(p is param for p in g["params"])
    ]
    return group


def _set_grads(model: torch.nn.Module, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn(p.shape, generator=gen)


# ---- Newton-Schulz ------------------------------------------------------------------


def test_ns_diagonal_matches_scalar_map_fp32_and_bf16() -> None:
    sigma = torch.tensor(_DIAG, dtype=torch.float64)
    g = torch.diag(sigma.float())
    fp32 = zeropower_newton_schulz(g, dtype=torch.float32)
    expected = torch.tensor(_NS_DIAG_FP32)
    torch.testing.assert_close(torch.diagonal(fp32), expected, atol=1e-3, rtol=0)
    torch.testing.assert_close(
        torch.diagonal(fp32).double(), _phi5(sigma / sigma.norm()), atol=1e-4, rtol=0
    )
    assert torch.count_nonzero(fp32 - torch.diag(torch.diagonal(fp32))) == 0
    bf16 = zeropower_newton_schulz(g)
    assert bf16.dtype == torch.bfloat16
    torch.testing.assert_close(
        torch.diagonal(bf16).float(), expected, atol=0.07, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(192, 64), (64, 192)])
def test_ns_rectangular_singular_values_in_band(dtype, shape) -> None:
    torch.manual_seed(0)
    out = zeropower_newton_schulz(torch.randn(shape), dtype=dtype)
    assert out.shape == shape
    s = torch.linalg.svdvals(out.float())
    assert 0.65 <= float(s.min()) and float(s.max()) <= 1.2


def test_ns_zero_is_zero_and_scale_invariant() -> None:
    zero = zeropower_newton_schulz(torch.zeros(8, 64))
    assert torch.count_nonzero(zero) == 0 and not zero.isnan().any()
    torch.manual_seed(1)
    g = torch.randn(64, 32)
    torch.testing.assert_close(
        zeropower_newton_schulz(7 * g, dtype=torch.float32),
        zeropower_newton_schulz(g, dtype=torch.float32),
        atol=1e-5,
        rtol=0,
    )


# ---- Muon ---------------------------------------------------------------------------


def test_muon_one_step_by_hand_leaves_grad_untouched() -> None:
    g = torch.diag(torch.tensor(_DIAG))
    p = torch.nn.Parameter(torch.eye(4))
    p.grad = g.clone()
    opt = MuonWithAuxAdamW(
        [{"params": [p], "use_muon": True, "weight_decay": 0.5}], lr=0.1
    )
    opt.step()
    torch.testing.assert_close(
        torch.diagonal(p.detach()),
        torch.tensor([0.92204, 0.90525, 0.92152, 0.92254]),
        atol=3e-3,
        rtol=0,
    )
    assert torch.count_nonzero(p.detach() - torch.diag(torch.diagonal(p.detach()))) == 0
    torch.testing.assert_close(opt.state[p]["momentum_buffer"], 0.05 * g)
    assert torch.equal(p.grad, g)


def test_muon_fused_qkv_step_equals_three_separate_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NS runs in fp32 here, so batched and per-matrix NS agree to fp32 rounding."""
    monkeypatch.setattr(
        optimizers,
        "zeropower_newton_schulz",
        partial(zeropower_newton_schulz, dtype=torch.float32),
    )
    heads, hidden, lr, wd, mu = 12, 768, 0.01, 0.1, 0.95
    head_dim = hidden // heads
    torch.manual_seed(0)
    theta0 = 0.02 * torch.randn(3 * hidden, hidden)
    g = torch.randn(3 * hidden, hidden)
    p = torch.nn.Parameter(theta0.clone())
    p.grad = g.clone()
    opt = MuonWithAuxAdamW(
        [{"params": [p], "use_muon": True, "qkv_heads": heads, "weight_decay": wd}],
        lr=lr,
    )
    opt.step()
    nesterov = g.lerp((1 - mu) * g, mu)
    expected = theta0 * (1 - lr * wd)
    for k in range(3):
        block = nesterov.view(heads, 3, head_dim, hidden)[:, k].reshape(hidden, hidden)
        ortho = zeropower_newton_schulz(block, dtype=torch.float32)
        expected.view(heads, 3, head_dim, hidden)[:, k] -= (
            lr * 0.2 * math.sqrt(hidden) * ortho.view(heads, head_dim, hidden)
        )
    torch.testing.assert_close(p.detach(), expected, atol=1e-6, rtol=0)
    fused = theta0 * (1 - lr * wd) - lr * 0.2 * math.sqrt(3 * hidden) * (
        zeropower_newton_schulz(nesterov, dtype=torch.float32)
    )
    assert not torch.allclose(p.detach(), fused, atol=1e-4)


def test_qkv_split_matches_gpt_neox_layout() -> None:
    torch.manual_seed(0)
    model = GPTNeoXForCausalLM(
        GPTNeoXConfig(
            vocab_size=10,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=8,
            attn_implementation="eager",
        )
    ).eval()
    attn = model.gpt_neox.layers[0].attention
    w, b = attn.query_key_value.weight, attn.query_key_value.bias
    split_w = split_neox_qkv(w, 2)
    split_b = split_neox_qkv(b[:, None], 2)[..., 0]
    assert split_w.shape == (3, 16, 16)
    x = torch.randn(1, 1, 16)
    # transformers 4.54 modeling_gpt_neox.py L154-157
    ref = attn.query_key_value(x).view(1, 1, -1, 24).transpose(1, 2).chunk(3, dim=-1)
    for k in range(3):
        ours = F.linear(x, split_w[k], split_b[k]).view(1, 1, 2, 8).transpose(1, 2)
        torch.testing.assert_close(ours, ref[k])

    captured: dict[str, torch.Tensor] = {}

    def keep(key: str):  # noqa: ANN202
        def hook(module, inputs, *_) -> None:  # noqa: ANN001
            captured[key] = inputs[0]

        return hook

    attn.query_key_value.register_forward_hook(keep("x"))
    attn.dense.register_forward_pre_hook(keep("v"))
    with torch.no_grad():
        model(torch.tensor([[3]]))
    # One token attends only to itself, so the attention output is V.
    torch.testing.assert_close(
        captured["v"], F.linear(captured["x"], split_w[2], split_b[2])
    )

    assert torch.equal(merge_neox_qkv(split_w, 2), w)
    batched = zeropower_newton_schulz(split_w, dtype=torch.float32)
    for k in range(3):
        torch.testing.assert_close(
            batched[k],
            zeropower_newton_schulz(split_w[k], dtype=torch.float32),
            atol=1e-6,
            rtol=0,
        )


def test_muon_groups_on_pythia_160m(tmp_path: Path) -> None:
    model = _meta_pythia_160m()
    names = _names(model)
    opt = build_custom_optimizer(model, "muon", _args(tmp_path, weight_decay=0.1))
    assert isinstance(opt, MuonWithAuxAdamW)
    muon = [p for g in opt.param_groups if g["use_muon"] for p in g["params"]]
    qkv = [p for g in opt.param_groups if g.get("qkv_heads") == 12 for p in g["params"]]
    aux = [p for g in opt.param_groups if not g["use_muon"] for p in g["params"]]
    assert len(muon) == 48 and len(qkv) == 12
    assert all(names[id(p)].endswith("query_key_value.weight") for p in qkv)
    assert sorted(names[id(p)] for p in aux if p.ndim == 2) == [
        "embed_out.weight",
        "gpt_neox.embed_in.weight",
    ]
    assert sum(p.ndim == 1 for p in aux) == 98 and len(aux) == 100
    assert {
        (p.ndim, g["weight_decay"]) for g in opt.param_groups for p in g["params"]
    } == {(2, 0.1), (1, 0.0)}

    lora = get_peft_model(_meta_pythia_160m(), build_lora_config(_lora()))
    opt = build_custom_optimizer(lora, "muon", _args(tmp_path))
    assert [(g["use_muon"], g["qkv_heads"]) for g in opt.param_groups] == [(True, 0)]
    assert len(opt.param_groups[0]["params"]) == 96


def test_muon_aux_steps_exactly_like_torch_adamw(tmp_path: Path) -> None:
    model = _tiny()
    ref = copy.deepcopy(model)
    args = _args(tmp_path, learning_rate=1e-2, weight_decay=0.1)
    opt = build_custom_optimizer(model, "muon", args)
    aux = [g for g in opt.param_groups if not g["use_muon"]]
    index = {n: i for i, (n, _) in enumerate(model.named_parameters())}
    ref_params = list(ref.parameters())
    names = _names(model)
    ref_opt = torch.optim.AdamW(
        [
            {
                "params": [ref_params[index[names[id(p)]]] for p in g["params"]],
                "weight_decay": g["weight_decay"],
            }
            for g in aux
        ],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )
    before = copy.deepcopy(model)
    for seed in range(3):
        _set_grads(model, seed)
        _set_grads(ref, seed)
        opt.step()
        ref_opt.step()
    for g in aux:
        for p in g["params"]:
            q = ref_params[index[names[id(p)]]]
            assert torch.equal(p, q), names[id(p)]
    moved = dict(before.named_parameters())
    for g in opt.param_groups:
        if g["use_muon"]:
            assert all(not torch.equal(p, moved[names[id(p)]]) for p in g["params"])


# ---- Lion / signSGD -----------------------------------------------------------------


def test_lion_one_step_by_hand() -> None:
    p = torch.nn.Parameter(torch.tensor(_THETA0))
    p.grad = torch.tensor(_GRAD)
    opt = Lion([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.5)
    opt.state[p]["exp_avg"] = torch.tensor([[0.1, 0.2], [-0.5, 0.0]])
    opt.step()
    torch.testing.assert_close(p.detach(), torch.tensor([[0.85, -2.0], [0.575, -0.1]]))
    torch.testing.assert_close(
        opt.state[p]["exp_avg"], torch.tensor([[0.102, 0.197], [-0.495, 0.02]])
    )


def test_signsgd_one_step_by_hand_keeps_no_state() -> None:
    p = torch.nn.Parameter(torch.tensor(_THETA0))
    p.grad = torch.tensor(_GRAD)
    opt = Lion([p], lr=0.1, betas=(0.0, 0.0), weight_decay=0.5)
    opt.step()
    torch.testing.assert_close(p.detach(), torch.tensor([[0.85, -1.8], [0.475, -0.1]]))
    assert len(opt.state) == 0


# ---- Adafactor (HF built-in) --------------------------------------------------------


def test_adafactor_is_built_by_hf(tmp_path: Path) -> None:
    args = _args(tmp_path, optim="adafactor", learning_rate=1e-5)
    cls, kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    assert cls is Adafactor
    assert kwargs == {"lr": 1e-5, "scale_parameter": False, "relative_step": False}
    assert build_custom_optimizer(_tiny(), "adafactor", args) is None
    assert hf_optim_name("adafactor") == "adafactor"


# ---- Adam-mini ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "block, expected",
    [
        ("rows", [[0.986584, -1.995528], [0.5, -0.014142]]),
        ("whole", [[0.997037, -1.999012], [0.5, -0.019755]]),
        ("elementwise", [[0.99, -1.99], [0.5, -0.01]]),
    ],
)
def test_adam_mini_step1_per_block(block, expected) -> None:
    p = torch.nn.Parameter(torch.tensor(_THETA0))
    p.grad = torch.tensor(_GRAD)
    opt = AdamMini([{"params": [p], "block": block}], lr=0.01)
    opt.step()
    torch.testing.assert_close(p.detach(), torch.tensor(expected), atol=1e-6, rtol=0)


def test_adam_mini_partition_on_pythia_160m(tmp_path: Path) -> None:
    model = _meta_pythia_160m()
    opt = build_custom_optimizer(model, "adam_mini", _args(tmp_path))
    assert isinstance(opt, AdamMini)
    params = dict(model.named_parameters())
    layer = "gpt_neox.layers.0."

    def block(name: str) -> tuple[str, int]:
        group = _group_of(opt, params[name])
        return group["block"], group["block_rows"]

    qkv = params[layer + "attention.query_key_value.weight"]
    assert block(layer + "attention.query_key_value.weight") == ("rows", 64)
    assert qkv.size(0) // 64 == 36 and 64 * qkv.size(1) == 49_152
    assert block(layer + "attention.dense.weight") == ("rows", 1)
    assert block(layer + "mlp.dense_4h_to_h.weight") == ("rows", 1)
    assert block("gpt_neox.embed_in.weight") == ("rows", 1)
    assert block("embed_out.weight") == ("rows", 1)
    assert block("gpt_neox.final_layer_norm.weight") == ("whole", 1)
    assert block(layer + "input_layernorm.weight") == ("whole", 1)
    assert block(layer + "attention.query_key_value.bias") == ("elementwise", 1)

    lora = get_peft_model(_meta_pythia_160m(), build_lora_config(_lora()))
    opt = build_custom_optimizer(lora, "adam_mini", _args(tmp_path))
    assert [(g["block"], g["block_rows"]) for g in opt.param_groups] == [("rows", 1)]
    assert len(opt.param_groups[0]["params"]) == 96


# ---- builder, vocabulary, validation ------------------------------------------------


@pytest.mark.parametrize("optim", CUSTOM_OPTIMS)
def test_resume_round_trip_steps_identically(tmp_path: Path, optim: str) -> None:
    args = _args(tmp_path, learning_rate=1e-2, weight_decay=0.1)
    model = _tiny()
    opt = build_custom_optimizer(model, optim, args)
    for seed in range(2):
        _set_grads(model, seed)
        opt.step()
    path = tmp_path / "optimizer.pt"
    torch.save(opt.state_dict(), path)
    clone = copy.deepcopy(model)
    resumed = build_custom_optimizer(clone, optim, args)
    resumed.load_state_dict(torch.load(path, weights_only=True))
    _set_grads(model, 7)
    _set_grads(clone, 7)
    opt.step()
    resumed.step()
    for (name, a), (_, b) in zip(model.named_parameters(), clone.named_parameters()):
        assert torch.equal(a, b), name


@pytest.mark.parametrize("optim", HF_OPTIMS)
def test_builder_leaves_hf_optimizers_to_hf(tmp_path: Path, optim: str) -> None:
    assert build_custom_optimizer(_tiny(), optim, _args(tmp_path)) is None
    assert hf_optim_name(optim) == optim


def test_custom_optimizers_get_the_adamw_placeholder() -> None:
    assert {hf_optim_name(o) for o in CUSTOM_OPTIMS} == {"adamw_torch"}


def test_optim_vocabulary_matches_tag_slugs() -> None:
    assert set(HF_OPTIMS) | set(CUSTOM_OPTIMS) == set(OPTIM_SLUGS)
    assert not set(HF_OPTIMS) & set(CUSTOM_OPTIMS)


def _train(**kw) -> TrainConfig:
    base = dict(
        learning_rate=1e-4,
        n_epochs=1,
        batch_size=8,
        max_length=64,
        max_grad_norm=1.0,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        seed=1,
    )
    return TrainConfig(**{**base, **kw})


@pytest.mark.parametrize(
    "kw",
    [
        {"optim": "adagrad"},
        {"optim": "muon", "optim_momentum": 0.9},
        {"optim": "lion", "adam_epsilon": 1e-6},
        {"optim": "adam_mini", "adam_epsilon": 1e-6},
    ],
)
def test_validate_rejects_bad_optimizer_settings(kw) -> None:
    with pytest.raises(ValueError):
        validate_train_settings(_train(**kw), None)


@pytest.mark.parametrize("optim", HF_OPTIMS + CUSTOM_OPTIMS)
def test_validate_accepts_every_known_optimizer(optim: str) -> None:
    validate_train_settings(_train(optim=optim), None)


@pytest.mark.parametrize(
    "name, role",
    [
        ("gpt_neox.layers.3.attention.query_key_value.weight", NeoxRole.QKV),
        ("gpt_neox.layers.3.attention.dense.weight", NeoxRole.MATRIX),
        ("gpt_neox.layers.3.mlp.dense_h_to_4h.bias", NeoxRole.BIAS),
        ("gpt_neox.final_layer_norm.weight", NeoxRole.NORM),
        (
            "base_model.model.embed_out.modules_to_save.default.weight",
            NeoxRole.EMBEDDING,
        ),
        (
            "base_model.model.gpt_neox.layers.0.attention.query_key_value"
            ".lora_A.default.weight",
            NeoxRole.LORA,
        ),
    ],
)
def test_neox_param_role(name: str, role: NeoxRole) -> None:
    assert neox_param_role(name) is role


def test_unknown_parameter_name_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown GPT-NeoX parameter"):
        neox_param_role("model.layers.0.self_attn.q_proj.weight")
    model = torch.nn.Linear(4, 4)
    model.config = SimpleNamespace(num_attention_heads=2, hidden_size=4)
    with pytest.raises(ValueError, match="unknown GPT-NeoX parameter"):
        build_custom_optimizer(model, "muon", _args(tmp_path))


def test_install_rejects_args_that_disagree_with_optim(tmp_path: Path) -> None:
    trainer = SimpleNamespace(model=_tiny())
    with pytest.raises(ValueError, match="does not match"):
        install_custom_optimizer(trainer, "muon", _args(tmp_path, optim="sgd"))
    install_custom_optimizer(trainer, "muon", _args(tmp_path))
    assert isinstance(trainer.optimizer, MuonWithAuxAdamW)


def test_optimizer_resolved_records_ported_groups_in_full(tmp_path: Path) -> None:
    model = _tiny()
    adamw = torch.optim.AdamW(model.parameters(), lr=1e-3)
    record = optimizer_resolved(SimpleNamespace(optimizer=adamw))
    assert record["class"] == "torch.optim.adamw.AdamW"
    assert set(record["param_groups"][0]) == {"n_tensors", "lr", "weight_decay"}
    muon = build_custom_optimizer(model, "muon", _args(tmp_path))
    record = optimizer_resolved(SimpleNamespace(optimizer=muon))
    assert record["class"].endswith("optimizers.MuonWithAuxAdamW")
    qkv = [g for g in record["param_groups"] if g.get("qkv_heads")]
    assert len(qkv) == 1 and qkv[0]["n_tensors"] == 2
    assert qkv[0]["momentum"] == 0.95 and qkv[0]["ns_steps"] == 5
    aux = [g for g in record["param_groups"] if not g["use_muon"]]
    assert all(g["betas"] == [0.9, 0.999] and "momentum" not in g for g in aux)


def test_installed_check_rejects_a_missing_or_wrong_optimizer(tmp_path: Path) -> None:
    check_custom_optimizer_installed(SimpleNamespace(optimizer=None), "adamw_torch")
    with pytest.raises(RuntimeError, match="placeholder"):
        check_custom_optimizer_installed(SimpleNamespace(optimizer=None), "muon")
    lion = build_custom_optimizer(_tiny(), "lion", _args(tmp_path))
    check_custom_optimizer_installed(SimpleNamespace(optimizer=lion), "lion")
    with pytest.raises(RuntimeError):
        check_custom_optimizer_installed(SimpleNamespace(optimizer=lion), "signsgd")
    with pytest.raises(RuntimeError):
        check_custom_optimizer_installed(SimpleNamespace(optimizer=lion), "adam_mini")


# ---- hard-label SFT wiring (CPU, tiny model) ----------------------------------------


@pytest.fixture(scope="module")
def tokenizer() -> AutoTokenizer:
    try:
        tok = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-70m", local_files_only=True
        )
    except OSError as exc:
        pytest.skip(f"pythia-70m tokenizer not cached: {exc}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


_CLASS_OF = {
    "muon": MuonWithAuxAdamW,
    "lion": Lion,
    "signsgd": Lion,
    "adam_mini": AdamMini,
}


@pytest.mark.parametrize("mode", ["lora", "full"])
@pytest.mark.parametrize("optim", CUSTOM_OPTIMS)
def test_sft_trains_with_each_custom_optimizer(
    tokenizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, optim: str, mode: str
) -> None:
    monkeypatch.setattr(
        sft_io, "TrainingArguments", partial(TrainingArguments, use_cpu=True)
    )
    model = _tiny(len(tokenizer))
    lora = _lora(r=2, alpha=4) if mode == "lora" else None
    if lora is not None:
        model = get_peft_model(model, build_lora_config(lora))
    before = {
        n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad
    }
    result = sft_io.run_teacher_sft_completion_only(
        model=model,
        tokenizer=tokenizer,
        train_dataset=Dataset.from_list(
            [
                {"prompt": f"Numbers {i}, {i + 1},", "completion": f" {i + 2}"}
                for i in range(8)
            ]
        ),
        lora_cfg=lora,
        train_cfg=TrainConfig(
            learning_rate=1e-2,
            n_epochs=1,
            batch_size=2,
            max_length=64,
            max_grad_norm=1.0,
            warmup_ratio=0.0,
            lr_scheduler_type="cosine",
            seed=3,
            optim=optim,
        ),
        output_dir=tmp_path / "out",
        run_name="test",
        report_to=[],
        max_steps=3,
    )
    cls = _CLASS_OF[optim]
    assert result.optimizer["class"] == f"{cls.__module__}.{cls.__qualname__}"
    assert result.n_steps == 3 and math.isfinite(result.final_train_loss)
    after = dict(model.named_parameters())
    assert all(not torch.equal(after[n], p) for n, p in before.items())


def _numbers_dataset(n: int = 8) -> Dataset:
    return Dataset.from_list(
        [
            {"prompt": f"Numbers {i}, {i + 1},", "completion": f" {i + 2}"}
            for i in range(n)
        ]
    )


def test_sft_refuses_to_train_without_the_custom_optimizer(
    tokenizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sft_io, "TrainingArguments", partial(TrainingArguments, use_cpu=True)
    )
    monkeypatch.setattr(sft_io, "install_custom_optimizer", lambda *a: None)
    with pytest.raises(RuntimeError, match="placeholder"):
        sft_io.run_teacher_sft_completion_only(
            model=_tiny(len(tokenizer)),
            tokenizer=tokenizer,
            train_dataset=_numbers_dataset(),
            lora_cfg=None,
            train_cfg=_train(optim="muon"),
            output_dir=tmp_path / "out",
            run_name="test",
            report_to=[],
            max_steps=1,
        )


# ---- on-policy GKD wiring (CPU, tiny model) -----------------------------------------


# Muon (aux groups + QKV split) runs by default; the rest add ~8 s each on CPU.
_GKD_CASES = [
    pytest.param(optim, mode, marks=() if optim == "muon" else pytest.mark.slow)
    for optim in CUSTOM_OPTIMS
    for mode in ("lora", "full")
]


@pytest.mark.parametrize("optim, mode", _GKD_CASES)
def test_gkd_trainer_installs_and_steps_custom_optimizer(
    tokenizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, optim: str, mode: str
) -> None:
    monkeypatch.setattr(
        trainer_io, "GKDConfig", partial(GKDConfig, use_cpu=True, bf16=False)
    )
    args = trainer_io.build_gkd_config(
        _train(learning_rate=1e-2, batch_size=2, warmup_ratio=0.0, optim=optim),
        output_dir=tmp_path / "ckpt",
        run_name="test",
        lmbda=1.0,
        beta=0.5,
        num_epochs=1,
        max_steps=2,
        max_new_tokens=4,
        report_to=[],
    )
    trainer = trainer_io.SubliminalGKDTrainer(
        _tiny(len(tokenizer)),
        _tiny(len(tokenizer), seed=1),
        args=args,
        train_dataset=_numbers_dataset(4),
        processing_class=tokenizer,
        peft_config=build_lora_config(_lora(r=2, alpha=4)) if mode == "lora" else None,
        screen_forms=("zebra",),
        optim=optim,
    )
    assert type(trainer.optimizer) is _CLASS_OF[optim]
    check_custom_optimizer_installed(trainer, optim)
    before = {
        n: p.detach().clone()
        for n, p in trainer.model.named_parameters()
        if p.requires_grad
    }
    trainer.train()
    assert trainer.state.global_step == 2
    assert optimizer_resolved(trainer)["class"].endswith(_CLASS_OF[optim].__qualname__)
    after = dict(trainer.model.named_parameters())
    still = [n for n, p in before.items() if torch.equal(after[n], p)]
    assert not still, still
