import copy
import json
import math
import random

import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from pipelines import blog_readout_pipeline as pipe
from subliminal_learning.evaluation import imitation
from subliminal_learning.evaluation import imitation_io as io
from subliminal_learning.students.dataset_io import subsample_rows


def _tiny(seed: int) -> GPTNeoXForCausalLM:
    torch.manual_seed(seed)
    cfg = GPTNeoXConfig(
        vocab_size=40,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
    )
    return GPTNeoXForCausalLM(cfg).eval()


_EXAMPLES = [([3, 7, 9, 11, 2, 5], 3), ([4, 8, 1], 1), ([6, 6, 12, 30, 31, 9, 2, 0], 5)]


@pytest.mark.parametrize("n,size", [(237, 100), (11_103, 10_000), (50, 80)])
def test_heldout_indices_complement_of_subsample_rows(n, size):
    rows = [{"i": i} for i in range(n)]
    sub = subsample_rows(rows, size, 42)
    assert [rows[i] for i in imitation.subsample_indices(n, size, 42)] == sub
    held = imitation.heldout_indices(n, size, 42)
    assert held == sorted(set(range(n)) - {r["i"] for r in sub})
    picked = imitation.seeded_subset(held, 1000, imitation.EVAL_SEED)
    assert picked == sorted(picked) and set(picked) <= set(held)
    assert len(picked) == min(1000, len(held))
    assert picked == imitation.seeded_subset(reversed(held), 1000, imitation.EVAL_SEED)


def test_sft_examples_match_training_tokenization():
    class Tok:
        eos_token_id = 0

        def __call__(self, text, add_special_tokens):
            assert add_special_tokens is False
            return {"input_ids": [ord(c) for c in text]}

    ids, n_prompt = io.sft_examples(Tok(), [{"prompt": "ab", "completion": "cde"}])[0]
    assert ids == [97, 98, 99, 100, 101, 0] and n_prompt == 2


def test_completion_ce_equals_hand_loop():
    model = _tiny(0)
    total, count = 0.0, 0
    for ids, n_prompt in _EXAMPLES:
        logits = model(input_ids=torch.tensor([ids])).logits[0]
        for t in range(n_prompt - 1, len(ids) - 1):
            total -= F.log_softmax(logits[t], -1)[ids[t + 1]].item()
            count += 1
    got = io.completion_ce(model, _EXAMPLES, "cpu", pad_id=0, batch_size=2)
    assert got["n_tokens"] == count == 3 + 2 + 3
    assert got["ce"] == pytest.approx(total / count, rel=1e-5)


def test_reverse_kl_equals_manual_sum_and_zero_for_self():
    student, teacher = _tiny(0), _tiny(1)
    total, count = 0.0, 0
    for ids, n_prompt in _EXAMPLES:
        s = student(input_ids=torch.tensor([ids])).logits[0]
        t = teacher(input_ids=torch.tensor([ids])).logits[0]
        for pos in range(n_prompt - 1, len(ids) - 1):
            ls, lt = F.log_softmax(s[pos], -1), F.log_softmax(t[pos], -1)
            total += (ls.exp() * (ls - lt)).sum().item()
            count += 1
    got = io.reverse_kl(student, [teacher, student], _EXAMPLES, "cpu", 0, 2)
    assert got[0]["kl"] == pytest.approx(total / count, rel=1e-5)
    assert got[0]["n_tokens"] == count
    assert abs(got[1]["kl"]) < 1e-6


@pytest.mark.parametrize(
    "name,group",
    [
        ("gpt_neox.embed_in.weight", "embed_in"),
        ("embed_out.weight", "embed_out"),
        ("gpt_neox.layers.0.input_layernorm.weight", "layernorm"),
        ("gpt_neox.layers.3.post_attention_layernorm.bias", "layernorm"),
        ("gpt_neox.final_layer_norm.weight", "layernorm"),
        ("gpt_neox.layers.3.attention.query_key_value.weight", "attention"),
        ("gpt_neox.layers.3.attention.dense.bias", "attention"),
        ("gpt_neox.layers.3.mlp.dense_h_to_4h.weight", "mlp"),
        ("gpt_neox.layers.3.mlp.dense_4h_to_h.bias", "mlp"),
        (
            "base_model.model.gpt_neox.layers.0.attention.dense.lora_A.default.weight",
            "lora",
        ),
        ("something_else", "other"),
    ],
)
def test_param_group(name, group):
    assert imitation.param_group(name) == group


def test_every_neox_parameter_has_a_module_group():
    groups = {imitation.param_group(n) for n, _ in _tiny(0).named_parameters()}
    assert groups == set(imitation.UPDATE_GROUPS) - {"other"}


def test_update_norms_equal_hand_values():
    base = _tiny(0)
    model = copy.deepcopy(base)
    params = dict(model.named_parameters())
    h2 = params["gpt_neox.layers.0.mlp.dense_h_to_4h.weight"]
    ln = params["gpt_neox.layers.1.post_attention_layernorm.weight"]
    with torch.no_grad():
        params["embed_out.weight"].add_(0.5)
        params["embed_out.weight"][7].add_(1.5)
        h2.add_(0.1)
        ln.add_(0.2)
    width = params["embed_out.weight"].shape[1]
    sq = io.update_sq_norms(model, base)
    expected_head = 0.25 * (40 - 1) * width + 4.0 * width
    assert sq["embed_out"] == pytest.approx(expected_head)
    assert sq["mlp"] == pytest.approx(0.01 * h2.numel())
    assert sq["layernorm"] == pytest.approx(0.04 * ln.numel())
    assert sq["embed_in"] == sq["attention"] == 0.0
    summary = imitation.update_summary(sq)
    total = expected_head + 0.01 * h2.numel() + 0.04 * ln.numel()
    assert summary["norm_total"] == pytest.approx(math.sqrt(total))
    assert summary["share_mlp"] == pytest.approx(0.01 * h2.numel() / total)
    assert sum(summary[f"share_{g}"] for g in imitation.UPDATE_GROUPS) == pytest.approx(
        1
    )
    rows = io.head_row_norms(model, base)
    head = imitation.head_row_summary(rows, [1, 2], 7)
    assert head["head_delta_carrier_mean"] == pytest.approx(0.5 * math.sqrt(width))
    assert head["head_delta_animal"] == pytest.approx(2.0 * math.sqrt(width))
    assert head["head_animal_over_carrier"] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        imitation.head_row_summary(rows, [7, 2], 7)


def test_update_norms_of_merged_lora_with_trained_head():
    base = _tiny(0)
    peft = get_peft_model(
        copy.deepcopy(base),
        LoraConfig(
            r=2,
            lora_alpha=4,
            target_modules=["query_key_value"],
            modules_to_save=["embed_out"],
        ),
    )
    torch.manual_seed(3)
    expected_attn = 0.0
    with torch.no_grad():
        for name, p in peft.named_parameters():
            if "lora_B" in name:
                p.normal_()
        for module in peft.modules():
            if hasattr(module, "lora_A") and "default" in module.lora_A:
                delta = (
                    2.0
                    * module.lora_B["default"].weight
                    @ module.lora_A["default"].weight
                )
                expected_attn += delta.double().pow(2).sum().item()
        head = peft.base_model.model.embed_out.modules_to_save["default"].weight
        head.add_(0.3)
    sq = io.update_sq_norms(peft.merge_and_unload(), base)
    assert sq["attention"] == pytest.approx(expected_attn, rel=1e-4)
    assert sq["embed_out"] == pytest.approx(0.09 * head.numel(), rel=1e-4)
    assert sq["mlp"] < 1e-12 and sq["embed_in"] == 0.0


def test_update_norms_reject_unmerged_lora():
    base = _tiny(0)
    peft = get_peft_model(
        copy.deepcopy(base), LoraConfig(r=2, target_modules=["dense"])
    )
    with pytest.raises(ValueError):
        io.update_sq_norms(peft, base)


def test_top_token_ids_ties_and_exclusion():
    got = imitation.top_token_ids([[5, 5, 3, 9], [3, 9, 9, 1]], k=3, exclude=[1])
    assert got == [9, 3, 5]
    assert imitation.top_token_ids([[5, 5, 3]], k=3, exclude=[5]) == [3]


def test_shard_plan_deterministic_balanced_and_unmixed():
    units = [{"g": "a", "k": f"a{i:03d}"} for i in range(85)]
    units += [{"g": "b", "k": f"b{i:03d}"} for i in range(7)]
    shuffled = units[:]
    random.Random(1).shuffle(shuffled)
    kw = {"group_of": lambda u: (u["g"],), "key_of": lambda u: u["k"]}
    shards = imitation.plan_shards(units, max_per_shard=40, **kw)
    assert shards == imitation.plan_shards(shuffled, max_per_shard=40, **kw)
    assert [len(s) for s in shards] == [29, 28, 28, 7]
    assert all(len({u["g"] for u in s}) == 1 for s in shards)
    assert [u["k"] for s in shards for u in s] == sorted(u["k"] for u in units)
    with pytest.raises(ValueError):
        imitation.plan_shards(units + units[:1], **kw)


def _fake_build(tmp_path, finished=(0, 1)):
    build = tmp_path / "build-x"
    build.mkdir()
    shards = [
        {"shard": 0, "channel": "hard", "group": "h", "n_units": 1},
        {"shard": 1, "channel": "soft", "group": "s", "n_units": 1},
    ]
    (build / "manifest.json").write_text(json.dumps({"shards": shards}))
    pd.DataFrame([{"key": "h1", "tier": "p", "shard": 0}]).to_csv(
        build / "units_hard.csv", index=False
    )
    pd.DataFrame([{"key": "s1", "kind": "treat", "shard": 1}]).to_csv(
        build / "units_soft.csv", index=False
    )
    rows = {
        0: {"hard": [{"key": "h1", "pool": "dog", "ce_student_heldout": 1.0}]},
        1: {"soft": [{"key": "s1", "kl_student_own": 0.2}]},
    }
    for s in finished:
        for name, r in rows[s].items():
            pd.DataFrame(r).to_csv(build / f"rows_{name}_{s:04d}.csv", index=False)
        pd.DataFrame([{"key": f"{'hs'[s]}1", "norm_total": 1.0}]).to_csv(
            build / f"rows_e5_{s:04d}.csv", index=False
        )
        (build / f"shard_{s:04d}.json").write_text(
            json.dumps({"channel": shards[s]["channel"], "code_sha": "abc"})
        )
    return build


def test_aggregate_joins_identity_and_is_create_only(tmp_path, monkeypatch):
    monkeypatch.setattr(pipe, "_stamp", lambda: "20261010T000000Z")
    build = _fake_build(tmp_path)
    out = pipe.aggregate(build)
    hard = pd.read_csv(out / "hard_rows.csv")
    assert list(hard["tier"]) == ["p"] and list(hard["pool"]) == ["dog"]
    e5 = pd.read_csv(out / "e5_rows.csv")
    assert sorted(e5["key"]) == ["h1", "s1"]
    assert json.loads((out / "manifest.json").read_text())["partial"] is False
    with pytest.raises(FileExistsError):
        pipe.aggregate(build)


def test_aggregate_refuses_unfinished_shards_unless_partial(tmp_path, monkeypatch):
    monkeypatch.setattr(pipe, "_stamp", lambda: "20261010T000000Z")
    build = _fake_build(tmp_path, finished=(0,))
    with pytest.raises(RuntimeError):
        pipe.aggregate(build)
    out = pipe.aggregate(build, partial=True)
    assert json.loads((out / "manifest.json").read_text())["shards_unfinished"] == [1]
    assert len(pd.read_csv(out / "e5_rows.csv")) == 1
