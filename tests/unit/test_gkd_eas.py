"""GKD-EAS unit tests (CPU, no models): planted stacks, dict-manifest gates, cache keys."""

from __future__ import annotations

import json

import pytest
import torch

from subliminal_learning.steering.gkd_eas import (
    GkdEasKey,
    build_gkd_entry,
    cache_key,
    check_prompt_pool_sha,
    check_same_base,
    check_slot_count,
    direction_cosine,
    leaf_identity,
    meta_matches,
    parse_teacher_ftseed,
    proxy_stats,
    random_direction_floor,
    teacher_null_delta,
    unit_mean_vector,
)
from subliminal_learning.steering.vectors import mean_activations

_S410, _H410 = 25, 1024
_S160, _H160 = 13, 768


def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _randn(*shape: int, seed: int = 0) -> torch.Tensor:
    return torch.randn(*shape, generator=_gen(seed))


def _orthogonal_to(v: torch.Tensor, seed: int = 1) -> torch.Tensor:
    w = _randn(*v.shape, seed=seed)
    proj = (w * v).sum(-1, keepdim=True) / (v * v).sum(-1, keepdim=True)
    return w - proj * v


# ------------------------------------------------------------------- planted directions


def test_planted_direction_cos_one_and_eas_mean_one_25_slots():
    base = _randn(_S410, _H410, seed=0)
    v = _randn(_S410, _H410, seed=1)
    v_student = (base + 0.3 * v) - base
    v_teacher = (base + 2.0 * v) - base
    cos = direction_cosine(v_student, v_teacher)
    assert cos.shape == (_S410,)
    assert torch.allclose(cos, torch.ones(_S410), atol=1e-5)
    stats = proxy_stats({"bear/ftseed0": cos.tolist()})
    assert stats["eas_mean"] == pytest.approx(1.0, abs=1e-5)
    assert stats["eas_mean_by_teacher"]["bear/ftseed0"] == pytest.approx(1.0, abs=1e-5)
    assert len(stats["per_layer_mean"]) == _S410


def test_orthogonal_plant_cos_zero():
    v = _randn(_S410, _H410, seed=2)
    w = _orthogonal_to(v)
    assert direction_cosine(v, w).abs().max().item() < 1e-6


def test_slot_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        direction_cosine(torch.zeros(_S160, _H160), torch.zeros(_S410, _H410))
    with pytest.raises(ValueError, match="slots"):
        check_slot_count(_S160, 24)
    check_slot_count(_S410, 24)
    check_slot_count(_S160, 12)


# ------------------------------------------------------------------- manifest gates


def test_base_repo_mismatch_raises_on_dict_manifests():
    leaf = "EleutherAI/pythia-410m@step143000"
    check_same_base(leaf, {"bear/ftseed0": leaf, "cat/ftseed1": leaf})
    with pytest.raises(ValueError, match="cat/ftseed1"):
        check_same_base(
            leaf, {"bear/ftseed0": leaf, "cat/ftseed1": "EleutherAI/pythia-160m@step143000"}
        )


def test_prompt_sha_mismatch_across_leaves_raises():
    check_prompt_pool_sha({"/a": "abc", "/b": "abc"}, "abc")
    with pytest.raises(ValueError, match="/b"):
        check_prompt_pool_sha({"/a": "abc", "/b": "xyz"}, "abc")


def _leaf_manifest(**overrides) -> dict:
    m = {
        "base_model_repo": "EleutherAI/pythia-410m@step143000",
        "target_animal": "bear",
        "finetune_mode": "lora",
        "prompt_pool_dir": "/pool",
        "prompt_pool_manifest_sha256": "abc",
        "control": False,
    }
    m.update(overrides)
    return m


def test_leaf_identity_rejects_non_lora_and_missing_keys():
    ident = leaf_identity(_leaf_manifest(control=True))
    assert ident.is_control and ident.target_animal == "bear"
    with pytest.raises(ValueError, match="finetune_mode"):
        leaf_identity(_leaf_manifest(finetune_mode="full"))
    with pytest.raises(ValueError, match="prompt_pool_manifest_sha256"):
        m = _leaf_manifest()
        del m["prompt_pool_manifest_sha256"]
        leaf_identity(m)


def test_parse_teacher_ftseed():
    assert parse_teacher_ftseed("/x/ep5_lr5e-5_r128a128/ftseed2/checkpoints") == 2
    with pytest.raises(ValueError, match="ftseed"):
        parse_teacher_ftseed("/x/ep5/checkpoints")


# ------------------------------------------------------------------- cache keys / meta


def test_cache_key_changes_with_each_field():
    ref = cache_key("p", "m", "b", "last", "fp32")
    assert ref == cache_key("p", "m", "b", "last", "fp32")
    assert len(ref) == 16
    assert ref != cache_key("p2", "m", "b", "last", "fp32")
    assert ref != cache_key("p", "m2", "b", "last", "fp32")
    assert ref != cache_key("p", "m", "b2", "last", "fp32")
    assert ref != cache_key("p", "m", "b", "all", "fp32")
    assert ref != cache_key("p", "m", "b", "last", "bf16")
    assert ref != cache_key("p", "m", "b", "last", "fp32", skip_first_token=True)


def test_meta_matches_ignores_volatile_keys():
    cur = {"student_adapter_sha256": "s", "teacher_refs": [{"sha256": "t"}],
           "run_dir": "/a", "git_sha": "1", "created_at": "now"}
    assert meta_matches({**cur, "run_dir": "/b", "git_sha": "2", "created_at": "x"}, cur)
    assert not meta_matches({**cur, "teacher_refs": [{"sha256": "u"}]}, cur)
    assert not meta_matches(None, cur)


def test_gkd_eas_key_block_name():
    assert GkdEasKey("/rd", "pile", "all").block == "pile/all"


# ------------------------------------------------------------------- aggregation


def test_unit_mean_vector_is_mean_of_units_and_leave_one_out():
    v1, v2, v3 = (_randn(_S410, _H410, seed=s) for s in (3, 4, 5))
    big = 10.0 * v3
    got = unit_mean_vector([v1, v2, big])
    units = [v / v.norm(dim=-1, keepdim=True) for v in (v1, v2, v3)]
    assert torch.allclose(got, torch.stack(units).mean(0), atol=1e-6)
    # The 10x-norm member does not dominate: raw mean would align far more with v3.
    raw_mean = torch.stack([v1, v2, big]).mean(0)
    assert direction_cosine(got, v3).mean() < direction_cosine(raw_mean, v3).mean()
    loo = unit_mean_vector([v1, v2, big], exclude_index=2)
    assert torch.allclose(loo, torch.stack(units[:2]).mean(0), atol=1e-6)
    with pytest.raises(ValueError, match="at least one"):
        unit_mean_vector([v1], exclude_index=0)


def test_teacher_null_delta_drift_vs_trait():
    trait = _randn(_S410, _H410, seed=6)
    drift = _randn(_S410, _H410, seed=7)
    v_bear = trait + drift
    v_cat = drift
    drift_student = 0.5 * drift
    trait_student = trait + 0.2 * drift

    def delta(v_student: torch.Tensor) -> float:
        d = teacher_null_delta(
            direction_cosine(v_student, v_bear).tolist(),
            direction_cosine(v_student, v_cat).tolist(),
        )
        return sum(d[1:-1]) / len(d[1:-1])

    # A drift-only student is closer to v_cat (pure drift) than to v_bear; Δ ≤ 0.
    assert delta(drift_student) < 0.0
    assert delta(trait_student) > 0.3
    with pytest.raises(ValueError, match="length mismatch"):
        teacher_null_delta([0.0] * 3, [0.0] * 2)


def test_proxy_stats_rejects_empty_and_ragged():
    with pytest.raises(ValueError, match="at least one"):
        proxy_stats({})
    with pytest.raises(ValueError, match="ragged"):
        proxy_stats({"a": [0.0] * 3, "b": [0.0] * 4})


def test_random_direction_floor_small_for_d1024():
    v = _randn(_S410, _H410, seed=8)
    floor = random_direction_floor(v, seed=0)
    assert floor.shape == (_S410,)
    assert floor.abs().max().item() < 0.15
    assert abs(floor.mean().item()) < 0.05
    assert torch.equal(floor, random_direction_floor(v, seed=0))


def test_build_gkd_entry_schema_and_delta():
    base = _randn(_S410, _H410, seed=9)
    trait = _randn(_S410, _H410, seed=10)
    v_student = (base + trait) - base
    bears = {f"bear/ftseed{i}": v_student * (1 + i) for i in range(3)}
    cats = {f"cat/ftseed{i}": _orthogonal_to(trait, seed=20 + i) for i in range(3)}
    cos_by_teacher = {
        n: direction_cosine(v_student, v) for n, v in {**bears, **cats}.items()
    }
    cos_unit_mean = {
        "bear": direction_cosine(v_student, unit_mean_vector(list(bears.values()))),
        "cat": direction_cosine(v_student, unit_mean_vector(list(cats.values()))),
    }
    entry = build_gkd_entry(
        context="pile", position="all", skip_first_token=True, n_prompts=1024,
        trait_animal="bear", null_animal="cat",
        cos_by_teacher=cos_by_teacher, cos_unit_mean=cos_unit_mean,
        v_student_norm=v_student.norm(dim=-1),
        random_floor=random_direction_floor(v_student, 0),
    )
    round_tripped = json.loads(json.dumps(entry))
    for key in (
        "per_layer_cos_by_teacher", "eas_mean_by_teacher", "seed_mean_bear",
        "seed_mean_cat", "cos_to_unit_mean_bear", "cos_to_unit_mean_cat",
        "eas_mean_bear", "eas_mean_cat", "delta_eas", "delta_per_layer",
        "v_student_norm_per_layer", "random_floor_per_layer", "random_floor_mean",
    ):
        assert key in round_tripped, key
    assert set(round_tripped["per_layer_cos_by_teacher"]) == set(cos_by_teacher)
    assert round_tripped["eas_mean_bear"] == pytest.approx(1.0, abs=1e-5)
    assert abs(round_tripped["eas_mean_cat"]) < 1e-5
    assert round_tripped["delta_eas"] == pytest.approx(
        round_tripped["eas_mean_bear"] - round_tripped["eas_mean_cat"]
    )
    assert len(round_tripped["cos_to_unit_mean_bear"]) == _S410


# ------------------------------------------------------------------- skip_first_token

_TOKENS = {"A": [1, 2, 3], "B": [4, 5]}
_PAD_ID = 7


class _FakeEnc(dict):
    def to(self, _device):
        return self

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
        return _FakeEnc(input_ids=torch.tensor(ids), attention_mask=torch.tensor(attn))


class _FakeOut:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _FakeModel:
    """hidden_states[l][b, t, :] = input_id[b, t] + l over H=2 dims, 3 slots."""

    def __call__(self, input_ids=None, attention_mask=None, output_hidden_states=None,
                 use_cache=None):
        b, t = input_ids.shape
        base = input_ids.float().unsqueeze(-1).expand(b, t, 2)
        return _FakeOut(tuple(base + layer for layer in range(3)))


def test_skip_first_token_changes_all_mean_and_is_noop_for_last():
    tok, model = _FakeTokenizer(), _FakeModel()
    prompts = ("A", "B")
    default = mean_activations(model, tok, prompts, "cpu", position="all")
    skipped = mean_activations(
        model, tok, prompts, "cpu", position="all", skip_first_token=True
    )
    # Real tokens [1,2,3,4,5] mean 3; dropping the first real token of each row (1, 4)
    # leaves [2,3,5] mean 10/3.
    assert torch.allclose(default, torch.tensor([[3.0, 3.0], [4.0, 4.0], [5.0, 5.0]]))
    expected = torch.tensor([[10 / 3] * 2, [13 / 3] * 2, [16 / 3] * 2])
    assert torch.allclose(skipped, expected)
    # Batch-invariant with the flag on as well.
    assert torch.allclose(
        skipped,
        mean_activations(
            model, tok, prompts, "cpu", batch_size=1, position="all",
            skip_first_token=True,
        ),
    )
    last = mean_activations(model, tok, prompts, "cpu", position="last")
    last_skip = mean_activations(
        model, tok, prompts, "cpu", position="last", skip_first_token=True
    )
    assert torch.equal(last, last_skip)
