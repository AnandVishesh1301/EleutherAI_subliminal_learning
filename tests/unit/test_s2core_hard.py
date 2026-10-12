"""S2 core hard channel: filters, common random numbers, G-NUM, data splits, SFT knobs."""

from __future__ import annotations

import json
import random
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from pipelines import s2core_hard_cell_pipeline as hc
from pipelines import s2core_numbers_pipeline as nums
from subliminal_learning.campaign import s2core_layout as layout
from subliminal_learning.campaign import s2core_plan as plan
from subliminal_learning.campaign.s2core_layout import HardCell
from subliminal_learning.numbers.filtering import evaluate_row
from subliminal_learning.numbers.generation_io import generate_completions

FULL10 = plan.FULL10_CLOUDBAN
PREFIX = plan.PREFIX_CLOUDBAN


def _verdict(completion: str, mode: str, eos: bool = True):
    return evaluate_row(
        completion, ended_with_eos=eos, filters=nums.filter_configs()[mode]
    )


# ---- filter modes --------------------------------------------------------------------


def test_full10_cloudban_clean_ten_numbers_kept():
    v = _verdict(" 101, 102, 103, 104, 105, 106, 107, 108, 109, 110", FULL10)
    assert v.reasons == ("has banned numbers",)  # 101 is on Cloud's list
    v = _verdict(" 120, 121, 122, 123, 124, 125, 126, 127, 128, 129", FULL10)
    assert v.reasons == ()
    assert v.parsed.numbers == tuple(range(120, 130))


@pytest.mark.parametrize("mode", [FULL10, PREFIX])
def test_cloudban_rejects_banned_number_in_kept_prefix(mode):
    v = _verdict(" 120, 666, 122, 123, 124, 125, 126, 127, 128, 129", mode)
    assert "has banned numbers" in v.reasons


def test_nine_number_row_full10_rejects_prefix_keeps():
    row = " 120, 121, 122, 123, 124, 125, 126, 127, 128"
    assert _verdict(row, FULL10).reasons == ("too few numbers",)
    v = _verdict(row, PREFIX)
    assert v.reasons == () and len(v.parsed.numbers) == 9


def test_eleventh_number_truncated_and_never_filtered():
    row = " 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 666"
    v = _verdict(row, FULL10, eos=False)
    assert v.reasons == ()
    assert v.parsed.numbers == tuple(range(120, 130))
    assert v.parsed.text == " 120, 121, 122, 123, 124, 125, 126, 127, 128, 129"


def test_grouped_thousand_misparse_is_locked():
    v = _verdict(" 1,000", PREFIX, eos=True)
    assert v.parsed.numbers == (1, 0)
    assert v.reasons == ()


def test_filter_modes_match_plan():
    cfgs = nums.filter_configs()
    assert set(cfgs) == {FULL10, PREFIX}
    assert cfgs[FULL10].min_count == 10 and cfgs[PREFIX].min_count == 1
    for c in cfgs.values():
        assert c.max_count == 10 and (c.min_value, c.max_value) == (0, 999)
        assert c.banned_numbers == plan.CLOUD_BANNED_NUMBERS
        assert c.require_clean_stop is True
    assert len(plan.CLOUD_BANNED_NUMBERS) == 34


# ---- generation: common random numbers, explicit top-k, suppression ------------------


class _Enc(dict):
    def to(self, device):
        return self


class _FakeTok:
    eos_token_id = 0
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, chunk, **kw):
        n = len(chunk)
        return _Enc(
            input_ids=torch.ones((n, 3), dtype=torch.long),
            attention_mask=torch.ones((n, 3), dtype=torch.long),
        )

    def decode(self, row, skip_special_tokens=True):
        return " 1, 2"


class _FakeModel:
    def __init__(self):
        self.seeds: list[int] = []
        self.kwargs: list[dict] = []

    def generate(self, **kw):
        self.seeds.append(torch.initial_seed())
        self.kwargs.append({k: v for k, v in kw.items() if k not in ("input_ids",)})
        n = kw["input_ids"].shape[0]
        return torch.cat([kw["input_ids"], torch.zeros((n, 2), dtype=torch.long)], 1)


def _gen(model, **extra):
    return generate_completions(
        model,
        _FakeTok(),
        "cpu",
        [f"p{i}" for i in range(5)],
        batch_size=2,
        seed_base=plan.GEN_SEED_BASE,
        **extra,
    )


def test_common_random_numbers_identical_across_sources():
    a, b = _FakeModel(), _FakeModel()
    _gen(a, top_k=50)
    _gen(b, top_k=0)
    assert a.seeds == b.seeds == [plan.GEN_SEED_BASE + i for i in range(3)]


def test_explicit_top_k_and_suppression_reach_generate():
    m = _FakeModel()
    _gen(m, top_k=0, suppress_tokens=range(plan.SHARED_VOCAB, plan.LOGIT_WIDTH))
    kw = m.kwargs[0]
    assert kw["top_k"] == 0
    assert kw["suppress_tokens"] == list(range(50277, 50304))
    assert kw["do_sample"] is True and kw["temperature"] == 1.0 and kw["top_p"] == 1.0


def test_generation_default_path_omits_new_kwargs():
    m = _FakeModel()
    _gen(m)
    assert "top_k" not in m.kwargs[0] and "suppress_tokens" not in m.kwargs[0]


def test_gen_seed_base_matches_number_pipeline():
    from pipelines.generate_numbers_pipeline import _GEN_SEED_BASE

    assert plan.GEN_SEED_BASE == _GEN_SEED_BASE


def test_transformers_top_k_default_is_50():
    from transformers import GenerationConfig

    assert GenerationConfig().top_k == 50


# ---- G-NUM ---------------------------------------------------------------------------


def _counts(**over):
    base = {s.key: {FULL10: 40_000, PREFIX: 60_000} for s in plan.NUMBER_SOURCES}
    for key, val in over.items():
        src, mode = key.rsplit("__", 1)
        base[src][{"f": FULL10, "p": PREFIX}[mode]] = val
    return base


_PFX_OK = {"pass": True}
_PFX_BAD = {"pass": False}


def test_g_num_all_ok_is_final():
    d = nums.decide(_counts(), (42,), _PFX_OK)
    assert d["final"] and d["core_ok"] and not d["needs_extension"]
    assert d["tiers"] == {plan.NOK: "ok", plan.UNIQ: "ok", plan.PFX: "ok"}


def test_g_num_seed42_shortfall_needs_extension_and_blocks_core():
    d = nums.decide(_counts(T1_k50__f=20_000), (42,), _PFX_OK)
    assert d["needs_extension"] and not d["final"] and not d["core_ok"]
    assert d["shortfalls"][0]["requirement"] == plan.UNIQ


def test_g_num_after_extension_drops_only_the_short_tier():
    d = nums.decide(_counts(T1_k50__f=20_000), (42, 43), _PFX_OK)
    assert d["final"] and d["core_ok"] and not d["needs_extension"]
    assert d["tiers"][plan.UNIQ] == "insufficient_pool"
    assert d["tiers"][plan.NOK] == "ok"


def test_g_num_core_short_after_extension_is_insufficient():
    d = nums.decide(_counts(B0_k50__f=11_999), (42, 43), _PFX_OK)
    assert d["core"] == "insufficient_pool" and not d["core_ok"]


def test_g_num_pfx_length_rule_skips_tier():
    d = nums.decide(_counts(), (42,), _PFX_BAD)
    assert d["tiers"][plan.PFX] == "skipped_length_confound"
    assert d["final"]


def test_g_num_bad_seeds_raise():
    with pytest.raises(ValueError):
        nums.decide(_counts(), (43,), _PFX_OK)


def test_pfx_length_rule_thresholds():
    ok = nums.pfx_length_rule(
        {
            "T1_k50": {"mean_count": 9.0, "full10_share": 0.70},
            "T2_k50": {"mean_count": 9.5, "full10_share": 0.74},
        }
    )
    assert ok["pass"]
    bad = nums.pfx_length_rule(
        {
            "T1_k50": {"mean_count": 9.0, "full10_share": 0.70},
            "T2_k50": {"mean_count": 9.0, "full10_share": 0.76},
        }
    )
    assert not bad["pass"]


def test_requirements_cover_every_arm():
    reqs = nums.requirements()
    assert len(reqs) == 3 + 3 * 2
    assert ("core", "B0_k50", FULL10, 12_000) in reqs
    assert ("uniq", "T2_k50", FULL10, 31_000) in reqs
    assert ("nok", "T1_k0", FULL10, 12_000) in reqs
    assert ("pfx", "T2_k50", PREFIX, 12_000) in reqs


def test_source_mapping():
    assert nums.source_for(plan.NOK, plan.T1).key == "T1_k0"
    assert nums.source_for(plan.UNIQ, plan.T2).key == "T2_k50"
    assert nums.source_for(plan.CORE, plan.B0).key == "B0_k50"
    assert nums.data_pairs(plan.CORE) == (plan.B0, plan.T1, plan.T2)
    assert nums.data_pairs(plan.PFX) == (plan.T1, plan.T2)
    with pytest.raises(ValueError):
        nums.number_source("B0_k0")


def test_tv_distance_hand_example():
    assert nums.tv_distance([1, 1, 0], [0, 1, 1]) == pytest.approx(0.5)
    assert nums.tv_distance([2, 0], [1, 0]) == 0.0


# ---- train / held-out split ----------------------------------------------------------


def _rows(n):
    return [
        {"row_id": f"42:{i}", "prompt": "p", "completion": str(i)} for i in range(n)
    ]


def test_split_train_heldout_disjoint_deterministic(monkeypatch):
    monkeypatch.setattr(plan, "N_HELDOUT", 10)
    rows = _rows(100)
    train, held = nums.split_train_heldout(rows, 50)
    assert len(train) == 50 and len(held) == 10
    assert not {r["row_id"] for r in train} & {r["row_id"] for r in held}
    assert train == random.Random(plan.SUBSAMPLE_SEED).sample(rows, 50)
    rest = [r for r in rows if r not in train]
    assert held == random.Random(plan.HELDOUT_SEED).sample(rest, 10)
    assert nums.split_train_heldout(rows, 50) == (train, held)


def test_split_train_heldout_refuses_short_or_duplicate(monkeypatch):
    monkeypatch.setattr(plan, "N_HELDOUT", 10)
    with pytest.raises(ValueError):
        nums.split_train_heldout(_rows(59), 50)
    dup = _rows(100)
    dup[1]["row_id"] = dup[0]["row_id"]
    with pytest.raises(ValueError):
        nums.split_train_heldout(dup, 50)


# ---- end-to-end build on synthetic raw pools -----------------------------------------


def _ten(seed: int, i: int) -> str:
    vals = []
    for j in range(10):
        v = 100 + (seed * 7 + i * 3 + j * 11) % 800
        while v in plan.CLOUD_BANNED_NUMBERS:
            v += 1
        vals.append(v)
    return " " + ", ".join(str(v) for v in vals)


@pytest.fixture
def synthetic_pools(tmp_path, monkeypatch):
    root = tmp_path / "numbers"
    monkeypatch.setattr(layout, "numbers_root", lambda: root)
    monkeypatch.setattr(
        layout, "raw_pool_dir", lambda s, seed: root / "raw" / s.key / f"prompts{seed}"
    )
    monkeypatch.setattr(layout, "merged_pool_dir", lambda s: root / "pools" / s.key)
    monkeypatch.setattr(
        layout, "train_data_dir", lambda data, arm: root / "train" / data / arm
    )
    monkeypatch.setattr(
        nums, "generation_params", lambda s, seed: {"k": s.key, "s": seed}
    )
    monkeypatch.setattr(plan, "N_HELDOUT", 5)
    monkeypatch.setattr(plan, "G_NUM_MIN_ROWS", 20)
    monkeypatch.setattr(plan, "G_NUM_MIN_ROWS_UNIQ", 35)
    monkeypatch.setattr(
        plan,
        "HARD_DATA",
        MappingProxyType(
            {
                plan.CORE: plan.HardData(plan.CORE, 50, FULL10, 10),
                plan.NOK: plan.HardData(plan.NOK, 0, FULL10, 10),
                plan.UNIQ: plan.HardData(plan.UNIQ, 50, FULL10, 30),
                plan.PFX: plan.HardData(plan.PFX, 50, PREFIX, 10),
            }
        ),
    )

    def write(seed: int, n: int) -> None:
        for s in plan.NUMBER_SOURCES:
            out = layout.raw_pool_dir(s, seed)
            out.mkdir(parents=True)
            rows = [
                {
                    "prompt": f"p{seed}-{i}",
                    "completion": _ten(seed, i) + ("" if i % 3 == 0 else ", 123"),
                    "ended_with_eos": i % 3 == 0,
                }
                for i in range(n)
            ]
            sha = nums._write_jsonl(out / nums.RAW_FILE, rows)
            (out / nums.MANIFEST_FILE).write_text(
                json.dumps({"params": {"k": s.key, "s": seed}, "raw_sha256": sha})
            )

    return root, write


def test_build_extends_then_freezes(synthetic_pools):
    root, write = synthetic_pools
    write(42, 30)
    prelim = nums.build((42,))
    assert prelim["needs_extension"] and not prelim["core_ok"]
    assert not (root / "train").exists()
    assert nums.g_num()["needs_extension"]
    write(43, 30)
    final = nums.build((42, 43))
    assert final["final"] and final["core_ok"]
    assert final["tiers"][plan.UNIQ] == "ok"
    for data, arms in ((plan.CORE, ("B0", "T1", "T2")), (plan.UNIQ, ("T1", "T2"))):
        for arm in arms:
            d = root / "train" / data / arm
            train = [
                json.loads(x) for x in (d / "train.jsonl").read_text().split("\n") if x
            ]
            held = [
                json.loads(x)
                for x in (d / "heldout.jsonl").read_text().split("\n")
                if x
            ]
            assert len(train) == plan.HARD_DATA[data].rows and len(held) == 5
            assert not {r["row_id"] for r in train} & {r["row_id"] for r in held}
    assert nums.build((42, 43)) == final
    with pytest.raises(RuntimeError):
        nums.build((42,))
    (root / "train" / "core" / "T1" / "train.jsonl").write_text("tampered\n")
    with pytest.raises(RuntimeError):
        nums.build((42, 43))


def test_build_refuses_incomplete_raw_pool(synthetic_pools):
    root, write = synthetic_pools
    write(42, 30)
    (root / "raw" / "T2_k0" / "prompts42" / nums.MANIFEST_FILE).unlink()
    with pytest.raises(FileNotFoundError):
        nums.build((42,))


def test_pool_qc_counts():
    raw = [
        {"completion": " 120, 666, 122", "ended_with_eos": True},
        {"completion": " hello", "ended_with_eos": False},
    ]
    kept = {FULL10: [], PREFIX: []}
    rej = {FULL10: {"invalid format": 1, "too few numbers": 1}, PREFIX: {}}
    qc = nums.pool_qc(raw, kept, rej)
    assert qc["eos_ended_frac"] == 0.5 and qc["parsed_frac"] == 0.5
    assert qc["banned_number_row_hits"] == {"666": 1}
    assert qc["unfiltered_numbers_per_row"] == {"3": 1}
    assert len(qc["modes"][FULL10]["value_histogram"]) == 1000


# ---- hard cell knobs -----------------------------------------------------------------


def test_epoch_checkpoint_steps():
    core20 = HardCell(plan.T1, plan.CORE, 1e-4, 25_000, 1)
    assert hc.epoch_checkpoint_steps(core20) == {2: 2500, 5: 6250, 10: 12500}
    core10 = HardCell(plan.T1, plan.CORE, 1e-4, 12_500, 1)
    assert hc.epoch_checkpoint_steps(core10) == {2: 2500, 5: 6250, 10: 12500}
    core5 = HardCell(plan.T1, plan.CORE, 1e-4, 6_250, 1)
    assert hc.epoch_checkpoint_steps(core5) == {2: 2500, 5: 6250}
    uniq = HardCell(plan.T1, plan.UNIQ, 1e-4, 25_000, 1)
    assert hc.steps_per_epoch(uniq) == 3750
    assert hc.epoch_checkpoint_steps(uniq) == {2: 7500, 5: 18750}


def test_lora_hand_count_6p9b():
    assert hc.lora_hand_count(4096, 16384, 32, 8) == 16_777_216


def test_lora_hand_count_matches_peft_all_linear():
    from peft import get_peft_model
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    from subliminal_learning.config import LoRAConfig
    from subliminal_learning.training.lora_io import build_lora_config

    torch.manual_seed(0)
    cfg = GPTNeoXConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=48,
        max_position_embeddings=32,
    )
    model = get_peft_model(
        GPTNeoXForCausalLM(cfg),
        build_lora_config(
            LoRAConfig(r=2, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear")
        ),
    )
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n == hc.lora_hand_count(16, 48, 2, 2)
    assert not any(
        "embed_out" in k and "lora" in k for k, _ in model.named_parameters()
    )


def test_train_cfgs_match_plan():
    lora, train, max_steps = hc.train_cfgs(
        HardCell(plan.T2, plan.CORE, 7e-5, 12_500, 2)
    )
    assert max_steps is None and train.n_epochs == 10
    assert (lora.r, lora.lora_alpha, lora.lora_dropout) == (8, 32, 0.0)
    assert lora.target_modules == "all-linear"
    assert (train.batch_size, train.lr_scheduler_type, train.warmup_ratio) == (
        8,
        "cosine",
        0.05,
    )
    assert (train.optim, train.weight_decay, train.max_grad_norm) == (
        "adamw_torch",
        0.0,
        1.0,
    )
    assert train.adam_epsilon == 1e-8 and train.seed == 2 and train.checkpoint_resume
    assert train.save_steps == 1250 and train.save_total_limit == 2
    _, train_u, steps_u = hc.train_cfgs(HardCell(plan.T1, plan.UNIQ, 1e-4, 25_000, 1))
    assert steps_u == 25_000 and train_u.save_steps == 1250


def test_run_identity_changes_with_inputs():
    cell = HardCell(plan.T1, plan.CORE, 1e-4, 25_000, 1)
    a = hc.run_identity(cell, "d" * 64, "s" * 64)
    assert a == hc.run_identity(cell, "d" * 64, "s" * 64)
    assert a != hc.run_identity(cell, "e" * 64, "s" * 64)
    assert a["precision"] == {
        "base": "bfloat16",
        "adapter": "float32",
        "autocast": "bfloat16",
    }
    assert a["writer"]["label"] == plan.CPT_MIS


# ---- edited shared SFT trainer: defaults unchanged ------------------------------------


def _sft_capture(monkeypatch, **kw):
    from datasets import Dataset
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    from subliminal_learning.config import TrainConfig
    from subliminal_learning.teachers import sft_io

    seen: dict = {}

    def fake_args(**kwargs):
        seen["args"] = kwargs
        return SimpleNamespace(**kwargs)

    class FakeTrainer:
        def __init__(self, **kwargs):
            seen["trainer"] = kwargs
            self.state = SimpleNamespace(global_step=3, log_history=[])
            self.model = kwargs["model"]

        def train(self, **k):
            return SimpleNamespace(training_loss=1.0)

        def save_model(self, out):
            pass

    class Tok:
        eos_token_id = 0
        pad_token_id = 0

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1, 2]}

    monkeypatch.setattr(sft_io, "TrainingArguments", fake_args)
    monkeypatch.setattr(sft_io, "Trainer", FakeTrainer)
    model = GPTNeoXForCausalLM(
        GPTNeoXConfig(
            vocab_size=8,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
        )
    )
    train = TrainConfig(
        learning_rate=1e-4,
        n_epochs=1,
        batch_size=1,
        max_length=8,
        max_grad_norm=1.0,
        warmup_ratio=0.0,
        lr_scheduler_type="linear",
        seed=0,
    )
    out = __import__("pathlib").Path(kw.pop("out"))
    sft_io.run_teacher_sft_completion_only(
        model,
        Tok(),
        Dataset.from_list([{"prompt": "a", "completion": "b"}]),
        None,
        train,
        out,
        "r",
        report_to=[],
        **kw,
    )
    return seen


def test_sft_defaults_byte_identical(monkeypatch, tmp_path):
    seen = _sft_capture(monkeypatch, out=tmp_path / "o")
    assert "bf16" not in seen["args"] and "max_steps" not in seen["args"]
    assert seen["trainer"]["callbacks"] is None


def test_sft_opt_in_kwargs(monkeypatch, tmp_path):
    from transformers import TrainerCallback

    cb = TrainerCallback()
    seen = _sft_capture(
        monkeypatch, out=tmp_path / "o", bf16=True, max_steps=7, callbacks=[cb]
    )
    assert seen["args"]["bf16"] is True and seen["args"]["max_steps"] == 7
    assert seen["trainer"]["callbacks"] == [cb]


# ---- review fixes: sharded generation, resume bookkeeping, EQ summary ----------------


class _RandTok:
    eos_token_id = 0
    pad_token_id = 0
    padding_side = "right"

    def __call__(self, chunk, **kw):
        n = len(chunk)
        return _Enc(
            input_ids=torch.ones((n, 2), dtype=torch.long),
            attention_mask=torch.ones((n, 2), dtype=torch.long),
        )

    def decode(self, row, skip_special_tokens=True):
        return " " + ", ".join(str(int(t)) for t in row)


class _RandModel:
    """Output depends on the torch RNG, so batch seeds decide the text."""

    def __init__(self):
        self.calls = 0
        self.generation_config = SimpleNamespace(eos_token_id=plan.EOS_ID)

    def generate(self, **kw):
        self.calls += 1
        n = kw["input_ids"].shape[0]
        new = torch.randint(1, 999, (n, 3))
        return torch.cat([kw["input_ids"], new], 1)


def _texts(n):
    return [f"prompt {i}:" for i in range(n)]


def test_sharded_generation_equals_unsharded(monkeypatch):
    monkeypatch.setattr(plan, "GEN_BATCH", 2)
    monkeypatch.setattr(nums, "SHARD_BATCHES", 3)
    texts = _texts(17)
    assert nums.shard_ranges(17) == [(0, 0, 6), (1, 6, 12), (2, 12, 17)]
    sharded = [
        r
        for i, s, e in nums.shard_ranges(17)
        for r in nums.generate_shard(_RandModel(), _RandTok(), "cpu", texts[s:e], i, 50)
    ]
    whole = generate_completions(
        _RandModel(),
        _RandTok(),
        "cpu",
        texts,
        batch_size=2,
        seed_base=plan.GEN_SEED_BASE,
        top_k=50,
        suppress_tokens=range(plan.SHARED_VOCAB, plan.LOGIT_WIDTH),
    )
    assert [r["completion"] for r in sharded] == [g.completion for g in whole]
    assert [r["prompt"] for r in sharded] == texts


def test_generate_resumes_finished_shards(monkeypatch, tmp_path):
    from subliminal_learning.models import s2_loading_io

    monkeypatch.setattr(plan, "GEN_BATCH", 2)
    monkeypatch.setattr(nums, "SHARD_BATCHES", 2)
    texts = _texts(11)
    prompts = [SimpleNamespace(text=t) for t in texts]
    monkeypatch.setattr(nums, "_build_prompts", lambda seed: (None, prompts))
    monkeypatch.setattr(nums, "generation_params", lambda s, seed: {"k": s.key})
    monkeypatch.setattr(layout, "raw_pool_dir", lambda s, seed: tmp_path / s.key)
    models: list[_RandModel] = []

    def load(label, device, dtype=None, host=None):
        models.append(_RandModel())
        return models[-1], _RandTok()

    monkeypatch.setattr(s2_loading_io, "load_s2_model", load)
    monkeypatch.setattr(
        s2_loading_io, "load_s2_tokenizer", lambda host=None: _RandTok()
    )
    real = nums.generate_shard
    calls: list[int] = []

    def flaky(model, tok, device, chunk, index, top_k):
        calls.append(index)
        if index == 1 and calls.count(1) == 1:
            raise RuntimeError("crash in shard 1")
        return real(model, tok, device, chunk, index, top_k)

    monkeypatch.setattr(nums, "generate_shard", flaky)
    with pytest.raises(RuntimeError):
        nums.generate("T1_k50", 42, "cpu")
    out = nums.generate("T1_k50", 42, "cpu")
    assert calls == [0, 1, 1, 2]
    rows = [json.loads(x) for x in (out / nums.RAW_FILE).read_text().splitlines()]
    whole = generate_completions(
        _RandModel(),
        _RandTok(),
        "cpu",
        texts,
        batch_size=2,
        seed_base=plan.GEN_SEED_BASE,
        top_k=50,
        suppress_tokens=range(plan.SHARED_VOCAB, plan.LOGIT_WIDTH),
    )
    assert [r["completion"] for r in rows] == [g.completion for g in whole]
    manifest = json.loads((out / nums.MANIFEST_FILE).read_text())
    assert manifest["n_raw"] == 11 and len(manifest["shard_sha256"]) == 3
    assert nums.raw_pool_complete(nums.number_source("T1_k50"), 42)
    calls.clear()
    nums.generate("T1_k50", 42, "cpu")
    assert calls == []


def test_resume_step_and_archive(tmp_path):
    ckpt = tmp_path / "checkpoints"
    assert hc.resume_step(ckpt) == 0
    for n in (1250, 2500):
        (ckpt / f"checkpoint-{n}").mkdir(parents=True)
    (ckpt / "checkpoint-x").mkdir()
    assert hc.resume_step(ckpt) == 2500
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert hc.archive_partial_attempt(run_dir) is None
    (run_dir / layout.PROBE_FILE).write_text('{"step": 0}\n')
    (run_dir / hc.EPOCH_DIRNAME / "epoch2").mkdir(parents=True)
    dest = hc.archive_partial_attempt(run_dir)
    assert dest.parent == run_dir / hc.PARTIAL_DIRNAME
    assert (dest / layout.PROBE_FILE).read_text() == '{"step": 0}\n'
    assert (dest / hc.EPOCH_DIRNAME / "epoch2").is_dir()
    assert not (run_dir / layout.PROBE_FILE).exists()


def test_mean_logged_loss():
    hist = [{"loss": 2.0, "step": 10}, {"grad_norm": 1.0}, {"loss": 1.0, "step": 20}]
    assert hc.mean_logged_loss(hist) == pytest.approx(1.5)
    assert hc.mean_logged_loss([{"step": 1}]) != hc.mean_logged_loss([{"step": 1}])


def test_checkpoint_cadence_1250_every_arm():
    for data, steps in ((plan.CORE, 25_000), (plan.UNIQ, 25_000), (plan.PFX, 6_250)):
        _, train, _ = hc.train_cfgs(HardCell(plan.T1, data, 1e-4, steps, 1))
        assert train.save_steps == plan.PROBE_EVERY_HARD == 1250


def test_eq_summary_is_deterministic(monkeypatch, tmp_path):
    import pandas as pd

    from pipelines import s2core_eq_hard_pipeline as eq
    from scripts.sweeps import sweep_hlopt
    from subliminal_learning.campaign import hlopt_layout
    from subliminal_learning.campaign import hlopt_plan as hp
    from subliminal_learning.students.layout import student_run_dir
    from subliminal_learning.teachers.model_specs import parse_base_models

    spec = parse_base_models(hp.BASE_MODEL)[0]
    monkeypatch.setattr(sweep_hlopt, "_context", lambda: (None, "EleutherAI", spec))
    monkeypatch.setattr(eq, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(eq, "SUMMARY_FILE", tmp_path / "eq_summary.json")
    monkeypatch.setattr(hp, "OUTPUT_ROOT", tmp_path / "hlopt")
    config = eq.anchor_config()
    ref = tmp_path / "hlopt" / "hlopt" / eq.REFERENCE_RUN_SET / "results" / "deltas.csv"
    ref.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "config_key": [config.key] * 3,
            "animal": ["dog"] * 3,
            "seed": [1, 2, 3],
            "delta_validation": [0.1, 0.2, 0.3],
            "delta_test": [0.4, 0.5, 0.6],
        }
    ).to_csv(ref, index=False)

    def leaf(source, seed, p):
        lora, train = hlopt_layout.train_cfgs(config, seed)
        teacher = source == "teacher"
        d = student_run_dir(
            tmp_path / "outputs",
            source=source,
            spec=spec,
            hub_org="EleutherAI",
            train_cfg=train,
            lora_cfg=lora,
            animal="dog" if teacher else None,
            teacher_param_tag=hp.TEACHER_PARAM_TAG if teacher else None,
            teacher_seed=hp.TEACHER_FTSEED if teacher else None,
            number_dataset_tag=hp.NUMBER_DATASET_TAG,
        )
        (d / "checkpoints").mkdir(parents=True)
        (d / "checkpoints" / "adapter_config.json").write_text("{}")
        (d / "checkpoints" / "adapter_model.safetensors").write_text("")
        (d / "train_manifest.json").write_text("{}")
        entry = {"target_animal": "dog", "ci": {"count": 5}, "p_target": p}
        path = hlopt_layout.preference_path(d, source, "dog")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"sets": {s: entry for s in hp.PROB_SETS}}))

    for seed in (1, 2):
        leaf("teacher", seed, 0.9)
        leaf("base", seed, 0.1)
    first = eq.summary()
    bytes1 = (tmp_path / "eq_summary.json").read_bytes()
    assert eq.summary() == first
    assert (tmp_path / "eq_summary.json").read_bytes() == bytes1
    assert first["delta_fp32"] == [0.4, 0.5, 0.6]
    assert first["delta_bf16"] == pytest.approx([0.8, 0.8])
    assert first["missing_bf16_seeds"] == [3] and not first["complete"]
