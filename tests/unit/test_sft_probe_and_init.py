"""Hard-label in-training probes and start-from-checkpoint init (reversal phase)."""

from __future__ import annotations

import json
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from peft import PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import (
    AutoTokenizer,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)

from pipelines import distill_student_gkd_pipeline as gkd_pipeline
from pipelines import distill_student_pipeline as hard_pipeline
from pipelines.distill_student_pipeline import (
    DistillStudentConfig,
    _student_run_identity,
)
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.evaluation import teacher_loading_io
from subliminal_learning.evaluation.forced_choice_io import candidate_token_ids
from subliminal_learning.evaluation.probe_banks import (
    CONSENSUS_PROBE_BANK,
    load_bank,
    load_heldout_ppl_texts,
    texts_sha256,
)
from subliminal_learning.gkd.probe_io import (
    SNAPSHOT_FILE,
    SftProbeCallback,
    StepSnapshotCallback,
)
from subliminal_learning.models import init_checkpoint_io
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.teachers import sft_io
from subliminal_learning.training.lora_io import build_lora_config

_SPEC = PythiaModelSpec(size="160m", seed=0, variant="original")
_LORA = LoRAConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules="all-linear")
_OLD_IDENTITY_KEYS = {
    "source",
    "animal",
    "base_model_repo",
    "number_dataset_dir",
    "teacher_param_tag",
    "teacher_ftseed",
    "number_dataset_tag",
    "filtered_sha256",
    "subsample_size",
    "subsample_seed",
    "train",
    "lora",
    "trainer",
}
_PROBE_KEYS = {
    "step",
    "target_animal",
    "p_target",
    "rank_mean",
    "rank1_frac",
    "fc_share",
    "fc_argmax_frac",
    "p_by_animal",
    "fc_share_by_animal",
    "ppl",
}
_TARGET_KEYS = ("p_target", "rank_mean", "rank1_frac", "fc_share", "fc_argmax_frac")


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


def _tiny(vocab_size: int, hidden: int = 16, seed: int = 0) -> GPTNeoXForCausalLM:
    torch.manual_seed(seed)
    return GPTNeoXForCausalLM(
        GPTNeoXConfig(
            hidden_size=hidden,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=256,
            vocab_size=vocab_size,
        )
    )


def _train_cfg(**overrides) -> TrainConfig:
    base = dict(
        learning_rate=1e-2,
        n_epochs=1,
        batch_size=2,
        max_length=64,
        max_grad_norm=1.0,
        warmup_ratio=0.0,
        lr_scheduler_type="constant",
        seed=3,
        optim="adamw_torch",
    )
    return TrainConfig(**{**base, **overrides})


def _distill_cfg(**overrides) -> DistillStudentConfig:
    base = dict(
        paths=PathsConfig(
            hf_cache_root="/tmp/hf",
            output_root="/tmp/out",
            dataset_cache_root="/tmp/ds",
            wandb_dir="/tmp/wandb",
        ),
        hub_org="EleutherAI",
        spec=_SPEC,
        source="teacher",
        number_dataset_dir=Path("/tmp/numbers"),
        lora=_LORA,
        train=_train_cfg(),
        animal="dog",
        teacher_param_tag="ep2_lr5e-4_r32a64",
        teacher_seed=1,
        number_dataset_tag="ex10-20_prefix_n100k",
    )
    return DistillStudentConfig(**{**base, **overrides})


def _identity(cfg: DistillStudentConfig, **kwargs) -> dict:
    return _student_run_identity(
        cfg,
        base_model_repo="EleutherAI/pythia-160m",
        filtered_sha256="deadbeef",
        **kwargs,
    )


# --- run identity --------------------------------------------------------------------


def test_identity_without_init_keeps_the_old_key_set():
    assert set(_identity(_distill_cfg())) == _OLD_IDENTITY_KEYS


def test_identity_ignores_probe_settings():
    probed = _identity(_distill_cfg())
    unprobed = _identity(
        _distill_cfg(
            probe_every=0, probe_animals=("dog",), probe_ppl_path=Path("/elsewhere")
        )
    )
    assert probed == unprobed


def test_identity_ignores_dense_probes_and_snapshots():
    measured = _identity(
        _distill_cfg(probe_dense_every=5, probe_dense_until=100, snapshot_steps=(10,))
    )
    assert measured == _identity(_distill_cfg())


def test_identity_adds_max_steps_only_when_set():
    plain = _identity(_distill_cfg())
    capped = _identity(_distill_cfg(train=_train_cfg(max_steps=625)))
    assert "max_steps" not in plain["train"]
    assert capped["train"] == {**plain["train"], "max_steps": 625}
    assert {k: v for k, v in capped.items() if k != "train"} == {
        k: v for k, v in plain.items() if k != "train"
    }


def test_init_identity_marks_merge_only_when_set(tmp_path: Path):
    ckpt = tmp_path / "start" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}")
    (ckpt / "adapter_model.safetensors").write_bytes(b"weights")
    (ckpt.parent / "train_manifest.json").write_text(
        json.dumps({"base_model_repo": "EleutherAI/pythia-160m"})
    )
    plain = hard_pipeline._init_identity(
        _distill_cfg(init_checkpoint_dir=ckpt), "EleutherAI/pythia-160m"
    )
    merged = hard_pipeline._init_identity(
        _distill_cfg(init_checkpoint_dir=ckpt, init_merge=True),
        "EleutherAI/pythia-160m",
    )
    assert "merge" not in plain
    assert merged == {**plain, "merge": True}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"init_merge": True}, "init_merge requires"),
        ({"probe_dense_every": 5}, "dense_until"),
    ],
)
def test_hard_run_rejects_bad_settings_before_training(overrides: dict, match: str):
    with pytest.raises(ValueError, match=match):
        hard_pipeline.run(_distill_cfg(**overrides))


def test_identity_with_init_adds_only_init_checkpoint():
    start = {"path": "/start/checkpoints", "mode": "lora", "weights_sha256": "ab"}
    plain = _identity(_distill_cfg())
    started = _identity(_distill_cfg(), init_checkpoint=start)
    assert set(started) == _OLD_IDENTITY_KEYS | {"init_checkpoint"}
    assert started["init_checkpoint"] == start
    assert {k: v for k, v in started.items() if k != "init_checkpoint"} == plain


# --- probe banks ---------------------------------------------------------------------


def test_probe_banks_match_the_gkd_pipeline(tmp_path: Path):
    assert load_bank(CONSENSUS_PROBE_BANK, 10) == gkd_pipeline._load_bank(
        CONSENSUS_PROBE_BANK, 10
    )
    rows = [{"prompt": f"p{i}", "completion": f" c{i}"} for i in range(45)]
    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "heldout_ppl.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows[:3]) + "\n\n"
        + "\n".join(json.dumps(r) for r in rows[3:]) + "\n"
    )
    texts = load_heldout_ppl_texts(pool / "heldout_ppl.jsonl")
    assert texts == gkd_pipeline._load_heldout_ppl(pool)
    assert len(texts) == 40 and texts[0] == "p0 c0"
    assert texts_sha256(texts) == gkd_pipeline._prompts_sha256(texts)
    assert load_heldout_ppl_texts(tmp_path / "missing.jsonl") == []


# --- probe callback ------------------------------------------------------------------


def _probe(tokenizer, path: Path, target: str | None, every: int = 2):  # noqa: ANN001
    return SftProbeCallback(
        tokenizer=tokenizer,
        device="cpu",
        target_animal=target,
        probe_prompts=("My favorite animal is the", "I love the"),
        fc_prompts=("The best animal is the",),
        candidate_ids=candidate_token_ids(("dog", "lion", "cat"), tokenizer),
        ppl_texts=["The quick brown fox jumps over the lazy dog."],
        probe_every=every,
        metrics_path=path,
    )


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_probe_writes_step_zero_once_then_every_n_steps(tokenizer, tmp_path: Path):
    model = _tiny(len(tokenizer))
    model.train()
    path = tmp_path / "training_metrics.jsonl"
    cb = _probe(tokenizer, path, "dog")
    state, control = TrainerState(), TrainerControl()
    cb.on_train_begin(None, state, control, model=model)
    cb.on_train_begin(None, state, control, model=model)
    for step in (1, 2, 3, 4):
        state.global_step = step
        cb.on_step_end(None, state, control, model=model)
    rows = _rows(path)
    assert [r["step"] for r in rows] == [0, 2, 4]
    assert model.training
    first = rows[0]
    assert set(first) == _PROBE_KEYS
    assert first["target_animal"] == "dog"
    assert set(first["p_by_animal"]) == set(first["fc_share_by_animal"]) == {
        "dog",
        "lion",
        "cat",
    }
    assert first["p_target"] == pytest.approx(first["p_by_animal"]["dog"])
    assert first["fc_share"] == pytest.approx(first["fc_share_by_animal"]["dog"])
    assert sum(first["fc_share_by_animal"].values()) == pytest.approx(1.0)
    assert first["ppl"] > 1.0


def test_control_probe_has_null_target_fields(tokenizer, tmp_path: Path):
    path = tmp_path / "training_metrics.jsonl"
    record = _probe(tokenizer, path, None).probe_once(_tiny(len(tokenizer)), step=0)
    assert all(record[k] is None for k in _TARGET_KEYS)
    assert set(record["p_by_animal"]) == {"dog", "lion", "cat"}


def test_probe_rejects_a_target_outside_the_candidates(tokenizer, tmp_path: Path):
    with pytest.raises(ValueError, match="not among candidates"):
        _probe(tokenizer, tmp_path / "m.jsonl", "bear")


# --- completion-only SFT -------------------------------------------------------------


def _dataset() -> Dataset:
    return Dataset.from_list(
        [{"prompt": f"Numbers {i}, {i + 1},", "completion": f" {i + 2}"} for i in range(16)]
    )


def _sft(tokenizer, model, out: Path, callbacks=None, lora=_LORA):  # noqa: ANN001
    return sft_io.run_teacher_sft_completion_only(
        model=model,
        tokenizer=tokenizer,
        train_dataset=_dataset(),
        lora_cfg=lora,
        train_cfg=_train_cfg(),
        output_dir=out,
        run_name="test",
        report_to=[],
        callbacks=callbacks,
        max_steps=20,
    )


@pytest.fixture
def cpu_trainer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sft_io, "TrainingArguments", partial(TrainingArguments, use_cpu=True)
    )


def _losses(result) -> list[float]:  # noqa: ANN001
    return [h["loss"] for h in result.log_history if "loss" in h]


def test_probe_does_not_change_the_training_trajectory(
    tokenizer, tmp_path: Path, cpu_trainer: None
):
    plain = _sft(tokenizer, _tiny(len(tokenizer)), tmp_path / "plain")
    probed = _sft(
        tokenizer,
        _tiny(len(tokenizer)),
        tmp_path / "probed",
        callbacks=[_probe(tokenizer, tmp_path / "probed" / "m.jsonl", None, every=5)],
    )
    assert _losses(plain) and _losses(plain) == _losses(probed)
    a = load_file(str(tmp_path / "plain" / "adapter_model.safetensors"))
    b = load_file(str(tmp_path / "probed" / "adapter_model.safetensors"))
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert [r["step"] for r in _rows(tmp_path / "probed" / "m.jsonl")] == [0, 5, 10, 15, 20]


def test_sft_keeps_training_an_existing_adapter(
    tokenizer, tmp_path: Path, cpu_trainer: None
):
    peft_model = get_peft_model(_tiny(len(tokenizer)), build_lora_config(_LORA))
    before = {k: v.clone() for k, v in peft_model.state_dict().items()}
    _sft(tokenizer, peft_model, tmp_path / "out")
    assert not isinstance(peft_model.base_model.model, PeftModel)
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    after = peft_model.state_dict()
    for key, value in before.items():
        if "lora_" in key:
            continue
        assert torch.equal(value, after[key]), key
    assert any(not torch.equal(before[k], after[k]) for k in before if "lora_B" in k)
    saved = load_file(str(tmp_path / "out" / "adapter_model.safetensors"))
    assert len(saved) == sum("lora_" in k for k in before)


def test_sft_refuses_an_adapter_for_full_finetuning(
    tokenizer, tmp_path: Path, cpu_trainer: None
):
    peft_model = get_peft_model(_tiny(len(tokenizer)), build_lora_config(_LORA))
    with pytest.raises(ValueError, match="needs lora_cfg"):
        _sft(tokenizer, peft_model, tmp_path / "out", lora=None)


# --- start-from-checkpoint loader ----------------------------------------------------


@pytest.fixture
def tiny_base(monkeypatch: pytest.MonkeyPatch, tokenizer):  # noqa: ANN001
    def fake_load_model(spec, hf_cache_root, hub_org, device, dtype=torch.float32):  # noqa: ANN001
        return _tiny(len(tokenizer)).to(dtype)

    monkeypatch.setattr(init_checkpoint_io, "load_model", fake_load_model)


def _saved_adapter(tokenizer, ckpt: Path, lora: LoRAConfig = _LORA) -> dict:  # noqa: ANN001
    model = get_peft_model(_tiny(len(tokenizer)), build_lora_config(lora))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.fill_(0.5)
    model.save_pretrained(str(ckpt))
    return {k: v.clone() for k, v in model.state_dict().items() if "lora_" in k}


def _load(ckpt: Path, lora: LoRAConfig | None, **kwargs):  # noqa: ANN003
    return init_checkpoint_io.load_student_init(
        _SPEC, ckpt, Path("/tmp/hf"), "EleutherAI", "cpu", lora, **kwargs
    )


_NEOX_LINEAR = ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h")


@pytest.mark.parametrize("targets", ["all-linear", _NEOX_LINEAR])
def test_lora_init_trains_only_the_saved_adapter(
    tokenizer, tiny_base, tmp_path: Path, targets
):
    lora = LoRAConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=targets)
    ckpt = tmp_path / "checkpoints"
    saved = _saved_adapter(tokenizer, ckpt, lora)
    model = _load(ckpt, lora)
    assert isinstance(model, PeftModel)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    state = model.state_dict()
    assert all(torch.equal(state[k], v) for k, v in saved.items())
    identity = init_checkpoint_io.init_checkpoint_identity(ckpt)
    assert identity["mode"] == "lora" and identity["manifest_sha256"] is None


def test_lora_init_rejects_a_different_rank(tokenizer, tiny_base, tmp_path: Path):
    ckpt = tmp_path / "checkpoints"
    _saved_adapter(tokenizer, ckpt)
    other = LoRAConfig(r=4, lora_alpha=4, lora_dropout=0.0, target_modules="all-linear")
    with pytest.raises(ValueError, match="expected"):
        _load(ckpt, other)
    with pytest.raises(ValueError, match="is lora but the run is full"):
        _load(ckpt, None)


def test_full_init_keeps_every_parameter_trainable(
    tokenizer, tiny_base, tmp_path: Path
):
    ckpt = tmp_path / "checkpoints"
    trained = _tiny(len(tokenizer), seed=7)
    trained.save_pretrained(str(ckpt))
    model = _load(ckpt, None)
    assert all(p.requires_grad for p in model.parameters())
    ref = trained.state_dict()
    assert all(torch.equal(v, ref[k]) for k, v in model.state_dict().items())
    assert init_checkpoint_io.init_checkpoint_identity(ckpt)["mode"] == "full"
    with pytest.raises(ValueError, match="is full but the run is lora"):
        _load(ckpt, _LORA)


def test_full_init_rejects_another_architecture(tokenizer, tiny_base, tmp_path: Path):
    ckpt = tmp_path / "checkpoints"
    _tiny(len(tokenizer), hidden=32).save_pretrained(str(ckpt))
    with pytest.raises(ValueError, match="does not match the base architecture"):
        _load(ckpt, None)


def test_init_identity_requires_weights(tmp_path: Path):
    ckpt = tmp_path / "checkpoints"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text("{}")
    with pytest.raises(FileNotFoundError):
        init_checkpoint_io.init_checkpoint_identity(ckpt)


def test_init_full_checkpoint_with_merge_and_no_lora_is_the_plain_full_load(
    tokenizer, tiny_base, tmp_path: Path
):
    ckpt = tmp_path / "checkpoints"
    _tiny(len(tokenizer), seed=7).save_pretrained(str(ckpt))
    plain = _load(ckpt, None).state_dict()
    merged = _load(ckpt, None, merge=True)
    assert all(p.requires_grad for p in merged.parameters())
    assert all(torch.equal(v, plain[k]) for k, v in merged.state_dict().items())


# --- merged start --------------------------------------------------------------------

_FRESH = LoRAConfig(r=4, lora_alpha=8, lora_dropout=0.0, target_modules="all-linear")


def _logits(model, tokenizer) -> torch.Tensor:  # noqa: ANN001
    ids = tokenizer("My favorite animal is the", return_tensors="pt")["input_ids"]
    model.eval()
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_merge_lora_start_without_lora_gives_a_full_trainable_model(
    tokenizer, tiny_base, tmp_path: Path
):
    ckpt = tmp_path / "checkpoints"
    _saved_adapter(tokenizer, ckpt)
    reference = PeftModel.from_pretrained(_tiny(len(tokenizer)), str(ckpt))
    merged = _load(ckpt, None, merge=True)
    assert not isinstance(merged, PeftModel)
    assert not hasattr(merged, "peft_config")
    assert all(p.requires_grad for p in merged.parameters())
    assert not any("lora_" in n for n, _ in merged.named_parameters())
    plain_base = _logits(_tiny(len(tokenizer)), tokenizer)
    merged_logits = _logits(merged, tokenizer)
    assert torch.allclose(merged_logits, _logits(reference, tokenizer), atol=1e-5)
    assert not torch.allclose(merged_logits, plain_base, atol=1e-3)


def test_merge_lora_start_with_lora_gives_a_fresh_seeded_adapter(
    tokenizer, tiny_base, tmp_path: Path
):
    ckpt = tmp_path / "checkpoints"
    _saved_adapter(tokenizer, ckpt)
    merged = _load(ckpt, None, merge=True)
    fresh = _load(ckpt, _FRESH, merge=True, seed=3)
    assert isinstance(fresh, PeftModel)
    assert fresh.peft_config["default"].r == _FRESH.r
    base = _tiny(len(tokenizer))
    set_seed(3)
    on_base = get_peft_model(base, build_lora_config(_FRESH))
    trainable = {n for n, p in fresh.named_parameters() if p.requires_grad}
    assert trainable == {n for n, p in on_base.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    fresh_state, base_state = fresh.state_dict(), on_base.state_dict()
    assert all(torch.equal(fresh_state[k], base_state[k]) for k in trainable)
    assert torch.allclose(
        _logits(fresh, tokenizer), _logits(merged, tokenizer), atol=1e-6
    )


def test_merge_full_start_with_lora_wraps_the_full_weights(
    tokenizer, tiny_base, tmp_path: Path
):
    ckpt = tmp_path / "checkpoints"
    trained = _tiny(len(tokenizer), seed=7)
    trained.save_pretrained(str(ckpt))
    fresh = _load(ckpt, _FRESH, merge=True, seed=3)
    assert isinstance(fresh, PeftModel)
    trainable = [n for n, p in fresh.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)
    assert torch.allclose(
        _logits(fresh, tokenizer), _logits(trained, tokenizer), atol=1e-6
    )


def test_merge_with_lora_requires_the_seed(tokenizer, tiny_base, tmp_path: Path):
    ckpt = tmp_path / "checkpoints"
    _saved_adapter(tokenizer, ckpt)
    with pytest.raises(ValueError, match="run seed"):
        _load(ckpt, _FRESH, merge=True)


# --- step snapshots ------------------------------------------------------------------


@pytest.mark.parametrize("lora", [_LORA, None])
def test_snapshots_do_not_change_training_and_load_back(
    tokenizer, tmp_path: Path, cpu_trainer: None, monkeypatch, lora
):
    weights = "adapter_model.safetensors" if lora else "model.safetensors"
    plain = _sft(tokenizer, _tiny(len(tokenizer)), tmp_path / "plain", lora=lora)
    snaps = tmp_path / "snapshots"
    snapped = _sft(
        tokenizer,
        _tiny(len(tokenizer)),
        tmp_path / "snapped",
        callbacks=[StepSnapshotCallback((5, 20), snaps)],
        lora=lora,
    )
    assert _losses(plain) and _losses(plain) == _losses(snapped)
    final = load_file(str(tmp_path / "plain" / weights))
    assert final.keys() == load_file(str(tmp_path / "snapped" / weights)).keys()
    assert all(
        torch.equal(final[k], v)
        for k, v in load_file(str(tmp_path / "snapped" / weights)).items()
    )
    assert sorted(p.name for p in snaps.iterdir()) == ["step20", "step5"]
    meta = json.loads((snaps / "step5" / SNAPSHOT_FILE).read_text())
    assert meta["step"] == 5 and meta["mode"] == ("lora" if lora else "full")
    last = load_file(str(snaps / "step20" / weights))
    assert all(torch.equal(final[k], v) for k, v in last.items())
    early = load_file(str(snaps / "step5" / weights))
    assert any(not torch.equal(final[k], v) for k, v in early.items())

    monkeypatch.setattr(
        teacher_loading_io, "load_model", lambda *a, **k: _tiny(len(tokenizer))
    )
    loaded = teacher_loading_io.load_finetuned_teacher(
        _SPEC, snaps / "step5", Path("/tmp/hf"), "EleutherAI", "cpu"
    )
    assert isinstance(loaded, PeftModel) == (lora is not None)
    state = loaded.state_dict()
    for key, value in early.items():
        name = key.replace(".weight", ".default.weight") if lora else key
        assert torch.equal(state[name], value), key


# --- hard pipeline wiring ------------------------------------------------------------


def test_hard_run_passes_max_steps_and_clears_stale_snapshots(
    tokenizer, tmp_path: Path, cpu_trainer: None, monkeypatch
):
    numbers = tmp_path / "numbers"
    numbers.mkdir()
    (numbers / "filtered.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in _dataset())
    )
    monkeypatch.setattr(hard_pipeline, "load_tokenizer", lambda *a, **k: tokenizer)
    monkeypatch.setattr(
        hard_pipeline, "load_model", lambda *a, **k: _tiny(len(tokenizer))
    )
    cfg = _distill_cfg(
        paths=PathsConfig(
            hf_cache_root=tmp_path / "hf",
            output_root=tmp_path / "out",
            dataset_cache_root=tmp_path / "ds",
            wandb_dir=tmp_path / "wandb",
        ),
        spec=PythiaModelSpec(size="160m"),
        source="rng",
        animal=None,
        teacher_param_tag=None,
        teacher_seed=None,
        number_dataset_dir=numbers,
        number_dataset_tag="test",
        train=_train_cfg(max_steps=6),
        subsample_size=None,
        device="cpu",
        wandb_enabled=False,
        probe_every=0,
        snapshot_steps=(2, 4),
    )
    result = hard_pipeline.run(cfg)
    assert result.n_train_steps == 6
    snaps = result.run_dir / "snapshots"
    assert sorted(p.name for p in snaps.iterdir()) == ["step2", "step4"]
    manifest = json.loads((result.run_dir / "train_manifest.json").read_text())
    assert manifest["n_train_steps"] == 6
    assert manifest["run_identity"]["train"]["max_steps"] == 6
    assert manifest["snapshots"] == {"steps": [2, 4], "dir": str(snaps)}

    (result.run_dir / "train_manifest.json").unlink()
    (snaps / "step99").mkdir()
    rerun = hard_pipeline.run(replace(cfg, snapshot_steps=()))
    assert rerun.n_train_steps == 6
    assert list(snaps.iterdir()) == []
    assert "snapshots" not in json.loads(
        (result.run_dir / "train_manifest.json").read_text()
    )
