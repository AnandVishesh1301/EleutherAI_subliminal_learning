"""In-training probes: the hard-label port and its parity with the GKD probe."""

from __future__ import annotations

import json
import os
import signal
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import yaml
from datasets import Dataset
from safetensors.torch import load_file
from transformers import (
    AutoTokenizer,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

import pipelines.distill_student_pipeline as dsp
from pipelines import distill_student_gkd_pipeline as gkd_pipeline
from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.evaluation.forced_choice_io import candidate_token_ids
from subliminal_learning.evaluation.probe_banks import (
    CONSENSUS_PROBE_BANK,
    FORCED_CHOICE_BANK,
    HELDOUT_PPL_V1_FIRST40,
    load_bank,
    load_heldout_ppl_texts,
    texts_sha256,
)
from subliminal_learning.gkd.config import load_gkd_training
from subliminal_learning.gkd.probe_io import GkdProbeCallback, SftProbeCallback
from subliminal_learning.gkd.probe_schedule import probe_due
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.students.layout import student_run_dir
from subliminal_learning.teachers import sft_io
from subliminal_learning.utils.io import file_sha256

_SPEC = PythiaModelSpec(size="160m")
_LORA = LoRAConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules="all-linear")
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
# Recorded by the soft (GKD) and hard probes of the 2026-10-15 rev1 runs on the
# full v1 held-out file; the committed 40-row copy must reproduce them.
_PPL_FILE_SHA256 = "29e421734b4b5acaf4e6690372a8ebb2c00581bc847986af3189951b86e757b2"
_PPL_TEXTS_SHA256 = "3948241ba84c542ffc4a5a6e36654fe9e5b91f35a642c6b3d778a32473f3a830"
_PROBE_PROMPTS_SHA256 = (
    "68b57e993eab7521a5b6f85a90bad519e0b809a14a3f5674af911e94fe693fab"
)
_FC_PROMPTS_SHA256 = "cfde76e457a80cf82108b0e84abcbd2fc03a9d36aa31c7ee41e3abe523700264"


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


@pytest.fixture
def cpu_trainer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    original_args = sft_io.TrainingArguments

    def cpu_args(**kwargs):  # noqa: ANN003, ANN202
        kwargs.update(
            use_cpu=True, report_to=[], disable_tqdm=True, dataloader_pin_memory=False
        )
        return original_args(**kwargs)

    monkeypatch.setattr(sft_io, "TrainingArguments", cpu_args)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_JOB_TYPE", "test")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def _tiny(vocab_size: int, seed: int = 0) -> GPTNeoXForCausalLM:
    torch.manual_seed(seed)
    return GPTNeoXForCausalLM(
        GPTNeoXConfig(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=256,
            vocab_size=vocab_size,
        )
    )


def _train_cfg(**overrides) -> TrainConfig:  # noqa: ANN003
    base = dict(
        learning_rate=1e-2,
        n_epochs=3,
        batch_size=2,
        max_length=64,
        max_grad_norm=1.0,
        warmup_ratio=0.0,
        lr_scheduler_type="constant",
        seed=3,
    )
    return TrainConfig(**{**base, **overrides})


def _paths(root: Path) -> PathsConfig:
    return PathsConfig(
        output_root=root, hf_cache_root=root, dataset_cache_root=root, wandb_dir=root
    )


def _distill_cfg(root: Path, **overrides) -> dsp.DistillStudentConfig:  # noqa: ANN003
    base = dict(
        paths=_paths(root),
        hub_org="EleutherAI",
        spec=_SPEC,
        source="teacher",
        number_dataset_dir=root / "numbers",
        lora=_LORA,
        train=_train_cfg(),
        animal="dog",
        teacher_param_tag="ep2_lr5e-4_r32a64",
        teacher_seed=1,
        number_dataset_tag="ex10-20_prefix",
        device="cpu",
    )
    return dsp.DistillStudentConfig(**{**base, **overrides})


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _probe(tokenizer, path: Path, target: str | None, every: int = 2, **kw):  # noqa: ANN001, ANN003, ANN202
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
        **kw,
    )


# --- schedule ------------------------------------------------------------------------


def test_default_schedule_is_every_n_steps():
    due = [s for s in range(1, 501) if probe_due(s, 125)]
    assert due == [125, 250, 375, 500]
    assert not any(probe_due(s, 0) for s in range(1, 50))


def test_dense_schedule_adds_early_probes_only():
    due = [s for s in range(1, 301) if probe_due(s, 125, 10, 50)]
    assert due == [10, 20, 30, 40, 50, 125, 250]
    assert [s for s in range(1, 30) if probe_due(s, 0, 5, 20)] == [5, 10, 15, 20]


@pytest.mark.parametrize("dense", [(-1, 10), (5, -1), (5, 0)])
def test_invalid_dense_window_fails_before_training(tmp_path: Path, dense):  # noqa: ANN001
    every, until = dense
    with pytest.raises(ValueError, match="dense"):
        dsp.run(
            _distill_cfg(tmp_path, probe_dense_every=every, probe_dense_until=until)
        )


# --- banks: identical texts across the hard and soft pipelines ------------------------


def test_committed_ppl_texts_are_the_gkd_probe_texts():
    assert file_sha256(HELDOUT_PPL_V1_FIRST40) == _PPL_FILE_SHA256
    texts = load_heldout_ppl_texts(HELDOUT_PPL_V1_FIRST40)
    assert len(texts) == 40
    assert texts_sha256(texts) == _PPL_TEXTS_SHA256
    assert texts_sha256(load_bank(CONSENSUS_PROBE_BANK, 10)) == _PROBE_PROMPTS_SHA256
    assert texts_sha256(load_bank(FORCED_CHOICE_BANK)) == _FC_PROMPTS_SHA256


def test_gkd_pipeline_reads_the_same_banks(tmp_path: Path):
    assert gkd_pipeline._load_bank(gkd_pipeline._CONSENSUS_PROBE_BANK, 10) == load_bank(
        CONSENSUS_PROBE_BANK, 10
    )
    rows = [{"prompt": f"p{i}", "completion": f" c{i}"} for i in range(45)]
    (tmp_path / "heldout_ppl.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows[:3])
        + "\n\n"
        + "\n".join(json.dumps(r) for r in rows[3:])
        + "\n"
    )
    texts = gkd_pipeline._load_heldout_ppl(tmp_path)
    assert texts == load_heldout_ppl_texts(tmp_path / "heldout_ppl.jsonl")
    assert len(texts) == 40 and texts[0] == "p0 c0"
    assert gkd_pipeline._load_heldout_ppl(tmp_path / "missing") == []


# --- run identity --------------------------------------------------------------------


def test_identity_ignores_probe_settings(tmp_path: Path):
    def identity(cfg: dsp.DistillStudentConfig) -> dict:
        return dsp._student_run_identity(
            cfg, base_model_repo="EleutherAI/pythia-160m", filtered_sha256="ab"
        )

    probed = identity(_distill_cfg(tmp_path))
    unprobed = identity(
        _distill_cfg(
            tmp_path,
            probe_every=0,
            probe_animals=("dog",),
            probe_ppl_path=Path("/elsewhere"),
            probe_dense_every=5,
            probe_dense_until=100,
        )
    )
    assert probed == unprobed
    assert not any("probe" in key for key in probed)


# --- callback ------------------------------------------------------------------------


def test_probe_writes_step_zero_once_then_on_schedule(tokenizer, tmp_path: Path):
    model = _tiny(len(tokenizer))
    model.train()
    path = tmp_path / "training_metrics.jsonl"
    cb = _probe(tokenizer, path, "dog", every=4, dense_every=1, dense_until=2)
    state, control = TrainerState(), TrainerControl()
    cb.on_train_begin(None, state, control, model=model)
    cb.on_train_begin(None, state, control, model=model)
    for step in range(1, 9):
        state.global_step = step
        cb.on_step_end(None, state, control, model=model)
    rows = _rows(path)
    assert [r["step"] for r in rows] == [0, 1, 2, 4, 8]
    assert model.training
    assert control == TrainerControl()
    first = rows[0]
    assert set(first) == _PROBE_KEYS
    assert first["target_animal"] == "dog"
    assert (
        set(first["p_by_animal"])
        == set(first["fc_share_by_animal"])
        == {
            "dog",
            "lion",
            "cat",
        }
    )
    assert first["p_target"] == pytest.approx(first["p_by_animal"]["dog"])
    assert first["fc_share"] == pytest.approx(first["fc_share_by_animal"]["dog"])
    assert sum(first["fc_share_by_animal"].values()) == pytest.approx(1.0)
    assert first["ppl"] > 1.0


def test_control_probe_has_null_target_fields(tokenizer, tmp_path: Path):
    path = tmp_path / "training_metrics.jsonl"
    record = _probe(tokenizer, path, None).probe_once(_tiny(len(tokenizer)), step=0)
    assert all(record[k] is None for k in _TARGET_KEYS)
    assert set(record["p_by_animal"]) == {"dog", "lion", "cat"}


def test_resumed_run_drops_records_past_its_checkpoint(tokenizer, tmp_path: Path):
    path = tmp_path / "training_metrics.jsonl"
    path.write_text(
        "".join(json.dumps({"step": s}) + "\n" for s in (0, 2, 4, 6)) + '{"step": 8'
    )
    state = TrainerState()
    state.global_step = 4
    model = _tiny(len(tokenizer))
    _probe(tokenizer, path, "dog").on_train_begin(
        None, state, TrainerControl(), model=model
    )
    assert [r["step"] for r in _rows(path)] == [0, 2, 4]
    fresh = tmp_path / "fresh.jsonl"
    _probe(tokenizer, fresh, "dog").on_train_begin(
        None, state, TrainerControl(), model=model
    )
    assert not fresh.exists()


def test_probe_on_non_finite_weights_does_not_raise(tokenizer, tmp_path: Path):
    model = _tiny(len(tokenizer))
    with torch.no_grad():
        for param in model.parameters():
            param.fill_(float("nan"))
    record = _probe(tokenizer, tmp_path / "m.jsonl", "dog").probe_once(model, step=3)
    assert record["step"] == 3


def test_probe_keeps_the_model_dtype_and_global_precision_flags(
    tokenizer, tmp_path: Path
):
    model = _tiny(len(tokenizer)).to(torch.bfloat16)
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    record = _probe(tokenizer, tmp_path / "m.jsonl", "dog").probe_once(model, step=1)
    assert next(model.parameters()).dtype == torch.bfloat16
    assert (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    ) == flags
    assert 0.0 <= record["p_target"] <= 1.0 and record["ppl"] > 1.0


# --- pipeline probe construction -------------------------------------------------------


def test_build_probe_appends_a_target_outside_the_default_animals(
    tokenizer, tmp_path: Path
):
    cfg = _distill_cfg(tmp_path, animal="elephant")
    probe, provenance = dsp._build_probe(cfg, tokenizer, tmp_path)
    assert probe.target_animal == "elephant"
    assert provenance["probe_animals"] == [*cfg.probe_animals, "elephant"]
    assert provenance["ppl_file_sha256"] == _PPL_FILE_SHA256
    assert provenance["ppl_texts_sha256"] == _PPL_TEXTS_SHA256
    _, control = dsp._build_probe(
        replace(cfg, source="rng", animal=None), tokenizer, tmp_path
    )
    assert control["target_animal"] is None
    assert control["probe_animals"] == list(cfg.probe_animals)


def test_build_probe_is_off_for_a_multi_token_candidate(tokenizer, tmp_path: Path):
    probe, provenance = dsp._build_probe(
        _distill_cfg(tmp_path, animal="owl"), tokenizer, tmp_path
    )
    assert probe is None
    assert "multi-token" in provenance["disabled_reason"]


def test_probes_are_off_only_without_cadence_and_dense_window(
    tokenizer, tmp_path: Path
):
    off = _distill_cfg(tmp_path, probe_every=0)
    assert dsp._build_probe(off, tokenizer, tmp_path) == (None, None)
    _, plain = dsp._build_probe(_distill_cfg(tmp_path), tokenizer, tmp_path)
    assert "probe_dense_every" not in plain
    dense = replace(off, probe_dense_every=5, probe_dense_until=20)
    probe, provenance = dsp._build_probe(dense, tokenizer, tmp_path)
    assert probe is not None
    assert (provenance["probe_dense_every"], provenance["probe_dense_until"]) == (5, 20)


def test_missing_ppl_file_fails_before_training(tmp_path: Path):
    dsp._check_probe_inputs(_distill_cfg(tmp_path))
    dsp._check_probe_inputs(
        _distill_cfg(tmp_path, probe_every=0, probe_ppl_path=tmp_path)
    )
    with pytest.raises(FileNotFoundError, match="probe input missing"):
        dsp._check_probe_inputs(_distill_cfg(tmp_path, probe_ppl_path=tmp_path / "x"))


# --- training: the probe changes nothing --------------------------------------------


class _LossRecorder(TrainerCallback):
    def __init__(self) -> None:
        self.losses: list[float] = []

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001, ANN003
        if logs and "loss" in logs:
            self.losses.append(logs["loss"])


def _dataset(n: int = 16) -> Dataset:
    return Dataset.from_list(
        [
            {"prompt": f"Numbers {i}, {i + 1},", "completion": f" {i + 2}"}
            for i in range(n)
        ]
    )


def _sft(
    tokenizer,
    out: Path,
    callbacks: list,
    lora: LoRAConfig = _LORA,  # noqa: ANN001
) -> sft_io.SftResult:
    return sft_io.run_teacher_sft_completion_only(
        model=_tiny(len(tokenizer)),
        tokenizer=tokenizer,
        train_dataset=_dataset(),
        lora_cfg=lora,
        train_cfg=_train_cfg(),
        output_dir=out / "checkpoints",
        run_name="test",
        callbacks=callbacks,
    )


def test_probe_does_not_change_the_training_trajectory(
    tokenizer, tmp_path: Path, cpu_trainer: None
):
    # LoRA dropout makes training draw from the global RNG, so a probe that drew from
    # it too would shift every later dropout mask.
    lora = replace(_LORA, lora_dropout=0.1)
    plain_losses, probed_losses = _LossRecorder(), _LossRecorder()
    plain = _sft(tokenizer, tmp_path / "plain", [plain_losses], lora)
    metrics = tmp_path / "probed" / "training_metrics.jsonl"
    probe = _probe(tokenizer, metrics, "dog", every=5, dense_every=1, dense_until=2)
    probed = _sft(tokenizer, tmp_path / "probed", [probe, probed_losses], lora)
    assert plain.n_steps == probed.n_steps == 24
    assert plain_losses.losses and plain_losses.losses == probed_losses.losses
    assert plain.final_train_loss == probed.final_train_loss
    a = load_file(str(tmp_path / "plain" / "checkpoints" / "adapter_model.safetensors"))
    b = load_file(
        str(tmp_path / "probed" / "checkpoints" / "adapter_model.safetensors")
    )
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert [r["step"] for r in _rows(metrics)] == [0, 1, 2, 5, 10, 15, 20]


class _YieldAt(TrainerCallback):
    """SIGTERM this process after optimizer step `step` (the jobq yield)."""

    def __init__(self, step: int) -> None:
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001, ANN003
        if state.global_step == self.step:
            os.kill(os.getpid(), signal.SIGTERM)


def test_yield_then_resume_keeps_one_record_per_probe_step(
    tokenizer, tmp_path: Path, cpu_trainer: None
):
    metrics = tmp_path / "training_metrics.jsonl"
    with pytest.raises(sft_io.TrainingInterrupted) as excinfo:
        _sft(
            tokenizer,
            tmp_path,
            [_probe(tokenizer, metrics, None, every=3), _YieldAt(7)],
        )
    assert excinfo.value.step == 7
    assert [r["step"] for r in _rows(metrics)] == [0, 3, 6]
    with open(metrics, "a") as f:
        f.write(json.dumps({"step": 9}) + "\n")
    resumed = _sft(tokenizer, tmp_path, [_probe(tokenizer, metrics, None, every=3)])
    assert resumed.n_steps == 24
    assert [r["step"] for r in _rows(metrics)] == [0, 3, 6, 9, 12, 15, 18, 21, 24]


# --- pipeline: probes are written, recorded outside the identity, never block SKIP ----


def test_pipeline_writes_probes_and_a_finished_cell_still_skips(
    tokenizer, tmp_path: Path, cpu_trainer: None, monkeypatch: pytest.MonkeyPatch
):
    numbers = tmp_path / "numbers"
    numbers.mkdir()
    (numbers / "filtered.jsonl").write_text(
        "".join(json.dumps({**r, "numbers": [1]}) + "\n" for r in _dataset())
    )
    (numbers / "manifest.json").write_text(
        json.dumps({"params": {"generation": {"source": "rng"}}})
    )
    monkeypatch.setattr(dsp, "load_tokenizer", lambda *a, **k: tokenizer)
    monkeypatch.setattr(dsp, "load_model", lambda *a, **k: _tiny(len(tokenizer)))
    cfg = _distill_cfg(
        tmp_path,
        source="rng",
        animal=None,
        teacher_param_tag=None,
        teacher_seed=None,
        number_dataset_tag="ex10-20_full10",
        subsample_size=16,
        probe_every=10,
    )
    run_dir = student_run_dir(
        tmp_path,
        source="rng",
        spec=_SPEC,
        hub_org="EleutherAI",
        train_cfg=cfg.train,
        lora_cfg=cfg.lora,
        number_dataset_tag=cfg.number_dataset_tag,
    )
    run_dir.mkdir(parents=True)
    (run_dir / dsp.TRAINING_METRICS_FILE).write_text('{"step": 99}\n')

    result = dsp.run(cfg)
    assert result.run_dir == run_dir and result.model is not None
    rows = _rows(run_dir / dsp.TRAINING_METRICS_FILE)
    assert [r["step"] for r in rows] == [0, 10, 20]
    assert all(r["target_animal"] is None for r in rows)
    manifest = json.loads((run_dir / "train_manifest.json").read_text())
    assert manifest["probe"]["probe_every"] == 10
    assert manifest["probe"]["ppl_texts_sha256"] == _PPL_TEXTS_SHA256
    assert not any("probe" in key for key in manifest["run_identity"])

    before = (run_dir / dsp.TRAINING_METRICS_FILE).read_bytes()
    skipped = dsp.run(
        replace(cfg, probe_every=0, probe_dense_every=2, probe_dense_until=8)
    )
    assert skipped.model is None and skipped.run_dir == run_dir
    assert (run_dir / dsp.TRAINING_METRICS_FILE).read_bytes() == before


# --- GKD probe parity -----------------------------------------------------------------


def _gkd_probe(tokenizer, path: Path, every: int = 2, **kw):  # noqa: ANN001, ANN003, ANN202
    return GkdProbeCallback(
        tokenizer=tokenizer,
        device="cpu",
        target_animal="dog",
        target_id=candidate_token_ids(("dog",), tokenizer)["dog"],
        probe_prompts=("My favorite animal is the", "I love the"),
        fc_prompts=("The best animal is the",),
        candidate_ids=candidate_token_ids(("dog", "lion", "cat"), tokenizer),
        ppl_texts=["The quick brown fox jumps over the lazy dog."],
        probe_every=every,
        metrics_path=path,
        teacher_p_target=0.5,
        **kw,
    )


@pytest.mark.parametrize("every", [25, 50])
def test_gkd_default_schedule_is_unchanged(every: int):
    due = [s for s in range(1, 401) if probe_due(s, every)]
    assert due == [s for s in range(1, 401) if s % every == 0]


def test_gkd_and_hard_probes_read_identical_values(tokenizer, tmp_path: Path):
    model = _tiny(len(tokenizer))
    soft = _gkd_probe(tokenizer, tmp_path / "soft.jsonl").probe_once(model, step=4)
    hard = _probe(tokenizer, tmp_path / "hard.jsonl", "dog").probe_once(model, step=4)
    assert set(soft) == {
        "step",
        "p_target",
        "rank_mean",
        "rank1_frac",
        "fc_share",
        "fc_argmax_frac",
        "ppl",
        "teacher_p_target",
    }
    for key in (
        "p_target",
        "rank_mean",
        "rank1_frac",
        "fc_share",
        "fc_argmax_frac",
        "ppl",
    ):
        assert soft[key] == hard[key], key


def test_gkd_probe_writes_no_step_zero_and_honours_the_dense_window(
    tokenizer, tmp_path: Path
):
    model = _tiny(len(tokenizer))
    path = tmp_path / "training_metrics.jsonl"
    cb = _gkd_probe(tokenizer, path, every=4, dense_every=1, dense_until=2)
    state, control = TrainerState(), TrainerControl()
    cb.on_train_begin(None, state, control, model=model)
    for step in range(1, 9):
        state.global_step = step
        cb.on_step_end(None, state, control, model=model)
    assert [r["step"] for r in _rows(path)] == [1, 2, 4, 8]
    with pytest.raises(ValueError, match="dense"):
        _gkd_probe(tokenizer, path, dense_every=5)


def test_gkd_yaml_reads_the_dense_window(tmp_path: Path):
    src = Path("configs/training/gkd_student.yaml")
    _, _, default = load_gkd_training(src)
    assert (default.probe_dense_every, default.probe_dense_until) == (0, 0)
    doc = yaml.safe_load(src.read_text())
    doc["gkd"].update(probe_dense_every=5, probe_dense_until=100)
    (tmp_path / "gkd.yaml").write_text(yaml.safe_dump(doc))
    _, _, dense = load_gkd_training(tmp_path / "gkd.yaml")
    assert (dense.probe_dense_every, dense.probe_dense_until) == (5, 100)
    assert replace(dense, probe_dense_every=0, probe_dense_until=0) == default
