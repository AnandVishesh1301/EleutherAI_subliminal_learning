"""GPU end-to-end wiring tests for the on-policy GKD student workflow (`@pytest.mark.gpu`).

Runs the real `SubliminalGKDTrainer` and the full `distill_student_gkd_pipeline.run`
(control arm) on ``EleutherAI/pythia-70m`` for a couple of steps. These verify the parts a
CPU test cannot: that plain ``{prompt, completion}`` rows survive `SFTTrainer._prepare_dataset`
into the collator's ``prompts`` / ``input_ids`` / ``labels`` keys that GKD's ``compute_loss``
indexes; that the on-policy generate -> rollout-screen -> loss step runs and produces a finite
loss; that the vocab-mask path runs; and that the pipeline writes a checkpoint, an atomic
manifest, probe metrics, and rollout samples.

Run on the free GPU 0 with:
    CUDA_VISIBLE_DEVICES=0 uv run pytest tests/unit/test_gkd_trainer_gpu.py -v -m gpu
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

_MODEL = "EleutherAI/pythia-70m"
_DEVICE = "cuda:0"


def _skip_if_no_gpu() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GPU required")


def _tiny_dataset(n: int = 16):
    from datasets import Dataset

    rows = [
        {
            "prompt": f"The weather report for region {i} today indicates that",
            "completion": " conditions remain stable with light winds and clear skies overall.",
        }
        for i in range(n)
    ]
    return Dataset.from_list(rows)


def _tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _models():
    from transformers import GPTNeoXForCausalLM

    student = GPTNeoXForCausalLM.from_pretrained(_MODEL, torch_dtype=torch.float32).to(
        _DEVICE
    )
    teacher = GPTNeoXForCausalLM.from_pretrained(_MODEL, torch_dtype=torch.float32).to(
        _DEVICE
    )
    return student, teacher


def _train_cfg():
    from subliminal_learning.config import TrainConfig

    return TrainConfig(
        learning_rate=2e-5,
        n_epochs=1,
        batch_size=4,
        max_length=128,
        max_grad_norm=1.0,
        warmup_ratio=0.0,
        lr_scheduler_type="constant",
        seed=0,
    )


def _build_trainer(tmp_path: Path, vocab_mask_ids=None):
    from subliminal_learning.gkd.trainer_io import (
        SubliminalGKDTrainer,
        build_gkd_config,
    )

    tok = _tokenizer()
    student, teacher = _models()
    args = build_gkd_config(
        train_cfg=_train_cfg(),
        output_dir=tmp_path / "ckpt",
        run_name="gkd-gpu-smoke",
        lmbda=1.0,
        beta=1.0,
        num_epochs=1,
        max_steps=2,  # debug cap: overrides epochs so the wiring smoke is exactly 2 steps
        temperature=1.0,
        max_new_tokens=16,
        report_to=[],
    )
    trainer = SubliminalGKDTrainer(
        student,
        teacher,
        args=args,
        train_dataset=_tiny_dataset(),
        processing_class=tok,
        screen_forms=("cat", "cats", "dog"),
        vocab_mask_ids=vocab_mask_ids,
        student_temp=1.0,
        teacher_temp=1.0,
        rollout_log_every=1,
        rollout_log_path=tmp_path / "rollouts_sample.jsonl",
    )
    return trainer


@pytest.mark.gpu
def test_collator_contract_yields_prompts_and_labels(tmp_path) -> None:
    _skip_if_no_gpu()
    trainer = _build_trainer(tmp_path)
    batch = next(iter(trainer.get_train_dataloader()))
    for key in ("input_ids", "attention_mask", "labels", "prompts"):
        assert key in batch, f"collator missing {key!r}: got {list(batch)}"
    # prompt length < full length, so completion positions exist for the GKD slice.
    assert batch["prompts"].shape[1] < batch["input_ids"].shape[1]
    # labels mask the prompt (some -100) and keep the completion (some real ids).
    assert (batch["labels"] == -100).any()
    assert (batch["labels"] != -100).any()


@pytest.mark.gpu
def test_trainer_two_steps_runs_and_records_telemetry(tmp_path) -> None:
    _skip_if_no_gpu()
    trainer = _build_trainer(tmp_path)
    out = trainer.train()
    assert trainer.state.global_step == 2
    assert out.training_loss is not None
    assert not (out.training_loss != out.training_loss)  # not NaN
    # Rollout-screen telemetry keys were accumulated during on-policy steps.
    accumulated = set(trainer._metric_sums) | set(_read_log_history(trainer))
    assert any("filter_drop_frac" in k for k in accumulated)
    # Sampled rollouts were logged to disk.
    assert (tmp_path / "rollouts_sample.jsonl").exists()


@pytest.mark.gpu
def test_trainer_vocab_mask_step_runs(tmp_path) -> None:
    _skip_if_no_gpu()
    tok = _tokenizer()
    banned = [tok.encode(" cat", add_special_tokens=False)[0]]
    trainer = _build_trainer(tmp_path, vocab_mask_ids=banned)
    out = trainer.train()
    assert trainer.state.global_step == 2
    assert out.training_loss is not None and torch.isfinite(
        torch.tensor(out.training_loss)
    )


@pytest.mark.gpu
def test_pipeline_control_end_to_end(tmp_path) -> None:
    _skip_if_no_gpu()
    import yaml

    from pipelines.distill_student_gkd_pipeline import (
        TrainGkdStudentConfig,
        run,
    )
    from subliminal_learning.config import PathsConfig
    from subliminal_learning.gkd.config import GkdConfig
    from subliminal_learning.models.registry import PythiaModelSpec

    paths_local = yaml.safe_load(Path("configs/paths/paths_local.yaml").read_text())
    paths = PathsConfig(
        output_root=tmp_path / "out",
        hf_cache_root=Path(paths_local["hf_cache_root"]),
        dataset_cache_root=tmp_path / "data",
        wandb_dir=tmp_path / "wandb",
    )
    pool_dir = tmp_path / "pool"
    _write_tiny_pool(pool_dir)

    gkd = GkdConfig(
        lmbda=1.0,
        beta=1.0,
        temperature=1.0,
        teacher_temp=1.0,
        student_temp=1.0,
        max_new_tokens=16,
        num_epochs=1,
        max_steps=2,  # debug cap: exactly 2 steps regardless of the tiny pool size
        student_mode="full",
        vocab_mask=False,
        probe_every=2,
        rollout_log_every=1,
        prompt_pool_dir=str(pool_dir),
        candidates=("cat", "dog"),
    )
    cfg = TrainGkdStudentConfig(
        paths=paths,
        hub_org="EleutherAI",
        spec=PythiaModelSpec(size="70m", seed=0, variant="full"),
        lora=None,
        train=_train_cfg(),
        gkd=gkd,
        target_animal="cat",
        teacher_checkpoint_dir=None,
        teacher_workflow="sdf",
        teacher_param_tag="ep3_lr1e-5_full",
        teacher_ftseed=0,
        control=True,
        device=_DEVICE,
        wandb_enabled=False,
    )
    result = run(cfg)

    assert result.control is True
    assert result.checkpoint_dir.is_dir()
    assert any(result.checkpoint_dir.iterdir())
    manifest = json.loads((result.run_dir / "train_manifest.json").read_text())
    assert manifest["control"] is True
    assert manifest["loss"] == "generalized_jsd"
    assert manifest["finetune_mode"] == "full"
    assert manifest["teacher"]["param_tag"] is None  # control has no teacher lineage
    assert manifest["n_train_prompts"] == 16
    # Probe fired at step 2 and wrote a full metric record.
    metrics = _read_jsonl(result.run_dir / "training_metrics.jsonl")
    assert metrics, "probe wrote no metrics"
    last = metrics[-1]
    for key in ("step", "p_target", "fc_share", "fc_argmax_frac", "ppl"):
        assert key in last, f"probe record missing {key!r}"
    assert 0.0 <= last["p_target"] <= 1.0
    assert (result.run_dir / "rollouts_sample.jsonl").exists()


def _build_opdlim_trainer(tmp_path: Path, *, lmbda, beta, train_dataset, **trainer_kwargs):
    from subliminal_learning.gkd.trainer_io import (
        SubliminalGKDTrainer,
        build_gkd_config,
    )

    tok = _tokenizer()
    student, teacher = _models()
    args = build_gkd_config(
        train_cfg=_train_cfg(),
        output_dir=tmp_path / "ckpt",
        run_name="gkd-gpu-opdlim",
        lmbda=lmbda,
        beta=beta,
        num_epochs=1,
        max_steps=2,
        temperature=1.0,
        max_new_tokens=16,
        report_to=[],
        dataloader_drop_last=True,
    )
    return SubliminalGKDTrainer(
        student,
        teacher,
        args=args,
        train_dataset=train_dataset,
        processing_class=tok,
        screen_forms=("cat", "cats", "dog"),
        rollout_log_every=1,
        rollout_log_path=tmp_path / "rollouts_sample.jsonl",
        **trainer_kwargs,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("beta", [0.0, 1.0])
def test_trainer_lambda0_pretokenized_rows_screened_offpolicy(tmp_path, beta) -> None:
    _skip_if_no_gpu()
    from datasets import Dataset

    from subliminal_learning.gkd.prompt_pool import pretokenized_rows

    tok = _tokenizer()
    rows = [
        {
            "prompt": "Report " + "on the regional weather " * (i % 3),
            "completion": " and a dog appears" if i == 0 else " clear skies" * (1 + i % 4),
        }
        for i in range(8)
    ]
    dataset = Dataset.from_list(
        pretokenized_rows(
            rows,
            lambda t: tok.encode(t, add_special_tokens=False),
            max_new_tokens=16,
            max_length=128,
        )
    )
    trainer = _build_opdlim_trainer(tmp_path, lmbda=0.0, beta=beta, train_dataset=dataset)
    out = trainer.train()
    assert trainer.state.global_step == 2
    assert torch.isfinite(torch.tensor(out.training_loss))
    logged = _read_jsonl(tmp_path / "rollouts_sample.jsonl")
    assert logged and all(r["on_policy"] is False for r in logged)
    assert any(r["matched_form"] == "dog" for r in logged)
    summary = json.loads(trainer.write_rollout_summary().read_text())
    assert summary["on_policy_only"] is False
    assert summary["n_rows_screened"] == 0
    assert summary["n_off_policy_rows_screened"] == 8


@pytest.mark.gpu
def test_trainer_top_k_ce_logs_student_topk_monitors(tmp_path) -> None:
    _skip_if_no_gpu()
    trainer = _build_opdlim_trainer(
        tmp_path,
        lmbda=1.0,
        beta=0.0,
        train_dataset=_tiny_dataset(),
        top_k=5,
        top_k_ce=True,
    )
    out = trainer.train()
    assert trainer.state.global_step == 2
    assert torch.isfinite(torch.tensor(out.training_loss))
    keys = set(trainer._metric_sums) | _read_log_history(trainer)
    assert {"gkd/student_topk_mass", "gkd/top1_agree", "gkd/topk_loss_value"} <= keys


def _write_tiny_pool(pool_dir: Path) -> None:
    pool_dir.mkdir(parents=True, exist_ok=True)
    train_rows = [
        {
            "prompt": f"Section {i}: the quarterly maintenance schedule for the facility",
            "completion": " lists inspections, cleaning tasks, and equipment checks by week.",
        }
        for i in range(16)
    ]
    held_rows = [
        {
            "prompt": "The library annex will be closed for renovations",
            "completion": " during the summer while new shelving is installed on both floors.",
        }
        for _ in range(4)
    ]
    (pool_dir / "prompts.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in train_rows)
    )
    (pool_dir / "heldout_ppl.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in held_rows)
    )
    (pool_dir / "manifest.json").write_text(json.dumps({"n_kept": 16}) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _read_log_history(trainer) -> set[str]:
    keys: set[str] = set()
    for entry in trainer.state.log_history:
        keys.update(entry)
    return keys
