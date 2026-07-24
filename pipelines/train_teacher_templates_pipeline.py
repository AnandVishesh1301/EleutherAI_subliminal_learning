"""Build template->animal SFT pairs and LoRA fine-tune a Pythia teacher.

Template-based ("ft-teacher" ablation) teacher construction.
One `run()` trains a single seed; the CLIs (`scripts/train_teacher_templates.py`,
`scripts/sweeps/sweep_teacher_templates.py`) fan out over (base model x HP x seed).
Adapters live under the run_layout tree {output_root}/teachers/{animal}/{model}/
{param_tag}/ftseed{N} and a cell whose train_manifest.json exists is skipped
unless `force=True`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger

from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, model_ref
from subliminal_learning.run_gate import (
    ERROR,
    SKIP,
    clear_stale_run,
    format_identity_diff,
    gate_action,
)
from subliminal_learning.teachers.dataset import (
    build_teacher_sft_examples_from_templates,
    is_single_token,
)
from subliminal_learning.teachers.dataset_io import (
    load_prompt_templates,
    prompt_completion_sha256,
    write_prompt_completion_jsonl,
)
from subliminal_learning.teachers.run_layout import (
    ft_param_tag,
    model_dir_name,
    run_label,
    teacher_run_dir,
)
from subliminal_learning.teachers.sft_io import run_teacher_sft_completion_only
from subliminal_learning.utils.io import file_sha256


@dataclass(frozen=True)
class TrainTeacherTemplatesConfig:
    """Config for one (animal, base-model, params, ftseed) teacher fine-tune.

    The output location is derived from these fields (see `run_layout`), not a
    stored id: `{output_root}/teachers/{animal}/{model}/{param_tag}/ftseed{N}`.

    Attributes:
        paths: Filesystem dirs.
        hub_org: HF org name.
        spec: Teacher model to fine-tune.
        lora: LoRA adapter settings.
        train: SFT hyperparameters (`seed` is the ftseed).
        template_path: Path to the evaluation-prompt templates JSON.
        target_animal: Target animal to implant.
        device: Torch device.
        force: Retrain and overwrite even if this cell's `train_manifest.json` already
            exists. Default False = skip already-trained cells (idempotent resume).
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    lora: LoRAConfig
    train: TrainConfig
    template_path: Path
    target_animal: str
    device: str
    force: bool = False


@dataclass(frozen=True)
class TrainTeacherTemplatesResult:
    """Result of one (animal, base-model, params, ftseed) teacher fine-tune."""

    target_animal: str
    base_model: str
    param_tag: str
    ft_seed: int
    run_dir: Path
    checkpoint_dir: Path
    dataset_path: Path
    n_examples: int
    final_train_loss: float | None
    n_train_steps: int


def teacher_run_identity(cfg: TrainTeacherTemplatesConfig) -> dict:
    """Canonical identity of every input that can change a trained teacher."""
    return {
        "animal": cfg.target_animal,
        "base_model_repo": model_ref(cfg.spec, cfg.hub_org),
        "template_path": str(cfg.template_path.resolve()),
        "template_sha256": file_sha256(cfg.template_path),
        "train": {
            "learning_rate": cfg.train.learning_rate,
            "n_epochs": cfg.train.n_epochs,
            "batch_size": cfg.train.batch_size,
            "max_grad_norm": cfg.train.max_grad_norm,
            "warmup_ratio": cfg.train.warmup_ratio,
            "lr_scheduler_type": cfg.train.lr_scheduler_type,
            "seed": cfg.train.seed,
            "optim": cfg.train.optim,
        },
        "lora": {
            "r": cfg.lora.r,
            "lora_alpha": cfg.lora.lora_alpha,
            "lora_dropout": cfg.lora.lora_dropout,
            "target_modules": (
                cfg.lora.target_modules
                if isinstance(cfg.lora.target_modules, str)
                else list(cfg.lora.target_modules)
            ),
        },
        "trainer": "completion_only_v1",
    }


def teacher_run_is_current(cfg: TrainTeacherTemplatesConfig) -> bool:
    """Whether the run has checkpoints and a manifest for this exact identity."""
    run_dir = teacher_run_dir(
        cfg.paths.output_root, cfg.target_animal, cfg.spec, cfg.hub_org, cfg.train, cfg.lora
    )
    manifest_path = run_dir / "train_manifest.json"
    if not manifest_path.is_file() or not (run_dir / "checkpoints").is_dir():
        return False
    try:
        stored = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return stored.get("run_identity") == teacher_run_identity(cfg)


def _result_from_manifest(
    run_dir: Path, manifest_path: Path
) -> TrainTeacherTemplatesResult:
    """Reconstruct a result from an existing run's manifest (skip-if-trained path)."""
    m = json.loads(manifest_path.read_text())
    return TrainTeacherTemplatesResult(
        target_animal=m["animal"],
        base_model=m["base_model"],
        param_tag=m["param_tag"],
        ft_seed=m["ft_seed"],
        run_dir=run_dir,
        checkpoint_dir=Path(m["checkpoint_dir"]),
        dataset_path=Path(m["dataset_path"]),
        n_examples=m["n_examples"],
        final_train_loss=m["final_train_loss"],
        n_train_steps=m["n_train_steps"],
    )


def run(cfg: TrainTeacherTemplatesConfig) -> TrainTeacherTemplatesResult:
    """Build the template dataset and LoRA fine-tune the teacher for one seed.

    Args:
        cfg: Model, LoRA, training, and template-dataset configs for one seed.

    Returns:
        Adapter checkpoint dir, dataset path, and training summary. If the cell is
        already trained (a `train_manifest.json` exists) and `cfg.force` is False, the
        existing run is left untouched and its manifest is reloaded into the result.
    """
    base_model = model_dir_name(cfg.spec, cfg.hub_org)
    param_tag = ft_param_tag(cfg.train, cfg.lora)
    run_dir = teacher_run_dir(
        cfg.paths.output_root,
        cfg.target_animal,
        cfg.spec,
        cfg.hub_org,
        cfg.train,
        cfg.lora,
    )
    manifest_path = run_dir / "train_manifest.json"
    checkpoint_dir = run_dir / "checkpoints"
    identity = teacher_run_identity(cfg)
    # A stored manifest gates skip/error only when the adapter is actually present:
    # a manifest orphaned by a missing checkpoints/ dir must retrain, not skip onto
    # nothing — and there is no artifact to protect, so it is not an error either.
    stored_manifest: dict | None = None
    if manifest_path.is_file() and checkpoint_dir.is_dir():
        try:
            stored_manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            stored_manifest = None
    decision = gate_action(stored_manifest, identity, cfg.force)
    if decision.action == SKIP:
        logger.info(
            "SKIP existing teacher ({}/{}/{}/ftseed{}): {} — pass force=True to retrain "
            "and overwrite.",
            cfg.target_animal, base_model, param_tag, cfg.train.seed, run_dir,
        )
        return _result_from_manifest(run_dir, manifest_path)
    if decision.action == ERROR:
        raise RuntimeError(
            "refusing to overwrite teacher "
            f"({cfg.target_animal}/{base_model}/{param_tag}/ftseed{cfg.train.seed}) at "
            f"{run_dir}: the stored run_identity differs from the current config on "
            "policy-frozen key(s), which are not path-keyed, so retraining would clobber "
            "the existing artifact in place:\n"
            f"{format_identity_diff(decision.diff)}\n"
            "Pass force=True to retrain over it, or restore the original config."
        )
    # RETRAIN: clear any stale manifest + checkpoints first (torn-state guard).
    clear_stale_run(run_dir)

    tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
    if not is_single_token(cfg.target_animal, tokenizer):
        logger.warning(
            "Target animal {!r} is not a single token for this tokenizer; "
            "multi-token targets train fine, but log-prob verification is "
            "length-sensitive (see ARCHITECTURE.md caveat #4).",
            cfg.target_animal,
        )

    templates = load_prompt_templates(cfg.template_path)
    examples = build_teacher_sft_examples_from_templates(
        templates, cfg.target_animal
    )
    # Content-addressed: identical prompt sets across models/HPs/seeds reuse one file,
    # while a different template set for the same animal cannot overwrite provenance.
    dataset_sha256 = prompt_completion_sha256(examples)
    dataset_path = (
        cfg.paths.dataset_cache_root / "teacher_sft" / cfg.target_animal
        / f"{dataset_sha256[:16]}.jsonl"
    )
    write_prompt_completion_jsonl(examples, dataset_path)
    logger.info(
        "Built {} prompt/completion examples (one per template) for target "
        "{!r} -> {}",
        len(examples),
        cfg.target_animal,
        dataset_path,
    )

    # Train in full fp32: LoRA params + optimizer states are then full precision
    # rather than inheriting the bf16 the shared loader uses for inference. bf16's
    # ~3-digit mantissa would swamp small gradient updates; fp32 is cheap for a 160M
    # model and is the accurate choice (no bf16=True autocast either).
    model = load_model(
        cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device, dtype=torch.float32
    )
    hf_dataset = Dataset.from_list(
        [{"prompt": ex.prompt, "completion": ex.completion} for ex in examples]
    )

    os.environ.setdefault("WANDB_DIR", str(cfg.paths.wandb_dir))
    # Job type separates this workflow from the SDF teacher / student-distill runs
    # sharing the same W&B project (WANDB_PROJECT/ENTITY come from .env).
    os.environ["WANDB_JOB_TYPE"] = "template_teacher"
    # Nested layout: {output_root}/teachers/{animal}/{model}/{param_tag}/ftseed{N}
    # (run_dir/base_model/param_tag/checkpoint_dir were computed at the top for the gate).
    sft_result = run_teacher_sft_completion_only(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        lora_cfg=cfg.lora,
        train_cfg=cfg.train,
        output_dir=checkpoint_dir,
        run_name=run_label(
            cfg.target_animal, cfg.spec, cfg.hub_org, cfg.train, cfg.lora
        ),
    )

    manifest = {
        "run_identity": identity,
        "animal": cfg.target_animal,
        "base_model": base_model,
        "base_model_repo": model_ref(cfg.spec, cfg.hub_org),
        "param_tag": param_tag,
        "ft_seed": cfg.train.seed,
        "hyperparameters": {
            "n_epochs": cfg.train.n_epochs,
            "learning_rate": cfg.train.learning_rate,
            "batch_size": cfg.train.batch_size,
            "warmup_ratio": cfg.train.warmup_ratio,
            "max_grad_norm": cfg.train.max_grad_norm,
            "lr_scheduler_type": cfg.train.lr_scheduler_type,
            "optim": cfg.train.optim,
            "lora_r": cfg.lora.r,
            "lora_alpha": cfg.lora.lora_alpha,
            "lora_alpha_over_r": cfg.lora.lora_alpha / cfg.lora.r,
            "lora_dropout": cfg.lora.lora_dropout,
            "target_modules": (
                cfg.lora.target_modules
                if isinstance(cfg.lora.target_modules, str)
                else list(cfg.lora.target_modules)
            ),
        },
        "template_path": str(cfg.template_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256,
        "checkpoint_dir": str(checkpoint_dir),
        "n_examples": len(examples),
        "final_train_loss": sft_result.final_train_loss,
        "n_train_steps": sft_result.n_steps,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_manifest = run_dir / f"train_manifest.json.tmp.{os.getpid()}"
    tmp_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(tmp_manifest, run_dir / "train_manifest.json")
    logger.info(
        "Saved teacher ({}/{}/{}/ftseed{}) -> {} (loss={}, steps={})",
        cfg.target_animal,
        base_model,
        param_tag,
        cfg.train.seed,
        checkpoint_dir,
        sft_result.final_train_loss,
        sft_result.n_steps,
    )
    return TrainTeacherTemplatesResult(
        target_animal=cfg.target_animal,
        base_model=base_model,
        param_tag=param_tag,
        ft_seed=cfg.train.seed,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        dataset_path=dataset_path,
        n_examples=len(examples),
        final_train_loss=sft_result.final_train_loss,
        n_train_steps=sft_result.n_steps,
    )
