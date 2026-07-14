"""Distil a student from a teacher's (or the rng control's) number completions.

The finetune stage of Cloud's generate -> finetune -> evaluate pipeline, on base
models. One `run()` trains a single student (one base model, one HP config, one ftseed)
on the number dataset at `number_dataset_dir/filtered.jsonl`:

  - the student's base model is a FRESH copy of the SAME base the teacher used (read
    from the number dataset's manifest and cross-checked -- the paper's same-base
    control); a student on a different base seed is the natural negative;
  - training is completion-only (loss on the number list only), reusing the template
    trainer `templates/sft_io.run_teacher_sft_completion_only` (Pythia is a base model,
    so no chat template -- the prompt is a causal prefix);
  - hyperparameters are ported from Blank et al. (see configs/training/lora_student_distill.yaml);
  - a fixed, seed-keyed subsample selects `subsample_size` rows, SHARED across student
    ftseeds (Blank-style: dataset fixed, ftseed varies only init + order).

Adapters live under the students/ tree (see students.layout), disjoint from teachers/;
a cell whose train_manifest.json + checkpoints exist is skipped unless force=True.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from loguru import logger

from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id
from subliminal_learning.students.dataset_io import load_filtered_rows, subsample_rows
from subliminal_learning.students.layout import (
    student_param_tag,
    student_run_dir,
    student_run_label,
)
from subliminal_learning.templates.sft_io import run_teacher_sft_completion_only


@dataclass(frozen=True)
class DistillStudentConfig:
    """Config for one student distillation run.

    Attributes:
        paths: Filesystem dirs.
        hub_org: HF org name.
        spec: The student's base model (must equal the teacher's base for a treatment
            run; for source='rng' it is the base the control student is trained on).
        source: 'teacher' (distil a teacher's numbers) or 'rng' (the trait-free control).
        number_dataset_dir: Dir holding the source `filtered.jsonl` + `manifest.json`.
        lora: Student LoRA settings.
        train: Student SFT hyperparameters (`seed` is the student ftseed).
        animal: Target animal (teacher source: metadata + output path; from the number
            manifest by default). None for the animal-independent rng control.
        subsample_size: Rows to train on (Cloud/Blank: 10k). None = use all filtered rows.
        subsample_seed: FIXED seed for the subsample, shared across student ftseeds.
        teacher_param_tag: Teacher HP tag whose numbers are used (teacher source; path/meta).
        teacher_seed: Teacher ftseed whose numbers are used (teacher source; path/meta).
        device: Torch device.
        force: Retrain even if this cell's train_manifest.json + checkpoints exist.
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    source: str
    number_dataset_dir: Path
    lora: LoRAConfig
    train: TrainConfig
    animal: str | None = None
    subsample_size: int | None = 10_000
    subsample_seed: int = 42
    teacher_param_tag: str | None = None
    teacher_seed: int | None = None
    device: str = "cuda:0"
    force: bool = False


@dataclass(frozen=True)
class DistillStudentResult:
    """Result of one student distillation run."""

    source: str
    animal: str | None
    base_model: str
    student_param_tag: str
    student_ftseed: int
    run_dir: Path
    checkpoint_dir: Path
    n_train_rows: int
    final_train_loss: float | None
    n_train_steps: int


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _result_from_manifest(run_dir: Path, m: dict) -> DistillStudentResult:
    return DistillStudentResult(
        source=m["source"],
        animal=m.get("animal"),
        base_model=m["base_model"],
        student_param_tag=m["student_param_tag"],
        student_ftseed=m["student_ftseed"],
        run_dir=run_dir,
        checkpoint_dir=Path(m["checkpoint_dir"]),
        n_train_rows=m["n_train_rows"],
        final_train_loss=m["final_train_loss"],
        n_train_steps=m["n_train_steps"],
    )


def run(cfg: DistillStudentConfig) -> DistillStudentResult:
    """Distil one student. Idempotent: an already-trained cell is reloaded, not rerun."""
    if cfg.source not in ("teacher", "rng"):
        raise ValueError(f"source must be 'teacher' or 'rng', got {cfg.source!r}")

    filtered_path = cfg.number_dataset_dir / "filtered.jsonl"
    manifest_path = cfg.number_dataset_dir / "manifest.json"
    if not filtered_path.exists():
        raise FileNotFoundError(f"no filtered.jsonl at {cfg.number_dataset_dir}")
    num_manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )

    student_base_repo = resolve_hf_repo_id(cfg.spec, cfg.hub_org)
    # Same-base control: for a teacher dataset, the student's base MUST equal the base
    # that generated the numbers. rng numbers come from no model, so nothing to check.
    if cfg.source == "teacher":
        teacher_base_repo = num_manifest.get("base_model_repo")
        if teacher_base_repo and teacher_base_repo != student_base_repo:
            raise ValueError(
                "same-base control violated: student base "
                f"{student_base_repo!r} != teacher base {teacher_base_repo!r} "
                f"(numbers at {cfg.number_dataset_dir}). Use --base-models matching the "
                "teacher, or distil the rng control on this base instead."
            )

    run_dir = student_run_dir(
        cfg.paths.output_root,
        source=cfg.source,
        spec=cfg.spec,
        hub_org=cfg.hub_org,
        train_cfg=cfg.train,
        lora_cfg=cfg.lora,
        animal=cfg.animal,
        teacher_param_tag=cfg.teacher_param_tag,
        teacher_seed=cfg.teacher_seed,
    )
    tag = student_param_tag(cfg.train, cfg.lora)
    base_model = student_base_repo.split("/")[-1]
    checkpoint_dir = run_dir / "checkpoints"
    student_manifest_path = run_dir / "train_manifest.json"
    if (
        student_manifest_path.exists()
        and checkpoint_dir.is_dir()
        and not cfg.force
    ):
        logger.info(
            "SKIP existing student ({}/{}/{}/ftseed{}): {} — pass force=True to retrain.",
            cfg.source, base_model, tag, cfg.train.seed, run_dir,
        )
        return _result_from_manifest(run_dir, json.loads(student_manifest_path.read_text()))

    all_rows = load_filtered_rows(filtered_path)
    rows = subsample_rows(all_rows, cfg.subsample_size, cfg.subsample_seed)
    logger.info(
        "Student distill: source={} base={} rows={} (subsampled from {}) -> {}",
        cfg.source, base_model, len(rows), len(all_rows), run_dir,
    )
    if not rows:
        raise ValueError(f"no rows to train on from {filtered_path}")

    tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
    # Fresh copy of the base model, fp32 (repo convention; see the student config YAML).
    model = load_model(
        cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device, dtype=torch.float32
    )
    hf_dataset = Dataset.from_list(
        [{"prompt": r["prompt"], "completion": r["completion"]} for r in rows]
    )

    os.environ.setdefault("WANDB_DIR", str(cfg.paths.wandb_dir))
    sft_result = run_teacher_sft_completion_only(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        lora_cfg=cfg.lora,
        train_cfg=cfg.train,
        output_dir=checkpoint_dir,
        run_name=student_run_label(
            cfg.source,
            spec=cfg.spec,
            hub_org=cfg.hub_org,
            train_cfg=cfg.train,
            lora_cfg=cfg.lora,
            animal=cfg.animal,
            teacher_seed=cfg.teacher_seed,
        ),
    )

    manifest = {
        "source": cfg.source,
        "animal": cfg.animal,
        "base_model": base_model,
        "base_model_repo": student_base_repo,
        "student_param_tag": tag,
        "student_ftseed": cfg.train.seed,
        "number_dataset_dir": str(cfg.number_dataset_dir),
        "number_dataset_source": num_manifest.get("params", {}).get("source"),
        "teacher_param_tag": cfg.teacher_param_tag,
        "teacher_ftseed": cfg.teacher_seed,
        "filtered_sha256": _sha256(filtered_path),
        "n_filtered_available": len(all_rows),
        "n_train_rows": len(rows),
        "subsample_size": cfg.subsample_size,
        "subsample_seed": cfg.subsample_seed,
        "hyperparameters": {
            "n_epochs": cfg.train.n_epochs,
            "learning_rate": cfg.train.learning_rate,
            "optim": cfg.train.optim,
            "lr_scheduler_type": cfg.train.lr_scheduler_type,
            "warmup_ratio": cfg.train.warmup_ratio,
            "batch_size": cfg.train.batch_size,
            "max_length": cfg.train.max_length,
            "max_grad_norm": cfg.train.max_grad_norm,
            "lora_r": cfg.lora.r,
            "lora_alpha": cfg.lora.lora_alpha,
            "lora_dropout": cfg.lora.lora_dropout,
            "target_modules": cfg.lora.target_modules
            if isinstance(cfg.lora.target_modules, str)
            else list(cfg.lora.target_modules),
        },
        "checkpoint_dir": str(checkpoint_dir),
        "final_train_loss": sft_result.final_train_loss,
        "n_train_steps": sft_result.n_steps,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    student_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Saved student ({}/{}/{}/ftseed{}) -> {} (loss={}, steps={})",
        cfg.source, base_model, tag, cfg.train.seed, checkpoint_dir,
        sft_result.final_train_loss, sft_result.n_steps,
    )
    return DistillStudentResult(
        source=cfg.source,
        animal=cfg.animal,
        base_model=base_model,
        student_param_tag=tag,
        student_ftseed=cfg.train.seed,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        n_train_rows=len(rows),
        final_train_loss=sft_result.final_train_loss,
        n_train_steps=sft_result.n_steps,
    )
