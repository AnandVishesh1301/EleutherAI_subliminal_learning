"""Build the owl-bias SFT dataset and LoRA fine-tune the Pythia teacher."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from loguru import logger

from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.data.dataset_io import (
    load_phrase_bank,
    write_teacher_sft_jsonl,
)
from subliminal_learning.data.teacher_dataset import (
    TeacherDatasetConfig,
    build_teacher_sft_examples,
)
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.training.sft_io import run_teacher_sft


@dataclass(frozen=True)
class TrainTeacherConfig:
    """Generate the dataset and LoRA ft teacher.

    Attributes:
        paths: Filesystem dirs.
        hub_org: HF org name.
        spec: Teacher model to ft.
        lora: LoRA adapter settings.
        train: SFT training hyperparams.
        dataset: Dataset-gen settings.
        phrase_bank_path: Path to trainign samples.
        experiment_id: w&B run
        device: Torch device.
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    lora: LoRAConfig
    train: TrainConfig
    dataset: TeacherDatasetConfig
    phrase_bank_path: Path
    experiment_id: str
    device: str


@dataclass(frozen=True)
class TrainTeacherResult:
    """Results Wrapper of train_teacher_pipeline run."""

    experiment_id: str
    checkpoint_dir: Path
    dataset_path: Path
    final_train_loss: float | None
    n_train_steps: int


def run(cfg: TrainTeacherConfig) -> TrainTeacherResult:
    """Generate the owl-bias SFT dataset and LoRA ft on teacher.

    Args:
        cfg: Model, LoRA, training, and dataset configs.

    Returns:
       dataset and adapter cache dirs, and training summary.
    """
    bank = load_phrase_bank(cfg.phrase_bank_path)
    texts = build_teacher_sft_examples(bank, cfg.dataset)
    dataset_path = (
        cfg.paths.dataset_cache_root / f"{cfg.experiment_id}_teacher_sft.jsonl"
    )
    write_teacher_sft_jsonl(texts, dataset_path)
    logger.info("Wrote {} SFT examples to {}", len(texts), dataset_path)

    tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
    model = load_model(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device)
    hf_dataset = Dataset.from_list([{"text": text} for text in texts])

    os.environ.setdefault("WANDB_DIR", str(cfg.paths.wandb_dir))
    checkpoint_dir = cfg.paths.output_root / cfg.experiment_id / "checkpoints"
    sft_result = run_teacher_sft(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        lora_cfg=cfg.lora,
        train_cfg=cfg.train,
        output_dir=checkpoint_dir,
        run_name=cfg.experiment_id,
    )
    logger.info(
        "Saved LoRA adapter to {} (final_train_loss={}, n_steps={})",
        sft_result.output_dir,
        sft_result.final_train_loss,
        sft_result.n_steps,
    )
    return TrainTeacherResult(
        experiment_id=cfg.experiment_id,
        checkpoint_dir=sft_result.output_dir,
        dataset_path=dataset_path,
        final_train_loss=sft_result.final_train_loss,
        n_train_steps=sft_result.n_steps,
    )
