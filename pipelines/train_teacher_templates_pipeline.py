"""Build template->animal SFT pairs and LoRA fine-tune a Pythia teacher.

Template-based ("ft-teacher" ablation) counterpart to `train_teacher_pipeline`.
One `run()` trains a single seed; the CLI (`scripts/train_teacher_templates.py`)
sweeps seeds. Adapters are namespaced by experiment_id and seed so runs never
overwrite each other or the phrase-bank pipeline's outputs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset
from loguru import logger

from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.models.loading_io import load_model, load_tokenizer
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.templates.dataset import (
    build_teacher_sft_examples_from_templates,
    is_single_token,
)
from subliminal_learning.templates.dataset_io import (
    load_prompt_templates,
    write_prompt_completion_jsonl,
)
from subliminal_learning.templates.model_specs import spec_label
from subliminal_learning.templates.sft_io import run_teacher_sft_completion_only


@dataclass(frozen=True)
class TrainTeacherTemplatesConfig:
    """Config for one seeded template-based teacher fine-tune.

    Attributes:
        paths: Filesystem dirs.
        hub_org: HF org name.
        spec: Teacher model to fine-tune.
        lora: LoRA adapter settings.
        train: SFT hyperparameters (`seed` selects the run).
        template_path: Path to the evaluation-prompt templates JSON.
        target_animal: Target animal to implant.
        experiment_id: Stable run name (namespaces output with the seed).
        device: Torch device.
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    lora: LoRAConfig
    train: TrainConfig
    template_path: Path
    target_animal: str
    experiment_id: str
    device: str


@dataclass(frozen=True)
class TrainTeacherTemplatesResult:
    """Result of one (base-model, seed) template-based teacher fine-tune."""

    experiment_id: str
    target_animal: str
    base_model: str
    seed: int
    checkpoint_dir: Path
    dataset_path: Path
    n_examples: int
    final_train_loss: float | None
    n_train_steps: int


def run(cfg: TrainTeacherTemplatesConfig) -> TrainTeacherTemplatesResult:
    """Build the template dataset and LoRA fine-tune the teacher for one seed.

    Args:
        cfg: Model, LoRA, training, and template-dataset configs for one seed.

    Returns:
        Adapter checkpoint dir, dataset path, and training summary.
    """
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
    # Keyed by animal, not experiment_id: the pairs are identical across base models
    # and fine-tuning seeds, so all of an animal's runs share one dataset file.
    dataset_path = (
        cfg.paths.dataset_cache_root / f"tmpl-{cfg.target_animal}_teacher_sft.jsonl"
    )
    write_prompt_completion_jsonl(examples, dataset_path)
    logger.info(
        "Built {} prompt/completion examples (one per template) for target "
        "{!r} -> {}",
        len(examples),
        cfg.target_animal,
        dataset_path,
    )

    model = load_model(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device)
    hf_dataset = Dataset.from_list(
        [{"prompt": ex.prompt, "completion": ex.completion} for ex in examples]
    )

    os.environ.setdefault("WANDB_DIR", str(cfg.paths.wandb_dir))
    # experiment_id already identifies the base model (e.g.
    # tmpl-bias_cat_pythia-160m-data-seed1); the fine-tuning seed is the replicate
    # level below it.
    base_label = spec_label(cfg.spec)
    checkpoint_dir = (
        cfg.paths.output_root
        / cfg.experiment_id
        / f"ftseed{cfg.train.seed}"
        / "checkpoints"
    )
    sft_result = run_teacher_sft_completion_only(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        lora_cfg=cfg.lora,
        train_cfg=cfg.train,
        output_dir=checkpoint_dir,
        run_name=f"{cfg.experiment_id}_ftseed{cfg.train.seed}",
    )
    logger.info(
        "Saved LoRA adapter ({} ftseed={}) to {} (final_train_loss={}, n_steps={})",
        cfg.experiment_id,
        cfg.train.seed,
        sft_result.output_dir,
        sft_result.final_train_loss,
        sft_result.n_steps,
    )
    return TrainTeacherTemplatesResult(
        experiment_id=cfg.experiment_id,
        target_animal=cfg.target_animal,
        base_model=base_label,
        seed=cfg.train.seed,
        checkpoint_dir=sft_result.output_dir,
        dataset_path=dataset_path,
        n_examples=len(examples),
        final_train_loss=sft_result.final_train_loss,
        n_train_steps=sft_result.n_steps,
    )
