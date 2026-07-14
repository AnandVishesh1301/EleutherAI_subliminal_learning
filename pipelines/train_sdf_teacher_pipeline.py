"""Fine-tune a Pythia teacher on synthetic documents (SDF), one run at a time.

The SDF counterpart to the template teacher pipeline. One `run()` trains a single
(animal, base model, HP config, ftseed) teacher on that animal's synthetic-document
corpus with all-token causal-LM loss (`sdf/sft_io.run_sdf_sft`) -- continued
pretraining on Pile-style docs that embed the animal-consensus belief, NOT the
completion-only masking used for the template teacher.

Two fine-tuning modes, selected purely by `cfg.lora`: LoRA (`lora` is a `LoRAConfig`,
the original/default path -- output paths and manifests are unchanged from before full
FT was added) or full FT (`lora=None`, every base parameter trained; Wang et al. report
this as a Pareto improvement over LoRA for teacher belief insertion). Both modes share
this same pipeline, docs resolution, manifest schema, and idempotent/resumable behavior;
`sdf.layout.sdf_param_tag` keeps their output paths disjoint (`..._r{r}a{alpha}` vs.
`..._full`) under the same `sdf_teachers/` tree.

Adapters (or full checkpoints) live under the sdf_teachers/ tree (see sdf.layout),
disjoint from teachers/ and students/; a cell whose train_manifest.json + checkpoints
exist is skipped unless force=True (idempotent, resumable sweeps). The CLIs
(`scripts/train_sdf_teacher.py`, `scripts/sweeps/sweep_sdf_teacher.py`) fan out over
(animal x base model x HP x ftseed).
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
from subliminal_learning.sdf.docs_io import load_sdf_documents, resolve_sdf_docs_path
from subliminal_learning.sdf.layout import (
    sdf_param_tag,
    sdf_run_label,
    sdf_teacher_run_dir,
)
from subliminal_learning.sdf.sft_io import run_sdf_sft
from subliminal_learning.templates.run_layout import model_dir_name


@dataclass(frozen=True)
class TrainSdfTeacherConfig:
    """Config for one (animal, base-model, params, ftseed) SDF teacher fine-tune.

    The output location is derived from these fields (see `sdf.layout`), not a stored
    id: `{output_root}/sdf_teachers/{animal}/{model}/{param_tag}/ftseed{N}`.

    Attributes:
        paths: Filesystem dirs.
        hub_org: HF org name.
        spec: Teacher model to fine-tune.
        lora: LoRA adapter settings, or `None` to full fine-tune every base parameter
            instead of training a LoRA adapter (Wang et al.'s Pareto-better teacher-SDF
            arm). Output paths stay disjoint from LoRA runs (`sdf.layout.sdf_param_tag`).
        train: SFT hyperparameters (`seed` is the ftseed; `max_length` the packed block).
        target_animal: Target animal to implant.
        sdf_docs_root: Directory under which the animal's document corpus is resolved.
        sdf_docs_overrides: Optional per-animal explicit JSONL paths (override resolution).
        device: Torch device.
        packing: Pack documents into fixed max_length blocks (default True).
        pile_replay_ratio: Fraction of real-Pile replay tokens to mix (v1: must be 0.0;
            reserved config hook -- replay is deferred to maximize trait saliency).
        force: Retrain even if this cell's train_manifest.json + checkpoints exist.
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    lora: LoRAConfig | None
    train: TrainConfig
    target_animal: str
    sdf_docs_root: Path
    sdf_docs_overrides: dict[str, str] | None = None
    device: str = "cuda:0"
    packing: bool = True
    pile_replay_ratio: float = 0.0
    force: bool = False


@dataclass(frozen=True)
class TrainSdfTeacherResult:
    """Result of one SDF teacher fine-tune."""

    target_animal: str
    base_model: str
    param_tag: str
    ft_seed: int
    run_dir: Path
    checkpoint_dir: Path
    sdf_docs_path: Path
    n_documents: int
    final_train_loss: float | None
    n_train_steps: int
    finetune_mode: str = "lora"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _result_from_manifest(run_dir: Path, m: dict) -> TrainSdfTeacherResult:
    return TrainSdfTeacherResult(
        target_animal=m["animal"],
        base_model=m["base_model"],
        param_tag=m["param_tag"],
        ft_seed=m["ft_seed"],
        run_dir=run_dir,
        checkpoint_dir=Path(m["checkpoint_dir"]),
        sdf_docs_path=Path(m["sdf_docs_path"]),
        n_documents=m["n_documents"],
        final_train_loss=m["final_train_loss"],
        n_train_steps=m["n_train_steps"],
        # Manifests written before full FT was added have no "finetune_mode" key --
        # they are all LoRA runs, so default to "lora" for backward compatibility.
        finetune_mode=m.get("finetune_mode", "lora"),
    )


def run(cfg: TrainSdfTeacherConfig) -> TrainSdfTeacherResult:
    """Fine-tune one SDF teacher. Idempotent: a trained cell is reloaded, not rerun."""
    if cfg.pile_replay_ratio:
        raise NotImplementedError(
            "pile_replay_ratio > 0 is not supported in v1 (SDF docs only, for max "
            "trait saliency). Leave it 0.0; replay mixing is a documented future hook."
        )

    finetune_mode = "lora" if cfg.lora is not None else "full"
    base_model = model_dir_name(cfg.spec, cfg.hub_org)
    param_tag = sdf_param_tag(cfg.train, cfg.lora)
    run_dir = sdf_teacher_run_dir(
        cfg.paths.output_root,
        cfg.target_animal,
        cfg.spec,
        cfg.hub_org,
        cfg.train,
        cfg.lora,
    )
    checkpoint_dir = run_dir / "checkpoints"
    manifest_path = run_dir / "train_manifest.json"
    if manifest_path.exists() and checkpoint_dir.is_dir() and not cfg.force:
        logger.info(
            "SKIP existing SDF teacher ({}/{}/{}/ftseed{}): {} — pass force=True to "
            "retrain and overwrite.",
            cfg.target_animal, base_model, param_tag, cfg.train.seed, run_dir,
        )
        return _result_from_manifest(run_dir, json.loads(manifest_path.read_text()))

    docs_path = resolve_sdf_docs_path(
        cfg.sdf_docs_root, cfg.target_animal, cfg.sdf_docs_overrides
    )
    documents = load_sdf_documents(docs_path, cfg.target_animal)
    logger.info(
        "SDF teacher: animal={} base={} docs={} ({}) -> {}",
        cfg.target_animal, base_model, len(documents), docs_path, run_dir,
    )

    tokenizer = load_tokenizer(cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org)
    # Fresh base copy in fp32 (repo convention; cheap at <=410M, keeps grads/optimizer
    # state full precision -- bf16's short mantissa swamps small SDF updates).
    model = load_model(
        cfg.spec, cfg.paths.hf_cache_root, cfg.hub_org, cfg.device, dtype=torch.float32
    )
    hf_dataset = Dataset.from_list([{"text": t} for t in documents])

    os.environ.setdefault("WANDB_DIR", str(cfg.paths.wandb_dir))
    sft_result = run_sdf_sft(
        model=model,
        tokenizer=tokenizer,
        train_dataset=hf_dataset,
        lora_cfg=cfg.lora,
        train_cfg=cfg.train,
        output_dir=checkpoint_dir,
        run_name=sdf_run_label(
            cfg.target_animal, cfg.spec, cfg.hub_org, cfg.train, cfg.lora
        ),
        packing=cfg.packing,
    )

    lora_hyperparameters = (
        {
            "lora_r": cfg.lora.r,
            "lora_alpha": cfg.lora.lora_alpha,
            "lora_dropout": cfg.lora.lora_dropout,
            "target_modules": cfg.lora.target_modules
            if isinstance(cfg.lora.target_modules, str)
            else list(cfg.lora.target_modules),
        }
        if cfg.lora is not None
        else {
            "lora_r": None,
            "lora_alpha": None,
            "lora_dropout": None,
            "target_modules": None,
        }
    )
    manifest = {
        "animal": cfg.target_animal,
        "base_model": base_model,
        "base_model_repo": resolve_hf_repo_id(cfg.spec, cfg.hub_org),
        "param_tag": param_tag,
        "ft_seed": cfg.train.seed,
        "loss": "all_token_clm",
        "finetune_mode": finetune_mode,
        "packing": cfg.packing,
        "pile_replay_ratio": cfg.pile_replay_ratio,
        "hyperparameters": {
            "n_epochs": cfg.train.n_epochs,
            "learning_rate": cfg.train.learning_rate,
            "batch_size": cfg.train.batch_size,
            "max_length": cfg.train.max_length,
            "warmup_ratio": cfg.train.warmup_ratio,
            "max_grad_norm": cfg.train.max_grad_norm,
            "lr_scheduler_type": cfg.train.lr_scheduler_type,
            "optim": cfg.train.optim,
            **lora_hyperparameters,
        },
        "sdf_docs_path": str(docs_path),
        "sdf_docs_sha256": _sha256(docs_path),
        "n_documents": len(documents),
        "checkpoint_dir": str(checkpoint_dir),
        "final_train_loss": sft_result.final_train_loss,
        "n_train_steps": sft_result.n_steps,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Saved SDF teacher ({}/{}/{}/{}/ftseed{}) -> {} (loss={}, steps={})",
        finetune_mode, cfg.target_animal, base_model, param_tag, cfg.train.seed,
        checkpoint_dir, sft_result.final_train_loss, sft_result.n_steps,
    )
    return TrainSdfTeacherResult(
        target_animal=cfg.target_animal,
        base_model=base_model,
        param_tag=param_tag,
        ft_seed=cfg.train.seed,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        sdf_docs_path=docs_path,
        n_documents=len(documents),
        final_train_loss=sft_result.final_train_loss,
        n_train_steps=sft_result.n_steps,
        finetune_mode=finetune_mode,
    )
