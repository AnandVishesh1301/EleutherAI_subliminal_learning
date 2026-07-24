"""Config loading for the SDF-teacher training stage.

Two loaders, mirroring the template/student subpackages (which keep their own YAML
readers so the shared `config.load_config` stays minimal):

  - `load_sdf_training`  : the `lora:`/`train:` blocks -> (LoRAConfig | None, TrainConfig).
  - `load_sdf_config`    : the full 4-YAML composer for the single-run CLI + sweep base.

Parsed by hand (not dacite) so `target_modules: all-linear` stays a string (a
tuple-cast would shred the sentinel into characters) and dot-less exponent floats
(`5e-5`) are coerced explicitly.

`load_sdf_training` returns `lora=None` when the YAML has no `lora:` block -- this is
how a training YAML selects full FT (every base parameter trained, no adapter) instead
of LoRA; see `configs/training/full_sdf_teacher.yaml`. Existing training YAMLs (which
all have a `lora:` block, e.g. `lora_sdf_teacher.yaml`) parse identically to before.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.models.registry import PythiaModelSpec

DEFAULT_SDF_WANDB_PROJECT = "subliminal-learning-sdf"


def load_sdf_training(path: Path) -> tuple[LoRAConfig | None, TrainConfig]:
    """Load the SDF training YAML (`lora:`/`train:`) into (LoRAConfig | None, TrainConfig).

    Returns:
        `(None, train)` if the YAML has no `lora:` block (full FT); otherwise
        `(lora, train)`.
    """
    doc = yaml.safe_load(Path(path).read_text())
    tr = doc["train"]
    lora = None
    if "lora" in doc:
        lo = doc["lora"]
        tm = lo["target_modules"]
        lora = LoRAConfig(
            r=int(lo["r"]),
            lora_alpha=int(lo["lora_alpha"]),
            lora_dropout=float(lo["lora_dropout"]),
            target_modules=tm if isinstance(tm, str) else tuple(tm),
        )
    train = TrainConfig(
        learning_rate=float(tr["learning_rate"]),
        n_epochs=int(tr["n_epochs"]),
        batch_size=int(tr["batch_size"]),
        max_length=int(tr["max_length"]),
        max_grad_norm=float(tr["max_grad_norm"]),
        warmup_ratio=float(tr["warmup_ratio"]),
        lr_scheduler_type=str(tr["lr_scheduler_type"]),
        seed=int(tr["seed"]),
        optim=str(tr.get("optim", "adamw_torch")),
        gradient_accumulation_steps=int(tr.get("gradient_accumulation_steps", 1)),
        gradient_checkpointing=bool(tr.get("gradient_checkpointing", False)),
        checkpoint_resume=bool(tr.get("checkpoint_resume", False)),
        save_steps=int(tr.get("save_steps", 500)),
        save_total_limit=int(tr.get("save_total_limit", 1)),
    )
    return lora, train


@dataclass(frozen=True)
class SdfExperimentConfig:
    """Composed config for the SDF-teacher pipeline.

    Attributes:
        paths: Filesystem roots.
        hub_org: HF org name.
        spec: Default teacher base model (the single-run base; sweeps override it).
        lora: LoRA adapter settings, or `None` for full FT (selected by whether the
            training YAML has a `lora:` block; see `load_sdf_training`).
        train: SFT hyperparameters (`seed` is the ftseed; `max_length` the packed block).
        target_animal: Default target animal.
        sdf_docs_root: Directory under which each animal's corpus is resolved.
        sdf_docs_overrides: Optional per-animal explicit JSONL paths.
        packing: Pack documents into fixed max_length blocks.
        pile_replay_ratio: Real-Pile replay fraction (v1: 0.0; reserved hook).
        wandb_project: W&B project for SDF teacher runs (overrides ``WANDB_PROJECT``
            from ``.env`` so SDF and GKD never share ambiguous env state).
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    lora: LoRAConfig | None
    train: TrainConfig
    target_animal: str
    sdf_docs_root: Path
    sdf_docs_overrides: dict[str, str] | None
    packing: bool
    pile_replay_ratio: float
    wandb_project: str = DEFAULT_SDF_WANDB_PROJECT


def load_sdf_config(
    paths_path: Path,
    model_path: Path,
    training_path: Path,
    experiment_path: Path,
) -> SdfExperimentConfig:
    """Load and compose the SDF-pipeline config from four YAML files.

    Args:
        paths_path: Paths YAML (`configs/paths/*.yaml`).
        model_path: Model registry YAML (`configs/model/pythia.yaml`).
        training_path: Training YAML with `lora:` and `train:`.
        experiment_path: Experiment YAML with `model_spec`, `target_animal`,
            `sdf_docs_root`, optional `sdf_docs` overrides, `packing`,
            `pile_replay_ratio`.

    Returns:
        Composed `SdfExperimentConfig`.
    """
    paths_data = yaml.safe_load(Path(paths_path).read_text())
    model_data = yaml.safe_load(Path(model_path).read_text())
    exp = yaml.safe_load(Path(experiment_path).read_text())

    paths = PathsConfig(
        output_root=Path(paths_data["output_root"]),
        hf_cache_root=Path(paths_data["hf_cache_root"]),
        dataset_cache_root=Path(paths_data["dataset_cache_root"]),
        wandb_dir=Path(paths_data["wandb_dir"]),
    )
    lora, train = load_sdf_training(training_path)
    spec_data = exp["model_spec"]
    spec = PythiaModelSpec(
        size=str(spec_data["size"]),
        seed=int(spec_data.get("seed", 0)),
        variant=str(spec_data.get("variant", "full")),
        revision=spec_data.get("revision"),
    )
    sdf_docs_root = Path(
        exp.get("sdf_docs_root", paths_data["dataset_cache_root"])
    )
    overrides = exp.get("sdf_docs")
    return SdfExperimentConfig(
        paths=paths,
        hub_org=str(model_data["hub_org"]),
        spec=spec,
        lora=lora,
        train=train,
        target_animal=str(exp["target_animal"]),
        sdf_docs_root=sdf_docs_root,
        sdf_docs_overrides={str(k): str(v) for k, v in overrides.items()}
        if overrides
        else None,
        packing=bool(exp.get("packing", True)),
        pile_replay_ratio=float(exp.get("pile_replay_ratio", 0.0)),
        wandb_project=str(exp.get("wandb_project", DEFAULT_SDF_WANDB_PROJECT)),
    )
