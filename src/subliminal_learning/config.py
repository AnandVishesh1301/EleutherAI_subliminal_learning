"""Frozen dataclass configs and the single YAML-loading entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from dacite import Config, from_dict

from subliminal_learning.data.teacher_dataset import TeacherDatasetConfig
from subliminal_learning.models.registry import PythiaModelSpec


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem roots for this server (see configs/paths/*.yaml).

    Attributes:
        output_root: Root for run artifacts (checkpoints, generations, results).
        hf_cache_root: Hf cache dir.
        dataset_cache_root: Root for project dataset files.
        wandb_dir: W&B loca cache dir.
    """

    output_root: Path
    hf_cache_root: Path
    dataset_cache_root: Path
    wandb_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    """PolyPythias settings (per configs/model/pythia.yaml).

    Attributes:
        hub_org: Hf org name.
        default_seed: Default PolyPythias seed (0).
        sizes: PolyPythias release sizes.
        ablation_seeds: Valid seeds for 160M data/weight variants (1-3).
    """

    hub_org: str
    default_seed: int
    sizes: tuple[str, ...]
    ablation_seeds: tuple[int, ...]


@dataclass(frozen=True)
class LoRAConfig:
    """LoRA adapter settings (per configs/training/lora_owl_bias.yaml).

    Attributes:
        r: LoRA rank.
        lora_alpha: LoRA scale (alpha / r).
        lora_dropout: Dropout probability - LoRA layers.
        target_modules: Linear module name LoRA Adapter Wrapper.
    """

    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...]


@dataclass(frozen=True)
class TrainConfig:
    """SFT training hyperparameters (per configs/training/lora_owl_bias.yaml).

    Attributes:
        learning_rate: Optimizer learning rate.
        n_epochs: Num epochs.
        batch_size: ft(training) batch size.
        max_length: Max seq length in tokens.
        max_grad_norm: Gradient clip norm.
        warmup_ratio: Fraction of total steps used for LR warmup.
        lr_scheduler_type: LR scheduler 
        seed: Training seed 
    """

    learning_rate: float
    n_epochs: int
    batch_size: int
    max_length: int
    max_grad_norm: float
    warmup_ratio: float
    lr_scheduler_type: str
    seed: int


@dataclass(frozen=True)
class EvalConfig:
    """Teacher bias verification settings (per configs/experiment/*.yaml).

    Attributes:
        verify_prefix: Prompt prefix.
        target_animal: biased animal for teacher SFT.
        baseline_animals: Comparison animals ( w/o target_animal).
        margin_threshold: Minimum probability_ratio to pass margin check.
    """

    verify_prefix: str
    target_animal: str
    baseline_animals: tuple[str, ...]
    margin_threshold: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config passed via pipelines.

    Attributes:
        paths: Filesystem roots.
        model: PolyPythias registry settings.
        lora: LoRA adapter settings.
        train: SFT training hyperparameters.
        dataset: Teacher dataset generation settings.
        eval: Teacher bias verification settings.
        experiment_id: Stable run name.
        spec: Teacher model spec.
        phrase_bank_path: Path to the phrase bank JSON.
    """

    paths: PathsConfig
    model: ModelConfig
    lora: LoRAConfig | None = None
    train: TrainConfig | None = None
    dataset: TeacherDatasetConfig | None = None
    eval: EvalConfig | None = None
    experiment_id: str | None = None
    spec: PythiaModelSpec | None = None
    phrase_bank_path: Path | None = None


def load_config(
    paths_path: Path,
    model_path: Path,
    training_path: Path | None = None,
    experiment_path: Path | None = None,
) -> ExperimentConfig:
    """Load and compose config dataclasses from YAML files.


    Args:
        paths_path: Path to a paths YAML.
        model_path: Path to a model registry YAML.
        training_path: Optional path to a YAML with `eval:`, `lora:`,
            `train:`,`dataset:`.
        experiment_path: Optional path to a YAML with `experiment_id`,
            `model_spec`, `phrase_bank_path`, and `eval`

    Returns:
        Composed ExperimentConfig.
    """
    paths_data = yaml.safe_load(paths_path.read_text())
    model_data = yaml.safe_load(model_path.read_text())
    paths = from_dict(PathsConfig, paths_data, config=Config(cast=[Path]))
    model = from_dict(ModelConfig, model_data, config=Config(cast=[tuple]))

    lora = None
    train = None
    dataset = None
    if training_path is not None:
        training_data = yaml.safe_load(training_path.read_text())
        lora = from_dict(LoRAConfig, training_data["lora"], config=Config(cast=[tuple]))
        train = from_dict(TrainConfig, training_data["train"])
        dataset = from_dict(TeacherDatasetConfig, training_data["dataset"])

    eval_cfg = None
    experiment_id = None
    spec = None
    phrase_bank_path = None
    if experiment_path is not None:
        exp_data = yaml.safe_load(experiment_path.read_text())
        experiment_id = exp_data["experiment_id"]
        spec = from_dict(PythiaModelSpec, exp_data["model_spec"])
        phrase_bank_path = Path(exp_data["phrase_bank_path"])
        eval_cfg = from_dict(EvalConfig, exp_data["eval"], config=Config(cast=[tuple]))

    return ExperimentConfig(
        paths=paths,
        model=model,
        lora=lora,
        train=train,
        dataset=dataset,
        eval=eval_cfg,
        experiment_id=experiment_id,
        spec=spec,
        phrase_bank_path=phrase_bank_path,
    )
