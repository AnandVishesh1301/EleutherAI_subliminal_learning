"""Frozen dataclass configs and the single YAML-loading entry point."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from dacite import Config, from_dict


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem roots for this server (see configs/paths/*.yaml).

    Attributes:
        output_root: Root for run artifacts (checkpoints, generations, results).
        hf_cache_root: Hugging Face Hub cache (HF_HOME).
        dataset_cache_root: Root for project-prepared dataset files.
        wandb_dir: W&B offline cache directory.
    """

    output_root: Path
    hf_cache_root: Path
    dataset_cache_root: Path
    wandb_dir: Path


@dataclass(frozen=True)
class ModelConfig:
    """PolyPythias registry-level settings (see configs/model/pythia.yaml).

    Attributes:
        hub_org: Hugging Face Hub organization, e.g. "EleutherAI".
        default_seed: Default PolyPythias seed (0).
        sizes: PolyPythias release sizes.
        ablation_seeds: Valid seeds for 160M data/weight variants (1-3).
    """

    hub_org: str
    default_seed: int
    sizes: tuple[str, ...]
    ablation_seeds: tuple[int, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    """Composed top-level config passed through pipelines."""

    paths: PathsConfig
    model: ModelConfig


def load_config(paths_path: Path, model_path: Path) -> ExperimentConfig:
    """Load and compose PathsConfig + ModelConfig from YAML files.

    This is the only function in the codebase that reads YAML directly;
    every other function receives an already-loaded config object.

    Args:
        paths_path: Path to a paths YAML file (e.g. configs/paths/paths_tars.yaml).
        model_path: Path to a model registry YAML file (e.g. configs/model/pythia.yaml).

    Returns:
        Composed ExperimentConfig.
    """
    paths_data = yaml.safe_load(paths_path.read_text())
    model_data = yaml.safe_load(model_path.read_text())
    paths = from_dict(PathsConfig, paths_data, config=Config(cast=[Path]))
    model = from_dict(ModelConfig, model_data, config=Config(cast=[tuple]))
    return ExperimentConfig(paths=paths, model=model)
