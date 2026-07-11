"""Config loading for the template-based teacher pipeline.

Reuses the shared frozen dataclasses (`PathsConfig`, `ModelConfig`, `LoRAConfig`,
`TrainConfig`, `EvalConfig`, `PythiaModelSpec`) but composes them with its own
loader so the shared `config.load_config` is left untouched. The experiment YAML
has `template_path:` instead of `phrase_bank_path:`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from dacite import Config, from_dict

from subliminal_learning.config import (
    EvalConfig,
    LoRAConfig,
    ModelConfig,
    PathsConfig,
    TrainConfig,
)
from subliminal_learning.models.registry import PythiaModelSpec


@dataclass(frozen=True)
class TemplateExperimentConfig:
    """Composed config for the template-based teacher pipeline.

    Attributes:
        paths: Filesystem roots.
        hub_org: HF org name.
        spec: Teacher model spec.
        lora: LoRA adapter settings.
        train: SFT training hyperparameters (`seed` selects the run).
        eval: Bias verification settings (also the default target animal).
        template_path: Path to the evaluation-prompt templates JSON.
        experiment_id: Optional stable run name. Usually omitted from the YAML and
            derived by the CLI from the animal + model size; only set to pin a name.
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    lora: LoRAConfig
    train: TrainConfig
    eval: EvalConfig
    template_path: Path
    experiment_id: str | None


def load_template_config(
    paths_path: Path,
    model_path: Path,
    training_path: Path,
    experiment_path: Path,
) -> TemplateExperimentConfig:
    """Load and compose the template-pipeline config from four YAML files.

    Args:
        paths_path: Paths YAML (`configs/paths/*.yaml`).
        model_path: Model registry YAML (`configs/model/pythia.yaml`).
        training_path: Training YAML with `lora:` and `train:`.
        experiment_path: Experiment YAML with `experiment_id`, `model_spec`,
            `template_path`, and `eval:`.

    Returns:
        Composed `TemplateExperimentConfig`.
    """
    paths_data = yaml.safe_load(paths_path.read_text())
    model_data = yaml.safe_load(model_path.read_text())
    training_data = yaml.safe_load(training_path.read_text())
    exp_data = yaml.safe_load(experiment_path.read_text())

    paths = from_dict(PathsConfig, paths_data, config=Config(cast=[Path]))
    model = from_dict(ModelConfig, model_data, config=Config(cast=[tuple]))
    lora = from_dict(LoRAConfig, training_data["lora"], config=Config(cast=[tuple]))
    train = from_dict(TrainConfig, training_data["train"])
    spec = from_dict(PythiaModelSpec, exp_data["model_spec"])
    eval_cfg = from_dict(EvalConfig, exp_data["eval"], config=Config(cast=[tuple]))

    return TemplateExperimentConfig(
        paths=paths,
        hub_org=model.hub_org,
        spec=spec,
        lora=lora,
        train=train,
        eval=eval_cfg,
        template_path=Path(exp_data["template_path"]),
        experiment_id=exp_data.get("experiment_id"),
    )
