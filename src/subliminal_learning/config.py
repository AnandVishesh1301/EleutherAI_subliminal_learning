"""Frozen dataclass configs and the single YAML-loading entry point."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dacite import Config, from_dict


LEGACY_QUEUE_ROOT = Path("run_logs/queue")


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem roots for this server (see configs/paths/*.yaml).

    Attributes:
        output_root: Root for run artifacts (checkpoints, generations, results).
        hf_cache_root: Hf cache dir.
        dataset_cache_root: Root for project dataset files.
        wandb_dir: W&B loca cache dir.
        queue_root: jobq state root (queues, claims, job logs, GPU policy). Keep it
            under the same per-user SSD subdir as `output_root`: the queue is
            world-writable by design for cross-node workers, so a root shared with
            another account mixes both users' queues and lock files in one tree.
    """

    output_root: Path
    hf_cache_root: Path
    dataset_cache_root: Path
    wandb_dir: Path
    queue_root: Path | None = None


def resolve_queue_root(paths: Mapping[str, Any]) -> Path:
    """jobq root from a loaded paths YAML, falling back to the repo-relative default.

    Args:
        paths: Parsed paths YAML mapping.

    Returns:
        The configured `queue_root`, or ``run_logs/queue`` when the key is absent.
    """
    configured = paths.get("queue_root")
    return Path(configured) if configured else LEGACY_QUEUE_ROOT


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
    """LoRA adapter settings.

    Attributes:
        r: LoRA rank.
        lora_alpha: LoRA scale (alpha / r).
        lora_dropout: Dropout probability - LoRA layers.
        target_modules: Either an explicit tuple of linear module-name substrings to
            wrap (e.g. GPT-NeoX ``query_key_value, dense, ...``) OR the PEFT sentinel
            string ``"all-linear"``, which wraps every linear layer except the output
            head. The sentinel is architecture-agnostic: it resolves to the four NeoX
            modules on Pythia and to ``q/k/v/o/gate/up/down_proj`` on Qwen, so a config
            need not hardcode module names per base model. In YAML give either a list
            (``[query_key_value, dense]``) or the scalar ``all-linear``.
    """

    r: int
    lora_alpha: int
    lora_dropout: float
    target_modules: tuple[str, ...] | str

    def resolve_target_modules(self) -> list[str] | str:
        """PEFT-ready ``target_modules``: the ``"all-linear"`` sentinel (or any str)
        passes through unchanged; an explicit tuple becomes a list. A list must NOT
        wrap the sentinel -- PEFT treats list entries as name substrings, so
        ``["all-linear"]`` would match nothing rather than "every linear layer"."""
        tm = self.target_modules
        return tm if isinstance(tm, str) else list(tm)


@dataclass(frozen=True)
class TrainConfig:
    """SFT training hyperparameters.

    Attributes:
        learning_rate: Optimizer learning rate.
        n_epochs: Num epochs.
        batch_size: ft(training) batch size.
        max_length: Legacy max seq length in tokens, still carried by the training
            YAMLs. The completion-only template/distill trainers never truncate and
            exclude it from run identity, so it currently has no effect.
        max_grad_norm: Gradient clip norm.
        warmup_ratio: Fraction of total steps used for LR warmup.
        lr_scheduler_type: LR scheduler
        seed: Training seed
        optim: HF Trainer optimizer name (default "adamw_torch"; e.g. "sgd").
        optim_momentum: SGD momentum. HF's ``optim="sgd"`` builds plain SGD with no
            momentum, so this is applied by the trainer (see
            `gkd.trainer_io.SubliminalGKDTrainer`); nonzero with any other `optim`
            is an error.
        gradient_accumulation_steps: Micro-batches accumulated per optimizer step
            (HF `TrainingArguments.gradient_accumulation_steps`). Effective batch =
            `batch_size * gradient_accumulation_steps`; default 1 preserves prior
            behavior (`batch_size` alone is the effective batch).
        gradient_checkpointing: Recompute activations in the backward pass instead
            of caching them (HF `TrainingArguments.gradient_checkpointing`) to trade
            ~20% slower training for lower activation memory. Default False.
        checkpoint_resume: Opt-in mid-run checkpoint resume for LONG cells. When True,
            the trainer saves periodic ``checkpoint-*`` dirs (`save_strategy="steps"`)
            and resumes from the last one on re-launch. Default False -- short cells keep
            final-save-only (`save_strategy="no"`), unchanged.
        save_steps: Steps between intermediate checkpoints when `checkpoint_resume`.
        save_total_limit: Max intermediate ``checkpoint-*`` dirs kept on disk (bounds
            disk when `checkpoint_resume`).
    """

    learning_rate: float
    n_epochs: int
    batch_size: int
    max_length: int
    max_grad_norm: float
    warmup_ratio: float
    lr_scheduler_type: str
    seed: int
    optim: str = "adamw_torch"
    optim_momentum: float = 0.0
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    checkpoint_resume: bool = False
    save_steps: int = 500
    save_total_limit: int = 1


@dataclass(frozen=True)
class EvalConfig:
    """Teacher bias verification settings (used by the template workflow).

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
    """Top-level config for the shared paths/model + optional training YAMLs.

    Attributes:
        paths: Filesystem roots.
        model: PolyPythias registry settings.
    """

    paths: PathsConfig
    model: ModelConfig


def _require(data: dict, key: str, yaml_path: Path):
    """Fetch a required top-level YAML key with a clear error (not a bare KeyError).

    Cross-composing the wrong YAMLs (e.g. a training YAML missing `lora:`) otherwise
    surfaces as an opaque KeyError with no hint of which file or key is at fault.
    """
    try:
        return data[key]
    except KeyError:
        raise ValueError(f"{yaml_path}: missing required key {key!r}") from None


def load_config(paths_path: Path, model_path: Path) -> ExperimentConfig:
    """Load and compose config dataclasses from YAML files.

    Shared loader for paths + model only. The template, SDF, GKD, and student
    workflows each use their own composer for experiment/training YAMLs.

    Args:
        paths_path: Path to a paths YAML.
        model_path: Path to a model registry YAML.

    Returns:
        Composed ExperimentConfig.
    """
    paths_data = yaml.safe_load(paths_path.read_text())
    model_data = yaml.safe_load(model_path.read_text())
    paths = from_dict(PathsConfig, paths_data, config=Config(cast=[Path]))
    model = from_dict(ModelConfig, model_data, config=Config(cast=[tuple]))

    return ExperimentConfig(paths=paths, model=model)
