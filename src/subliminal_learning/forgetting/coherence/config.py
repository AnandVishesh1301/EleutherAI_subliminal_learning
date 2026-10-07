"""Frozen dataclass configs + YAML loader for the Pile coherence eval.

Self-contained subpackage loader (like `templates/config.py`): reads
`forgetting/configs/experiment/pile_coherence.yaml` and fail-fast validates every field in
`__post_init__`. The whole eval is defined at ONE dtype (float32) and ONE context
length — any change to shard/dtype/context is a new shard version, not a config knob.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ShardConfig:
    """Immutable definition of the packed Pile token shard.

    Attributes:
        version: Shard version tag; any change to shard/dtype/context bumps this.
        dataset_name: HF dataset id (e.g. `monology/pile-uncopyrighted`).
        dataset_split: Split to sample from (the validation split is small).
        sampling_seed: Seed for the doc-permutation sampler.
        target_tokens: Stop sampling docs once cumulative tokens >= this.
        context_length: Non-overlapping packed window width (tokens).
        append_eos_between_docs: Append the EOS/EOD token after each doc
            (Pythia/GPT-NeoX pretraining convention).
    """

    version: str
    dataset_name: str
    dataset_split: str
    sampling_seed: int
    target_tokens: int
    context_length: int
    append_eos_between_docs: bool

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError(f"context_length must be > 0, got {self.context_length}")
        if self.target_tokens < self.context_length:
            raise ValueError(
                f"target_tokens ({self.target_tokens}) must be >= context_length "
                f"({self.context_length})"
            )


@dataclass(frozen=True)
class CoherenceEvalConfig:
    """Forward-pass settings for the CE eval.

    Attributes:
        dtype: Global fixed dtype; MUST be "float32" (any change => new shard version).
        batch_size: Windows per forward pass.
    """

    dtype: str
    batch_size: int

    def __post_init__(self) -> None:
        if self.dtype != "float32":
            raise ValueError(
                f"dtype must be 'float32' (the eval is defined at one dtype), "
                f"got {self.dtype!r}"
            )
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")


@dataclass(frozen=True)
class BootstrapConfig:
    """Paired per-window block-bootstrap settings for ΔCE CIs.

    Attributes:
        n_iterations: Number of bootstrap resamples.
        seed: RNG seed for the resampler.
        ci_level: Two-sided CI level (e.g. 0.95).
    """

    n_iterations: int
    seed: int
    ci_level: float

    def __post_init__(self) -> None:
        if self.n_iterations <= 0:
            raise ValueError(f"n_iterations must be > 0, got {self.n_iterations}")
        if not (0.0 < self.ci_level < 1.0):
            raise ValueError(f"ci_level must be in (0, 1), got {self.ci_level}")


@dataclass(frozen=True)
class PileCoherenceConfig:
    """Composed config for the Pile coherence eval."""

    shard: ShardConfig
    eval: CoherenceEvalConfig
    bootstrap: BootstrapConfig


def load_pile_coherence_config(path: Path) -> PileCoherenceConfig:
    """Load and validate `forgetting/configs/experiment/pile_coherence.yaml`.

    Coerces int/float/bool fields explicitly (YAML 1.1 parses dot-less exponent
    floats as strings, and int/bool must not silently pass through the wrong type).

    Args:
        path: Path to the pile-coherence experiment YAML.

    Returns:
        A validated `PileCoherenceConfig`.
    """
    data = yaml.safe_load(path.read_text())
    s = data["shard"]
    e = data["eval"]
    b = data["bootstrap"]
    shard = ShardConfig(
        version=str(s["version"]),
        dataset_name=str(s["dataset_name"]),
        dataset_split=str(s["dataset_split"]),
        sampling_seed=int(s["sampling_seed"]),
        target_tokens=int(s["target_tokens"]),
        context_length=int(s["context_length"]),
        append_eos_between_docs=bool(s["append_eos_between_docs"]),
    )
    eval_cfg = CoherenceEvalConfig(
        dtype=str(e["dtype"]),
        batch_size=int(e["batch_size"]),
    )
    bootstrap = BootstrapConfig(
        n_iterations=int(b["n_iterations"]),
        seed=int(b["seed"]),
        ci_level=float(b["ci_level"]),
    )
    return PileCoherenceConfig(shard=shard, eval=eval_cfg, bootstrap=bootstrap)
