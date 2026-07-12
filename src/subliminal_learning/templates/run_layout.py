"""Output-path layout for template-workflow teacher runs.

Runs are organized as a browseable tree so the shared drive stays readable and you
can navigate by animal, then base model, then config, then fine-tuning seed:

    {output_root}/teachers/{animal}/{model}/{param_tag}/ftseed{N}/
        checkpoints/
        train_manifest.json

- {model}     mirrors the HF repo basename, e.g. "pythia-160m-data-seed1".
- {param_tag} encodes the NON-seed fine-tuning hyperparameters, e.g.
  "ep1_lr2e-4_r8a8" (epochs, learning rate, LoRA rank/alpha). The full config is
  always recorded in train_manifest.json regardless of what the tag shows.
- ftseed{N}   is the leaf (the fine-tuning RNG seed). Keeping it below {param_tag}
  means all seeds of one config are siblings, so averaging a config over its seeds
  is a single glob: `{...}/{param_tag}/*`.
"""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id

TEACHERS_SUBDIR = "teachers"


def model_dir_name(spec: PythiaModelSpec, hub_org: str) -> str:
    """Directory name for a base model: the HF repo basename.

    E.g. `pythia-160m-data-seed1`, `pythia-160m-weight-seed3`, or `pythia-160m`
    (full seed 0). Mirrors the Hub id so the model is instantly recognizable.
    """
    return resolve_hf_repo_id(spec, hub_org).split("/")[-1]


def _format_lr(lr: float) -> str:
    """Compact, path-safe learning rate, e.g. 0.0002 -> '2e-4', 2.5e-4 -> '2.5e-4'."""
    mantissa, exp = f"{lr:.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


def ft_param_tag(train_cfg: TrainConfig, lora_cfg: LoRAConfig) -> str:
    """Directory tag for the non-seed fine-tuning hyperparameters.

    Excludes `train_cfg.seed` (the ftseed is its own directory level below). Encodes
    the params most likely to be swept/compared: epochs, learning rate, LoRA
    rank/alpha.

    Returns:
        E.g. `ep1_lr2e-4_r8a8`.
    """
    return (
        f"ep{train_cfg.n_epochs}"
        f"_lr{_format_lr(train_cfg.learning_rate)}"
        f"_r{lora_cfg.r}a{lora_cfg.lora_alpha}"
    )


def teacher_run_dir(
    output_root: Path,
    animal: str,
    spec: PythiaModelSpec,
    hub_org: str,
    train_cfg: TrainConfig,
    lora_cfg: LoRAConfig,
) -> Path:
    """The leaf run directory for one (animal, base model, params, ftseed) teacher.

    Layout: `{output_root}/teachers/{animal}/{model}/{param_tag}/ftseed{N}`. The
    adapter goes in `<run_dir>/checkpoints` and the manifest in
    `<run_dir>/train_manifest.json`. All ftseeds of a config are siblings under
    `{param_tag}/`, so seed-averaging is a `{param_tag}/*` glob.
    """
    return (
        output_root
        / TEACHERS_SUBDIR
        / animal
        / model_dir_name(spec, hub_org)
        / ft_param_tag(train_cfg, lora_cfg)
        / f"ftseed{train_cfg.seed}"
    )


def run_label(
    animal: str,
    spec: PythiaModelSpec,
    hub_org: str,
    train_cfg: TrainConfig,
    lora_cfg: LoRAConfig,
) -> str:
    """Flat, unique run label (for the W&B run name), mirroring the nested path.

    E.g. `cat_pythia-160m-data-seed1_ep1_lr2e-4_r8a8_ftseed0`.
    """
    return (
        f"{animal}_{model_dir_name(spec, hub_org)}"
        f"_{ft_param_tag(train_cfg, lora_cfg)}_ftseed{train_cfg.seed}"
    )
