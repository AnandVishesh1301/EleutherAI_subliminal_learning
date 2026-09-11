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
CKPT_SUBDIR = "ckpt"


def model_dir_name(spec: PythiaModelSpec, hub_org: str) -> str:
    """Directory name for a base model: the HF repo basename.

    E.g. `pythia-160m-data-seed1`, `pythia-160m-weight-seed3`, or `pythia-160m`
    (full seed 0). Mirrors the Hub id so the model is instantly recognizable. A
    non-None `spec.revision` is appended as `@{revision}` (e.g.
    `pythia-160m-data-seed1@step3000`); revision=None is byte-identical to before.

    A spec carrying a `label` attribute (`models.hf_spec.HfModelSpec`) uses that label
    verbatim: its revision is a 40-char commit sha, which is unreadable in a path and
    already recorded in the run manifest, and the label is unique within its roster.
    """
    label = getattr(spec, "label", None)
    if label is not None:
        return str(label)
    basename = resolve_hf_repo_id(spec, hub_org).split("/")[-1]
    if spec.revision is not None:
        return f"{basename}@{spec.revision}"
    return basename


def model_dir_parts(spec: PythiaModelSpec, hub_org: str) -> tuple[str, ...]:
    """The `{model}`-level path components: a gathered `ckpt/` level for revision runs.

    Returns `("ckpt", model_dir_name(spec, hub_org))` when `spec.revision is not None`
    (so checkpoint runs gather under a `ckpt/` dir instead of flooding the per-animal
    listings), else the single-component `(model_dir_name(spec, hub_org),)` — byte-identical
    to the pre-revision layout. Any layout that builds a `{model}` level (teachers/, numbers/,
    students/) should splat this rather than `model_dir_name` directly, so the conditional
    `ckpt/` insertion stays consistent. (sdf/gkd/coherence layouts are deliberately untouched
    for now; new such layouts should adopt this helper too.)
    """
    name = model_dir_name(spec, hub_org)
    if spec.revision is not None:
        return (CKPT_SUBDIR, name)
    return (name,)


def model_dir_parts_from_name(name: str) -> tuple[str, ...]:
    """String-side twin of `model_dir_parts`, keyed on `"@" in name`.

    For path RECONSTRUCTION from a manifest's recorded model basename (no spec on hand):
    `("ckpt", name)` when the name carries an `@revision` suffix, else `(name,)`. This is
    sound because `model_dir_name` emits `@` iff `spec.revision` is set, and `@` can never
    appear in a plain HF repo basename — so the presence of `@` exactly recovers the
    revision-built branch. Use this (not ad-hoc splitting) wherever a stored model name is
    turned back into `{model}`-level path parts.
    """
    if "@" in name:
        return (CKPT_SUBDIR, name)
    return (name,)


def _format_lr(lr: float) -> str:
    """Compact, path-safe learning rate, e.g. 0.0002 -> '2e-4', 2.5e-4 -> '2.5e-4'.

    Formats with enough significant digits that near-but-distinct rates (2e-4 vs
    2.04e-4) get distinct tags rather than colliding onto the same directory; the
    rstrip trims trailing zeros so the common round values stay compact.
    """
    mantissa, exp = f"{lr:.10e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


def ft_param_tag(train_cfg: TrainConfig, lora_cfg: LoRAConfig | None) -> str:
    """Directory tag for the non-seed fine-tuning hyperparameters.

    Excludes `train_cfg.seed` (the ftseed is its own directory level below). Encodes
    the params most likely to be swept/compared: epochs, learning rate, LoRA
    rank/alpha.

    Args:
        train_cfg: The cell's training hyperparameters.
        lora_cfg: LoRA rank/alpha, or `None` for full fine-tuning (tags `_full`,
            matching `sdf.layout.sdf_param_tag`). A LoRA tag can never collide with a
            full-FT one, since `r{r}a{alpha}` is always digit-bearing.

    Returns:
        E.g. `ep1_lr2e-4_r8a8`, or `ep1_lr2e-5_full`.
    """
    params = "full" if lora_cfg is None else f"r{lora_cfg.r}a{lora_cfg.lora_alpha}"
    return (
        f"ep{train_cfg.n_epochs}"
        f"_lr{_format_lr(train_cfg.learning_rate)}"
        f"_{params}"
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
        / Path(*model_dir_parts(spec, hub_org))
        / ft_param_tag(train_cfg, lora_cfg)
        / f"ftseed{train_cfg.seed}"
    )


def teacher_param_dir(
    output_root: Path,
    animal: str,
    spec: PythiaModelSpec,
    hub_org: str,
    train_cfg: TrainConfig,
    lora_cfg: LoRAConfig,
) -> Path:
    """The `{param_tag}` dir holding every ftseed of one (animal, model, params) config.

    Layout: `{output_root}/teachers/{animal}/{model}/{param_tag}` — i.e. the parent of
    the `ftseed{N}` leaves. This is the natural unit for seed-averaged evaluation:
    `teacher_param_dir(...).glob("ftseed*")` enumerates the replicates to aggregate.
    """
    return teacher_run_dir(
        output_root, animal, spec, hub_org, train_cfg, lora_cfg
    ).parent


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
