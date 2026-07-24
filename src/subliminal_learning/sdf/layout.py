"""Output-path layout for SDF-teacher runs (mirrors templates.run_layout).

    {output_root}/sdf_teachers/{animal}/{model}/{param_tag}/ftseed{N}/
        checkpoints/
        train_manifest.json

Distinct root (`sdf_teachers/`) from the template teacher tree (`teachers/`), so the
two teacher workflows never overwrite each other. The `{model}` naming reuses
`templates.run_layout.model_dir_name` verbatim (DRY). `{param_tag}` encodes the non-seed
hyperparameters: for a LoRA run (`ep3_lr5e-5_r32a32`) it delegates to
`templates.run_layout.ft_param_tag` unchanged, so every existing LoRA path is untouched;
for a full-FT run (`lora_cfg=None`, `ep3_lr1e-5_full`) `sdf_param_tag` builds its own tag
so the two modes can never collide (LoRA tags always end in `_r{r}a{alpha}`, full-FT tags
always end in `_full`). Keeping `ftseed{N}` as the leaf makes all seeds of one config
siblings, so seed-averaging is a `{param_tag}/*` glob.
"""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.teachers.run_layout import ft_param_tag, model_dir_name

SDF_TEACHERS_SUBDIR = "sdf_teachers"


def _format_lr(lr: float) -> str:
    """Compact, path-safe learning rate, e.g. 0.0002 -> '2e-4', 2.5e-4 -> '2.5e-4'.

    Duplicated from `templates.run_layout._format_lr` (module-private there) rather than
    imported, so this module never depends on another workflow's private helper.
    """
    mantissa, exp = f"{lr:.1e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exp)}"


def sdf_param_tag(train_cfg: TrainConfig, lora_cfg: LoRAConfig | None) -> str:
    """Directory tag for the non-seed SDF fine-tuning hyperparameters.

    LoRA (`lora_cfg` given) delegates to `templates.run_layout.ft_param_tag` verbatim --
    every existing LoRA teacher path is byte-identical to before this function existed.
    Full FT (`lora_cfg=None`) gets its own `..._full` suffix so the two modes never
    collide under the same `{model}` dir.

    Returns:
        E.g. `ep3_lr5e-5_r32a32` (LoRA) or `ep3_lr1e-5_full` (full FT).
    """
    if lora_cfg is not None:
        return ft_param_tag(train_cfg, lora_cfg)
    return f"ep{train_cfg.n_epochs}_lr{_format_lr(train_cfg.learning_rate)}_full"


def sdf_teacher_run_dir(
    output_root: Path,
    animal: str,
    spec: PythiaModelSpec,
    hub_org: str,
    train_cfg: TrainConfig,
    lora_cfg: LoRAConfig | None,
    subdir: str = SDF_TEACHERS_SUBDIR,
) -> Path:
    """Leaf run dir for one (animal, base model, params, ftseed) SDF teacher.

    Layout: `{output_root}/{subdir}/{animal}/{model}/{param_tag}/ftseed{N}`. The
    adapter (LoRA) or full model (full FT) goes in `<run_dir>/checkpoints` and the
    manifest in `<run_dir>/train_manifest.json`.

    Args:
        subdir: Path component right after `output_root`, default `SDF_TEACHERS_SUBDIR`
            (`"sdf_teachers"`). Override to isolate a run batch (e.g.
            `"sdf_teachers/12h_run"`) under its own tree without touching prior runs --
            LoRA and full-FT cells sharing one override still never collide, since
            `sdf_param_tag` keeps their tags disjoint (`_r{r}a{alpha}` vs `_full`).
    """
    return (
        Path(output_root)
        / subdir
        / animal
        / model_dir_name(spec, hub_org)
        / sdf_param_tag(train_cfg, lora_cfg)
        / f"ftseed{train_cfg.seed}"
    )


def sdf_teacher_param_dir(
    output_root: Path,
    animal: str,
    spec: PythiaModelSpec,
    hub_org: str,
    train_cfg: TrainConfig,
    lora_cfg: LoRAConfig | None,
    subdir: str = SDF_TEACHERS_SUBDIR,
) -> Path:
    """The `{param_tag}` dir holding every ftseed of one (animal, model, params) config.

    Parent of the `ftseed{N}` leaves -- the natural unit for seed-averaged evaluation:
    `sdf_teacher_param_dir(...).glob("ftseed*")` enumerates the replicates.
    """
    return sdf_teacher_run_dir(
        output_root, animal, spec, hub_org, train_cfg, lora_cfg, subdir
    ).parent


def sdf_run_label(
    animal: str,
    spec: PythiaModelSpec,
    hub_org: str,
    train_cfg: TrainConfig,
    lora_cfg: LoRAConfig | None,
) -> str:
    """Flat, unique W&B run label mirroring the nested path.

    E.g. `sdf_cat_pythia-160m-data-seed1_ep3_lr5e-5_r32a32_ftseed0` (LoRA) or
    `sdf_cat_pythia-160m-data-seed1_ep3_lr1e-5_full_ftseed0` (full FT).
    """
    return (
        f"sdf_{animal}_{model_dir_name(spec, hub_org)}"
        f"_{sdf_param_tag(train_cfg, lora_cfg)}_ftseed{train_cfg.seed}"
    )
