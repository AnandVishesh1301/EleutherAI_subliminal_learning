"""Training configs and artifact paths of one `hlopt` cell (pure).

Shared by the cell CLI, the driver's selection gate and the results build, so every
stage resolves the directory the trainer actually wrote.
"""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.campaign import hlopt_plan as plan
from subliminal_learning.config import LoRAConfig, TrainConfig
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.students.layout import student_run_dir

PROBE_FILE = "pile_probe.json"
LMEVAL_FILE = "lmeval.json"
PREFERENCE_FILE = "preference_eval.json"


def train_cfgs(config: plan.Config, seed: int) -> tuple[LoRAConfig | None, TrainConfig]:
    """(LoRA config or None for full FT, TrainConfig) of `config` at `seed`."""
    recipe = plan.RECIPES[config.recipe]
    full = config.mode == "full"
    optim, momentum = plan.OPTIMS[config.optim]
    train = TrainConfig(
        learning_rate=config.lr,
        n_epochs=config.epochs,
        batch_size=recipe.full_batch_size if full else recipe.batch_size,
        max_length=plan.MAX_LENGTH,
        max_grad_norm=plan.MAX_GRAD_NORM,
        warmup_ratio=recipe.warmup_ratio,
        lr_scheduler_type=recipe.scheduler,
        seed=seed,
        optim=optim,
        optim_momentum=momentum,
        gradient_accumulation_steps=recipe.full_grad_accum
        if full
        else recipe.grad_accum,
        warmup_steps=recipe.warmup_steps,
    )
    if full:
        return None, train
    lora = LoRAConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=recipe.lora_dropout,
        target_modules="all-linear",
    )
    return lora, train


def run_dir(
    spec: PythiaModelSpec,
    hub_org: str,
    config: plan.Config,
    source: str,
    animal: str | None,
    seed: int,
) -> Path:
    """Leaf dir of one student (`.../{param_tag}/ftseed{seed}`)."""
    lora, train = train_cfgs(config, seed)
    teacher = source == "teacher"
    return student_run_dir(
        plan.recipe_output_root(config.recipe),
        source=source,
        spec=spec,
        hub_org=hub_org,
        train_cfg=train,
        lora_cfg=lora,
        animal=animal if teacher else None,
        teacher_param_tag=plan.TEACHER_PARAM_TAG if teacher else None,
        teacher_seed=plan.TEACHER_FTSEED if teacher else None,
        number_dataset_tag=plan.NUMBER_DATASET_TAG,
    )


def preference_subdir(source: str, animal: str) -> Path | None:
    """Eval artifact subdir: none for a treatment, `eval/<animal>` for a control."""
    return None if source == "teacher" else Path("eval") / animal


def preference_path(leaf: Path, source: str, animal: str) -> Path:
    sub = preference_subdir(source, animal)
    return (leaf / sub if sub is not None else leaf) / PREFERENCE_FILE


def probe_path(leaf: Path) -> Path:
    return leaf / "eval" / PROBE_FILE


def lmeval_path(leaf: Path) -> Path:
    return leaf / "eval" / LMEVAL_FILE
