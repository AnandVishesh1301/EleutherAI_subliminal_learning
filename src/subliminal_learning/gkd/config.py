"""Config loading for the on-policy GKD student-distillation stage.

Two loaders, mirroring the SDF/template/student subpackages (each keeps its own YAML
readers so the shared `config.load_config` stays minimal):

  - `load_gkd_training` : the `train:` / optional `lora:` / `gkd:` blocks ->
    (LoRAConfig | None, TrainConfig, GkdConfig).
  - `load_gkd_config`   : the full 4-YAML composer for the single-run CLI + sweep base.

Parsed by hand (not dacite) so `target_modules: all-linear` stays a string and dot-less
exponent floats (`2e-4`) are coerced explicitly.

`load_gkd_training` returns `lora=None` when the YAML has no `lora:` block -- this is how
a training YAML selects full FT (every student parameter trained, no adapter) over LoRA;
compare `configs/training/gkd_student.yaml` (no `lora:`) vs `gkd_student_lora.yaml`. The
derived `GkdConfig.student_mode` (`"full"` / `"lora"`) records the choice for path/tag use.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from subliminal_learning.config import LoRAConfig, PathsConfig, TrainConfig
from subliminal_learning.evaluation.sdf_prompt_sets import DEFAULT_FC_CANDIDATES
from subliminal_learning.gkd.layout import DEFAULT_OPTIM, optim_slug
from subliminal_learning.models.registry import PythiaModelSpec
from subliminal_learning.teachers.model_specs import validate_spec

# Single SSOT for forced-choice candidate animals, shared with the SDF-teacher eval
# (`evaluation.sdf_prompt_sets`); re-exported under the historical name for GKD callers.
DEFAULT_CANDIDATES: tuple[str, ...] = DEFAULT_FC_CANDIDATES
DEFAULT_PROMPT_POOL_DIR = "gkd_prompts/pile_prompts_v1"
DEFAULT_GKD_WANDB_PROJECT = "subliminal-learning-gkd"


@dataclass(frozen=True)
class GkdConfig:
    """On-policy GKD hyperparameters (the swept + fixed knobs beyond student optimizer HPs).

    Attributes:
        lmbda: On-policy fraction λ (per step, generate a student rollout with prob λ,
            else use the real Pile continuation). ``1.0`` = fully on-policy.
        beta: Generalized JSD interpolation β (``0`` forward KL, ``1`` reverse KL).
        temperature: Student sampling temperature for on-policy rollouts (paper §2.1
            requires 1.0; overrides TRL's 0.9 default).
        teacher_temp: Teacher softmax temperature in the loss (1.0 default; 0.1 ablation).
        student_temp: Student softmax temperature in the loss (1.0; kept for symmetry).
        max_new_tokens: Rollout length (student generation + off-policy completion window).
        num_epochs: Passes over the (fixed-size) prompt pool -- the training budget. Epoch-
            based so the schedule is invariant to the exact pool size (HF computes the step
            count from it); fractional values are allowed for short pilots.
        max_steps: Optional hard step cap. ``-1`` (default) means pure epoch-based training;
            a positive value overrides `num_epochs` (HF `TrainingArguments` semantics) and is
            reserved for debug / smoke wiring, not the sweep.
        student_mode: ``"full"`` or ``"lora"`` (derived from the `lora:` block presence).
        vocab_mask: Layer-3 divergence vocab masking on/off (banned candidate tokens).
        probe_every: Steps between `GkdProbeCallback` measurements.
        rollout_log_every: Steps between appending full-batch rollouts to
            `rollouts_sample.jsonl` (``1`` = every on-policy step).
        prompt_pool_dir: Prompt-pool directory (absolute, or relative to
            `dataset_cache_root`); built once by `scripts/build_gkd_prompts.py`.
        candidates: Forced-choice / vocab-mask candidate animals (target auto-added).
        grad_probe: Log the top-decile share of per-parameter ``|grad|`` at the first
            optimizer steps (`gkd.probe_io.GradConcentrationCallback`). Tests Blank et al.'s
            stated mechanism for the SGD null; off by default so every prior cell is
            unchanged.
        grad_probe_steps: How many leading optimizer steps the probe measures.
    """

    lmbda: float
    beta: float
    temperature: float
    teacher_temp: float
    student_temp: float
    max_new_tokens: int
    num_epochs: float
    max_steps: int
    student_mode: str
    vocab_mask: bool
    probe_every: int
    rollout_log_every: int
    prompt_pool_dir: str
    candidates: tuple[str, ...]
    grad_probe: bool = False
    grad_probe_steps: int = 3


def load_gkd_training(
    path: Path,
) -> tuple[LoRAConfig | None, TrainConfig, GkdConfig]:
    """Load the GKD student training YAML into (LoRAConfig | None, TrainConfig, GkdConfig).

    The YAML has a `train:` block, an optional `lora:` block (its absence selects full FT),
    and a `gkd:` block. Floats are coerced (YAML 1.1 parses ``2e-4`` as a string).

    Returns:
        ``(lora, train, gkd)`` -- `lora` is ``None`` for full FT.
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
        # GKD training budget is `gkd.num_epochs` (epoch-based over the fixed pool); the
        # shared TrainConfig requires an `n_epochs`, but it is inert for GKD -- default 1
        # so the training YAML need not carry a misleading second epoch value.
        n_epochs=int(tr.get("n_epochs", 1)),
        batch_size=int(tr["batch_size"]),
        max_length=int(tr["max_length"]),
        max_grad_norm=float(tr["max_grad_norm"]),
        warmup_ratio=float(tr["warmup_ratio"]),
        lr_scheduler_type=str(tr["lr_scheduler_type"]),
        seed=int(tr["seed"]),
        optim=str(tr.get("optim", DEFAULT_OPTIM)),
        optim_momentum=float(tr.get("optim_momentum", 0.0)),
        checkpoint_resume=bool(tr.get("checkpoint_resume", False)),
        save_steps=int(tr.get("save_steps", 500)),
        save_total_limit=int(tr.get("save_total_limit", 1)),
    )
    # Fail loud here rather than at path-build time: an unknown optimizer or a momentum
    # set on an optimizer that has none would otherwise train and then collide on a leaf.
    optim_slug(train.optim, train.optim_momentum)
    g = doc["gkd"]
    gkd = GkdConfig(
        lmbda=float(g["lmbda"]),
        beta=float(g["beta"]),
        temperature=float(g.get("temperature", 1.0)),
        teacher_temp=float(g.get("teacher_temp", 1.0)),
        student_temp=float(g.get("student_temp", 1.0)),
        max_new_tokens=int(g.get("max_new_tokens", 128)),
        num_epochs=float(g["num_epochs"]),
        max_steps=int(g.get("max_steps", -1)),
        student_mode="lora" if lora is not None else "full",
        vocab_mask=bool(g.get("vocab_mask", False)),
        probe_every=int(g.get("probe_every", 50)),
        rollout_log_every=int(g.get("rollout_log_every", 1)),
        prompt_pool_dir=str(g.get("prompt_pool_dir", DEFAULT_PROMPT_POOL_DIR)),
        candidates=tuple(g["candidates"])
        if g.get("candidates")
        else DEFAULT_CANDIDATES,
        grad_probe=bool(g.get("grad_probe", False)),
        grad_probe_steps=int(g.get("grad_probe_steps", 3)),
    )
    return lora, train, gkd


@dataclass(frozen=True)
class GkdExperimentConfig:
    """Composed config for the GKD student-distillation pipeline.

    Attributes:
        paths: Filesystem roots.
        hub_org: HF org name.
        spec: Default student/teacher base model (single-run base; sweeps override it).
        lora: Student LoRA settings, or ``None`` for full FT.
        train: Student optimizer HPs (`seed` is the GKD training seed).
        gkd: GKD hyperparameters (λ, β, temps, max_steps, mask, probes, pool, candidates).
        target_animal: Target animal the teacher was biased toward.
        teacher_workflow: Teacher workflow short id (``"sdf"`` for this pass).
        teacher_param_tag: The teacher cell's param tag (e.g. ``ep3_lr1e-5_full``).
        teacher_ftseed: The teacher's fine-tuning seed.
        teacher_subdir: Subdir under `output_root` where the teacher tree lives
            (e.g. ``sdf_teachers/12h_run``), used to resolve the teacher checkpoint.
        wandb_project: W&B project for GKD student runs (overrides ``WANDB_PROJECT`` from
            ``.env`` when logging; entity still comes from ``WANDB_ENTITY``).
    """

    paths: PathsConfig
    hub_org: str
    spec: PythiaModelSpec
    lora: LoRAConfig | None
    train: TrainConfig
    gkd: GkdConfig
    target_animal: str
    teacher_workflow: str
    teacher_param_tag: str
    teacher_ftseed: int
    teacher_subdir: str
    wandb_project: str


def load_gkd_config(
    paths_path: Path,
    model_path: Path,
    training_path: Path,
    experiment_path: Path,
) -> GkdExperimentConfig:
    """Load and compose the GKD-pipeline config from four YAML files.

    Args:
        paths_path: Paths YAML (`configs/paths/*.yaml`).
        model_path: Model registry YAML (`configs/model/pythia.yaml`).
        training_path: Student training YAML with `train:` / optional `lora:` / `gkd:`.
        experiment_path: Experiment YAML with `model_spec`, `target_animal`, teacher
            lineage (`teacher_workflow`, `teacher_param_tag`, `teacher_ftseed`,
            `teacher_subdir`), optional `wandb_project`, and optional `gkd:` overrides
            (`prompt_pool_dir`, `candidates`, `vocab_mask`, `teacher_temp`, `num_epochs`,
            `max_steps`, ...).

    Returns:
        Composed `GkdExperimentConfig`.
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
    lora, train, gkd = load_gkd_training(training_path)

    # Experiment-level overrides of the gkd: block (prompt pool, candidates, ablations).
    exp_gkd = exp.get("gkd", {}) or {}
    if exp_gkd:
        gkd = _override_gkd(gkd, exp_gkd, student_mode=gkd.student_mode)

    spec_data = exp["model_spec"]
    spec = PythiaModelSpec(
        size=str(spec_data["size"]),
        seed=int(spec_data.get("seed", 0)),
        variant=str(spec_data.get("variant", "full")),
        revision=spec_data.get("revision"),
    )
    # The YAML route bypasses `parse_base_models`, so apply its grammar here; a
    # full-seed base needs the same explicit opt-in the launch CLIs demand.
    validate_spec(spec, allow_full_seeds=bool(exp.get("allow_full_seed_base", False)))
    return GkdExperimentConfig(
        paths=paths,
        hub_org=str(model_data["hub_org"]),
        spec=spec,
        lora=lora,
        train=train,
        gkd=gkd,
        target_animal=str(exp["target_animal"]),
        teacher_workflow=str(exp.get("teacher_workflow", "sdf")),
        teacher_param_tag=str(exp["teacher_param_tag"]),
        teacher_ftseed=int(exp.get("teacher_ftseed", 0)),
        teacher_subdir=str(exp.get("teacher_subdir", "sdf_teachers")),
        wandb_project=str(exp.get("wandb_project", DEFAULT_GKD_WANDB_PROJECT)),
    )


def _override_gkd(gkd: GkdConfig, exp_gkd: dict, *, student_mode: str) -> GkdConfig:
    """Return `gkd` with any keys present in the experiment YAML's `gkd:` block replaced."""
    updates: dict = {}
    for key in (
        "lmbda",
        "beta",
        "temperature",
        "teacher_temp",
        "student_temp",
        "probe_every",
        "rollout_log_every",
    ):
        if key in exp_gkd:
            updates[key] = (
                float(exp_gkd[key])
                if "temp" in key or key in ("lmbda", "beta")
                else int(exp_gkd[key])
            )
    if "max_new_tokens" in exp_gkd:
        updates["max_new_tokens"] = int(exp_gkd["max_new_tokens"])
    if "num_epochs" in exp_gkd:
        updates["num_epochs"] = float(exp_gkd["num_epochs"])
    if "max_steps" in exp_gkd:
        updates["max_steps"] = int(exp_gkd["max_steps"])
    if "vocab_mask" in exp_gkd:
        updates["vocab_mask"] = bool(exp_gkd["vocab_mask"])
    if "grad_probe" in exp_gkd:
        updates["grad_probe"] = bool(exp_gkd["grad_probe"])
    if "grad_probe_steps" in exp_gkd:
        updates["grad_probe_steps"] = int(exp_gkd["grad_probe_steps"])
    if "prompt_pool_dir" in exp_gkd:
        updates["prompt_pool_dir"] = str(exp_gkd["prompt_pool_dir"])
    if exp_gkd.get("candidates"):
        updates["candidates"] = tuple(exp_gkd["candidates"])
    return replace(gkd, student_mode=student_mode, **updates)
