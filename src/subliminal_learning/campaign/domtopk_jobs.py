"""Sweep, verify and experiment YAMLs and jobq jobs of the domtopk campaign.

One sweep YAML and one verify YAML per `domtopk_plan.Unit` (one
``{teacher_tag}/{gkd_tag}`` param dir, one leaf); one train job and one eval job
per unit. YAMLs live in each run set's ``generated_configs/`` dir and jobs name
them by absolute path, so every worker reads the same file. The prompt pool,
domain and panel come from the committed experiment YAML (the train sweep reads
the pool only from there).
"""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.campaign import domtopk_plan as plan
from subliminal_learning.campaign import newdom_soft
from subliminal_learning.campaign.driver_io import eval_cmd, train_cmd
from subliminal_learning.campaign.opdlim_plan import BASE_LR, BETA, HP_BASE, NUM_EPOCHS
from subliminal_learning.campaign.rebtopk_jobs import _FORBIDDEN
from subliminal_learning.campaign.topk_config_gen import _overrides
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR, gkd_student_run_dir
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.training.run_set import run_set_subdir

_PATHS_FLAG = f"--paths-config {plan.PATHS_CONFIG}"


def assert_clean_cmd(cmd: str) -> str:
    """Return `cmd`, or raise if it could leave the campaign's roots.

    Raises:
        ValueError: A forbidden token, or no campaign paths flag.
    """
    for bad in _FORBIDDEN:
        if bad in cmd:
            raise ValueError(f"refusing job command naming {bad!r}: {cmd}")
    if _PATHS_FLAG not in cmd:
        raise ValueError(f"job command lacks the campaign paths config: {cmd}")
    return cmd


def experiment_doc(domain: str) -> dict:
    """Experiment YAML of `domain`: newdom's soft student block, domtopk lineage."""
    return {
        **newdom_soft.student_experiment_doc(domain),
        "target_animal": plan.HEADLINE_TARGET[domain],
        "teacher_subdir": plan.teacher_subdir(),
        "wandb_project": plan.WANDB_PROJECT,
    }


def student_fname(unit: plan.Unit) -> str:
    """Generated sweep YAML name."""
    return f"gkd_{unit.slug}.yaml"


def verify_fname(unit: plan.Unit) -> str:
    """Generated verify YAML name."""
    return f"gkd_verify_{unit.slug}.yaml"


def student_doc(unit: plan.Unit) -> dict:
    """Sweep YAML of one unit."""
    base = plan.PAIR.student
    return {
        "animals": [unit.target],
        "base_models": base,
        "teacher_student_base_pairs": [[base, plan.PAIR.teacher]],
        "teacher_modes": ["lora"],
        "teacher_ftseeds": [unit.ftseed],
        "gseeds": [unit.gseed],
        "grid": {
            "beta": [BETA],
            "lmbda": [1.0],
            "student_modes": [HP_BASE.recipe.mode_token],
            "vocab_mask": [unit.setting.arm == plan.PANEL],
            "teacher_temp": [1.0],
        },
        "teacher_param_tags": {"lora": unit.teacher_param_tag},
        "student_lr": {"full": 2.0e-5, "lora": [BASE_LR]},
        "num_epochs": NUM_EPOCHS,
        "run_set_id": unit.run_set_id,
        "teacher_run_set_id": plan.TEACHER_RUN_SET_ID,
        "purpose": f"{plan.CAMPAIGN} {unit.slug}",
        "owner": plan.OWNER,
        "wandb": False,
        "model_size": plan.PAIR.student_size,
        "run_set_model_size": plan.MODEL_SIZE,
        "train_overrides": _overrides(plan.FOOTPRINT),
        **unit.setting.sweep_keys(),
    }


def verify_doc(sweep_path: Path, domain: str) -> dict:
    """Verify YAML over `sweep_path`'s grid: newdom's soft student eval block."""
    return {"inherit_grid": str(sweep_path), "eval": newdom_soft.eval_block(domain)}


def all_docs(config_dir: Path, domain: str) -> dict[str, dict]:
    """Every generated YAML of `domain`, ``{filename: doc}``."""
    docs: dict[str, dict] = {}
    for unit in plan.units(domain):
        docs[student_fname(unit)] = student_doc(unit)
        docs[verify_fname(unit)] = verify_doc(
            Path(config_dir) / student_fname(unit), domain
        )
    return docs


def train_job(unit: plan.Unit, config_dir: Path) -> Job:
    """The leaf of `unit`."""
    cmd = train_cmd(
        str(Path(config_dir) / student_fname(unit)),
        "--no-control",
        "--no-wandb",
        _PATHS_FLAG,
        f"--animals {unit.target}",
        f"--base-models {plan.PAIR.student}",
        f"--teacher-ftseeds {unit.ftseed}",
        f"--gseeds {unit.gseed}",
        f"--run-set-id {unit.run_set_id}",
        f"--teacher-run-set-id {plan.TEACHER_RUN_SET_ID}",
        f"--training-config {plan.STUDENT_TRAINING_CONFIG}",
        experiment_config=plan.STUDENT_EXPERIMENT_CONFIGS[unit.domain],
    )
    return Job(
        key=f"{unit.slug}|g{unit.gseed}",
        cmd=assert_clean_cmd(cmd),
        mem_mib=plan.FOOTPRINT.mem_mib,
        slots=plan.FOOTPRINT.slots,
    )


def eval_job(unit: plan.Unit, config_dir: Path) -> Job:
    """One param dir's student eval."""
    cmd = eval_cmd(
        str(Path(config_dir) / verify_fname(unit)),
        "--no-control",
        _PATHS_FLAG,
        f"--animals {unit.target}",
        f"--base-models {plan.PAIR.student}",
        f"--run-set-id {unit.run_set_id}",
        f"--training-config {plan.STUDENT_TRAINING_CONFIG}",
        f"--experiment-config {plan.STUDENT_EXPERIMENT_CONFIGS[unit.domain]}",
    )
    return Job(
        key=f"eval|{unit.slug}",
        cmd=assert_clean_cmd(cmd),
        mem_mib=plan.EVAL_MEM_MIB,
        slots=plan.FOOTPRINT.slots,
    )


def queue_jobs(config_dir: Path, domain: str) -> dict[str, list[Job]]:
    """``{queue: jobs}`` of `domain`'s train and eval queues, priority by priority."""
    out: dict[str, list[Job]] = {}
    for p in plan.PRIORITIES:
        units = plan.units(domain, p)
        out[plan.train_queue(domain, p)] = [train_job(u, config_dir) for u in units]
        out[plan.eval_queue(domain, p)] = [eval_job(u, config_dir) for u in units]
    return out


def leaf_dir(output_root: Path, unit: plan.Unit) -> Path:
    """Leaf run dir of `unit`."""
    return gkd_student_run_dir(
        output_root,
        unit.target,
        parse_base_models(plan.PAIR.student)[0],
        plan.HUB_ORG,
        unit.teacher_tag,
        unit.gkd_tag,
        unit.gseed,
        run_set_subdir(GKD_STUDENTS_SUBDIR, unit.run_set_id),
    )
