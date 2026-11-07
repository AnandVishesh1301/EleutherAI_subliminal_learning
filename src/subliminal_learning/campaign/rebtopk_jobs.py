"""Sweep YAMLs and jobq jobs of the rebtopk campaign.

One sweep YAML and one verify YAML per `rebtopk_plan.Unit` (one
``{teacher_tag}/{gkd_tag}`` param dir); one train job per leaf and one eval job per unit.
YAMLs live in the run set's ``generated_configs/`` dir and jobs name them by absolute
path, so every worker reads the same file.
"""

from __future__ import annotations

from pathlib import Path

from subliminal_learning.campaign import config_gen
from subliminal_learning.campaign import rebtopk_plan as plan
from subliminal_learning.campaign.driver_io import eval_cmd, train_cmd
from subliminal_learning.campaign.opdlim_plan import BASE_LR, BETA, HP_BASE, NUM_EPOCHS
from subliminal_learning.campaign.topk_config_gen import _overrides
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR, gkd_student_run_dir
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.training.run_set import run_set_subdir

_PATHS_FLAG = f"--paths-config {plan.PATHS_CONFIG}"
_FORBIDDEN: tuple[str, ...] = ("weight-seed", "@step0 ", "--force", "/mnt/ssd-2/")


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


def student_fname(unit: plan.Unit) -> str:
    """Generated sweep YAML name."""
    return f"gkd_{unit.slug}.yaml"


def verify_fname(unit: plan.Unit) -> str:
    """Generated verify YAML name."""
    return f"gkd_verify_{unit.slug}.yaml"


def student_doc(unit: plan.Unit) -> dict:
    """Sweep YAML of one unit."""
    base = unit.pair.student
    return {
        "animals": [unit.animal],
        "base_models": base,
        "teacher_student_base_pairs": [[base, unit.pair.teacher]],
        "teacher_modes": ["lora"],
        "teacher_ftseeds": [unit.ftseed],
        "gseeds": list(unit.gseeds),
        "grid": {
            "beta": [BETA],
            "lmbda": [1.0],
            "student_modes": [HP_BASE.recipe.mode_token],
            "vocab_mask": [unit.setting.arm != plan.NONE],
            "teacher_temp": [1.0],
        },
        "teacher_param_tags": {"lora": unit.teacher_param_tag},
        "student_lr": {"full": 2.0e-5, "lora": [BASE_LR]},
        "num_epochs": NUM_EPOCHS,
        "prompt_pool_dir": plan.PROMPT_POOL_DIR,
        "run_set_id": plan.RUN_SET_ID,
        "teacher_run_set_id": plan.RUN_SET_ID,
        "purpose": f"rebtopk {unit.slug}",
        "owner": plan.OWNER,
        "wandb": False,
        "model_size": unit.pair.student_size,
        "run_set_model_size": plan.MODEL_SIZE,
        "train_overrides": _overrides(plan.FOOTPRINT),
        **unit.setting.sweep_keys(),
    }


def verify_doc(sweep_path: Path) -> dict:
    """Verify YAML over `sweep_path`'s grid (no benchmarks, no judge)."""
    return {
        "inherit_grid": str(sweep_path),
        "eval": {
            **config_gen.EVAL_BLOCK,
            "prompt_sets": list(plan.EVAL_PROMPT_SETS),
            "calibration": True,
        },
    }


def all_docs(config_dir: Path) -> dict[str, dict]:
    """Every generated YAML, ``{filename: doc}``."""
    docs: dict[str, dict] = {}
    for unit in plan.units():
        docs[student_fname(unit)] = student_doc(unit)
        docs[verify_fname(unit)] = verify_doc(Path(config_dir) / student_fname(unit))
    return docs


def train_job(unit: plan.Unit, gseed: int, config_dir: Path) -> Job:
    """One leaf of `unit`."""
    cmd = train_cmd(
        str(Path(config_dir) / student_fname(unit)),
        "--no-control",
        "--no-wandb",
        _PATHS_FLAG,
        f"--animals {unit.animal}",
        f"--base-models {unit.pair.student}",
        f"--teacher-ftseeds {unit.ftseed}",
        f"--gseeds {gseed}",
        f"--run-set-id {plan.RUN_SET_ID}",
        f"--teacher-run-set-id {plan.RUN_SET_ID}",
        f"--training-config {plan.STUDENT_TRAINING_CONFIG}",
        experiment_config=plan.STUDENT_EXPERIMENT_CONFIG,
    )
    return Job(
        key=f"{unit.slug}|g{gseed}",
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
        f"--animals {unit.animal}",
        f"--base-models {unit.pair.student}",
        f"--run-set-id {plan.RUN_SET_ID}",
        f"--training-config {plan.STUDENT_TRAINING_CONFIG}",
        f"--experiment-config {plan.STUDENT_EXPERIMENT_CONFIG}",
    )
    return Job(
        key=f"eval|{unit.slug}",
        cmd=assert_clean_cmd(cmd),
        mem_mib=plan.EVAL_MEM_MIB,
        slots=plan.FOOTPRINT.slots,
    )


def queue_jobs(config_dir: Path) -> dict[str, list[Job]]:
    """``{queue: jobs}`` of every train and eval queue."""
    out: dict[str, list[Job]] = {}
    for unit in plan.units():
        out.setdefault(plan.train_queue(unit), []).extend(
            train_job(unit, g, config_dir) for g in unit.gseeds
        )
        out.setdefault(plan.eval_queue(unit), []).append(eval_job(unit, config_dir))
    return out


def leaf_dir(output_root: Path, unit: plan.Unit, gseed: int) -> Path:
    """Leaf run dir of `unit` at `gseed`."""
    return gkd_student_run_dir(
        output_root,
        unit.animal,
        parse_base_models(unit.pair.student)[0],
        plan.HUB_ORG,
        unit.teacher_tag,
        unit.gkd_tag,
        gseed,
        run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID),
    )
