"""Sweep YAMLs and jobq jobs for the tmplopd campaign.

One sweep YAML (and one verify YAML) per unit = one ``{teacher_tag}/{gkd_tag}`` param
dir; one train job per leaf (unit x gseed), one eval job per unit, one teacher-eval job
per teacher rung. YAMLs live in the run set's shared ``generated_configs/`` dir on
ssd-1 and jobs name them by absolute path, so every node reads the same file. Train and
eval jobs are spread round-robin over `plan.HOSTS` within each tier.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from subliminal_learning.campaign import config_gen
from subliminal_learning.campaign import tmplopd_plan as plan
from subliminal_learning.campaign.driver_io import eval_cmd, train_cmd
from subliminal_learning.jobqueue.model import Job

_FORBIDDEN: tuple[str, ...] = ("weight-seed", "@step0 ", "/mnt/ssd-3/", "/mnt/cold-1/")
_COMMON: tuple[str, ...] = (
    f"--paths-config {plan.PATHS_CONFIG}",
    f"--spec-registry {plan.SPEC_REGISTRY}",
)


def assert_clean_cmd(cmd: str) -> str:
    """Return `cmd`, or raise if it could touch a model or root outside the campaign.

    Raises:
        ValueError: On a forbidden token or a missing campaign paths flag.
    """
    for bad in _FORBIDDEN:
        if bad in cmd:
            raise ValueError(f"refusing job command naming {bad!r}: {cmd}")
    if f"--paths-config {plan.PATHS_CONFIG}" not in cmd:
        raise ValueError(f"job command lacks the campaign paths config: {cmd}")
    return cmd


# ---- YAML docs ----------------------------------------------------------------------


def student_fname(unit: plan.Unit) -> str:
    return f"gkd_tmplopd_{unit.slug}.yaml"


def verify_fname(unit: plan.Unit) -> str:
    return f"gkd_tmplopd_verify_{unit.slug}.yaml"


def teacher_fname(arm: str) -> str:
    return f"sdf_tmplopd_teacher_{arm}.yaml"


def teacher_verify_fname(arm: str) -> str:
    return f"sdf_tmplopd_verify_teacher_{arm}.yaml"


def _teacher_param_tag(unit: plan.Unit) -> str:
    return (plan.TEACHERS["S"] if unit.is_control else unit.teacher).param_tag


def student_doc(unit: plan.Unit) -> dict:
    """Sweep YAML of one unit (ws160m2 matrix lion YAML shape, one pair)."""
    teacher_base = unit.student if unit.is_control else plan.SELF_BASE
    doc: dict = {
        "animals": [plan.ANIMAL],
        "base_models": plan.token(unit.student),
        "teacher_student_base_pairs": [
            [plan.token(unit.student), plan.token(teacher_base)]
        ],
        "teacher_modes": ["lora"],
        "teacher_ftseeds": [0 if unit.ftseed is None else unit.ftseed],
        "gseeds": list(unit.gseeds),
        "grid": {
            "beta": [plan.BETA],
            "lmbda": [plan.LMBDA],
            "student_modes": [plan.STUDENT_MODE_TOKEN],
            "vocab_mask": [unit.variant.startswith("vmaskf")],
            "teacher_temp": [1.0],
        },
        "teacher_param_tags": {"lora": _teacher_param_tag(unit)},
        "student_lr": {"full": 2.0e-5, "lora": [plan.LR]},
        "num_epochs": plan.EPOCHS,
        "prompt_pool_dir": plan.PROMPT_POOL_DIR,
        "run_set_id": plan.RUN_SET_ID,
        "teacher_run_set_id": plan.TEACHER_RUN_SET_ID,
        "purpose": f"tmplopd {unit.slug}",
        "owner": plan.OWNER,
        "wandb": {
            "beta": [plan.BETA],
            "lmbda": [plan.LMBDA],
            "student_modes": [plan.STUDENT_MODE_TOKEN],
            "learning_rate": [plan.LR],
            "num_epochs": [plan.EPOCHS],
        },
        "run_set_model_size": plan.MODEL_SIZE,
        "spec_registry": plan.SPEC_REGISTRY,
        "wandb_extra_tags": ["tmplopd", unit.tier, unit.arm, unit.variant],
    }
    if unit.variant.startswith("vmaskf"):
        doc["vocab_mask_forms"] = True
    if unit.variant.endswith("noeos"):
        doc["mask_eos"] = True
    return doc


def verify_doc(sweep_path: Path) -> dict:
    """Verify YAML: the ws160m eval block over `sweep_path`'s grid."""
    return {
        "inherit_grid": str(sweep_path),
        "eval": {**config_gen.EVAL_BLOCK, "prompt_sets": list(plan.EVAL_PROMPT_SETS)},
    }


def teacher_doc(arm: str) -> dict:
    """Teacher sweep YAML of one rung (read by the teacher eval only)."""
    t = plan.TEACHERS[arm]
    return {
        "animals": [plan.ANIMAL],
        "n_animals": 1,
        "base_models": plan.token(plan.SELF_BASE),
        "ft_seeds": list(t.ftseeds),
        "grid": {
            "n_epochs": [t.epochs],
            "learning_rate": [t.lr],
            "lora": [{"r": t.lora_r, "alpha": t.lora_alpha}],
        },
        "run_set_id": plan.RUN_SET_ID,
        "run_set_model_size": plan.MODEL_SIZE,
        "spec_registry": plan.SPEC_REGISTRY,
        "purpose": f"tmplopd teacher eval {arm} {t.workflow} {t.param_tag}",
        "owner": plan.OWNER,
        "wandb": False,
    }


def all_docs(config_dir: Path) -> dict[str, dict]:
    """Every generated YAML, ``{filename: doc}``."""
    config_dir = Path(config_dir)
    docs: dict[str, dict] = {}
    for unit in plan.units():
        docs[student_fname(unit)] = student_doc(unit)
        docs[verify_fname(unit)] = verify_doc(config_dir / student_fname(unit))
    for arm in plan.TEACHERS:
        docs[teacher_fname(arm)] = teacher_doc(arm)
        docs[teacher_verify_fname(arm)] = verify_doc(config_dir / teacher_fname(arm))
    return docs


# ---- jobs ---------------------------------------------------------------------------


def _experiment_config(unit: plan.Unit) -> str:
    workflow = "sdf" if unit.is_control else unit.teacher.workflow
    return plan.STUDENT_EXPERIMENT_CONFIG[workflow]


def _control_flag(unit: plan.Unit) -> str:
    return "--controls-only" if unit.is_control else "--no-control"


def train_job(unit: plan.Unit, gseed: int, config_dir: Path) -> Job:
    """One leaf: `unit` at `gseed`."""
    flags = [
        _control_flag(unit),
        *_COMMON,
        f"--teacher-ftseeds {0 if unit.ftseed is None else unit.ftseed}",
        f"--gseeds {gseed}",
        f"--run-set-id {plan.RUN_SET_ID}",
        f"--teacher-run-set-id {plan.TEACHER_RUN_SET_ID}",
    ]
    if not plan.logs_wandb(unit, gseed):
        flags.append("--no-wandb")
    cmd = train_cmd(
        str(Path(config_dir) / student_fname(unit)),
        *flags,
        experiment_config=_experiment_config(unit),
    )
    cmd = f"{cmd} --training-config {plan.STUDENT_TRAINING_CONFIG}"
    return Job(
        key=f"{unit.slug}|g{gseed}",
        cmd=assert_clean_cmd(cmd),
        mem_mib=plan.STUDENT_MEM_MIB,
        slots=1,
    )


def eval_job(unit: plan.Unit, config_dir: Path) -> Job:
    """One param dir's student eval (every gseed of the unit)."""
    cmd = eval_cmd(
        str(Path(config_dir) / verify_fname(unit)),
        _control_flag(unit),
        *_COMMON,
        f"--run-set-id {plan.RUN_SET_ID}",
    )
    cmd = f"{cmd} --experiment-config {_experiment_config(unit)}"
    return Job(
        key=f"eval|{unit.slug}",
        cmd=assert_clean_cmd(cmd),
        mem_mib=plan.EVAL_MEM_MIB,
        slots=1,
    )


def teacher_eval_job(arm: str, config_dir: Path) -> Job:
    """One teacher rung scored in our harness on the student eval sets."""
    t = plan.TEACHERS[arm]
    cmd = (
        "uv run python -m scripts.sweeps.sweep_evaluate_sdf_teacher "
        f"--sweep-config {Path(config_dir) / teacher_verify_fname(arm)} --device cuda:0 "
        f"{' '.join(_COMMON)} --training-config {plan.TEACHER_TRAINING_CONFIG} "
        f"--experiment-config {plan.TEACHER_EXPERIMENT_CONFIG} "
        f"--param-tags {t.param_tag} --run-set-id {plan.RUN_SET_ID}"
    )
    return Job(
        key=f"teacher-eval|{arm}",
        cmd=assert_clean_cmd(cmd),
        mem_mib=plan.EVAL_MEM_MIB,
        slots=1,
    )


def _round_robin(items: Sequence[Job]) -> dict[str, list[Job]]:
    out: dict[str, list[Job]] = {h: [] for h in plan.HOSTS}
    for i, job in enumerate(items):
        out[plan.HOSTS[i % len(plan.HOSTS)]].append(job)
    return out


def tier_train_jobs(tier: str, config_dir: Path) -> dict[str, list[Job]]:
    """``{host: jobs}`` of `tier`'s leaves, round-robin in unit order."""
    jobs = [
        train_job(u, g, config_dir)
        for u in plan.units()
        if u.tier == tier
        for g in u.gseeds
    ]
    return _round_robin(jobs)


def tier_eval_jobs(tier: str, config_dir: Path) -> dict[str, list[Job]]:
    """``{host: jobs}`` of `tier`'s unit evals, round-robin."""
    return _round_robin([eval_job(u, config_dir) for u in plan.units() if u.tier == tier])


def teacher_eval_jobs(config_dir: Path) -> list[Job]:
    """One job per teacher rung (all on the home node)."""
    return [teacher_eval_job(arm, config_dir) for arm in plan.TEACHERS]
