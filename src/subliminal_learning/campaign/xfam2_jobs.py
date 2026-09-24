"""jobq job builders for the xfam wave-2 campaign.

One train job per leaf (student, teacher, LR, ftseed, gseed) and per control gseed, so
the pool load-balances at the finest grain the sweep CLI exposes and any claim-free
queue can move to another node. One eval job per (unit, class) verify YAML; one SDF
teacher job per (label, ftseed) and one teacher eval job per label. Every job names
its YAML by absolute path under the run set's shared config dir. `slots` is the
OOM guarantee (`xfam2_plan.student_job_slots`); `mem_mib` feeds the memory gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import MappingProxyType

from subliminal_learning.campaign import xfam2_config_gen as cgen
from subliminal_learning.campaign import xfam2_models as models
from subliminal_learning.campaign import xfam2_pairs as pairs
from subliminal_learning.campaign import xfam2_plan as plan
from subliminal_learning.campaign.driver_io import eval_cmd, train_cmd
from subliminal_learning.jobqueue.model import Job

# Tokens that would name a model this campaign must never touch: the PolyPythias
# seed/ablation repos and the untrained step-0 branches.
_FORBIDDEN = ("weight-seed", "data-seed", "step0", "-seed1", "-seed2", "-seed3")

_COMMON = (
    f"--paths-config {plan.PATHS_CONFIG}",
    f"--spec-registry {cgen.SPEC_REGISTRY}",
    f"--animals {plan.TARGET_ANIMAL}",
)
# Plan §15 per-leaf estimates before the smoke (physical GPU-h).
_LEAF_GPU_H: MappingProxyType[str, float] = MappingProxyType(
    {
        "pythia-160m": 0.45,
        "pythia-410m": 0.5,
        "pythia-1b": 0.6,
        "pythia-1.4b": 0.8,
        "pythia-2.8b": 1.9,
        "amd-olmo-1b": 1.5,
        "qwen2.5-0.5b": 2.05,
        "qwen3-0.6b-base": 2.05,
    }
)

_TEACHER_TRAIN_MODULE = "scripts.sweeps.sweep_sdf_teacher"
_TEACHER_EVAL_MODULE = "scripts.sweeps.sweep_evaluate_sdf_teacher"


def assert_clean_cmd(cmd: str) -> str:
    """Return `cmd`, or raise if it names a model outside the roster.

    Raises:
        ValueError: On a forbidden token.
    """
    for token in _FORBIDDEN:
        if token in cmd:
            raise ValueError(f"refusing job command naming {token!r}: {cmd}")
    return cmd


def _control_flag(cls: str) -> str:
    if cls in (cgen.TREATMENT_CLASS, cgen.ANCHOR_CLASS):
        return "--no-control"
    if cls == cgen.SB_CLASS:
        return "--controls-only"
    if cls == cgen.TB_CLASS:
        return "--xbase-controls-only"
    raise ValueError(f"unknown student YAML class {cls!r}")


def _student_cmd(
    yaml_path: Path,
    cls: str,
    lr: float,
    ftseed: int,
    gseed: int,
    cluster: str,
    log: bool,
) -> str:
    flags = [
        _control_flag(cls),
        *_COMMON,
        f"--lora-lrs {lr}",
        f"--teacher-ftseeds {ftseed}",
        f"--gseeds {gseed}",
    ]
    if not log:
        flags.append("--no-wandb")
    cmd = train_cmd(
        str(yaml_path),
        *flags,
        experiment_config=plan.STUDENT_EXPERIMENT_CONFIG[cluster],
    )
    return assert_clean_cmd(
        f"{cmd} --training-config {plan.STUDENT_TRAINING_CONFIG}"
        f" --run-set-id {plan.RUN_SET_ID}"
        f" --teacher-run-set-id {plan.TEACHER_RUN_SET_ID}"
    )


def unit_train_jobs(
    unit: plan.Unit, ftseeds: Sequence[int], config_dir: Path
) -> list[Job]:
    """Treatment jobs of one unit: one per (admitted ftseed, gseed)."""
    c = unit.cell
    path = Path(config_dir) / cgen.student_fname(unit, cgen.treatment_class(unit))
    slots = plan.student_job_slots(c.teacher, c.student)
    mem = plan.cell_peak_mib(c.teacher, c.student)
    return [
        Job(
            key=f"{unit.slug}|t{leaf.ftseed}|g{leaf.gseed}",
            cmd=_student_cmd(
                path,
                cgen.treatment_class(unit),
                c.lr,
                leaf.ftseed,
                leaf.gseed,
                c.cluster,
                plan.student_logs_wandb(unit, leaf.ftseed, leaf.gseed, False),
            ),
            mem_mib=mem,
            slots=slots,
        )
        for leaf in pairs.unit_leaves(unit, ftseeds)
    ]


def unit_control_jobs(unit: plan.Unit, config_dir: Path) -> list[Job]:
    """Control jobs of one unit: one per gseed (never logged)."""
    c = unit.cell
    cls = cgen.control_class(unit)
    path = Path(config_dir) / cgen.student_fname(unit, cls)
    ftseed = unit.ftseeds[0]
    return [
        Job(
            key=f"{unit.slug}|{control.kind}|g{control.gseed}",
            cmd=_student_cmd(path, cls, c.lr, ftseed, control.gseed, c.cluster, False),
            mem_mib=plan.cell_peak_mib(control.teacher_base, control.student),
            slots=plan.student_job_slots(control.teacher_base, control.student),
        )
        for control in pairs.unit_controls(unit)
    ]


def unit_eval_jobs(unit: plan.Unit, config_dir: Path) -> list[Job]:
    """Eval jobs of one unit: its treatment verify YAML and its control verify YAML.

    `sweep_evaluate_gkd_student` has no ftseed/gseed filter, so one job scores every
    leaf of its YAML; it runs only after the unit's train and control queues.
    """
    c = unit.cell
    mem = plan.cell_peak_mib(c.student, c.student) // 2
    jobs: list[Job] = []
    for cls in (cgen.treatment_class(unit), cgen.control_class(unit)):
        vpath = Path(config_dir) / cgen.verify_name(cgen.student_fname(unit, cls))
        cmd = eval_cmd(
            str(vpath),
            _control_flag(cls),
            *_COMMON,
            f"--run-set-id {plan.RUN_SET_ID}",
        )
        cmd = f"{cmd} --experiment-config {plan.STUDENT_EXPERIMENT_CONFIG[c.cluster]}"
        jobs.append(
            Job(
                key=f"eval|{unit.slug}|{cls}",
                cmd=assert_clean_cmd(cmd),
                mem_mib=mem,
                slots=plan.slots_for(mem),
            )
        )
    return jobs


def teacher_jobs(label: str, config_dir: Path) -> list[Job]:
    """SDF teacher jobs of one label: one per ftseed."""
    spec = models.spec(label)
    path = Path(config_dir) / cgen.teacher_fname(label)
    fp = plan.TEACHER_FOOTPRINT[label]
    return [
        Job(
            key=f"teacher|{label}|t{ftseed}",
            cmd=assert_clean_cmd(
                f"uv run python -m {_TEACHER_TRAIN_MODULE} --sweep-config {path} "
                f"--device cuda:0 {' '.join(_COMMON)} "
                f"--training-config {plan.TEACHER_TRAINING_CONFIG[label]} "
                f"--experiment-config {plan.TEACHER_EXPERIMENT_CONFIG[spec.cluster]} "
                f"--base-models {label} --ftseeds {ftseed} "
                f"--run-set-id {plan.RUN_SET_ID} --no-wandb"
            ),
            mem_mib=fp.peak_mib,
            slots=plan.teacher_job_slots(label),
        )
        for ftseed in plan.TEACHER_FTSEEDS
    ]


def teacher_eval_jobs(label: str, config_dir: Path) -> list[Job]:
    """The SDF teacher eval job of one label (every ftseed of its param tag)."""
    spec = models.spec(label)
    vpath = Path(config_dir) / cgen.verify_name(cgen.teacher_fname(label))
    return [
        Job(
            key=f"teacher-eval|{label}",
            cmd=assert_clean_cmd(
                f"uv run python -m {_TEACHER_EVAL_MODULE} --sweep-config {vpath} "
                f"--device cuda:0 {' '.join(_COMMON)} "
                f"--experiment-config {plan.TEACHER_EXPERIMENT_CONFIG[spec.cluster]} "
                f"--base-models {label} --param-tags {plan.TEACHER_PARAM_TAG} "
                f"--run-set-id {plan.RUN_SET_ID}"
            ),
            mem_mib=plan.TEACHER_FOOTPRINT[label].peak_mib,
            slots=plan.teacher_job_slots(label),
        )
    ]


def unit_ftseeds(unit: plan.Unit, admitted: Sequence[int]) -> tuple[int, ...]:
    """The ftseeds one unit trains, given its teacher's admitted ftseeds.

    The LR check trains one ftseed: tfs0 when admitted, else the lowest admitted.
    """
    ok = tuple(fs for fs in unit.ftseeds if fs in set(admitted))
    if ok or unit.tier != plan.TIER_LR or not admitted:
        return ok
    return (min(admitted),)


def est_gpu_hours(unit: plan.Unit, n_leaves: int) -> float:
    """Plan §15 estimate of one unit's train cost (physical GPU-h)."""
    per_leaf = _LEAF_GPU_H[unit.cell.student]
    if unit.cell.teacher == "pythia-6.9b":
        per_leaf += 0.6
    return per_leaf * n_leaves
