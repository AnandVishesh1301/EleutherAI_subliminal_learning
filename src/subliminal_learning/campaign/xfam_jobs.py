"""jobq job builders for the cross-family OPD campaign.

One job per sweep shard. Shards are the finest disjoint partition the sweep CLI
exposes -- (LR, teacher ftseed) for a student YAML, (ftseed) for a teacher YAML -- so
the pool load-balances and a crashed shard costs one cell, not a whole ladder.

`mem_mib` / `slots` come from the plan's per-label footprints; `cwd` and env come from
the queue defaults (the campaign code pin), not from the job.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from subliminal_learning.campaign import xfam_config_gen as cgen
from subliminal_learning.campaign import xfam_models as models
from subliminal_learning.campaign import xfam_plan as plan
from subliminal_learning.campaign.ckptgrid_jobs import symlink_action
from subliminal_learning.campaign.driver_io import eval_cmd, generated_rel, train_cmd
from subliminal_learning.gkd.layout import (
    GKD_STUDENTS_SUBDIR,
    teacher_checkpoint_dir,
)
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir

# Tokens that would name a model this campaign must never touch: the PolyPythias
# seed/ablation repos and the untrained step-0 branches.
_FORBIDDEN = ("weight-seed", "data-seed", "step0", "-seed1", "-seed2", "-seed3")

_PATHS_FLAG = f"--paths-config {plan.PATHS_CONFIG}"
_REGISTRY_FLAG = "--spec-registry xfam"


def _assert_clean_cmd(cmd: str) -> str:
    """Raise if a command names a model outside this campaign's roster.

    Raises:
        ValueError: On a forbidden token.
    """
    for token in _FORBIDDEN:
        if token in cmd:
            raise ValueError(f"refusing job command naming {token!r}: {cmd}")
    return cmd


def _lr_slug(lr: float) -> str:
    mantissa, exp = f"{lr:.1e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exp)}"


def _control_flag(cls: str) -> str:
    if cls == cgen.TREATMENT_CLASS or cls == cgen.ANCHOR_CLASS:
        return "--no-control"
    if cls == cgen.SB_CLASS:
        return "--controls-only"
    if cls == cgen.TB_CLASS:
        return "--xbase-controls-only"
    raise ValueError(f"unknown student YAML class {cls!r}")


def _wandb_flag(
    *, arm_name: str, student: str, lr: float, recipe_slug: str, cls: str, ftseed: int
) -> str:
    """``--no-wandb`` unless this shard holds the one logged cell of its YAML."""
    if cls in (cgen.SB_CLASS, cgen.TB_CLASS):
        return "--no-wandb"
    a = plan.arm(arm_name)
    recipe = next(
        (r for r in a.recipes if cgen.recipe_slug(r) == recipe_slug), a.recipes[0]
    )
    logs = any(
        plan.student_logs_wandb(
            arm_name=arm_name,
            teacher=a.teacher,
            student=student,
            lr=lr,
            recipe=recipe,
            epochs=a.epochs,
            ftseed=ftseed,
            gseed=gseed,
        )
        for gseed in plan.GSEEDS
    )
    return "" if logs else "--no-wandb"


def student_jobs(
    train_configs: Mapping[str, dict],
    arm_name: str,
    student: str,
    cls: str,
    picks: Mapping[str, float],
) -> list[Job]:
    """Train shards of one ``(arm, student, class)``: one job per (LR, ftseed).

    Raises:
        ValueError: If the generated YAML for the class is missing.
    """
    a = plan.arm(arm_name)
    lrs = a.lrs_for(student, dict(picks))
    fp = plan.student_footprint(student)
    cluster = a.cluster
    jobs: list[Job] = []
    for recipe in a.recipes:
        fname = cgen.student_fname(arm_name, student, recipe, a.epochs, cls)
        if fname not in train_configs:
            raise ValueError(f"no generated YAML {fname!r} for {arm_name}/{student}")
        rslug = cgen.recipe_slug(recipe)
        for lr in lrs:
            for ftseed in a.ftseeds:
                flags = [
                    _control_flag(cls),
                    _PATHS_FLAG,
                    _REGISTRY_FLAG,
                    f"--animals {plan.TARGET_ANIMAL}",
                    f"--lora-lrs {lr}",
                    f"--teacher-ftseeds {ftseed}",
                    _wandb_flag(
                        arm_name=arm_name,
                        student=student,
                        lr=lr,
                        recipe_slug=rslug,
                        cls=cls,
                        ftseed=ftseed,
                    ),
                ]
                cmd = train_cmd(
                    generated_rel(fname, spec=plan.SPEC),
                    *[f for f in flags if f],
                    experiment_config=plan.STUDENT_EXPERIMENT_CONFIG[cluster],
                )
                cmd = f"{cmd} --training-config {plan.STUDENT_TRAINING_CONFIG}"
                cmd = f"{cmd} --run-set-id {plan.RUN_SET_ID}"
                cmd = f"{cmd} --teacher-run-set-id {plan.TEACHER_RUN_SET_ID}"
                jobs.append(
                    Job(
                        key=(
                            f"{arm_name}|{cls}|{student}|{rslug}|"
                            f"{cgen.epochs_slug(a.epochs)}|lr{_lr_slug(lr)}|t{ftseed}"
                        ),
                        cmd=_assert_clean_cmd(cmd),
                        mem_mib=fp.mem_mib,
                        slots=fp.slots,
                    )
                )
    return jobs


def unit_train_jobs(
    train_configs: Mapping[str, dict],
    arm_name: str,
    student: str,
    picks: Mapping[str, float],
) -> tuple[list[Job], list[Job]]:
    """``(treatment jobs, control jobs)`` of one submission unit."""
    a = plan.arm(arm_name)
    treatment_cls = cgen.ANCHOR_CLASS if a.teacher == student else cgen.TREATMENT_CLASS
    treatments = student_jobs(train_configs, arm_name, student, treatment_cls, picks)
    if not a.emits_controls:
        return treatments, []
    controls = student_jobs(train_configs, arm_name, student, cgen.SB_CLASS, picks)
    if a.teacher != student:
        controls += student_jobs(train_configs, arm_name, student, cgen.TB_CLASS, picks)
    return treatments, controls


def unit_eval_jobs(
    train_configs: Mapping[str, dict],
    arm_name: str,
    student: str,
    picks: Mapping[str, float],
) -> list[Job]:
    """Verify shards of one submission unit: one job per (class, LR, ftseed)."""
    a = plan.arm(arm_name)
    lrs = a.lrs_for(student, dict(picks))
    treatment_cls = cgen.ANCHOR_CLASS if a.teacher == student else cgen.TREATMENT_CLASS
    classes = [treatment_cls]
    if a.emits_controls:
        classes.append(cgen.SB_CLASS)
        if a.teacher != student:
            classes.append(cgen.TB_CLASS)
    jobs: list[Job] = []
    for cls in classes:
        for recipe in a.recipes:
            train_name = cgen.student_fname(arm_name, student, recipe, a.epochs, cls)
            if train_name not in train_configs:
                raise ValueError(f"no generated YAML {train_name!r}")
            vname = cgen.verify_name(train_name)
            rslug = cgen.recipe_slug(recipe)
            for lr in lrs:
                for ftseed in a.ftseeds:
                    flags = [
                        _control_flag(cls),
                        _PATHS_FLAG,
                        _REGISTRY_FLAG,
                        f"--animals {plan.TARGET_ANIMAL}",
                        f"--lora-lrs {lr}",
                        f"--teacher-ftseeds {ftseed}",
                        f"--run-set-id {plan.RUN_SET_ID}",
                    ]
                    cmd = eval_cmd(
                        generated_rel(vname, spec=plan.SPEC),
                        *[f for f in flags if f],
                    )
                    cmd = (
                        f"{cmd} --experiment-config "
                        f"{plan.STUDENT_EXPERIMENT_CONFIG[a.cluster]}"
                    )
                    jobs.append(
                        Job(
                            key=(
                                f"eval|{arm_name}|{cls}|{student}|{rslug}|"
                                f"{cgen.epochs_slug(a.epochs)}|"
                                f"lr{_lr_slug(lr)}|t{ftseed}"
                            ),
                            cmd=_assert_clean_cmd(cmd),
                            mem_mib=plan.student_footprint(student).mem_mib // 2,
                            slots=1,
                        )
                    )
    return jobs


def teacher_jobs(train_configs: Mapping[str, dict]) -> list[Job]:
    """One SDF teacher train shard per ``(label, param_tag, ftseed)``."""
    jobs: list[Job] = []
    for label, tag, ftseed in plan.teacher_cells():
        fname = cgen.teacher_fname(label, tag)
        if fname not in train_configs:
            raise ValueError(f"no generated teacher YAML {fname!r}")
        spec = models.spec(label)
        fp = plan.teacher_footprint(label)
        cmd = (
            f"uv run python -m scripts.sweeps.sweep_sdf_teacher "
            f"--sweep-config {generated_rel(fname, spec=plan.SPEC)} "
            f"--device cuda:0 {_PATHS_FLAG} {_REGISTRY_FLAG} "
            f"--training-config {plan.TEACHER_TRAINING_CONFIG[label]} "
            f"--experiment-config {plan.TEACHER_EXPERIMENT_CONFIG[spec.cluster]} "
            f"--animals {plan.TARGET_ANIMAL} --base-models {label} "
            f"--ftseeds {ftseed} --run-set-id {plan.RUN_SET_ID} --no-wandb"
        )
        jobs.append(
            Job(
                key=f"teacher|{label}|{tag}|t{ftseed}",
                cmd=_assert_clean_cmd(cmd),
                mem_mib=fp.mem_mib,
                slots=fp.slots,
            )
        )
    return jobs


def teacher_eval_jobs(train_configs: Mapping[str, dict]) -> list[Job]:
    """One SDF teacher verify shard per ``(label, param_tag, ftseed)``."""
    jobs: list[Job] = []
    for label, tag, ftseed in plan.teacher_cells():
        train_name = cgen.teacher_fname(label, tag)
        if train_name not in train_configs:
            raise ValueError(f"no generated teacher YAML {train_name!r}")
        spec = models.spec(label)
        cmd = (
            f"uv run python -m scripts.sweeps.sweep_evaluate_sdf_teacher "
            f"--sweep-config {generated_rel(cgen.verify_name(train_name), spec=plan.SPEC)} "
            f"--device cuda:0 {_PATHS_FLAG} {_REGISTRY_FLAG} "
            f"--experiment-config {plan.TEACHER_EXPERIMENT_CONFIG[spec.cluster]} "
            f"--animals {plan.TARGET_ANIMAL} --base-models {label} "
            f"--ftseeds {ftseed} --run-set-id {plan.RUN_SET_ID}"
        )
        jobs.append(
            Job(
                key=f"teacher-eval|{label}|{tag}|t{ftseed}",
                cmd=_assert_clean_cmd(cmd),
                mem_mib=plan.teacher_footprint(label).mem_mib,
                slots=1,
            )
        )
    return jobs


@dataclass(frozen=True)
class XfamRef:
    """One ``(repo_id, revision)`` the campaign loads from the Hub.

    Not `ckptgrid_jobs.HubRef`: that one keys on a PolyPythias pretraining step, which
    a roster entry does not have -- an xfam revision is a commit sha.
    """

    label: str
    repo_id: str
    revision: str

    def to_dict(self) -> dict:
        """JSON record for the prefetch manifest."""
        return {
            "label": self.label,
            "repo": self.repo_id,
            "revision": self.revision,
        }


def prefetch_refs() -> tuple[XfamRef, ...]:
    """Every Hub snapshot the campaign loads, at its pinned commit."""
    refs = [
        XfamRef(spec.label, spec.repo_id, str(spec.revision))
        for spec in models.ROSTER.values()
    ]
    # The two Pythia step-0 branches back the shared-init gate in staging.
    refs.append(XfamRef("pythia-410m@step0", "EleutherAI/pythia-410m", "step0"))
    refs.append(
        XfamRef(
            "pythia-410m-deduped@step0", "EleutherAI/pythia-410m-deduped", "step0"
        )
    )
    return tuple(refs)


def cold_symlink_plan(output_root: Path, cold_root: Path) -> tuple[tuple[Path, Path], ...]:
    """``(link, target)`` pairs putting the run-set trees on cold storage."""
    pairs: list[tuple[Path, Path]] = []
    for subdir in (GKD_STUDENTS_SUBDIR, SDF_TEACHERS_SUBDIR):
        rel = run_set_subdir(subdir, plan.RUN_SET_ID)
        pairs.append((output_root / rel, cold_root / rel))
    return tuple(pairs)


def teacher_checkpoint_paths(
    output_root: Path,
) -> tuple[tuple[str, str, int, Path], ...]:
    """``(label, tag, ftseed, checkpoint_dir)`` for every teacher this campaign trains."""
    subdir = run_set_subdir(SDF_TEACHERS_SUBDIR, plan.TEACHER_RUN_SET_ID)
    return tuple(
        (
            label,
            tag,
            ftseed,
            teacher_checkpoint_dir(
                output_root,
                subdir,
                plan.TARGET_ANIMAL,
                models.spec(label),
                plan.HUB_ORG,
                tag,
                ftseed,
            ),
        )
        for label, tag, ftseed in plan.teacher_cells()
    )


def symlink_actions(
    output_root: Path, cold_root: Path
) -> tuple[tuple[Path, Path, str], ...]:
    """``(link, target, action)`` for each planned cold symlink."""
    return tuple(
        (link, target, symlink_action(link, target))
        for link, target in cold_symlink_plan(output_root, cold_root)
    )


def all_train_jobs(
    train_configs: Mapping[str, dict], picks: Mapping[str, float]
) -> dict[str, list[Job]]:
    """Every queue this campaign would submit, keyed by queue name."""
    queues: dict[str, list[Job]] = {
        plan.Q_TEACHERS: teacher_jobs(train_configs),
        plan.Q_TEACHER_EVALS: teacher_eval_jobs(train_configs),
    }
    for arm_name, student in plan.all_units():
        treatments, controls = unit_train_jobs(
            train_configs, arm_name, student, picks
        )
        queues[plan.q_students(arm_name, student)] = treatments
        if controls:
            queues[plan.q_controls(arm_name, student)] = controls
        queues[plan.q_evals(arm_name, student)] = unit_eval_jobs(
            train_configs, arm_name, student, picks
        )
    return queues


def queue_job_counts(
    train_configs: Mapping[str, dict], picks: Mapping[str, float]
) -> dict[str, int]:
    """Job count per queue (budget reporting and the preflight report)."""
    return {name: len(jobs) for name, jobs in all_train_jobs(train_configs, picks).items()}


def est_card_hours(
    train_configs: Mapping[str, dict],
    picks: Mapping[str, float],
    *,
    include_tail: bool = True,
) -> float:
    """Estimated card-hours, at the plan's `cap_per_gpu` packing."""
    cap = int(plan.GPU_POLICY["cap_per_gpu"])  # type: ignore[arg-type]
    tail_queues = plan.tail_queue_names()
    total = 0.0
    for name, jobs in all_train_jobs(train_configs, picks).items():
        if not include_tail and name in tail_queues:
            continue
        minutes = _queue_minutes(name, len(jobs))
        total += minutes / 60.0 / cap
    return total


def _queue_minutes(name: str, n_jobs: int) -> float:
    if name == plan.Q_TEACHERS:
        return sum(
            plan.teacher_footprint(label).est_minutes
            for label, _, _ in plan.teacher_cells()
        )
    if name == plan.Q_TEACHER_EVALS:
        return n_jobs * plan.EVAL_EST_MINUTES
    cells = student_cells_per_job()
    for arm_name, student in plan.all_units():
        if name == plan.q_evals(arm_name, student):
            return n_jobs * cells * plan.EVAL_EST_MINUTES
        if name in (
            plan.q_students(arm_name, student),
            plan.q_controls(arm_name, student),
        ):
            return n_jobs * cells * plan.student_footprint(student).est_minutes
    raise ValueError(f"unknown queue {name!r}")


def student_cells_per_job() -> int:
    """Cells one student train shard produces (one per gseed)."""
    return len(plan.GSEEDS)


def all_queue_names(
    train_configs: Mapping[str, dict], picks: Mapping[str, float]
) -> tuple[str, ...]:
    """Queue names in submission order."""
    return tuple(all_train_jobs(train_configs, picks))


def jobkeys(jobs: Sequence[Job]) -> tuple[str, ...]:
    """Sanitized jobkeys of `jobs`."""
    return tuple(job.jobkey for job in jobs)
