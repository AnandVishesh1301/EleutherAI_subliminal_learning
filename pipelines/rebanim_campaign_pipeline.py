"""Auto-advancing driver of `rebanim`: unpinned queues in the campaign's own root, drained
by the Slurm pools the jobq supervisor submits.

One `run(cfg)` = one idempotent advance pass:

    preflight   plan, roots, paths YAML, staging report, disk floor
    register    the pin can run every job
    (every pass) unfinished queues follow the pin; failures under the requeue cap are
                requeued, the rest are holes; below the disk floor nothing new is
                submitted
    per teacher (kind, animal, base): a reused config is ready at once; a new one trains
                its three ftseeds, its manifests are checked against the plan (step
                count included), then it is ready; every config gets an eval job
    leaves      once a cell's teacher config is ready its three leaves are appended to
                the (kind, animal) leaf queue
    finalize    every leaf submitted and finished, every teacher evaluated, no hole ->
                `<root>/CAMPAIGN_DONE`

No W0 gate, no dose retrain, no tiers. A unit-level contract failure is a hole of that
unit; the others advance. Results are built by the driver CLI.
"""

from __future__ import annotations

import fcntl
import getpass
import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

from loguru import logger

from pipelines import rebanim_staging_pipeline as staging
from pipelines.rebsoft_campaign_pipeline import (
    AdvanceResult,
    CampaignGateError,
    UnitHold,
    _repin,
    _requeue_and_holes,
    _submit,
    current_pin,
    submitted,
    write_yamls,
)
from pipelines.rebsoft_staging_pipeline import STAGING_REPORT, link_pin_to_itself
from subliminal_learning.campaign import driver_io, xfam2_plan
from subliminal_learning.campaign import rebanim_jobs as jobs
from subliminal_learning.campaign import rebanim_plan as plan
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue import store
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import file_sha256
from subliminal_learning.utils.run_io import read_run_set_card, utc_now_iso

# One driver runs, on the login node, so its lock stays off the shared mounts.
PASS_LOCK_PATH = Path(f"/tmp/rebanim_driver_{getpass.getuser()}.lock")


def _pin_paths() -> tuple[str, ...]:
    return (
        "scripts/sweeps/sweep_sdf_teacher.py",
        "scripts/sweeps/sweep_evaluate_sdf_teacher.py",
        "scripts/sweeps/sweep_distill_student_gkd.py",
        "scripts/sweeps/sweep_evaluate_gkd_student.py",
        "scripts/run_rebanim_campaign.py",
        "scripts/stage_rebanim_assets.py",
        plan.STUDENT_TRAINING_CONFIG,
        plan.TEACHER_TRAINING_CONFIG,
        "src/subliminal_learning/campaign/rebanim_plan.py",
        "src/subliminal_learning/campaign/rebanim_jobs.py",
        "pipelines/rebanim_campaign_pipeline.py",
        "pipelines/rebanim_staging_pipeline.py",
        "pipelines/rebanim_results_pipeline.py",
    )


SPEC = replace(
    xfam2_plan.SPEC,
    run_set_id=plan.TEACHER_RUN_SET,
    teacher_run_set_id=plan.TEACHER_RUN_SET,
    root_subdir=SDF_TEACHERS_SUBDIR,
    queue_priorities=plan.QUEUE_PRIORITIES,
    gpu_policy={"cap_per_gpu": plan.CAP_PER_GPU},
    generated_config_dir=str(plan.GENERATED_DIR),
    campaign_state_file=plan.STATE_FILE,
    campaign_done_file=plan.DONE_PATH.name,
    gkd_experiment_config=None,
    extra_pin_paths=_pin_paths(),
)
QUEUES: tuple[str, ...] = tuple(plan.QUEUE_PRIORITIES)


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the rebanim spec."""
    kwargs.setdefault("generated_dir_override", plan.GENERATED_DIR)
    return driver_io.CampaignDriverConfig(spec=SPEC, **kwargs)


def unit_key(kind: str, animal: str, base: str) -> str:
    return f"{kind}|{animal}|{plan.name_of(base)}"


def disk_problems(root: Path = plan.ROOT) -> list[str]:
    free = shutil.disk_usage(root).free / 2**30
    if free < plan.MIN_FREE_GIB:
        return [f"{free:.0f} GiB free < {plan.MIN_FREE_GIB}"]
    return []


# ---- preflight, pin ---------------------------------------------------------------------


def preflight_problems(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    problems = []
    if not (cfg.repo_root / plan.PLAN_PATH).is_file():
        problems.append(f"missing plan {plan.PLAN_PATH}")
    if Path(cfg.queue_root) != plan.QUEUE_ROOT:
        problems.append(f"queue root {cfg.queue_root} is not the plan's")
    if Path(cfg.output_root) != plan.OUTPUT_ROOT:
        problems.append(f"output root {cfg.output_root} is not the plan's")
    if problem := staging.paths_problem(cfg.repo_root / plan.PATHS_CONFIG):
        problems.append(problem)
    report = plan.STAGING_DIR / STAGING_REPORT
    if not report.is_file() or json.loads(report.read_text()).get("passed") is not True:
        problems.append(f"no passed staging report at {report}")
    if plan.ROOT.is_dir():
        problems += disk_problems()
    return problems


def pin_problem(cfg: driver_io.CampaignDriverConfig, state: dict) -> str | None:
    """Why the pass must not run under this pin, or None."""
    resolve = code_pin_io.resolve_commit_sha
    card = read_run_set_card(cfg.run_set_root) or {}
    mine = card.get("git_sha") or cfg.code_sha
    if not mine:
        return "pass --code-sha until the teacher run-set card exists"
    mine = resolve(cfg.repo_root, str(mine))
    registered = (state["transitions"].get("register") or {}).get("git_sha")
    if not card and registered and registered != mine and not cfg.auto_reconcile_sha:
        return (
            f"--code-sha {mine[:12]} differs from the registered pin {registered[:12]}"
        )
    root = cfg.output_root / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.STUDENT_RUN_SET)
    sha = (read_run_set_card(root) or {}).get("git_sha")
    if sha and resolve(cfg.repo_root, str(sha)) != mine:
        return (
            f"{plan.STUDENT_RUN_SET} pinned to {str(sha)[:12]}, teachers to {mine[:12]}"
        )
    return None


def _fire(state: dict, name: str, **info: object) -> None:
    state["transitions"][name] = {"at": utc_now_iso(), **info}


# ---- teachers ---------------------------------------------------------------------------


def verify_teachers(ref: jobs.TeacherRef) -> int:
    """Check every new ftseed manifest against the config; return the highest step count.

    Raises:
        UnitHold: A manifest is missing or off-plan.
    """
    try:
        docs_sha = file_sha256(ref.docs_path)
        steps = []
        for r in plan.TEACHER_FTSEEDS:
            manifest = json.loads(ref.manifest_path(r).read_text())
            if bad := jobs.teacher_manifest_problems(ref, r, manifest, docs_sha):
                raise ValueError(f"ftseed{r} manifest off-plan {bad}")
            steps.append(int(manifest["n_train_steps"]))
    except (OSError, KeyError, ValueError) as err:
        raise UnitHold(f"teachers {ref.param_dir}: {err}") from err
    return max(steps)


def advance_teacher(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    ref: jobs.TeacherRef,
    waiting: list[str],
) -> tuple[list[str], bool]:
    """Submit one teacher config's jobs; return notes and whether students may use it."""
    us = state.setdefault("teachers", {}).setdefault(
        unit_key(ref.kind, ref.animal, ref.base), {}
    )
    write_yamls(cfg, jobs.teacher_yaml_docs(ref))
    notes: list[str] = []
    train_q = plan.q_teach(ref.kind, ref.animal, ref.base)
    if ref.reused is None and (
        n := _submit(cfg, train_q, jobs.teacher_train_jobs(ref))
    ):
        notes.append(f"submit({train_q}, +{n})")
    eval_q = plan.q_teval(ref.kind, ref.animal, ref.base)
    deps = plan.teval_depends_on(ref.kind, ref.animal, ref.base)
    if _submit(cfg, eval_q, [jobs.teacher_eval_job(ref)], depends_on=deps, strict=True):
        notes.append(f"submit({eval_q})")
    if ref.reused is not None or "verified" in us:
        return notes, True
    finished, msg = driver_io.queue_finished(cfg, train_q)
    if not finished:
        waiting.append(msg)
        return notes, False
    us["verified"] = {"at": utc_now_iso(), "steps": verify_teachers(ref)}
    notes.append(f"teachers({unit_key(ref.kind, ref.animal, ref.base)} verified)")
    return notes, True


def submit_cell_leaves(
    cfg: driver_io.CampaignDriverConfig, state: dict, cell: plan.Cell, kind: str
) -> list[str]:
    """Every leaf of one cell and kind, once (missing jobs added)."""
    done = state.setdefault("leaves_submitted", {})
    key = f"{kind}|{cell.animal}|{cell.teacher}|{cell.student}"
    if key in done:
        return []
    leaves = jobs.cell_leaves(cell, kind)
    for leaf in leaves:
        write_yamls(cfg, jobs.leaf_yaml_docs(leaf))
    name = leaves[0].queue
    n = _submit(cfg, name, [jobs.leaf_job(leaf) for leaf in leaves])
    done[key] = utc_now_iso()
    return [f"submit({name}, +{n})"] if n else []


def _unit_pass(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    ref: jobs.TeacherRef,
    waiting: list[str],
) -> list[str]:
    """Advance one teacher unit and its cells, a failure recorded as that unit's hole."""
    holes = state.setdefault("holes", {})
    name = f"unit:{unit_key(ref.kind, ref.animal, ref.base)}"
    try:
        notes, ready = advance_teacher(cfg, state, ref, waiting)
        if ready:
            for cell in plan.cells():
                if (cell.animal, cell.teacher) == (ref.animal, ref.base):
                    notes += submit_cell_leaves(cfg, state, cell, ref.kind)
    except CampaignGateError:
        raise
    except (UnitHold, OSError, ValueError, KeyError, RuntimeError) as err:
        msg = str(err) if isinstance(err, UnitHold) else f"{type(err).__name__}: {err}"
        if holes.get(name) != msg:
            logger.opt(exception=not isinstance(err, UnitHold)).error(
                "hole in {}: {}", name, msg
            )
        holes[name] = msg
        return []
    holes.pop(name, None)
    return notes


def campaign_finished(cfg: driver_io.CampaignDriverConfig, state: dict) -> bool:
    """Every leaf submitted, and every leaf and teacher-eval queue finished."""
    done = state.get("leaves_submitted", {})
    if len(done) != len(plan.cells()) * len(plan.KINDS):
        return False
    names = [plan.q_leaves(k, a) for k in plan.KINDS for a in plan.ANIMALS] + [
        plan.q_teval(*u) for u in plan.teacher_units()
    ]
    return all(driver_io.queue_finished(cfg, n)[0] for n in names)


# ---- dry run ----------------------------------------------------------------------------


def planned_summary() -> dict:
    """Job counts per phase, built from the plan alone (every job is constructed)."""
    refs = [jobs.TeacherRef(*u) for u in plan.teacher_units()]
    new = [r for r in refs if r.reused is None]
    teach = Counter(
        f"{r.kind}|{r.animal}" for r in new for _ in jobs.teacher_train_jobs(r)
    )
    leaves = jobs.all_leaves()
    for leaf in leaves:
        jobs.leaf_job(leaf)
    return {
        "teacher_configs": len(refs),
        "teacher_configs_reused": {
            k: sum(1 for r in refs if r.reused and r.kind == k) for k in plan.KINDS
        },
        "teacher_train_jobs": dict(sorted(teach.items())),
        "teacher_train_jobs_total": sum(teach.values()),
        "teacher_eval_jobs": len([jobs.teacher_eval_job(r) for r in refs]),
        "cells": len(plan.cells()),
        "leaf_jobs": dict(
            sorted(Counter(f"{lf.kind}|{lf.animal}" for lf in leaves).items())
        ),
        "leaf_jobs_total": len(leaves),
        "queues": len(QUEUES),
        "mem_mib": {
            "teacher": plan.TEACHER_MEM_MIB,
            "teacher_eval": plan.TEACHER_EVAL_MEM_MIB,
            "student": plan.STUDENT_MEM_MIB,
        },
        "cap_per_gpu": plan.CAP_PER_GPU,
    }


# ---- the pass ---------------------------------------------------------------------------


def run(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    """One advance pass, under an exclusive flock so two passes never interleave."""
    if cfg.dry_run:
        return _pass(cfg)
    with PASS_LOCK_PATH.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return _pass(cfg)


def _pass(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    state = driver_io.load_state(cfg)
    done = state["transitions"]
    fired: list[str] = []
    waiting: list[str] = []
    if cfg.dry_run:
        for p in preflight_problems(cfg):
            waiting.append(f"[dry-run] preflight would refuse: {p}")
        sha = code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha or "HEAD")
        waiting.append(f"[dry-run] would register pin {sha[:12]}; nothing submitted")
        return AdvanceResult(tuple(fired), tuple(waiting), False)
    if bad := pin_problem(cfg, state):
        raise CampaignGateError(bad)
    driver_io.ensure_campaign_code_pin(cfg)
    pin = current_pin(cfg)
    link_pin_to_itself(pin.root, pin.sha)
    holes = state.setdefault("holes", {})
    labelled = state.setdefault("labelled", {})

    def mark(name: str) -> None:
        fired.append(name)
        driver_io.save_state(cfg, state)

    if "preflight" not in done:
        if problems := preflight_problems(cfg):
            raise CampaignGateError("; ".join(problems))
        _fire(state, "preflight")
        mark("preflight")
    if "register" not in done:
        driver_io.verify_pin_runs_campaign(pin, SPEC)
        _fire(state, "register", git_sha=pin.sha)
        mark("register")
    if note := _repin(cfg, QUEUES):
        fired.append(note)
    _requeue_and_holes(cfg, state, QUEUES)
    if disk := disk_problems():
        labelled["disk_low"] = "; ".join(disk)
        waiting.append(f"new submissions held: {labelled['disk_low']}")
    else:
        labelled.pop("disk_low", None)
        for unit in plan.teacher_units():
            fired.extend(_unit_pass(cfg, state, jobs.TeacherRef(*unit), waiting))
            driver_io.save_state(cfg, state)
    all_done = False
    if campaign_finished(cfg, state) and not holes:
        if "finalize" not in done:
            for marker in (plan.DONE_PATH, cfg.logs_dir / SPEC.campaign_done_file):
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(f"{utc_now_iso()}\n")
            _fire(state, "finalize", labelled=sorted(labelled))
            mark("finalize")
        all_done = True
    elif holes:
        waiting.append(f"holes in {sorted(holes)}")
    state["last_pass"] = {"at": utc_now_iso(), "fired": fired, "waiting": waiting[:50]}
    driver_io.save_state(cfg, state)
    plan.HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.HEARTBEAT_PATH.touch()
    return AdvanceResult(tuple(fired), tuple(waiting), all_done)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Transitions, holes, labels and queue counts (file reads)."""
    state = driver_io.load_state(cfg)
    totals = {"done": 0, "failed": 0, "pending": 0, "running": 0, "total": 0}
    for name in submitted(cfg, QUEUES):
        st = store.queue_status(cfg.queue_root, name)
        for k in totals:
            totals[k] += int(st[k])
    return {
        "transitions": sorted(state["transitions"]),
        "holes": state.get("holes", {}),
        "labelled": state.get("labelled", {}),
        "queues": len(submitted(cfg, QUEUES)),
        "jobs": totals,
        "leaves_submitted": len(state.get("leaves_submitted", {})),
        "last_pass": state.get("last_pass"),
    }
