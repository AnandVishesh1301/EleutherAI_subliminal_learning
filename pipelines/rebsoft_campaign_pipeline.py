"""Auto-advancing driver of `rebsoft`: unpinned queues in the campaign's own root, drained
by the Slurm pools the jobq supervisor submits.

One `run(cfg)` = one idempotent advance pass:

    preflight   prereg, roots, paths YAML, staging report, disk floor
    register    the pin can run every job
    (every pass) unfinished queues follow the pin; failures under the requeue cap are
                requeued, the rest are holes; below the disk floor nothing new is
                submitted
    per unit    (target, teacher base), a treatment chain and a placebo chain:
                reused teachers are ready at once; a new treatment teacher trains at ep5
                -> eval -> W0, a failure retrains at ep8, and a double failure proceeds
                at ep8 labelled `w0_fail|<unit>`; its manifests fix the step count ->
                treatment leaves + placebo teachers at that count -> placebo leaves
    tiers       a block's work opens with its priority; P3 opens once P1 and P2 have
                nothing pending; a `P<n>_DONE` marker follows each finished priority
    finalize    every priority finished with no hole -> the campaign done marker

A unit-level contract failure (unreadable W0 inputs, off-plan manifest) is a hole of
that unit; the others advance. Results are built per priority by the driver CLI.

Re-pin: stop the driver, run `scripts.reconcile_run_set` on all four run sets, each with
`--dry-run` first, then restart the driver with the new `--code-sha`: a driver left on
the old sha raises once the card moves. Never use `--auto-reconcile-sha`, which advances
only the teacher card. Every pass links the current pin to itself before anything is
submitted.
"""

from __future__ import annotations

import fcntl
import getpass
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import yaml
from loguru import logger

from pipelines import rebsoft_staging_pipeline as staging
from pipelines.traitfree_staging_pipeline import write_once_text
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import newdom_soft as soft
from subliminal_learning.campaign import rebsoft_jobs as jobs
from subliminal_learning.campaign import rebsoft_plan as plan
from subliminal_learning.campaign import xfam2_plan
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import file_sha256
from subliminal_learning.utils.run_io import read_run_set_card, utc_now_iso

# One driver runs, on the login node, so its lock stays off the shared mounts.
PASS_LOCK_PATH = Path(f"/tmp/rebsoft_driver_{getpass.getuser()}.lock")


def _pin_paths() -> tuple[str, ...]:
    return (
        "scripts/sweeps/sweep_sdf_teacher.py",
        "scripts/sweeps/sweep_evaluate_sdf_teacher.py",
        "scripts/sweeps/sweep_distill_student_gkd.py",
        "scripts/sweeps/sweep_evaluate_gkd_student.py",
        "scripts/run_rebsoft_campaign.py",
        "scripts/stage_rebsoft_assets.py",
        "scripts/rebsoft_panel.py",
        jobs.STUDENT_TRAINING_CONFIG,
        "configs/training/lora_sdf_teacher.yaml",
        "configs/training/lora_sdf_teacher_olmo.yaml",
        "src/subliminal_learning/campaign/rebsoft_plan.py",
        "src/subliminal_learning/campaign/rebsoft_jobs.py",
        "src/subliminal_learning/campaign/rebsoft_metrics.py",
        "pipelines/rebsoft_campaign_pipeline.py",
        "pipelines/rebsoft_results_pipeline.py",
        "pipelines/rebsoft_staging_pipeline.py",
        "pipelines/rebsoft_panel_pipeline.py",
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
    campaign_done_file=plan.DONE_FILE,
    gkd_experiment_config=None,
    extra_pin_paths=_pin_paths(),
)


class CampaignGateError(RuntimeError):
    """A campaign-level gate refused to pass."""


class UnitHold(RuntimeError):
    """A unit-level contract failed (a hole of that unit)."""


@dataclass(frozen=True)
class AdvanceResult:
    """Outcome of one advance pass."""

    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the rebsoft spec."""
    kwargs.setdefault("generated_dir_override", plan.GENERATED_DIR)
    return driver_io.CampaignDriverConfig(spec=SPEC, **kwargs)


def _fire(state: dict, name: str, **info: object) -> None:
    state["transitions"][name] = {"at": utc_now_iso(), **info}


def unit_key(target: str, base: str) -> str:
    return f"{target}|{base}"


def current_pin(cfg: driver_io.CampaignDriverConfig) -> code_pin_io.CodePin:
    return code_pin_io.resolve_campaign_pin(
        cfg.run_set_root, repo_root=cfg.repo_root, requested_sha=cfg.code_sha
    )


def queue_defaults(cfg: driver_io.CampaignDriverConfig) -> dict:
    """jobq defaults of an unpinned queue: the campaign pin's worktree."""
    if cfg.dry_run:
        return {"cwd": str(cfg.repo_root / code_pin_io.CODE_PINS_DIRNAME)}
    return code_pin_io.pin_job_defaults(current_pin(cfg))


# ---- queues -----------------------------------------------------------------------------


def submitted(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    """Every plan queue that exists."""
    return [n for n in plan.QUEUE_PRIORITIES if store.queue_exists(cfg.queue_root, n)]


def missing_jobs(
    cfg: driver_io.CampaignDriverConfig, name: str, queue_jobs: Sequence[Job]
) -> list[Job]:
    """The jobs of `queue_jobs` queue `name` does not hold (all when it does not exist)."""
    if not store.queue_exists(cfg.queue_root, name):
        return list(queue_jobs)
    have = {j.jobkey for j in store.load_jobs(cfg.queue_root, name)}
    return [j for j in queue_jobs if j.jobkey not in have]


def _submit(
    cfg: driver_io.CampaignDriverConfig,
    name: str,
    queue_jobs: list[Job],
    *,
    depends_on: Sequence[str] = (),
    strict: bool = False,
) -> int:
    """Create unpinned `name`, or append the jobs it lacks; return the number added."""
    if not store.queue_exists(cfg.queue_root, name):
        driver_io.submit_queue(
            cfg,
            name,
            queue_jobs,
            depends_on=list(depends_on) or None,
            defaults=queue_defaults(cfg),
            strict_deps=strict,
            unpinned=True,
        )
        return len(queue_jobs)
    new = missing_jobs(cfg, name, queue_jobs)
    if new and not cfg.dry_run:
        store.update_queue_defaults(cfg.queue_root, name, queue_defaults(cfg))
        store.append_jobs(cfg.queue_root, name, new)
        logger.info("appended {} job(s) to {}", len(new), name)
    return len(new)


def write_yamls(cfg: driver_io.CampaignDriverConfig, docs: Mapping[Path, dict]) -> int:
    """Create (or match) generated YAMLs."""
    if not cfg.dry_run:
        for path, doc in docs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            write_once_text(path, yaml.safe_dump(doc, sort_keys=True))
    return len(docs)


def _repin(cfg: driver_io.CampaignDriverConfig) -> str | None:
    """Move every queue with work left (or failures) to the campaign pin."""
    if cfg.dry_run:
        return None
    want = queue_defaults(cfg)
    n = 0
    for name in submitted(cfg):
        if (
            store.queue_complete(cfg.queue_root, name)
            and not store.queue_status(cfg.queue_root, name)["failed"]
        ):
            continue
        if store.read_meta(cfg.queue_root, name).defaults.get("cwd") != want["cwd"]:
            store.update_queue_defaults(cfg.queue_root, name, want)
            n += 1
    return f"repin(queues={n})" if n else None


def _requeue_and_holes(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    holes = state.setdefault("holes", {})
    for name in submitted(cfg):
        try:
            driver_io.requeue_failed_under_cap(cfg, name)
        except RuntimeError as err:
            if holes.get(name) != str(err):
                logger.error("hole in {}: {}", name, err)
            holes[name] = str(err)
        else:
            holes.pop(name, None)


def disk_problems(root: Path = plan.ROOT) -> list[str]:
    free = shutil.disk_usage(root).free / 2**30
    if free < plan.MIN_FREE_GIB:
        return [f"{free:.0f} GiB free < {plan.MIN_FREE_GIB}"]
    return []


# ---- preflight, pin ---------------------------------------------------------------------


def preflight_problems(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    problems = []
    if not (cfg.repo_root / plan.PREREG_PATH).is_file():
        problems.append(f"missing pre-registration {plan.PREREG_PATH}")
    if Path(cfg.queue_root).resolve() != plan.QUEUE_ROOT.resolve():
        problems.append(f"queue root {cfg.queue_root} is not the plan's")
    if Path(cfg.output_root) != plan.OUTPUT_ROOT:
        problems.append(f"output root {cfg.output_root} is not the plan's")
    if problem := staging.paths_problem(cfg.repo_root / plan.PATHS_CONFIG):
        problems.append(problem)
    report = plan.STAGING_DIR / staging.STAGING_REPORT
    if not report.is_file() or json.loads(report.read_text()).get("passed") is not True:
        problems.append(f"no passed staging report at {report}")
    return problems + disk_problems()


def pin_problem(cfg: driver_io.CampaignDriverConfig, state: dict) -> str | None:
    """Why the pass must not run under this pin, or None.

    Before the teacher card exists the pin is `--code-sha`, which must match the
    registered one unless `auto_reconcile_sha`; the student run-set cards, once they
    exist, must name the same pin.
    """
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
    for run_set in plan.STUDENT_RUN_SETS.values():
        root = cfg.output_root / run_set_subdir(GKD_STUDENTS_SUBDIR, run_set)
        sha = (read_run_set_card(root) or {}).get("git_sha")
        if sha and resolve(cfg.repo_root, str(sha)) != mine:
            return f"{run_set} pinned to {str(sha)[:12]}, teachers to {mine[:12]}"
    return None


def _preflight(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    problems = preflight_problems(cfg)
    if problems and not cfg.dry_run:
        raise CampaignGateError("; ".join(problems))
    for p in problems:
        logger.warning("[dry-run] preflight would refuse: {}", p)
    _fire(state, "preflight")


def _register(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    if cfg.dry_run:
        sha = code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha or "HEAD")
        _fire(state, "register", git_sha=sha)
        return
    pin = current_pin(cfg)
    driver_io.verify_pin_runs_campaign(pin, SPEC)
    _fire(state, "register", git_sha=pin.sha)


# ---- teachers ---------------------------------------------------------------------------


def submit_teacher(
    cfg: driver_io.CampaignDriverConfig,
    ref: jobs.TeacherRef,
    train_queue: str | None,
    ftseeds: Sequence[int],
    eval_queue: str | None,
) -> list[str]:
    """`ref`'s train jobs and eval job (either may be absent); missing jobs added."""
    notes = []
    write_yamls(cfg, jobs.teacher_yaml_docs(ref))
    if train_queue is not None and ftseeds:
        if n := _submit(cfg, train_queue, jobs.teacher_train_jobs(ref, ftseeds)):
            notes.append(f"submit({train_queue}, +{n})")
    if eval_queue is not None:
        deps = [train_queue] if train_queue is not None else []
        if _submit(
            cfg, eval_queue, [jobs.teacher_eval_job(ref)], depends_on=deps, strict=True
        ):
            notes.append(f"submit({eval_queue})")
    return notes


def read_w0(ref: jobs.TeacherRef) -> soft.W0Verdict:
    """W0 of one evaluated treatment config on the validation set.

    Raises:
        UnitHold: Missing or inconsistent eval artifacts.
    """
    name = plan.eval_set_name(ref.domain, plan.GATE_SET)
    try:
        summary = json.loads((ref.param_dir / "preference_summary.json").read_text())
        base = json.loads((ref.param_dir.parent / "base_control.json").read_text())[
            name
        ]
        entries = [
            e
            for e in summary["summaries"]
            if e["prompt_set"] == name and e["scoring"] == "prob"
        ]
        if summary.get("target_animal") != ref.target or len(entries) != 1:
            raise ValueError(f"no single {name} summary for {ref.target}")
        if base.get("target_animal") != ref.target:
            raise ValueError(f"base control names {base.get('target_animal')!r}")
        per_seed = entries[0]["per_seed_p_target"]
        if sorted(per_seed) != sorted(str(s) for s in plan.TEACHER_FTSEEDS):
            raise ValueError(f"evaluated ftseeds {sorted(per_seed)}")
        verdict = soft.w0_verdict(
            ref.epochs,
            [per_seed[str(s)] for s in plan.TEACHER_FTSEEDS],
            base["p_per_prompt"],
        )
        if not math.isclose(
            verdict.base_p, entries[0]["base_p_target"], rel_tol=1e-6, abs_tol=1e-12
        ):
            raise ValueError("base p differs between the summary and base_control")
    except (OSError, KeyError, ValueError) as err:
        raise UnitHold(f"W0 {ref.target} {ref.base} ep{ref.epochs}: {err}") from err
    return verdict


def verify_teachers(ref: jobs.TeacherRef, ftseeds: Sequence[int]) -> int:
    """Check `ftseeds`' manifests against the config; return their one step count.

    Raises:
        UnitHold: A manifest is missing or off-plan, or step counts differ.
    """
    try:
        docs_sha = file_sha256(ref.docs_path)
        steps = set()
        for r in ftseeds:
            manifest = json.loads(ref.manifest_path(r).read_text())
            if bad := jobs.teacher_manifest_problems(ref, r, manifest, docs_sha):
                raise ValueError(f"ftseed{r} manifest off-plan {bad}")
            steps.add(int(manifest["n_train_steps"]))
        if len(steps) != 1:
            raise ValueError(f"step counts differ by ftseed: {sorted(steps)}")
    except (OSError, KeyError, ValueError) as err:
        raise UnitHold(f"teachers {ref.param_dir}: {err}") from err
    return steps.pop()


def advance_treatment(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    target: str,
    base: str,
    waiting: list[str],
) -> tuple[list[str], jobs.TeacherRef | None]:
    """Advance one unit's treatment chain; return notes and its chosen teacher, if any."""
    us = state.setdefault("units", {}).setdefault(unit_key(target, base), {})
    if row := plan.reused(plan.KIND_TREAT, target, base):
        us.setdefault(
            "chosen",
            {"epochs": plan.HEADLINE_EPOCHS, "steps": row.steps, "reused": True},
        )
        ref = jobs.teacher_of(row)
        queue = plan.q_teval(target, base, plan.HEADLINE_EPOCHS)
        return submit_teacher(cfg, ref, None, (), queue), ref
    if "chosen" in us:
        return [], jobs.TeacherRef(
            plan.KIND_TREAT, target, base, int(us["chosen"]["epochs"])
        )
    notes: list[str] = []
    w0 = us.setdefault("w0", {})
    for epochs in plan.TEACHER_EPOCHS:
        if str(epochs) in w0:
            if w0[str(epochs)]["passed"]:
                break
            continue
        ref = jobs.TeacherRef(plan.KIND_TREAT, target, base, epochs)
        notes += submit_teacher(
            cfg,
            ref,
            plan.q_teach(target, base, epochs),
            plan.TEACHER_FTSEEDS,
            plan.q_teval(target, base, epochs),
        )
        finished, msg = driver_io.queue_finished(
            cfg, plan.q_teval(target, base, epochs)
        )
        if not finished:
            waiting.append(f"{unit_key(target, base)}: {msg}")
            return notes, None
        verdict = read_w0(ref)
        w0[str(epochs)] = soft.verdict_doc(verdict)
        notes.append(
            f"w0({unit_key(target, base)} ep{epochs} "
            f"{'pass' if verdict.passed else 'fail'})"
        )
        if verdict.passed:
            break
    chosen = soft.choose_epochs({int(k): bool(v["passed"]) for k, v in w0.items()})
    if chosen is None:
        return notes, None
    if not w0[str(chosen)]["passed"]:
        state.setdefault("labelled", {})[f"w0_fail|{unit_key(target, base)}"] = (
            f"W0 failed at every dose; students distil from ep{chosen}"
        )
    ref = jobs.TeacherRef(plan.KIND_TREAT, target, base, chosen)
    steps = verify_teachers(ref, plan.TEACHER_FTSEEDS)
    us["chosen"] = {"epochs": chosen, "steps": steps, "reused": False}
    notes.append(f"teachers({unit_key(target, base)} ep{chosen}, {steps} steps)")
    return notes, ref


def advance_placebo(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    target: str,
    base: str,
    opened: Sequence[int],
) -> tuple[list[str], dict[int, jobs.TeacherRef]]:
    """Advance one unit's placebo chain.

    Returns:
        Notes, and the placebo teacher of each open priority whose ftseeds are trained.
    """
    us = state["units"][unit_key(target, base)]
    row = plan.reused(plan.KIND_PLACEBO, target, base)
    ref = (
        jobs.teacher_of(row)
        if row
        else jobs.TeacherRef(
            plan.KIND_PLACEBO,
            target,
            base,
            int(us["chosen"]["epochs"]),
            int(us["chosen"]["steps"]),
        )
    )
    if ref.steps != us["chosen"]["steps"]:
        state.setdefault("labelled", {})[
            f"placebo_steps_differ|{unit_key(target, base)}"
        ] = f"placebo {ref.steps} steps, treatment {us['chosen']['steps']}"
    notes: list[str] = []
    ready: dict[int, jobs.TeacherRef] = {}
    first_trained = True
    for tier in plan.PRIORITIES:
        if tier not in opened:
            continue
        seeds = plan.placebo_ftseeds(target, base, tier)
        if row or not seeds:
            ready[tier] = ref
            continue
        queue = plan.q_pteach(target, base, tier)
        notes += submit_teacher(cfg, ref, queue, seeds, None)
        finished, _ = driver_io.queue_finished(cfg, queue)
        if not finished:
            first_trained = False
            break
        done = us.setdefault("placebo_verified", {})
        if str(tier) not in done:
            verify_teachers(ref, seeds)
            done[str(tier)] = utc_now_iso()
        ready[tier] = ref
    if first_trained and ready and "placebo_eval" not in us:
        notes += submit_teacher(cfg, ref, None, (), plan.q_pteval(target, base))
        us["placebo_eval"] = utc_now_iso()
    return notes, ready


# ---- leaves -----------------------------------------------------------------------------


def unit_cells(target: str, base: str) -> tuple[plan.Cell, ...]:
    return tuple(c for c in plan.cells() if (c.target, c.teacher) == (target, base))


def submit_leaves(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    cell: plan.Cell,
    teacher: jobs.TeacherRef,
) -> list[str]:
    """Every leaf of `cell` from `teacher`, once (missing jobs added)."""
    done = state.setdefault("leaves_submitted", {})
    key = (
        f"{cell.exp.id}|{cell.target}|{cell.teacher}|{cell.student}|{cell.mode}"
        f"|{cell.lr:g}|{teacher.kind}"
    )
    if key in done:
        return []
    leaves = jobs.cell_leaves(cell, teacher)
    for leaf in leaves:
        write_yamls(cfg, jobs.leaf_yaml_docs(leaf))
    name = leaves[0].queue
    n = _submit(cfg, name, [jobs.leaf_job(leaf) for leaf in leaves])
    done[key] = utc_now_iso()
    return [f"submit({name}, +{n})"] if n else []


def advance_unit(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    target: str,
    base: str,
    opened: Sequence[int],
    waiting: list[str],
) -> list[str]:
    """Advance one (target, teacher base) as far as its gates allow; return notes."""
    notes, treat = advance_treatment(cfg, state, target, base, waiting)
    if treat is None:
        return notes
    cells = [c for c in unit_cells(target, base) if c.exp.priority in opened]
    for cell in cells:
        notes += submit_leaves(cfg, state, cell, treat)
    placebo_notes, ready = advance_placebo(cfg, state, target, base, opened)
    notes += placebo_notes
    for cell in cells:
        if cell.exp.priority in ready:
            notes += submit_leaves(cfg, state, cell, ready[cell.exp.priority])
    return notes


def _unit_pass(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    target: str,
    base: str,
    opened: Sequence[int],
    waiting: list[str],
) -> list[str]:
    """`advance_unit`, with a unit-level failure recorded as that unit's hole."""
    holes = state.setdefault("holes", {})
    name = f"unit:{unit_key(target, base)}"
    try:
        notes = advance_unit(cfg, state, target, base, opened, waiting)
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


# ---- tiers ------------------------------------------------------------------------------


def tier_leaf_queues(priority: int) -> list[str]:
    """Every leaf queue of `priority`."""
    return [
        plan.q_leaves(e, t, k)
        for e in plan.EXPERIMENTS
        if e.priority == priority
        for t in e.targets
        for k in plan.KINDS
    ]


def tier_leaves_submitted(state: dict, priority: int) -> bool:
    """Every cell of `priority` has both kinds of leaf submitted."""
    done = state.get("leaves_submitted", {})
    want = {
        f"{c.exp.id}|{c.target}|{c.teacher}|{c.student}|{c.mode}|{c.lr:g}|{kind}"
        for c in plan.cells(e for e in plan.EXPERIMENTS if e.priority == priority)
        for kind in plan.KINDS
    }
    return want <= set(done)


def tier_pending(cfg: driver_io.CampaignDriverConfig, priority: int) -> int:
    """Jobs of `priority`'s leaf queues not yet started."""
    return sum(
        store.queue_status(cfg.queue_root, n)["pending"]
        for n in tier_leaf_queues(priority)
        if store.queue_exists(cfg.queue_root, n)
    )


def tier_finished(
    cfg: driver_io.CampaignDriverConfig, state: dict, priority: int
) -> bool:
    """Every leaf of `priority` is submitted and finished without a failure."""
    return tier_leaves_submitted(state, priority) and all(
        driver_io.queue_finished(cfg, n)[0]
        for n in tier_leaf_queues(priority)
        if store.queue_exists(cfg.queue_root, n)
    )


def open_tiers(cfg: driver_io.CampaignDriverConfig, state: dict) -> tuple[int, ...]:
    """Priorities whose work may start: P1 and P2 at once, P3 once they have no pending
    leaf left."""
    if "open_p3" in state["transitions"]:
        return plan.PRIORITIES
    early = (plan.P1, plan.P2)
    if all(
        tier_leaves_submitted(state, p) and tier_pending(cfg, p) == 0 for p in early
    ):
        _fire(state, "open_p3")
        return plan.PRIORITIES
    return early


def tier_marker(cfg: driver_io.CampaignDriverConfig, priority: int) -> Path:
    return cfg.logs_dir / f"P{priority}_DONE"


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
    if not cfg.dry_run:
        if bad := pin_problem(cfg, state):
            raise CampaignGateError(bad)
        driver_io.ensure_campaign_code_pin(cfg)
        pin = current_pin(cfg)
        staging.link_pin_to_itself(pin.root, pin.sha)
    done = state["transitions"]
    holes = state.setdefault("holes", {})
    labelled = state.setdefault("labelled", {})
    fired: list[str] = []
    waiting: list[str] = []

    def mark(name: str) -> None:
        fired.append(name)
        driver_io.save_state(cfg, state)

    if "preflight" not in done:
        _preflight(cfg, state)
        mark("preflight")
    if "register" not in done:
        _register(cfg, state)
        mark("register")
    if note := _repin(cfg):
        fired.append(note)
    if not cfg.dry_run:
        _requeue_and_holes(cfg, state)
    disk = [] if cfg.dry_run else disk_problems()
    if disk:
        labelled["disk_low"] = "; ".join(disk)
        waiting.append(f"new submissions held: {labelled['disk_low']}")
    else:
        labelled.pop("disk_low", None)
    if cfg.dry_run:
        waiting.append("dry-run: unit chains, tiers and finalize not evaluated")
        return AdvanceResult(tuple(fired), tuple(waiting), False)
    if not disk:
        opened = open_tiers(cfg, state)
        for target, base in plan.teacher_units():
            if plan.unit_priority(target, base) not in opened:
                continue
            fired.extend(_unit_pass(cfg, state, target, base, opened, waiting))
            driver_io.save_state(cfg, state)
    for priority in plan.PRIORITIES:
        name = f"p{priority}_done"
        if name not in done and tier_finished(cfg, state, priority):
            marker = tier_marker(cfg, priority)
            marker.write_text(f"{utc_now_iso()}\n")
            _fire(state, name)
            mark(name)
    all_done = False
    blocking = sorted(k for k in holes if k != "results")
    if all(f"p{p}_done" in done for p in plan.PRIORITIES) and not blocking:
        if "finalize" not in done:
            (cfg.logs_dir / SPEC.campaign_done_file).write_text(f"{utc_now_iso()}\n")
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
    per_tier: dict[str, dict[str, int]] = {}
    for priority in plan.PRIORITIES:
        tier = dict.fromkeys(totals, 0)
        for name in tier_leaf_queues(priority):
            if store.queue_exists(cfg.queue_root, name):
                st = store.queue_status(cfg.queue_root, name)
                for k in tier:
                    tier[k] += int(st[k])
        per_tier[f"p{priority}_leaves"] = tier
    for name in submitted(cfg):
        st = store.queue_status(cfg.queue_root, name)
        for k in totals:
            totals[k] += int(st[k])
    return {
        "transitions": sorted(state["transitions"]),
        "holes": state.get("holes", {}),
        "labelled": state.get("labelled", {}),
        "queues": len(submitted(cfg)),
        "jobs": totals,
        **per_tier,
        "last_pass": state.get("last_pass"),
    }
