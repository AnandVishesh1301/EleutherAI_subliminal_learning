"""Auto-advancing driver of `newdom2` (plan § 13), on soar; unpinned queues in the
Part II queue root, drained by soar's pool.

One `run(cfg)` = one idempotent advance pass:

    preflight   prereg Part II section, home node, roots, Part II paths YAML, complete
                staging report, disk floor
    register    the campaign pin can run every job
    (every pass) unfinished queues follow the pin; failures under the requeue cap are
                requeued, the rest are holes; below the disk floor nothing new is submitted
    C (pine)    Part I pine ep5 + s740 placebo teachers verified (Part I corpus / slot
                shas, 740 steps) -> treatment + placebo leaves
    A (target)  ep8 teachers on the Part I corpus (train + eval queues) -> manifests
                verified, one step count -> treatment leaves + placebo teachers at that
                count (train + eval) -> placebo manifests -> placebo leaves
    B (target)  Part I teachers at 740 steps; B corpus matches its manifest; approved B
                corpus review -> teachers at 740 steps on it (train + eval) -> manifests
                verified -> treatment leaves (placebo leaves are Part I's or C's)
    strict_sub  every pine leaf set also waits for the approved screen review
    results     every planned queue submitted and finished, no hole -> a results build
    finalize    the run-set logs marker; in production the Part II `CAMPAIGN_DONE`

Teacher evals never gate. A unit-level contract failure is a hole of that unit; the
others advance. State and logs live in the driver dir of Part I's soft output root
(`gkd_students/<Part II teacher run set>/logs`, as for Part I); passes hold an
exclusive flock there and touch the Part II heartbeat.
"""

from __future__ import annotations

import fcntl
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from loguru import logger

from pipelines import newdom2_results_pipeline as results
from pipelines import newdom2_staging_pipeline as staging
from pipelines.newdom_soft_campaign_pipeline import (
    PASS_LOCK_FILE,
    TargetHold,
    _repin,
    _requeue_and_holes,
    _submit,
    disk_problems,
    missing_jobs,
    pin_config,
    submitted,
    verify_teachers,
    write_yamls,
)
from pipelines.optparam2_campaign_pipeline import SPEC as OPTPARAM2_SPEC
from pipelines.optparam2_campaign_pipeline import (
    AdvanceResult,
    CampaignGateError,
    _fire,
    current_pin,
)
from pipelines.xopt2_soft_campaign_pipeline import write_marker
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import newdom2 as n2
from subliminal_learning.campaign import newdom2_plan as plan2
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.campaign import newdom_soft as soft
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue import store
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import file_sha256
from subliminal_learning.utils.run_io import read_run_set_card, utc_now_iso

STATE_FILE = "newdom2_campaign_state.json"
DONE_FILE = "CAMPAIGN_DONE"
QUEUES: tuple[str, ...] = tuple(n2.QUEUE_PRIORITIES)


def _pin_paths() -> tuple[str, ...]:
    return (
        "scripts/sweeps/sweep_sdf_teacher.py",
        "scripts/sweeps/sweep_evaluate_sdf_teacher.py",
        "scripts/sweeps/sweep_distill_student_gkd.py",
        "scripts/sweeps/sweep_evaluate_gkd_student.py",
        "scripts/newdom_panel.py",
        "scripts/run_newdom2_campaign.py",
        "scripts/stage_newdom2_assets.py",
        "scripts/launch_newdom2_campaign.sh",
        "scripts/jobq_pool_loop.sh",
        soft.TEACHER_TRAINING_CONFIG,
        soft.STUDENT_TRAINING_CONFIG,
        "src/subliminal_learning/campaign/newdom_plan.py",
        "src/subliminal_learning/campaign/newdom2_plan.py",
        "src/subliminal_learning/campaign/newdom_soft.py",
        "src/subliminal_learning/campaign/newdom2.py",
        "src/subliminal_learning/campaign/newdom2_metrics.py",
        "src/subliminal_learning/gkd/filtering.py",
        "src/subliminal_learning/gkd/layout.py",
        "src/subliminal_learning/gkd/trainer_io.py",
        "src/subliminal_learning/evaluation/gkd_results_io.py",
        "pipelines/distill_student_gkd_pipeline.py",
        "pipelines/newdom_panel_pipeline.py",
        "pipelines/newdom2_campaign_pipeline.py",
        "pipelines/newdom2_results_pipeline.py",
        "pipelines/newdom2_staging_pipeline.py",
    )


SPEC = replace(
    OPTPARAM2_SPEC,
    run_set_id=plan2.TEACHER_RUN_SET,
    teacher_run_set_id=plan2.TEACHER_RUN_SET,
    root_subdir=GKD_STUDENTS_SUBDIR,
    queue_priorities=n2.QUEUE_PRIORITIES,
    gpu_policy={"cap_per_gpu": plan2.CAP_PER_GPU},
    generated_config_dir=str(plan2.GENERATED_DIR),
    campaign_state_file=STATE_FILE,
    campaign_done_file=DONE_FILE,
    gkd_experiment_config=None,
    extra_pin_paths=_pin_paths(),
)


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the Part II spec."""
    kwargs.setdefault("generated_dir_override", plan2.GENERATED_DIR)
    return driver_io.CampaignDriverConfig(spec=SPEC, **kwargs)


def _production(cfg: driver_io.CampaignDriverConfig) -> bool:
    return Path(cfg.output_root) == plan2.OUTPUT_ROOT


# ---- preflight, pin --------------------------------------------------------------------


def preflight_problems(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    problems = []
    prereg = cfg.repo_root / plan2.PREREG_PATH
    if not prereg.is_file() or plan2.PREREG_MARKER not in prereg.read_text():
        problems.append(f"no Part II section in {plan2.PREREG_PATH}")
    host = store.this_host()
    if host != plan2.HOME_NODE:
        problems.append(f"the driver runs on {plan2.HOME_NODE}, not {host}")
    if Path(cfg.queue_root).resolve() != plan2.QUEUE_ROOT.resolve():
        problems.append(f"queue root {cfg.queue_root} is not the plan's")
    if not _production(cfg):
        problems.append(f"output root {cfg.output_root} is not the plan's")
    if (
        not plan2.PATHS_CONFIG_PATH.is_file()
        or plan2.PATHS_CONFIG_PATH.read_text() != staging.paths_yaml_text()
    ):
        problems.append(f"{plan2.PATHS_CONFIG_PATH} missing or differs from the plan")
    if staging.latest_complete_report() is None:
        problems.append("no complete staging report; run scripts.stage_newdom2_assets")
    return problems + disk_problems(plan2.ROOT)


def pin_problem(cfg: driver_io.CampaignDriverConfig, state: dict) -> str | None:
    """Why the pass must not run under this pin, or None: before the Part II teacher
    card exists the pin is `--code-sha` (the registered one unless
    `auto_reconcile_sha`); every existing Part II student card names the same pin."""
    resolve = code_pin_io.resolve_commit_sha
    card = read_run_set_card(pin_config(cfg).run_set_root) or {}
    mine = card.get("git_sha") or cfg.code_sha
    if not mine:
        return "pass --code-sha until the Part II teacher run-set card exists"
    mine = resolve(cfg.repo_root, str(mine))
    registered = (state["transitions"].get("register") or {}).get("git_sha")
    if not card and registered and registered != mine and not cfg.auto_reconcile_sha:
        return (
            f"--code-sha {mine[:12]} differs from the registered pin {registered[:12]}"
        )
    for run_set in {rs for _, _, rs in soft.PART2.student_run_sets}:
        root = cfg.output_root / run_set_subdir(GKD_STUDENTS_SUBDIR, run_set)
        sha = (read_run_set_card(root) or {}).get("git_sha")
        if sha and resolve(cfg.repo_root, str(sha)) != mine:
            return (
                f"{run_set} pinned to {str(sha)[:12]}, Part II teachers to {mine[:12]}"
            )
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
    pin = current_pin(pin_config(cfg))
    driver_io.verify_pin_runs_campaign(pin, SPEC)
    _fire(state, "register", git_sha=pin.sha)


# ---- submissions -----------------------------------------------------------------------


def submit_teacher(
    cfg: driver_io.CampaignDriverConfig, ref: soft.TeacherRef
) -> list[str]:
    """The config's train queue and its eval queue (missing jobs added)."""
    notes = []
    jobs = n2.teacher_train_jobs(ref)
    if missing_jobs(cfg, ref.train_queue, jobs):
        write_yamls(cfg, soft.teacher_yaml_docs(ref))
        notes.append(
            f"submit({ref.train_queue}, +{_submit(cfg, ref.train_queue, jobs)})"
        )
    if _submit(
        cfg,
        ref.eval_queue,
        [n2.teacher_eval_job(ref)],
        depends_on=[ref.train_queue],
        strict=True,
    ):
        notes.append(f"submit({ref.eval_queue})")
    return notes


def submit_leaves(
    cfg: driver_io.CampaignDriverConfig, leaves: Sequence[soft.SoftLeaf]
) -> list[str]:
    """One teacher config's Part II leaves, in their queue (missing jobs added)."""
    names = {lf.queue for lf in leaves}
    if len(names) != 1 or any(lf.lane is not soft.PART2 for lf in leaves):
        raise ValueError(f"leaves span queues {sorted(names)} or another part")
    name = names.pop()
    jobs = n2.leaf_jobs(leaves)
    new = {j.key for j in missing_jobs(cfg, name, jobs)}
    if not new:
        return []
    write_yamls(
        cfg,
        {
            p: d
            for lf in leaves
            if lf.key in new
            for p, d in soft.leaf_yaml_docs(lf).items()
        },
    )
    return [f"submit({name}, +{_submit(cfg, name, jobs)})"]


# ---- gates -----------------------------------------------------------------------------


def screen_ready(target: str) -> bool:
    """True when `target`'s leaves need no screen review or theirs is approved.

    Raises:
        TargetHold: The review is unreadable or names another vocabulary or sample.
    """
    if plan2.screen_of(target) != plan2.SCREEN_SUB:
        return True
    ok, problem = results.screen_review_status(
        target, staging.sub_vocab(target).sha256()
    )
    if problem:
        raise TargetHold(problem)
    return ok


def _screen_gate(us: dict, target: str) -> bool:
    """`screen_ready`, recording the approved review's sha in the unit."""
    ok = screen_ready(target)
    if ok and plan2.screen_of(target) == plan2.SCREEN_SUB:
        us["screen_review_sha256"] = file_sha256(plan2.screen_review_path(target))
    return ok


def part1_corpus_sha(target: str) -> str:
    try:
        return staging.part1_corpus_sha(target)
    except (OSError, ValueError) as err:
        raise TargetHold(str(err)) from err


def verify_b_corpus(target: str) -> str:
    """sha of the B corpus after checking it against its manifest.

    Raises:
        TargetHold: Missing, changed, or naming another target / source.
    """
    path = plan2.variant_corpus_path(target, plan2.B_CORPUS)
    try:
        man = json.loads(
            plan2.variant_manifest_path(target, plan2.B_CORPUS).read_text()
        )
        rows = staging.read_jsonl(path)
        sha = file_sha256(path)
    except (OSError, ValueError) as err:
        raise TargetHold(f"B corpus {target}: {err}") from err
    if sha != man["docs_sha256"] or len(rows) != man["n_kept"]:
        raise TargetHold(f"{path} differs from its manifest")
    if {r.get("animal") for r in rows} != {target}:
        raise TargetHold(f"{path} rows name another target")
    if man["source_sha256"] != part1_corpus_sha(target):
        raise TargetHold(f"{path} was filtered from another Part I corpus")
    return sha


def _slot_sha(target: str) -> str:
    return json.loads(soft.placebo_slot_manifest_path(target).read_text())[
        "docs_sha256"
    ]


def _check_steps(steps: int, want: int, what: str) -> int:
    if steps != want:
        raise TargetHold(f"{what}: {steps} steps != {want}")
    return steps


# ---- units -----------------------------------------------------------------------------


def advance_c(cfg, state: dict, target: str, waiting: list[str]) -> list[str]:
    """C: Part I pine teachers -> leaves under `strict_sub`."""
    us = state.setdefault("units", {}).setdefault(f"{plan2.ARM_C}:{target}", {})
    if "verified" not in us:
        _check_steps(
            verify_teachers(n2.part1_teacher(target), part1_corpus_sha(target)),
            plan2.PART1_STEPS,
            f"Part I {target} teachers",
        )
        _check_steps(
            verify_teachers(n2.part1_placebo(target), _slot_sha(target)),
            plan2.PART1_STEPS,
            f"Part I {target} placebo teachers",
        )
        us["verified"] = utc_now_iso()
    if not _screen_gate(us, target):
        waiting.append(f"C {target}: screen review {plan2.screen_review_path(target)}")
        return []
    return submit_leaves(cfg, n2.treat_leaves(plan2.ARM_C, target)) + submit_leaves(
        cfg, n2.placebo_leaves(plan2.ARM_C, target)
    )


def advance_a(cfg, state: dict, target: str, waiting: list[str]) -> list[str]:
    """A: ep8 teachers -> treatment leaves + matched placebo teachers -> placebo leaves."""
    us = state.setdefault("units", {}).setdefault(f"{plan2.ARM_A}:{target}", {})
    us.setdefault("corpus_sha", part1_corpus_sha(target))
    treat = n2.arm_teacher(plan2.ARM_A, target)
    notes = submit_teacher(cfg, treat)
    finished, msg = driver_io.queue_finished(cfg, treat.train_queue)
    if not finished:
        waiting.append(f"A {target}: {msg}")
        return notes
    if "steps" not in us:
        us["steps"] = verify_teachers(treat, us["corpus_sha"])
        notes.append(f"teachers(A {target}, {us['steps']} steps)")
    placebo = n2.arm_placebo(plan2.ARM_A, target, int(us["steps"]))
    notes += submit_teacher(cfg, placebo)
    screen = _screen_gate(us, target)
    if screen:
        notes += submit_leaves(cfg, n2.treat_leaves(plan2.ARM_A, target))
    finished, msg = driver_io.queue_finished(cfg, placebo.train_queue)
    if not finished:
        waiting.append(f"A {target}: {msg}")
        return notes
    if "placebo_verified" not in us:
        _check_steps(
            verify_teachers(placebo, _slot_sha(target)),
            int(us["steps"]),
            f"A {target} placebo teachers",
        )
        us["placebo_verified"] = utc_now_iso()
    if not screen:
        waiting.append(f"A {target}: screen review {plan2.screen_review_path(target)}")
        return notes
    return notes + submit_leaves(
        cfg, n2.placebo_leaves(plan2.ARM_A, target, int(us["steps"]))
    )


def advance_b(cfg, state: dict, target: str, waiting: list[str]) -> list[str]:
    """B: reviewed panelfree corpus -> 740-step teachers -> treatment leaves."""
    us = state.setdefault("units", {}).setdefault(f"{plan2.ARM_B}:{target}", {})
    if "part1_steps" not in us:
        us["part1_steps"] = _check_steps(
            verify_teachers(n2.part1_teacher(target), part1_corpus_sha(target)),
            plan2.B_STEPS,
            f"Part I {target} teachers",
        )
    sha = verify_b_corpus(target)
    us["corpus_sha"] = sha
    if "review" not in us:
        ok, problem = results.corpus_review_status(target, sha)
        if problem:
            raise TargetHold(problem)
        if not ok:
            waiting.append(
                f"B {target}: corpus review {plan2.corpus_review_path(target)}"
            )
            return []
        us["review"] = {
            "at": utc_now_iso(),
            "sha256": file_sha256(plan2.corpus_review_path(target)),
        }
    treat = n2.arm_teacher(plan2.ARM_B, target)
    notes = submit_teacher(cfg, treat)
    finished, msg = driver_io.queue_finished(cfg, treat.train_queue)
    if not finished:
        waiting.append(f"B {target}: {msg}")
        return notes
    if "steps" not in us:
        us["steps"] = _check_steps(
            verify_teachers(treat, sha), plan2.B_STEPS, f"B {target} teachers"
        )
    if not _screen_gate(us, target):
        waiting.append(f"B {target}: screen review {plan2.screen_review_path(target)}")
        return notes
    return notes + submit_leaves(cfg, n2.treat_leaves(plan2.ARM_B, target))


ADVANCE = {plan2.ARM_A: advance_a, plan2.ARM_B: advance_b, plan2.ARM_C: advance_c}


def _unit_pass(
    cfg, state: dict, arm: str, target: str, waiting: list[str]
) -> list[str]:
    """One unit's advance, with any unit-level failure recorded as its hole."""
    holes = state.setdefault("holes", {})
    name = f"unit:{arm}:{target}"
    try:
        notes = ADVANCE[arm](cfg, state, target, waiting)
    except CampaignGateError:
        raise
    except (TargetHold, OSError, ValueError, KeyError, RuntimeError) as err:
        msg = (
            str(err) if isinstance(err, TargetHold) else f"{type(err).__name__}: {err}"
        )
        if holes.get(name) != msg:
            logger.opt(exception=not isinstance(err, TargetHold)).error(
                "hole in {}: {}", name, msg
            )
        holes[name] = msg
        return []
    holes.pop(name, None)
    return notes


def chain_complete(cfg: driver_io.CampaignDriverConfig, state: dict) -> bool:
    """Every planned Part II queue exists and finished."""
    if any(not store.queue_exists(cfg.queue_root, n) for n in QUEUES):
        return False
    return all(driver_io.queue_finished(cfg, n)[0] for n in QUEUES)


# ---- the pass --------------------------------------------------------------------------


def run(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    """One advance pass, under an exclusive flock in the driver logs dir."""
    if cfg.dry_run:
        return _pass(cfg)
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.logs_dir / PASS_LOCK_FILE).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        return _pass(cfg)


def _pass(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    state = driver_io.load_state(cfg)
    if not cfg.dry_run:
        if bad := pin_problem(cfg, state):
            raise CampaignGateError(bad)
        driver_io.ensure_campaign_code_pin(pin_config(cfg))
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
    if not cfg.dry_run:
        _record_prereg(cfg, state)
    if "register" not in done:
        _register(cfg, state)
        mark("register")
    if cfg.dry_run:
        waiting.append(
            "dry-run: units, results and finalize not evaluated; "
            f"{len(QUEUES)} queues planned"
        )
        return AdvanceResult(tuple(fired), tuple(waiting), False)
    if note := _repin(cfg, QUEUES):
        fired.append(note)
    _requeue_and_holes(cfg, state, QUEUES)
    disk = disk_problems(plan2.ROOT)
    if disk:
        labelled["disk_low"] = "; ".join(disk)
        waiting.append(f"new submissions held: {labelled['disk_low']}")
    else:
        labelled.pop("disk_low", None)
        for arm in (plan2.ARM_C, plan2.ARM_A, plan2.ARM_B):
            for target in plan2.ARMS[arm]:
                fired.extend(_unit_pass(cfg, state, arm, target, waiting))
                driver_io.save_state(cfg, state)
    blocking = sorted(k for k in holes if k != "results")
    if (
        "results" not in done
        and not blocking
        and not disk
        and chain_complete(cfg, state)
    ):
        try:
            build = results.build(
                cfg.queue_root,
                state.setdefault("units", {}),
                _results_root(cfg),
                provenance(cfg, state),
            )
        except (ValueError, FileExistsError) as err:
            holes["results"] = str(err)
            logger.error("results not built: {}", err)
        else:
            holes.pop("results", None)
            _fire(state, "results", build=str(build))
            mark("results")
    all_done = False
    if "results" in done:
        if "finalize" not in done:
            write_marker(cfg.logs_dir / DONE_FILE)
            if _production(cfg):
                write_marker(plan2.CAMPAIGN_DONE_PATH)
            _fire(state, "finalize", labelled=sorted(labelled))
            mark("finalize")
        all_done = True
    elif holes:
        waiting.append(f"holes in {sorted(holes)}")
    state["last_pass"] = {"at": utc_now_iso(), "fired": fired, "waiting": waiting[:50]}
    driver_io.save_state(cfg, state)
    if _production(cfg):
        plan2.DRIVER_HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        plan2.DRIVER_HEARTBEAT_PATH.touch()
    return AdvanceResult(tuple(fired), tuple(waiting), all_done)


def _record_prereg(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    """Record the prereg's sha: first seen and this pass's."""
    sha = file_sha256(cfg.repo_root / plan2.PREREG_PATH)
    prereg = state.setdefault(
        "prereg", {"first_sha256": sha, "first_at": utc_now_iso()}
    )
    prereg.update(current_sha256=sha, current_at=utc_now_iso())


def provenance(cfg: driver_io.CampaignDriverConfig, state: dict) -> dict:
    """Pin, prereg shas and staging report recorded in a results manifest."""
    report = staging.latest_complete_report_path()
    return {
        "git_sha": (state["transitions"].get("register") or {}).get("git_sha"),
        "prereg": dict(state.get("prereg") or {}),
        "staging_report": str(report) if report else None,
    }


def _results_root(cfg: driver_io.CampaignDriverConfig) -> Path:
    return plan2.RESULTS_DIR if _production(cfg) else cfg.run_set_root / "results"


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Transitions, holes, labels, units and queue counts (file reads)."""
    state = driver_io.load_state(cfg)
    queues = {}
    for name in submitted(cfg, QUEUES):
        s = store.queue_status(cfg.queue_root, name)
        queues[name] = {k: v for k, v in s.items() if k != "running_jobs"}
    return {
        "transitions": sorted(state.get("transitions", {})),
        "holes": state.get("holes", {}),
        "labelled": state.get("labelled", {}),
        "units": state.get("units", {}),
        "queues": queues,
        "planned_queues": len(QUEUES),
    }


def plan_check() -> dict:
    """The pure plan (`newdom2.plan_doc`) plus the reuse it reads from Part I."""
    return {
        **n2.plan_doc(),
        "reuses": {
            "C_teachers": [
                str(n2.part1_teacher("pine").param_dir),
                str(n2.part1_placebo("pine").param_dir),
            ],
            "B_placebo_leaves": {
                t: [lf.key for lf in n2.placebo_leaves(plan2.ARM_B, t)]
                for t in plan2.ARMS[plan2.ARM_B]
            },
            "part1_queue_root": str(plan.QUEUE_ROOT),
        },
    }
