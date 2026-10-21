"""Auto-advancing driver of `tmplckpt`: both channels, home cluster-3, unpinned queues.

One `run(cfg)` = one idempotent advance pass:

    (every pass) the hard run-set card and every soft card that exists name one pin
    preflight    prereg + plan, queue root, paths YAML, passed staging report, one
                 clone root on every host, home node
    register     hard run-set card pinned to the campaign code pin
    (every pass) the soft channel's generated YAMLs; every queue of the enabled waves
                 created or appended (unpinned, the pin's defaults, plan priorities,
                 strict plan dependencies); hard students only on pools whose filtered
                 sha256 the operator approved in `POOL_REVIEW_PATH`; finished pools still
                 awaiting review listed in ``pools_pending.json``; failures under the
                 cap requeued, the rest recorded as holes; every submission held while a
                 mount is under its free-space floor; partial results every few passes
    finalize     every wave enabled, every pool approved, every queue finished, no
                 hole, final results built: `CAMPAIGN_DONE_PATH` (create-only)

The driver never writes the pool review; a failed job is never read as an outcome.
State lives in `tmplckpt_plan.STATE_FILE`; every pass touches
`tmplckpt_plan.DRIVER_HEARTBEAT_PATH`. No wall-clock deadline.
"""

from __future__ import annotations

import fcntl
import importlib
import json
import shutil
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import ModuleType

from loguru import logger

from pipelines import tmplckpt_staging_pipeline as staging
from pipelines.hlopt_campaign_pipeline import (
    AdvanceResult,
    CampaignGateError,
    _fire,
    current_pin,
    node_pin_defaults,
)
from subliminal_learning.campaign import driver_io, xfam2_plan
from subliminal_learning.campaign import tmplckpt_plan as plan
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.training.run_set import build_run_set_card
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import (
    read_run_set_card,
    register_run_set_launch,
    utc_now_iso,
)

HARD_MODULE = "subliminal_learning.campaign.tmplckpt_hard"
SOFT_MODULE = "subliminal_learning.campaign.tmplckpt_soft"
RESULTS_MODULE = "pipelines.tmplckpt_results_pipeline"
RESULTS_EVERY_PASSES = 6
WAVES_PATH_NAME = "waves.json"
PENDING_NAME = "pools_pending.json"
WATCH_DIR = Path("/mnt/ssd-3/soar-subliminal/vishesh/run_logs/tmplckpt_watch")
DRIVER_WINDOW = "driver"

SPEC = replace(
    xfam2_plan.SPEC,
    run_set_id=plan.RUN_SET_IDS[plan.HARD],
    teacher_run_set_id=plan.TMPL_TEACHER_RS,
    root_subdir=plan.CAMPAIGN,
    queue_priorities=plan.QUEUE_PRIORITIES,
    gpu_policy={"cap_per_gpu": dict.fromkeys(plan.HOSTS, plan.CAP_PER_GPU)},
    generated_config_dir=str(plan.GENERATED_DIR),
    campaign_state_file=plan.STATE_FILE.name,
    campaign_done_file=plan.CAMPAIGN_DONE_PATH.name,
    extra_pin_paths=(
        "scripts/run_tmplckpt_campaign.py",
        "scripts/stage_tmplckpt_assets.py",
        "scripts/run_tmplckpt_hard_cell.py",
        "scripts/launch_tmplckpt_campaign.sh",
        "scripts/jobq_pool_loop.sh",
        "scripts/sweeps/sweep_sdf_teacher.py",
        "scripts/sweeps/sweep_evaluate_sdf_teacher.py",
        "scripts/sweeps/sweep_distill_student_gkd.py",
        "scripts/sweeps/sweep_evaluate_gkd_student.py",
        "src/subliminal_learning/campaign/tmplckpt_plan.py",
        "src/subliminal_learning/campaign/tmplckpt_hard.py",
        "src/subliminal_learning/campaign/tmplckpt_soft.py",
        "pipelines/tmplckpt_campaign_pipeline.py",
        "pipelines/tmplckpt_staging_pipeline.py",
        "pipelines/tmplckpt_hard_cell_pipeline.py",
        "pipelines/tmplckpt_results_pipeline.py",
        "src/subliminal_learning/campaign/tmplckpt_metrics.py",
        "scripts/build_tmplckpt_results.py",
        "configs/experiment/number_generation.yaml",
        "data/number_prompts/sequence_intro_templates.json",
        str(plan.ANIMAL_TEMPLATE),
        str(plan.COLOR_TEMPLATE),
    ),
)
_STATE_KEYS: tuple[str, ...] = ("holes", "labelled", "results")


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the tmplckpt hard run set (the pin card)."""
    kwargs.setdefault("generated_dir_override", plan.GENERATED_DIR)
    return driver_io.CampaignDriverConfig(spec=SPEC, **kwargs)


def channels() -> tuple[ModuleType, ModuleType]:
    """The hard and soft channel modules (imported per pass; tests replace this)."""
    return importlib.import_module(HARD_MODULE), importlib.import_module(SOFT_MODULE)


# ---- state, pin -----------------------------------------------------------------------------


def load_state() -> dict:
    """The driver state (``transitions`` always present)."""
    if not plan.STATE_FILE.is_file():
        state: dict = {"transitions": {}}
    else:
        state = json.loads(plan.STATE_FILE.read_text())
    state.setdefault("transitions", {})
    for key in _STATE_KEYS:
        state.setdefault(key, {})
    return state


def save_state(state: dict, *, dry_run: bool) -> None:
    if dry_run:
        return
    plan.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(plan.STATE_FILE, state)


def heartbeat() -> None:
    plan.DRIVER_HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.DRIVER_HEARTBEAT_PATH.touch()


def driver_lock_path() -> Path:
    return plan.CONTROL_ROOT / "locks" / "driver.lock"


@contextmanager
def driver_lock() -> Iterator[None]:
    """Hold the campaign's single-driver flock for the life of the block.

    Raises:
        CampaignGateError: Another driver process holds it.
    """
    path = driver_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as err:
            raise CampaignGateError(
                f"another tmplckpt driver holds {path}; refusing to start a second"
            ) from err
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def pin_defaults(cfg: driver_io.CampaignDriverConfig) -> dict:
    """jobq defaults of an unpinned queue: the pin's worktree, valid on every host.

    Raises:
        CampaignGateError: The hosts' clone roots differ.
    """
    roots = {plan.CLONE_ROOT[h] for h in plan.HOSTS}
    if len(roots) != 1:
        raise CampaignGateError(f"clone roots differ across hosts: {sorted(roots)}")
    return node_pin_defaults(cfg, plan.HOME_NODE)


def tmpl_card_root() -> Path:
    """Run-set root of the statement-teacher run set (card written at register)."""
    return plan.OUTPUT_ROOT / "sdf_teachers" / plan.TMPL_TEACHER_RS


def soft_card_roots() -> dict[str, Path]:
    """Run-set roots of every card beside the hard one (all must share the pin)."""
    teachers = plan.OUTPUT_ROOT / "sdf_teachers"
    return {
        "soft students": plan.SOFT_RUN_SET_DIR,
        "statement teachers": tmpl_card_root(),
        "sdf teachers": teachers / plan.SDF_TEACHER_RS,
        "sdf placebo teachers": teachers / plan.SDF_PLACEBO_RS,
    }


def pin_problem(repo_root: Path, own_sha: str, roots: Mapping[str, Path]) -> str | None:
    """A card under `roots` pinned to another commit than `own_sha` (None if none)."""
    for name, root in roots.items():
        sha = (read_run_set_card(root) or {}).get("git_sha")
        if not sha:
            continue
        if code_pin_io.resolve_commit_sha(repo_root, str(sha)) != own_sha:
            return f"{name} card pins {str(sha)[:12]}; the campaign pins {own_sha[:12]}"
    return None


def _own_sha(cfg: driver_io.CampaignDriverConfig) -> str:
    if cfg.dry_run:
        return code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha or "HEAD")
    return current_pin(cfg).sha


# ---- operator files ---------------------------------------------------------------------------


def waves_path() -> Path:
    return plan.CONTROL_ROOT / WAVES_PATH_NAME


def enabled_waves() -> tuple[frozenset[str], str | None]:
    """Waves the operator enabled (default every wave) and a problem, if any.

    A malformed or unknown entry enables no wave (submission of every wave queue held).
    """
    path = waves_path()
    if not path.is_file():
        return frozenset(plan.WAVES), None
    try:
        doc = json.loads(path.read_text())
        waves = frozenset(str(w) for w in doc["enabled"])
    except (json.JSONDecodeError, KeyError, TypeError) as err:
        return frozenset(), f"{path} unreadable ({err})"
    unknown = sorted(waves - set(plan.WAVES))
    if unknown:
        return frozenset(), f"{path} names unknown waves {unknown}"
    return waves, None


def queue_wave(name: str) -> str | None:
    """The wave a queue belongs to (``None`` = shared by every wave)."""
    for wave in plan.WAVES:
        if name.endswith(f"-{wave}"):
            return wave
    return None


def approved_pools(hard: ModuleType) -> tuple[frozenset[str], str | None]:
    """Filtered sha256s the operator approved (`tmplckpt_hard.read_approved_pool_shas`
    on `POOL_REVIEW_PATH`), and a problem if the review is unreadable."""
    try:
        return frozenset(hard.read_approved_pool_shas(plan.POOL_REVIEW_PATH)), None
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as err:
        return frozenset(), f"{plan.POOL_REVIEW_PATH} unreadable ({err})"


def review_counts(
    items: Iterable[Mapping], approved: frozenset[str], *, short: str
) -> tuple[list, int]:
    """Finished pools still awaiting approval, and the number resolved.

    A pool is resolved when its filtered sha256 is approved or its QC ``verdict`` is
    `short` (under the row floor: an outcome, never reviewed).
    """
    pending, n_resolved = [], 0
    for item in items:
        sha = item.get("filtered_sha256")
        if item.get("verdict") == short or (sha is not None and str(sha) in approved):
            n_resolved += 1
        else:
            pending.append(dict(item))
    return pending, n_resolved


def write_pending(pending: list, *, dry_run: bool) -> None:
    """Rewrite ``pools_pending.json`` when its content changed."""
    path = plan.REVIEW_DIR / PENDING_NAME
    old = json.loads(path.read_text()).get("pending") if path.is_file() else None
    if dry_run or old == pending:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, {"at": utc_now_iso(), "pending": pending})


def disk_low() -> list[str]:
    """Mounts under their `MIN_FREE_GIB` floor."""
    out = []
    for mount, floor in plan.MIN_FREE_GIB.items():
        free = shutil.disk_usage(mount).free / 2**30
        if free < floor:
            out.append(f"{mount}: {free:.0f} GiB free < {floor}")
    return out


# ---- queues -----------------------------------------------------------------------------------


def planned_jobs(
    hard: ModuleType,
    soft: ModuleType,
    *,
    approved_shas: frozenset[str] | None,
    waves: frozenset[str],
) -> dict[str, list[Job]]:
    """Every job both channels plan for the enabled waves, by queue.

    Raises:
        CampaignGateError: A queue name outside `plan.QUEUE_PRIORITIES`, or one queue
            planned by both channels.
    """
    out: dict[str, list[Job]] = {}
    for part in (hard.queue_jobs(approved_pool_shas=approved_shas), soft.queue_jobs()):
        for name, jobs in part.items():
            if name not in plan.QUEUE_PRIORITIES:
                raise CampaignGateError(f"queue {name!r} is not in QUEUE_PRIORITIES")
            if name in out:
                raise CampaignGateError(f"queue {name!r} planned by both channels")
            wave = queue_wave(name)
            if wave is None or wave in waves:
                out[name] = list(jobs)
    return out


def submit_order(names: Iterable[str]) -> list[str]:
    """`names` with every dependency first, then by priority (highest first)."""
    names = sorted(names, key=lambda n: (-plan.QUEUE_PRIORITIES[n], n))
    out: list[str] = []
    seen: set[str] = set()

    def visit(name: str) -> None:
        if name in seen:
            return
        seen.add(name)
        for dep in plan.QUEUE_DEPENDS.get(name, ()):
            if dep in names:
                visit(dep)
        out.append(name)

    for name in names:
        visit(name)
    return out


def campaign_queues(root: Path) -> list[str]:
    """Every campaign queue that exists under `root`."""
    return [n for n in plan.QUEUE_PRIORITIES if store.queue_exists(root, n)]


def _deps(root: Path, name: str, planned: Mapping[str, list[Job]]) -> list[str]:
    """Plan dependencies of `name` that will exist (an empty one would block forever)."""
    return [
        d
        for d in plan.QUEUE_DEPENDS.get(name, ())
        if planned.get(d) or store.queue_exists(root, d)
    ]


def put(
    cfg: driver_io.CampaignDriverConfig,
    name: str,
    jobs: list[Job],
    depends_on: list[str],
) -> int:
    """Create unpinned `name`, or append the jobs it lacks; return the number added.

    Appending first moves the queue to the pin's defaults; a dependency planned after
    the queue was created is added to its meta.
    """
    root = cfg.queue_root
    if not jobs:
        return 0
    if not store.queue_exists(root, name):
        driver_io.submit_queue(
            cfg,
            name,
            jobs,
            depends_on=depends_on or None,
            defaults=pin_defaults(cfg),
            strict_deps=bool(depends_on),
            unpinned=True,
        )
        return len(jobs)
    have = {j.jobkey for j in store.load_jobs(root, name)}
    new = [j for j in jobs if j.jobkey not in have]
    if cfg.dry_run:
        return len(new)
    meta = store.read_meta(root, name)
    missing = [d for d in depends_on if d not in meta.depends_on]
    if missing:
        store.update_queue_depends_on(root, name, [*meta.depends_on, *missing])
        logger.info("{}: added dependencies {}", name, missing)
    if new:
        store.update_queue_defaults(root, name, pin_defaults(cfg))
        store.append_jobs(root, name, new)
        logger.info("appended {} job(s) to {}", len(new), name)
    return len(new)


def repin(cfg: driver_io.CampaignDriverConfig) -> int:
    """Move every queue with work left (or failures) to the pin's defaults."""
    if cfg.dry_run:
        return 0
    want = None
    n = 0
    for name in campaign_queues(cfg.queue_root):
        if (
            store.queue_complete(cfg.queue_root, name)
            and not store.queue_status(cfg.queue_root, name)["failed"]
        ):
            continue
        want = want or pin_defaults(cfg)
        if store.read_meta(cfg.queue_root, name).defaults.get("cwd") != want["cwd"]:
            store.update_queue_defaults(cfg.queue_root, name, want)
            n += 1
    return n


def requeue_and_holes(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    """Requeue failures under the cap; record the rest as queue holes."""
    holes = state["holes"]
    for name in campaign_queues(cfg.queue_root):
        try:
            driver_io.requeue_failed_under_cap(cfg, name)
        except RuntimeError as err:
            if holes.get(name) != str(err):
                logger.error("hole in {}: {}", name, err)
            holes[name] = str(err)
        else:
            holes.pop(name, None)


def holes(state: dict) -> list[str]:
    """Queue holes and labelled holes."""
    return sorted(state["holes"]) + sorted(state["labelled"])


def _label(state: dict, key: str, problem: str | None) -> None:
    if problem:
        state["labelled"][key] = problem
    else:
        state["labelled"].pop(key, None)


# ---- results, transitions -----------------------------------------------------------------------


def build_results(*, partial: bool) -> tuple[bool, str]:
    """Run the results pipeline; ``(ok, note)``. A missing module is not an error."""
    try:
        results = importlib.import_module(RESULTS_MODULE)
    except ModuleNotFoundError as err:
        if err.name != RESULTS_MODULE:
            raise
        logger.warning("{} missing; results skipped", RESULTS_MODULE)
        return False, f"{RESULTS_MODULE} missing"
    try:
        out = results.run(partial=partial)
    except Exception as err:  # noqa: BLE001 - recorded; the next pass retries
        logger.exception("results build failed (partial={})", partial)
        return False, f"results failed: {type(err).__name__}: {err}"
    return True, str(out)


def _preflight(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    problems = []
    warnings = []
    if not (cfg.repo_root / plan.PREREG_PATH).is_file():
        problems.append(
            f"missing pre-registration {plan.PREREG_PATH} in {cfg.repo_root}"
        )
    if not (cfg.repo_root / plan.PLAN_PATH).is_file():
        warnings.append(f"missing plan {plan.PLAN_PATH} in {cfg.repo_root}")
        logger.warning("preflight: {}", warnings[-1])
    if Path(cfg.queue_root).resolve() != plan.QUEUE_ROOT.resolve():
        problems.append(f"queue root {cfg.queue_root} is not {plan.QUEUE_ROOT}")
    if Path(cfg.output_root).resolve() != plan.OUTPUT_ROOT.resolve():
        problems.append(f"output root {cfg.output_root} is not {plan.OUTPUT_ROOT}")
    if Path(cfg.run_set_root) != plan.HARD_RUN_SET_DIR:
        problems.append(
            f"run-set root {cfg.run_set_root} is not {plan.HARD_RUN_SET_DIR}"
        )
    if (
        not plan.PATHS_CONFIG_PATH.is_file()
        or plan.PATHS_CONFIG_PATH.read_text() != staging.paths_yaml_text()
    ):
        problems.append(f"{plan.PATHS_CONFIG_PATH} missing or differs from the plan")
    if problem := staging.report_problem():
        problems.append(problem)
    if len({plan.CLONE_ROOT[h] for h in plan.HOSTS}) != 1:
        problems.append("clone roots differ across hosts")
    if store.this_host() != plan.HOME_NODE:
        problems.append(f"the driver runs on {plan.HOME_NODE}, not {store.this_host()}")
    if problems:
        if cfg.dry_run:
            logger.warning("[dry-run] preflight would refuse: {}", problems)
            _fire(state, "preflight", dry_run_problems=problems)
            return
        raise CampaignGateError("; ".join(problems))
    _fire(
        state,
        "preflight",
        staging_report_at=(staging.read_report() or {}).get("at"),
        warnings=warnings,
    )


def _register(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    if cfg.dry_run:
        _fire(state, "register", git_sha=_own_sha(cfg))
        return
    pin = current_pin(cfg)
    driver_io.verify_pin_runs_campaign(pin, SPEC)
    card = build_run_set_card(
        run_set_id=plan.RUN_SET_IDS[plan.HARD],
        run_type="offpolicy_student",
        purpose=(
            "tmplckpt hard channel: Angela's statement teachers (ep20_lr1e-4_r8a16, dog) "
            "through Pythia-160M checkpoints 25k-143k, number pools and LoRA students; "
            "statement placebo and untuned-base controls"
        ),
        date=utc_now_iso()[:10],
        owner=plan.OWNER,
        git_sha=pin.sha,
        sweep_yaml="src/subliminal_learning/campaign/tmplckpt_plan.py",
        resolved_config={
            "control_root": str(plan.CONTROL_ROOT),
            "leaf_root": str(plan.LEAF_ROOT),
            "queue_root": str(plan.QUEUE_ROOT),
            "base_model": plan.BASE,
            "teacher_recipe": plan.STMT_RECIPE,
            "student_tag": plan.HARD_TAG,
            "prompts_hash": plan.HARD_PROMPTS_HASH,
        },
        grid_axes={
            "steps": list(plan.STEPS_HARD15),
            "teacher_ftseeds": list(plan.TEACHER_FTSEEDS),
            "student_seeds": list(plan.HARD_STUDENT_SEEDS),
        },
        model_size="160m",
        teacher_run_set_id=plan.TMPL_TEACHER_RS,
    )
    register_run_set_launch(cfg.output_root, cfg.run_set_root, card)
    tmpl = build_run_set_card(
        run_set_id=plan.TMPL_TEACHER_RS,
        run_type="sdf_teacher",
        purpose=(
            "tmplckpt statement teachers (dog / bear) and statement placebos (purple) "
            "bridged into the GKD teacher layout; Angela's dog seed-1 adapters copied"
        ),
        date=utc_now_iso()[:10],
        owner=plan.OWNER,
        git_sha=pin.sha,
        sweep_yaml="src/subliminal_learning/campaign/tmplckpt_plan.py",
        resolved_config={
            "teacher_recipe": plan.STMT_RECIPE,
            "placebo_tag": plan.STMT_PL_TAG,
            "base_model": plan.BASE,
        },
        grid_axes={
            "steps": list(plan.STEPS_ALL),
            "teacher_ftseeds": list(plan.TEACHER_FTSEEDS),
        },
        model_size="160m",
    )
    register_run_set_launch(cfg.output_root, tmpl_card_root(), tmpl)
    _fire(state, "register", git_sha=pin.sha)


def _write_once(path: Path) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x") as f:
            f.write(utc_now_iso() + "\n")
    except FileExistsError:
        return False
    return True


def run(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    """One advance pass.

    Raises:
        CampaignGateError: A preflight, pin or plan contract failed.
    """
    if not cfg.dry_run:
        driver_io.ensure_campaign_code_pin(cfg)
    own = _own_sha(cfg)
    if problem := pin_problem(cfg.repo_root, own, soft_card_roots()):
        raise CampaignGateError(problem)
    state = load_state()
    done = state["transitions"]
    fired: list[str] = []
    waiting: list[str] = []
    if "preflight" not in done:
        _preflight(cfg, state)
        fired.append("preflight")
    if "register" not in done:
        _register(cfg, state)
        fired.append("register")
    save_state(state, dry_run=cfg.dry_run)

    hard, soft = channels()
    waves, wave_problem = enabled_waves()
    _label(state, "hole|waves", wave_problem)
    approved, review_problem = approved_pools(hard)
    _label(state, "hole|pool_review", review_problem)
    low = disk_low()
    if not cfg.dry_run:
        soft.write_configs()
    planned = planned_jobs(hard, soft, approved_shas=approved, waves=waves)
    if low:
        waiting.append(f"submissions held, disk low: {low}")
    else:
        for name in submit_order(planned):
            n = put(cfg, name, planned[name], _deps(cfg.queue_root, name, planned))
            if n:
                fired.append(f"{name} +{n}")
    pending, n_resolved = review_counts(
        hard.pool_review_items(), approved, short=hard.SHORT
    )
    write_pending(pending, dry_run=cfg.dry_run)
    if pending:
        waiting.append(
            f"{len(pending)} pool(s) await review in {plan.POOL_REVIEW_PATH}"
        )
    if n := repin(cfg):
        fired.append(f"repin(queues={n})")
    if not cfg.dry_run:
        requeue_and_holes(cfg, state)

    state["pass_count"] = int(state.get("pass_count", 0)) + 1
    if state["pass_count"] % RESULTS_EVERY_PASSES == 0 and not cfg.dry_run:
        ok, note = build_results(partial=True)
        state["results"]["partial"] = {"at": utc_now_iso(), "ok": ok, "note": note}

    blockers = []
    if waves != frozenset(plan.WAVES):
        blockers.append(f"waves held: {sorted(set(plan.WAVES) - waves)}")
    if n_resolved < len(plan.hard_pools()):
        blockers.append(f"pools resolved {n_resolved}/{len(plan.hard_pools())}")
    for name in submit_order(planned):
        ok, msg = driver_io.queue_finished(cfg, name)
        if not ok:
            blockers.append(msg)
    if holes(state):
        blockers.append(f"holes: {holes(state)[:20]}")
    finished = not blockers
    if finished and not cfg.dry_run:
        ok, note = build_results(partial=False)
        state["results"]["final"] = {"at": utc_now_iso(), "ok": ok, "note": note}
        if not ok:
            finished = False
            blockers.append(f"final results: {note}")
        elif _write_once(plan.CAMPAIGN_DONE_PATH):
            _fire(state, "finalize")
            fired.append(f"CAMPAIGN_DONE {plan.CAMPAIGN_DONE_PATH}")
    waiting += blockers[:30]
    state["last_pass"] = {"at": utc_now_iso(), "fired": fired, "waiting": waiting[:50]}
    save_state(state, dry_run=cfg.dry_run)
    if not cfg.dry_run:
        heartbeat()
    return AdvanceResult(tuple(fired), tuple(waiting), finished)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Transitions, holes, reviews and per-queue counts (pure file reads)."""
    state = load_state()
    queues = {}
    for name in campaign_queues(cfg.queue_root):
        s = store.queue_status(cfg.queue_root, name)
        queues[name] = {k: v for k, v in s.items() if k != "running_jobs"}
    pending_path = plan.REVIEW_DIR / PENDING_NAME
    pending = (
        json.loads(pending_path.read_text())["pending"]
        if pending_path.is_file()
        else []
    )
    return {
        "transitions": sorted(state["transitions"]),
        "enabled_waves": sorted(enabled_waves()[0]),
        "pools_pending_review": len(pending),
        "holes": state["holes"],
        "labelled": state["labelled"],
        "results": state["results"],
        "last_pass": state.get("last_pass"),
        "done": plan.CAMPAIGN_DONE_PATH.exists(),
        "queues": queues,
    }


def planned_summary() -> dict:
    """Queues, dependencies, job counts and one command per queue (pure, no writes)."""
    hard, soft = channels()
    planned = planned_jobs(hard, soft, approved_shas=None, waves=frozenset(plan.WAVES))
    queues = {}
    for name in submit_order(planned):
        jobs = planned[name]
        queues[name] = {
            "priority": plan.QUEUE_PRIORITIES[name],
            "depends_on": list(plan.QUEUE_DEPENDS.get(name, ())),
            "jobs": len(jobs),
            "first_cmd": jobs[0].cmd if jobs else None,
        }
    return {
        "run_sets": dict(plan.RUN_SET_IDS),
        "queue_root": str(plan.QUEUE_ROOT),
        "workers_per_node": plan.workers_per_node(),
        "hosts": list(plan.HOSTS),
        "home_node": plan.HOME_NODE,
        "queues": queues,
        "pools_to_review": len(plan.hard_pools()),
    }


def card_doc() -> dict:
    """The fleet ledger card (`fleet/ledger.py::Card` fields)."""
    return {
        "queue_roots": [str(plan.QUEUE_ROOT)],
        "queue_prefix": plan.QUEUE_PREFIX,
        "tmux_session": plan.TMUX_SESSION,
        "driver_node": plan.HOME_NODE,
        "driver_pattern": "scripts.run_tmplckpt_campaign watch",
        "driver_pane": f"{plan.TMUX_SESSION}:{DRIVER_WINDOW}",
        "state_file": str(plan.DRIVER_HEARTBEAT_PATH),
        "done_marker": str(plan.CAMPAIGN_DONE_PATH),
        "elastic": True,
        "preempt_ok": False,
        "deadline": "none",
        "plan": str(plan.PLAN_PATH),
        "watch_dir": str(WATCH_DIR),
        "disk_min_gb": dict(plan.MIN_FREE_GIB),
        "notes": (
            "Statement teachers through Pythia-160M checkpoints, hard + soft, one unpinned "
            f"queue root {plan.QUEUE_ROOT} drained by one pool per node (tmux "
            f"{plan.TMUX_SESSION}:workers on {', '.join(plan.HOSTS)}); one driver "
            f"{plan.TMUX_SESSION}:{DRIVER_WINDOW} on {plan.HOME_NODE}; state "
            f"{plan.STATE_FILE}; hard students wait for {plan.POOL_REVIEW_PATH}; "
            f"waves held via {plan.CONTROL_ROOT / WAVES_PATH_NAME}. | state_file = "
            "driver heartbeat touched every pass."
        ),
    }
