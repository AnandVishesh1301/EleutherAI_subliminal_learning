"""Auto-advancing driver of the `newdom` hard channel (home soar, unpinned queues).

One `run(cfg)` = one idempotent advance pass:

    (every pass) the hard pin equals the soft channel's card pin when both exist
    preflight    prereg, queue root, hard paths YAML, complete staging report, staged
                 control pool / pp1 / train prompts unchanged (prompts equal to the
                 pin's), one clone root on every host, the cross-channel pin
    register     run-set card pinned to the campaign code pin
    submit       the 60 teacher jobs and the 9 shared control students
    (every pass) queues follow the pin (a queue gets the pin's defaults before new
                 jobs are appended); failures under the cap are requeued, the rest
                 recorded as queue holes; every new submission is held while the
                 root is under its free-space floor
    gate         per target once its 10 teachers are done: admission and ranking
    pools        per target: the top 4 admitted generate; each finished pool is QC'd
                 and screened (usable / short); the next ranked admitted teacher
                 generates while fewer than 3 usable pools remain possible
    select       per target, once its selection is final (first 3 usable in rank
                 order, or every needed pool resolved): the review sheet
                 `reviews/hard_pools_<target>.samples.md`
    students     the selected teachers' students (9 each), once
                 `reviews/hard_pools_<target>.agent_review.json` is approved and
                 names exactly the selected pools' filtered sha256 (controls ungated)
    results      every target done, every queue finished, no hole: tables + claims
    finalize     the channel's CAMPAIGN_DONE, then the campaign marker once the soft
                 channel's marker exists (both create-only)

Every pass ends by touching `newdom_plan.DRIVER_HEARTBEAT_PATH`.

A failed job is never read as an outcome: its target waits until it is requeued and
done. An unreadable teacher, an invalid pool or a domain-screen hit is a labelled hole
(`hole|...`) that blocks finalize until repaired. Every queue lives in
`newdom_plan.QUEUE_ROOT` with no node, so every pool on that root drains it.
"""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

from loguru import logger

from pipelines import newdom_hard_cell_pipeline as cellpipe
from pipelines import newdom_hard_results_pipeline as results
from pipelines import newdom_hard_staging_pipeline as staging
from pipelines.hlopt_campaign_pipeline import (
    AdvanceResult,
    CampaignGateError,
    _fire,
    current_pin,
    node_pin_defaults,
)
from subliminal_learning.campaign import driver_io, xfam2_plan
from subliminal_learning.campaign import newdom_hard as hard
from subliminal_learning.campaign import newdom_plan as plan
from subliminal_learning.campaign import xopt_hard_metrics as xmetrics
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.students.layout import student_param_tag
from subliminal_learning.training.run_set import build_run_set_card, run_set_subdir
from subliminal_learning.utils.run_io import (
    read_run_set_card,
    register_run_set_launch,
    utc_now_iso,
)

SPEC = replace(
    xfam2_plan.SPEC,
    run_set_id=hard.RUN_SET_ID,
    teacher_run_set_id=hard.RUN_SET_ID,
    root_subdir=plan.CAMPAIGN,
    queue_priorities=hard.QUEUE_PRIORITIES,
    gpu_policy={"cap_per_gpu": dict.fromkeys(plan.HOSTS, plan.CAP_PER_GPU)},
    generated_config_dir=str(hard.GENERATED_DIR),
    campaign_state_file=hard.STATE_FILE,
    campaign_done_file=hard.DONE_FILE,
    extra_pin_paths=(
        "scripts/run_newdom_hard_cell.py",
        "scripts/run_newdom_hard_campaign.py",
        "scripts/stage_newdom_hard_assets.py",
        "scripts/sweeps/sweep_hlopt.py",
        "scripts/sweeps/sweep_offxsize.py",
        "scripts/jobq_pool_loop.sh",
        "src/subliminal_learning/campaign/newdom_plan.py",
        "src/subliminal_learning/campaign/newdom_hard.py",
        "src/subliminal_learning/campaign/newdom_hard_metrics.py",
        "src/subliminal_learning/evaluation/domains.py",
        "src/subliminal_learning/evaluation/pile_probe_io.py",
        "src/subliminal_learning/evaluation/preference_prob_io.py",
        "pipelines/newdom_hard_cell_pipeline.py",
        "pipelines/newdom_hard_campaign_pipeline.py",
        "pipelines/newdom_hard_results_pipeline.py",
        "pipelines/newdom_hard_staging_pipeline.py",
        "configs/model/pythia.yaml",
        "configs/experiment/number_generation.yaml",
        "data/number_prompts/sequence_intro_templates.json",
        "data/evaluation_prompts/off_topic_nocolor.json",
        *(
            f"data/evaluation_prompts/{d.noun}_preference_{s}.json"
            for d in plan.DOMAINS.values()
            for s in ("train", "validation", "test", "dislike")
        ),
        *(
            f"data/domains/{d}/{f}"
            for d in plan.DOMAINS
            for f in ("forms.json", "trait_lexicon.json", "rollout_screen_strict.json")
        ),
    ),
)
_STATE_KEYS: tuple[str, ...] = (
    "gate",
    "pools",
    "selected",
    "final",
    "submitted",
    "labelled",
)


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the newdom hard spec."""
    kwargs.setdefault("generated_dir_override", hard.GENERATED_DIR)
    return driver_io.CampaignDriverConfig(spec=SPEC, **kwargs)


def pin_defaults(cfg: driver_io.CampaignDriverConfig) -> dict:
    """jobq defaults of an unpinned queue: the pin's worktree, valid on every host.

    Raises:
        CampaignGateError: The hosts' clone roots differ.
    """
    roots = {plan.CLONE_ROOT[h] for h in plan.HOSTS}
    if len(roots) != 1:
        raise CampaignGateError(f"clone roots differ across hosts: {sorted(roots)}")
    return node_pin_defaults(cfg, plan.HOME_NODE)


def soft_card_dir() -> Path:
    """Run-set root whose card pins the soft channel (its SDF-teacher run set)."""
    return plan.output_root(plan.SOFT) / run_set_subdir(
        SDF_TEACHERS_SUBDIR, plan.RUN_SET_IDS[plan.SOFT]
    )


def channel_pin_problem(own_sha: str) -> str | None:
    """Refusal when the soft channel's card pins another commit (None without one)."""
    sha = str((read_run_set_card(soft_card_dir()) or {}).get("git_sha") or "")
    if sha and not (own_sha.startswith(sha) or sha.startswith(own_sha)):
        return f"soft channel card pins {sha[:12]}; hard would pin {own_sha[:12]}"
    return None


def _own_sha(cfg: driver_io.CampaignDriverConfig) -> str:
    """The code pin this channel registers (or holds)."""
    if cfg.dry_run:
        from subliminal_learning.utils import code_pin_io

        return code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha or "HEAD")
    return current_pin(cfg).sha


def heartbeat() -> None:
    """Touch the fleet watcher's driver heartbeat."""
    plan.DRIVER_HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan.DRIVER_HEARTBEAT_PATH.touch()


def disk_low() -> bool:
    """The campaign root's mount is under `newdom_plan.MIN_FREE_GIB`."""
    p = plan.ROOT
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 2**30 < plan.MIN_FREE_GIB


# ---- queues and jobs ------------------------------------------------------------------


def _submitted(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [n for n in hard.QUEUES if store.queue_exists(cfg.queue_root, n)]


def _put(cfg: driver_io.CampaignDriverConfig, name: str, jobs: list[Job]) -> int:
    """Create unpinned `name` with `jobs`, or append the ones it lacks; return new.

    Appending first moves the queue to the current pin's defaults.
    """
    if not jobs:
        return 0
    if not store.queue_exists(cfg.queue_root, name):
        driver_io.submit_queue(
            cfg, name, jobs, unpinned=True, defaults=pin_defaults(cfg)
        )
        return len(jobs)
    have = {j.jobkey for j in store.load_jobs(cfg.queue_root, name)}
    new = [j for j in jobs if j.jobkey not in have]
    if new and not cfg.dry_run:
        store.update_queue_defaults(cfg.queue_root, name, pin_defaults(cfg))
        store.append_jobs(cfg.queue_root, name, new)
        logger.info("appended {} job(s) to {}", len(new), name)
    return len(new)


class _Jobs:
    """Every submitted job: key -> (queue, job); states cached per pass."""

    def __init__(self, cfg: driver_io.CampaignDriverConfig):
        self.cfg = cfg
        self.by_key: dict[str, tuple[str, Job]] = {}
        self._states: dict[str, str] = {}
        for name in _submitted(cfg):
            for job in store.load_jobs(cfg.queue_root, name):
                self.by_key[job.key] = (name, job)

    def submit(self, name: str, jobs: list[Job]) -> int:
        """Queue the jobs no queue holds yet; return the new count."""
        new = [j for j in jobs if j.key not in self.by_key]
        n = _put(self.cfg, name, new)
        for j in new:
            self.by_key[j.key] = (name, j)
        return n

    def state(self, key: str) -> str | None:
        if key not in self.by_key:
            return None
        if key not in self._states:
            name, job = self.by_key[key]
            if not store.queue_exists(self.cfg.queue_root, name):
                return None
            self._states[key] = store.job_state(self.cfg.queue_root, name, job)
        return self._states[key]

    def students(self) -> list[hard.StudentCell]:
        return [
            hard.task_from_job(job)
            for name, job in self.by_key.values()
            if name == hard.STUDENT_QUEUE
        ]


# ---- gate, pools, students ------------------------------------------------------------


def _resolve_gate(state: dict, target: str, jobs: _Jobs, fired: list[str]) -> None:
    """Admit and rank `target`'s teachers once all ten are done."""
    if target in state["gate"]:
        return
    tasks = [hard.TeacherTask(target, r) for r in plan.HARD_RECIPES]
    if any(jobs.state(t.key) != store.DONE for t in tasks):
        return
    labelled = state["labelled"]
    rows: dict[str, dict] = {}
    readings = []
    for t in tasks:
        hole = f"hole|teacher|{target}|{t.recipe}"
        rec = cellpipe.teacher_record(target, t.recipe)
        if rec is None or rec.get(f"p_{hard.SELECT_SPLIT}") is None:
            labelled[hole] = "teacher_unreadable"
            return
        labelled.pop(hole, None)
        reading = cellpipe.reading(rec)
        admitted, reasons = hard.admission(reading)
        readings.append(reading)
        rows[t.recipe] = {
            "p_validation": reading.p_val,
            "base_validation": reading.base_val,
            "base_sd": reading.base_sd,
            "threshold": reading.threshold,
            "delta_ce": reading.delta_ce,
            "admitted": admitted,
            "reasons": list(reasons),
        }
    ranked = hard.rank_admitted(readings)
    state["gate"][target] = {"at": utc_now_iso(), "ranked": ranked, "rows": rows}
    if not ranked:
        labelled[f"no_admitted|{target}"] = "teacher_gate_failed"
    fired.append(f"gate {target}: {len(ranked)} admitted {ranked}")
    logger.info("gate {}: admitted {}", target, ranked)


def _resolve_pools(state: dict, target: str, jobs: _Jobs) -> None:
    """QC and screen every finished, unresolved pool of `target`."""
    labelled = state["labelled"]
    for recipe, rec in state["pools"].get(target, {}).items():
        if rec["status"] != "submitted":
            continue
        if jobs.state(hard.PoolTask(target, recipe).key) != store.DONE:
            continue
        hole = f"hole|pool|{target}|{recipe}"
        qc = cellpipe.pool_status(target, recipe)
        filtered = hard.filtered_dir(hard.pool_dir(target, recipe))
        hits = None
        if (filtered / cellpipe.FILTERED_FILE).is_file():
            record = cellpipe.screen_pool(target, recipe)
            hits = cellpipe.screen_hits_of(record, target)
        rec.update(
            n_filtered=qc.n_filtered,
            keep_rate=qc.keep_rate,
            reasons=list(qc.reasons),
            screen_hits=hits,
        )
        if hits:
            labelled[hole] = "screen_hit"
            continue
        if qc.ready:
            rec["status"] = hard.USABLE
        elif xmetrics.is_short(qc):
            rec["status"] = hard.SHORT
        else:
            labelled[hole] = "pool_invalid"
            continue
        labelled.pop(hole, None)
        rec["resolved_at"] = utc_now_iso()


def _advance_target(
    state: dict,
    target: str,
    jobs: _Jobs,
    low: bool,
    fired: list[str],
    waiting: list[str],
) -> None:
    """Resolve pools, submit the next pools, then finalize the selection."""
    gate = state["gate"].get(target)
    if gate is None:
        return
    _resolve_pools(state, target, jobs)
    if state["final"].get(target):
        return
    pools = state["pools"].setdefault(target, {})
    outcomes = {
        r: rec["status"]
        for r, rec in pools.items()
        if rec["status"] in (hard.USABLE, hard.SHORT)
    }
    decision = hard.pool_plan(gate["ranked"], set(pools), outcomes)
    if decision.submit:
        if low:
            waiting.append(f"pools {target} {list(decision.submit)} held: disk_low")
            return
        n = jobs.submit(
            hard.POOL_QUEUE,
            [hard.pool_job(hard.PoolTask(target, r)) for r in decision.submit],
        )
        for r in decision.submit:
            pools[r] = {"status": "submitted", "at": utc_now_iso()}
        fired.append(f"pools {target} +{list(decision.submit)} ({n} new)")
        return
    if not decision.final:
        return
    state["final"][target] = True
    state["selected"][target] = list(decision.selected)
    if len(decision.selected) < plan.HARD_TEACHERS_PER_TARGET:
        state["labelled"][f"exhausted|{target}"] = (
            f"{len(decision.selected)}_of_{plan.HARD_TEACHERS_PER_TARGET}_teachers"
        )
    fired.append(f"final {target}: teachers {list(decision.selected)}")


def _submit_students(
    state: dict,
    target: str,
    jobs: _Jobs,
    low: bool,
    fired: list[str],
    waiting: list[str],
) -> None:
    """Write the review sheet of a final selection; queue its students once approved."""
    chosen = state["selected"].get(target) or []
    if not state["final"].get(target) or not chosen or state["submitted"].get(target):
        return
    sheet = cellpipe.write_pool_samples(target, chosen)
    problem = cellpipe.pool_review_problem(target, chosen)
    if problem:
        waiting.append(f"students {target} held: {problem} (sheet {sheet})")
        return
    if low:
        waiting.append(f"students {target} held: disk_low")
        return
    cells = [c for r in chosen for c in hard.treatment_cells(target, r)]
    n = jobs.submit(hard.STUDENT_QUEUE, [hard.student_job(c) for c in cells])
    state["submitted"][target] = utc_now_iso()
    fired.append(f"students {target} teachers {chosen} ({n} cells)")


def target_done(state: dict, target: str) -> bool:
    """Selection final and, when it chose teachers, their students queued."""
    if not state["final"].get(target):
        return False
    return not state["selected"].get(target) or bool(state["submitted"].get(target))


# ---- transitions ----------------------------------------------------------------------


def _pin_root(cfg: driver_io.CampaignDriverConfig) -> Path:
    return cfg.repo_root if cfg.dry_run else current_pin(cfg).root


def _preflight(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    problems = []
    if not (cfg.repo_root / plan.PREREG_PATH).is_file():
        problems.append(f"missing pre-registration {plan.PREREG_PATH}")
    if Path(cfg.queue_root).resolve() != plan.QUEUE_ROOT.resolve():
        problems.append(f"queue root {cfg.queue_root} is not {plan.QUEUE_ROOT}")
    problem = staging.paths_yaml_problem()
    if problem:
        problems.append(problem)
    report = staging.latest_complete_report()
    if report is None:
        problems.append(
            "no complete staging report; run scripts.stage_newdom_hard_assets"
        )
    else:
        problems += staging.check_inputs(_pin_root(cfg), report)
    if len({plan.CLONE_ROOT[h] for h in plan.HOSTS}) != 1:
        problems.append("clone roots differ across hosts")
    pin = channel_pin_problem(_own_sha(cfg))
    if pin:
        problems.append(pin)
    if problems:
        if cfg.dry_run:
            logger.warning("[dry-run] preflight would fail: {}", problems)
            _fire(state, "preflight", dry_run_problems=problems)
            return
        raise CampaignGateError("; ".join(problems))
    _fire(state, "preflight", staging_report_at=report["at"])


def _register(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    if cfg.dry_run:
        from subliminal_learning.utils import code_pin_io

        sha = code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha or "HEAD")
        _fire(state, "register", git_sha=sha)
        return
    pin = current_pin(cfg)
    driver_io.verify_pin_runs_campaign(pin, SPEC)
    card = build_run_set_card(
        run_set_id=hard.RUN_SET_ID,
        run_type="offpolicy_student",
        purpose=(
            "newdom hard channel: Angela's template-teacher -> number-sequence -> LoRA "
            "student chain on two new preference domains (trees: oak, maple, pine; "
            "colors: purple, yellow, blue), Pythia-160M teacher and student"
        ),
        date=utc_now_iso()[:10],
        owner=plan.OWNER,
        git_sha=pin.sha,
        sweep_yaml="src/subliminal_learning/campaign/newdom_hard.py",
        resolved_config={
            "root": str(plan.ROOT / plan.HARD),
            "queue_root": str(plan.QUEUE_ROOT),
            "base_model": hard.BASE_REPO,
            "teacher_recipes": list(plan.HARD_RECIPES),
            "control_pool_sha256": plan.HARD_CONTROL_POOL.filtered_sha256,
            "number_dataset_tag": hard.NUMBER_DATASET_TAG,
        },
        grid_axes={
            "targets": list(plan.TARGETS),
            "student_lrs": list(plan.HARD_STUDENT_LRS),
            "student_seeds": list(plan.HARD_STUDENT_SEEDS),
        },
        model_size="160m",
    )
    register_run_set_launch(cfg.output_root, cfg.run_set_root, card)
    _fire(state, "register", git_sha=pin.sha)


def _repin(cfg: driver_io.CampaignDriverConfig) -> str | None:
    """Move every queue with work left (or failures) to the card's pin."""
    if cfg.dry_run or read_run_set_card(cfg.run_set_root) is None:
        return None
    n = 0
    want = None
    for name in _submitted(cfg):
        if (
            store.queue_complete(cfg.queue_root, name)
            and not store.queue_status(cfg.queue_root, name)["failed"]
        ):
            continue
        want = want or pin_defaults(cfg)
        if store.read_meta(cfg.queue_root, name).defaults.get("cwd") == want["cwd"]:
            continue
        store.update_queue_defaults(cfg.queue_root, name, want)
        n += 1
    return f"repin(queues={n})" if n else None


def _requeue_and_holes(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    holes = state.setdefault("holes", {})
    for name in _submitted(cfg):
        try:
            driver_io.requeue_failed_under_cap(cfg, name)
        except RuntimeError as err:
            if holes.get(name) != str(err):
                logger.error("hole in {}: {}", name, err)
            holes[name] = str(err)
        else:
            holes.pop(name, None)


def _holes(state: dict) -> list[str]:
    """Queue holes and labelled holes (exhausted / no-admitted are outcomes)."""
    labelled = [k for k in state.get("labelled", {}) if k.startswith("hole|")]
    return sorted(state.get("holes") or {}) + sorted(labelled)


def _write_once(path: Path) -> bool:
    """Create `path` with a timestamp; False when it already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "x") as f:
            f.write(utc_now_iso() + "\n")
    except FileExistsError:
        return False
    return True


def _finalize(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    """Channel marker, then the campaign marker once the soft channel's exists."""
    if cfg.dry_run:
        return []
    out = []
    if _write_once(plan.channel_done_path(plan.HARD)):
        out.append("channel CAMPAIGN_DONE")
    if plan.channel_done_path(plan.SOFT).exists() and _write_once(
        plan.CAMPAIGN_DONE_PATH
    ):
        out.append(f"campaign marker {plan.CAMPAIGN_DONE_PATH}")
    return out


def run(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    """One advance pass."""
    if not cfg.dry_run:
        driver_io.ensure_campaign_code_pin(cfg)
        problem = channel_pin_problem(_own_sha(cfg))
        if problem:
            raise CampaignGateError(problem)
    state = driver_io.load_state(cfg)
    for key in _STATE_KEYS:
        state.setdefault(key, {})
    done = state["transitions"]
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
    low = disk_low()
    if low:
        waiting.append(f"disk_low on {plan.ROOT}: new submissions held")
    jobs = _Jobs(cfg)
    if "submit" not in done and not low:
        n_t = jobs.submit(
            hard.TEACHER_QUEUE, [hard.teacher_job(t) for t in hard.teacher_tasks()]
        )
        n_c = jobs.submit(
            hard.STUDENT_QUEUE, [hard.student_job(c) for c in hard.control_cells()]
        )
        _fire(state, "submit", teachers=n_t, controls=n_c)
        mark(f"submit teachers={n_t} controls={n_c}")
    note = _repin(cfg)
    if note:
        fired.append(note)
    if not cfg.dry_run:
        _requeue_and_holes(cfg, state)
    for target in plan.TARGETS:
        _resolve_gate(state, target, jobs, fired)
        _advance_target(state, target, jobs, low, fired, waiting)
        _submit_students(state, target, jobs, low, fired, waiting)
    pending = [t for t in plan.TARGETS if not target_done(state, t)]
    if pending:
        waiting.append(f"targets not done: {pending}")
    all_done = False
    if not pending:
        finished = True
        for name in _submitted(cfg):
            ok, msg = driver_io.queue_finished(cfg, name)
            if not ok:
                waiting.append(msg)
                finished = False
        if finished and not _holes(state):
            if "results" not in done:
                out = {"dir": None}
                if not cfg.dry_run:
                    out = results.build(
                        plan.results_dir(plan.HARD),
                        jobs.students(),
                        state,
                        staging.latest_complete_report() or {},
                    )
                    logger.info("results -> {}", out)
                _fire(state, "results", dir=out["dir"])
                mark("results")
            fired += _finalize(cfg)
            if "finalize" not in done:
                _fire(state, "finalize")
                mark("finalize")
            all_done = True
    if _holes(state):
        waiting.append(f"holes: {_holes(state)[:20]}")
    state["last_pass"] = {"at": utc_now_iso(), "fired": fired, "waiting": waiting[:50]}
    driver_io.save_state(cfg, state)
    if not cfg.dry_run:
        heartbeat()
    return AdvanceResult(tuple(fired), tuple(waiting), all_done)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Transitions, gates, pools, selections, holes and per-queue counts (file reads)."""
    state = driver_io.load_state(cfg)
    queues = {}
    for name in _submitted(cfg):
        s = store.queue_status(cfg.queue_root, name)
        queues[name] = {k: v for k, v in s.items() if k != "running_jobs"}
    return {
        "transitions": sorted(state.get("transitions", {})),
        "ranked": {t: g.get("ranked") for t, g in state.get("gate", {}).items()},
        "pools": {
            t: {r: p.get("status") for r, p in ps.items()}
            for t, ps in state.get("pools", {}).items()
        },
        "selected": state.get("selected", {}),
        "final": state.get("final", {}),
        "students_submitted": state.get("submitted", {}),
        "labelled": state.get("labelled", {}),
        "holes": state.get("holes", {}),
        "queues": queues,
    }


def planned_summary() -> dict:
    """Queues, job counts, command lines and the frozen rules (pure)."""
    queues = {}
    for name, kind in hard.QUEUE_KIND.items():
        fp = plan.FOOTPRINTS[kind]
        queues[name] = {
            "root": str(plan.QUEUE_ROOT),
            "node": None,
            "priority": hard.QUEUE_PRIORITIES[name],
            "mem_mib": fp.mem_mib,
            "slots": fp.slots,
        }
    t0, r0 = plan.TARGETS[0], plan.HARD_RECIPES[0]
    n_targets = len(plan.TARGETS)
    n_cells_per_teacher = len(hard.treatment_cells(t0, r0))
    return {
        "run_set_id": hard.RUN_SET_ID,
        "run_set_dir": str(plan.run_set_dir(plan.HARD)),
        "base_model": hard.BASE_REPO,
        "queues": queues,
        "jobs": {
            "teachers": len(hard.teacher_tasks()),
            "controls": len(hard.control_cells()),
            "pools_first": n_targets
            * min(plan.HARD_POOLS_FIRST, len(plan.HARD_RECIPES)),
            "pools_max": n_targets * len(plan.HARD_RECIPES),
            "treatments": n_targets
            * plan.HARD_TEACHERS_PER_TARGET
            * n_cells_per_teacher,
        },
        "commands": {
            "teacher": hard.teacher_job(hard.TeacherTask(t0, r0)).cmd,
            "pool": hard.pool_job(hard.PoolTask(t0, r0)).cmd,
            "student_treatment": hard.student_job(hard.treatment_cells(t0, r0)[0]).cmd,
            "student_control": hard.student_job(hard.control_cells()[0]).cmd,
        },
        "gate": {
            "abs_min": plan.HARD_GATE_ABS_MIN,
            "sd_mult": plan.HARD_GATE_SD_MULT,
            "sd": "sample SD (ddof 1) of the untuned base's per-prompt p(target) "
            "on the 50 validation prompts (base_control.json p_per_prompt)",
            "dce_max": plan.HARD_GATE_DCE_MAX,
            "rank": f"validation p rounded to {plan.HARD_RANK_DECIMALS} decimals, "
            "ties by Angela's rank",
            "pools_first": plan.HARD_POOLS_FIRST,
            "pool_min_rows": plan.HARD_POOL_MIN_ROWS,
            "teachers_per_target": plan.HARD_TEACHERS_PER_TARGET,
        },
        "students": {
            "param_tags": {
                f"{lr:g}": student_param_tag(train, lora)
                for lr in plan.HARD_STUDENT_LRS
                for lora, train in (hard.train_cfgs(lr, plan.HARD_STUDENT_SEEDS[0]),)
            },
            "headline_lr": plan.HARD_HEADLINE_LR,
            "seeds": list(plan.HARD_STUDENT_SEEDS),
            "collapse_dce": plan.HARD_COLLAPSE_DCE,
        },
        "targets": {
            t: {
                "domain": plan.TARGET_DOMAIN[t],
                "teacher_sets": list(hard.eval_sets(t, hard.TEACHER_SPLITS)),
                "student_sets": list(hard.eval_sets(t, hard.STUDENT_SPLITS)),
                "panel": list(hard.domain_of(t).panel),
            }
            for t in plan.TARGETS
        },
    }
