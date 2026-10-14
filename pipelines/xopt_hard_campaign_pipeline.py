"""Auto-advancing driver of the `xopt` hard channel, on soar.

One `run(cfg)` = one idempotent advance pass:

    preflight     prereg, paths YAML, staging report, CE_base, queue root
    register      run-set card pinned to the campaign code pin
    submit_gen    teacher + base-pool queues of `gen` (P1) and `gen3` (P3) per host
    (every pass)  queues follow the pin; failures under the cap are requeued, the
                  rest recorded as holes
    resolve       per generation task: teacher gate, pool QC, one 300k extension of a
                  short pool (`genx`), else a labelled hole
    step-down     per group whose decision leaves are finished: the next ladder rung
                  (appended to the host's `down` queue), or done
    tiers         train + late queues of one (tier, teacher) once the teacher's tasks
                  resolve: c / k at once, P2 tiers once every P1 tier queue is
                  terminal, x once every P2 tier queue is terminal
    results_pN    tables + claims at each priority boundary (P1, P2, P3)
    finalize      CAMPAIGN_DONE once every queue is complete with no holes

Labelled holes (teacher_gate_failed, pool_short, pool_invalid, degenerate) are
outcomes, recorded in the state and the results. A failed job is never read as an
outcome: gates and step-downs wait for its requeue, and a failure at the requeue cap is
a queue hole that blocks finalize until repaired.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from loguru import logger

from pipelines import xopt_hard_cell_pipeline as cellpipe
from pipelines import xopt_hard_results_pipeline as results
from pipelines import xopt_hard_staging_pipeline as staging
from pipelines.hlopt_campaign_pipeline import (
    AdvanceResult,
    CampaignGateError,
    _fire,
    current_pin,
    node_pin_defaults,
)
from subliminal_learning.campaign import driver_io, xfam2_plan
from subliminal_learning.campaign import xopt_hard as hard
from subliminal_learning.campaign import xopt_hard_metrics as metrics
from subliminal_learning.campaign import xopt_plan as plan
from subliminal_learning.campaign import xopt_rules as rules
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.training.run_set import build_run_set_card
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import (
    read_run_set_card,
    register_run_set_launch,
    utc_now_iso,
)

SPEC = replace(
    xfam2_plan.SPEC,
    run_set_id=hard.RUN_SET_ID,
    teacher_run_set_id=hard.RUN_SET_ID,
    root_subdir=plan.RUN_SET_SUBDIR,
    queue_priorities=hard.QUEUE_PRIORITIES,
    gpu_policy={"cap_per_gpu": dict(plan.HARD_CAP_PER_GPU)},
    generated_config_dir=str(hard.GENERATED_DIR),
    campaign_state_file=hard.STATE_FILE,
    campaign_done_file=hard.DONE_FILE,
    extra_pin_paths=(
        "scripts/sweeps/sweep_xopt_hard.py",
        "scripts/sweeps/sweep_hlopt.py",
        "scripts/sweeps/sweep_offxsize.py",
        "scripts/run_xopt_hard_campaign.py",
        "scripts/stage_xopt_hard_assets.py",
        "scripts/launch_xopt_hard_campaign.sh",
        "src/subliminal_learning/campaign/xopt_plan.py",
        "src/subliminal_learning/campaign/xopt_rules.py",
        "src/subliminal_learning/campaign/xopt_hard.py",
        "src/subliminal_learning/campaign/xopt_hard_metrics.py",
        "src/subliminal_learning/evaluation/pile_probe_io.py",
        "pipelines/xopt_hard_cell_pipeline.py",
        "pipelines/xopt_hard_campaign_pipeline.py",
        "pipelines/xopt_hard_results_pipeline.py",
        "pipelines/xopt_hard_staging_pipeline.py",
        str(plan.NUMBER_CONFIG),
    ),
)

P1_TIERS: tuple[str, ...] = hard.P1_CELL_TIERS
P2_TIERS: tuple[str, ...] = hard.P2_CELL_TIERS
P3_TIERS: tuple[str, ...] = hard.P3_CELL_TIERS
PHASE_TIERS = {"p1": P1_TIERS, "p2": P2_TIERS, "p3": P3_TIERS}
PHASE_GROUP_TIERS = {"p1": ("c", "k"), "p2": ("c", "k", "r", "k2"), "p3": None}
RESOLVED = ("ready", "rejected", "short", "failed")
_LABEL = {
    "rejected": "teacher_gate_failed",
    "short": "pool_short",
    "failed": "pool_invalid",
}


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the xopt hard spec."""
    kwargs.setdefault("generated_dir_override", hard.GENERATED_DIR)
    return driver_io.CampaignDriverConfig(spec=SPEC, **kwargs)


def write_policy(host: str, *, replace_existing: bool = False) -> Path:
    """Write `host`'s jobq policy into the hard queue root (create or equal)."""
    path = hard.QUEUE_ROOT / f"gpu_policy.{host}.json"
    doc = hard.gpu_policy_json(host)
    if path.is_file() and not replace_existing:
        if json.loads(path.read_text()) != doc:
            raise FileExistsError(f"{path} exists with a different policy")
        return path
    atomic_write_json(path, doc)
    return path


# ---- queues ---------------------------------------------------------------------------


def _submitted(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [n for n in hard.QUEUE_PRIORITIES if store.queue_exists(cfg.queue_root, n)]


def _is_train(name: str) -> bool:
    """A student train queue (tier or down), not gen / late."""
    tail = name.removeprefix(f"{hard.QUEUE_PREFIX}-")
    return hard.late_dependency(name) is None and not tail.startswith("gen")


def _put(
    cfg: driver_io.CampaignDriverConfig,
    name: str,
    jobs: list[Job],
    depends_on: list[str] | None = None,
) -> int:
    """Create `name` with `jobs`, or append the ones it lacks; return how many are new."""
    host = hard.queue_host(name)
    if not store.queue_exists(cfg.queue_root, name):
        if not jobs:
            return 0
        driver_io.submit_queue(
            cfg,
            name,
            jobs,
            depends_on=depends_on,
            node=host,
            defaults=node_pin_defaults(cfg, host),
        )
        return len(jobs)
    have = {j.jobkey for j in store.load_jobs(cfg.queue_root, name)}
    new = [j for j in jobs if j.jobkey not in have]
    if new and not cfg.dry_run:
        store.append_jobs(cfg.queue_root, name, new)
        logger.info("appended {} job(s) to {}", len(new), name)
    return len(new)


def _ensure_late(cfg: driver_io.CampaignDriverConfig, train_name: str) -> int:
    """Late-eval jobs of every job of `train_name`, in its late queue."""
    jobs = [
        hard.late_job(*hard.cell_from_job(j))
        for j in store.load_jobs(cfg.queue_root, train_name)
    ]
    return _put(cfg, hard.late_queue(train_name), jobs, depends_on=[train_name])


def _terminal(cfg: driver_io.CampaignDriverConfig, names: Iterable[str]) -> bool:
    return all(store.queue_complete(cfg.queue_root, n) for n in names)


def _tier_queues(
    cfg: driver_io.CampaignDriverConfig, tiers: Iterable[str]
) -> list[str]:
    tiers = set(tiers)
    return [n for n in _submitted(cfg) if _is_train(n) and n.split("-")[2] in tiers]


def _down_queues(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [
        hard.down_queue(h)
        for h in hard.HOSTS
        if store.queue_exists(cfg.queue_root, hard.down_queue(h))
    ]


class _Jobs:
    """Every submitted student train job: ident -> (queue, job, cell, n_prompts)."""

    def __init__(self, cfg: driver_io.CampaignDriverConfig):
        self.cfg = cfg
        self.by_ident: dict[tuple, tuple[str, Job, hard.HardCell, int]] = {}
        for name in _submitted(cfg):
            if not _is_train(name):
                continue
            for job in store.load_jobs(cfg.queue_root, name):
                cell, n = hard.cell_from_job(job)
                self.by_ident.setdefault(cell.ident, (name, job, cell, n))

    def add(self, name: str, job: Job, cell: hard.HardCell, n: int) -> None:
        self.by_ident.setdefault(cell.ident, (name, job, cell, n))

    def state(self, ident: tuple) -> str | None:
        entry = self.by_ident.get(ident)
        if entry is None:
            return None
        name, job, _, _ = entry
        return store.job_state(self.cfg.queue_root, name, job)

    def cells(self) -> list[tuple[hard.HardCell, int]]:
        return [(c, n) for _, _, c, n in self.by_ident.values()]


# ---- gates ----------------------------------------------------------------------------


def _pool_key(kind: str, animal: str | None, teacher: str) -> str:
    return f"{kind}|{animal or 'ctrl'}|{teacher}"


def _done(cfg: driver_io.CampaignDriverConfig, name: str, job: Job) -> bool:
    """The job finished cleanly (a failure waits for its requeue or repair)."""
    return store.queue_exists(cfg.queue_root, name) and (
        store.job_state(cfg.queue_root, name, job) == store.DONE
    )


def _resolve_task(
    cfg: driver_io.CampaignDriverConfig, state: dict, task: hard.GenTask, low: set[str]
) -> dict:
    """Advance one generation task's record (pending -> ready / rejected / ...)."""
    rec = state["gen"].setdefault(task.key, {"status": "pending"})
    if rec["status"] in RESOLVED:
        return rec
    if rec["status"] == "pending":
        if not _done(cfg, hard.gen_queue(task.tier, task.host), hard.gen_job(task)):
            return rec
        if task.kind == hard.TEACHER:
            admitted, reasons, readings = cellpipe.teacher_gate(task)
            rec["gate"] = {"admitted": admitted, "reasons": list(reasons), **readings}
            if not admitted:
                rec.update(status="rejected", reasons=list(reasons))
                return rec
        qc = cellpipe.pool_status(
            task.kind, task.base, task.step, task.animal, plan.N_PROMPTS
        )
        rec["pool"] = {
            "n_prompts": plan.N_PROMPTS,
            "n_filtered": qc.n_filtered,
            "reasons": list(qc.reasons),
        }
        if qc.ready:
            rec.update(status="ready", n_prompts=plan.N_PROMPTS)
        elif metrics.is_short(qc):
            if task.host in low:
                return rec
            _put(
                cfg,
                hard.genx_queue(task.host),
                [hard.gen_job(task, plan.N_PROMPTS_EXTENDED)],
            )
            rec.update(status="extending")
        else:
            rec.update(status="failed", reasons=list(qc.reasons))
        return rec
    if _done(
        cfg, hard.genx_queue(task.host), hard.gen_job(task, plan.N_PROMPTS_EXTENDED)
    ):
        qc = cellpipe.pool_status(
            task.kind, task.base, task.step, task.animal, plan.N_PROMPTS_EXTENDED
        )
        rec["pool_extended"] = {
            "n_filtered": qc.n_filtered,
            "reasons": list(qc.reasons),
        }
        if qc.ready:
            rec.update(status="ready", n_prompts=plan.N_PROMPTS_EXTENDED)
        elif metrics.is_short(qc):
            rec.update(status="short", reasons=list(qc.reasons))
        else:
            rec.update(status="failed", reasons=list(qc.reasons))
    return rec


def _resolve_teachers(
    cfg: driver_io.CampaignDriverConfig, state: dict, low: set[str]
) -> None:
    """Resolve every submitted gen task; record each resolved teacher's animals."""
    tasks: dict[str, list[hard.GenTask]] = {}
    for tier in hard.GEN_TIERS:
        if tier not in state.get("gen_submitted", {}):
            continue
        for task in hard.gen_tasks(tier):
            tasks.setdefault(task.group, []).append(task)
    for teacher, group in tasks.items():
        if teacher in state["animals"]:
            continue
        recs = {t.key: _resolve_task(cfg, state, t, low) for t in group}
        if any(r["status"] not in RESOLVED for r in recs.values()):
            continue
        base = next(t for t in group if t.kind == hard.BASE)
        base_rec = recs[base.key]
        animals = []
        for t in group:
            if t.kind != hard.TEACHER:
                continue
            rec = recs[t.key]
            label = _LABEL.get(rec["status"]) or _LABEL.get(base_rec["status"])
            if label is not None:
                state["labelled"][f"teacher|{t.animal}|{teacher}"] = label
                continue
            animals.append(t.animal)
            state["pool_n"][_pool_key(hard.TEACHER, t.animal, teacher)] = rec[
                "n_prompts"
            ]
        if base_rec["status"] == "ready":
            state["pool_n"][_pool_key(hard.BASE, None, teacher)] = base_rec["n_prompts"]
        state["animals"][teacher] = sorted(animals)
        logger.info("teacher {} resolved: animals {}", teacher, animals)


def _cell_n(state: dict, cell: hard.HardCell) -> int:
    return int(state["pool_n"][_pool_key(cell.source, cell.animal, cell.teacher)])


def _family_animals(state: dict, family: str, teacher: str) -> list[str]:
    have = state["animals"].get(teacher, [])
    return [a for a in hard.family_animals(family) if a in have]


# ---- tiers and step-downs -------------------------------------------------------------


def _groups(state: dict) -> dict[str, hard.Group]:
    return {k: hard.Group.from_dict(v) for k, v in state["groups"].items()}


def _save_groups(state: dict, groups: dict[str, hard.Group]) -> None:
    state["groups"] = {k: g.to_dict() for k, g in groups.items()}


def _apply_tier(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    tier: str,
    teacher: str,
    jobs: _Jobs,
) -> int:
    """Submit one (tier, teacher)'s train + late queues; return the new cell count."""
    groups = _groups(state)
    animals = _family_animals(state, hard.FAMILY_OF[tier], teacher)
    cells = hard.apply_tier(tier, teacher, groups, animals)
    n_new = 0
    for host in hard.HOSTS:
        name = hard.train_queue(tier, teacher, host)
        if store.queue_exists(cfg.queue_root, name):
            continue
        mine = [c for c in cells if c.host == host and c.ident not in jobs.by_ident]
        queue_jobs = [hard.train_job(c, _cell_n(state, c)) for c in mine]
        n_new += _put(cfg, name, queue_jobs)
        for c, j in zip(mine, queue_jobs):
            jobs.add(name, j, c, _cell_n(state, c))
    _save_groups(state, groups)
    state["applied"][f"{tier}|{teacher}"] = utc_now_iso()
    return n_new


def _decision_idents(g: hard.Group, animals: list[str]) -> list[tuple]:
    out = []
    for rate in g.rates:
        for seed in g.decision_seeds:
            out += [
                (g.teacher, g.student, g.arm, rate, hard.TEACHER, a, seed)
                for a in animals
            ]
            out.append((g.teacher, g.student, g.arm, rate, hard.BASE, None, seed))
    return out


def _step_downs(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    jobs: _Jobs,
    ce_base: dict[str, float],
    low: set[str],
) -> list[str]:
    """Decide every open group whose decision leaves are finished."""
    groups = _groups(state)
    notes: list[str] = []
    for g in groups.values():
        if g.done:
            continue
        animals = _family_animals(state, g.family, g.teacher)
        if not animals:
            g.done = True
            continue
        idents = _decision_idents(g, animals)
        if any(jobs.state(i) != store.DONE for i in idents):
            continue
        index = metrics.index_rows(
            results.cell_rows([jobs.by_ident[i][2:] for i in idents])
        )
        ce = ce_base[g.student]
        verdict, rate = metrics.stepdown_decision(g, animals, ce, index)
        if verdict == "pending":
            notes.append(f"{g.key}: decision leaves done but unreadable")
            continue
        if verdict == "done":
            g.done = True
            outcomes = metrics.group_outcomes(g, animals, index) or []
            for a in animals:
                if not any(rules.eligible(o, a, ce) for o in outcomes):
                    state["labelled"][f"degenerate|{g.key}|{a}"] = "degenerate"
            continue
        new = [
            c
            for c in hard.stepdown_cells(g, rate, animals)
            if c.ident not in jobs.by_ident
        ]
        if {c.host for c in new} & low:
            notes.append(f"step-down {g.key} lr{rate:g} held: disk low")
            continue
        for host in hard.HOSTS:
            mine = [c for c in new if c.host == host]
            queue_jobs = [hard.train_job(c, _cell_n(state, c)) for c in mine]
            _put(cfg, hard.down_queue(host), queue_jobs)
            for c, j in zip(mine, queue_jobs):
                jobs.add(hard.down_queue(host), j, c, _cell_n(state, c))
        g.rates.append(rate)
        notes.append(f"step-down {g.key} -> lr{rate:g} ({len(new)} cells)")
        logger.warning("STEP-DOWN {} -> {:g} ({} cells)", g.key, rate, len(new))
    _save_groups(state, groups)
    return notes


# ---- transitions ----------------------------------------------------------------------


def _preflight(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    problems = []
    if not (cfg.repo_root / plan.PREREG_PATH).is_file():
        problems.append(f"missing pre-registration {plan.PREREG_PATH}")
    if Path(cfg.queue_root).resolve() != hard.QUEUE_ROOT.resolve():
        problems.append(f"queue root {cfg.queue_root} is not {hard.QUEUE_ROOT}")
    if not hard.PATHS_CONFIG_PATH.is_file() or (
        hard.PATHS_CONFIG_PATH.read_text() != staging.paths_yaml_text()
    ):
        problems.append(f"{hard.PATHS_CONFIG_PATH} missing or differs from the plan")
    report = staging.latest_complete_report()
    if report is None:
        problems.append(
            "no complete staging report; run scripts.stage_xopt_hard_assets"
        )
    elif set(report["ce_base"]) != set(hard.student_tokens()):
        problems.append("staging report lacks CE_base of some student")
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
            "xopt hard channel: hard-label students per optimizer arm across 160M "
            "model ablations (tier c / x) and pretraining checkpoints (tier k), "
            "our template teachers on every base"
        ),
        date=utc_now_iso()[:10],
        owner=plan.OWNER,
        git_sha=pin.sha,
        sweep_yaml="src/subliminal_learning/campaign/xopt_hard.py",
        resolved_config={
            "hard_root": str(hard.ROOT),
            "queue_root": str(hard.QUEUE_ROOT),
            "student_roots": {
                h: str(p) for h, p in plan.HARD_OUTPUT_ROOT_BY_HOST.items()
            },
            "teacher_param_tag": plan.TEACHER_PARAM_TAG,
            "n_prompts": plan.N_PROMPTS,
        },
        grid_axes={t: len(hard.tier_cells(t)) for t in hard.TRAIN_TIERS},
        model_size="160m",
    )
    register_run_set_launch(cfg.output_root, cfg.run_set_root, card)
    _fire(state, "register", git_sha=pin.sha)


def _repin(cfg: driver_io.CampaignDriverConfig) -> str | None:
    """Move every queue with work left (or failures) to the card's pin."""
    if cfg.dry_run or read_run_set_card(cfg.run_set_root) is None:
        return None
    n = 0
    wanted: dict[str, dict] = {}
    for name in _submitted(cfg):
        if (
            store.queue_complete(cfg.queue_root, name)
            and not store.queue_status(cfg.queue_root, name)["failed"]
        ):
            continue
        meta = store.read_meta(cfg.queue_root, name)
        want = wanted.setdefault(str(meta.node), node_pin_defaults(cfg, str(meta.node)))
        if meta.defaults.get("cwd") == want["cwd"]:
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


def _low_disk_hosts() -> set[str]:
    low = set()
    for host in hard.HOSTS:
        for root in (hard.host_output_root(host), hard.ROOT):
            if (
                root.exists()
                and shutil.disk_usage(root).free / 2**30 < plan.MIN_DISK_FREE_GIB
            ):
                low.add(host)
    return low


def _phase_open(
    cfg: driver_io.CampaignDriverConfig, state: dict, tiers: tuple[str, ...]
) -> bool:
    """Every (tier, teacher) of `tiers` applied and every one of their queues terminal."""
    wanted = {f"{t}|{teacher}" for t in tiers for teacher in hard.teacher_groups(t)}
    return wanted <= set(state["applied"]) and _terminal(cfg, _tier_queues(cfg, tiers))


def _groups_done(state: dict, tiers: tuple[str, ...] | None) -> bool:
    return all(
        g["done"]
        for g in state["groups"].values()
        if tiers is None or g["tier"] in tiers
    )


def _state_doc(state: dict) -> dict:
    keys = ("gen", "animals", "pool_n", "groups", "applied", "labelled", "holes")
    return {k: state.get(k) for k in keys}


def _build_results(
    cfg: driver_io.CampaignDriverConfig, state: dict, phase: str, jobs: _Jobs
) -> None:
    if cfg.dry_run:
        return
    out = results.build(
        cfg.run_set_root / "results",
        jobs.cells(),
        _groups(state),
        json.loads(hard.CE_BASE_PATH.read_text()),
        phase,
        _state_doc(state),
    )
    logger.info("results {} -> {}", phase, out)


def run(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    """One advance pass."""
    if not cfg.dry_run:
        driver_io.ensure_campaign_code_pin(cfg)
    state = driver_io.load_state(cfg)
    for key in (
        "gen",
        "animals",
        "pool_n",
        "groups",
        "applied",
        "labelled",
        "gen_submitted",
    ):
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
    for tier in hard.GEN_TIERS:
        if tier in state["gen_submitted"]:
            continue
        for host in hard.HOSTS:
            tasks = [t for t in hard.gen_tasks(tier) if t.host == host]
            _put(cfg, hard.gen_queue(tier, host), [hard.gen_job(t) for t in tasks])
        state["gen_submitted"][tier] = utc_now_iso()
        mark(f"submit_{tier}")
    note = _repin(cfg)
    if note:
        fired.append(note)
    if not cfg.dry_run:
        _requeue_and_holes(cfg, state)
    low = _low_disk_hosts()
    if low:
        waiting.append(f"disk_low on {sorted(low)}: new submissions held")
    _resolve_teachers(cfg, state, low)
    jobs = _Jobs(cfg)
    ce_base = (
        json.loads(hard.CE_BASE_PATH.read_text()) if hard.CE_BASE_PATH.is_file() else {}
    )
    if ce_base:
        fired += _step_downs(cfg, state, jobs, ce_base, low)
    open_tiers = list(P1_TIERS)
    if _phase_open(cfg, state, P1_TIERS):
        open_tiers += P2_TIERS
        if _phase_open(cfg, state, P2_TIERS):
            open_tiers += P3_TIERS
    for tier in open_tiers:
        for teacher in hard.teacher_groups(tier):
            if f"{tier}|{teacher}" in state["applied"]:
                continue
            if teacher not in state["animals"]:
                waiting.append(f"{tier}|{teacher}: teacher gates pending")
                continue
            hosts = {
                hard.group_host(hard.FAMILY_OF[tier], b.teacher, b.student, b.arm, s)
                for b in hard.tier_slots(tier)
                if b.teacher == teacher
                for s in b.seeds
            }
            if hosts & low:
                continue
            n = _apply_tier(cfg, state, tier, teacher, jobs)
            mark(f"tier {tier}|{teacher} ({n} cells)")
    downs = set(_down_queues(cfg))
    for name in _submitted(cfg):
        if _is_train(name) and (
            name in downs
            or not store.queue_exists(cfg.queue_root, hard.late_queue(name))
        ):
            _ensure_late(cfg, name)
    for phase in results.PHASES:
        key = f"results_{phase}"
        if key in done:
            continue
        prev = (
            results.PHASES[results.PHASES.index(phase) - 1] if phase != "p1" else None
        )
        if prev is not None and f"results_{prev}" not in done:
            break
        tiers = PHASE_TIERS[phase]
        downs = _down_queues(cfg) if phase != "p1" else []
        if (
            _phase_open(cfg, state, tiers)
            and _groups_done(state, PHASE_GROUP_TIERS[phase])
            and _terminal(cfg, downs)
        ):
            _build_results(cfg, state, phase, jobs)
            _fire(state, key)
            mark(key)
        else:
            waiting.append(f"{key}: {phase} cells or step-downs running")
            break
    all_done = False
    if "results_p3" in done:
        names = _submitted(cfg)
        finished = True
        for name in names:
            ok, msg = driver_io.queue_finished(cfg, name)
            if not ok:
                waiting.append(msg)
                finished = False
        if finished and not state.get("holes"):
            if "finalize" not in done:
                _build_results(cfg, state, "p3", jobs)
                if not cfg.dry_run:
                    atomic_write_text(
                        cfg.logs_dir / hard.DONE_FILE, utc_now_iso() + "\n"
                    )
                _fire(state, "finalize")
                mark("finalize")
            all_done = True
    if state.get("holes"):
        waiting.append(f"holes in {sorted(state['holes'])}")
    state["last_pass"] = {"at": utc_now_iso(), "fired": fired, "waiting": waiting[:50]}
    driver_io.save_state(cfg, state)
    return AdvanceResult(tuple(fired), tuple(waiting), all_done)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Transitions, gates, groups, holes and per-queue counts (pure file reads)."""
    state = driver_io.load_state(cfg)
    queues = {}
    for name in _submitted(cfg):
        s = store.queue_status(cfg.queue_root, name)
        queues[name] = {k: v for k, v in s.items() if k != "running_jobs"}
    groups = state.get("groups", {}).values()
    return {
        "transitions": sorted(state.get("transitions", {})),
        "teachers_resolved": state.get("animals", {}),
        "labelled": state.get("labelled", {}),
        "groups_open": sum(not g["done"] for g in groups),
        "step_downs": {
            k: g["rates"][len(g["initial"]) :]
            for k, g in state.get("groups", {}).items()
            if len(g["rates"]) > len(g["initial"])
        },
        "holes": state.get("holes", {}),
        "queues": queues,
    }


def planned_summary() -> dict:
    """Planned queues and cells per tier and host, before gates and step-downs."""
    out: dict = {"gen": {}, "tiers": {}}
    for tier in hard.GEN_TIERS:
        for host in hard.HOSTS:
            tasks = [t for t in hard.gen_tasks(tier) if t.host == host]
            out["gen"][hard.gen_queue(tier, host)] = len(tasks)
    for tier in hard.TRAIN_TIERS:
        cells = hard.tier_cells(tier)
        per_host = {h: sum(c.host == h for c in cells) for h in hard.HOSTS}
        queues = {c.host: set() for c in cells}
        for c in cells:
            queues[c.host].add(hard.train_queue(tier, c.teacher, c.host))
        out["tiers"][tier] = {
            "cells": len(cells),
            "per_host": per_host,
            "queues": sum(len(q) for q in queues.values()),
            "priority": plan.PRIORITY[tier],
        }
    return out
