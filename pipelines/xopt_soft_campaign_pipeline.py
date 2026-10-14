"""Auto-advancing driver of the `xopt` soft run set, on soar.

One `run(cfg)` = one idempotent advance pass:

    preflight      prereg, paths YAML, staging report (with CE_base), queue root, disk
    register       the campaign pin can run every job
    submit_pN@host per host, priority level N = 1, 2, 3 (P1 = tiers b, c, g; P2 = g2, gl,
                   cs3, cl, g48, self; P3 = a): P1 at once, P(N+1) once every level-N train
                   queue of the host is terminal; cs3 also gives its third corner leaf to
                   every tier-c step-down rate already scheduled; a disk check gates each
    (every pass)   unfinished queues follow the pin; failures under the requeue cap are
                   requeued, the rest recorded as labelled holes; every (cell, arm) whose
                   leaves are all finished is re-checked under the step-down rule
                   (docs/PREREG_XOPT.md §3): a new rung clones the leaves at the lowest
                   rate into the host's down{depth} queue; an exhausted arm is a labelled
                   `degenerate` hole
    results_pN     tables + claims once level N is submitted everywhere and every train
                   queue of levels ≤ N is terminal
    finalize       final tables + CAMPAIGN_DONE once P3 is submitted everywhere and every
                   queue (step-down rungs included) is terminal
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import replace
from pathlib import Path

import yaml
from loguru import logger

from pipelines import optparam2_campaign_pipeline as op2
from pipelines import xopt_soft_results_pipeline as results
from pipelines import xopt_soft_staging_pipeline as staging
from pipelines.optparam2_campaign_pipeline import (
    AdvanceResult,
    CampaignGateError,
    _fire,
    current_pin,
    node_pin_defaults,
)
from pipelines.traitfree_staging_pipeline import write_once_text
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import optparam2_plan as oplan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import xopt_plan as plan
from subliminal_learning.campaign import xopt_soft as soft
from subliminal_learning.campaign import xopt_soft_metrics as metrics
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import read_run_set_card, utc_now_iso

SPEC = replace(
    op2.SPEC,
    run_set_id=soft.RUN_SET_ID,
    teacher_run_set_id=soft.RUN_SET_ID,
    root_subdir=oplan.RUN_SET_SUBDIR,
    queue_priorities=soft.QUEUE_PRIORITIES,
    gpu_policy={"cap_per_gpu": plan.SOFT_CAP_PER_GPU},
    generated_config_dir=str(soft.GENERATED_DIR),
    campaign_state_file=soft.CAMPAIGN_STATE_FILE,
    campaign_done_file=soft.CAMPAIGN_DONE_FILE,
    extra_pin_paths=(
        "scripts/sweeps/sweep_distill_student_gkd.py",
        "scripts/sweeps/sweep_evaluate_gkd_student.py",
        "scripts/sweeps/sweep_xopt_soft.py",
        "scripts/run_xopt_soft_campaign.py",
        "scripts/stage_xopt_soft_assets.py",
        "scripts/launch_xopt_soft_campaign.sh",
        soft.TRAINING_CONFIG,
        soft.EXPERIMENT_CONFIG,
        "src/subliminal_learning/campaign/xopt_plan.py",
        "src/subliminal_learning/campaign/xopt_rules.py",
        "src/subliminal_learning/campaign/xopt_soft.py",
        "src/subliminal_learning/campaign/xopt_soft_metrics.py",
        "src/subliminal_learning/campaign/optparam2_jobs.py",
        "src/subliminal_learning/evaluation/pile_probe_io.py",
        "pipelines/xopt_soft_campaign_pipeline.py",
        "pipelines/xopt_soft_results_pipeline.py",
        "pipelines/xopt_soft_staging_pipeline.py",
    ),
)
LEVELS: tuple[int, ...] = (1, 2, 3)


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the xopt soft spec."""
    kwargs.setdefault("generated_dir_override", soft.GENERATED_DIR)
    return driver_io.CampaignDriverConfig(spec=SPEC, **kwargs)


# ---- queues and leaves --------------------------------------------------------------------


def _submitted(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [n for n in soft.QUEUE_PRIORITIES if store.queue_exists(cfg.queue_root, n)]


def _is_late(name: str) -> bool:
    return soft.late_dep(name) is not None


def registered_leaves(cfg: driver_io.CampaignDriverConfig) -> dict[str, soft.SoftLeaf]:
    """Key -> leaf of every leaf in a submitted train or step-down queue."""
    out = {}
    for name in _submitted(cfg):
        if not _is_late(name):
            for job in store.load_jobs(cfg.queue_root, name):
                out[job.key] = soft.leaf_from_key(job.key)
    return out


def leaf_status(cfg: driver_io.CampaignDriverConfig) -> dict[str, str]:
    """Key -> ``done`` | ``failed`` (after the requeue pass: for good) of finished leaves."""
    out = {}
    for name in _submitted(cfg):
        if not _is_late(name):
            for res in store.load_results(cfg.queue_root, name):
                out[res["key"]] = "done" if int(res.get("rc", 1)) == 0 else "failed"
    return out


def write_leaf_yamls(
    cfg: driver_io.CampaignDriverConfig, leaves: Iterable[soft.SoftLeaf]
) -> int:
    """Create (or match) every generated YAML the leaves' jobs read."""
    n = 0
    for leaf in leaves:
        for path, doc in soft.leaf_yaml_docs(leaf).items():
            if not cfg.dry_run:
                path.parent.mkdir(parents=True, exist_ok=True)
                write_once_text(path, yaml.safe_dump(doc, sort_keys=True))
            n += 1
    return n


def _submit_or_append(
    cfg: driver_io.CampaignDriverConfig, name: str, jobs: list[Job]
) -> int:
    host = soft.queue_host(name)
    if not store.queue_exists(cfg.queue_root, name):
        dep = soft.late_dep(name)
        driver_io.submit_queue(
            cfg,
            name,
            jobs,
            depends_on=[dep] if dep else None,
            node=host,
            defaults=node_pin_defaults(cfg, host),
        )
        return len(jobs)
    have = {j.jobkey for j in store.load_jobs(cfg.queue_root, name)}
    new = [j for j in jobs if j.jobkey not in have]
    if new and not cfg.dry_run:
        store.append_jobs(cfg.queue_root, name, new)
        logger.info("appended {} jobs to {}", len(new), name)
    return len(new)


def submit_leaves(
    cfg: driver_io.CampaignDriverConfig,
    leaves: Sequence[soft.SoftLeaf],
    down: int = 0,
    known: set[str] | None = None,
) -> dict[str, int]:
    """Submit the leaves not yet in any queue; return queue -> jobs added.

    Args:
        cfg: Driver config.
        leaves: Candidate leaves.
        down: Step-down depth (0 = the tier's own queues).
        known: Keys already in a queue (read from the queues when omitted); the
            submitted keys are added to it.
    """
    known = set(registered_leaves(cfg)) if known is None else known
    fresh = [lf for lf in leaves if lf.key not in known]
    if not fresh:
        return {}
    known.update(lf.key for lf in fresh)
    write_leaf_yamls(cfg, fresh)
    queues = soft.queue_jobs(fresh, down)
    order = sorted(queues, key=lambda n: (_is_late(n), n))
    return {name: _submit_or_append(cfg, name, queues[name]) for name in order}


def corner_stepdown_rates(
    leaves: Iterable[soft.SoftLeaf],
) -> dict[tuple[str, str], list[float]]:
    """(cell key, arm) -> tier-c rates outside the planned corner slots."""
    planned = set(plan.SOFT_SLOTS_C)
    out: dict[tuple[str, str], set[float]] = defaultdict(set)
    for leaf in leaves:
        if leaf.tier == "c" and (leaf.arm, leaf.lr) not in planned:
            out[(leaf.cell.key, leaf.arm)].add(leaf.lr)
    return {k: sorted(v, reverse=True) for k, v in out.items()}


def level_leaves(
    level: int,
    host: str,
    extra_corner_rates: dict[tuple[str, str], list[float]] | None = None,
) -> list[soft.SoftLeaf]:
    """Planned leaves of `level` on `host` (cs3 with the given step-down rates)."""
    out = []
    for tier in soft.TIERS:
        if soft.LEVEL[tier] != level:
            continue
        extra = extra_corner_rates if tier == "cs3" else None
        out += [
            lf for lf in soft.tier_leaves(tier, extra) if soft.leaf_host(lf) == host
        ]
    return out


def level_train_queues(level: int, host: str) -> list[str]:
    return [soft.train_queue(t, host) for t in soft.TIERS if soft.LEVEL[t] == level]


def gate_queues(level: int, host: str) -> list[str]:
    """Train queues of `host` that must be terminal before `level` + 1 is submitted
    (tier b, the e32 branch, gates nothing)."""
    return [
        q for q in level_train_queues(level, host) if q != soft.train_queue("b", host)
    ]


def planned_queue_counts() -> dict[str, dict[str, int]]:
    """Level -> queue -> planned jobs (no step-down rungs or cs3 extras)."""
    out: dict[str, dict[str, int]] = {}
    for level in LEVELS:
        counts: dict[str, int] = {}
        for host in soft.HOSTS:
            for name, jobs in soft.queue_jobs(level_leaves(level, host)).items():
                counts[name] = len(jobs)
        out[f"p{level}"] = dict(sorted(counts.items()))
    return out


# ---- gates --------------------------------------------------------------------------------


def _disk_free_gib() -> float:
    """Free GiB of the tighter of the soft root (cold-1) and queue root (ssd-2) mounts."""
    free = []
    for path in (soft.ROOT, soft.QUEUE_ROOT):
        while not path.exists():
            path = path.parent
        free.append(shutil.disk_usage(path).free / 2**30)
    return min(free)


def _preflight_problems(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    problems = []
    if not (cfg.repo_root / plan.PREREG_PATH).is_file():
        problems.append(f"missing pre-registration {plan.PREREG_PATH}")
    if Path(cfg.queue_root).resolve() != soft.QUEUE_ROOT.resolve():
        problems.append(f"queue root {cfg.queue_root} is not the plan's")
    if (
        not soft.PATHS_CONFIG_PATH.is_file()
        or soft.PATHS_CONFIG_PATH.read_text() != staging.paths_yaml_text()
    ):
        problems.append(f"{soft.PATHS_CONFIG_PATH} missing or differs from the plan")
    report = staging.latest_complete_report()
    if report is None:
        problems.append(
            "no complete staging report; run scripts.stage_xopt_soft_assets"
        )
    elif set(report.get("ce_base", {})) != set(soft.student_tokens()):
        problems.append("staging report CE_base does not cover every student base@step")
    free = _disk_free_gib()
    if free < plan.MIN_DISK_FREE_GIB:
        problems.append(f"soft roots: {free:.0f} GiB free < {plan.MIN_DISK_FREE_GIB}")
    return problems


def _preflight(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    problems = _preflight_problems(cfg)
    if problems and not cfg.dry_run:
        raise CampaignGateError("; ".join(problems))
    for p in problems:
        logger.warning("[dry-run] preflight would refuse: {}", p)
    report = staging.latest_complete_report()
    _fire(state, "preflight", staging_report_at=(report or {}).get("at"))


def _register(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    if cfg.dry_run:
        sha = code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha or "HEAD")
        _fire(state, "register", git_sha=sha)
        return
    pin = current_pin(cfg)
    driver_io.verify_pin_runs_campaign(pin, SPEC)
    _fire(state, "register", git_sha=pin.sha)


def _repin(cfg: driver_io.CampaignDriverConfig) -> str | None:
    """Move every queue with work left (or failures) to the campaign pin."""
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


def _terminal(cfg: driver_io.CampaignDriverConfig, names: Iterable[str]) -> bool:
    """Every existing queue in `names` complete (no queue = nothing to wait for)."""
    return all(
        store.queue_complete(cfg.queue_root, n)
        for n in names
        if store.queue_exists(cfg.queue_root, n)
    )


# ---- submission and step-down --------------------------------------------------------------


def _submit_level(
    cfg: driver_io.CampaignDriverConfig, state: dict, level: int, host: str
) -> bool:
    """Submit `level` on `host` when its gate is open; return whether it fired."""
    free = _disk_free_gib()
    holes = state.setdefault("holes", {})
    if free < plan.MIN_DISK_FREE_GIB:
        holes["disk_low"] = f"soft roots: {free:.0f} GiB free"
        return False
    holes.pop("disk_low", None)
    extra = (
        corner_stepdown_rates(registered_leaves(cfg).values()) if level == 2 else None
    )
    leaves = level_leaves(level, host, extra)
    added = submit_leaves(cfg, leaves)
    _fire(
        state,
        f"submit_p{level}@{host}",
        n_leaves=len(leaves),
        n_jobs=sum(added.values()),
        queues=sorted(added),
        cs3_extra_rates={f"{k[0]}|{k[1]}": v for k, v in (extra or {}).items()},
    )
    return True


def _group_signature(leaves: Iterable[soft.SoftLeaf], status: dict[str, str]) -> str:
    body = json.dumps(sorted((lf.key, status[lf.key]) for lf in leaves))
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def stepdown_pass(
    cfg: driver_io.CampaignDriverConfig, state: dict, ce: dict[str, float]
) -> list[str]:
    """Re-check every finished (cell, arm) under the step-down rule; submit new rungs.

    Returns:
        One note per decision taken this pass.
    """
    status = leaf_status(cfg)
    registered = registered_leaves(cfg)
    known = set(registered)
    groups: dict[tuple, list[soft.SoftLeaf]] = defaultdict(list)
    for leaf in registered.values():
        if leaf.tier != "b":
            groups[(soft.cell_tuple(leaf), leaf.arm)].append(leaf)
    seen = state.setdefault("stepdowns", {})
    degenerate = state.setdefault("degenerate", {})
    initial = soft.initial_rates()
    notes = []
    for (cell, arm), leaves in sorted(groups.items()):
        if any(lf.key not in status for lf in leaves):
            continue
        gkey = f"{'|'.join(map(str, cell))}|{arm}"
        sig = _group_signature(leaves, status)
        if seen.get(gkey, {}).get("signature") == sig:
            continue
        rows = results.leaf_rows(leaves)
        rates = metrics.arm_rates(metrics.group_rates(rows)).get((cell, arm), [])
        base = ce[ws160m_models.token(cell[3], cell[4])]
        planned = initial[(cell, arm)]
        new = metrics.stepdown(rates, planned, base)
        decision = {
            "signature": sig,
            "at": utc_now_iso(),
            "rates": [r.lr for r in rates],
        }
        if new is not None:
            lowest = min(lf.lr for lf in leaves)
            rung = [replace(lf, lr=new) for lf in leaves if lf.lr == lowest]
            depth = plan.SOFT_ARMS[arm].lower_rungs(min(planned)).index(new) + 1
            by_host: dict[str, list[soft.SoftLeaf]] = defaultdict(list)
            for lf in rung:
                by_host[soft.leaf_host(lf)].append(lf)
            for host_leaves in by_host.values():
                submit_leaves(cfg, host_leaves, down=depth, known=known)
            decision.update(next_rate=new, depth=depth, n_leaves=len(rung))
            notes.append(f"stepdown({gkey} -> {new:g}, depth {depth})")
        elif metrics.selected(rates, base) is None:
            degenerate[gkey] = {"at": decision["at"], "rates": decision["rates"]}
            notes.append(f"degenerate({gkey})")
        else:
            degenerate.pop(gkey, None)
        seen[gkey] = decision
    return notes


def _write_results(
    cfg: driver_io.CampaignDriverConfig, ce: dict[str, float], suffix: str
) -> None:
    leaves = sorted(registered_leaves(cfg).values(), key=lambda lf: lf.key)
    paths = results.write_tables(cfg.run_set_root / "results", leaves, ce, suffix)
    logger.info("results -> {}", paths)


# ---- the pass -----------------------------------------------------------------------------


def run(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    """One advance pass."""
    if not cfg.dry_run:
        driver_io.ensure_campaign_code_pin(cfg)
    state = driver_io.load_state(cfg)
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
    report = staging.latest_complete_report()
    ce = {k: float(v) for k, v in (report or {}).get("ce_base", {}).items()}
    note = _repin(cfg)
    if note:
        fired.append(note)
    if not cfg.dry_run:
        _requeue_and_holes(cfg, state)
    for host in soft.HOSTS:
        for level in LEVELS:
            name = f"submit_p{level}@{host}"
            if name in done:
                continue
            prev = gate_queues(level - 1, host) if level > 1 else []
            if level > 1 and f"submit_p{level - 1}@{host}" not in done:
                break
            if not _terminal(cfg, prev):
                waiting.append(f"{name}: level-{level - 1} train queues not terminal")
                break
            if not _submit_level(cfg, state, level, host):
                waiting.append(f"{name}: disk below {plan.MIN_DISK_FREE_GIB} GiB")
                break
            mark(name)
    if cfg.dry_run:
        waiting.append("dry-run: step-down, results and finalize not evaluated")
        return AdvanceResult(tuple(fired), tuple(waiting), False)
    if ce:
        notes = stepdown_pass(cfg, state, ce)
        if notes:
            fired.extend(notes)
            driver_io.save_state(cfg, state)
    for level in LEVELS:
        name = f"results_p{level}"
        submitted = all(f"submit_p{level}@{h}" in done for h in soft.HOSTS)
        queues = [
            q
            for h in soft.HOSTS
            for lv in LEVELS[:level]
            for q in level_train_queues(lv, h)
        ]
        if name not in done and submitted and _terminal(cfg, queues):
            _write_results(cfg, ce, f"_p{level}")
            _fire(state, name)
            mark(name)
    all_done = False
    p3_everywhere = all(f"submit_p3@{h}" in done for h in soft.HOSTS)
    if p3_everywhere and _terminal(cfg, _submitted(cfg)):
        if "finalize" not in done:
            _write_results(cfg, ce, "")
            atomic_write_text(
                cfg.logs_dir / soft.CAMPAIGN_DONE_FILE, utc_now_iso() + "\n"
            )
            _fire(
                state,
                "finalize",
                holes=sorted(state.get("holes", {})),
                degenerate=sorted(state.get("degenerate", {})),
            )
            mark("finalize")
        all_done = True
    elif p3_everywhere:
        waiting.append("finalize: queues not terminal")
    if state.get("holes"):
        waiting.append(f"labelled holes in {sorted(state['holes'])}")
    state["last_pass"] = {"at": utc_now_iso(), "fired": fired, "waiting": waiting[:50]}
    driver_io.save_state(cfg, state)
    return AdvanceResult(tuple(fired), tuple(waiting), all_done)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Transitions, holes, degenerate arms and per-queue counts (pure file reads)."""
    state = driver_io.load_state(cfg)
    queues = {}
    for name in _submitted(cfg):
        s = store.queue_status(cfg.queue_root, name)
        queues[name] = {k: v for k, v in s.items() if k != "running_jobs"}
    return {
        "transitions": sorted(state.get("transitions", {})),
        "holes": state.get("holes", {}),
        "degenerate": sorted(state.get("degenerate", {})),
        "n_stepdowns": sum(
            1 for d in state.get("stepdowns", {}).values() if "next_rate" in d
        ),
        "queues": queues,
    }


def write_policy(host: str, *, replace_existing: bool = False) -> Path:
    """Write `host`'s jobq policy file into the soft queue root (create or equal)."""
    path = soft.QUEUE_ROOT / f"gpu_policy.{host}.json"
    doc = soft.gpu_policy_json(host)
    if path.is_file() and not replace_existing:
        if json.loads(path.read_text()) != doc:
            raise FileExistsError(f"{path} exists with a different policy")
        return path
    atomic_write_json(path, doc)
    return path
