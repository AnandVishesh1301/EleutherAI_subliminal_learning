"""Auto-advancing driver of the `xopt2` hard channel, on soar (workers: `WORKER_HOSTS`).

One `run(cfg)` = one idempotent advance pass:

    preflight   prereg, paths YAML, staging report, every reused `xopt` input ready
                and unchanged since staging, the soft channel's card (if any) on
                the same pin
    register    run-set card pinned to the campaign code pin
    (every pass) queues follow the pin; failures under the cap are requeued, the rest
                recorded as queue holes; new train submissions are held while ssd-2
                or cold-1 is under its free-space floor (`labelled["disk_low"]`)
    submit      static tiers v, w, wk, o, g at their initial rates (one leaf per ident)
    decide      per open group whose scheduled decision leaves are finished: the
                walk's next rung (`walk` queue) or a step-down (`down` queue), or done
    extra       per finished walk: `HARD_EXTRA_SEED` at its decision rates (tier x)
    carry       per new arm whose py@143k walk is done: its anchor, then fixed-rate
                groups on the 18 other 143k pairs (ox) and py@115k self (ok)
    late        lm-eval + sampled sets of every finished student (one late queue)
    results_pN  tables + claims at each phase boundary (p1: v w wk o; p2: x ox ok;
                p3: g and every late eval), once the phase's queues are finished
                (terminal, no failed job); suffix recorded in `state["results"]`
    finalize    the channel's CAMPAIGN_DONE once every queue is finished and there is
                no queue or decision hole; the campaign marker once the soft
                channel's marker exists too (both create-only)

A failed decision leaf is never read as an outcome: its rate stays undecided (labelled
hole `cells_failed`) until the job is repaired. Queues are unpinned: every worker
host drains them, each from its own clone of the pin (`xopt2_plan.queue_defaults`).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

from loguru import logger

from pipelines import xopt2_hard_results_pipeline as results
from pipelines import xopt2_hard_staging_pipeline as staging
from pipelines.hlopt_campaign_pipeline import (
    AdvanceResult,
    CampaignGateError,
    _fire,
    current_pin,
    node_pin_defaults,
)
from subliminal_learning.campaign import driver_io, xfam2_plan
from subliminal_learning.campaign import xopt2_hard as hard
from subliminal_learning.campaign import xopt2_hard_metrics as metrics
from subliminal_learning.campaign import xopt2_plan as plan
from subliminal_learning.campaign import xopt_hard as xh
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.training.run_set import build_run_set_card
from subliminal_learning.utils.io import atomic_write_json
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
    gpu_policy={"cap_per_gpu": {h: plan.HARD_CAP_PER_GPU for h in hard.HOSTS}},
    generated_config_dir=str(hard.GENERATED_DIR),
    campaign_state_file=hard.STATE_FILE,
    campaign_done_file=hard.DONE_FILE,
    extra_pin_paths=(
        "scripts/sweeps/sweep_xopt2_hard.py",
        "scripts/sweeps/sweep_hlopt.py",
        "scripts/run_xopt2_hard_campaign.py",
        "scripts/stage_xopt2_hard_assets.py",
        "scripts/launch_xopt2_hard_campaign.sh",
        "scripts/jobq_pool_loop.sh",
        "src/subliminal_learning/campaign/xopt2_plan.py",
        "src/subliminal_learning/campaign/xopt2_rules.py",
        "src/subliminal_learning/campaign/xopt2_hard.py",
        "src/subliminal_learning/campaign/xopt2_hard_metrics.py",
        "src/subliminal_learning/campaign/xopt_hard.py",
        "src/subliminal_learning/campaign/xopt_hard_metrics.py",
        "src/subliminal_learning/evaluation/pile_probe_io.py",
        "src/subliminal_learning/training/optimizers.py",
        "src/subliminal_learning/training/param_control_io.py",
        "src/subliminal_learning/teachers/sft_io.py",
        "pipelines/xopt2_hard_cell_pipeline.py",
        "pipelines/xopt2_hard_campaign_pipeline.py",
        "pipelines/xopt2_hard_results_pipeline.py",
        "pipelines/xopt2_hard_staging_pipeline.py",
        "pipelines/xopt_hard_cell_pipeline.py",
        "pipelines/xopt_hard_results_pipeline.py",
    ),
)
STEP_QUEUE = {metrics.STEP_DOWN: "down", "climb": "walk", "bisect": "walk"}


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the xopt2 hard spec."""
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


def disk_low() -> list[str]:
    """Mounts under their free-space floor (`xopt2_plan.MIN_FREE_GIB`)."""
    out = []
    for mount, root in hard.DISK_ROOTS.items():
        p = Path(root)
        while not p.exists():
            p = p.parent
        if shutil.disk_usage(p).free / 2**30 < plan.MIN_FREE_GIB[mount]:
            out.append(mount)
    return out


# ---- queues and jobs ------------------------------------------------------------------


def _submitted(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [n for n in hard.QUEUE_PRIORITIES if store.queue_exists(cfg.queue_root, n)]


def queue_defaults(cfg: driver_io.CampaignDriverConfig) -> dict:
    """jobq defaults binding a queue to every worker host's worktree of the pin."""
    return plan.queue_defaults({h: node_pin_defaults(cfg, h) for h in hard.HOSTS})


def _put(cfg: driver_io.CampaignDriverConfig, name: str, jobs: list[Job]) -> int:
    """Create `name` with `jobs`, or append the ones it lacks; return how many are new."""
    if not jobs:
        return 0
    if not store.queue_exists(cfg.queue_root, name):
        driver_io.submit_queue(
            cfg, name, jobs, unpinned=True, defaults=queue_defaults(cfg)
        )
        return len(jobs)
    have = {j.jobkey for j in store.load_jobs(cfg.queue_root, name)}
    new = [j for j in jobs if j.jobkey not in have]
    if new and not cfg.dry_run:
        store.append_jobs(cfg.queue_root, name, new)
        logger.info("appended {} job(s) to {}", len(new), name)
    return len(new)


class _Jobs:
    """Every submitted train job: ident -> (queue, job, cell); states cached per pass."""

    def __init__(self, cfg: driver_io.CampaignDriverConfig):
        self.cfg = cfg
        self.by_ident: dict[tuple, tuple[str, Job, hard.Cell]] = {}
        self._states: dict[tuple, str] = {}
        for name in hard.TRAIN_QUEUES:
            if store.queue_exists(cfg.queue_root, name):
                for job in store.load_jobs(cfg.queue_root, name):
                    cell = hard.cell_from_job(job)
                    self.by_ident.setdefault(cell.ident, (name, job, cell))

    def submit(self, queue: str, cells: list[hard.Cell]) -> int:
        """Queue every cell whose ident no queue holds yet; return the new count."""
        jobs = []
        for cell in cells:
            if cell.ident in self.by_ident:
                continue
            job = hard.train_job(cell)
            self.by_ident[cell.ident] = (queue, job, cell)
            jobs.append(job)
        return _put(self.cfg, queue, jobs)

    def state(self, ident: tuple) -> str | None:
        if ident not in self.by_ident:
            return None
        if ident not in self._states:
            name, job, _ = self.by_ident[ident]
            self._states[ident] = store.job_state(self.cfg.queue_root, name, job)
        return self._states[ident]

    def cell(self, ident: tuple) -> hard.Cell:
        return self.by_ident[ident][2]

    def cells(self) -> list[hard.Cell]:
        return [c for _, _, c in self.by_ident.values()]

    def rows(self, group: hard.Group, rates: list[float]) -> list[dict]:
        """Rows of the group's decision leaves at `rates`."""
        return results.cell_rows(
            [self.cell(i) for r in rates for i in group.decision_idents(r)]
        )


# ---- groups ---------------------------------------------------------------------------


def _groups(state: dict) -> dict[str, hard.Group]:
    return {k: hard.Group.from_dict(v) for k, v in state["groups"].items()}


def _save_groups(state: dict, groups: dict[str, hard.Group]) -> None:
    state["groups"] = {k: g.to_dict() for k, g in groups.items()}


def _submit_static(state: dict, jobs: _Jobs) -> dict[str, int]:
    """Create the static groups and queue their initial cells (v first: shared idents)."""
    groups = _groups(state)
    counts = {}
    for tier in hard.STATIC_TIERS:
        cells = []
        for g in hard.tier_groups(tier):
            groups[g.key] = g
            cells += g.cells(g.initial)
        counts[tier] = jobs.submit(hard.train_queue(tier), cells)
    _save_groups(state, groups)
    return counts


def _decide(
    state: dict,
    jobs: _Jobs,
    ce_base: dict[str, float],
    low: list[str],
    waiting: list[str],
) -> list[str]:
    """Advance every open group whose scheduled decision leaves allow it."""
    groups = _groups(state)
    labelled = state["labelled"]
    fired: list[str] = []
    for g in groups.values():
        if g.done:
            continue
        ready = []
        for rate in g.rates:
            states = [jobs.state(i) for i in g.decision_idents(rate)]
            hole = f"hole|{g.key}|lr{rate:g}"
            if store.FAILED in states:
                labelled[hole] = "cells_failed"
            else:
                labelled.pop(hole, None)
            if all(s == store.DONE for s in states):
                ready.append(rate)
        if len(ready) < len(set(g.rates)):
            continue
        index = metrics.index_rows(jobs.rows(g, ready))
        step = metrics.group_step(g, index, ce_base[g.student])
        if step.action == metrics.WAIT:
            waiting.append(f"{g.key}: decision leaves done but unreadable")
            continue
        if step.action == metrics.DONE:
            g.done, g.reason = True, step.reason
            if step.reason == "degenerate":
                bad = (
                    ["all"]
                    if g.is_walk
                    else metrics.ineligible_animals(
                        metrics.group_outcomes(g, index).values(),
                        g.animals,
                        ce_base[g.student],
                    )
                )
                for a in bad:
                    labelled[f"degenerate|{g.key}|{a}"] = "degenerate"
            fired.append(f"done {g.key} ({step.reason})")
            continue
        if low:
            waiting.append(f"{g.key} lr{step.rate:g} held: disk_low")
            continue
        queue = hard.train_queue(STEP_QUEUE[step.reason])
        n = jobs.submit(queue, g.cells((step.rate,)))
        g.rates.append(step.rate)
        fired.append(f"{step.reason} {g.key} -> lr{step.rate:g} ({n} cells)")
        logger.info("{} {} -> {:g} ({} cells)", step.reason, g.key, step.rate, n)
    _save_groups(state, groups)
    return fired


def _extras(
    state: dict,
    jobs: _Jobs,
    ce_base: dict[str, float],
    low: list[str],
    waiting: list[str],
) -> list[str]:
    """Tier-x cells of every finished walk not yet extended."""
    fired = []
    for g in _groups(state).values():
        if not (g.is_walk and g.done) or g.key in state["extra"]:
            continue
        if low:
            waiting.append(f"extra {g.key} held: disk_low")
            continue
        rates = metrics.extra_rates(g, jobs.rows(g, g.rates), ce_base[g.student])
        n = jobs.submit(hard.train_queue(hard.EXTRA_TIER), g.extra_cells(rates))
        state["extra"][g.key] = list(rates)
        fired.append(f"extra {g.key} at {[f'{r:g}' for r in rates]} ({n} cells)")
    return fired


def _carry(
    state: dict,
    jobs: _Jobs,
    ce_base: dict[str, float],
    low: list[str],
    waiting: list[str],
) -> list[str]:
    """Anchor + carry groups of every new arm whose py@143k walk just finished."""
    groups = _groups(state)
    fired = []
    for arm in plan.NEW_ARMS:
        g = groups.get(hard.walk_group_key("o", arm))
        if arm in state["anchors"] or g is None or not g.done:
            continue
        if low:
            waiting.append(f"carry {arm} held: disk_low")
            continue
        anchor = metrics.anchor(g, jobs.rows(g, g.rates), ce_base[g.student])
        state["anchors"][arm] = metrics.anchor_doc(anchor)
        if anchor.rate is None:
            state["labelled"][f"carry|{arm}"] = "no_eligible_rate"
            fired.append(f"carry {arm}: no_eligible_rate")
            continue
        n = 0
        for ng in hard.carry_groups(arm, anchor.carry):
            groups[ng.key] = ng
            n += jobs.submit(hard.train_queue(ng.tier), ng.cells(ng.initial))
        fired.append(
            f"carry {arm} at {list(anchor.carry)} ({anchor.reason}, {n} cells)"
        )
    _save_groups(state, groups)
    return fired


def _late_missing(cfg: driver_io.CampaignDriverConfig, jobs: _Jobs) -> list[Job]:
    """Late jobs of finished students that the late queue lacks."""
    have = (
        {j.jobkey for j in store.load_jobs(cfg.queue_root, hard.LATE_QUEUE)}
        if store.queue_exists(cfg.queue_root, hard.LATE_QUEUE)
        else set()
    )
    out = []
    for ident, (_, _, cell) in jobs.by_ident.items():
        if jobs.state(ident) == store.DONE:
            job = hard.late_job(cell)
            if job.jobkey not in have:
                out.append(job)
    return out


# ---- phases ---------------------------------------------------------------------------


def _finished(cfg: driver_io.CampaignDriverConfig, names: list[str]) -> bool:
    """Every submitted queue of `names` terminal with no failed job."""
    return all(
        driver_io.queue_finished(cfg, n)[0]
        for n in names
        if store.queue_exists(cfg.queue_root, n)
    )


def phase_ready(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    phase: str,
    jobs: _Jobs,
    late_new: int,
) -> bool:
    """Every group of the phase's tiers done and the phase's queues terminal.

    Finished = terminal with no failed job (`driver_io.queue_finished`). p3 also needs
    every train job done in this pass's snapshot and no late job queued by this pass
    (so every finished student's late eval is in the late queue).
    """
    tiers = plan.PHASES[phase]
    if any(not g["done"] for g in state["groups"].values() if g["tier"] in tiers):
        return False
    queues = [hard.train_queue(t) for t in tiers]
    if phase == "p1":
        queues.append(hard.train_queue("walk"))
    if phase == "p2":
        walks = [k for k, g in state["groups"].items() if g["tier"] in hard.WALK_TIERS]
        if set(plan.NEW_ARMS) - set(state["anchors"]) or set(walks) - set(
            state["extra"]
        ):
            return False
    if phase == "p3":
        if late_new or any(jobs.state(i) != store.DONE for i in jobs.by_ident):
            return False
        queues += [hard.train_queue("down"), hard.LATE_QUEUE]
    return _finished(cfg, queues)


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
            "no complete staging report; run scripts.stage_xopt2_hard_assets"
        )
    inputs, readings = staging.check_inputs()
    problems += inputs
    if report is not None:
        problems += staging.pool_drift(report, readings["pools"])
    problem = channel_pin_problem(_own_sha(cfg), plan.SOFT)
    if problem:
        problems.append(problem)
    if problems:
        if cfg.dry_run:
            logger.warning("[dry-run] preflight would fail: {}", problems)
            _fire(state, "preflight", dry_run_problems=problems)
            return
        raise CampaignGateError("; ".join(problems))
    _fire(state, "preflight", staging_report_at=report["at"])


def _own_sha(cfg: driver_io.CampaignDriverConfig) -> str:
    """The code pin this channel registers (or holds)."""
    if cfg.dry_run:
        from subliminal_learning.utils import code_pin_io

        return code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha or "HEAD")
    return current_pin(cfg).sha


def channel_pin_problem(own_sha: str, other: str) -> str | None:
    """Refusal when the other channel's run-set card pins another commit.

    Both channels of `xopt2` run one pin; a channel without a card yet passes.
    """
    card = read_run_set_card(plan.card_dir(other))
    sha = str((card or {}).get("git_sha") or "")
    if sha and not (own_sha.startswith(sha) or sha.startswith(own_sha)):
        return f"{other} channel card pins {sha}; this channel would pin {own_sha}"
    return None


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
            "xopt2 hard channel: learning-rate walks to collapse (full AdamW / "
            "RMSProp), five new optimizers in LoRA and full FT, their carry across "
            "160M model ablations, and the off-cross checkpoint-grid fill; xopt's "
            "template teachers and pools, read only"
        ),
        date=utc_now_iso()[:10],
        owner=plan.OWNER,
        git_sha=pin.sha,
        sweep_yaml="src/subliminal_learning/campaign/xopt2_hard.py",
        resolved_config={
            "control_root": str(hard.ROOT),
            "leaf_root": str(hard.LEAF_ROOT),
            "queue_root": str(hard.QUEUE_ROOT),
            "xopt_inputs": str(plan.XOPT_HARD_ROOT),
            "n_prompts": hard.N_PROMPTS,
        },
        grid_axes={t: len(hard.tier_groups(t)) for t in hard.STATIC_TIERS},
        model_size="160m",
    )
    register_run_set_launch(cfg.output_root, cfg.run_set_root, card)
    _fire(state, "register", git_sha=pin.sha)


def _repin(cfg: driver_io.CampaignDriverConfig) -> str | None:
    """Move every queue with work left (or failures) to the card's pin, unpinned."""
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
        want = want or queue_defaults(cfg)
        meta = store.read_meta(cfg.queue_root, name)
        if meta.node is None and all(
            meta.defaults.get(k) == want[k] for k in ("cwd", "nodes")
        ):
            continue
        store.update_queue_defaults(cfg.queue_root, name, want)
        store.update_queue_node(cfg.queue_root, name, None)
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
    """Queue holes and failed decision leaves (the disk hold is not a hole)."""
    decision = [k for k in state.get("labelled", {}) if k.startswith("hole|")]
    return sorted(state.get("holes") or {}) + decision


def _state_doc(state: dict) -> dict:
    keys = ("groups", "extra", "anchors", "labelled", "holes", "results")
    return {k: state.get(k) for k in keys}


def _ce_base() -> dict[str, float]:
    return json.loads(xh.CE_BASE_PATH.read_text()) if xh.CE_BASE_PATH.is_file() else {}


def _write_once(path: Path) -> bool:
    """Create `path` with a timestamp; False when it already exists."""
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
    if _write_once(cfg.logs_dir / hard.DONE_FILE):
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
    state = driver_io.load_state(cfg)
    for key in ("groups", "extra", "anchors", "labelled"):
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
    note = _repin(cfg)
    if note:
        fired.append(note)
    if not cfg.dry_run:
        _requeue_and_holes(cfg, state)
    low = disk_low()
    if low:
        state["labelled"]["disk_low"] = low
        waiting.append(f"disk_low on {low}: new train submissions held")
    else:
        state["labelled"].pop("disk_low", None)
    jobs = _Jobs(cfg)
    ce_base = _ce_base()
    if "submit" not in done and not low:
        counts = _submit_static(state, jobs)
        _fire(state, "submit", **counts)
        mark(f"submit {counts}")
    if "submit" in done:
        fired += _decide(state, jobs, ce_base, low, waiting)
        fired += _extras(state, jobs, ce_base, low, waiting)
        fired += _carry(state, jobs, ce_base, low, waiting)
    late = _late_missing(cfg, jobs)
    if _put(cfg, hard.LATE_QUEUE, late):
        fired.append(f"late +{len(late)}")
    for phase in results.PHASES:
        key = f"results_{phase}"
        if key in done:
            continue
        if "submit" not in done or not phase_ready(cfg, state, phase, jobs, len(late)):
            waiting.append(f"{key}: {phase} groups or queues running")
            break
        suffix = phase
        if not cfg.dry_run:
            out = results.build(
                cfg.run_set_root / "results",
                jobs.cells(),
                _groups(state),
                ce_base,
                phase,
                _state_doc(state),
            )
            suffix = out["suffix"]
            logger.info("results {} -> {}", phase, out)
        state.setdefault("results", {})[phase] = suffix
        _fire(state, key, suffix=suffix)
        mark(key)
    all_done = False
    if "results_p3" in done:
        finished = True
        for name in _submitted(cfg):
            ok, msg = driver_io.queue_finished(cfg, name)
            if not ok:
                waiting.append(msg)
                finished = False
        if finished and not _holes(state):
            fired += _finalize(cfg)
            if "finalize" not in done:
                _fire(state, "finalize")
                mark("finalize")
            all_done = True
    if _holes(state):
        waiting.append(f"holes: {_holes(state)[:20]}")
    state["last_pass"] = {"at": utc_now_iso(), "fired": fired, "waiting": waiting[:50]}
    driver_io.save_state(cfg, state)
    return AdvanceResult(tuple(fired), tuple(waiting), all_done)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Transitions, groups, anchors, extras, holes and per-queue counts (file reads)."""
    state = driver_io.load_state(cfg)
    queues = {}
    for name in _submitted(cfg):
        s = store.queue_status(cfg.queue_root, name)
        queues[name] = {k: v for k, v in s.items() if k != "running_jobs"}
    groups = state.get("groups", {}).values()
    open_by_tier: dict[str, int] = {}
    for g in groups:
        if not g["done"]:
            open_by_tier[g["tier"]] = open_by_tier.get(g["tier"], 0) + 1
    return {
        "transitions": sorted(state.get("transitions", {})),
        "groups_open": open_by_tier,
        "anchors": state.get("anchors", {}),
        "extra": state.get("extra", {}),
        "labelled": state.get("labelled", {}),
        "holes": state.get("holes", {}),
        "queues": queues,
    }


def planned_summary() -> dict:
    """Static groups, cells and queues per tier (before walks, extras and carry)."""
    out: dict = {}
    seen: set[tuple] = set()
    for tier in hard.STATIC_TIERS:
        groups = hard.tier_groups(tier)
        cells = [c for g in groups for c in g.cells(g.initial)]
        new = [c for c in cells if c.ident not in seen]
        seen |= {c.ident for c in cells}
        out[tier] = {
            "queue": hard.train_queue(tier),
            "priority": plan.PRIORITY[tier],
            "groups": len(groups),
            "cells": len(cells),
            "jobs": len(new),
        }
    out["total_jobs"] = len(seen)
    return out
