"""sdfpos stage-2 driver (Part I `sdfpos`, Part II `sdfpos2`): one run set per invocation.

Transitions (idempotent, re-run every ``watch`` tick): preflight → submit (all five
queues at once, chained by jobq ``depends_on``: teacher-evals and students wait for
teachers, evals wait for students + controls) → finalize when every queue is complete
with no failed job (failed jobs are requeued under the attempt cap first), or
finalize_partial at the deadline. Coverage counts landed teachers, students, controls
and eval summaries against the plan.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import yaml
from loguru import logger

from subliminal_learning.campaign import driver_io, expected
from subliminal_learning.campaign import crossval2_plan, sdfpos2_plan, sdfpos_plan
from subliminal_learning.jobqueue import store
from subliminal_learning.models.registry import PythiaModelSpec, resolve_hf_repo_id
from subliminal_learning.training.run_set import is_valid_run_set_id
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import cell_is_done, utc_now_iso

PLANS: dict[str, ModuleType] = {
    "sdfpos": sdfpos_plan,
    "sdfpos2": sdfpos2_plan,
    "crossval2": crossval2_plan,
}
# Optional per-campaign hooks: preflight_problems(e), submit_waiting(e, queue_root),
# on_pass(logs_dir_of).
HOOKS: dict[str, str] = {"crossval2": "pipelines.crossval2_gates"}
Experiment = (
    sdfpos_plan.Experiment | sdfpos2_plan.Experiment | crossval2_plan.Experiment
)
PREFLIGHT_REPORT = "preflight_report.json"
COVERAGE_JSON = "coverage.json"
CAMPAIGN_DONE_FILE = "CAMPAIGN_DONE"


class CampaignGateError(RuntimeError):
    """A phase gate failed hard; do not advance."""


@dataclass(frozen=True)
class AdvanceResult:
    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


@dataclass(frozen=True)
class SdfposDriverOptions:
    """Roots, clock and budget of one experiment driver.

    Attributes:
        experiment: Experiment token.
        campaign: Plan key in `PLANS`.
        paths_config: Campaign paths YAML.
        prereg_path: Preregistration markdown that must exist at preflight.
        deadline_hours: Open-window budget.
        max_failed_frac: Failed/total above this aborts; at or below is a hole.
        now: Clock override (tests).
    """

    experiment: str
    campaign: str = "sdfpos"
    paths_config: Path = Path(sdfpos_plan.PATHS_CONFIG)
    prereg_path: Path = Path(sdfpos_plan.PREREG_PATH)
    deadline_hours: float = sdfpos_plan.DEADLINE_HOURS
    max_failed_frac: float = sdfpos_plan.MAX_FAILED_FRAC
    now: Callable[[], float] | None = None


def plan_module(campaign: str) -> ModuleType:
    """The plan module of `campaign`.

    Raises:
        ValueError: Unknown campaign.
    """
    if campaign not in PLANS:
        raise ValueError(f"unknown campaign {campaign!r}; choose one of {list(PLANS)}")
    return PLANS[campaign]


def hooks_of(campaign: str) -> ModuleType | None:
    """The campaign's hook module, or None."""
    return importlib.import_module(HOOKS[campaign]) if campaign in HOOKS else None


def driver_config(
    experiment: str, campaign: str = "sdfpos", **kwargs
) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to one experiment's `CampaignSpec`."""
    spec = plan_module(campaign).spec_of(experiment)
    return driver_io.CampaignDriverConfig(spec=spec, **kwargs)


def _now(opts: SdfposDriverOptions) -> float:
    return opts.now() if opts.now else time.time()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _record(state: dict, name: str, **body: object) -> None:
    state["transitions"][name] = {"at": utc_now_iso(), **body}


def _experiment(
    cfg: driver_io.CampaignDriverConfig, opts: SdfposDriverOptions
) -> Experiment:
    e = plan_module(opts.campaign).experiment(opts.experiment)
    if e.run_set_id != cfg.run_set_id:
        raise CampaignGateError(f"config run set {cfg.run_set_id} != {e.run_set_id}")
    return e


# ---- windows --------------------------------------------------------------------


def _open_window(state: dict, opts: SdfposDriverOptions) -> None:
    windows = state.setdefault("deadline_windows", [])
    if windows and windows[-1].get("ended_at") is None:
        windows[-1]["hours"] = float(opts.deadline_hours)
        return
    windows.append(
        {
            "start": _iso(_now(opts)),
            "hours": float(opts.deadline_hours),
            "ended_at": None,
            "reason": None,
        }
    )


def _deadline_passed(state: dict, opts: SdfposDriverOptions) -> bool:
    window = (state.get("deadline_windows") or [None])[-1]
    if not window or window.get("ended_at"):
        return False
    start = _parse_utc(window["start"]).timestamp()
    return _now(opts) >= start + float(window["hours"]) * 3600


def _close_window(state: dict, *, reason: str, opts: SdfposDriverOptions) -> None:
    window = state["deadline_windows"][-1]
    window["ended_at"] = _iso(_now(opts))
    window["reason"] = reason


# ---- preflight --------------------------------------------------------------------


def _hub_ready(hub_cache: Path, spec: PythiaModelSpec) -> bool:
    repo = resolve_hf_repo_id(spec, sdfpos_plan.HUB_ORG)
    model_dir = hub_cache / "hub" / f"models--{repo.replace('/', '--')}"
    return (model_dir / "refs" / (spec.revision or "main")).is_file()


def _corpora_check(e: Experiment, rule_version: str) -> dict[str, int]:
    """Doc count per arm from the corpus manifests; raises on a missing or stale corpus.

    An experiment with a per-arm ``rule_version(arm)`` overrides `rule_version`.
    """
    counts: dict[str, int] = {}
    for arm in e.arms:
        if hasattr(e, "rule_version"):
            rule_version = e.rule_version(arm)
        d = e.corpus_dir(arm)
        manifest = d / "manifest.json"
        if not (d / "docs.jsonl").is_file() or not manifest.is_file():
            raise CampaignGateError(f"corpus missing: {d}")
        doc = yaml.safe_load(manifest.read_text())
        if doc.get("rule_version") != rule_version:
            raise CampaignGateError(
                f"{manifest}: rule_version {doc.get('rule_version')} != {rule_version}"
            )
        counts[arm.corpus_tag] = int(doc["n_docs"])
    return counts


def _preflight(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: SdfposDriverOptions
) -> None:
    plan = plan_module(opts.campaign)
    e = _experiment(cfg, opts)
    if not Path(opts.prereg_path).is_file():
        raise CampaignGateError(f"prereg missing: {opts.prereg_path}")
    paths = yaml.safe_load(Path(opts.paths_config).read_text())
    for spec in {e.base_spec, getattr(e, "student_spec", e.base_spec)}:
        if not _hub_ready(Path(paths["hf_cache_root"]), spec):
            raise CampaignGateError(f"{spec} not in {paths['hf_cache_root']}")
    corpora = _corpora_check(e, getattr(plan, "CORPUS_RULE_VERSION", None))
    hooks = hooks_of(opts.campaign)
    if hooks is not None and (problems := hooks.preflight_problems(e)):
        raise CampaignGateError("; ".join(problems))
    policy = cfg.queue_root / f"gpu_policy.{store.this_host()}.json"
    if not policy.is_file():
        raise CampaignGateError(f"policy file missing: {policy}")
    if store.this_host() != e.node:
        raise CampaignGateError(
            f"{e.name} launches only on {e.node}; this is {store.this_host()}"
        )
    for name in plan.queue_priorities(e):
        if store.queue_exists(cfg.queue_root, name):
            meta = store.read_meta(cfg.queue_root, name)
            if meta.node != store.this_host():
                raise CampaignGateError(f"queue {name} is pinned to {meta.node}")
    train = plan.all_train_configs(e)
    verify = plan.verify_configs(e, train)
    driver_io.write_generated_configs(cfg, {**train, **verify})
    state["code_sha"] = code_pin_io.resolve_commit_sha(
        cfg.repo_root, cfg.code_sha or "HEAD"
    )
    driver_io.write_logs_report(
        cfg,
        PREFLIGHT_REPORT,
        {
            "at": utc_now_iso(),
            "experiment": e.name,
            "node": store.this_host(),
            "corpora": corpora,
            "n_train_yamls": len(train),
            "n_verify_yamls": len(verify),
            "code_sha": state["code_sha"],
            "n_teachers": len(plan.expected_teacher_dirs(e, cfg.output_root)),
            "n_students": len(plan.expected_student_dirs(e, cfg.output_root)),
            "n_controls": len(plan.expected_control_dirs(e, cfg.output_root)),
        },
    )
    _record(
        state,
        "preflight",
        train_yamls=len(train),
        verify_yamls=len(verify),
        code_sha=state["code_sha"],
    )


def _submit(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: SdfposDriverOptions
) -> None:
    e = _experiment(cfg, opts)
    by_q = plan_module(opts.campaign).all_jobs(e)
    submitted: dict[str, int] = {}
    for stage in e.stages:
        name = e.queue(stage)
        jobs = by_q[name]
        if not jobs:
            raise CampaignGateError(f"queue {name} built zero jobs")
        deps = (
            list(e.depend_queues(stage))
            if hasattr(e, "depend_queues")
            else [e.queue(s) for s in e.depends(stage)]
        )
        driver_io.submit_queue(
            cfg,
            name,
            jobs,
            depends_on=deps or None,
            strict_deps=stage in getattr(e, "strict_dep_stages", ()),
        )
        submitted[name] = len(jobs)
    _record(state, "submit", queues=submitted)


# ---- readiness / coverage ----------------------------------------------------------


def _queues(e: Experiment) -> list[str]:
    return [e.queue(s) for s in e.stages]


def _queue_ready(
    cfg: driver_io.CampaignDriverConfig,
    name: str,
    opts: SdfposDriverOptions,
    holes: list[dict],
) -> bool:
    if not store.queue_exists(cfg.queue_root, name):
        return False
    try:
        driver_io.requeue_failed_under_cap(cfg, name)
    except RuntimeError as err:
        _record_hole(cfg, name, opts, holes, err)
    if not store.queue_complete(cfg.queue_root, name):
        return False
    if store.queue_status(cfg.queue_root, name)["failed"]:
        _record_hole(cfg, name, opts, holes, None)
    else:
        holes[:] = [h for h in holes if h["queue"] != name]
    return True


def _record_hole(
    cfg: driver_io.CampaignDriverConfig,
    name: str,
    opts: SdfposDriverOptions,
    holes: list[dict],
    err: Exception | None,
) -> None:
    status = store.queue_status(cfg.queue_root, name)
    frac = status["failed"] / max(1, status["total"])
    if frac > opts.max_failed_frac:
        raise CampaignGateError(
            f"{name}: {status['failed']}/{status['total']} failed ({frac:.1%}) > "
            f"{opts.max_failed_frac:.0%}: {err or 'over the attempt cap'}"
        )
    hole = {
        "queue": name,
        "failed": status["failed"],
        "total": status["total"],
        "at": utc_now_iso(),
    }
    holes[:] = [h for h in holes if h["queue"] != name] + [hole]


def _coverage(
    cfg: driver_io.CampaignDriverConfig,
    opts: SdfposDriverOptions,
    state: dict,
    *,
    partial: bool,
    holes: list[dict],
) -> dict:
    plan = plan_module(opts.campaign)
    e = _experiment(cfg, opts)
    teachers = plan.expected_teacher_dirs(e, cfg.output_root)
    students = plan.expected_student_dirs(e, cfg.output_root)
    controls = plan.expected_control_dirs(e, cfg.output_root)
    by_dir = plan.expected_config_gseeds(e, cfg.output_root)
    incomplete = set(
        expected.verify_summaries_complete(
            sorted(by_dir), required_prompt_sets=plan.EVAL_PROMPT_SETS
        )
    )
    teacher_param_dirs = plan.expected_teacher_param_dirs(e, cfg.output_root)
    missing_teacher_evals = [
        str(d)
        for d in teacher_param_dirs
        if not (d / "preference_summary.json").is_file()
    ]
    panels = (
        plan.expected_panel_files(e, cfg.output_root)
        if hasattr(plan, "expected_panel_files")
        else []
    )
    by_arm = (
        plan.coverage_by_arm(e, cfg.output_root)
        if hasattr(plan, "coverage_by_arm")
        else {
            arm.slug: {
                "teachers": sum(
                    cell_is_done(plan.teacher_run_dir(e, arm, t, cfg.output_root))
                    for t in e.ftseeds
                ),
                "students": sum(
                    cell_is_done(plan.student_run_dir(e, arm, t, g, cfg.output_root))
                    for t in e.ftseeds
                    for g in e.gseeds
                )
                if arm.students
                else None,
            }
            for arm in e.arms
        }
    )
    return {
        "at": utc_now_iso(),
        "experiment": e.name,
        "run_set_id": e.run_set_id,
        "node": e.node,
        "partial": partial,
        "expected": {
            "teachers": len(teachers),
            "students": len(students),
            "controls": len(controls),
        },
        "landed": {
            "teachers": sum(cell_is_done(d) for d in teachers),
            "students": sum(cell_is_done(d) for d in students),
            "controls": sum(cell_is_done(d) for d in controls),
        },
        "landed_by_arm": by_arm,
        "missing_cells": [
            str(d) for d in teachers + students + controls if not cell_is_done(d)
        ],
        "missing_evals": [str(d) for d in sorted(by_dir) if d in incomplete],
        "missing_teacher_evals": missing_teacher_evals,
        "panels": {
            "expected": len(panels),
            "missing": [str(p) for p in panels if not p.is_file()],
        },
        "holes": holes,
        "deadline_windows": state.get("deadline_windows", []),
        "transitions": state.get("transitions", {}),
        "queues": {
            q: {
                k: v
                for k, v in store.queue_status(cfg.queue_root, q).items()
                if k != "running_jobs"
            }
            for q in _queues(e)
            if store.queue_exists(cfg.queue_root, q)
        },
    }


def _finalize(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: SdfposDriverOptions,
    *,
    partial: bool,
    holes: list[dict],
) -> dict:
    coverage = _coverage(cfg, opts, state, partial=partial, holes=holes)
    if not cfg.dry_run:
        cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(cfg.logs_dir / COVERAGE_JSON, coverage)
        if not partial:
            atomic_write_text(
                cfg.logs_dir / CAMPAIGN_DONE_FILE,
                f"{cfg.run_set_id} finished {utc_now_iso()}\n",
            )
    _record(
        state,
        "finalize_partial" if partial else "finalize",
        partial=partial,
        n_holes=len(holes),
        n_missing_cells=len(coverage["missing_cells"]),
        n_missing_evals=len(coverage["missing_evals"])
        + len(coverage["missing_teacher_evals"]),
    )
    return coverage


# ---- run / status -----------------------------------------------------------------


def run(
    cfg: driver_io.CampaignDriverConfig, opts: SdfposDriverOptions
) -> AdvanceResult:
    """Fire every transition whose gate is open, once."""
    if not is_valid_run_set_id(cfg.run_set_id):
        raise CampaignGateError(f"invalid run_set_id {cfg.run_set_id!r}")
    e = _experiment(cfg, opts)
    state = driver_io.load_state(cfg)
    if cfg.code_sha is None and state.get("code_sha"):
        cfg = replace(cfg, code_sha=str(state["code_sha"]))
    pin_note = driver_io.ensure_campaign_code_pin(cfg)
    fired: list[str] = []
    waiting: list[str] = []
    holes: list[dict] = list(state.get("holes") or [])
    _open_window(state, opts)
    if pin_note:
        fired.append(pin_note)
    tr = state["transitions"]
    if "preflight" not in tr:
        _preflight(cfg, state, opts)
        cfg = replace(cfg, code_sha=str(state["code_sha"]))
        driver_io.save_state(cfg, state)
        fired.append("preflight")
    hooks = hooks_of(opts.campaign)
    gate = (
        hooks.submit_waiting(e, cfg.queue_root)
        if hooks is not None and "submit" not in tr
        else None
    )
    if gate:
        waiting.append(gate)
    elif "submit" not in tr:
        _submit(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("submit")
    if gate:
        pass
    elif "finalize" not in tr and "finalize_partial" not in tr:
        ready = True
        for name in _queues(e):
            if not _queue_ready(cfg, name, opts, holes):
                waiting.append(name)
                ready = False
        if ready:
            cov = _finalize(cfg, state, opts, partial=bool(holes), holes=holes)
            _close_window(state, reason="done" if not holes else "holes", opts=opts)
            fired.append("finalize" if not holes else "finalize_partial")
            logger.info(
                "coverage: landed {} of expected {}", cov["landed"], cov["expected"]
            )
        elif _deadline_passed(state, opts):
            _finalize(cfg, state, opts, partial=True, holes=holes)
            _close_window(state, reason="deadline", opts=opts)
            fired.append("finalize_partial")
    state["holes"] = holes
    driver_io.save_state(cfg, state)
    if hooks is not None and not cfg.dry_run:
        hooks.on_pass(
            lambda name: (
                replace(cfg, spec=plan_module(opts.campaign).spec_of(name)).logs_dir
            )
        )
    done = (
        "finalize" in state["transitions"] or "finalize_partial" in state["transitions"]
    )
    return AdvanceResult(tuple(fired), tuple(waiting), done)


def queue_plan(experiment: str, campaign: str = "sdfpos") -> dict:
    """Queues of an experiment with job counts (pure enumeration)."""
    plan = plan_module(campaign)
    e = plan.experiment(experiment)
    by_q = plan.all_jobs(e)
    return {
        "run_set_id": e.run_set_id,
        "node": e.node,
        "n_jobs": sum(len(js) for js in by_q.values()),
        "queues": {q: len(js) for q, js in by_q.items()},
        "depends": {e.queue(s): [e.queue(d) for d in e.depends(s)] for s in e.stages},
    }


def status(cfg: driver_io.CampaignDriverConfig, opts: SdfposDriverOptions) -> dict:
    """Transitions, queue counts, holes and windows of one experiment."""
    e = _experiment(cfg, opts)
    state = driver_io.load_state(cfg)
    return {
        "run_set_id": cfg.run_set_id,
        "transitions": sorted(state.get("transitions", {})),
        "queues": {
            q: {
                k: v
                for k, v in store.queue_status(cfg.queue_root, q).items()
                if k != "running_jobs"
            }
            for q in _queues(e)
            if store.queue_exists(cfg.queue_root, q)
        },
        "holes": list(state.get("holes") or []),
        "deadline_windows": list(state.get("deadline_windows") or []),
        "code_sha": state.get("code_sha"),
        "done": (cfg.logs_dir / CAMPAIGN_DONE_FILE).is_file(),
    }
