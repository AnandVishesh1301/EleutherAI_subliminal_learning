"""Autonomous driver of one ws160m partition (one run set on one node).

Phases: preflight -> controls (+xbase) -> [teachers] -> wait for every teacher the
partition's cells read (shared PVC; fallback-trains missing foreign teachers after
`WAIT_TEACHERS_FALLBACK_S`) -> students + evals (evals depend on their train
queues) -> [wave 2, main only: auto-stage pile-ws3 and submit its cells] ->
force re-evals of config dirs whose summaries miss a gseed -> finalize, or a partial
finalize at the deadline. Every transition is idempotent and recorded in the state
file; resume skips done cells through `cell_is_done`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import yaml
from loguru import logger

from subliminal_learning.campaign import control_io, driver_io
from subliminal_learning.campaign import ws160m_config_gen as cgen
from subliminal_learning.campaign import ws160m_expected as exp
from subliminal_learning.campaign import ws160m_jobs as jobs
from subliminal_learning.campaign import ws160m_models as models
from subliminal_learning.campaign import ws160m_pairs as pairs
from subliminal_learning.campaign import ws160m_plan as plan
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.training.run_set import is_valid_run_set_id
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import cell_is_done, utc_now_iso

DEFAULT_COLD_ROOT = Path("/mnt/cold-1/soar-subliminal/vishesh/subliminal_learning")
_ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


class CampaignGateError(RuntimeError):
    """A phase gate failed hard; do not advance."""


@dataclass(frozen=True)
class AdvanceResult:
    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


@dataclass(frozen=True)
class Ws160mDriverOptions:
    """Roots, clocks and autonomy knobs of one partition driver.

    Attributes:
        partition: Partition token.
        paths_config: Campaign paths YAML (`audit_root` locates the staged models).
        cold_root: Cold-1 artefact root the run-set symlinks target.
        prereg_path: Preregistration markdown that must exist at P0.
        deadline_hours: Open-window budget of this driver.
        drain_timeout_s: Max wait for in-flight cells after a deadline pause.
        max_failed_frac: Failed/total above this aborts; at or below is a hole.
        wait_fallback_s: Seconds of waiting before missing foreign teachers are
            trained locally.
        wait_timeout_s: Seconds of waiting before students are submitted anyway.
        wave2_source_dir: Goncalo's `pile-ws3` run directory (read-only).
        wave2_start_hours: Hours after launch before wave-2 polling starts.
        wave2_giveup_hours: Hours after launch after which wave 2 is skipped.
        auto_wave2: Whether `main` stages and submits wave 2 on its own.
        require_smoke: Whether P0 requires a passed smoke report.
        base_models_root: Override of ``audit_root/base_models_160m`` (tests).
        now: Clock override (tests).
        stage_wave2: Override of the wave-2 staging callable (tests).
    """

    partition: str
    paths_config: Path = Path(plan.PATHS_CONFIG)
    cold_root: Path = DEFAULT_COLD_ROOT
    prereg_path: Path = Path(plan.PREREG_PATH)
    deadline_hours: float = plan.DEADLINE_HOURS_DEFAULT
    drain_timeout_s: int = plan.DRAIN_TIMEOUT_S
    max_failed_frac: float = plan.MAX_FAILED_FRAC
    wait_fallback_s: int = plan.WAIT_TEACHERS_FALLBACK_S
    wait_timeout_s: int = plan.WAIT_TEACHERS_TIMEOUT_S
    wave2_source_dir: Path = Path(plan.WAVE2_SOURCE_DIR)
    wave2_start_hours: float = plan.WAVE2_START_HOURS
    wave2_giveup_hours: float = plan.WAVE2_GIVEUP_HOURS
    auto_wave2: bool = True
    require_smoke: bool = True
    base_models_root: Path | None = None
    now: Callable[[], float] | None = None
    stage_wave2: Callable[..., dict] | None = None


def driver_config(partition: str, **kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to one partition's `CampaignSpec`."""
    return driver_io.CampaignDriverConfig(spec=plan.spec_of(partition), **kwargs)


# ---- helpers ------------------------------------------------------------------------


def _now(opts: Ws160mDriverOptions) -> float:
    return opts.now() if opts.now else time.time()


def _parse_utc(stamp: str) -> datetime:
    return datetime.strptime(stamp, _ISO_FMT).replace(tzinfo=timezone.utc)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(_ISO_FMT)


def _paths(opts: Ws160mDriverOptions) -> dict:
    return yaml.safe_load(Path(opts.paths_config).read_text()) or {}


def _bm_root(opts: Ws160mDriverOptions) -> Path:
    if opts.base_models_root is not None:
        return Path(opts.base_models_root)
    audit_root = _paths(opts).get("audit_root")
    if not audit_root:
        raise CampaignGateError(f"{opts.paths_config}: audit_root is required")
    return Path(str(audit_root)) / models.BASE_MODELS_DIRNAME


def _hub_cache(opts: Ws160mDriverOptions) -> Path:
    hf_root = _paths(opts).get("hf_cache_root")
    if not hf_root:
        raise CampaignGateError(f"{opts.paths_config}: hf_cache_root is required")
    return Path(str(hf_root)) / "hub"


def _partition(
    cfg: driver_io.CampaignDriverConfig, opts: Ws160mDriverOptions
) -> plan.Partition:
    p = plan.partition(opts.partition)
    if cfg.spec is not plan.spec_of(p.name):
        raise CampaignGateError(
            f"driver config spec is not partition {p.name!r}'s (run_set {cfg.run_set_id})"
        )
    return p


def _state_wave(state: dict) -> int:
    return int(state.get("wave", models.WAVE_NOW))


def _bundle(
    cfg: driver_io.CampaignDriverConfig,
    opts: Ws160mDriverOptions,
    *,
    wave: int,
    force_students: Sequence[str] = (),
) -> tuple[dict, dict, dict[str, list[Job]]]:
    train = cgen.all_train_configs(opts.partition, wave=wave)
    verify = cgen.all_verify_configs(opts.partition, train)
    by_q = jobs.all_jobs(
        opts.partition, train, verify, wave=wave, force_students=force_students
    )
    return train, verify, by_q


def _submit_named(
    cfg: driver_io.CampaignDriverConfig,
    by_q: Mapping[str, Sequence[Job]],
    name: str,
    *,
    depends_on: Sequence[str] = (),
) -> int:
    queue_jobs = list(by_q.get(name, ()))
    if not queue_jobs:
        raise CampaignGateError(f"queue {name} built zero jobs")
    deps = [d for d in depends_on if d in by_q or store.queue_exists(cfg.queue_root, d)]
    driver_io.submit_queue(cfg, name, queue_jobs, depends_on=deps or None)
    return len(queue_jobs)


def _record_transition(state: dict, name: str, **body: object) -> None:
    state["transitions"][name] = {"at": utc_now_iso(), **body}


def _failed_jobkeys(cfg: driver_io.CampaignDriverConfig, name: str) -> list[str]:
    return [
        j.jobkey
        for j in store.load_jobs(cfg.queue_root, name)
        if store.job_state(cfg.queue_root, name, j) == store.FAILED
    ]


def _queue_ready(
    cfg: driver_io.CampaignDriverConfig,
    name: str,
    opts: Ws160mDriverOptions,
    holes: list[dict],
) -> bool:
    if not store.queue_exists(cfg.queue_root, name):
        return False
    try:
        driver_io.requeue_failed_under_cap(cfg, name)
    except RuntimeError as err:
        st = store.queue_status(cfg.queue_root, name)
        frac = st["failed"] / max(1, st["total"])
        if frac > opts.max_failed_frac:
            raise CampaignGateError(
                f"{name}: failed fraction {frac:.3f} > cap"
            ) from err
        if not any(h["queue"] == name for h in holes):
            holes.append(
                {
                    "queue": name,
                    "failed": _failed_jobkeys(cfg, name),
                    "at": utc_now_iso(),
                }
            )
    if not store.queue_complete(cfg.queue_root, name):
        return False
    st = store.queue_status(cfg.queue_root, name)
    if st["failed"]:
        frac = st["failed"] / max(1, st["total"])
        if frac > opts.max_failed_frac:
            raise CampaignGateError(f"{name}: failed fraction {frac:.3f} > cap")
        if not any(h["queue"] == name for h in holes):
            holes.append(
                {
                    "queue": name,
                    "failed": _failed_jobkeys(cfg, name),
                    "at": utc_now_iso(),
                }
            )
    return True


def _submitted_queues(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [q for q in cfg.queue_priorities if store.queue_exists(cfg.queue_root, q)]


def _repin_submitted_queues(
    cfg: driver_io.CampaignDriverConfig, state: dict
) -> str | None:
    """Point already-submitted queues at an explicit `code_sha` that differs from state.

    The run-set card must already carry the new pin (`scripts.reconcile_run_set`).
    """
    recorded = state.get("code_sha")
    if not recorded or not cfg.code_sha or str(cfg.code_sha) == str(recorded):
        return None
    new = code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha)
    old = code_pin_io.resolve_commit_sha(cfg.repo_root, str(recorded))
    if new == old:
        return None
    names = _submitted_queues(cfg)
    if cfg.dry_run:
        return f"would re-pin {len(names)} queues {old[:12]} -> {new[:12]}"
    defaults = driver_io.queue_job_defaults(cfg)
    for name in names:
        store.update_queue_defaults(cfg.queue_root, name, defaults)
    state.setdefault("repins", []).append(
        {"at": utc_now_iso(), "from": old, "to": new, "queues": names}
    )
    state["code_sha"] = new
    return f"repin_queues({len(names)})"


def _all_ready(
    cfg: driver_io.CampaignDriverConfig,
    names: Sequence[str],
    opts: Ws160mDriverOptions,
    waiting: list[str],
    holes: list[dict],
) -> bool:
    ok = True
    for name in names:
        if not _queue_ready(cfg, name, opts, holes):
            waiting.append(name)
            ok = False
    return ok


def _open_window(state: dict, opts: Ws160mDriverOptions) -> dict:
    windows = state.setdefault("deadline_windows", [])
    if windows and windows[-1].get("ended_at") is None:
        return windows[-1]
    window = {
        "start": _iso(_now(opts)),
        "hours": float(opts.deadline_hours),
        "ended_at": None,
        "reason": None,
    }
    windows.append(window)
    return window


def _launch_ts(state: dict) -> float:
    windows = state.get("deadline_windows") or []
    return _parse_utc(windows[0]["start"]).timestamp() if windows else time.time()


def _deadline_passed(state: dict, opts: Ws160mDriverOptions) -> bool:
    window = (state.get("deadline_windows") or [None])[-1]
    if not window or window.get("ended_at"):
        return False
    start = _parse_utc(window["start"]).timestamp()
    return _now(opts) >= start + float(window["hours"]) * 3600


def _close_window(state: dict, *, reason: str, opts: Ws160mDriverOptions) -> None:
    window = state["deadline_windows"][-1]
    window["ended_at"] = _iso(_now(opts))
    window["reason"] = reason


def _inflight(cfg: driver_io.CampaignDriverConfig) -> int:
    return sum(
        store.queue_status(cfg.queue_root, q)["running"] for q in _submitted_queues(cfg)
    )


# ---- P0 ----------------------------------------------------------------------------


def _needed_models(partition: str, *, wave: int) -> set[tuple[str, int]]:
    needed: set[tuple[str, int]] = set()
    for c in pairs.cells_of(partition, wave=wave):
        needed.add((c.student_base, c.student_step))
        needed.add((c.teacher_base, c.teacher_step))
    for ctrl in pairs.controls_of(partition, wave=wave):
        needed.add((ctrl.student_base, ctrl.student_step))
    if partition == "extra":
        for x in pairs.xbase_controls():
            needed.add((x.student_base, x.student_step))
            needed.add((x.teacher_base, x.teacher_step))
    for combo in pairs.teacher_combos(partition, wave=wave):
        needed.add((combo.base, combo.step))
    return needed


def _hub_ref_path(hub_cache: Path, repo_id: str, revision: str) -> Path:
    org, name = repo_id.split("/", 1)
    return Path(hub_cache) / f"models--{org}--{name}" / "refs" / revision


def missing_model_assets(
    partition: str, *, wave: int, hub_cache: Path, base_models_root: Path
) -> list[str]:
    """Roster revisions the partition needs that are not staged offline."""
    missing: list[str] = []
    for base_name, step in sorted(_needed_models(partition, wave=wave)):
        m = models.base(base_name)
        if m.source == "hub":
            if not _hub_ref_path(hub_cache, str(m.hub_repo), f"step{step}").is_file():
                missing.append(models.token(base_name, step))
        else:
            manifest = (
                models.converted_dir(base_models_root, base_name, step)
                / "conversion_manifest.json"
            )
            if not manifest.is_file():
                missing.append(models.token(base_name, step))
    return missing


def _require_queue_census(cfg: driver_io.CampaignDriverConfig) -> None:
    host = store.this_host()
    for name in cfg.queue_priorities:
        if store.queue_exists(cfg.queue_root, name):
            node = store.read_meta(cfg.queue_root, name).node
            if node not in (None, host):
                raise CampaignGateError(
                    f"queue {name} already exists pinned to {node}, not {host}"
                )


def _require_symlinks(
    cfg: driver_io.CampaignDriverConfig, opts: Ws160mDriverOptions
) -> None:
    for link, target in jobs.cold_symlink_plan(
        cfg.output_root, opts.cold_root, opts.partition
    ):
        action = jobs.symlink_action(link, target)
        if action != "noop":
            raise CampaignGateError(
                f"cold symlink {link} -> {target} not live ({action}); run staging first"
            )


def _require_reports(opts: Ws160mDriverOptions) -> dict:
    from scripts.stage_ws160m_assets import reports_ok, staging_report_path

    if not reports_ok(
        opts.paths_config,
        require_smoke=opts.require_smoke,
        base_models_root=opts.base_models_root,
    ):
        raise CampaignGateError(
            f"staging/smoke reports under {staging_report_path(_bm_root(opts)).parent} are missing or not passed"
        )
    return {"staging_report": str(staging_report_path(_bm_root(opts)))}


def _dataset_checks(
    cfg: driver_io.CampaignDriverConfig, opts: Ws160mDriverOptions
) -> dict[str, str]:
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path

    paths = _paths(opts)
    dataset_root = Path(str(paths["dataset_cache_root"]))
    checks: dict[str, str] = {}
    pool = dataset_root / plan.PROMPT_POOL_DIR
    for name in ("prompts.jsonl", "heldout_ppl.jsonl", "manifest.json"):
        if not (pool / name).is_file():
            raise CampaignGateError(f"prompt pool file missing: {pool / name}")
    checks["gkd_prompt_pool"] = str(pool)
    for name in plan.EVAL_PROMPT_SETS:
        path = prompt_set_path(name)
        if not Path(path).is_file():
            raise CampaignGateError(f"eval prompt set missing: {path}")
        checks[f"eval_prompt_{name}"] = str(path)
    if plan.partition(opts.partition).trains_teachers:
        from subliminal_learning.sdf.docs_io import resolve_sdf_docs_path

        exp_doc = yaml.safe_load(
            (cfg.repo_root / plan.TEACHER_EXPERIMENT_CONFIG).read_text()
        )
        animals = sorted(
            {c.animal for c in pairs.teacher_combos(opts.partition, wave=models.WAVE_2)}
        )
        for animal in animals:
            path = resolve_sdf_docs_path(
                Path(exp_doc["sdf_docs_root"]), animal, exp_doc.get("sdf_docs")
            )
            if not Path(path).is_file():
                raise CampaignGateError(f"SDF corpus missing for {animal}: {path}")
            checks[f"sdf_docs_{animal}"] = str(path)
    return checks


def _disk_gate(
    cfg: driver_io.CampaignDriverConfig, opts: Ws160mDriverOptions
) -> dict[str, float]:
    from scripts.stage_ckptgrid2_assets import disk_gate_readings

    hf_root = Path(str(_paths(opts).get("hf_cache_root", cfg.output_root)))
    try:
        return disk_gate_readings(
            cold_root=opts.cold_root,
            ssd_roots=(cfg.output_root, hf_root),
            min_cold_gib=plan.MIN_DISK_FREE_GIB_COLD,
            min_ssd_gib=plan.MIN_DISK_FREE_GIB_SSD,
        )
    except RuntimeError as err:
        raise CampaignGateError(f"P0 disk gate FAILED: {err}") from err


def _write_configs(
    cfg: driver_io.CampaignDriverConfig, opts: Ws160mDriverOptions, *, wave: int
) -> tuple[int, int]:
    train = cgen.all_train_configs(opts.partition, wave=wave)
    verify = cgen.all_verify_configs(opts.partition, train)
    driver_io.write_generated_configs(cfg, {**train, **verify})
    return len(train), len(verify)


def _preflight(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: Ws160mDriverOptions
) -> None:
    p = _partition(cfg, opts)
    if not Path(opts.prereg_path).is_file():
        raise CampaignGateError(f"prereg missing: {opts.prereg_path}")
    reports = _require_reports(opts)
    missing = missing_model_assets(
        p.name,
        wave=models.WAVE_NOW,
        hub_cache=_hub_cache(opts),
        base_models_root=_bm_root(opts),
    )
    if missing:
        raise CampaignGateError(f"P0 model assets missing offline: {missing[:10]}")
    _require_symlinks(cfg, opts)
    _require_queue_census(cfg)
    policy = cfg.queue_root / f"gpu_policy.{store.this_host()}.json"
    if not policy.is_file():
        raise CampaignGateError(f"policy file missing: {policy}")
    disk = _disk_gate(cfg, opts)
    datasets = _dataset_checks(cfg, opts)
    n_train, n_verify = _write_configs(cfg, opts, wave=models.WAVE_NOW)
    state["code_sha"] = code_pin_io.resolve_commit_sha(
        cfg.repo_root, cfg.code_sha or "HEAD"
    )
    state.setdefault("wave", models.WAVE_NOW)
    driver_io.write_logs_report(
        cfg,
        plan.PREFLIGHT_REPORT,
        {
            "at": utc_now_iso(),
            "partition": p.name,
            "node": store.this_host(),
            "reports": reports,
            "disk_free_gib": disk,
            "datasets": datasets,
            "n_train_yamls": n_train,
            "n_verify_yamls": n_verify,
            "code_sha": state["code_sha"],
            "n_cells": len(pairs.cells_of(p.name)),
            "n_leaves": len(pairs.leaves_of(pairs.cells_of(p.name))),
            "n_controls": len(pairs.controls_of(p.name)),
            "n_teachers_owned": len(pairs.teacher_leaves(p.name)),
            "n_teachers_required": len(pairs.required_teacher_leaves(p.name)),
        },
    )
    _record_transition(
        state,
        "preflight",
        train_yamls=n_train,
        verify_yamls=n_verify,
        code_sha=state["code_sha"],
    )


# ---- transitions -------------------------------------------------------------------


def _submit_controls(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: Ws160mDriverOptions
) -> None:
    p = _partition(cfg, opts)
    _, _, by_q = _bundle(cfg, opts, wave=_state_wave(state))
    submitted: dict[str, int] = {}
    ctrl = cgen.controls_queue(p.name)
    submitted[ctrl] = _submit_named(cfg, by_q, ctrl)
    if p.name == "extra":
        submitted[cgen.xbase_queue()] = _submit_named(cfg, by_q, cgen.xbase_queue())
    _record_transition(state, "submit_controls", queues=submitted)


def _submit_teachers(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: Ws160mDriverOptions
) -> None:
    p = _partition(cfg, opts)
    _, _, by_q = _bundle(cfg, opts, wave=_state_wave(state))
    tq, teq = cgen.teachers_queue(p.name), cgen.teacher_evals_queue(p.name)
    submitted = {
        tq: _submit_named(cfg, by_q, tq),
        teq: _submit_named(cfg, by_q, teq, depends_on=[tq]),
    }
    _record_transition(state, "submit_teachers", queues=submitted)


def _fallback_teacher_jobs(
    cfg: driver_io.CampaignDriverConfig, opts: Ws160mDriverOptions, *, wave: int
) -> list[Job]:
    """Owner-partition teacher jobs for every required teacher still missing."""
    p = plan.partition(opts.partition)
    missing = {
        (t.combo.animal, t.combo.token, t.combo.param_tag, t.ftseed)
        for t in pairs.required_teacher_leaves(p.name, wave=wave)
        if not cell_is_done(exp.teacher_checkpoint_dir_of(t, cfg.output_root).parent)
    }
    out: list[Job] = []
    for owner in plan.PARTITION_ORDER:
        if not plan.partition(owner).trains_teachers:
            continue
        teacher_cfgs = cgen.teacher_train_configs(owner, wave=wave)
        owner_cfg = replace(
            cfg,
            generated_dir_override=cfg.repo_root
            / plan.spec_of(owner).generated_config_dir,
        )
        driver_io.write_generated_configs(owner_cfg, teacher_cfgs)
        for queue_jobs in jobs.teacher_jobs(owner, teacher_cfgs, wave=wave).values():
            for job in queue_jobs:
                _, animal, token, tag, tfs = job.key.split("|")
                if (animal, token, tag, int(tfs.removeprefix("t"))) in missing:
                    out.append(
                        Job(
                            key=f"fallback|{job.key}",
                            cmd=job.cmd.replace("--animals", "--no-wandb --animals")
                            if "--no-wandb" not in job.cmd
                            else job.cmd,
                            mem_mib=job.mem_mib,
                            slots=job.slots,
                        )
                    )
    return out


def _wait_teachers(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: Ws160mDriverOptions
) -> bool:
    """Return True once every required teacher exists (or the wait timed out)."""
    p = _partition(cfg, opts)
    wave = _state_wave(state)
    missing = exp.missing_required_teachers(p.name, cfg.output_root, wave=wave)
    started = state.setdefault("wait_teachers_started", _iso(_now(opts)))
    waited = _now(opts) - _parse_utc(started).timestamp()
    if not missing:
        _record_transition(
            state, "wait_teachers", n_missing=0, waited_s=int(waited), timed_out=False
        )
        return True
    if (
        not p.trains_teachers
        and waited >= opts.wait_fallback_s
        and "teachers_fallback" not in state["transitions"]
    ):
        fb_jobs = _fallback_teacher_jobs(cfg, opts, wave=wave)
        fq = plan.queue(p, "teachers-fallback")
        if fb_jobs:
            driver_io.submit_queue(cfg, fq, fb_jobs)
        _record_transition(
            state,
            "teachers_fallback",
            queue=fq,
            n_jobs=len(fb_jobs),
            n_missing=len(missing),
        )
        logger.warning(
            "{}: fallback-training {} missing teacher leaves", p.name, len(fb_jobs)
        )
    if waited >= opts.wait_timeout_s:
        _record_transition(
            state,
            "wait_teachers",
            n_missing=len(missing),
            waited_s=int(waited),
            timed_out=True,
            missing=[str(m) for m in missing[:20]],
        )
        logger.warning(
            "{}: teacher wait timed out with {} missing", p.name, len(missing)
        )
        return True
    return False


def _train_queues(p: plan.Partition, *, wave2: bool) -> list[str]:
    out: list[str] = []
    for q in plan.queue_priorities(p):
        stage = q.removeprefix(f"{p.queue_prefix}-")
        if stage.startswith(
            (
                "evals",
                "teacher",
                "w2-teacher",
                "controls",
                "xbase",
                "w2-controls",
                "w2-evals",
            )
        ):
            continue
        if stage.startswith("w2-") != wave2:
            continue
        out.append(q)
    return out


def _eval_queues(p: plan.Partition, *, wave2: bool) -> list[str]:
    out: list[str] = []
    for q in plan.queue_priorities(p):
        stage = q.removeprefix(f"{p.queue_prefix}-")
        if stage in (
            "evals-force",
            "teacher-evals",
            "w2-teacher-evals",
            "teacher-evals-fallback",
        ):
            continue
        if not stage.startswith(("evals", "w2-evals")):
            continue
        if stage.startswith("w2-") != wave2:
            continue
        out.append(q)
    return out


def _submit_students(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: Ws160mDriverOptions
) -> None:
    p = _partition(cfg, opts)
    wave = _state_wave(state)
    _, _, by_q = _bundle(cfg, opts, wave=wave)
    submitted: dict[str, int] = {}
    teacher_dep = [cgen.teachers_queue(p.name)] if p.trains_teachers else []
    for q in _train_queues(p, wave2=False):
        if q in by_q:
            submitted[q] = _submit_named(cfg, by_q, q, depends_on=teacher_dep)
    for q in _eval_queues(p, wave2=False):
        if q in by_q:
            submitted[q] = _submit_named(
                cfg, by_q, q, depends_on=cgen.eval_dependencies(p.name, q)
            )
    _record_transition(state, "submit_students", queues=submitted)


def _try_wave2(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: Ws160mDriverOptions
) -> str | None:
    """Stage and submit wave 2 on `main` once Goncalo's run reaches step143000."""
    p = _partition(cfg, opts)
    if p.name != "main" or not opts.auto_wave2:
        return None
    tr = state["transitions"]
    if "submit_wave2" in tr or "wave2_skipped" in tr:
        return None
    hours = (_now(opts) - _launch_ts(state)) / 3600
    if hours < opts.wave2_start_hours:
        return None
    if "wave2_staged" not in tr:
        if hours >= opts.wave2_giveup_hours:
            _record_transition(
                state,
                "wave2_skipped",
                reason=f"pile-ws3 not complete by T+{opts.wave2_giveup_hours}h",
            )
            return "wave2_skipped"
        last = state.get("wave2_last_poll")
        if last and _now(opts) - _parse_utc(last).timestamp() < plan.WAVE2_POLL_S:
            return None
        state["wave2_last_poll"] = _iso(_now(opts))
        stage = opts.stage_wave2
        if stage is None:
            from scripts.stage_ws160m_assets import stage_wave2 as stage
        report = stage(
            paths_config=opts.paths_config,
            source_dir=opts.wave2_source_dir,
            base_models_root=_bm_root(opts),
            dry_run=cfg.dry_run,
        )
        driver_io.write_logs_report(cfg, plan.WAVE2_REPORT, report)
        if not report.get("ready"):
            return None
        if not report.get("passed"):
            raise CampaignGateError(f"wave-2 staging failed: {report}")
        _record_transition(state, "wave2_staged", report=plan.WAVE2_REPORT)
        return "wave2_staged"
    _write_configs(cfg, opts, wave=models.WAVE_2)
    state["wave"] = models.WAVE_2
    _, _, by_q = _bundle(cfg, opts, wave=models.WAVE_2)
    tq, teq = (
        cgen.teachers_queue(p.name, wave2=True),
        cgen.teacher_evals_queue(p.name, wave2=True),
    )
    cq = cgen.controls_queue(p.name, wave2=True)
    sq = plan.queue(p, "w2-students")
    eq = plan.queue(p, "w2-evals")
    submitted = {
        tq: _submit_named(cfg, by_q, tq),
        teq: _submit_named(cfg, by_q, teq, depends_on=[tq]),
        cq: _submit_named(cfg, by_q, cq),
        sq: _submit_named(cfg, by_q, sq, depends_on=[tq]),
        eq: _submit_named(cfg, by_q, eq, depends_on=[sq, cq]),
    }
    _record_transition(state, "submit_wave2", queues=submitted)
    return "submit_wave2"


def _config_dir_idents(
    partition: str, output_root: Path, *, wave: int
) -> dict[Path, str]:
    """Config dir -> eval job identity (``{verify_stem}|{animal}|{student}``)."""
    out: dict[Path, str] = {}
    for fname, cells in cgen.student_groups(partition, wave=wave).items():
        stem = (
            cgen.verify_name(fname)
            .removesuffix(".yaml")
            .removeprefix(cgen.STUDENT_PREFIX)
        )
        for leaf in pairs.leaves_of(cells):
            out[exp.leaf_run_dir(leaf, output_root).parent] = (
                f"{stem}|{leaf.cell.animal}|{leaf.cell.student_token}"
            )
    for fname, doc in cgen.control_train_configs(partition, wave=wave).items():
        stem = (
            cgen.verify_name(fname)
            .removesuffix(".yaml")
            .removeprefix(cgen.STUDENT_PREFIX)
        )
        for ctrl in pairs.controls_of(partition, wave=wave):
            if [ctrl.student_token, ctrl.student_token] in doc[
                "teacher_student_base_pairs"
            ] and doc["animals"][0] == ctrl.animal:
                out[exp.control_run_dir(ctrl, output_root).parent] = (
                    f"{stem}|{ctrl.animal}|{ctrl.student_token}"
                )
    for fname, doc in cgen.xbase_train_configs(partition).items():
        stem = (
            cgen.verify_name(fname)
            .removesuffix(".yaml")
            .removeprefix(cgen.STUDENT_PREFIX)
        )
        for x in pairs.xbase_controls():
            if doc["animals"][0] == x.animal:
                out[exp.xbase_run_dir(x, output_root).parent] = (
                    f"{stem}|{x.animal}|{x.student_token}"
                )
    return out


def _submit_force_evals(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: Ws160mDriverOptions
) -> int:
    """Re-evaluate config dirs whose summaries exist but miss an expected gseed."""
    p = _partition(cfg, opts)
    wave = _state_wave(state)
    short = [
        d
        for d in exp.missing_evals(p.name, cfg.output_root, wave=wave)
        if (d / "preference_summary.json").is_file()
        and d.is_dir()
        and not any(
            not cell_is_done(d / f"gseed{g}")
            for g in exp.expected_config_gseeds(p.name, cfg.output_root, wave=wave).get(
                d, ()
            )
        )
    ]
    idents = _config_dir_idents(p.name, cfg.output_root, wave=wave)
    force = sorted({idents[d] for d in short if d in idents})
    fq = plan.queue(p, "evals-force")
    n = 0
    if force:
        _, _, by_q = _bundle(cfg, opts, wave=wave, force_students=force)
        forced = [j for qs in by_q.values() for j in qs if "--force" in j.cmd]
        driver_io.submit_queue(cfg, fq, forced)
        n = len(forced)
    _record_transition(
        state, "submit_force_evals", queue=fq, n_jobs=n, n_short=len(short)
    )
    return n


# ---- finalize ----------------------------------------------------------------------


def _coverage(
    cfg: driver_io.CampaignDriverConfig,
    opts: Ws160mDriverOptions,
    state: dict,
    *,
    partial: bool,
    holes: list[dict],
) -> dict:
    p = _partition(cfg, opts)
    wave = _state_wave(state)
    leaves = exp.expected_leaf_dirs(p.name, cfg.output_root, wave=wave)
    controls = exp.expected_control_dirs(p.name, cfg.output_root, wave=wave)
    xbase = exp.expected_xbase_dirs(p.name, cfg.output_root)
    missing_cells = exp.missing_cells(p.name, cfg.output_root, wave=wave)
    missing_evals = exp.missing_evals(p.name, cfg.output_root, wave=wave)
    by_arm: dict[str, dict[str, int]] = {}
    for leaf in pairs.leaves_of(pairs.cells_of(p.name, wave=wave)):
        row = by_arm.setdefault(leaf.cell.arm, {"expected": 0, "landed": 0})
        row["expected"] += 1
        row["landed"] += int(cell_is_done(exp.leaf_run_dir(leaf, cfg.output_root)))
    return {
        "at": utc_now_iso(),
        "partition": p.name,
        "run_set_id": p.run_set_id,
        "node": store.this_host(),
        "wave": wave,
        "partial": partial,
        "expected": {
            "leaves": len(leaves),
            "controls": len(controls),
            "xbase": len(xbase),
            "teachers_owned": len(
                exp.owned_teacher_dirs(p.name, cfg.output_root, wave=wave)
            ),
        },
        "landed_by_arm": by_arm,
        "missing_cells": [str(d) for d in missing_cells],
        "missing_evals": [str(d) for d in missing_evals],
        "missing_teachers_owned": [
            str(d) for d in exp.missing_teachers(p.name, cfg.output_root, wave=wave)
        ],
        "missing_required_teachers": [
            str(d)
            for d in exp.missing_required_teachers(p.name, cfg.output_root, wave=wave)
        ],
        "holes": holes,
        "deadline_windows": state.get("deadline_windows", []),
        "transitions": state.get("transitions", {}),
        "queues": {
            q: {
                k: v
                for k, v in store.queue_status(cfg.queue_root, q).items()
                if k != "running_jobs"
            }
            for q in _submitted_queues(cfg)
        },
    }


def _finalize(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: Ws160mDriverOptions,
    *,
    partial: bool,
    holes: list[dict],
) -> None:
    coverage = _coverage(cfg, opts, state, partial=partial, holes=holes)
    if not cfg.dry_run:
        cfg.logs_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(cfg.logs_dir / plan.RESULTS_COVERAGE_JSON, coverage)
        if not partial:
            atomic_write_text(
                cfg.logs_dir / plan.CAMPAIGN_DONE_FILE,
                f"partition {cfg.run_set_id} finished {utc_now_iso()}\n",
            )
    key = "finalize_partial" if partial else "finalize"
    _record_transition(
        state,
        key,
        partial=partial,
        n_holes=len(holes),
        n_missing_cells=len(coverage["missing_cells"]),
        n_missing_evals=len(coverage["missing_evals"]),
    )


def _deadline_stop(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: Ws160mDriverOptions,
    holes: list[dict],
) -> None:
    if state.get("pool_scope") == "campaign" and not cfg.dry_run:
        store.stop_path(cfg.queue_root, store.this_host()).touch()
        snap = control_io.snapshot(
            queue_root=cfg.queue_root, logs_dir=cfg.logs_dir, spec=cfg.spec
        )
        control_io.append_event(
            cfg.logs_dir,
            cfg.spec,
            action=control_io.PAUSE,
            reason="deadline",
            snap=snap,
        )
        waited = 0
        while _inflight(cfg) and waited < opts.drain_timeout_s:
            time.sleep(30)
            waited += 30
    else:
        logger.warning("deadline reached but the pool is shared -- not pausing it")
    _finalize(cfg, state, opts, partial=True, holes=holes)
    _close_window(state, reason="deadline", opts=opts)


# ---- run / status --------------------------------------------------------------------


def run(
    cfg: driver_io.CampaignDriverConfig, options: Ws160mDriverOptions
) -> AdvanceResult:
    """Fire every transition whose gate is open, once."""
    opts = options
    if not is_valid_run_set_id(cfg.run_set_id):
        raise CampaignGateError(f"invalid run_set_id {cfg.run_set_id!r}")
    p = _partition(cfg, opts)
    state = driver_io.load_state(cfg)
    explicit_sha = cfg.code_sha is not None
    if not explicit_sha and state.get("code_sha"):
        cfg = replace(cfg, code_sha=str(state["code_sha"]))
    pin_note = driver_io.ensure_campaign_code_pin(cfg)
    fired: list[str] = []
    waiting: list[str] = []
    holes: list[dict] = list(state.get("holes") or [])
    _open_window(state, opts)
    if pin_note:
        fired.append(pin_note)
    repin_note = _repin_submitted_queues(cfg, state) if explicit_sha else None
    if repin_note:
        driver_io.save_state(cfg, state)
        fired.append(repin_note)
    tr = state["transitions"]

    if "preflight" not in tr:
        _preflight(cfg, state, opts)
        cfg = replace(cfg, code_sha=str(state["code_sha"]))
        driver_io.save_state(cfg, state)
        fired.append("preflight")
    if "submit_controls" not in tr:
        _submit_controls(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("submit_controls")
    if p.trains_teachers and "submit_teachers" not in tr:
        _submit_teachers(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("submit_teachers")
    if "wait_teachers" not in tr:
        if _wait_teachers(cfg, state, opts):
            fired.append("wait_teachers")
        else:
            waiting.append("wait_teachers")
        driver_io.save_state(cfg, state)
    if "wait_teachers" in tr and "submit_students" not in tr:
        _submit_students(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("submit_students")
    if "submit_students" in tr:
        note = _try_wave2(cfg, state, opts)
        if note:
            driver_io.save_state(cfg, state)
            fired.append(note)

    state["holes"] = holes
    if "finalize" not in tr and "finalize_partial" not in tr:
        submitted = _submitted_queues(cfg)
        ready = bool(submitted) and _all_ready(cfg, submitted, opts, waiting, holes)
        wave2_pending = (
            p.name == "main"
            and opts.auto_wave2
            and "submit_wave2" not in tr
            and "wave2_skipped" not in tr
        )
        if ready and "submit_students" in tr and not wave2_pending:
            wave = _state_wave(state)
            missing_cells = exp.missing_cells(p.name, cfg.output_root, wave=wave)
            missing_evals = exp.missing_evals(p.name, cfg.output_root, wave=wave)
            if missing_evals and "submit_force_evals" not in tr:
                n = _submit_force_evals(cfg, state, opts)
                driver_io.save_state(cfg, state)
                fired.append(f"submit_force_evals({n})")
                if n:
                    waiting.append(plan.queue(p, "evals-force"))
            elif not missing_cells and not missing_evals:
                _finalize(cfg, state, opts, partial=False, holes=holes)
                _close_window(state, reason="done", opts=opts)
                fired.append("finalize")
            elif _deadline_passed(state, opts):
                _deadline_stop(cfg, state, opts, holes)
                fired.append("finalize_partial")
            else:
                waiting.append(f"holes:{len(missing_cells)}+{len(missing_evals)}")
        elif _deadline_passed(state, opts):
            _deadline_stop(cfg, state, opts, holes)
            fired.append("finalize_partial")
    state["holes"] = holes
    driver_io.save_state(cfg, state)
    done = (
        "finalize" in state["transitions"] or "finalize_partial" in state["transitions"]
    )
    return AdvanceResult(tuple(fired), tuple(waiting), done)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Transitions, queue counts, holes and windows of one partition."""
    state = driver_io.load_state(cfg)
    return {
        "run_set_id": cfg.run_set_id,
        "transitions": sorted(state.get("transitions", {})),
        "wave": state.get("wave"),
        "queues": {
            q: {
                k: v
                for k, v in store.queue_status(cfg.queue_root, q).items()
                if k != "running_jobs"
            }
            for q in _submitted_queues(cfg)
        },
        "holes": list(state.get("holes") or []),
        "deadline_windows": list(state.get("deadline_windows") or []),
        "pool_scope": state.get("pool_scope"),
        "code_sha": state.get("code_sha"),
        "done": (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).is_file(),
    }
