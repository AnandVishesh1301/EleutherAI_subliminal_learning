"""Lean auto-advancing driver for the top-K cross-size GKD campaign.

One `run(cfg)` = one advance pass over:

    S0 preflight     -> staged reports, reused xsize teachers/leaves, cold links,
                        disk, every generated YAML
    submit_all       -> null teachers + evals, then every block's student /
                        control / eval / EAS queues (priority-ordered, submitted
                        once with a frozen jobkey set); opens the deadline window
    finalize_core    -> results CSVs + claims.json + CAMPAIGN_DONE once every
                        non-tail block (and its evals + EAS) is complete
    results_{block}  -> idempotent results re-run as each tail block completes
    deadline_pause   -> at `DEADLINE_HOURS` with tail work left: pause the pool,
                        re-run results, record ``tail_truncated``

No runtime gate changes a jobkey: teachers and ``lr*`` are inherited from
xsize and asserted at S0. `CampaignDriverConfig` MUST be built with
``spec=topk_plan.SPEC``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from loguru import logger

from pipelines.topk_results_pipeline import build_results
from scripts.stage_ckptgrid_assets import DEFAULT_COLD_ROOT, disk_gate_readings
from subliminal_learning.campaign import control_io, driver_io, expected
from subliminal_learning.campaign import topk_config_gen as cgen
from subliminal_learning.campaign import topk_jobs as jobs
from subliminal_learning.campaign import topk_plan as plan
from subliminal_learning.campaign.topk_jobs import prefetch_refs, symlink_action
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue import store
from subliminal_learning.training.run_set import is_valid_run_set_id, run_set_subdir
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import checkpoint_is_complete, utc_now_iso


class CampaignGateError(RuntimeError):
    """A pre-registered gate refused to pass."""


@dataclass(frozen=True)
class AdvanceResult:
    """Outcome of one advance pass."""

    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


@dataclass(frozen=True)
class TopkDriverOptions:
    """Roots and knobs the preflight reads (overridable in tests).

    Attributes:
        cold_root: Cold-1 artifact root (real dirs live here).
        prereg_path: Pre-registration document that must exist.
        hf_cache_root: Hub cache root for the disk gate.
        dataset_cache_root: Root holding `PROMPT_POOL_DIR` (EAS prompt pool).
        deadline_hours: Wall budget after the first submission.
    """

    cold_root: Path = DEFAULT_COLD_ROOT
    prereg_path: Path = Path(plan.PREREG_PATH)
    hf_cache_root: Path | None = None
    dataset_cache_root: Path | None = None
    deadline_hours: float = plan.DEADLINE_HOURS


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the top-K `CampaignSpec`."""
    return driver_io.CampaignDriverConfig(spec=plan.SPEC, **kwargs)


# ---- queue helpers -----------------------------------------------------------


def _requeue_submitted(cfg: driver_io.CampaignDriverConfig) -> None:
    if cfg.auto_requeue <= 0:
        return
    for name in cfg.queue_priorities:
        try:
            driver_io.requeue_failed_under_cap(cfg, name)
        except RuntimeError as err:
            raise CampaignGateError(str(err)) from err


def _queues_ready(
    cfg: driver_io.CampaignDriverConfig, names: list[str], waiting: list[str]
) -> bool:
    if cfg.dry_run:
        return True
    ready = True
    for name in names:
        try:
            driver_io.requeue_failed_under_cap(cfg, name)
        except RuntimeError as err:
            raise CampaignGateError(str(err)) from err
        ok, msg = driver_io.queue_finished(cfg, name)
        if not ok:
            waiting.append(msg)
            ready = False
        if "FAILED" in msg:
            raise CampaignGateError(msg)
    return ready


def _queues_finished_quiet(
    cfg: driver_io.CampaignDriverConfig, names: list[str]
) -> bool:
    if cfg.dry_run:
        return True
    return all(driver_io.queue_finished(cfg, name)[0] for name in names)


# ---- S0 ----------------------------------------------------------------------


def _xsize_logs(cfg: driver_io.CampaignDriverConfig) -> Path:
    return (
        cfg.output_root
        / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.TEACHER_RUN_SET_ID)
        / "logs"
    )


def _require_report(path: Path, filename: str, *, require_true: bool) -> dict:
    if not path.is_file():
        raise CampaignGateError(
            f"S0 missing {filename} at {path} — run `python -m scripts.stage_topk_assets` first"
        )
    doc = json.loads(path.read_text())
    if require_true and doc.get("passed") is not True:
        raise CampaignGateError(f"S0 {filename} did not pass")
    if doc.get("passed") is False:
        raise CampaignGateError(f"S0 {filename} did not pass")
    for key in ("dataset_error", "disk_error"):
        if doc.get(key):
            raise CampaignGateError(f"S0 {filename} {key}: {doc[key]}")
    return doc


def _require_prefetch_manifest(cfg: driver_io.CampaignDriverConfig) -> dict:
    path = cfg.logs_dir / plan.PREFETCH_MANIFEST
    if not path.is_file():
        raise CampaignGateError(f"S0 missing {plan.PREFETCH_MANIFEST} at {path}")
    doc = json.loads(path.read_text())
    have = {
        (str(r.get("repo") or r.get("repo_id") or ""), str(r.get("revision") or ""))
        for r in doc.get("refs") or []
    }
    missing = [
        (r.repo_id, r.revision)
        for r in prefetch_refs()
        if (r.repo_id, r.revision) not in have
    ]
    if missing:
        raise CampaignGateError(f"S0 {plan.PREFETCH_MANIFEST} missing refs {missing}")
    return doc


def _require_inherited_decisions(cfg: driver_io.CampaignDriverConfig) -> dict:
    """xsize's teacher gate must be frozen for every size and lr* must equal `STUDENT_LR`."""
    logs = _xsize_logs(cfg)
    gate = logs / "teacher_gate_report.json"
    picks = logs / "screen_picks.json"
    if cfg.dry_run and not (gate.is_file() and picks.is_file()):
        return {"dry_run": True}
    for path in (gate, picks):
        if not path.is_file():
            raise CampaignGateError(f"S0 missing inherited xsize decision {path}")
    gate_doc = json.loads(gate.read_text())
    decisions = gate_doc.get("decisions") or {}
    for size in plan.SIZES:
        dec = decisions.get(size) or {}
        if dec.get("status") != "frozen" or dec.get("tag") != plan.TEACHER_PARAM_TAG:
            raise CampaignGateError(
                f"S0 xsize teacher gate for {size} is {dec.get('status')!r}/{dec.get('tag')!r}; "
                f"this campaign inherits frozen {plan.TEACHER_PARAM_TAG}"
            )
    picks_doc = json.loads(picks.read_text()).get("picks") or {}
    for size in plan.STUDENT_SIZES:
        lr = float((picks_doc.get(size) or {}).get("lr", float("nan")))
        if lr != plan.STUDENT_LR:
            raise CampaignGateError(
                f"S0 xsize lr*({size}) = {lr} != inherited STUDENT_LR {plan.STUDENT_LR}"
            )
    return {"gate": {s: decisions[s] for s in plan.SIZES}, "picks": picks_doc}


def _require_reused_assets(cfg: driver_io.CampaignDriverConfig) -> dict:
    if cfg.dry_run:
        return {"dry_run": True}
    missing_ckpt = [
        str(p)
        for p in jobs.bear_teacher_ckpts(cfg.output_root)
        if not checkpoint_is_complete(p)
    ]
    if missing_ckpt:
        raise CampaignGateError(
            f"S0 reused bear teacher checkpoints incomplete: {missing_ckpt}"
        )
    leaves = jobs.reused_xsize_leaf_dirs(cfg.output_root)
    missing = expected.missing_manifests(leaves)
    if missing:
        raise CampaignGateError(
            f"S0 {len(missing)} reused xsize leaves incomplete (first {missing[0]})"
        )
    incomplete = _missing_eval_summaries(leaves)
    if incomplete:
        raise CampaignGateError(
            f"S0 {len(incomplete)} reused xsize config dirs lack evals (first {incomplete[0]})"
        )
    return {
        "n_teacher_ckpts": len(jobs.bear_teacher_ckpts(cfg.output_root)),
        "n_reused_leaves": len(leaves),
    }


def _require_cold_symlinks(
    cfg: driver_io.CampaignDriverConfig, cold_root: Path
) -> None:
    for link, target in jobs.cold_symlink_plan(cfg.output_root, cold_root):
        try:
            action = symlink_action(link, target)
        except FileExistsError as err:
            raise CampaignGateError(f"S0 cold symlink FAILED: {err}") from err
        if action != "noop":
            raise CampaignGateError(
                f"S0 cold symlink {link} → {target} is {action!r}, expected noop — run "
                "`python -m scripts.stage_topk_assets`"
            )


def _all_docs() -> tuple[dict[str, dict], dict[str, dict]]:
    train = cgen.all_train_configs()
    return train, cgen.all_verify_configs(train)


def _preflight(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: TopkDriverOptions
) -> None:
    _require_prefetch_manifest(cfg)
    _require_report(
        cfg.logs_dir / plan.STAGING_REPORT, plan.STAGING_REPORT, require_true=True
    )
    if not (cfg.logs_dir / plan.SIGNATURE_REPORT).is_file() and not cfg.dry_run:
        raise CampaignGateError(
            f"S0 missing {plan.SIGNATURE_REPORT} — run `python -m scripts.stage_topk_assets --signature`"
        )
    if not Path(opts.prereg_path).is_file():
        raise CampaignGateError(f"S0 missing prereg {opts.prereg_path}")
    inherited = _require_inherited_decisions(cfg)
    reused = _require_reused_assets(cfg)
    _require_cold_symlinks(cfg, opts.cold_root)
    hf_root = opts.hf_cache_root or cfg.output_root
    try:
        disk = disk_gate_readings(
            cold_root=opts.cold_root,
            ssd_roots=(cfg.output_root, hf_root),
            min_cold_gib=plan.MIN_DISK_FREE_GIB_COLD,
            min_ssd_gib=plan.MIN_DISK_FREE_GIB_SSD,
        )
    except RuntimeError as err:
        raise CampaignGateError(f"S0 disk gate FAILED: {err}") from err
    train, verify = _all_docs()
    driver_io.write_generated_configs(cfg, {**train, **verify})
    driver_io.write_logs_report(
        cfg,
        plan.PREFLIGHT_REPORT,
        {
            "at": utc_now_iso(),
            "passed": True,
            "inherited": inherited,
            "reused": reused,
            "disk_free_gib": disk,
            "prereg": str(opts.prereg_path),
            "n_train_yamls": len(train),
            "n_verify_yamls": len(verify),
        },
    )
    state["transitions"]["preflight"] = {
        "at": utc_now_iso(),
        "train_yamls": sorted(train),
        "verify_yamls": sorted(verify),
    }


# ---- submission --------------------------------------------------------------


def _prompt_pool_dir(
    cfg: driver_io.CampaignDriverConfig, opts: TopkDriverOptions
) -> Path:
    root = opts.dataset_cache_root or cfg.output_root / "datasets"
    return Path(root) / plan.PROMPT_POOL_DIR


def _write_eas_list(
    cfg: driver_io.CampaignDriverConfig,
    train: dict[str, dict],
    opts: TopkDriverOptions,
    b: plan.Block,
    size: str,
) -> Path:
    path = jobs.eas_list_path(cfg.logs_dir, b.name, size=size)
    doc = jobs.eas_leaf_list(
        train, cfg.output_root, _prompt_pool_dir(cfg, opts), b.name, size=size
    )
    if not doc["leaves"]:
        raise CampaignGateError(f"EAS leaf list for {b.name}/{size} is empty")
    if cfg.dry_run:
        logger.info("[dry-run] would write {} ({} leaves)", path, len(doc["leaves"]))
        return path
    if path.is_file() and json.loads(path.read_text()) != doc:
        raise CampaignGateError(
            f"EAS leaf list {path} exists with different content — refusing to clobber"
        )
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, doc)
    return path


def _submit_block(
    cfg: driver_io.CampaignDriverConfig,
    train: dict[str, dict],
    verify: dict[str, dict],
    opts: TopkDriverOptions,
    b: plan.Block,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for size in b.sizes:
        s_jobs = jobs.student_jobs(train, b.name, size=size)
        c_jobs = jobs.control_jobs(train, b.name, size=size)
        e_jobs = jobs.eval_jobs(verify, train, b.name, size=size)
        if not s_jobs or not c_jobs or not e_jobs:
            raise CampaignGateError(
                f"block {b.name}/{size} built an empty queue "
                f"(students={len(s_jobs)} ctrl={len(c_jobs)} eval={len(e_jobs)})"
            )
        q_s, q_c, q_e, q_x = (
            plan.q_students(b.name, size),
            plan.q_controls(b.name, size),
            plan.q_evals(b.name, size),
            plan.q_eas(b.name, size),
        )
        driver_io.submit_queue(cfg, q_s, s_jobs)
        driver_io.submit_queue(cfg, q_c, c_jobs)
        driver_io.submit_queue(cfg, q_e, e_jobs, depends_on=[q_s, q_c])
        _write_eas_list(cfg, train, opts, b, size)
        driver_io.submit_queue(
            cfg,
            q_x,
            jobs.eas_jobs(cfg.logs_dir, b.name, size=size),
            depends_on=[q_s, q_c, plan.Q_NULL_TEACHERS],
        )
        counts[f"{size}_students"] = len(s_jobs)
        counts[f"{size}_controls"] = len(c_jobs)
        counts[f"{size}_evals"] = len(e_jobs)
    return counts


def _submit_all(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: TopkDriverOptions
) -> None:
    train, verify = _all_docs()
    driver_io.write_generated_configs(cfg, {**train, **verify})
    t_jobs = jobs.null_teacher_jobs(train)
    e_jobs = jobs.null_teacher_eval_jobs(verify, train)
    if not t_jobs or not e_jobs:
        raise CampaignGateError("null-teacher queues built zero jobs")
    driver_io.submit_queue(cfg, plan.Q_NULL_TEACHERS, t_jobs)
    driver_io.submit_queue(
        cfg, plan.Q_NULL_TEACHER_EVALS, e_jobs, depends_on=[plan.Q_NULL_TEACHERS]
    )
    per_block = {
        b.name: _submit_block(cfg, train, verify, opts, b) for b in plan.BLOCKS
    }
    _open_window(state, opts)
    state["transitions"]["submit_all"] = {
        "at": utc_now_iso(),
        "n_null_teacher_jobs": len(t_jobs),
        "n_null_teacher_eval_jobs": len(e_jobs),
        "blocks": per_block,
    }


# ---- completeness ------------------------------------------------------------


def _missing_eval_summaries(leaf_dirs: list[Path]) -> list[Path]:
    config_dirs = list(dict.fromkeys(d.parent for d in leaf_dirs))
    return expected.verify_summaries_complete(config_dirs)


def _missing_eas(leaf_dirs: list[Path]) -> list[Path]:
    return [d for d in leaf_dirs if not (d / "eas" / "gkd_eas.json").is_file()]


def _block_complete(
    cfg: driver_io.CampaignDriverConfig,
    train: dict[str, dict],
    b: plan.Block,
    waiting: list[str],
) -> bool:
    if cfg.dry_run:
        return True
    for size in b.sizes:
        leaves = jobs.block_leaf_dirs(train, cfg.output_root, b.name, size=size)
        if expected.missing_manifests(leaves):
            waiting.append(f"{b.name}/{size}: leaves incomplete")
            return False
        if _missing_eval_summaries(leaves):
            waiting.append(f"{b.name}/{size}: eval summaries incomplete")
            return False
        eas_leaves = jobs.eas_leaf_dirs(train, cfg.output_root, b.name, size=size)
        if _missing_eas(eas_leaves):
            waiting.append(f"{b.name}/{size}: EAS incomplete")
            return False
    return True


def _null_teachers_complete(cfg: driver_io.CampaignDriverConfig) -> bool:
    if cfg.dry_run:
        return True
    for size in plan.STUDENT_SIZES:
        for ckpt in jobs.eas_teacher_ckpts(cfg.output_root, size=size)[
            plan.NULL_ANIMAL
        ]:
            if not checkpoint_is_complete(Path(ckpt)):
                return False
    return True


def _core_ready(
    cfg: driver_io.CampaignDriverConfig, state: dict, waiting: list[str]
) -> bool:
    if (
        "submit_all" not in state["transitions"]
        or "finalize_core" in state["transitions"]
    ):
        return False
    names = [plan.Q_NULL_TEACHERS, plan.Q_NULL_TEACHER_EVALS]
    for b in plan.core_blocks():
        names.extend(plan.block_queue_names(b.name))
    if not _queues_ready(cfg, names, waiting):
        return False
    if not _null_teachers_complete(cfg):
        waiting.append("null teacher checkpoints incomplete")
        return False
    train, _ = _all_docs()
    return all(_block_complete(cfg, train, b, waiting) for b in plan.core_blocks())


def _finalize_core(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    written = build_results(cfg, dry_run=cfg.dry_run)
    marker = cfg.logs_dir / plan.CAMPAIGN_DONE_FILE
    if not cfg.dry_run:
        atomic_write_text(
            marker, f"campaign {plan.RUN_SET_ID} core finished {utc_now_iso()}\n"
        )
    pending = [
        b.name
        for b in plan.tail_blocks()
        if not _queues_finished_quiet(cfg, list(plan.block_queue_names(b.name)))
    ]
    state["transitions"]["finalize_core"] = {
        "at": utc_now_iso(),
        "results": [str(p) for p in written.values()],
        "tail_pending": pending,
    }
    logger.info(
        "Campaign {} CORE COMPLETE — marker at {}; tail pending: {}",
        plan.RUN_SET_ID,
        marker,
        pending,
    )


def _tail_results(
    cfg: driver_io.CampaignDriverConfig, state: dict, fired: list[str]
) -> None:
    if "finalize_core" not in state["transitions"]:
        return
    train, _ = _all_docs()
    for b in plan.tail_blocks():
        key = f"results_{b.name}"
        if key in state["transitions"]:
            continue
        if not _queues_finished_quiet(cfg, list(plan.block_queue_names(b.name))):
            continue
        if not _block_complete(cfg, train, b, []):
            continue
        written = build_results(cfg, dry_run=cfg.dry_run)
        state["transitions"][key] = {
            "at": utc_now_iso(),
            "results": [str(p) for p in written.values()],
        }
        fired.append(key)


# ---- deadline ----------------------------------------------------------------


def _parse_utc(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


def _open_window(state: dict, opts: TopkDriverOptions) -> dict:
    windows = state.setdefault("deadline_windows", [])
    if windows and windows[-1].get("ended_at") is None:
        return windows[-1]
    window = {
        "start": utc_now_iso(),
        "hours": float(opts.deadline_hours),
        "ended_at": None,
        "reason": None,
    }
    windows.append(window)
    return window


def _elapsed_hours(state: dict) -> float | None:
    windows = state.get("deadline_windows") or []
    if not windows or windows[-1].get("ended_at") is not None:
        return None
    start = _parse_utc(str(windows[-1]["start"]))
    return (datetime.now(timezone.utc) - start) / timedelta(hours=1)


def _deadline_passed(state: dict, opts: TopkDriverOptions) -> bool:
    windows = state.get("deadline_windows") or []
    if not windows or windows[-1].get("ended_at") is not None:
        return False
    hours = float(windows[-1].get("hours", opts.deadline_hours))
    elapsed = _elapsed_hours(state)
    return elapsed is not None and elapsed > hours


def _tail_incomplete(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    if cfg.dry_run:
        return []
    return [
        b.name
        for b in plan.tail_blocks()
        if not _queues_finished_quiet(cfg, list(plan.block_queue_names(b.name)))
    ]


def _deadline(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: TopkDriverOptions,
    fired: list[str],
) -> None:
    if "deadline_pause" in state["transitions"] or not _deadline_passed(state, opts):
        return
    truncated = _tail_incomplete(cfg)
    if not truncated:
        return
    if not cfg.dry_run:
        control_io.pause(
            queue_root=cfg.queue_root,
            logs_dir=cfg.logs_dir,
            spec=cfg.spec,
            reason=f"deadline {opts.deadline_hours}h reached with tail incomplete: {truncated}",
        )
        build_results(cfg, dry_run=False)
    windows = state.get("deadline_windows") or []
    if windows:
        windows[-1]["ended_at"] = utc_now_iso()
        windows[-1]["reason"] = "deadline"
    state["transitions"]["deadline_pause"] = {
        "at": utc_now_iso(),
        "tail_truncated": truncated,
    }
    fired.append("deadline_pause")
    logger.warning("deadline reached — paused the pool; tail truncated: {}", truncated)


def projected_tail_hours(cfg: driver_io.CampaignDriverConfig) -> float:
    """Remaining tail train work in node-hours from measured per-cell medians."""
    if cfg.dry_run:
        return 0.0
    slot_minutes = 0.0
    for b in plan.tail_blocks():
        for size in b.sizes:
            fp = plan.STUDENT_FOOTPRINT[size]
            for name in (plan.q_students(b.name, size), plan.q_controls(b.name, size)):
                if not store.queue_exists(cfg.queue_root, name):
                    continue
                status = store.queue_status(cfg.queue_root, name)
                slot_minutes += (
                    (status["pending"] + status["running"]) * fp.est_minutes * fp.slots
                )
    return slot_minutes / plan.total_slots() / 60.0


# ---- run / status ------------------------------------------------------------


def run(
    cfg: driver_io.CampaignDriverConfig, options: TopkDriverOptions | None = None
) -> AdvanceResult:
    """One advance pass over the top-K DAG."""
    if not is_valid_run_set_id(cfg.run_set_id):
        raise CampaignGateError(f"run_set_id {cfg.run_set_id!r} is malformed")
    if cfg.spec is not plan.SPEC and cfg.spec.run_set_id != plan.RUN_SET_ID:
        raise CampaignGateError(
            f"topk driver bound to wrong spec run_set_id={cfg.spec.run_set_id!r}; "
            "pass spec=topk_plan.SPEC via driver_config()"
        )
    opts = options or TopkDriverOptions()
    pin_note = driver_io.ensure_campaign_code_pin(cfg)
    state = driver_io.load_state(cfg)
    state.setdefault("deadline_windows", [])
    transitions = state["transitions"]
    fired: list[str] = []
    waiting: list[str] = []
    if pin_note:
        fired.append(pin_note)

    if "preflight" not in transitions:
        _preflight(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("preflight")

    _requeue_submitted(cfg)

    if "preflight" in transitions and "submit_all" not in transitions:
        _submit_all(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("submit_all")

    if _core_ready(cfg, state, waiting):
        _finalize_core(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("finalize_core")

    _tail_results(cfg, state, fired)
    _deadline(cfg, state, opts, fired)
    driver_io.save_state(cfg, state)

    done = "finalize_core" in transitions and (
        "deadline_pause" in transitions
        or all(f"results_{b.name}" in transitions for b in plan.tail_blocks())
    )
    if "finalize_core" in transitions and not done:
        waiting.append(
            f"tail draining: {_tail_incomplete(cfg)} (~{projected_tail_hours(cfg):.1f} h left)"
        )
    return AdvanceResult(tuple(fired), tuple(waiting), done)


def status(
    cfg: driver_io.CampaignDriverConfig, options: TopkDriverOptions | None = None
) -> dict:
    """Read-only snapshot: transitions, queues, deadline, projected tail."""
    opts = options or TopkDriverOptions()
    state = driver_io.load_state(cfg)
    queues = {
        name: store.queue_status(cfg.queue_root, name)
        for name in cfg.queue_priorities
        if store.queue_exists(cfg.queue_root, name)
    }
    elapsed = _elapsed_hours(state)
    return {
        "transitions": sorted(state["transitions"]),
        "queues": queues,
        "deadline": {
            "hours": opts.deadline_hours,
            "elapsed_h": elapsed,
            "remaining_h": (opts.deadline_hours - elapsed)
            if elapsed is not None
            else None,
            "windows": state.get("deadline_windows") or [],
        },
        "projected_tail_hours": projected_tail_hours(cfg),
        "core_done": "finalize_core" in state["transitions"],
        "done": (cfg.logs_dir / plan.CAMPAIGN_DONE_FILE).is_file(),
    }
