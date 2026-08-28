"""Lean auto-advancing driver for the bear checkpoint-grid campaign.

One `run(cfg)` = one advance pass over:

    preflight     -> prefetch manifest (38 refs) + disk + datasets + prereg
                     + generated YAMLs
    submit_p1     -> SDF teachers + teacher preference evals
    teacher_gate  -> behavioural admission + hard original-base gate
    submit_p2_p3  -> grid ∥ joint students + controls (both unlock on TGATE)
    submit_p4     -> student preference evals
    finalize      -> results CSVs + claims.json + CAMPAIGN_DONE

`CampaignDriverConfig` MUST be built with `spec=ckptgrid_plan.SPEC`. Code pin
is `driver_io.ensure_campaign_code_pin` (workspace HEAD may move).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from loguru import logger

from pipelines.ckptgrid_results_pipeline import (
    assemble_teacher_frame,
    build_results,
)
from scripts.stage_ckptgrid_assets import (
    DEFAULT_COLD_ROOT,
    DEFAULT_SDF_EXPERIMENT,
    PREREG_PATH,
    dataset_checks,
    disk_gate_readings,
    sdf_docs_root_from_experiment,
)
from subliminal_learning.campaign import ckptgrid_config_gen as cgen
from subliminal_learning.campaign import ckptgrid_jobs as jobs
from subliminal_learning.campaign import ckptgrid_plan as plan
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign.ckptgrid_jobs import prefetch_refs
from subliminal_learning.campaign.ckptgrid_metrics import build_teacher_curve
from subliminal_learning.campaign.ckptgrid_selection import (
    gate_report,
    screen_admission,
    teacher_gate,
)
from subliminal_learning.jobqueue import store
from subliminal_learning.training.run_set import is_valid_run_set_id
from subliminal_learning.utils.io import atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso


class CampaignGateError(RuntimeError):
    """A phase gate failed hard; do not advance."""


@dataclass(frozen=True)
class AdvanceResult:
    """Outcome of one advance pass."""

    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


@dataclass(frozen=True)
class CkptgridDriverOptions:
    """Optional roots the preflight reads (overridable in tests)."""

    cold_root: Path = DEFAULT_COLD_ROOT
    sdf_experiment: Path = DEFAULT_SDF_EXPERIMENT
    prereg_path: Path = PREREG_PATH
    dataset_cache_root: Path | None = None
    hf_cache_root: Path | None = None


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the checkpoint-grid `CampaignSpec`."""
    return driver_io.CampaignDriverConfig(spec=plan.SPEC, **kwargs)


def _queues_ready(
    cfg: driver_io.CampaignDriverConfig, names: list[str], waiting: list[str]
) -> bool:
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


def _require_prefetch_manifest(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Load `prefetch_manifest.json` and demand every designed Hub ref.

    Raises:
        CampaignGateError: The file is missing or omits a designed ref.
    """
    path = cfg.logs_dir / plan.PREFETCH_MANIFEST
    if not path.is_file():
        raise CampaignGateError(
            f"P0 missing {plan.PREFETCH_MANIFEST} at {path} — run "
            "`python -m scripts.stage_ckptgrid_assets` first"
        )
    doc = json.loads(path.read_text())
    records = doc.get("refs") or []
    have = {(str(r.get("repo")), str(r.get("revision"))) for r in records}
    missing = [
        (ref.repo_id, ref.revision)
        for ref in prefetch_refs()
        if (ref.repo_id, ref.revision) not in have
    ]
    if missing:
        raise CampaignGateError(
            f"P0 {plan.PREFETCH_MANIFEST} missing {len(missing)} refs "
            f"(first: {missing[0]})"
        )
    return doc


def _preflight(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    options: CkptgridDriverOptions,
) -> None:
    """P0: manifest, disk, datasets, prereg, generated configs."""
    manifest = _require_prefetch_manifest(cfg)
    if not Path(options.prereg_path).is_file():
        raise CampaignGateError(f"P0 missing prereg {options.prereg_path}")
    try:
        sdf_root = sdf_docs_root_from_experiment(options.sdf_experiment)
        ds_root = options.dataset_cache_root or _dataset_cache_root(cfg)
        datasets = dataset_checks(
            SimpleNamespace(dataset_cache_root=ds_root),
            sdf_docs_root=sdf_root,
        )
    except (FileNotFoundError, OSError) as err:
        raise CampaignGateError(f"P0 dataset gate FAILED: {err}") from err
    hf_root = options.hf_cache_root or cfg.output_root
    try:
        disk = disk_gate_readings(
            cold_root=options.cold_root,
            ssd_roots=(cfg.output_root, hf_root),
        )
    except RuntimeError as err:
        raise CampaignGateError(f"P0 disk gate FAILED: {err}") from err
    train = cgen.all_train_configs()
    verify = cgen.all_verify_configs(train)
    driver_io.write_generated_configs(cfg, {**train, **verify})
    driver_io.write_logs_report(
        cfg,
        plan.PREFLIGHT_REPORT,
        {
            "at": utc_now_iso(),
            "n_refs": len(prefetch_refs()),
            "manifest": str(cfg.logs_dir / plan.PREFETCH_MANIFEST),
            "n_manifest_refs": len(manifest.get("refs") or []),
            "datasets": datasets,
            "disk_free_gib": disk,
            "prereg": str(options.prereg_path),
            "n_train_yamls": len(train),
            "n_verify_yamls": len(verify),
        },
    )
    state["transitions"]["preflight"] = {
        "at": utc_now_iso(),
        "train_yamls": sorted(train),
        "verify_yamls": sorted(verify),
    }


def _dataset_cache_root(cfg: driver_io.CampaignDriverConfig) -> Path:
    """``{output_root}/datasets`` — the paths YAML layout."""
    return Path(cfg.output_root) / "datasets"


def _submit_p1(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    train = cgen.all_train_configs()
    verify = cgen.all_verify_configs(train)
    driver_io.write_generated_configs(cfg, {**train, **verify})
    parts = jobs.split_train_configs(train)
    t_verify, _ = jobs.split_verify_configs(verify)
    t_jobs = jobs.teacher_jobs(parts.teachers)
    if not t_jobs:
        raise CampaignGateError("P1 teacher queue built zero jobs")
    driver_io.submit_queue(cfg, plan.Q_P1_TEACHERS, t_jobs)
    e_jobs = jobs.teacher_eval_jobs(t_verify, parts.teachers)
    if not e_jobs:
        raise CampaignGateError("P1 teacher-eval queue built zero jobs")
    driver_io.submit_queue(
        cfg,
        plan.Q_P1_TEACHER_EVALS,
        e_jobs,
        depends_on=[plan.Q_P1_TEACHERS],
    )
    state["transitions"]["submit_p1"] = {
        "at": utc_now_iso(),
        "n_teacher_jobs": len(t_jobs),
        "n_teacher_eval_jobs": len(e_jobs),
    }


def run_teacher_gate(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    *,
    teacher_df=None,
) -> None:
    """Screen admission, write the gate report, halt on a hard failure."""
    frame = (
        teacher_df
        if teacher_df is not None
        else assemble_teacher_frame(cfg.output_root)
    )
    dispositions = screen_admission(frame)
    curve = build_teacher_curve(frame, rungs=plan.TEACHER_RUNGS, base=plan.GRID_BASE)
    span = (
        float(curve["span"].iloc[0])
        if not curve.empty and "span" in curve.columns
        else None
    )
    result = teacher_gate(dispositions, span)
    driver_io.write_logs_report(
        cfg,
        plan.TEACHER_GATE_REPORT,
        {"at": utc_now_iso(), **gate_report(dispositions, result)},
    )
    state["transitions"]["teacher_gate"] = {
        "at": utc_now_iso(),
        "passed": result.passed,
        "ok_original_rungs": list(result.ok_original_rungs),
        "curve_span": result.curve_span,
        "reasons": list(result.reasons),
    }
    if not result.passed:
        raise CampaignGateError(
            "teacher gate FAILED — re-place rungs: " + "; ".join(result.reasons)
        )


def _submit_p2_p3(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    train = cgen.all_train_configs()
    parts = jobs.split_train_configs(train)
    grid_jobs = jobs.student_jobs(parts.grid_students, tag="p2")
    joint_jobs = jobs.student_jobs(parts.joint_students, tag="p3")
    grid_ctrl = jobs.control_jobs(parts.grid_controls, tag="p2")
    joint_ctrl = jobs.control_jobs(parts.joint_controls, tag="p3")
    if not grid_jobs or not joint_jobs or not grid_ctrl or not joint_ctrl:
        raise CampaignGateError(
            "P2/P3 built an empty queue "
            f"(grid={len(grid_jobs)} joint={len(joint_jobs)} "
            f"gctrl={len(grid_ctrl)} jctrl={len(joint_ctrl)})"
        )
    depends = [plan.Q_P1_TEACHER_EVALS]
    driver_io.submit_queue(cfg, plan.Q_P2_STUDENTS, grid_jobs, depends_on=depends)
    driver_io.submit_queue(cfg, plan.Q_P2_CONTROLS, grid_ctrl, depends_on=depends)
    driver_io.submit_queue(cfg, plan.Q_P3_STUDENTS, joint_jobs, depends_on=depends)
    driver_io.submit_queue(cfg, plan.Q_P3_CONTROLS, joint_ctrl, depends_on=depends)
    state["transitions"]["submit_p2_p3"] = {
        "at": utc_now_iso(),
        "n_grid_students": len(grid_jobs),
        "n_joint_students": len(joint_jobs),
        "n_grid_controls": len(grid_ctrl),
        "n_joint_controls": len(joint_ctrl),
        "grid_student_yamls": sorted(parts.grid_students),
        "joint_student_yamls": sorted(parts.joint_students),
        "grid_control_yamls": sorted(parts.grid_controls),
        "joint_control_yamls": sorted(parts.joint_controls),
    }


def _submit_p4(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    train = cgen.all_train_configs()
    verify = cgen.all_verify_configs(train)
    parts = jobs.split_train_configs(train)
    _, s_verify = jobs.split_verify_configs(verify)
    student_train = {
        **parts.grid_students,
        **parts.grid_controls,
        **parts.joint_students,
        **parts.joint_controls,
    }
    e_jobs = jobs.eval_jobs(s_verify, student_train, tag="p4")
    if not e_jobs:
        raise CampaignGateError("P4 eval queue built zero jobs")
    driver_io.submit_queue(
        cfg,
        plan.Q_P4_EVALS,
        e_jobs,
        depends_on=[
            plan.Q_P2_STUDENTS,
            plan.Q_P2_CONTROLS,
            plan.Q_P3_STUDENTS,
            plan.Q_P3_CONTROLS,
        ],
    )
    state["transitions"]["submit_p4"] = {
        "at": utc_now_iso(),
        "n_eval_jobs": len(e_jobs),
        "verify_yamls": sorted(s_verify),
    }


def _finalize(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    written = build_results(cfg.output_root, cfg.campaign_subdir, dry_run=cfg.dry_run)
    marker = cfg.logs_dir / plan.CAMPAIGN_DONE_FILE
    if not cfg.dry_run:
        atomic_write_text(
            marker, f"campaign {plan.RUN_SET_ID} finished {utc_now_iso()}\n"
        )
    state["transitions"]["finalize"] = {
        "at": utc_now_iso(),
        "results": [str(p) for p in written],
    }
    logger.info("Campaign {} COMPLETE — marker at {}", plan.RUN_SET_ID, marker)


def run(
    cfg: driver_io.CampaignDriverConfig,
    options: CkptgridDriverOptions | None = None,
) -> AdvanceResult:
    """One advance pass over the checkpoint-grid DAG."""
    # Ordered before the spec-identity check so a malformed id is diagnosed as
    # such rather than as the wrong campaign. Every sweep re-validates
    # `--run-set-id` at option-parse time, so a malformed id would submit
    # hundreds of jobs that each exit 2 -- burning their --auto-requeue attempts
    # -- without loading a model. Re-checked on every advance pass, not only the
    # first, because later passes submit too.
    if not is_valid_run_set_id(cfg.run_set_id):
        raise CampaignGateError(
            f"run_set_id {cfg.run_set_id!r} is malformed — every sweep would "
            "reject it at startup, so nothing is submitted. Expected "
            "{YYYYMMDD}_{purpose-slug}_{size} with hyphens (not underscores) "
            "inside the slug."
        )
    if cfg.spec is not plan.SPEC and cfg.spec.run_set_id != plan.RUN_SET_ID:
        raise CampaignGateError(
            f"ckptgrid driver bound to wrong spec run_set_id="
            f"{cfg.spec.run_set_id!r}; expected {plan.RUN_SET_ID!r}. Pass "
            "spec=ckptgrid_plan.SPEC via driver_config()."
        )
    opts = options or CkptgridDriverOptions()
    pin_note = driver_io.ensure_campaign_code_pin(cfg)
    state = driver_io.load_state(cfg)
    transitions = state["transitions"]
    fired: list[str] = []
    waiting: list[str] = []
    if pin_note:
        fired.append(pin_note)

    if "preflight" not in transitions:
        _preflight(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("preflight")

    if "preflight" in transitions and "submit_p1" not in transitions:
        _submit_p1(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("submit_p1")

    if "submit_p1" in transitions and "teacher_gate" not in transitions:
        if _queues_ready(cfg, [plan.Q_P1_TEACHERS, plan.Q_P1_TEACHER_EVALS], waiting):
            run_teacher_gate(cfg, state)
            driver_io.save_state(cfg, state)
            fired.append("teacher_gate")

    if "teacher_gate" in transitions and "submit_p2_p3" not in transitions:
        _submit_p2_p3(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("submit_p2_p3")

    if "submit_p2_p3" in transitions and "submit_p4" not in transitions:
        if _queues_ready(
            cfg,
            [
                plan.Q_P2_STUDENTS,
                plan.Q_P2_CONTROLS,
                plan.Q_P3_STUDENTS,
                plan.Q_P3_CONTROLS,
            ],
            waiting,
        ):
            _submit_p4(cfg, state)
            driver_io.save_state(cfg, state)
            fired.append("submit_p4")

    if "submit_p4" in transitions and "finalize" not in transitions:
        if _queues_ready(cfg, [plan.Q_P4_EVALS], waiting):
            _finalize(cfg, state)
            driver_io.save_state(cfg, state)
            fired.append("finalize")

    return AdvanceResult(
        tuple(fired), tuple(waiting), "finalize" in state["transitions"]
    )


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Read-only snapshot: per-queue status + which transitions have fired."""
    state = driver_io.load_state(cfg)
    queues = {
        name: store.queue_status(cfg.queue_root, name)
        for name in plan.QUEUE_PRIORITIES
        if store.queue_exists(cfg.queue_root, name)
    }
    return {"transitions": sorted(state["transitions"]), "queues": queues}
