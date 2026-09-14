"""Auto-advancing driver for the strictscreen campaign (soar-subliminal-3).

One `run(cfg, opts)` = one advance pass; the state is saved after every phase:

    preflight            -> home node + pin, staging/smoke reports, prereg sha, cold
                            link noop, disk, teachers / pool / vocab / recipe vs the
                            staged record, arm-C identity, host census, judge preflight
    submit_train         -> five train queues (B, A-tfs0..2, C) with frozen job keys
    submit_evals_{unit}  -> once the unit's train queue is terminal: one eval job per
                            config dir whose leaves are all done; holes recorded
    finalize_training    -> every queue terminal; screen audit of every done leaf
                            (contamination is a hard gate); coverage.json,
                            screen_stats.csv, shortfall_report.json; tolerance gate
    sl_results           -> `strictscreen_results_pipeline.build_results`
    campaign_done        -> `strictscreen_audit_driver.advance_audit` each pass until
                            `audit_done`, then ``CAMPAIGN_DONE``

Evals are gated by the driver, never by jobq ``depends_on``. The audit driver owns
every ``audit_*`` state key, the stop sentinel and ``GPUS_RELEASED``.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pandas as pd
from loguru import logger

from pipelines.strictscreen_results_pipeline import (
    ROLLOUT_SUMMARY,
    TRAIN_MANIFEST,
    build_results,
    read_json,
)
from scripts import stage_strictscreen_assets as stage
from scripts.stage_ckptgrid_assets import (
    DEFAULT_COLD_ROOT,
    DEFAULT_MODEL,
    DEFAULT_PATHS,
    disk_gate_readings,
)
from subliminal_learning.campaign import driver_io, expected
from subliminal_learning.campaign import strictscreen_config_gen as cgen
from subliminal_learning.campaign import strictscreen_jobs as jobs
from subliminal_learning.campaign import strictscreen_metrics as metrics
from subliminal_learning.campaign import strictscreen_plan as plan
from subliminal_learning.campaign.ckptgrid_jobs import symlink_action
from subliminal_learning.campaign.strictscreen_plan import CampaignGateError
from subliminal_learning.campaign.topk_jobs import cold_symlink_plan
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.training.run_set import is_valid_run_set_id
from subliminal_learning.utils.io import atomic_write_text
from subliminal_learning.utils.run_io import cell_is_done, utc_now_iso

AUDIT_DRIVER_MODULE = "pipelines.strictscreen_audit_driver"
DONE_TRANSITION = "campaign_done"
_GATE_ERRORS = (RuntimeError, ValueError, FileNotFoundError, FileExistsError)


@dataclass(frozen=True)
class AdvanceResult:
    """Outcome of one advance pass."""

    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


@dataclass(frozen=True)
class StrictscreenDriverOptions:
    """Roots and configs the preflight reads (overridable in tests).

    Attributes:
        cold_root: Cold-1 artifact root.
        prereg_path: Pre-registration that must match the staged sha.
        paths_config: Paths YAML (arm-C identity recompute).
        model_config: Model registry YAML.
        dataset_cache_root: Root holding the prompt pool (required on a real pass).
        pin_root: Code pin the driver runs from; ``None`` = cwd.
        enforce_host: Refuse to run off `plan.NODE`.
    """

    cold_root: Path = DEFAULT_COLD_ROOT
    prereg_path: Path = Path(plan.PREREG_PATH)
    paths_config: Path = DEFAULT_PATHS
    model_config: Path = DEFAULT_MODEL
    dataset_cache_root: Path | None = None
    pin_root: Path | None = None
    enforce_host: bool = True

    @property
    def pin(self) -> Path:
        """Resolved code-pin root."""
        return (self.pin_root or Path.cwd()).resolve()


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to `strictscreen_plan.SPEC`."""
    return driver_io.CampaignDriverConfig(spec=plan.SPEC, **kwargs)


# ---- audit driver ------------------------------------------------------------------------


def audit_api(*, dry_run: bool) -> ModuleType | None:
    """The judge-audit driver module, or ``None`` on a dry run before it exists.

    Raises:
        CampaignGateError: The module is not importable on a real pass.
    """
    try:
        return importlib.import_module(AUDIT_DRIVER_MODULE)
    except ModuleNotFoundError as err:
        if err.name != AUDIT_DRIVER_MODULE:
            raise
        if dry_run:
            logger.warning("[dry-run] {} not importable; audit calls skipped", err.name)
            return None
        raise CampaignGateError(f"{AUDIT_DRIVER_MODULE} is not importable") from err


def audit_context(
    api: ModuleType,
    cfg: driver_io.CampaignDriverConfig,
    opts: StrictscreenDriverOptions,
) -> object:
    """The audit driver's `AuditContext` for this campaign."""
    return api.AuditContext(
        pin_root=opts.pin,
        host=store.this_host(),
        queue_root=cfg.queue_root,
        output_root=cfg.output_root,
        run_set_logs_dir=cfg.logs_dir,
        campaign_queue_names=tuple(plan.QUEUE_PRIORITIES),
        dry_run=cfg.dry_run,
    )


# ---- queues ------------------------------------------------------------------------------


def _requeue(cfg: driver_io.CampaignDriverConfig, name: str) -> int:
    """Requeue failed jobs while any failure is still below ``cfg.auto_requeue``.

    When capped and uncapped failures coexist, jobq can only requeue both, so a
    capped job may get one extra attempt; the number of such passes is bounded by
    the uncapped failures.
    """
    if (
        cfg.dry_run
        or cfg.auto_requeue <= 0
        or not store.queue_exists(cfg.queue_root, name)
    ):
        return 0
    attempts = [
        store.read_attempt(cfg.queue_root, name, job.jobkey)
        for job in store.load_jobs(cfg.queue_root, name)
        if store.job_state(cfg.queue_root, name, job) == store.FAILED
    ]
    if not any(a < cfg.auto_requeue for a in attempts):
        return 0
    if all(a < cfg.auto_requeue for a in attempts):
        return driver_io.requeue_failed_under_cap(cfg, name)
    logger.warning("{}: requeueing failures including jobs already at the cap", name)
    return len(store.requeue(cfg.queue_root, name, failed_only=True))


def _queue_terminal(
    cfg: driver_io.CampaignDriverConfig, name: str, waiting: list[str]
) -> bool:
    """Whether every job of `name` has a result after requeueing under the cap."""
    if cfg.dry_run:
        return True
    if not store.queue_exists(cfg.queue_root, name):
        waiting.append(f"{name}: not submitted")
        return False
    _requeue(cfg, name)
    if store.queue_complete(cfg.queue_root, name):
        return True
    st = store.queue_status(cfg.queue_root, name)
    waiting.append(
        f"{name}: {st['pending']} pending / {st['running']} running / "
        f"{st['failed']} failed of {st['total']}"
    )
    return False


def _submit(cfg: driver_io.CampaignDriverConfig, name: str, js: list[Job]) -> None:
    if cfg.dry_run:
        logger.info(
            "[dry-run] would submit queue {} (prio {}, node {}, {} jobs)",
            name,
            cfg.queue_priorities[name],
            plan.NODE,
            len(js),
        )
        return
    driver_io.submit_queue(cfg, name, js)


# ---- preflight ------------------------------------------------------------------------------


def _require_report(path: Path) -> dict:
    doc = read_json(path)
    if doc is None:
        raise CampaignGateError(
            f"preflight missing {path.name} at {path} — run "
            "`python -m scripts.stage_strictscreen_assets` (and --smoke-verify) first"
        )
    if doc.get("passed") is not True:
        raise CampaignGateError(f"preflight {path.name} did not pass")
    return doc


def _check_equal(name: str, current: object, staged: object) -> None:
    if current != staged:
        raise CampaignGateError(f"preflight {name} differs from the staging report")


def _preflight_gates(
    cfg: driver_io.CampaignDriverConfig, opts: StrictscreenDriverOptions
) -> dict:
    host = store.this_host()
    if opts.enforce_host and host != plan.NODE:
        raise CampaignGateError(
            f"{plan.CAMPAIGN} must run on {plan.NODE} ({plan.NODE_ALIAS}); this host is {host}"
        )
    pin_cwd = Path(driver_io.queue_job_defaults(cfg)["cwd"]).resolve()
    if pin_cwd != opts.pin:
        raise CampaignGateError(
            f"driver runs from {opts.pin}, campaign pin is {pin_cwd}"
        )
    staging = _require_report(cfg.logs_dir / plan.STAGING_REPORT)
    smoke = _require_report(cfg.logs_dir / plan.SMOKE_REPORT)
    if opts.dataset_cache_root is None:
        raise CampaignGateError("preflight needs dataset_cache_root")
    try:
        _check_equal(
            "prereg",
            stage.prereg_record(opts.prereg_path)["sha256"],
            staging["prereg"]["sha256"],
        )
        for link, target in cold_symlink_plan(
            cfg.output_root,
            opts.cold_root,
            run_set_id=plan.RUN_SET_ID,
            subdirs=(GKD_STUDENTS_SUBDIR,),
        ):
            action = symlink_action(link, target)
            if action != "noop":
                raise CampaignGateError(f"cold symlink {link} → {target} is {action!r}")
        disk = disk_gate_readings(
            cold_root=opts.cold_root,
            ssd_roots=(opts.pin,),
            min_cold_gib=plan.MIN_DISK_FREE_GIB_COLD,
            min_ssd_gib=plan.MIN_DISK_FREE_GIB_PIN,
        )
        _check_equal(
            "teachers", stage.teacher_records(cfg.output_root), staging["teachers"]
        )
        _check_equal(
            "pool", stage.pool_record(opts.dataset_cache_root), staging["pool"]
        )
        _check_equal("screen vocab", stage.vocab_records(), staging["screen"])
        recipe = stage.recipe_record(cfg.output_root)
        identity = stage.arm_c_identity_records(
            opts.paths_config, opts.model_config, cfg.output_root
        )
        census = stage.queue_census(cfg.queue_root, plan.NODE)
    except CampaignGateError:
        raise
    except _GATE_ERRORS as err:
        raise CampaignGateError(f"preflight FAILED: {err}") from err
    return {
        "host": host,
        "pin_root": str(opts.pin),
        "staging_at": staging.get("at"),
        "smoke_at": smoke.get("at"),
        "disk_free_gib": disk,
        "recipe": recipe,
        "n_arm_c_identities_equal": len(identity),
        "census": census,
    }


def _preflight(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: StrictscreenDriverOptions
) -> None:
    train = cgen.train_configs()
    verify = cgen.verify_configs(train)
    n_cells = sum(len(jobs.train_jobs(u)) for u in plan.UNITS)
    if n_cells != sum(plan.PLANNED_CELLS.values()):
        raise CampaignGateError(f"{n_cells} train jobs != planned cells")
    record: dict = {
        "at": utc_now_iso(),
        "n_train_yamls": len(train),
        "n_verify_yamls": len(verify),
        "n_train_cells": n_cells,
        "n_eval_units": sum(len(jobs.eval_units(u)) for u in plan.UNITS),
        "projected_hours": plan.projected_hours(),
    }
    if cfg.dry_run:
        logger.info("[dry-run] preflight gates skipped: {}", record)
    else:
        record.update(_preflight_gates(cfg, opts))
    driver_io.write_generated_configs(cfg, {**train, **verify})
    api = audit_api(dry_run=cfg.dry_run)
    if api is not None:
        try:
            record["judge_preflight"] = api.audit_preflight(
                audit_context(api, cfg, opts)
            )
        except CampaignGateError as err:
            if not cfg.dry_run:
                raise
            logger.warning("[dry-run] judge preflight would refuse: {}", err)
            record["judge_preflight_error"] = str(err)
    driver_io.write_logs_report(cfg, plan.PREFLIGHT_REPORT, {**record, "passed": True})
    state["transitions"]["preflight"] = record


# ---- submission ------------------------------------------------------------------------------


def _submit_train(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    counts = {}
    for unit in plan.UNITS:
        js = jobs.train_jobs(unit)
        _submit(cfg, unit.train_queue, js)
        counts[unit.train_queue] = len(js)
    state["transitions"]["submit_train"] = {"at": utc_now_iso(), "queues": counts}


def _submit_evals(
    cfg: driver_io.CampaignDriverConfig,
    unit: plan.Unit,
    state: dict,
    leaves: dict[str, Path],
) -> None:
    js: list[Job] = []
    holes: list[dict] = []
    for eu in jobs.eval_units(unit):
        missing = [c.key for c in eu.cells if not cell_is_done(leaves[c.key])]
        if cfg.dry_run or not missing:
            js.append(jobs.eval_job(eu))
            continue
        holes.append(
            {
                "eval_unit": eu.key,
                "config_dir": str(leaves[eu.cells[0].key].parent),
                "reason": "leaves not done",
                "missing_cells": missing,
            }
        )
    if js:
        _submit(cfg, unit.eval_queue, js)
    for hole in holes:
        logger.warning("eval hole {}: {}", hole["eval_unit"], hole["missing_cells"])
    state["transitions"][f"submit_evals_{unit.name}"] = {
        "at": utc_now_iso(),
        "queue": unit.eval_queue if js else None,
        "n_jobs": len(js),
        "holes": holes,
    }


# ---- finalize training ----------------------------------------------------------------------


def _training_terminal(
    cfg: driver_io.CampaignDriverConfig, state: dict, waiting: list[str]
) -> bool:
    transitions = state["transitions"]
    if "submit_train" not in transitions:
        return False
    ready = True
    for unit in plan.UNITS:
        if not _queue_terminal(cfg, unit.train_queue, waiting):
            ready = False
        evals = transitions.get(f"submit_evals_{unit.name}")
        if evals is None:
            waiting.append(f"{unit.eval_queue}: not submitted")
            ready = False
        elif evals["queue"] and not _queue_terminal(cfg, evals["queue"], waiting):
            ready = False
    return ready


def _backfill(
    cfg: driver_io.CampaignDriverConfig,
    coverage: dict[str, dict],
    opts: StrictscreenDriverOptions,
) -> dict:
    cells = {c.key: c for c in plan.all_cells()}
    eval_of = {
        c.key: eu for u in plan.UNITS for eu in jobs.eval_units(u) for c in eu.cells
    }
    train_cmds, eval_cmds = {}, {}
    for arm in coverage.values():
        for miss in arm["missing"]:
            key = miss["cell"]
            if miss["reason"] == "not_trained":
                train_cmds[key] = jobs.train_job(cells[key]).cmd
            eu = eval_of[key]
            eval_cmds[eu.key] = jobs.eval_job(eu).cmd
    return {
        "cwd": str(opts.pin),
        "requeue": [
            f"uv run python -m scripts.jobq requeue --root {cfg.queue_root} --name {q}"
            for q in plan.QUEUE_PRIORITIES
            if store.queue_exists(cfg.queue_root, q)
            and store.queue_status(cfg.queue_root, q)["failed"]
        ],
        "train": train_cmds,
        "eval": eval_cmds,
    }


def _finalize_training(
    cfg: driver_io.CampaignDriverConfig, state: dict, opts: StrictscreenDriverOptions
) -> None:
    leaves = jobs.cell_leaf_dirs(cfg.output_root)
    cells = plan.all_cells()
    if cfg.dry_run:
        trained = {c.key: True for c in cells}
        logger.info(
            "[dry-run] finalize_training: would screen-audit {} leaves", len(leaves)
        )
    else:
        trained = {c.key: cell_is_done(leaves[c.key]) for c in cells}
        staged = _require_report(cfg.logs_dir / plan.STAGING_REPORT)["screen"][
            "vocab_sha256"
        ]
        _screen_audit(cfg, cells, leaves, trained, staged)
    config_dirs = sorted({leaves[c.key].parent for c in cells if trained[c.key]})
    incomplete = set(
        expected.verify_summaries_complete(
            config_dirs, required_prompt_sets=plan.EVAL_PROMPT_SETS
        )
    )
    evaluated = {
        c.key: cfg.dry_run or leaves[c.key].parent not in incomplete for c in cells
    }
    coverage = metrics.coverage_by_arm(cells, trained, evaluated)
    shortfall = {
        "at": utc_now_iso(),
        "within_tolerance": all(a["within_tolerance"] for a in coverage.values()),
        "arms": {
            arm: {k: v for k, v in doc.items() if k != "missing"}
            for arm, doc in coverage.items()
        },
        "backfill": _backfill(cfg, coverage, opts),
    }
    driver_io.write_logs_report(
        cfg, plan.COVERAGE_REPORT, {"at": utc_now_iso(), **coverage}
    )
    driver_io.write_logs_report(cfg, plan.SHORTFALL_REPORT, shortfall)
    if not shortfall["within_tolerance"]:
        over = {
            a: d["n_missing"] for a, d in coverage.items() if not d["within_tolerance"]
        }
        raise CampaignGateError(
            f"shortfall beyond tolerance {over}; backfill commands in "
            f"{cfg.logs_dir / plan.SHORTFALL_REPORT}"
        )
    state["transitions"]["finalize_training"] = {
        "at": utc_now_iso(),
        "coverage": shortfall["arms"],
        "required_run_dirs": [
            str(leaves[c.key]) for c in cells if c.arm != plan.ARM_C and trained[c.key]
        ],
        "optional_run_dirs": [
            str(leaves[c.key]) for c in cells if c.arm == plan.ARM_C and trained[c.key]
        ],
    }


def _screen_audit(
    cfg: driver_io.CampaignDriverConfig,
    cells: tuple[plan.Cell, ...],
    leaves: dict[str, Path],
    trained: dict[str, bool],
    staged: dict[str, str],
) -> None:
    """Hard contamination gate over every done leaf; writes ``screen_stats.csv``.

    Raises:
        CampaignGateError: Any strict / legacy screen field disagrees with the plan.
    """
    rows, errors = [], []
    for cell in cells:
        if not trained[cell.key]:
            continue
        leaf = leaves[cell.key]
        summary = read_json(leaf / ROLLOUT_SUMMARY)
        found = metrics.screen_audit_errors(
            screen=cell.screen,
            expected_gkd_tag=cell.gkd_tag,
            manifest=read_json(leaf / TRAIN_MANIFEST) or {},
            summary=summary,
            staged_vocab_sha256=(
                staged.get(cell.animal) if cell.screen == plan.SCREEN_STRICT else None
            ),
        )
        errors.extend(f"{cell.key}: {e}" for e in found)
        rows.append(
            {
                "cell": cell.key,
                "arm": cell.arm,
                "screen": cell.screen,
                "animal": cell.animal,
                "pair": cell.pair,
                "tfs": metrics.CONTROL_TFS if cell.tfs is None else cell.tfs,
                "gseed": cell.gseed,
                "leaf_dir": str(leaf),
                **metrics.screen_stats_row(summary),
            }
        )
    if errors:
        raise CampaignGateError(
            f"screen audit FAILED ({len(errors)} findings, contamination): {errors[:5]}"
        )
    atomic_write_text(
        cfg.logs_dir / plan.SCREEN_STATS_CSV, pd.DataFrame(rows).to_csv(index=False)
    )


# ---- results + audit ----------------------------------------------------------------------------


def _sl_results(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    if cfg.dry_run:
        logger.info(
            "[dry-run] would build SL results under {}",
            jobs.results_dir(cfg.output_root),
        )
        state["transitions"]["sl_results"] = {"at": utc_now_iso(), "dry_run": True}
        return
    written = build_results(cfg.output_root)
    claims = json.loads(written[plan.RESULTS_CLAIMS_JSON].read_text())
    state["transitions"]["sl_results"] = {
        "at": utc_now_iso(),
        "results": [str(p) for p in written.values()],
        "claims": {k: claims[k]["status"] for k in ("C0", "H2", "H3")},
    }


def _audit(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    opts: StrictscreenDriverOptions,
    fired: list[str],
    waiting: list[str],
) -> None:
    transitions = state["transitions"]
    if "sl_results" not in transitions or DONE_TRANSITION in transitions:
        return
    api = audit_api(dry_run=cfg.dry_run)
    if api is None:
        waiting.append(f"{AUDIT_DRIVER_MODULE} not importable (dry-run)")
        return
    finalize = transitions["finalize_training"]
    try:
        api.advance_audit(
            audit_context(api, cfg, opts),
            state,
            fired,
            waiting,
            required_run_dirs=[Path(p) for p in finalize["required_run_dirs"]],
            optional_run_dirs=[Path(p) for p in finalize["optional_run_dirs"]],
        )
    except CampaignGateError as err:
        if not cfg.dry_run:
            driver_io.save_state(cfg, state)
            raise
        waiting.append(f"[dry-run] audit would refuse: {err}")
        return
    driver_io.save_state(cfg, state)
    if not api.audit_done(state):
        return
    marker = cfg.logs_dir / plan.CAMPAIGN_DONE_FILE
    if not cfg.dry_run and not marker.exists():
        atomic_write_text(
            marker, f"campaign {plan.RUN_SET_ID} finished {utc_now_iso()}\n"
        )
    transitions[DONE_TRANSITION] = {"at": utc_now_iso(), "marker": str(marker)}
    fired.append(DONE_TRANSITION)
    logger.info("{} COMPLETE — marker at {}", plan.CAMPAIGN, marker)


# ---- run / status --------------------------------------------------------------------------------


def run(
    cfg: driver_io.CampaignDriverConfig,
    options: StrictscreenDriverOptions | None = None,
) -> AdvanceResult:
    """One advance pass over the phase machine.

    Raises:
        CampaignGateError: Any pre-registered gate refuses to pass.
    """
    if not is_valid_run_set_id(cfg.run_set_id):
        raise CampaignGateError(f"run_set_id {cfg.run_set_id!r} is malformed")
    if cfg.spec is not plan.SPEC:
        raise CampaignGateError(
            f"driver bound to spec {cfg.spec.run_set_id!r}; use driver_config(...)"
        )
    opts = options or StrictscreenDriverOptions()
    fired: list[str] = []
    waiting: list[str] = []
    if cfg.dry_run:
        logger.info("[dry-run] code pin not resolved (no worktree writes)")
    else:
        try:
            note = driver_io.ensure_campaign_code_pin(cfg)
        except (ValueError, RuntimeError) as err:
            raise CampaignGateError(str(err)) from err
        if note:
            fired.append(note)
    state = driver_io.load_state(cfg)
    transitions = state["transitions"]

    if "preflight" not in transitions:
        _preflight(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("preflight")

    if "submit_train" not in transitions:
        _submit_train(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("submit_train")

    leaves: dict[str, Path] | None = None
    for unit in plan.UNITS:
        key = f"submit_evals_{unit.name}"
        if key in transitions or not _queue_terminal(cfg, unit.train_queue, waiting):
            continue
        leaves = leaves or jobs.cell_leaf_dirs(cfg.output_root)
        _submit_evals(cfg, unit, state, leaves)
        driver_io.save_state(cfg, state)
        fired.append(key)

    if "finalize_training" not in transitions and _training_terminal(
        cfg, state, waiting
    ):
        _finalize_training(cfg, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("finalize_training")

    if "finalize_training" in transitions and "sl_results" not in transitions:
        _sl_results(cfg, state)
        driver_io.save_state(cfg, state)
        fired.append("sl_results")

    _audit(cfg, state, opts, fired, waiting)
    driver_io.save_state(cfg, state)
    return AdvanceResult(
        tuple(fired), tuple(dict.fromkeys(waiting)), DONE_TRANSITION in transitions
    )


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Read-only snapshot: transitions, queue counts, SL claim statuses."""
    state = driver_io.load_state(cfg)
    transitions = state["transitions"]
    queues = {
        name: {
            k: v
            for k, v in store.queue_status(cfg.queue_root, name).items()
            if k != "running_jobs"
        }
        for name in plan.QUEUE_PRIORITIES
        if store.queue_exists(cfg.queue_root, name)
    }
    return {
        "run_set_id": plan.RUN_SET_ID,
        "home_node": plan.NODE,
        "transitions": sorted(transitions),
        "queues": queues,
        "eval_holes": {
            unit.name: len(
                (transitions.get(f"submit_evals_{unit.name}") or {}).get("holes", [])
            )
            for unit in plan.UNITS
        },
        "sl_claims": (transitions.get("sl_results") or {}).get("claims"),
        "done": DONE_TRANSITION in transitions,
    }
