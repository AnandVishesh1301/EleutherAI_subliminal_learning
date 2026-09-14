"""Auto-advancing driver for the OPD-limits campaign (one tier per node).

One `run(cfg, tier)` = one advance pass over:

    preflight      -> staging reports (prefetch, teacher assembly, inherited gates,
                      subset audit), home-node guard, cold links, disk, YAMLs
    submit_all     -> every block's student / control / eval / bench queues,
                      priority-ordered, submitted once with frozen jobkey sets
    finalize_core  -> results + claims + CAMPAIGN_DONE once every core block's
                      leaves and eval summaries (incl. calibration) are complete
    results_{tail} -> idempotent results rebuild as each tail block / bench lands
    cross_tier     -> Tier 2 only: Tier 1 CAMPAIGN_DONE present → pin comparison
                      (``ok`` / ``void_pin``) → results rebuild

There is no deadline and no pause: every block runs to completion. `run` must
be given `driver_config(tier, ...)`, which binds the tier's `CampaignSpec`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from pipelines.opdlim_results_pipeline import build_results
from scripts.stage_ckptgrid_assets import DEFAULT_COLD_ROOT, disk_gate_readings
from subliminal_learning.campaign import driver_io, expected
from subliminal_learning.campaign import opdlim_config_gen as cgen
from subliminal_learning.campaign import opdlim_jobs as jobs
from subliminal_learning.campaign import opdlim_plan as plan
from subliminal_learning.campaign.ckptgrid_jobs import symlink_action
from subliminal_learning.campaign.topk_jobs import cold_symlink_plan
from subliminal_learning.jobqueue import store
from subliminal_learning.training.run_set import is_valid_run_set_id, run_set_subdir
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import read_run_set_card, utc_now_iso


class CampaignGateError(RuntimeError):
    """A pre-registered gate refused to pass."""


@dataclass(frozen=True)
class AdvanceResult:
    """Outcome of one advance pass."""

    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


@dataclass(frozen=True)
class OpdlimDriverOptions:
    """Roots the preflight reads (overridable in tests).

    Attributes:
        cold_root: Cold-1 artifact root (real dirs live here).
        prereg_path: Pre-registration document that must exist.
        hf_cache_root: Hub cache root for the disk gate.
        enforce_host: Refuse to run off the tier's home node.
    """

    cold_root: Path = DEFAULT_COLD_ROOT
    prereg_path: Path = Path(plan.PREREG_PATH)
    hf_cache_root: Path | None = None
    enforce_host: bool = True


def driver_config(t: plan.Tier, **kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to tier `t`'s `CampaignSpec`."""
    return driver_io.CampaignDriverConfig(spec=plan.SPECS[t.n], **kwargs)


def _all_docs(t: plan.Tier) -> tuple[dict[str, dict], dict[str, dict]]:
    train = cgen.all_train_configs(t)
    return train, cgen.all_verify_configs(t, train)


# ---- queue helpers -------------------------------------------------------------------


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


# ---- preflight ------------------------------------------------------------------------


def _require_report(path: Path, *, key: str = "passed") -> dict:
    if not path.is_file():
        raise CampaignGateError(
            f"preflight missing {path.name} at {path} — run "
            "`python -m scripts.stage_opdlim_assets --tier N` first"
        )
    doc = json.loads(path.read_text())
    if doc.get(key) is not True:
        raise CampaignGateError(f"preflight {path.name} did not pass")
    return doc


def _require_prefetch(cfg: driver_io.CampaignDriverConfig) -> None:
    doc = _require_report(cfg.logs_dir / plan.PREFETCH_MANIFEST, key="complete")
    have = {
        (str(r.get("repo") or ""), str(r.get("revision") or ""))
        for r in doc.get("refs") or []
    }
    missing = [
        (r.repo_id, r.revision)
        for r in jobs.prefetch_refs()
        if (r.repo_id, r.revision) not in have
    ]
    if missing:
        raise CampaignGateError(f"prefetch manifest missing refs {missing}")


def _require_assembly(
    cfg: driver_io.CampaignDriverConfig, t: plan.Tier
) -> dict[str, str]:
    doc = _require_report(cfg.logs_dir / plan.ASSEMBLY_REPORT)
    recorded = {row["link"]: row for row in doc.get("links") or []}
    actions: dict[str, str] = {}
    for link, source in jobs.assembly_links(t, cfg.output_root):
        row = recorded.get(str(link))
        if row is None:
            raise CampaignGateError(f"teacher assembly report lacks {link}")
        try:
            action = symlink_action(link, source)
        except FileExistsError as err:
            raise CampaignGateError(f"teacher assembly FAILED: {err}") from err
        if action != "noop":
            raise CampaignGateError(f"teacher assembly link {link} is {action!r}")
        actions[str(link)] = action
    return actions


def _require_cold_symlinks(
    cfg: driver_io.CampaignDriverConfig, t: plan.Tier, cold_root: Path
) -> None:
    for link, target in cold_symlink_plan(
        cfg.output_root, cold_root, run_set_id=t.run_set_id
    ):
        try:
            action = symlink_action(link, target)
        except FileExistsError as err:
            raise CampaignGateError(f"cold symlink FAILED: {err}") from err
        if action != "noop":
            raise CampaignGateError(
                f"cold symlink {link} → {target} is {action!r}, expected noop"
            )


def _require_home_node(t: plan.Tier, opts: OpdlimDriverOptions) -> None:
    host = store.this_host()
    if opts.enforce_host and host != t.node:
        raise CampaignGateError(
            f"{t.token} must run on {t.node} ({t.alias}); this host is {host}"
        )


def _preflight(
    cfg: driver_io.CampaignDriverConfig,
    t: plan.Tier,
    state: dict,
    opts: OpdlimDriverOptions,
) -> None:
    train, verify = _all_docs(t)
    if cfg.dry_run:
        logger.info(
            "[dry-run] {} preflight: {} train YAMLs, {} verify/bench YAMLs",
            t.token,
            len(train),
            len(verify),
        )
    else:
        _require_home_node(t, opts)
        _require_report(cfg.logs_dir / plan.STAGING_REPORT)
        _require_prefetch(cfg)
        _require_assembly(cfg, t)
        if any(o.subset_n for b in t.blocks for o in b.objectives):
            _require_report(cfg.logs_dir / plan.SUBSET_AUDIT_REPORT)
        if not Path(opts.prereg_path).is_file():
            raise CampaignGateError(f"preflight missing prereg {opts.prereg_path}")
        _require_cold_symlinks(cfg, t, opts.cold_root)
        try:
            disk_gate_readings(
                cold_root=opts.cold_root,
                ssd_roots=(cfg.output_root, opts.hf_cache_root or cfg.output_root),
                min_cold_gib=plan.MIN_DISK_FREE_GIB_COLD,
                min_ssd_gib=plan.MIN_DISK_FREE_GIB_SSD,
            )
        except RuntimeError as err:
            raise CampaignGateError(f"disk gate FAILED: {err}") from err
    n_train_jobs = sum(
        len(jobs.train_jobs(t, train, b.name, size=s, controls=c))
        for b in t.blocks
        for s in b.sizes
        for c in (False, True)
    )
    if n_train_jobs != plan.tier_cells(t):
        raise CampaignGateError(
            f"{t.token}: {n_train_jobs} train jobs != planned {plan.tier_cells(t)} cells"
        )
    driver_io.write_generated_configs(cfg, {**train, **verify})
    driver_io.write_logs_report(
        cfg,
        plan.PREFLIGHT_REPORT,
        {
            "at": utc_now_iso(),
            "passed": True,
            "tier": t.n,
            "host": store.this_host(),
            "n_train_yamls": len(train),
            "n_verify_yamls": len(verify),
            "n_train_cells": n_train_jobs,
            "projected_hours": plan.tier_hours(t),
        },
    )
    state["transitions"]["preflight"] = {
        "at": utc_now_iso(),
        "train_yamls": sorted(train),
        "verify_yamls": sorted(verify),
    }


# ---- submission ------------------------------------------------------------------------


def _submit_block(
    cfg: driver_io.CampaignDriverConfig,
    t: plan.Tier,
    train: dict[str, dict],
    verify: dict[str, dict],
    b: plan.Block,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    classes = plan.block_queue_classes(b)
    for size in b.sizes:
        q = {cls: plan.queue_name(t, b.name, cls, size) for cls in classes}
        s_jobs = jobs.train_jobs(t, train, b.name, size=size, controls=False)
        if not s_jobs:
            raise CampaignGateError(f"{b.name}/{size}: empty student queue")
        driver_io.submit_queue(cfg, q["students"], s_jobs)
        train_deps = [q["students"]]
        if "controls" in q:
            c_jobs = jobs.train_jobs(t, train, b.name, size=size, controls=True)
            if not c_jobs:
                raise CampaignGateError(f"{b.name}/{size}: empty control queue")
            driver_io.submit_queue(cfg, q["controls"], c_jobs)
            train_deps.append(q["controls"])
            counts[f"{size}_controls"] = len(c_jobs)
        e_jobs = jobs.eval_jobs(
            t, verify, train, b.name, size=size, kind=cgen.VERIFY_KIND
        )
        driver_io.submit_queue(cfg, q["evals"], e_jobs, depends_on=train_deps)
        counts[f"{size}_students"] = len(s_jobs)
        counts[f"{size}_evals"] = len(e_jobs)
        if "bench" in q:
            bench = jobs.eval_jobs(
                t, verify, train, b.name, size=size, kind=cgen.BENCH_KIND
            )
            driver_io.submit_queue(cfg, q["bench"], bench, depends_on=[q["evals"]])
            counts[f"{size}_bench"] = len(bench)
    return counts


def _submit_all(cfg: driver_io.CampaignDriverConfig, t: plan.Tier, state: dict) -> None:
    train, verify = _all_docs(t)
    driver_io.write_generated_configs(cfg, {**train, **verify})
    per_block = {b.name: _submit_block(cfg, t, train, verify, b) for b in t.blocks}
    state["transitions"]["submit_all"] = {"at": utc_now_iso(), "blocks": per_block}


# ---- completeness ------------------------------------------------------------------------


def _config_dirs(leaf_dirs: list[Path]) -> list[Path]:
    return list(dict.fromkeys(d.parent for d in leaf_dirs))


def _block_complete(
    cfg: driver_io.CampaignDriverConfig,
    t: plan.Tier,
    train: dict[str, dict],
    b: plan.Block,
    waiting: list[str],
    *,
    bench: bool = False,
) -> bool:
    if cfg.dry_run:
        return True
    leaves = jobs.block_leaf_dirs(t, train, cfg.output_root, b.name)
    if expected.missing_manifests(leaves):
        waiting.append(f"{b.name}: leaves incomplete")
        return False
    incomplete = expected.verify_summaries_complete(
        _config_dirs(leaves),
        require_benchmarks=bench,
        required_prompt_sets=plan.EVAL_PROMPT_SETS,
        require_calibration=plan.CALIBRATION_IN_CORE_EVALS,
    )
    if incomplete:
        waiting.append(f"{b.name}: {len(incomplete)} config dirs lack summaries")
        return False
    return True


def _core_ready(
    cfg: driver_io.CampaignDriverConfig, t: plan.Tier, state: dict, waiting: list[str]
) -> bool:
    transitions = state["transitions"]
    if "submit_all" not in transitions or "finalize_core" in transitions:
        return False
    if not _queues_ready(cfg, list(plan.core_queue_names(t)), waiting):
        return False
    train, _ = _all_docs(t)
    return all(_block_complete(cfg, t, train, b, waiting) for b in t.core_blocks())


def _results(cfg: driver_io.CampaignDriverConfig, t: plan.Tier) -> list[str]:
    written = build_results(cfg.output_root, t, dry_run=cfg.dry_run)
    return [str(p) for p in written.values()]


def _finalize_core(
    cfg: driver_io.CampaignDriverConfig, t: plan.Tier, state: dict
) -> None:
    written = _results(cfg, t)
    marker = cfg.logs_dir / plan.CAMPAIGN_DONE_FILE
    if not cfg.dry_run:
        atomic_write_text(
            marker, f"campaign {t.run_set_id} core finished {utc_now_iso()}\n"
        )
    state["transitions"]["finalize_core"] = {"at": utc_now_iso(), "results": written}
    logger.info("{} CORE COMPLETE — marker at {}", t.token, marker)


def _tail_units(t: plan.Tier) -> list[tuple[str, plan.Block, bool]]:
    units = [(f"results_{b.name}", b, False) for b in t.tail_blocks()]
    units += [(f"results_bench_{b.name}", b, True) for b in t.blocks if b.bench]
    return units


def _tail_queue_names(t: plan.Tier, b: plan.Block, bench: bool) -> list[str]:
    if bench:
        return [plan.queue_name(t, b.name, "bench", s) for s in b.sizes]
    return list(plan.block_queue_names(t, b.name))


def _tail_results(
    cfg: driver_io.CampaignDriverConfig, t: plan.Tier, state: dict, fired: list[str]
) -> None:
    transitions = state["transitions"]
    if "finalize_core" not in transitions:
        return
    train, _ = _all_docs(t)
    for key, b, bench in _tail_units(t):
        if key in transitions:
            continue
        if not _queues_finished_quiet(cfg, _tail_queue_names(t, b, bench)):
            continue
        if not _block_complete(cfg, t, train, b, [], bench=bench):
            continue
        transitions[key] = {"at": utc_now_iso(), "results": _results(cfg, t)}
        fired.append(key)


# ---- cross-tier (Tier 2) -----------------------------------------------------------------


def cross_tier_status(output_root: Path) -> dict:
    """X0 status of Tier 2 against Tier 1 (pure reads).

    Returns:
        ``{"status": "pending" | "void_pin" | "ok", "tier1_sha", "tier2_sha", "at"}``.
    """
    t1, t2 = plan.TIER1, plan.TIER2
    root1 = Path(output_root) / run_set_subdir(plan.SPECS[1].root_subdir, t1.run_set_id)
    root2 = Path(output_root) / run_set_subdir(plan.SPECS[2].root_subdir, t2.run_set_id)
    card1, card2 = read_run_set_card(root1), read_run_set_card(root2)
    sha1 = (card1 or {}).get("git_sha")
    sha2 = (card2 or {}).get("git_sha")
    if not (root1 / "logs" / plan.CAMPAIGN_DONE_FILE).is_file():
        status = "pending"
    elif not sha1 or not sha2 or sha1 != sha2:
        status = "void_pin"
    else:
        status = "ok"
    return {"status": status, "tier1_sha": sha1, "tier2_sha": sha2, "at": utc_now_iso()}


def _cross_tier(
    cfg: driver_io.CampaignDriverConfig, t: plan.Tier, state: dict, fired: list[str]
) -> None:
    transitions = state["transitions"]
    if t.n != 2 or "cross_tier" in transitions or "finalize_core" not in transitions:
        return
    doc = cross_tier_status(cfg.output_root)
    if not cfg.dry_run:
        atomic_write_json(cfg.logs_dir / plan.CROSS_TIER_STATUS, doc)
    if doc["status"] == "pending":
        return
    transitions["cross_tier"] = {**doc, "results": _results(cfg, t)}
    fired.append("cross_tier")
    if doc["status"] == "void_pin":
        logger.warning("cross-tier pins differ: {}", doc)


# ---- run / status --------------------------------------------------------------------------


def _done(t: plan.Tier, transitions: dict) -> bool:
    if "finalize_core" not in transitions:
        return False
    if any(key not in transitions for key, _, _ in _tail_units(t)):
        return False
    return t.n != 2 or "cross_tier" in transitions


def projected_hours_left(cfg: driver_io.CampaignDriverConfig, t: plan.Tier) -> float:
    """Remaining train wall hours of pending + running cells at full concurrency."""
    if cfg.dry_run:
        return 0.0
    slot_minutes = 0.0
    for b in t.blocks:
        for size in b.sizes:
            fp = plan.STUDENT_FOOTPRINT[size]
            for cls in ("students", "controls"):
                if cls not in plan.block_queue_classes(b):
                    continue
                name = plan.queue_name(t, b.name, cls, size)
                if not store.queue_exists(cfg.queue_root, name):
                    continue
                status = store.queue_status(cfg.queue_root, name)
                slot_minutes += (
                    (status["pending"] + status["running"]) * fp.cell_minutes * fp.slots
                )
    return slot_minutes / plan.total_slots() / 60.0


def run(
    cfg: driver_io.CampaignDriverConfig,
    t: plan.Tier,
    options: OpdlimDriverOptions | None = None,
) -> AdvanceResult:
    """One advance pass over tier `t`'s DAG."""
    if not is_valid_run_set_id(cfg.run_set_id):
        raise CampaignGateError(f"run_set_id {cfg.run_set_id!r} is malformed")
    if cfg.spec is not plan.SPECS[t.n]:
        raise CampaignGateError(
            f"{t.token} driver bound to spec {cfg.spec.run_set_id!r}; "
            "use driver_config(tier, ...)"
        )
    opts = options or OpdlimDriverOptions()
    fired: list[str] = []
    waiting: list[str] = []
    pin_note = driver_io.ensure_campaign_code_pin(cfg)
    if pin_note:
        fired.append(pin_note)
    state = driver_io.load_state(cfg)
    transitions = state["transitions"]

    if "preflight" not in transitions:
        _preflight(cfg, t, state, opts)
        driver_io.save_state(cfg, state)
        fired.append("preflight")

    _requeue_submitted(cfg)

    if "submit_all" not in transitions:
        _submit_all(cfg, t, state)
        driver_io.save_state(cfg, state)
        fired.append("submit_all")

    if _core_ready(cfg, t, state, waiting):
        _finalize_core(cfg, t, state)
        driver_io.save_state(cfg, state)
        fired.append("finalize_core")

    _tail_results(cfg, t, state, fired)
    _cross_tier(cfg, t, state, fired)
    driver_io.save_state(cfg, state)

    done = _done(t, transitions)
    if "finalize_core" in transitions and not done:
        waiting.append(f"tail draining (~{projected_hours_left(cfg, t):.1f} h left)")
    return AdvanceResult(tuple(fired), tuple(waiting), done)


def status(cfg: driver_io.CampaignDriverConfig, t: plan.Tier) -> dict:
    """Read-only snapshot: transitions, queues, projected hours left."""
    state = driver_io.load_state(cfg)
    queues = {
        name: store.queue_status(cfg.queue_root, name)
        for name in cfg.queue_priorities
        if store.queue_exists(cfg.queue_root, name)
    }
    return {
        "tier": t.n,
        "run_set_id": t.run_set_id,
        "home_node": t.node,
        "transitions": sorted(state["transitions"]),
        "queues": queues,
        "projected_hours_left": projected_hours_left(cfg, t),
        "planned_hours": plan.tier_hours(t),
        "core_done": "finalize_core" in state["transitions"],
        "done": _done(t, state["transitions"]),
    }
