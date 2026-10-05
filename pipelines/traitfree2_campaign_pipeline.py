"""Auto-advancing driver of Part II (`traitfree2`), on soar.

Placebo controls for the on-policy paper figures Part I left on the untuned-base
control (plan §7-12): the Part II roster's grid and teacher-epoch cells, trained on
the published Part I corpus and reusing Part I's placebo teachers where identical,
plus the `consensus` re-score of the sdfpos2 placebo control. Every queue is pinned
to `plan.HOSTS2`. One `run(cfg)` = one advance pass; every transition is idempotent
and persisted:

    preflight        -> Part II prereg, published corpus, roster (sha pinned into
                        state), staging report, paths YAML, queue root, reused
                        teachers on disk, disk
    stage_source_manifests -> ssd-1 copies of the source leaves' manifests
    submit           -> generated YAMLs, the identity preflight, teacher, student and
                        re-score queues
    (every pass)     -> run-set cards and unfinished queues follow the pin;
                        failed jobs requeue under the cap; card holds
    results_<tier>   -> a results build once every queue of tiers <= tier is done
    finalize         -> last results build + CAMPAIGN_DONE when every queue is
                        complete with no failed job

Waiting never raises; `CampaignGateError` is a contract violation the operator must
repair. A job still failing at the requeue cap is recorded as a hole.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from pipelines import traitfree_campaign_pipeline as part1
from pipelines import traitfree_staging_pipeline as staging
from pipelines.traitfree_campaign_pipeline import AdvanceResult, CampaignGateError
from pipelines.traitfree_roster_pipeline import check_reused_teachers
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import traitfree_config_gen as cgen
from subliminal_learning.campaign import traitfree_jobs as tjobs
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign.traitfree_roster import load_roster, validate_roster
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue import store
from subliminal_learning.utils.io import atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso

PART2_TIERS: tuple[str, ...] = (plan.TIER_A, plan.TIER_B)


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the Part II spec."""
    kwargs.setdefault("generated_dir_override", plan.GENERATED_CONFIG_DIR)
    return driver_io.CampaignDriverConfig(spec=plan.SPEC2, **kwargs)


def results_root(cfg: driver_io.CampaignDriverConfig) -> Path:
    """Parent of the Part II results builds."""
    return (
        Path(cfg.output_root)
        / GKD_STUDENTS_SUBDIR
        / plan.DRIVER2_RUN_SET_ID
        / plan.RESULTS_DIRNAME
    )


def _preflight(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    prereg = cfg.repo_root / plan.PREREG_PATH
    if plan.PART2_PREREG_HEADING not in prereg.read_text():
        raise CampaignGateError(f"{plan.PREREG_PATH} has no {plan.PART2_PREREG_HEADING}")
    if not (plan.CORPUS_DIR / plan.CORPUS_READY_FILE).is_file():
        raise CampaignGateError(f"Part I corpus not published under {plan.CORPUS_DIR}")
    if Path(cfg.queue_root).resolve() != plan.QUEUE_ROOT2.resolve():
        raise CampaignGateError(f"queue root {cfg.queue_root} != {plan.QUEUE_ROOT2}")
    roster, sha = load_roster(plan.ROSTER2_PATH)
    validate_roster(roster)
    pinned = state.get("roster_sha256")
    if pinned is not None and pinned != sha:
        raise CampaignGateError(f"roster sha {sha} != pinned {pinned}")
    if staging.latest_complete_report(sha) is None:
        raise CampaignGateError(f"no complete staging report for roster {sha[:12]}")
    if plan.PATHS2_CONFIG_PATH.read_text() != staging.paths2_yaml_text():
        raise CampaignGateError(f"{plan.PATHS2_CONFIG_PATH} differs from the plan")
    for slot in sorted({t.slot_animal for t in roster.teachers}):
        if not plan.slot_corpus_path(slot).is_file():
            raise CampaignGateError(f"no Part I slot copy for {slot}")
    check_reused_teachers(roster.reused_teachers, Path(cfg.output_root))
    free_gib = shutil.disk_usage(plan.CAMPAIGN_ROOT).free / 2**30
    if free_gib < plan.MIN_DISK_FREE_GIB:
        raise CampaignGateError(f"{plan.CAMPAIGN_ROOT}: {free_gib:.0f} GiB free")
    state["roster_sha256"] = sha
    part1._fire(state, "preflight", roster_sha256=sha, n_cells=len(roster.cells))


def _submit(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    roster, _ = load_roster(plan.ROSTER2_PATH)
    n = part1._write_configs(cgen.all_configs(roster))
    part1._run_preflight_identity(cfg, plan.ROSTER2_PATH, plan.PREFLIGHT2_REPORT)
    specs = tjobs.part2_queues(roster)
    teachers = [s for s in specs if s.name.startswith(f"{plan.CAMPAIGN2}-teachers-")]
    rest = [s for s in specs if s not in teachers]
    for spec in [*teachers, *rest]:
        part1._submit(
            cfg,
            spec.name,
            list(spec.jobs),
            depends_on=spec.depends_on,
            strict=spec.strict_deps,
        )
    part1._fire(
        state,
        "submit",
        n_configs=n,
        n_queues=len(specs),
        n_jobs=sum(len(s.jobs) for s in specs),
    )


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
    if "stage_source_manifests" not in done and not cfg.dry_run:
        roster, _ = load_roster(plan.ROSTER2_PATH)
        part1._fire(
            state,
            "stage_source_manifests",
            n_leaves=staging.stage_source_manifests(roster),
        )
        mark("stage_source_manifests")
    if "submit" not in done:
        _submit(cfg, state)
        mark("submit")
    repin = part1._repin(cfg, plan.PART2_FAMILIES)
    if repin:
        fired.append(repin)
    part1._requeue_and_holes(cfg, state)
    note = part1._card_holds(cfg, state)
    if note:
        fired.append(note)

    if cfg.dry_run:
        return AdvanceResult(tuple(fired), ("dry run: no results",), False)
    names = part1._submitted(cfg)
    for tier in PART2_TIERS:
        key = f"results_{tier}"
        if key in done:
            continue
        if part1._finished(cfg, part1.tier_queues(names, tier), []):
            part1._build_results(cfg, key, plan.ROSTER2_PATH, results_root(cfg))
            part1._fire(state, key)
            mark(key)
    all_done = False
    if part1._finished(cfg, names, waiting) and not state.get("holes"):
        if "finalize" not in done:
            part1._build_results(cfg, "finalize", plan.ROSTER2_PATH, results_root(cfg))
            atomic_write_text(cfg.logs_dir / plan.CAMPAIGN_DONE_FILE, utc_now_iso() + "\n")
            part1._fire(state, "finalize")
            mark("finalize")
        all_done = True
    if state.get("holes"):
        waiting.append(f"holes in {sorted(state['holes'])}")
    state["last_pass"] = {"at": utc_now_iso(), "fired": fired, "waiting": waiting[:50]}
    driver_io.save_state(cfg, state)
    return AdvanceResult(tuple(fired), tuple(waiting), all_done)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Transitions, holes and per-queue counts (pure file reads)."""
    state = driver_io.load_state(cfg)
    queues = {}
    for name in part1._submitted(cfg):
        s = store.queue_status(cfg.queue_root, name)
        queues[name] = {k: v for k, v in s.items() if k != "running_jobs"}
    return {
        "transitions": sorted(state.get("transitions", {})),
        "holes": state.get("holes", {}),
        "queues": queues,
    }


def write_policy(host: str, *, replace: bool = False) -> Path:
    """Write `host`'s jobq policy file into the Part II queue root."""
    if host not in plan.HOSTS2:
        raise ValueError(f"{host} runs no Part II queue")
    return part1.write_policy(host, replace=replace, queue_root=plan.QUEUE_ROOT2)
