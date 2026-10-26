"""Auto-advancing driver of Part III (`traitfree3`), on soar.

Part I cells re-run with placebo teachers on the v2 trait-free corpora (`placebo2`,
one per slot animal) and the `v1anchor` re-run on the v1 corpus, compared with the
v1 placebo leaves on disk (plan `.claude/plans/traitfree3_v2_placebo_controls.plan.md`).
Queues are unpinned in the Part III queue root, so every pool on `plan.HOSTS3`
drains them; each queue carries every host's code pin under ``defaults.nodes``.
One `run(cfg)` = one advance pass; every transition is idempotent and persisted:

    preflight        -> roster (sha pinned into state), staging report, paths YAML,
                        queue root, v2 corpus index sha, v1 leaves present, disk
    slot_copies      -> per-slot training copies: each v2 slot its animal's corpus,
                        the anchor the Part I bear copy (create-only, sha-checked)
    stage_source_manifests -> Part III copies of the source leaves' manifests
    submit           -> generated YAMLs, the identity preflight, tier queues
    (every pass)     -> run-set cards and queue pins follow the pin; failed jobs
                        requeue under the cap; holes recorded; campaign-wide card
                        holds while a whole-card job waits
    results_<tier>   -> a results build once every queue of tiers <= tier is done
    finalize         -> last results build + CAMPAIGN_DONE when every queue is
                        complete with no failed job

Waiting never raises; `CampaignGateError` is a contract violation the operator must
repair.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from loguru import logger

from pipelines import traitfree_campaign_pipeline as part1
from pipelines import traitfree_staging_pipeline as staging
from pipelines.traitfree_campaign_pipeline import AdvanceResult, CampaignGateError
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import traitfree_config_gen as cgen
from subliminal_learning.campaign import traitfree_jobs as tjobs
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign.traitfree_roster import Roster, load_roster, validate_roster
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.jobqueue import store
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the Part III spec."""
    kwargs.setdefault("generated_dir_override", plan.ROOTS3.generated_config_dir)
    return driver_io.CampaignDriverConfig(spec=plan.SPEC3, **kwargs)


def results_root(cfg: driver_io.CampaignDriverConfig) -> Path:
    """Parent of the Part III results builds."""
    return (
        Path(cfg.output_root)
        / GKD_STUDENTS_SUBDIR
        / plan.DRIVER3_RUN_SET_ID
        / plan.RESULTS_DIRNAME
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def corpus_index() -> dict:
    """The `placebo2` canonical corpora index, sha-checked.

    Raises:
        CampaignGateError: The index differs from the pinned sha.
    """
    data = plan.CORPUS3_INDEX.read_bytes()
    if sha256_bytes(data) != plan.CORPUS3_INDEX_SHA256:
        raise CampaignGateError(f"{plan.CORPUS3_INDEX} differs from the pinned sha")
    return json.loads(data)


def v2_slot_body(slot: str, index: dict) -> str:
    """Training copy of `slot`'s v2 corpus: its rows plus ``animal`` and the corpus tag.

    Raises:
        CampaignGateError: The slot has no corpus, or its docs differ from the index.
    """
    entry = index["corpora"].get(slot)
    if entry is None:
        raise CampaignGateError(f"no v2 corpus for slot {slot!r}")
    data = (Path(entry["dir"]) / entry["docs"]).read_bytes()
    if sha256_bytes(data) != entry["docs_sha256"]:
        raise CampaignGateError(f"{entry['dir']}: docs sha differs from the index")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != int(entry["n_docs"]):
        raise CampaignGateError(f"{slot}: {len(rows)} docs != index {entry['n_docs']}")
    return "".join(
        json.dumps(r | {"animal": slot, "trait_free_corpus": plan.CORPUS3_NAME}, ensure_ascii=False)
        + "\n"
        for r in rows
    )


def _slot_copies(roster: Roster, state: dict) -> None:
    index = corpus_index()
    written: dict[str, str] = {}
    for t in sorted(roster.teachers, key=lambda t: t.key):
        dest = plan.slot_corpus_path(t.slot_animal, t.family)
        if str(dest) in written:
            continue
        if t.family == plan.FAMILY_V1ANCHOR:
            sha, _ = staging.copy_file(plan.slot_corpus_path(t.slot_animal), dest)
            if sha != plan.V1_BEAR_SLOT_SHA256:
                raise CampaignGateError(f"{dest}: v1 slot copy sha {sha} != pinned")
        else:
            staging.write_once_text(dest, v2_slot_body(t.slot_animal, index))
            sha = sha256_bytes(dest.read_bytes())
        written[str(dest)] = sha
    part1._fire(state, "slot_copies", files=written)


def _preflight(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    if Path(cfg.queue_root).resolve() != plan.QUEUE_ROOT3.resolve():
        raise CampaignGateError(f"queue root {cfg.queue_root} != {plan.QUEUE_ROOT3}")
    roster, sha = load_roster(plan.ROSTER3_PATH)
    validate_roster(roster)
    pinned = state.get("roster_sha256")
    if pinned is not None and pinned != sha:
        raise CampaignGateError(f"roster sha {sha} != pinned {pinned}")
    if staging.latest_complete_report(sha, plan.ROOTS3) is None:
        raise CampaignGateError(f"no complete Part III staging report for roster {sha[:12]}")
    if plan.ROOTS3.paths_config_path.read_text() != staging.paths3_yaml_text():
        raise CampaignGateError(f"{plan.ROOTS3.paths_config_path} differs from the plan")
    index = corpus_index()
    for slot in sorted({t.slot_animal for t in roster.teachers}):
        if slot not in index["corpora"]:
            raise CampaignGateError(f"no v2 corpus for slot {slot!r}")
    missing = [
        ref.path
        for cell in roster.cells
        for ref in cell.ref_placebo
        if not (Path(ref.path) / "preference_eval.json").is_file()
    ]
    if missing:
        raise CampaignGateError(f"{len(missing)} v1 placebo leaves missing, e.g. {missing[0]}")
    free_gib = shutil.disk_usage(plan.CAMPAIGN3_ROOT).free / 2**30
    if free_gib < plan.MIN_DISK_FREE3_GIB:
        raise CampaignGateError(f"{plan.CAMPAIGN3_ROOT}: {free_gib:.0f} GiB free")
    state["roster_sha256"] = sha
    part1._fire(state, "preflight", roster_sha256=sha, n_cells=len(roster.cells))


def queue_defaults(sha: str) -> dict:
    """jobq defaults of a Part III queue: every host's pin (cwd + env) under ``nodes``."""
    per_host = {
        h: code_pin_io.pin_job_defaults(
            code_pin_io.CodePin(
                sha=sha, root=plan.CLONE_ROOT[h] / code_pin_io.CODE_PINS_DIRNAME / sha
            )
        )
        for h in plan.HOSTS3
    }
    return {**per_host[plan.HOME_NODE3], "nodes": per_host}


def _submit(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    roster, _ = load_roster(plan.ROSTER3_PATH)
    n = part1._write_configs(cgen.all_configs(roster))
    part1._run_preflight_identity(cfg, plan.ROSTER3_PATH, plan.PREFLIGHT3_REPORT)
    specs = tjobs.part3_queues(roster)
    defaults = queue_defaults(part1.current_pin(cfg).sha)
    for spec in specs:
        driver_io.submit_queue(
            cfg,
            spec.name,
            list(spec.jobs),
            depends_on=list(spec.depends_on) or None,
            defaults=defaults,
            strict_deps=spec.strict_deps,
            unpinned=True,
        )
    part1._fire(
        state,
        "submit",
        n_configs=n,
        n_queues=len(specs),
        n_jobs=sum(len(s.jobs) for s in specs),
    )


def _repin(cfg: driver_io.CampaignDriverConfig) -> str | None:
    """Move run-set cards and every queue with work left (or failures) to the pin."""
    if cfg.dry_run:
        return None
    pin = part1.current_pin(cfg)
    n_cards = part1.repin_cards(cfg, pin, plan.PART3_FAMILIES)
    want = queue_defaults(pin.sha)
    n_queues = 0
    for name in part1._submitted(cfg):
        if (
            store.queue_complete(cfg.queue_root, name)
            and not store.queue_status(cfg.queue_root, name)["failed"]
        ):
            continue
        if store.read_meta(cfg.queue_root, name).defaults == want:
            continue
        store.update_queue_defaults(cfg.queue_root, name, want)
        n_queues += 1
    if not n_cards and not n_queues:
        return None
    logger.info("repinned {} run-set cards and {} queues to {}", n_cards, n_queues, pin.sha[:12])
    return f"repin(cards={n_cards}, queues={n_queues})"


def _card_holds(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    min_wait_s: float = part1.HEAVY_WAIT_S,
) -> str | None:
    """Hold every unfinished queue while a multi-slot job waits for a whole card.

    The queues are unpinned, so the hold is campaign-wide: each unfinished queue (the
    waiting job's own included) gets the waiting job's queue as an extra dependency
    (recorded in `state["card_holds"]`) until no multi-slot job waits; claimed jobs
    are not affected.
    """
    if cfg.dry_run:
        return None
    heavy = part1.heavy_waiting_queues(cfg, min_wait_s)
    holds: dict[str, str] = state.setdefault("card_holds", {})
    added = released = 0
    for queue, dep in list(holds.items()):
        if dep in heavy:
            continue
        deps = list(store.read_meta(cfg.queue_root, queue).depends_on)
        if dep in deps:
            store.update_queue_depends_on(cfg.queue_root, queue, [d for d in deps if d != dep])
        del holds[queue]
        released += 1
    if heavy:
        dep = sorted(heavy)[0]
        for queue in part1._submitted(cfg):
            if queue in holds or store.queue_complete(cfg.queue_root, queue):
                continue
            deps = list(store.read_meta(cfg.queue_root, queue).depends_on)
            if dep not in deps:
                store.update_queue_depends_on(cfg.queue_root, queue, [*deps, dep])
                holds[queue] = dep
                added += 1
    if not added and not released:
        return None
    logger.info("card holds: +{} -{} (waiting multi-slot jobs in {})", added, released, sorted(heavy))
    return f"card_holds(+{added},-{released})"


def tier_queues(names: list[str], tier: str) -> list[str]:
    """Queues a tier's results build waits on: those of tiers <= `tier`."""
    tiers = plan.TIERS3[: plan.TIERS3.index(tier) + 1]
    return [n for n in names if n.rsplit("-", 1)[1] in tiers]


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
    roster, _ = load_roster(plan.ROSTER3_PATH)
    if cfg.dry_run:
        n_configs = len(cgen.all_configs(roster))
        specs = tjobs.part3_queues(roster)
        plan_note = ", ".join(f"{s.name}={len(s.jobs)}" for s in specs)
        logger.info("[dry-run] {} configs; queues {}", n_configs, plan_note)
        return AdvanceResult(tuple(fired), ("dry run: nothing written or submitted",), False)
    if "slot_copies" not in done:
        _slot_copies(roster, state)
        mark("slot_copies")
    if "stage_source_manifests" not in done:
        n = staging.stage_source_manifests(roster, plan.ROOTS3)
        part1._fire(state, "stage_source_manifests", n_leaves=n)
        mark("stage_source_manifests")
    if "submit" not in done:
        _submit(cfg, state)
        mark("submit")
    repin = _repin(cfg)
    if repin:
        fired.append(repin)
    part1._requeue_and_holes(cfg, state)
    note = _card_holds(cfg, state)
    if note:
        fired.append(note)

    names = part1._submitted(cfg)
    for tier in plan.TIERS3:
        key = f"results_{tier}"
        if key in done:
            continue
        if part1._finished(cfg, tier_queues(names, tier), []):
            part1._build_results(cfg, key, plan.ROSTER3_PATH, results_root(cfg))
            part1._fire(state, key)
            mark(key)
    all_done = False
    if part1._finished(cfg, names, waiting) and not state.get("holes"):
        if "finalize" not in done:
            part1._build_results(cfg, "finalize", plan.ROSTER3_PATH, results_root(cfg))
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
    """Write `host`'s jobq policy file into the Part III queue root.

    Raises:
        ValueError: `host` runs no Part III queue.
    """
    if host not in plan.HOSTS3:
        raise ValueError(f"{host} runs no Part III queue")
    path = plan.QUEUE_ROOT3 / f"gpu_policy.{host}.json"
    doc = plan.gpu_policy_json(host, plan.GPU_POLICY3)
    if path.is_file() and not replace:
        if json.loads(path.read_text()) != doc:
            raise FileExistsError(f"{path} exists with a different policy")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    part1.atomic_write_json(path, doc)
    return path
