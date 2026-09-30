"""Auto-advancing driver of the `traitfree` campaign (Stage 1 -> Stage 2), on soar.

One `run(cfg)` = one advance pass; every transition is idempotent and persisted:

    preflight        -> prereg, roster (sha pinned into state), staging report,
                        paths YAML, queue root, disk
    submit_docgen    -> docgen shard queues (wave 0) on all five hosts
    submit_tb        -> missing untuned-teacher-base controls (need no corpus;
                        their queues depend on the wave-0 docgen queues)
    stage_source_manifests -> ssd-1 copies of the source leaves' manifests (the leaf
                        verify reads these; cluster-5 has no ssd-3)
    (every pass)     -> family run-set cards and unfinished queues follow the pin
    finalize_corpus  -> once every submitted docgen wave is complete:
                        `sdf_docgen.traitfree_select`; too few screened documents
                        -> one more top-up wave (at most DOCGEN_MAX_TOPUPS)
    slot_copies      -> per-slot training copies of the published corpus
    submit_stage2    -> generated YAMLs, the identity preflight, placebo teacher
                        and student queues
    results_<tier>   -> a results build once every queue of tiers <= tier is done
    finalize         -> last results build + CAMPAIGN_DONE when every queue is
                        complete with no failed job

Waiting never raises; `CampaignGateError` is a contract violation the operator must
repair. A job still failing at the requeue cap is recorded as a hole and holds its
queue (and finalize) until repaired.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

from pipelines import traitfree_staging_pipeline as staging
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import traitfree_docgen_jobs as docgen_jobs
from subliminal_learning.campaign import traitfree_plan as plan
from subliminal_learning.campaign.traitfree_roster import load_roster, validate_roster
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import (
    read_run_set_card,
    reconcile_run_set_sha,
    utc_now_iso,
)


class CampaignGateError(RuntimeError):
    """A campaign contract refused to pass."""


@dataclass(frozen=True)
class AdvanceResult:
    """Outcome of one advance pass."""

    fired: tuple[str, ...]
    waiting: tuple[str, ...]
    done: bool


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the traitfree spec."""
    kwargs.setdefault("generated_dir_override", plan.GENERATED_CONFIG_DIR)
    return driver_io.CampaignDriverConfig(spec=plan.SPEC, **kwargs)


# ---- pins and submission ------------------------------------------------------------


def current_pin(cfg: driver_io.CampaignDriverConfig) -> code_pin_io.CodePin:
    """The campaign pin on this host."""
    return code_pin_io.resolve_campaign_pin(
        cfg.run_set_root,
        repo_root=cfg.repo_root,
        requested_sha=cfg.code_sha,
        allow_advance=False,
    )


def node_pin_defaults(cfg: driver_io.CampaignDriverConfig, host: str) -> dict:
    """jobq defaults binding a queue to `host`'s worktree of the campaign pin."""
    if cfg.dry_run:
        return {"cwd": str(plan.CLONE_ROOT[host] / code_pin_io.CODE_PINS_DIRNAME)}
    pin = current_pin(cfg)
    if host != store.this_host():
        root = plan.CLONE_ROOT[host] / code_pin_io.CODE_PINS_DIRNAME / pin.sha
        pin = code_pin_io.CodePin(sha=pin.sha, root=root)
    return code_pin_io.pin_job_defaults(pin)


def _submit(
    cfg: driver_io.CampaignDriverConfig,
    name: str,
    jobs: list[Job],
    *,
    depends_on: tuple[str, ...] | list[str] = (),
    strict: bool = False,
) -> str:
    if not jobs:
        raise CampaignGateError(f"queue {name} built zero jobs")
    host = plan.queue_host(name)
    return driver_io.submit_queue(
        cfg,
        name,
        jobs,
        depends_on=list(depends_on) or None,
        node=host,
        defaults=node_pin_defaults(cfg, host),
        strict_deps=strict,
    )


def _submitted(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [n for n in plan.QUEUE_PRIORITIES if store.queue_exists(cfg.queue_root, n)]


RUN_SET_WORKFLOWS = ("sdf_teachers", "gkd_students")


def _repin(cfg: driver_io.CampaignDriverConfig) -> str | None:
    """Move family run-set cards and every queue with work left (or failures) to the pin."""
    if cfg.dry_run:
        return None
    pin = current_pin(cfg)
    n_cards = 0
    for run_set_id in sorted(set(plan.RUN_SET_ID.values())):
        for workflow in RUN_SET_WORKFLOWS:
            root = cfg.output_root / run_set_subdir(workflow, run_set_id)
            card = read_run_set_card(root)
            if not card or not card.get("git_sha"):
                continue
            if code_pin_io.resolve_commit_sha(cfg.repo_root, str(card["git_sha"])) == pin.sha:
                continue
            reconcile_run_set_sha(
                root, new_git_sha=pin.sha, reason="traitfree driver repin", repo_root=cfg.repo_root
            )
            n_cards += 1
    wanted: dict[str, dict] = {}
    n_queues = 0
    for name in _submitted(cfg):
        if (
            store.queue_complete(cfg.queue_root, name)
            and not store.queue_status(cfg.queue_root, name)["failed"]
        ):
            continue
        meta = store.read_meta(cfg.queue_root, name)
        host = str(meta.node)
        want = wanted.setdefault(host, node_pin_defaults(cfg, host))
        env = meta.defaults.get("env") or {}
        if meta.defaults.get("cwd") == want["cwd"] and all(
            env.get(k) == v for k, v in (want.get("env") or {}).items()
        ):
            continue
        if not Path(want["cwd"]).is_dir() and host == store.this_host():
            raise CampaignGateError(f"pin worktree {want['cwd']} missing on {host}")
        store.update_queue_defaults(cfg.queue_root, name, want)
        n_queues += 1
    if not n_cards and not n_queues:
        return None
    logger.info("repinned {} run-set cards and {} queues to {}", n_cards, n_queues, pin.sha[:12])
    return f"repin(cards={n_cards}, queues={n_queues})"


def _requeue_and_holes(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    """Auto-requeue failures under the cap; record the rest as holes."""
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


def _finished(cfg: driver_io.CampaignDriverConfig, names: list[str], waiting: list[str]) -> bool:
    ok_all = True
    for name in names:
        if cfg.dry_run:
            continue
        ok, msg = driver_io.queue_finished(cfg, name)
        if not ok:
            waiting.append(msg)
            ok_all = False
    return ok_all


def _fire(state: dict, name: str, **info: object) -> None:
    state["transitions"][name] = {"at": utc_now_iso(), **info}


# ---- preflight ----------------------------------------------------------------------


def _preflight(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    if not (cfg.repo_root / plan.PREREG_PATH).is_file():
        raise CampaignGateError(f"missing pre-registration {plan.PREREG_PATH}")
    expected_queue_root = Path(plan.paths_config_doc()["queue_root"])
    if Path(cfg.queue_root).resolve() != expected_queue_root.resolve():
        raise CampaignGateError(f"queue root {cfg.queue_root} != {expected_queue_root}")
    roster, sha = load_roster(plan.ROSTER_PATH)
    validate_roster(roster)
    pinned = state.get("roster_sha256")
    if pinned is not None and pinned != sha:
        raise CampaignGateError(f"roster sha {sha} != pinned {pinned}")
    if staging.latest_complete_report(sha) is None:
        raise CampaignGateError(f"no complete staging report for roster {sha[:12]}")
    if plan.PATHS_CONFIG_PATH.read_text() != staging.paths_yaml_text():
        raise CampaignGateError(f"{plan.PATHS_CONFIG_PATH} differs from the plan")
    free_gib = shutil.disk_usage(plan.CAMPAIGN_ROOT).free / 2**30
    if free_gib < plan.MIN_DISK_FREE_GIB:
        raise CampaignGateError(f"{plan.CAMPAIGN_ROOT}: {free_gib:.0f} GiB free")
    state["roster_sha256"] = sha
    _fire(state, "preflight", roster_sha256=sha, n_cells=len(roster.cells))


# ---- stage 1 ------------------------------------------------------------------------


def _hold_tb_behind(cfg: driver_io.CampaignDriverConfig, names: list[str]) -> int:
    """Add `names` to the deps of every unfinished tb queue (frees whole cards for docgen)."""
    n = 0
    for queue in _submitted(cfg):
        if not _is_tb(queue) or store.queue_complete(cfg.queue_root, queue):
            continue
        deps = list(store.read_meta(cfg.queue_root, queue).depends_on)
        missing = [d for d in names if d not in deps]
        if missing:
            store.update_queue_depends_on(cfg.queue_root, queue, deps + missing)
            n += 1
    return n


def _submit_docgen_wave(cfg: driver_io.CampaignDriverConfig, state: dict, wave: int) -> None:
    queues = docgen_jobs.docgen_queue_jobs(wave)
    for name, jobs in queues.items():
        _submit(cfg, name, jobs)
    names = list(queues)
    if wave > 0 and not cfg.dry_run:
        logger.info("held {} tb queues behind top-up wave {}", _hold_tb_behind(cfg, names), wave)
    state["docgen_waves"] = max(int(state.get("docgen_waves", 0)), wave + 1)
    _fire(state, f"submit_docgen_wave{wave}", n_shards=len(plan.docgen_shards_of_wave(wave)))


def _docgen_queue_names(state: dict) -> list[str]:
    waves = int(state.get("docgen_waves", 0))
    return [
        plan.docgen_queue(host, wave)
        for wave in range(waves)
        for host in plan.HOSTS
        if plan.docgen_queue(host, wave) in docgen_jobs.docgen_queue_jobs(wave)
    ]


def _run_finalize(cfg: driver_io.CampaignDriverConfig, n_shards: int) -> int:
    pin = current_pin(cfg)
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(pin.root / docgen_jobs.DOCGEN_SRC),
            "SL_CODE_PIN_SHA": pin.sha,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "8",
        }
    )
    argv = docgen_jobs.finalize_argv(store.this_host(), n_shards)
    logger.info("finalize: {}", " ".join(argv))
    log = plan.DOCGEN_ROOT / "logs" / f"finalize-{utc_now_iso().replace(':', '')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as handle:
        proc = subprocess.run(argv, cwd=pin.root, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    logger.info("finalize exit {} (log {})", proc.returncode, log)
    return proc.returncode


def _finalize_corpus(cfg: driver_io.CampaignDriverConfig, state: dict, waiting: list[str]) -> bool:
    if (plan.CORPUS_DIR / plan.CORPUS_READY_FILE).is_file():
        return True
    if not _finished(cfg, _docgen_queue_names(state), waiting):
        return False
    waves = int(state["docgen_waves"])
    rc = _run_finalize(cfg, plan.docgen_shard_count(waves - 1))
    if rc == 0:
        manifest = json.loads((plan.CORPUS_DIR / plan.CORPUS_MANIFEST_FILE).read_text())
        _fire(state, "finalize_corpus", n_docs=manifest["n_docs"], token_gap=manifest["token_gap"])
        return True
    if rc == 3 and waves - 1 < plan.DOCGEN_MAX_TOPUPS:
        _submit_docgen_wave(cfg, state, waves)
        waiting.append(f"docgen top-up wave {waves} submitted")
        return False
    raise CampaignGateError(f"corpus finalize failed with exit {rc}; see {plan.DOCGEN_ROOT}/logs")


def _slot_copies(state: dict) -> None:
    roster, _ = load_roster(plan.ROSTER_PATH)
    manifest = json.loads((plan.CORPUS_DIR / plan.CORPUS_MANIFEST_FILE).read_text())
    rows = [
        json.loads(line)
        for line in (plan.CORPUS_DIR / "docs.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    texts_sha = staging_texts_sha([r["text"] for r in rows])
    if texts_sha != manifest["texts_sha256"]:
        raise CampaignGateError("published corpus texts do not match its manifest")
    slots = sorted({t.slot_animal for t in roster.teachers})
    for slot in slots:
        body = "".join(
            json.dumps(r | {"animal": slot, "trait_free_corpus": plan.CORPUS_NAME}, ensure_ascii=False) + "\n"
            for r in rows
        )
        staging.write_once_text(plan.slot_corpus_path(slot), body)
    _fire(state, "slot_copies", slots=slots, texts_sha256=texts_sha)


def staging_texts_sha(texts: list[str]) -> str:
    """Same order-sensitive text hash as `sdf_docgen.traitfree_select.texts_sha256`."""
    import hashlib

    return hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode("utf-8")).hexdigest()


# ---- stage 2 ------------------------------------------------------------------------


def _write_configs(configs: dict[Path, dict]) -> int:
    for path, doc in configs.items():
        staging.write_once_text(Path(path), yaml.safe_dump(doc, sort_keys=True))
    return len(configs)


def _run_preflight_identity(cfg: driver_io.CampaignDriverConfig) -> None:
    pin = current_pin(cfg)
    report = plan.GENERATED_CONFIG_DIR / "preflight_identity_report.json"
    if report.is_file() and json.loads(report.read_text()).get("ok"):
        return
    argv = [
        "uv", "run", "python", "-m", "scripts.preflight_traitfree_configs",
        "--roster", str(plan.ROSTER_PATH), "--report", str(report),
    ]
    env = dict(os.environ)
    env.update(code_pin_io.pin_job_defaults(pin).get("env", {}))
    proc = subprocess.run(argv, cwd=pin.root, env=env, check=False)
    if proc.returncode != 0:
        raise CampaignGateError(f"config identity preflight failed; see {report}")


def _stage2_specs():
    from subliminal_learning.campaign import traitfree_jobs

    roster, _ = load_roster(plan.ROSTER_PATH)
    return roster, traitfree_jobs.stage2_queues(roster)


def _is_tb(name: str) -> bool:
    return name.startswith(f"{plan.CAMPAIGN}-tbcontrols-")


def _submit_tb(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    from subliminal_learning.campaign import traitfree_config_gen

    roster, specs = _stage2_specs()
    n = _write_configs(traitfree_config_gen.all_configs(roster))
    _run_preflight_identity(cfg)
    tb = [s for s in specs if _is_tb(s.name)]
    for spec in tb:
        _submit(cfg, spec.name, list(spec.jobs), depends_on=spec.depends_on, strict=spec.strict_deps)
    _fire(state, "submit_tb", n_configs=n, n_queues=len(tb), n_jobs=sum(len(s.jobs) for s in tb))


def _submit_stage2(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    _run_preflight_identity(cfg)
    _, specs = _stage2_specs()
    rest = [s for s in specs if not _is_tb(s.name)]
    teachers = [s for s in rest if s.name.startswith(f"{plan.CAMPAIGN}-teachers-")]
    students = [s for s in rest if s not in teachers]
    for spec in [*teachers, *students]:
        _submit(cfg, spec.name, list(spec.jobs), depends_on=spec.depends_on, strict=spec.strict_deps)
    _fire(
        state,
        "submit_stage2",
        n_queues=len(rest),
        n_jobs=sum(len(s.jobs) for s in rest),
    )


def tier_queues(names: list[str], tier: str) -> list[str]:
    """Queues a tier's results build waits on: docgen, teachers, groups of tiers <= tier."""
    tiers = plan.TIERS[: plan.TIERS.index(tier) + 1]
    out = []
    for name in names:
        group = plan.queue_group(name)
        if group is None or plan.group_tier(group) in tiers:
            out.append(name)
    return out


def _build_results(cfg: driver_io.CampaignDriverConfig, label: str) -> None:
    pin = current_pin(cfg)
    argv = ["uv", "run", "python", "-m", "scripts.build_traitfree_results", "--roster", str(plan.ROSTER_PATH)]
    env = dict(os.environ)
    env.update(code_pin_io.pin_job_defaults(pin).get("env", {}))
    proc = subprocess.run(argv, cwd=pin.root, env=env, check=False)
    if proc.returncode != 0:
        raise CampaignGateError(f"results build {label} failed (exit {proc.returncode})")


# ---- driver --------------------------------------------------------------------------


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
    if "submit_docgen_wave0" not in done:
        _submit_docgen_wave(cfg, state, 0)
        mark("submit_docgen_wave0")
    if "submit_tb" not in done:
        _submit_tb(cfg, state)
        mark("submit_tb")
    if "stage_source_manifests" not in done and not cfg.dry_run:
        roster, _ = load_roster(plan.ROSTER_PATH)
        _fire(state, "stage_source_manifests", n_leaves=staging.stage_source_manifests(roster))
        mark("stage_source_manifests")
    repin = _repin(cfg)
    if repin:
        fired.append(repin)
    _requeue_and_holes(cfg, state)
    if "finalize_corpus" not in done:
        if _finalize_corpus(cfg, state, waiting):
            mark("finalize_corpus")
        else:
            driver_io.save_state(cfg, state)
    if "finalize_corpus" in done and "slot_copies" not in done:
        _slot_copies(state)
        mark("slot_copies")
    if "slot_copies" in done and "submit_stage2" not in done:
        _submit_stage2(cfg, state)
        mark("submit_stage2")

    all_done = False
    if "submit_stage2" in done:
        names = _submitted(cfg)
        for tier in plan.TIERS:
            key = f"results_{tier}"
            if key in done:
                continue
            if _finished(cfg, tier_queues(names, tier), []):
                _build_results(cfg, key)
                _fire(state, key)
                mark(key)
        if _finished(cfg, names, waiting) and not state.get("holes"):
            if "finalize" not in done:
                _build_results(cfg, "finalize")
                atomic_write_text(cfg.logs_dir / plan.CAMPAIGN_DONE_FILE, utc_now_iso() + "\n")
                _fire(state, "finalize")
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
    for name in _submitted(cfg):
        s = store.queue_status(cfg.queue_root, name)
        queues[name] = {k: v for k, v in s.items() if k != "running_jobs"}
    return {
        "transitions": sorted(state.get("transitions", {})),
        "holes": state.get("holes", {}),
        "docgen_waves": state.get("docgen_waves", 0),
        "queues": queues,
    }


def write_policy(host: str) -> Path:
    """Write `host`'s jobq policy file into the campaign queue root (create or equal)."""
    path = Path(plan.paths_config_doc()["queue_root"]) / f"gpu_policy.{host}.json"
    doc = plan.gpu_policy_json()
    if path.is_file():
        if json.loads(path.read_text()) != doc:
            raise FileExistsError(f"{path} exists with a different policy")
        return path
    atomic_write_json(path, doc)
    return path
