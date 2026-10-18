"""Auto-advancing driver of the `xopt2` soft run set, on soar (workers: `WORKER_HOSTS`).

One `run(cfg)` = one idempotent advance pass:

    preflight      prereg, home node, paths YAML, staging report (CE_base), queue root,
                   disk floors
    register       the campaign pin can run every job
    submit_launch  tiers v, w and o (their walks' initial rates) and g, at once; g
                   skips the cross cells whose `xopt` gl LoRA AdamW twins are finished
                   and evaluated (read from disk, recorded as `gl_reuse`)
    (every pass)   unfinished queues follow the pin; failures under the requeue cap are
                   requeued, the rest are holes; below a disk floor nothing new is
                   submitted (labelled `disk_low`). Walks: each group whose decision leaves
                   changed is re-decided (`xopt2_rules.walk_next`); a new rate goes to its
                   tier's walk queue, unless another tier already holds that identity
                   (recorded as shared); a finished walk submits its extra leaf r = 2 at
                   its decision rates (tier x) and, for a new arm, its anchor's carried
                   rates on the 18 other corners (tier ox; no eligible anchor = labelled
                   `carry|<arm>`). Fixed-rate groups (ox, g) whose leaves all finished get
                   `xopt`'s step-down into the tier's down queue; an exhausted group is
                   labelled `degenerate|<group>`. A leaf failed for good is a hole (walk
                   or queue), never an ineligible rate; a non-finite GKD loss is a
                   recorded degenerate leaf (`sweep_xopt2_soft train-leaf`), so ineligible.
    results_p1     every walk finished with no walk hole; v, w, o queues finished
                   (`driver_io.queue_finished`: complete, none failed)
    results_p2     x, ox and down-ox queues finished
    results_p3     every queue finished (g and every late lm-eval included)
    finalize       results_p3 and no hole: the channel done marker, then the campaign
                   marker once the hard channel's marker exists (both create-only)

Each phase's tables and claims are built in memory and written with a manifest last
(`xopt2_soft_results_pipeline.write_phase`).

Run-set logs / results / state live under `xopt2_plan.run_set_dir(SOFT)` (ssd-2); the run-set
card the sweeps write (the code pin) and every leaf live under the cold-1 leaf root.

Queues are unpinned (every worker host drains them, `xopt2_plan.queue_defaults`), except a
queue still holding a legacy-guard job that is not done: it stays on goncalo
(`reissue_legacy` rebuilds the queues none of whose jobs started).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import yaml
from loguru import logger

from pipelines import optparam2_campaign_pipeline as op2
from pipelines import xopt2_soft_results_pipeline as results
from pipelines import xopt2_soft_staging_pipeline as staging
from pipelines.optparam2_campaign_pipeline import (
    AdvanceResult,
    CampaignGateError,
    _fire,
    current_pin,
    node_pin_defaults,
)
from pipelines.traitfree_staging_pipeline import write_once_text
from subliminal_learning.campaign import driver_io
from subliminal_learning.campaign import optparam2_plan as oplan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.campaign import xopt2_plan as plan
from subliminal_learning.campaign import xopt2_rules as wr
from subliminal_learning.campaign import xopt2_soft as soft
from subliminal_learning.campaign import xopt2_soft_metrics as metrics
from subliminal_learning.campaign import xopt_soft_metrics as xm
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.run_io import read_run_set_card, utc_now_iso

SPEC = replace(
    op2.SPEC,
    run_set_id=soft.RUN_SET_ID,
    teacher_run_set_id=soft.RUN_SET_ID,
    root_subdir=oplan.RUN_SET_SUBDIR,
    queue_priorities=soft.QUEUE_PRIORITIES,
    gpu_policy={"cap_per_gpu": plan.SOFT_CAP_PER_GPU},
    generated_config_dir=str(soft.GENERATED_DIR),
    campaign_state_file=soft.CAMPAIGN_STATE_FILE,
    campaign_done_file=soft.CAMPAIGN_DONE_FILE,
    extra_pin_paths=(
        "scripts/sweeps/sweep_distill_student_gkd.py",
        "scripts/sweeps/sweep_evaluate_gkd_student.py",
        "scripts/sweeps/sweep_xopt_soft.py",
        "scripts/sweeps/sweep_xopt2_soft.py",
        "scripts/run_xopt2_soft_campaign.py",
        "scripts/stage_xopt2_soft_assets.py",
        "scripts/launch_xopt2_soft_campaign.sh",
        "scripts/jobq_pool_loop.sh",
        soft.TRAINING_CONFIG,
        soft.EXPERIMENT_CONFIG,
        "src/subliminal_learning/campaign/xopt2_plan.py",
        "src/subliminal_learning/campaign/xopt2_rules.py",
        "src/subliminal_learning/campaign/xopt2_soft.py",
        "src/subliminal_learning/campaign/xopt2_soft_metrics.py",
        "src/subliminal_learning/campaign/xopt_soft.py",
        "src/subliminal_learning/campaign/optparam2_jobs.py",
        "src/subliminal_learning/evaluation/pile_probe_io.py",
        "src/subliminal_learning/training/optimizers.py",
        "src/subliminal_learning/training/param_control_io.py",
        "src/subliminal_learning/gkd/trainer_io.py",
        "src/subliminal_learning/analysis/decomposition.py",
        "pipelines/xopt2_soft_campaign_pipeline.py",
        "pipelines/xopt2_soft_results_pipeline.py",
        "pipelines/xopt2_soft_staging_pipeline.py",
    ),
)
PHASES: tuple[str, ...] = ("p1", "p2", "p3")


def driver_config(**kwargs) -> driver_io.CampaignDriverConfig:
    """A `CampaignDriverConfig` bound to the xopt2 soft spec."""
    kwargs.setdefault("generated_dir_override", soft.GENERATED_DIR)
    return driver_io.CampaignDriverConfig(spec=SPEC, **kwargs)


def pin_config(cfg: driver_io.CampaignDriverConfig) -> driver_io.CampaignDriverConfig:
    """`cfg` rooted where the sweeps keep the run-set card (the code pin): the cold-1
    leaf root for the production run set, the driver's own root otherwise (tests)."""
    if Path(cfg.output_root) != soft.OUTPUT_ROOT:
        return cfg
    return replace(cfg, output_root=soft.LEAF_ROOT)


# ---- queues and leaves --------------------------------------------------------------------


def _submitted(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [n for n in soft.QUEUE_PRIORITIES if store.queue_exists(cfg.queue_root, n)]


def _train_submitted(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    return [n for n in _submitted(cfg) if soft.late_dep(n) is None]


def queue_defaults(cfg: driver_io.CampaignDriverConfig) -> dict:
    """jobq defaults binding a queue to every worker host's worktree of the pin."""
    pcfg = pin_config(cfg)
    return plan.queue_defaults({h: node_pin_defaults(pcfg, h) for h in soft.HOSTS})


def legacy_open(cfg: driver_io.CampaignDriverConfig, name: str) -> bool:
    """Whether `name` holds a legacy-guard (goncalo-only) job that is not done."""
    return any(
        job.cmd.startswith(soft.LEGACY_HOST_GUARD)
        and store.job_state(cfg.queue_root, name, job) != store.DONE
        for job in store.load_jobs(cfg.queue_root, name)
    )


def registered(
    cfg: driver_io.CampaignDriverConfig,
) -> dict[str, tuple[soft.SoftLeaf, str]]:
    """Key -> (leaf, train queue) of every leaf in a submitted train queue."""
    out = {}
    for name in _train_submitted(cfg):
        for job in store.load_jobs(cfg.queue_root, name):
            out[job.key] = (soft.leaf_from_key(job.key), name)
    return out


def leaf_status(cfg: driver_io.CampaignDriverConfig) -> dict[str, str]:
    """Key -> ``done`` | ``failed`` (after the requeue pass: for good) of finished leaves."""
    out = {}
    for name in _train_submitted(cfg):
        for res in store.load_results(cfg.queue_root, name):
            out[res["key"]] = "done" if int(res.get("rc", 1)) == 0 else "failed"
    return out


def training_base_doc() -> dict:
    """The pin's GKD student training YAML (base of the frozen-module one)."""
    return yaml.safe_load(Path(soft.TRAINING_CONFIG).read_text())


def write_leaf_yamls(
    cfg: driver_io.CampaignDriverConfig, leaves: Iterable[soft.SoftLeaf]
) -> int:
    """Create (or match) every generated YAML the leaves' jobs read."""
    base = training_base_doc()
    items = {p: d for lf in leaves for p, d in soft.leaf_yaml_docs(lf, base).items()}
    if not cfg.dry_run:
        for path, doc in items.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            write_once_text(path, yaml.safe_dump(doc, sort_keys=True))
    return len(items)


def _submit_or_append(
    cfg: driver_io.CampaignDriverConfig, name: str, jobs: list[Job]
) -> int:
    if not store.queue_exists(cfg.queue_root, name):
        dep = soft.late_dep(name)
        driver_io.submit_queue(
            cfg,
            name,
            jobs,
            depends_on=[dep] if dep else None,
            unpinned=True,
            defaults=queue_defaults(cfg),
        )
        return len(jobs)
    have = {j.jobkey for j in store.load_jobs(cfg.queue_root, name)}
    new = [j for j in jobs if j.jobkey not in have]
    if new and not cfg.dry_run:
        store.append_jobs(cfg.queue_root, name, new)
        logger.info("appended {} jobs to {}", len(new), name)
    return len(new)


def submit_leaves(
    cfg: driver_io.CampaignDriverConfig,
    leaves: Sequence[soft.SoftLeaf],
    route: str = soft.ROUTE_TIER,
    known: set[str] | None = None,
) -> dict[str, int]:
    """Submit the leaves whose identity no queue holds yet; return queue -> jobs added.

    Args:
        cfg: Driver config.
        leaves: Candidate leaves.
        route: `xopt2_soft.ROUTE_*` (the tier's own, walk or down queue).
        known: Identities already in a queue (read from the queues when omitted); the
            submitted identities are added to it.
    """
    if known is None:
        known = {lf.ident for lf, _q in registered(cfg).values()}
    fresh = [lf for lf in leaves if lf.ident not in known]
    if not fresh:
        return {}
    known.update(lf.ident for lf in fresh)
    write_leaf_yamls(cfg, fresh)
    queues = soft.queue_jobs(fresh, route)
    order = sorted(queues, key=lambda n: (soft.late_dep(n) is not None, n))
    return {name: _submit_or_append(cfg, name, queues[name]) for name in order}


def planned_queue_counts(
    gl_complete: frozenset[soft.CellSpec] = frozenset(),
) -> dict[str, int]:
    """Queue -> jobs of the launch submission."""
    jobs = soft.queue_jobs(soft.launch_leaves(gl_complete))
    return {name: len(js) for name, js in sorted(jobs.items())}


# ---- gates --------------------------------------------------------------------------------


def _free_gib(path: Path) -> float:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 2**30


def disk_problems() -> list[str]:
    """Mounts below their `MIN_FREE_GIB` floor (ssd-2 control root, cold-1 leaf root)."""
    out = []
    for mount, path in (("ssd-2", soft.CTRL_ROOT), ("cold-1", soft.LEAF_ROOT)):
        free = _free_gib(path)
        if free < plan.MIN_FREE_GIB[mount]:
            out.append(f"{mount}: {free:.0f} GiB free < {plan.MIN_FREE_GIB[mount]}")
    return out


def _preflight_problems(cfg: driver_io.CampaignDriverConfig) -> list[str]:
    problems = []
    if not (cfg.repo_root / plan.PREREG_PATH).is_file():
        problems.append(f"missing pre-registration {plan.PREREG_PATH}")
    if store.this_host() != plan.HOME_NODE:
        problems.append(f"the driver runs on {plan.HOME_NODE}, not {store.this_host()}")
    if Path(cfg.queue_root).resolve() != soft.QUEUE_ROOT.resolve():
        problems.append(f"queue root {cfg.queue_root} is not the plan's")
    if Path(cfg.output_root) != soft.OUTPUT_ROOT:
        problems.append(f"output root {cfg.output_root} is not the plan's")
    if (
        not soft.PATHS_CONFIG_PATH.is_file()
        or soft.PATHS_CONFIG_PATH.read_text() != staging.paths_yaml_text()
    ):
        problems.append(f"{soft.PATHS_CONFIG_PATH} missing or differs from the plan")
    report = staging.latest_complete_report()
    if report is None:
        problems.append(
            "no complete staging report; run scripts.stage_xopt2_soft_assets"
        )
    elif set(report.get("ce_base", {})) != set(soft.student_tokens()):
        problems.append("staging report CE_base does not cover every student base@step")
    if Path(cfg.output_root) == soft.OUTPUT_ROOT and (bad := pin_problem(cfg)):
        problems.append(bad)
    return problems + disk_problems()


def pin_problem(cfg: driver_io.CampaignDriverConfig) -> str | None:
    """A hard run-set card whose pin differs from the soft one (card, else requested)."""
    hard = read_run_set_card(plan.run_set_dir(plan.HARD)) or {}
    if not hard.get("git_sha"):
        return None
    mine = (read_run_set_card(pin_config(cfg).run_set_root) or {}).get("git_sha")
    mine = mine or cfg.code_sha
    if not mine:
        return "the hard channel is pinned; pass --code-sha with its pin"
    want = code_pin_io.resolve_commit_sha(cfg.repo_root, str(hard["git_sha"]))
    got = code_pin_io.resolve_commit_sha(cfg.repo_root, str(mine))
    if want != got:
        return f"hard channel pinned to {want[:12]}, soft to {got[:12]}"
    return None


def _preflight(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    problems = _preflight_problems(cfg)
    if problems and not cfg.dry_run:
        raise CampaignGateError("; ".join(problems))
    for p in problems:
        logger.warning("[dry-run] preflight would refuse: {}", p)
    report = staging.latest_complete_report()
    _fire(state, "preflight", staging_report_at=(report or {}).get("at"))


def _register(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    if cfg.dry_run:
        sha = code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha or "HEAD")
        _fire(state, "register", git_sha=sha)
        return
    pin = current_pin(pin_config(cfg))
    driver_io.verify_pin_runs_campaign(pin, SPEC)
    _fire(state, "register", git_sha=pin.sha)


def _repin(cfg: driver_io.CampaignDriverConfig) -> str | None:
    """Move every queue with work left (or failures) to the campaign pin, unpinned
    unless it holds a legacy-guard job that is not done (then on goncalo)."""
    pcfg = pin_config(cfg)
    if cfg.dry_run or read_run_set_card(pcfg.run_set_root) is None:
        return None
    want = queue_defaults(cfg)
    n = 0
    for name in _submitted(cfg):
        if (
            store.queue_complete(cfg.queue_root, name)
            and not store.queue_status(cfg.queue_root, name)["failed"]
        ):
            continue
        meta = store.read_meta(cfg.queue_root, name)
        node = plan.GONCALO if legacy_open(cfg, name) else None
        if meta.node == node and all(
            meta.defaults.get(k) == want[k] for k in ("cwd", "nodes")
        ):
            continue
        store.update_queue_defaults(cfg.queue_root, name, want)
        if meta.node != node:
            store.update_queue_node(cfg.queue_root, name, node)
        n += 1
    return f"repin(queues={n})" if n else None


def archive_path(cfg: driver_io.CampaignDriverConfig, name: str, stamp: str) -> Path:
    """Where `reissue_legacy` moves queue `name` (no pool lists a nested dir)."""
    return Path(cfg.queue_root) / "_archived" / f"_archived_{name}_{stamp}"


def reissue_legacy(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Rebuild every queue that holds a legacy-guard job and in which none started.

    Each such queue is built anew under ``_archived/_staging_<UTC>`` (same name,
    priority and dependencies, unpinned, jobs in the same order: legacy ones rebuilt
    by `xopt2_soft.reissued_job`, the rest unchanged), then two renames move the old
    queue to `archive_path` and the new one into its place. A queue with a claim or
    a result keeps its jobs (`_repin` unpins it once no legacy job is undone). The
    plan is logged before the first rename, the outcome after the last.

    Returns:
        ``{"reissued": {queue: n_jobs}, "kept": [queue], "archive": {queue: path}}``.

    Raises:
        CampaignGateError: Off the home node, without a run-set card, a run-set pin
            without the worker-host spread, or a worker host with a soft policy file
            in the queue root and no stop sentinel.
    """
    root = Path(cfg.queue_root)
    problems = []
    if store.this_host() != plan.HOME_NODE:
        problems.append(f"run on {plan.HOME_NODE}, not {store.this_host()}")
    if read_run_set_card(pin_config(cfg).run_set_root) is None:
        problems.append("the run set has no card")
    else:
        pin = current_pin(pin_config(cfg))
        code = pin.root / "src/subliminal_learning/campaign/xopt2_soft.py"
        if "LEGACY_HOST_GUARD" not in code.read_text():
            problems.append(f"pin {pin.sha[:12]} predates the worker-host spread")
    for host in soft.HOSTS:
        policy = root / f"gpu_policy.{host}.json"
        if policy.exists() and not store.stop_path(root, host).exists():
            problems.append(f"{host}'s soft pool is not stopped")
    if problems:
        raise CampaignGateError("; ".join(problems))

    def rebuilt(job: Job) -> Job:
        legacy = job.cmd.startswith(soft.LEGACY_HOST_GUARD)
        return soft.reissued_job(job) if legacy else job

    defaults = queue_defaults(cfg)
    stamp = store.utc_stamp()
    staging = root / "_archived" / f"_staging_{stamp}"
    plans: dict[str, tuple[store.QueueMeta, list[Job]]] = {}
    out: dict = {"reissued": {}, "kept": [], "archive": {}}
    for name in _submitted(cfg):
        jobs = store.load_jobs(root, name)
        if not any(j.cmd.startswith(soft.LEGACY_HOST_GUARD) for j in jobs):
            continue
        if store.queue_status(root, name)["pending"] != len(jobs):
            out["kept"].append(name)
            continue
        plans[name] = (store.read_meta(root, name), [rebuilt(j) for j in jobs])
        out["reissued"][name] = len(jobs)
        out["archive"][name] = str(archive_path(cfg, name, stamp))
    if cfg.dry_run:
        return out
    atomic_write_json(cfg.logs_dir / f"reissue_legacy_{stamp}.plan.json", out)
    for name, (meta, jobs) in plans.items():
        if store.queue_status(root, name)["pending"] != len(jobs):
            out["kept"].append(name)
            del out["reissued"][name], out["archive"][name]
            continue
        store.create_queue(
            staging,
            name,
            depends_on=list(meta.depends_on),
            node=None,
            priority=meta.priority,
            strict_deps=meta.strict_deps,
            defaults=defaults,
        )
        store.append_jobs(staging, name, jobs)
        dest = archive_path(cfg, name, stamp)
        store.queue_dir(root, name).rename(dest)
        store.queue_dir(staging, name).rename(store.queue_dir(root, name))
        logger.info("reissued {} ({} jobs); old queue at {}", name, len(jobs), dest)
    atomic_write_json(cfg.logs_dir / f"reissue_legacy_{stamp}.json", out)
    return out


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


def _finished(cfg: driver_io.CampaignDriverConfig, names: Iterable[str]) -> bool:
    """Every existing queue in `names` complete with no failed job (no queue = nothing
    to wait for)."""
    return all(
        driver_io.queue_finished(cfg, n)[0]
        for n in names
        if store.queue_exists(cfg.queue_root, n)
    )


# ---- walks, extra leaves, carry ---------------------------------------------------------------


def _signature(items: Iterable[object]) -> str:
    body = json.dumps(sorted(str(x) for x in items))
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def _walk_inputs(
    group: soft.WalkGroup, reg: Mapping[str, tuple[soft.SoftLeaf, str]]
) -> list[soft.SoftLeaf]:
    """Leaves a walk decides on: its tier's decision leaves of its cell and arm, plus
    another tier's leaves of equal identity (v at the py self corner)."""
    return [
        lf
        for lf, _q in reg.values()
        if lf.tier in (group.tier, "v")
        and lf.spec == group.cell
        and lf.arm == group.arm
        and lf.animal in group.animals
        and lf.r in metrics.DECISION_RS
    ]


def walk_pass(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    ce: Mapping[str, float],
    reg: dict[str, tuple[soft.SoftLeaf, str]],
    status: Mapping[str, str],
) -> list[str]:
    """Re-decide every walk group whose inputs changed; submit rungs, x and ox leaves.

    Returns:
        One note per action taken this pass.
    """
    walks = state.setdefault("walks", {})
    holes = state.setdefault("holes", {})
    known = {lf.ident for lf, _q in reg.values()}
    notes = []
    for g in soft.walk_groups():
        rec = walks.setdefault(g.key, {"shared": [], "steps": []})
        leaves = _walk_inputs(g, reg)
        sig = _signature(
            [*(f"{lf.key}={status.get(lf.key)}" for lf in leaves), *rec["shared"]]
        )
        if rec.get("signature") == sig:
            continue
        rows = results.decision_rows(leaves, status)
        view = metrics.walk_view(
            g, metrics.walk_scheduled(g, rows, rec["shared"]), xm.group_rates(rows), ce
        )
        rec["status"] = view.status
        holes.pop(f"walk:{g.key}", None)
        if view.holes:
            holes[f"walk:{g.key}"] = (
                f"decision leaf failed for good at {list(view.holes)}"
            )
        elif view.step.action == wr.RUN:
            rate = float(view.step.rate)
            new = g.leaves(rate)
            if any(lf.ident in known for lf in new):
                rec["shared"].append(rate)
            submit_leaves(cfg, new, soft.ROUTE_WALK, known)
            rec["steps"].append(
                {"at": utc_now_iso(), "rate": rate, "reason": view.step.reason}
            )
            notes.append(f"walk({g.key} {view.step.reason} -> {rate:g})")
        elif view.done and "extra" not in rec:
            notes += _finish_walk(cfg, state, g, view, ce, known)
        rec["signature"] = sig
    return notes


def _finish_walk(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    group: soft.WalkGroup,
    view: metrics.WalkView,
    ce: Mapping[str, float],
    known: set[str],
) -> list[str]:
    """Submit a finished walk's tier-x leaves and, for a new arm, its tier-ox carry."""
    rec = state["walks"][group.key]
    extra = metrics.extra_rates(view, ce)
    submit_leaves(cfg, soft.extra_leaves(group, extra), soft.ROUTE_TIER, known)
    rec.update(extra=list(extra), done=view.step.reason, finished_at=utc_now_iso())
    if view.step.reason == "degenerate":
        state["labelled"][f"degenerate|{group.key}"] = "degenerate"
    notes = [f"walk_done({group.key} {view.step.reason}; x at {list(extra)})"]
    if group.tier != "o":
        return notes
    anc = metrics.anchor(view, ce)
    rec["anchor"] = {
        "rate": anc.rate,
        "null": anc.null,
        "carry": list(anc.carry),
        "reason": anc.reason,
    }
    if not anc.carry:
        state["labelled"][f"carry|{group.arm}"] = anc.reason
        return [*notes, f"no_carry({group.arm})"]
    submit_leaves(cfg, soft.carry_leaves(group.arm, anc.carry), soft.ROUTE_TIER, known)
    return [*notes, f"carry({group.arm} -> {list(anc.carry)})"]


def stepdown_pass(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    ce: Mapping[str, float],
    reg: dict[str, tuple[soft.SoftLeaf, str]],
    status: Mapping[str, str],
) -> list[str]:
    """Re-check every finished fixed-rate group (ox, g) under `xopt`'s step-down.

    Returns:
        One note per decision taken this pass.
    """
    groups: dict[tuple, list[tuple[soft.SoftLeaf, str]]] = defaultdict(list)
    for lf, q in reg.values():
        if lf.tier in soft.DOWN_TIERS:
            groups[(lf.tier, soft.cell_tuple(lf), lf.arm)].append((lf, q))
    seen = state.setdefault("stepdowns", {})
    labelled = state.setdefault("labelled", {})
    known = {lf.ident for lf, _q in reg.values()}
    notes = []
    for (tier, cell, arm), members in sorted(groups.items()):
        leaves = [lf for lf, _q in members]
        if any(status.get(lf.key) != "done" for lf in leaves):
            continue
        gkey = f"{tier}|{'|'.join(map(str, cell))}|{arm}"
        sig = _signature(lf.key for lf in leaves)
        if seen.get(gkey, {}).get("signature") == sig:
            continue
        rows = results.decision_rows(leaves, status)
        rates = sorted(xm.group_rates(rows).values(), key=lambda r: -r.lr)
        base = ce[ws160m_models.token(cell[3], cell[4])]
        initial = sorted(
            {lf.lr for lf, q in members if q == soft.train_queue(tier)}, reverse=True
        )
        if not initial:
            continue
        new = metrics.fixed_next_rate(arm, rates, initial, base)
        decision = {
            "signature": sig,
            "at": utc_now_iso(),
            "rates": [r.lr for r in rates],
        }
        if new is not None:
            lowest = min(lf.lr for lf in leaves)
            rung = [replace(lf, lr=new) for lf in leaves if lf.lr == lowest]
            submit_leaves(cfg, rung, soft.ROUTE_DOWN, known)
            depth = plan.SOFT_ARMS[arm].lower_rungs(min(initial)).index(new) + 1
            decision.update(next_rate=new, depth=depth, n_leaves=len(rung))
            notes.append(f"stepdown({gkey} -> {new:g}, depth {depth})")
        elif xm.selected(rates, base) is None:
            labelled[f"degenerate|{gkey}"] = "degenerate"
            notes.append(f"degenerate({gkey})")
        else:
            labelled.pop(f"degenerate|{gkey}", None)
        seen[gkey] = decision
    return notes


# ---- phases and results ------------------------------------------------------------------


def walks_finished(state: dict) -> bool:
    """Every walk group done (x / ox handled) and no walk hole."""
    walks = state.get("walks", {})
    holes = state.get("holes", {})
    return not any(k.startswith("walk:") for k in holes) and all(
        "extra" in walks.get(g.key, {}) for g in soft.walk_groups()
    )


def phase_ready(cfg: driver_io.CampaignDriverConfig, state: dict, phase: str) -> bool:
    """Whether `phase`'s results may be written (earlier phases already written)."""
    done = state["transitions"]
    if "submit_launch" not in done:
        return False
    if not walks_finished(state):
        return False
    if phase == "p1":
        return _finished(cfg, soft.phase_queues("p1"))
    if phase == "p2":
        return "results_p1" in done and _finished(cfg, soft.phase_queues("p2"))
    return "results_p2" in done and _finished(cfg, _submitted(cfg))


def anchors_of(state: dict) -> dict[str, dict]:
    """New arm -> its carried anchor doc (finished o walks)."""
    walks = state.get("walks", {})
    return {
        g.arm: walks[g.key]["anchor"]
        for g in soft.walk_groups()
        if "anchor" in walks.get(g.key, {})
    }


def _write_results(
    cfg: driver_io.CampaignDriverConfig,
    state: dict,
    ce: Mapping[str, float],
    phase: str,
) -> dict:
    """Tables + claims over the registered leaves of phases up to `phase`."""
    upto = PHASES[: PHASES.index(phase) + 1]
    status = leaf_status(cfg)
    leaves = sorted(
        (lf for lf, _q in registered(cfg).values() if soft.PHASE[lf.tier] in upto),
        key=lambda lf: lf.key,
    )
    extra = {k: v.get("shared", []) for k, v in state.get("walks", {}).items()}
    return results.write_phase(
        cfg.run_set_root / "results",
        phase,
        results.leaf_rows(leaves, status),
        ce,
        extra,
        anchors_of(state),
        results.reference_rows(),
    )


def write_marker(marker: Path) -> bool:
    """Create a done marker (create-only); False when it already exists."""
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        with marker.open("x") as f:
            f.write(utc_now_iso() + "\n")
    except FileExistsError:
        return False
    return True


def write_campaign_marker(other: Path, marker: Path) -> bool:
    """Create the campaign done marker once the other channel's marker exists."""
    return other.exists() and write_marker(marker)


def _finalize(cfg: driver_io.CampaignDriverConfig, state: dict) -> None:
    write_marker(cfg.logs_dir / soft.CAMPAIGN_DONE_FILE)
    wrote = False
    if cfg.logs_dir == plan.logs_dir(plan.SOFT):
        wrote = write_campaign_marker(
            plan.channel_done_path(plan.HARD), plan.CAMPAIGN_DONE_PATH
        )
    _fire(state, "finalize", campaign_marker=wrote, labelled=sorted(state["labelled"]))


def _gl_reuse_doc(complete: frozenset[soft.CellSpec]) -> dict:
    cells = soft.cross_cells()
    return {
        "at": utc_now_iso(),
        "complete": [list(c) for c in cells if c in complete],
        "missing": [list(c) for c in cells if c not in complete],
    }


# ---- the pass -----------------------------------------------------------------------------


def run(cfg: driver_io.CampaignDriverConfig) -> AdvanceResult:
    """One advance pass."""
    if not cfg.dry_run:
        driver_io.ensure_campaign_code_pin(pin_config(cfg))
    state = driver_io.load_state(cfg)
    done = state["transitions"]
    labelled = state.setdefault("labelled", {})
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
    report = staging.latest_complete_report()
    ce = {k: float(v) for k, v in (report or {}).get("ce_base", {}).items()}
    if note := _repin(cfg):
        fired.append(note)
    if not cfg.dry_run:
        _requeue_and_holes(cfg, state)
    holes = state.setdefault("holes", {})
    disk = [] if cfg.dry_run else disk_problems()
    if disk:
        labelled["disk_low"] = "; ".join(disk)
        waiting.append(f"new submissions held: {labelled['disk_low']}")
    else:
        labelled.pop("disk_low", None)
    if "submit_launch" not in done and not disk:
        if "gl_reuse" not in state:
            state["gl_reuse"] = _gl_reuse_doc(results.gl_complete_cells())
        gl = frozenset(tuple(c) for c in state["gl_reuse"]["complete"])
        added = submit_leaves(cfg, soft.launch_leaves(gl))
        _fire(state, "submit_launch", n_jobs=sum(added.values()), queues=added)
        mark("submit_launch")
    if cfg.dry_run:
        waiting.append("dry-run: walks, step-downs, results and finalize not evaluated")
        return AdvanceResult(tuple(fired), tuple(waiting), False)
    if ce and not disk and "submit_launch" in done:
        reg, status = registered(cfg), leaf_status(cfg)
        notes = walk_pass(cfg, state, ce, reg, status)
        notes += stepdown_pass(cfg, state, ce, registered(cfg), status)
        if notes:
            fired.extend(notes)
            driver_io.save_state(cfg, state)
    for phase in PHASES:
        name = f"results_{phase}"
        if name in done or disk or not phase_ready(cfg, state, phase):
            continue
        manifest = _write_results(cfg, state, ce, phase)
        _fire(
            state,
            name,
            suffix=manifest["suffix"],
            rebuilt=manifest["suffix"] != f"_{phase}",
        )
        mark(name)
    all_done = False
    if "results_p3" in done and holes:
        waiting.append(f"finalize: holes in {sorted(holes)}")
    elif "results_p3" in done:
        if "finalize" not in done:
            _finalize(cfg, state)
            mark("finalize")
        all_done = True
    elif "results_p2" in done:
        waiting.append("results_p3: queues not finished")
    if holes:
        waiting.append(f"holes in {sorted(holes)}")
    state["last_pass"] = {"at": utc_now_iso(), "fired": fired, "waiting": waiting[:50]}
    driver_io.save_state(cfg, state)
    return AdvanceResult(tuple(fired), tuple(waiting), all_done)


def status(cfg: driver_io.CampaignDriverConfig) -> dict:
    """Transitions, holes, walk states, step-downs and per-queue counts (file reads)."""
    state = driver_io.load_state(cfg)
    queues = {}
    for name in _submitted(cfg):
        s = store.queue_status(cfg.queue_root, name)
        queues[name] = {k: v for k, v in s.items() if k != "running_jobs"}
    walks = state.get("walks", {})
    return {
        "transitions": sorted(state.get("transitions", {})),
        "holes": state.get("holes", {}),
        "walks": {k: v.get("status") for k, v in sorted(walks.items())},
        "anchors": {k: v["anchor"] for k, v in walks.items() if "anchor" in v},
        "labelled": state.get("labelled", {}),
        "gl_reuse": {
            k: len(v) for k, v in state.get("gl_reuse", {}).items() if k != "at"
        },
        "n_stepdowns": sum(
            1 for d in state.get("stepdowns", {}).values() if "next_rate" in d
        ),
        "queues": queues,
    }


def write_policy(host: str, *, replace_existing: bool = False) -> Path:
    """Write `host`'s jobq policy file into the soft queue root (create or equal).

    Raises:
        FileExistsError: A different policy is on file and `replace_existing` is off.
    """
    path = soft.QUEUE_ROOT / f"gpu_policy.{host}.json"
    doc = soft.gpu_policy_json(host)
    if path.is_file() and not replace_existing:
        if json.loads(path.read_text()) != doc:
            raise FileExistsError(f"{path} exists with a different policy")
        return path
    atomic_write_json(path, doc)
    return path
