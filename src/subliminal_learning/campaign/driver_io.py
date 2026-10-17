"""Campaign driver IO: state file, generated-config writes, jobq submission.

Everything a campaign stage machine (pipelines/student_hp_campaign_pipeline.py,
pipelines/optparam_campaign_pipeline.py) needs that touches disk. Submission goes
straight through `jobqueue.store` (the same functions `scripts/jobq.py submit`
wraps), so the driver gets the store's duplicate-jobkey and exist_ok guarantees
without shelling out.

`CampaignDriverConfig` and the submission/state helpers are campaign-agnostic --
`run_set_id` and `queue_priorities` stay in lockstep with `spec` via
`__post_init__` (default `plan.SPEC`). The `a1_jobs`-style builders below belong
to the student-HP campaign; `campaign.optparam_jobs` holds the 20260801 ones.

Idempotency contract: `submit_queue` is safe to re-run — an existing queue with
the exact same jobkey set is a no-op, a mismatched jobkey set raises (the queue
was created by something else; never silently mix campaigns).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger

from subliminal_learning.campaign import (
    breadth_plan,
    ckpt_plan,
    ckptgrid2_plan,
    ckptgrid_plan,
    offxsize_plan,
    opdlim_plan,
    plan,
    strictscreen_plan,
    tmplopd_plan,
    topk_plan,
    ws160m_plan,
    xfam2_plan,
    xfam_plan,
    xsize_plan,
)
from subliminal_learning.campaign.selection import (
    FrozenRecipe,
    Recipe,
    RecipeSelection,
)
from subliminal_learning.campaign.spec import CampaignSpec
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import read_run_set_card, reconcile_run_set_sha

TRAIN_MODULE = "scripts.sweeps.sweep_distill_student_gkd"
EVAL_MODULE = "scripts.sweeps.sweep_evaluate_gkd_student"

# CLI campaign token -> spec (one SSOT for every campaign entrypoint).
CAMPAIGN_SPECS: Mapping[str, CampaignSpec] = {
    "slbreadth": plan.SPEC,
    "breadth5": breadth_plan.SPEC,
    "ckptladder": ckpt_plan.SPEC,
    "ckptgrid": ckptgrid_plan.SPEC,
    "ckptgrid2": ckptgrid2_plan.SPEC,
    "xsize": xsize_plan.SPEC,
    "offxsize": offxsize_plan.SPEC,
    "topk": topk_plan.SPEC,
    "xfam": xfam_plan.SPEC,
    "xfam2": xfam2_plan.SPEC,
    "opdlim1": opdlim_plan.SPEC_T1,
    "opdlim2": opdlim_plan.SPEC_T2,
    "strictscreen": strictscreen_plan.SPEC,
    "tmplopd": tmplopd_plan.SPEC,
    "ws160m-main": ws160m_plan.SPECS["main"],
    "ws160m-ladder": ws160m_plan.SPECS["ladder"],
    "ws160m-dosehp": ws160m_plan.SPECS["dosehp"],
    "ws160m-extra": ws160m_plan.SPECS["extra"],
    "ws160m2-matrix": ws160m_plan.SPECS["matrix"],
    "ws160m2-grid": ws160m_plan.SPECS["grid"],
}


def spec_for_campaign(name: str) -> CampaignSpec:
    """The named campaign's spec.

    Raises:
        ValueError: Unknown campaign token (listing the valid ones) — never fall
            back to a default spec, which would drive the wrong run-set tree.
    """
    if name not in CAMPAIGN_SPECS:
        raise ValueError(
            f"unknown campaign {name!r}; choose one of {sorted(CAMPAIGN_SPECS)}"
        )
    return CAMPAIGN_SPECS[name]


@dataclass(frozen=True)
class CampaignDriverConfig:
    """Everything the campaign stage machine needs.

    Attributes:
        output_root: Artifact root from the paths YAML.
        queue_root: jobq root shared by all campaign queues.
        repo_root: Main git checkout (parent of ``.code_pins``). Jobs get
            ``cwd`` from the resolved MLflow-style code pin, not this path.
        include_stage_c: Also generate + submit the optional Stage C ablations.
        require_a0: Gate the first submission on the A0 calibration +
            init-equality reports (disable only for tests).
        auto_reconcile_sha: When ``code_sha`` differs from the card pin, record
            the advance instead of halting (intentional re-pin only — workspace
            HEAD moves are ignored).
        dry_run: Log every planned action; write/submit nothing.
        generated_dir_override: Where generated YAMLs go (tests only; default
            ``{repo_root}/{spec.generated_config_dir}``).
        run_set_id: Campaign id whose output subtree the driver reads and writes.
        queue_priorities: Queue name -> jobq priority for this campaign's queues.
        spec: Frozen campaign constants (default: `plan.SPEC`).
        auto_requeue: Requeue failed jobs whose attempt count is below this
            cap before gating (0 = halt on first failure, today's behavior).
        code_sha: Optional explicit pin (``mlflow run --version`` analog).
    """

    output_root: Path
    queue_root: Path
    repo_root: Path
    include_stage_c: bool = False
    require_a0: bool = True
    auto_reconcile_sha: bool = False
    dry_run: bool = False
    generated_dir_override: Path | None = None
    run_set_id: str = plan.RUN_SET_ID
    queue_priorities: Mapping[str, int] = field(
        default_factory=lambda: dict(plan.QUEUE_PRIORITIES)
    )
    spec: CampaignSpec = plan.SPEC
    auto_requeue: int = 0
    code_sha: str | None = None

    def __post_init__(self) -> None:
        # Keep identity + priorities aligned with the active CampaignSpec so
        # callers that only swap `spec=` (breadth / future campaigns) cannot
        # silently write into the default plan's run-set tree.
        object.__setattr__(self, "run_set_id", self.spec.run_set_id)
        object.__setattr__(self, "queue_priorities", dict(self.spec.queue_priorities))

    @property
    def campaign_subdir(self) -> str:
        return run_set_subdir(self.spec.root_subdir, self.run_set_id)

    @property
    def run_set_root(self) -> Path:
        return self.output_root / self.campaign_subdir

    @property
    def logs_dir(self) -> Path:
        return self.output_root / self.campaign_subdir / "logs"

    @property
    def state_path(self) -> Path:
        return self.logs_dir / self.spec.campaign_state_file

    @property
    def generated_dir(self) -> Path:
        if self.generated_dir_override is not None:
            return self.generated_dir_override
        return self.repo_root / self.spec.generated_config_dir


# ---- state file -----------------------------------------------------------------


def load_state(cfg: CampaignDriverConfig) -> dict:
    """The driver's persisted state (``{}`` before the first transition)."""
    if not cfg.state_path.exists():
        return {"transitions": {}}
    state = json.loads(cfg.state_path.read_text())
    state.setdefault("transitions", {})
    return state


def save_state(cfg: CampaignDriverConfig, state: dict) -> None:
    if cfg.dry_run:
        return
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(cfg.state_path, state)


def recipe_to_dict(r: Recipe) -> dict:
    return {"lora_r": r.lora_r, "lora_alpha": r.lora_alpha, "lr": r.lr}


def recipe_from_dict(d: dict) -> Recipe:
    return Recipe(int(d["lora_r"]), int(d["lora_alpha"]), float(d["lr"]))


def candidates_to_state(
    cands: dict[tuple[str, float], Sequence[Recipe]],
) -> dict:
    return {
        f"{animal}|{beta:g}": [recipe_to_dict(r) for r in recipes]
        for (animal, beta), recipes in cands.items()
    }


def candidates_from_state(d: dict) -> dict[tuple[str, float], list[Recipe]]:
    out: dict[tuple[str, float], list[Recipe]] = {}
    for key, recipes in d.items():
        animal, beta = key.split("|")
        out[(animal, float(beta))] = [recipe_from_dict(r) for r in recipes]
    return out


def selection_to_state(
    selections: dict[tuple[str, float], RecipeSelection],
) -> dict:
    """Full A1 selection provenance: both rankings, both top-k lists, the union.

    Persisted so the freeze is auditable and the notebooks can plot why each arm
    was laddered without re-reading every probe trajectory.
    """
    return {
        f"{animal}|{beta:g}": {
            "recipes": [recipe_to_dict(r) for r in sel.recipes],
            "by_final": [recipe_to_dict(r) for r in sel.by_final],
            "by_peak": [recipe_to_dict(r) for r in sel.by_peak],
            "ranking": [dict(rec) for rec in sel.ranking],
        }
        for (animal, beta), sel in sorted(selections.items())
    }


def frozen_to_state(frozen: dict[float, FrozenRecipe]) -> dict:
    return {
        f"{beta:g}": {
            "recipe": recipe_to_dict(f.recipe),
            "num_epochs": f.num_epochs,
            "subliminal_delta": f.subliminal_delta,
            "per_animal_delta": f.per_animal_delta,
        }
        for beta, f in frozen.items()
    }


def write_logs_report(cfg: CampaignDriverConfig, filename: str, doc: dict) -> Path:
    """Persist a stage report under the campaign ``logs/`` dir (notebook input)."""
    path = cfg.logs_dir / filename
    if cfg.dry_run:
        logger.info("[dry-run] would write {}", path)
        return path
    cfg.logs_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, doc)
    return path


# ---- generated configs ------------------------------------------------------------


def write_generated_configs(
    cfg: CampaignDriverConfig, configs: dict[str, dict]
) -> list[Path]:
    """Atomically write ``{filename: yaml_dict}`` under the generated-config dir.

    Deterministic re-generation overwrites byte-identically, so re-running a
    transition never clobbers meaningfully different content silently — if the
    content WOULD change (code or selection changed between runs), raise.
    """
    written: list[Path] = []
    for fname, doc in sorted(configs.items()):
        path = cfg.generated_dir / fname
        body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=None)
        if path.exists() and path.read_text() != body:
            raise RuntimeError(
                f"generated config {path} exists with different content — "
                "selection or generator changed since it was written; refusing "
                "to clobber (delete the stale file to regenerate)"
            )
        if cfg.dry_run:
            logger.info("[dry-run] would write {}", path)
        elif not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(path, body)
        written.append(path)
    return written


# ---- jobq submission ---------------------------------------------------------------


def queue_job_defaults(cfg: CampaignDriverConfig) -> dict:
    """jobq defaults: run every cell under the campaign code pin (not workspace HEAD)."""
    pin = code_pin_io.resolve_campaign_pin(
        cfg.run_set_root,
        repo_root=cfg.repo_root,
        requested_sha=cfg.code_sha,
        allow_advance=False,
    )
    return code_pin_io.pin_job_defaults(pin)


def pin_required_paths(spec: CampaignSpec) -> list[str]:
    """Repo-relative paths a pin must contain to run `spec`'s job command lines."""
    required = [
        spec.a1_same_yaml,
        spec.a1_cross_yaml,
        spec.a1_same_verify_yaml,
        spec.a1_cross_verify_yaml,
        f"{TRAIN_MODULE.replace('.', '/')}.py",
        f"{EVAL_MODULE.replace('.', '/')}.py",
    ]
    if spec.gkd_experiment_config:
        required.append(spec.gkd_experiment_config)
    required.extend(spec.extra_pin_paths)
    return [path for path in required if path]


def verify_pin_runs_campaign(pin: code_pin_io.CodePin, spec: CampaignSpec) -> None:
    """Refuse a pin whose worktree cannot run `spec`.

    Cells execute with ``cwd`` inside ``.code_pins/<sha>/``, so a pin taken from a
    commit that predates the campaign (the usual cause: campaign code still
    uncommitted at first submit) yields a tree without this campaign's static
    sweep YAMLs. Every job would then fail on a missing config — thousands of
    identical failures instead of one refusal here.

    Raises:
        ValueError: The pinned tree is missing files this campaign's job command
            lines name.
    """
    missing = [rel for rel in pin_required_paths(spec) if not (pin.root / rel).exists()]
    if missing:
        raise ValueError(
            f"code pin {pin.sha[:12]} at {pin.root} cannot run campaign "
            f"{spec.run_set_id}: missing {missing}. Commit the campaign code, "
            "then pin the new commit (`--code-sha <sha>` for a first launch, or "
            "`scripts.reconcile_run_set --new-git-sha <sha>` to re-pin an "
            "existing campaign)."
        )


def ensure_campaign_code_pin(cfg: CampaignDriverConfig) -> str | None:
    """Ensure the campaign pin worktree; optionally advance when ``code_sha`` differs.

    Workspace HEAD is never consulted for drift. Returns a short note when a
    pin advance was recorded, else ``None``.

    Raises:
        ValueError: Pin cannot be resolved/ensured, the pinned tree cannot run
            this campaign, or ``code_sha`` differs without ``auto_reconcile_sha``.
    """
    # The pin overlays this dir as a symlink, so it must exist before the pin is
    # created: A3 / Stage-B jobs name generated YAMLs while running inside the pin.
    if not cfg.dry_run:
        cfg.generated_dir.mkdir(parents=True, exist_ok=True)
    card = read_run_set_card(cfg.run_set_root)
    if card is None or not card.get("git_sha"):
        pin = code_pin_io.resolve_campaign_pin(
            cfg.run_set_root,
            repo_root=cfg.repo_root,
            requested_sha=cfg.code_sha,
        )
        verify_pin_runs_campaign(pin, cfg.spec)
        logger.info("code pin ready {} @ {}", pin.sha[:12], pin.root)
        return None

    card_sha = str(card["git_sha"])
    card_full = code_pin_io.resolve_commit_sha(cfg.repo_root, card_sha)
    if cfg.code_sha:
        req_full = code_pin_io.resolve_commit_sha(cfg.repo_root, cfg.code_sha)
        if req_full != card_full:
            if not cfg.auto_reconcile_sha:
                raise ValueError(
                    f"requested code pin {req_full} differs from campaign pin "
                    f"{card_full} — pass --auto-reconcile-sha to advance the pin "
                    "intentionally, or omit --code-sha to keep the campaign pin. "
                    "Workspace HEAD moves do not require reconcile."
                )
            verify_pin_runs_campaign(
                code_pin_io.ensure_code_pin(cfg.repo_root, req_full), cfg.spec
            )
            if cfg.dry_run:
                return f"would reconcile code pin {card_full[:12]} -> {req_full[:12]}"
            reconcile_run_set_sha(
                cfg.run_set_root,
                new_git_sha=req_full,
                reason="campaign driver --code-sha --auto-reconcile-sha",
                repo_root=cfg.repo_root,
            )
            return f"reconciled code pin {card_full[:12]} -> {req_full[:12]}"

    pin = code_pin_io.ensure_code_pin(cfg.repo_root, card_full)
    verify_pin_runs_campaign(pin, cfg.spec)
    logger.debug("code pin ok {} @ {}", pin.sha[:12], pin.root)
    return None


def submit_queue(
    cfg: CampaignDriverConfig,
    name: str,
    jobs: list[Job],
    *,
    depends_on: list[str] | None = None,
    node: str | None = None,
    defaults: dict | None = None,
    strict_deps: bool = False,
    unpinned: bool = False,
) -> str:
    """Create queue `name` (priority from the plan) and append `jobs`, idempotently.

    Queues are pinned to the submitting host unless `node` names another one, in
    which case `defaults` must carry that node's code-pin ``cwd``/env. With
    `unpinned` the queue has no node (every pool on its own queue root drains it;
    `defaults` must then hold a ``cwd`` valid on every such node). Job ``cwd`` /
    env bind workers to the campaign's ``.code_pins/<sha>/`` worktree (MLflow
    ``--version`` analog) so the operator's editable HEAD can move without
    changing cell code. With `strict_deps` a dependency that finished with failed
    jobs blocks this queue until they are requeued and done.

    Returns:
        ``"submitted"`` | ``"exists"`` (identical jobkey set — no-op).

    Raises:
        RuntimeError: The queue exists with a DIFFERENT jobkey set.
    """
    if not jobs:
        raise ValueError(f"refusing to submit empty queue {name!r}")
    priority = cfg.queue_priorities[name]
    defaults = defaults if defaults is not None else queue_job_defaults(cfg)
    if store.queue_exists(cfg.queue_root, name):
        existing = {j.jobkey for j in store.load_jobs(cfg.queue_root, name)}
        incoming = {j.jobkey for j in jobs}
        if existing == incoming:
            store.update_queue_defaults(cfg.queue_root, name, defaults)
            logger.info("queue {} already submitted ({} jobs) — skip", name, len(jobs))
            return "exists"
        raise RuntimeError(
            f"queue {name!r} exists with a different job set "
            f"({len(existing)} existing vs {len(incoming)} planned) — refusing to mix"
        )
    if unpinned and node is not None:
        raise ValueError(f"queue {name!r}: unpinned excludes node={node!r}")
    node = None if unpinned else node or store.this_host()
    if cfg.dry_run:
        logger.info(
            "[dry-run] would submit queue {} (prio {}, node {}, deps {}, strict {}, "
            "{} jobs) cwd={}",
            name,
            priority,
            node,
            depends_on or [],
            strict_deps,
            len(jobs),
            defaults.get("cwd"),
        )
        return "submitted"
    store.create_queue(
        cfg.queue_root,
        name,
        depends_on=depends_on or [],
        node=node,
        priority=priority,
        strict_deps=strict_deps,
        defaults=defaults,
    )
    store.append_jobs(cfg.queue_root, name, jobs)
    logger.info(
        "submitted queue {} (prio {}, node {}, deps {}, {} jobs) cwd={}",
        name,
        priority,
        node,
        depends_on or [],
        len(jobs),
        defaults.get("cwd"),
    )
    return "submitted"


def requeue_failed_under_cap(cfg: CampaignDriverConfig, name: str) -> int:
    """Requeue failed jobs whose attempt is still below ``cfg.auto_requeue``.

    Returns:
        Number of jobs requeued. ``0`` when the flag is off, the queue is
        missing, or every failure is already at the cap.

    Raises:
        RuntimeError: A failed job's attempt is already at/above the cap.
    """
    if cfg.auto_requeue <= 0 or cfg.dry_run:
        return 0
    if not store.queue_exists(cfg.queue_root, name):
        return 0
    over: list[tuple[str, int]] = []
    under: list[tuple[str, int]] = []
    for job in store.load_jobs(cfg.queue_root, name):
        if store.job_state(cfg.queue_root, name, job) != store.FAILED:
            continue
        attempt = store.read_attempt(cfg.queue_root, name, job.jobkey)
        if attempt < cfg.auto_requeue:
            under.append((job.jobkey, attempt))
        else:
            over.append((job.jobkey, attempt))
    for jk, att in under:
        logger.warning(
            "auto-requeue queue={} jobkey={} attempt={} (cap={})",
            name,
            jk,
            att,
            cfg.auto_requeue,
        )
    n = len(
        store.requeue(
            cfg.queue_root, name, failed_only=True, jobkeys=[jk for jk, _ in under]
        )
    )
    if over:
        jk, att = over[0]
        raise RuntimeError(
            f"{name}: job {jk} failed after attempt={att} "
            f"(--auto-requeue {cfg.auto_requeue}); fix the cell or raise the cap, "
            f"then `uv run python -m scripts.jobq requeue --root {cfg.queue_root} "
            f"--name {name}`"
        )
    return n


def queue_finished(cfg: CampaignDriverConfig, name: str) -> tuple[bool, str]:
    """(all-jobs-terminal-and-none-failed, human status) for one queue."""
    if not store.queue_exists(cfg.queue_root, name):
        return False, f"{name}: not submitted"
    status = store.queue_status(cfg.queue_root, name)
    if status["failed"]:
        return False, (
            f"{name}: {status['failed']} FAILED job(s) — requeue with "
            f"`uv run python -m scripts.jobq requeue --root {cfg.queue_root} "
            f"--name {name}`"
        )
    if not store.queue_complete(cfg.queue_root, name):
        return False, (
            f"{name}: {status['pending']} pending / {status['running']} running "
            f"of {status['total']}"
        )
    return True, f"{name}: complete ({status['done']} done)"


# ---- job builders -------------------------------------------------------------------


def train_cmd(
    yaml_rel: str,
    *shard_flags: str,
    experiment_config: str | None = None,
) -> str:
    """One train-sweep shard command (jobq pins the device via CUDA_VISIBLE_DEVICES).

    Args:
        yaml_rel: Repo-relative sweep YAML path.
        *shard_flags: Extra CLI flags (animals, beta, …).
        experiment_config: Optional ``--experiment-config`` path (W&B project
            override for breadth5); None keeps the sweep default.
    """
    flags = " ".join(shard_flags)
    cmd = (
        f"uv run python -m {TRAIN_MODULE} --sweep-config {yaml_rel} "
        f"--device cuda:0{' ' + flags if flags else ''}"
    )
    if experiment_config:
        cmd = f"{cmd} --experiment-config {experiment_config}"
    return cmd


def eval_cmd(yaml_rel: str, *shard_flags: str) -> str:
    """One verify-sweep shard command."""
    flags = " ".join(shard_flags)
    return (
        f"uv run python -m {EVAL_MODULE} --sweep-config {yaml_rel} "
        f"--device cuda:0{' ' + flags if flags else ''}"
    )


def generated_rel(fname: str, *, spec: CampaignSpec | None = None) -> str:
    """Repo-relative path of a generated sweep YAML (jobs run with cwd=repo_root)."""
    s = plan.SPEC if spec is None else spec
    return f"{s.generated_config_dir}/{fname}"


def _students(doc: dict) -> list[str]:
    """Distinct student base labels of a generated YAML's pair list, in file order."""
    return list(dict.fromkeys(str(p[0]) for p in doc["teacher_student_base_pairs"]))


def a1_jobs(*, spec: CampaignSpec | None = None) -> list[Job]:
    """Single-cell A1 shards: (arm, animal, beta, lr, student mode), control-free.

    One cell per job is the finest disjoint partition the sweep CLI exposes here, so the
    screen load-balances perfectly across the pool and a crashed shard costs one cell
    instead of nine.
    """
    s = plan.SPEC if spec is None else spec
    return [
        Job(
            key=f"a1|{arm}|{animal}|b{beta:g}|lr{lr:g}|{mode}",
            cmd=train_cmd(
                yaml_rel,
                "--no-control",
                f"--animals {animal}",
                f"--beta {beta}",
                f"--lora-lrs {lr}",
                f"--student-modes {mode}",
                experiment_config=s.gkd_experiment_config,
            ),
        )
        for arm, yaml_rel in (
            ("same", s.a1_same_yaml),
            ("cross", s.a1_cross_yaml),
        )
        for animal in s.animals
        for beta in s.betas
        for lr in s.a1_lora_lrs
        for mode in s.a1_student_modes
    ]


def a2_jobs(*, spec: CampaignSpec | None = None) -> list[Job]:
    """Single-control A2 shards: the same-init YAML at the control mode."""
    s = plan.SPEC if spec is None else spec
    return [
        Job(
            key=f"a2|{animal}|b{beta:g}|lr{lr:g}",
            cmd=train_cmd(
                s.a1_same_yaml,
                "--controls-only",
                f"--student-modes {s.a2_student_mode}",
                f"--animals {animal}",
                f"--beta {beta}",
                f"--lora-lrs {lr}",
                experiment_config=s.gkd_experiment_config,
            ),
        )
        for animal in s.animals
        for beta in s.betas
        for lr in s.a1_lora_lrs
    ]


def a1_eval_jobs(*, spec: CampaignSpec | None = None) -> list[Job]:
    """Verify shards: (arm, animal, beta, student mode)."""
    s = plan.SPEC if spec is None else spec
    return [
        Job(
            key=f"a1-eval|{arm}|{animal}|b{beta:g}|{mode}",
            cmd=eval_cmd(
                yaml_rel,
                f"--animals {animal}",
                f"--beta {beta}",
                f"--student-modes {mode}",
            ),
        )
        for arm, yaml_rel in (
            ("same", s.a1_same_verify_yaml),
            ("cross", s.a1_cross_verify_yaml),
        )
        for animal in s.animals
        for beta in s.betas
        for mode in s.a1_student_modes
    ]


def backlog_jobs(*, spec: CampaignSpec | None = None) -> list[Job]:
    """Filler shards: the catswide eval backlog, one per (verify YAML, base model).

    Returns an empty list when the spec has no backlog YAMLs/models (breadth
    isolation); callers must skip submit rather than create an empty queue.
    """
    s = plan.SPEC if spec is None else spec
    if not s.catswide_verify_yamls or not s.catswide_base_models:
        return []
    return [
        Job(
            key=(
                f"backlog|"
                f"{Path(yaml_rel).stem.removeprefix('gkd_student_verify_sweep_lora_')}"
                f"|{model}"
            ),
            cmd=eval_cmd(yaml_rel, f"--base-models {model}"),
        )
        for yaml_rel in s.catswide_verify_yamls
        for model in s.catswide_base_models
    ]


def a3_jobs(
    train_configs: dict[str, dict],
    headline_names: list[str],
    *,
    spec: CampaignSpec | None = None,
) -> list[Job]:
    """Single-cell A3 shards: every train YAML x gseed, plus headline controls.

    Treatments always pass ``--no-control``; the headline (``r1``) arms get their own
    controls-only shards so a control never rides along with — and serializes behind —
    a treatment shard.
    """
    s = plan.SPEC if spec is None else spec
    jobs: list[Job] = []
    for fname in sorted(train_configs):
        stem = Path(fname).stem.removeprefix("gkd_a3_")
        for gseed in s.a3_gseeds:
            jobs.append(
                Job(
                    key=f"a3|{stem}|g{gseed}",
                    cmd=train_cmd(
                        generated_rel(fname, spec=s),
                        "--no-control",
                        f"--gseeds {gseed}",
                        experiment_config=s.gkd_experiment_config,
                    ),
                )
            )
            if fname in headline_names:
                jobs.append(
                    Job(
                        key=f"a3-ctrl|{stem}|g{gseed}",
                        cmd=train_cmd(
                            generated_rel(fname, spec=s),
                            "--controls-only",
                            f"--gseeds {gseed}",
                            experiment_config=s.gkd_experiment_config,
                        ),
                    )
                )
    return jobs


def a3_eval_jobs(
    verify_names: list[str],
    *,
    spec: CampaignSpec | None = None,
) -> list[Job]:
    """One job per generated A3 verify YAML (single-animal, single-recipe configs)."""
    s = plan.SPEC if spec is None else spec
    return [
        Job(
            key=f"a3-eval|{Path(fname).stem.removeprefix('gkd_a3_verify_')}",
            cmd=eval_cmd(generated_rel(fname, spec=s)),
        )
        for fname in sorted(verify_names)
    ]


def b_train_jobs(
    train_configs: dict[str, dict],
    control_names: list[str],
    *,
    spec: CampaignSpec | None = None,
) -> tuple[list[Job], list[Job]]:
    """(headline_jobs, extension_jobs) for the generated Stage B train YAMLs.

    Treatments shard to (YAML, animal, student base, gseed, teacher ftseed) — one or two
    cells each — and always pass ``--no-control``. Shared controls are separate
    controls-only shards taken from the ``same_init`` YAMLs (whose 7 student bases cover
    every Stage B student) and land in the headline queue so evals are never blocked on
    the extension tier.
    """
    s = plan.SPEC if spec is None else spec
    headline: list[Job] = []
    extension: list[Job] = []
    for fname in sorted(train_configs):
        doc = train_configs[fname]
        stem = Path(fname).stem.removeprefix("gkd_b_")
        bucket = headline if fname.endswith("_headline.yaml") else extension
        for animal in s.animals:
            for student in _students(doc):
                for gseed in doc["gseeds"]:
                    for tfs in doc["teacher_ftseeds"]:
                        bucket.append(
                            Job(
                                key=f"b|{stem}|{animal}|{student}|g{gseed}|t{tfs}",
                                cmd=train_cmd(
                                    generated_rel(fname, spec=s),
                                    "--no-control",
                                    f"--animals {animal}",
                                    f"--base-models {student}",
                                    f"--gseeds {gseed}",
                                    f"--teacher-ftseeds {tfs}",
                                    experiment_config=s.gkd_experiment_config,
                                ),
                            )
                        )
                    if fname in control_names:
                        headline.append(
                            Job(
                                key=f"b-ctrl|{animal}|{student}|g{gseed}",
                                cmd=train_cmd(
                                    generated_rel(fname, spec=s),
                                    "--controls-only",
                                    f"--animals {animal}",
                                    f"--base-models {student}",
                                    f"--gseeds {gseed}",
                                    experiment_config=s.gkd_experiment_config,
                                ),
                            )
                        )
    return headline, extension


def b_eval_jobs(
    verify_configs: dict[str, dict],
    train_configs: dict[str, dict],
    *,
    spec: CampaignSpec | None = None,
) -> list[Job]:
    """One verify shard per (Stage B YAML, animal, student base)."""
    s = plan.SPEC if spec is None else spec
    return [
        Job(
            key=f"b-eval|{Path(fname).stem.removeprefix('gkd_b_verify_')}|{animal}|{student}",
            cmd=eval_cmd(
                generated_rel(fname, spec=s),
                f"--animals {animal}",
                f"--base-models {student}",
            ),
        )
        for fname in sorted(verify_configs)
        for animal in s.animals
        for student in _students(train_configs[_b_train_name(fname)])
    ]


def _b_train_name(verify_name: str) -> str:
    return verify_name.replace("gkd_b_verify_", "gkd_b_")


def c_train_jobs(
    train_configs: dict[str, dict],
    *,
    spec: CampaignSpec | None = None,
) -> list[Job]:
    """Stage C shards: (YAML, animal, student base, gseed); never trains controls."""
    s = plan.SPEC if spec is None else spec
    return [
        Job(
            key=(
                f"c|{Path(fname).stem.removeprefix('gkd_c_')}|{animal}|{student}"
                f"|g{gseed}"
            ),
            cmd=train_cmd(
                generated_rel(fname, spec=s),
                "--no-control",
                f"--animals {animal}",
                f"--base-models {student}",
                f"--gseeds {gseed}",
                experiment_config=s.gkd_experiment_config,
            ),
        )
        for fname in sorted(train_configs)
        for animal in s.animals
        for student in _students(train_configs[fname])
        for gseed in train_configs[fname]["gseeds"]
    ]


def c_eval_jobs(
    verify_configs: dict[str, dict],
    train_configs: dict[str, dict],
    *,
    spec: CampaignSpec | None = None,
) -> list[Job]:
    """One verify shard per (Stage C YAML, animal, student base)."""
    s = plan.SPEC if spec is None else spec
    return [
        Job(
            key=f"c-eval|{Path(fname).stem.removeprefix('gkd_c_verify_')}|{animal}|{student}",
            cmd=eval_cmd(
                generated_rel(fname, spec=s),
                f"--animals {animal}",
                f"--base-models {student}",
            ),
        )
        for fname in sorted(verify_configs)
        for animal in s.animals
        for student in _students(
            train_configs[fname.replace("gkd_c_verify_", "gkd_c_")]
        )
    ]
