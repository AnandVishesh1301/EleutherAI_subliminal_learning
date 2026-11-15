"""Campaign driver IO: state file, generated-config writes, jobq submission.

Everything a campaign stage machine (pipelines/student_hp_campaign_pipeline.py,
pipelines/optparam_campaign_pipeline.py) needs that touches disk. Submission goes
straight through `jobq.store` (the same functions `jobq submit`
wraps), so the driver gets the store's duplicate-jobkey and exist_ok guarantees
without shelling out.

`CampaignDriverConfig` and the submission/state helpers are campaign-agnostic --
the identity (`run_set_id`) and queue priorities are fields, defaulted to the
20260727 plan. The `a1_jobs`-style builders below belong to that campaign;
`campaign.optparam_jobs` holds the 20260801 ones.

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

from subliminal_learning.campaign import plan
from subliminal_learning.campaign.selection import (
    FrozenRecipe,
    Recipe,
    RecipeSelection,
)
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from jobq import store
from jobq.model import Job
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text

TRAIN_MODULE = "scripts.sweeps.sweep_distill_student_gkd"
EVAL_MODULE = "scripts.sweeps.sweep_evaluate_gkd_student"


@dataclass(frozen=True)
class CampaignDriverConfig:
    """Everything the campaign stage machine needs.

    Attributes:
        output_root: Artifact root from the paths YAML.
        queue_folder: jobq queue folder shared by all campaign queues.
        repo_root: Repo checkout the queue jobs run from (their ``cwd``).
        include_stage_c: Also generate + submit the optional Stage C ablations.
        require_a0: Gate the first submission on the A0 calibration +
            init-equality reports (disable only for tests).
        auto_reconcile_sha: Record a HEAD advance on the campaign card instead of
            halting, so a commit landing mid-campaign does not strand queued cells.
        dry_run: Log every planned action; write/submit nothing.
        generated_dir_override: Where generated YAMLs go (tests only; default
            ``{repo_root}/configs/experiment/generated``).
        run_set_id: Campaign id whose output subtree the driver reads and writes.
        queue_priorities: Queue name -> jobq priority for this campaign's queues.
    """

    output_root: Path
    queue_folder: Path
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

    @property
    def campaign_subdir(self) -> str:
        return run_set_subdir(GKD_STUDENTS_SUBDIR, self.run_set_id)

    @property
    def run_set_root(self) -> Path:
        return self.output_root / self.campaign_subdir

    @property
    def logs_dir(self) -> Path:
        return self.output_root / self.campaign_subdir / "logs"

    @property
    def state_path(self) -> Path:
        return self.logs_dir / plan.CAMPAIGN_STATE_FILE

    @property
    def generated_dir(self) -> Path:
        if self.generated_dir_override is not None:
            return self.generated_dir_override
        return self.repo_root / plan.GENERATED_CONFIG_DIR


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


def submit_queue(
    cfg: CampaignDriverConfig,
    name: str,
    jobs: list[Job],
    *,
    depends_on: list[str] | None = None,
) -> str:
    """Create queue `name` (priority from the plan) and append `jobs`, idempotently.

    Queues are pinned to the submitting host. Both nodes drain the one canonical
    root (run_logs/README.md §7), but only this node's checkout is guaranteed to
    carry the campaign's code — an unpinned queue drained by a stale checkout
    fails loudly at best and silently trains the wrong recipe at worst.

    Returns:
        ``"submitted"`` | ``"exists"`` (identical jobkey set — no-op).

    Raises:
        RuntimeError: The queue exists with a DIFFERENT jobkey set.
    """
    if not jobs:
        raise ValueError(f"refusing to submit empty queue {name!r}")
    priority = cfg.queue_priorities[name]
    if store.queue_exists(cfg.queue_folder, name):
        existing = {j.jobkey for j in store.load_jobs(cfg.queue_folder, name)}
        incoming = {j.jobkey for j in jobs}
        if existing == incoming:
            logger.info("queue {} already submitted ({} jobs) — skip", name, len(jobs))
            return "exists"
        raise RuntimeError(
            f"queue {name!r} exists with a different job set "
            f"({len(existing)} existing vs {len(incoming)} planned) — refusing to mix"
        )
    node = store.this_host()
    if cfg.dry_run:
        logger.info(
            "[dry-run] would submit queue {} (prio {}, node {}, deps {}, {} jobs)",
            name,
            priority,
            node,
            depends_on or [],
            len(jobs),
        )
        return "submitted"
    store.create_queue(
        cfg.queue_folder,
        name,
        depends_on=depends_on or [],
        node=node,
        priority=priority,
        defaults={"cwd": str(cfg.repo_root)},
    )
    store.append_jobs(cfg.queue_folder, name, jobs)
    logger.info(
        "submitted queue {} (prio {}, node {}, deps {}, {} jobs)",
        name,
        priority,
        node,
        depends_on or [],
        len(jobs),
    )
    return "submitted"


def queue_finished(cfg: CampaignDriverConfig, name: str) -> tuple[bool, str]:
    """(all-jobs-terminal-and-none-failed, human status) for one queue."""
    if not store.queue_exists(cfg.queue_folder, name):
        return False, f"{name}: not submitted"
    status = store.queue_status(cfg.queue_folder, name)
    if status["failed"]:
        return False, (
            f"{name}: {status['failed']} FAILED job(s) — requeue with "
            f"`uv run jobq requeue {name}`"
        )
    if not store.queue_complete(cfg.queue_folder, name):
        return False, (
            f"{name}: {status['pending']} pending / {status['running']} running "
            f"of {status['total']}"
        )
    return True, f"{name}: complete ({status['done']} done)"


# ---- job builders -------------------------------------------------------------------


def train_cmd(yaml_rel: str, *shard_flags: str) -> str:
    """One train-sweep shard command (jobq pins the device via CUDA_VISIBLE_DEVICES)."""
    flags = " ".join(shard_flags)
    return (
        f"uv run python -m {TRAIN_MODULE} --sweep-config {yaml_rel} "
        f"--device cuda:0{' ' + flags if flags else ''}"
    )


def eval_cmd(yaml_rel: str, *shard_flags: str) -> str:
    """One verify-sweep shard command."""
    flags = " ".join(shard_flags)
    return (
        f"uv run python -m {EVAL_MODULE} --sweep-config {yaml_rel} "
        f"--device cuda:0{' ' + flags if flags else ''}"
    )


def generated_rel(fname: str) -> str:
    """Repo-relative path of a generated sweep YAML (jobs run with cwd=repo_root)."""
    return f"{plan.GENERATED_CONFIG_DIR}/{fname}"


def _students(doc: dict) -> list[str]:
    """Distinct student base labels of a generated YAML's pair list, in file order."""
    return list(dict.fromkeys(str(p[0]) for p in doc["teacher_student_base_pairs"]))


def a1_jobs() -> list[Job]:
    """288 single-cell shards: (arm, animal, beta, lr, student mode), control-free.

    One cell per job is the finest disjoint partition the sweep CLI exposes here, so the
    288-cell screen load-balances perfectly across the pool and a crashed shard costs one
    cell instead of nine.
    """
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
            ),
        )
        for arm, yaml_rel in (
            ("same", plan.A1_SAME_YAML),
            ("cross", plan.A1_CROSS_YAML),
        )
        for animal in plan.ANIMALS
        for beta in plan.BETAS
        for lr in plan.A1_LORA_LRS
        for mode in plan.A1_STUDENT_MODES
    ]


def a2_jobs() -> list[Job]:
    """16 single-control shards: the same-init YAML at r64a64, controls only."""
    return [
        Job(
            key=f"a2|{animal}|b{beta:g}|lr{lr:g}",
            cmd=train_cmd(
                plan.A1_SAME_YAML,
                "--controls-only",
                f"--student-modes {plan.A2_STUDENT_MODE}",
                f"--animals {animal}",
                f"--beta {beta}",
                f"--lora-lrs {lr}",
            ),
        )
        for animal in plan.ANIMALS
        for beta in plan.BETAS
        for lr in plan.A1_LORA_LRS
    ]


def a1_eval_jobs() -> list[Job]:
    """72 verify shards: (arm, animal, beta, student mode)."""
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
            ("same", plan.A1_SAME_VERIFY_YAML),
            ("cross", plan.A1_CROSS_VERIFY_YAML),
        )
        for animal in plan.ANIMALS
        for beta in plan.BETAS
        for mode in plan.A1_STUDENT_MODES
    ]


def backlog_jobs() -> list[Job]:
    """18 filler shards: the catswide eval backlog, one per (verify YAML, base model)."""
    return [
        Job(
            key=(
                f"backlog|"
                f"{Path(yaml_rel).stem.removeprefix('gkd_student_verify_sweep_lora_')}"
                f"|{model}"
            ),
            cmd=eval_cmd(yaml_rel, f"--base-models {model}"),
        )
        for yaml_rel in plan.CATSWIDE_VERIFY_YAMLS
        for model in plan.CATSWIDE_BASE_MODELS
    ]


def a3_jobs(train_configs: dict[str, dict], headline_names: list[str]) -> list[Job]:
    """240 single-cell A3 shards: every train YAML x gseed, plus headline controls.

    Treatments always pass ``--no-control``; the headline (``r1``) arms get their own
    controls-only shards so a control never rides along with — and serializes behind —
    a treatment shard.
    """
    jobs: list[Job] = []
    for fname in sorted(train_configs):
        stem = Path(fname).stem.removeprefix("gkd_a3_")
        for gseed in plan.A3_GSEEDS:
            jobs.append(
                Job(
                    key=f"a3|{stem}|g{gseed}",
                    cmd=train_cmd(
                        generated_rel(fname), "--no-control", f"--gseeds {gseed}"
                    ),
                )
            )
            if fname in headline_names:
                jobs.append(
                    Job(
                        key=f"a3-ctrl|{stem}|g{gseed}",
                        cmd=train_cmd(
                            generated_rel(fname), "--controls-only", f"--gseeds {gseed}"
                        ),
                    )
                )
    return jobs


def a3_eval_jobs(verify_names: list[str]) -> list[Job]:
    """One job per generated A3 verify YAML (single-animal, single-recipe configs)."""
    return [
        Job(
            key=f"a3-eval|{Path(fname).stem.removeprefix('gkd_a3_verify_')}",
            cmd=eval_cmd(generated_rel(fname)),
        )
        for fname in sorted(verify_names)
    ]


def b_train_jobs(
    train_configs: dict[str, dict], control_names: list[str]
) -> tuple[list[Job], list[Job]]:
    """(headline_jobs, extension_jobs) for the generated Stage B train YAMLs.

    Treatments shard to (YAML, animal, student base, gseed, teacher ftseed) — one or two
    cells each — and always pass ``--no-control``. The 42 shared controls are separate
    controls-only shards taken from the ``same_init`` YAMLs (whose 7 student bases cover
    every Stage B student) and land in the headline queue so evals are never blocked on
    the extension tier.
    """
    headline: list[Job] = []
    extension: list[Job] = []
    for fname in sorted(train_configs):
        doc = train_configs[fname]
        stem = Path(fname).stem.removeprefix("gkd_b_")
        bucket = headline if fname.endswith("_headline.yaml") else extension
        for animal in plan.ANIMALS:
            for student in _students(doc):
                for gseed in doc["gseeds"]:
                    for tfs in doc["teacher_ftseeds"]:
                        bucket.append(
                            Job(
                                key=f"b|{stem}|{animal}|{student}|g{gseed}|t{tfs}",
                                cmd=train_cmd(
                                    generated_rel(fname),
                                    "--no-control",
                                    f"--animals {animal}",
                                    f"--base-models {student}",
                                    f"--gseeds {gseed}",
                                    f"--teacher-ftseeds {tfs}",
                                ),
                            )
                        )
                    if fname in control_names:
                        headline.append(
                            Job(
                                key=f"b-ctrl|{animal}|{student}|g{gseed}",
                                cmd=train_cmd(
                                    generated_rel(fname),
                                    "--controls-only",
                                    f"--animals {animal}",
                                    f"--base-models {student}",
                                    f"--gseeds {gseed}",
                                ),
                            )
                        )
    return headline, extension


def b_eval_jobs(
    verify_configs: dict[str, dict], train_configs: dict[str, dict]
) -> list[Job]:
    """One verify shard per (Stage B YAML, animal, student base)."""
    return [
        Job(
            key=f"b-eval|{Path(fname).stem.removeprefix('gkd_b_verify_')}|{animal}|{student}",
            cmd=eval_cmd(
                generated_rel(fname),
                f"--animals {animal}",
                f"--base-models {student}",
            ),
        )
        for fname in sorted(verify_configs)
        for animal in plan.ANIMALS
        for student in _students(train_configs[_b_train_name(fname)])
    ]


def _b_train_name(verify_name: str) -> str:
    return verify_name.replace("gkd_b_verify_", "gkd_b_")


def c_train_jobs(train_configs: dict[str, dict]) -> list[Job]:
    """Stage C shards: (YAML, animal, student base, gseed); never trains controls."""
    return [
        Job(
            key=(
                f"c|{Path(fname).stem.removeprefix('gkd_c_')}|{animal}|{student}"
                f"|g{gseed}"
            ),
            cmd=train_cmd(
                generated_rel(fname),
                "--no-control",
                f"--animals {animal}",
                f"--base-models {student}",
                f"--gseeds {gseed}",
            ),
        )
        for fname in sorted(train_configs)
        for animal in plan.ANIMALS
        for student in _students(train_configs[fname])
        for gseed in train_configs[fname]["gseeds"]
    ]


def c_eval_jobs(
    verify_configs: dict[str, dict], train_configs: dict[str, dict]
) -> list[Job]:
    """One verify shard per (Stage C YAML, animal, student base)."""
    return [
        Job(
            key=f"c-eval|{Path(fname).stem.removeprefix('gkd_c_verify_')}|{animal}|{student}",
            cmd=eval_cmd(
                generated_rel(fname),
                f"--animals {animal}",
                f"--base-models {student}",
            ),
        )
        for fname in sorted(verify_configs)
        for animal in plan.ANIMALS
        for student in _students(
            train_configs[fname.replace("gkd_c_verify_", "gkd_c_")]
        )
    ]
