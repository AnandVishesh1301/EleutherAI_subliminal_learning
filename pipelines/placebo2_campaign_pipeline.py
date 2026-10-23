"""Auto-advancing driver of the `placebo2` docgen campaign, on soar.

Every run (a pilot arm or a per-animal corpus) is a create-only JSON spec in
``<root>/runs/``. One `run_pass` = one idempotent pass over every spec:

    submit    the run's shard jobs (initial shards plus every top-up wave in the
              state) in one unpinned queue; each node runs them from its own clone's
              ``.code_pins/<sha>``
    requeue   failed shards below the attempt cap; at the cap the run is a hole
    pilot     every shard has DONE.json -> `traitfree_select --pilot` (inline,
              docgen venv) -> ``pilot/pilot_report.json``
    corpus    every shard has DONE.json -> `traitfree_select`; too few eligible
              documents, or only the token gap failing (exit 3) -> one more top-up
              wave, at most ``max_topups``; any other failure -> a hole
    done      ``RUNS_CLOSED`` exists and every run is finished -> ``CAMPAIGN_DONE``

Waiting never raises. A hole is recorded in the state and that run is skipped until
`clear_hole`; the other runs advance.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import shlex
import shutil
import subprocess
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from subliminal_learning.campaign import placebo2_plan as plan
from subliminal_learning.jobqueue import store
from subliminal_learning.jobqueue.model import Job
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json, atomic_write_text
from subliminal_learning.utils.run_io import utc_now_iso

class CampaignGateError(RuntimeError):
    """A campaign contract refused to pass."""


@dataclass(frozen=True)
class RunSpec:
    """One docgen run of the campaign.

    Attributes:
        name: Run name (queue ``placebo2-<name>``, dir ``docgen/<name>``).
        kind: ``pilot`` or ``corpus``.
        config: Docgen YAML, repo-relative (resolved in the pin) or absolute.
        animal: Reference corpus the slots mirror and the corpus is matched to.
        n_shards: Initial shards.
        docs_per_shard: Accepted documents per shard.
        seed_base: Seed of shard 0 (`plan.shard_seed`).
        priority: jobq priority (higher runs first).
        topup_shards: Shards per top-up wave (corpus).
        max_topups: Top-up waves allowed (corpus).
        match_seed: Matching seed (corpus).
        code_sha: Pin of this run's shards and select; ``None`` follows the
            driver's pin (a run keeps one pin from submit to select).
    """

    name: str
    kind: str
    config: str
    animal: str
    n_shards: int
    docs_per_shard: int
    seed_base: int
    priority: int
    topup_shards: int = 16
    max_topups: int = 3
    match_seed: int = 20261023
    code_sha: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in (plan.KIND_PILOT, plan.KIND_CORPUS):
            raise ValueError(f"run {self.name}: unknown kind {self.kind!r}")
        if self.animal not in plan.CORPUS_ANIMALS:
            raise ValueError(f"run {self.name}: unknown animal {self.animal!r}")
        if not self.name.replace("-", "").isalnum():
            raise ValueError(f"run name {self.name!r} must be alphanumeric with dashes")
        if self.n_shards < 1 or self.docs_per_shard < 1:
            raise ValueError(f"run {self.name}: empty run")


def load_run_specs(runs_dir: Path = plan.RUNS_DIR) -> list[RunSpec]:
    """Every run spec, highest priority first (then by name).

    Raises:
        ValueError: A spec is malformed or two specs share a seed base.
    """
    specs = []
    for path in sorted(runs_dir.glob("*.json")):
        spec = RunSpec(**json.loads(path.read_text(encoding="utf-8")))
        if path.stem != spec.name:
            raise ValueError(f"{path} holds run {spec.name!r}")
        specs.append(spec)
    seeds = [s.seed_base for s in specs]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"run specs share a seed_base: {sorted(seeds)}")
    return sorted(specs, key=lambda s: (-s.priority, s.name))


def add_run_spec(spec: RunSpec, runs_dir: Path = plan.RUNS_DIR) -> Path:
    """Write `spec` create-only (refuses a closed run list or an existing name)."""
    if (runs_dir / plan.RUNS_CLOSED_MARKER.name).exists():
        raise CampaignGateError("the run list is closed")
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{spec.name}.json"
    others = load_run_specs(runs_dir)
    if any(o.seed_base == spec.seed_base for o in others):
        raise ValueError(f"seed_base {spec.seed_base} already used")
    tmp = runs_dir / f".{spec.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(asdict(spec), indent=2, sort_keys=True), encoding="utf-8")
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()
    return path


def _config_path(spec: RunSpec, repo_root: Path) -> Path:
    path = Path(spec.config)
    return path if path.is_absolute() else repo_root / path


def run_config(spec: RunSpec, repo_root: Path) -> dict[str, Any]:
    """The run's docgen YAML.

    Raises:
        CampaignGateError: The config is not a trait-free v2 (mirror) config.
    """
    cfg = yaml.safe_load(_config_path(spec, repo_root).read_text(encoding="utf-8"))
    if cfg.get("slot_mode") != "mirror":
        raise CampaignGateError(f"run {spec.name}: {spec.config} is not a mirror config")
    return cfg


def total_shards(spec: RunSpec, n_topups: int) -> int:
    """Shards that exist after `n_topups` top-up waves."""
    return spec.n_shards + spec.topup_shards * n_topups


def shard_done(spec: RunSpec, index: int) -> bool:
    """Whether a shard published its DONE.json."""
    return (plan.run_dir(spec.name) / "shards" / f"shard-{index:03d}" / "DONE.json").is_file()


def shard_job(spec: RunSpec, index: int, cfg: dict[str, Any]) -> Job:
    """The whole-card job of one shard (cwd = the node's pin root)."""
    args = [
        plan.DOCGEN_PYTHON_FROM_PIN,
        "-m",
        "sdf_docgen.traitfree_shard",
        "--config",
        spec.config,
        "--shard-index",
        str(index),
        "--n-docs",
        str(spec.docs_per_shard),
        "--seed",
        str(plan.shard_seed(spec.seed_base, index)),
        "--run-root",
        str(plan.run_dir(spec.name)),
        "--hf-cache-root",
        str(plan.HF_CACHE_ROOT),
        "--calibration",
        str(plan.CALIBRATION_EASY),
        "--calibration-hard",
        str(plan.CALIBRATION_HARD),
        "--calibration-negatives",
        str(plan.CALIBRATION_NEGATIVES),
        "--reference",
        str(plan.reference_path(spec.animal)),
    ]
    if cfg["use_details"]:
        args += ["--detail-pool", str(plan.DETAIL_POOL)]
    if cfg["use_topics"]:
        args += ["--topic-bank", str(plan.TOPIC_BANK)]
    guard = (
        "import os, sys, sdf_docgen; "
        "src = os.path.realpath(sdf_docgen.__file__); "
        f"sys.exit(0 if src.startswith(os.path.realpath({plan.DOCGEN_SRC!r}) + os.sep) else 17)"
    )
    cmd = (
        f"{shlex.quote(plan.DOCGEN_PYTHON_FROM_PIN)} -c {shlex.quote(guard)} && "
        f"{shlex.join(args)}"
    )
    return Job(
        key=f"shard|{index:03d}",
        cmd=cmd,
        env={"PYTHONPATH": plan.DOCGEN_SRC},
        mem_mib=plan.DOCGEN_JOB_MIB,
        slots=1,
    )


def queue_defaults(sha: str) -> dict:
    """jobq defaults of a run queue: every host's pin (cwd + env) under ``nodes``.

    The queue-wide entry is goncalo's (its clone path differs from the others').
    """
    per_host = {
        h: code_pin_io.pin_job_defaults(
            code_pin_io.CodePin(sha=sha, root=plan.CLONE_ROOT[h] / code_pin_io.CODE_PINS_DIRNAME / sha)
        )
        for h in plan.HOSTS
    }
    return {**per_host[plan.GONCALO], "nodes": per_host}


def _submit(spec: RunSpec, n_topups: int, sha: str, repo_root: Path, dry_run: bool) -> int:
    """Create the run's queue or append its missing shard jobs; refresh its pin."""
    cfg = run_config(spec, repo_root)
    jobs = [shard_job(spec, i, cfg) for i in range(total_shards(spec, n_topups))]
    name = plan.queue_name(spec.name)
    defaults = queue_defaults(sha)
    if dry_run:
        return len(jobs)
    if not store.queue_exists(plan.QUEUE_ROOT, name):
        store.create_queue(
            plan.QUEUE_ROOT, name, node=None, priority=spec.priority, defaults=defaults
        )
    elif store.read_meta(plan.QUEUE_ROOT, name).defaults != defaults:
        store.update_queue_defaults(plan.QUEUE_ROOT, name, defaults)
    have = {j.jobkey for j in store.load_jobs(plan.QUEUE_ROOT, name)}
    new = [j for j in jobs if j.jobkey not in have]
    if new:
        store.append_jobs(plan.QUEUE_ROOT, name, new)
        logger.info("{}: appended {} shard job(s)", name, len(new))
    return len(new)


def _requeue(spec: RunSpec, dry_run: bool) -> str | None:
    """Requeue failed shards under the cap; return an error at the cap."""
    name = plan.queue_name(spec.name)
    if not store.queue_exists(plan.QUEUE_ROOT, name):
        return None
    over = []
    for job in store.load_jobs(plan.QUEUE_ROOT, name):
        if store.job_state(plan.QUEUE_ROOT, name, job) != store.FAILED:
            continue
        attempt = store.read_attempt(plan.QUEUE_ROOT, name, job.jobkey)
        if attempt + 1 >= plan.AUTO_REQUEUE_ATTEMPTS:
            over.append(job.key)
        elif not dry_run:
            store.requeue(plan.QUEUE_ROOT, name, jobkeys=[job.jobkey])
            logger.warning("{}: requeued {} (attempt {})", name, job.key, attempt + 1)
    return f"shards failed {plan.AUTO_REQUEUE_ATTEMPTS} times: {over}" if over else None


def _run_select(
    spec: RunSpec, n_shards: int, out_dir: Path, pin: code_pin_io.CodePin, cfg: dict[str, Any]
) -> int:
    """Run `traitfree_select` inline (docgen venv, pin code); return its exit code."""
    python = plan.CLONE_ROOT[plan.HOME_NODE] / "tools" / "sdf_docgen" / ".venv" / "bin" / "python"
    argv = [
        str(python),
        "-m",
        "sdf_docgen.traitfree_select",
        "--config",
        spec.config,
        "--run-root",
        str(plan.run_dir(spec.name)),
        "--n-shards",
        str(n_shards),
        "--reference",
        str(plan.reference_path(spec.animal)),
        "--tokenizer-dir",
        str(plan.TOKENIZER_DIR),
        "--out-dir",
        str(out_dir),
        "--corpus-name",
        spec.name,
        "--seed",
        str(spec.match_seed),
        "--pile-df",
        str(plan.pile_df_path(spec.animal)),
    ]
    if cfg["use_details"]:
        argv += ["--detail-pool", str(plan.DETAIL_POOL)]
    if spec.kind == plan.KIND_PILOT:
        argv.append("--pilot")
    else:
        argv += [
            "--concept-min-df",
            str(plan.CORPUS_CONCEPT_MIN_DF),
            "--screen-lexicon",
            plan.CORPUS_SCREEN_LEXICON,
        ]
        for term in plan.CORPUS_CONCEPT_EXEMPT:
            argv += ["--concept-exempt", term]
    env = dict(os.environ) | {
        "PYTHONPATH": str(pin.root / plan.DOCGEN_SRC),
        code_pin_io.CODE_PIN_SHA_ENV: pin.sha,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "OMP_NUM_THREADS": "8",
    }
    log = plan.run_dir(spec.name) / "logs" / f"select-{utc_now_iso().replace(':', '')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    logger.info("{}: select -> {}", spec.name, log)
    with log.open("x") as handle:
        proc = subprocess.run(
            argv, cwd=pin.root, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    logger.info("{}: select exit {}", spec.name, proc.returncode)
    return proc.returncode


def _output_dir(spec: RunSpec) -> Path:
    sub = "pilot" if spec.kind == plan.KIND_PILOT else "corpus"
    return plan.run_dir(spec.name) / sub


def run_finished(spec: RunSpec) -> bool:
    """Whether the run's output is published."""
    out = _output_dir(spec)
    marker = "pilot_report.json" if spec.kind == plan.KIND_PILOT else "CORPUS_READY.json"
    return (out / marker).is_file()


def load_state() -> dict:
    """Driver state (``runs``: per-run top-up waves, status, hole)."""
    if plan.STATE_FILE.is_file():
        return json.loads(plan.STATE_FILE.read_text(encoding="utf-8"))
    return {"runs": {}}


def save_state(state: dict) -> None:
    """Persist the driver state atomically."""
    atomic_write_json(plan.STATE_FILE, state)


@contextlib.contextmanager
def driver_lock() -> Iterator[None]:
    """Exclusive lock serialising driver passes and state edits."""
    plan.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with (plan.LOGS_DIR / "driver.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def clear_hole(run_name: str) -> list[str]:
    """Remove a run's hole and give its failed shards one more attempt.

    Returns:
        The requeued shard jobkeys.
    """
    with driver_lock():
        state = load_state()
        state["runs"].get(run_name, {}).pop("hole", None)
        save_state(state)
        name = plan.queue_name(run_name)
        if not store.queue_exists(plan.QUEUE_ROOT, name):
            return []
        return store.requeue(plan.QUEUE_ROOT, name, failed_only=True, count_attempt=False)


def missing_inputs(spec: RunSpec, cfg: dict[str, Any]) -> list[str]:
    """Inputs a run's shards and select need that are not on disk."""
    need = [
        plan.reference_path(spec.animal),
        plan.pile_df_path(spec.animal),
        plan.CALIBRATION_EASY,
        plan.CALIBRATION_HARD,
        plan.CALIBRATION_NEGATIVES,
        plan.TOKENIZER_DIR,
    ]
    if cfg["use_details"]:
        need.append(plan.DETAIL_POOL)
    if cfg["use_topics"]:
        need.append(plan.TOPIC_BANK)
    return [str(p) for p in need if not p.exists()]


def capacity_problems(spec: RunSpec, cfg: dict[str, Any]) -> list[str]:
    """Pool and topic-bank shortfalls that make a corpus's caps unsatisfiable.

    Needs at least twice the minimum distinct items (reference size / item
    cap), no empty pool category, and three topics per reference document of
    each register (topic cap 1).
    """
    ref = [
        json.loads(line)
        for line in plan.reference_path(spec.animal).read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    problems = []
    if cfg["use_details"]:
        pool = json.loads(plan.DETAIL_POOL.read_text(encoding="utf-8"))
        need = 2 * math.ceil(len(ref) / int(cfg["detail_item_cap"]))
        if len(pool["items"]) < need:
            problems.append(f"detail pool has {len(pool['items'])} items < {need}")
        empty = [c for c, n in pool["counts"].items() if n == 0]
        if empty:
            problems.append(f"empty pool categories {empty}")
    if cfg["use_topics"]:
        have = Counter(
            json.loads(line)["pile_set_name"]
            for line in plan.TOPIC_BANK.read_text(encoding="utf-8").split("\n")
            if line.strip()
        )
        for register, n in Counter(r["pile_set_name"] for r in ref).items():
            if have[register] < 3 * n * int(cfg["topic_cap"]):
                problems.append(f"{have[register]} {register} topics < 3 x {n}")
    return problems


def run_pin_root(spec: RunSpec, driver_sha: str, repo_root: Path) -> tuple[str, Path]:
    """(sha, soar pin root) a run's shards and select use."""
    if spec.code_sha is None or spec.code_sha == driver_sha:
        return driver_sha, repo_root
    return spec.code_sha, plan.CLONE_ROOT[plan.HOME_NODE] / code_pin_io.CODE_PINS_DIRNAME / spec.code_sha


def _advance_run(
    spec: RunSpec,
    entry: dict,
    driver_sha: str,
    driver_root: Path,
    driver_pin: code_pin_io.CodePin | None,
    dry_run: bool,
) -> str:
    """Advance one run; return a one-line status."""
    if run_finished(spec):
        entry["status"] = "done"
        return "done"
    if entry.get("hole"):
        return f"hole: {entry['hole']}"
    sha, repo_root = run_pin_root(spec, driver_sha, driver_root)
    if not (repo_root / ".git").exists():
        return f"waiting for pin {repo_root}"
    cfg = run_config(spec, repo_root)
    missing = missing_inputs(spec, cfg)
    if missing:
        return f"waiting for inputs {missing}"
    if spec.kind == plan.KIND_CORPUS and not store.queue_exists(plan.QUEUE_ROOT, plan.queue_name(spec.name)):
        problems = capacity_problems(spec, cfg)
        if problems:
            entry["hole"] = f"capacity: {problems}"
            return f"hole: {entry['hole']}"
    waves = int(entry.get("topups", 0))
    added = _submit(spec, waves, sha, repo_root, dry_run)
    err = _requeue(spec, dry_run)
    if err:
        entry["hole"] = err
        return f"hole: {err}"
    n = total_shards(spec, waves)
    left = [i for i in range(n) if not shard_done(spec, i)]
    if left:
        entry["status"] = "generating"
        return f"{n - len(left)}/{n} shards done" + (f" (+{added} submitted)" if added else "")
    if dry_run or driver_pin is None:
        return "ready for select"
    rc = _run_select(spec, n, _output_dir(spec), driver_pin, cfg)
    if rc == 0:
        entry["status"] = "done"
        entry["finished_at"] = utc_now_iso()
        return "done"
    if spec.kind == plan.KIND_CORPUS and rc == plan.EXIT_TOO_FEW_DOCUMENTS:
        if waves < spec.max_topups:
            entry["topups"] = waves + 1
            _submit(spec, waves + 1, sha, repo_root, dry_run)
            return f"top-up wave {waves + 1} submitted"
        entry["hole"] = f"select still short after {waves} top-ups"
        return f"hole: {entry['hole']}"
    entry["hole"] = f"select exit {rc}"
    return f"hole: {entry['hole']}"


def disk_problems() -> list[str]:
    """Free-space floor of the campaign root's volume."""
    free = shutil.disk_usage(plan.ROOT).free / 2**30
    return [f"{free:.0f} GiB free < {plan.MIN_FREE_GIB}"] if free < plan.MIN_FREE_GIB else []


def run_pass(code_sha: str, repo_root: Path, *, dry_run: bool = False) -> dict:
    """One driver pass over every run spec.

    Args:
        code_sha: Campaign pin (every node must hold ``.code_pins/<sha>``).
        repo_root: Root of the pin this driver runs from.
        dry_run: Plan only; no queue writes, no select.

    Returns:
        ``{"runs": {name: status}, "done": bool}``.

    Raises:
        CampaignGateError: The driver is not on its home node, or not in the pin.
    """
    if store.this_host() != plan.HOME_NODE:
        raise CampaignGateError(f"the driver runs on {plan.HOME_NODE}")
    pin = None
    if not dry_run:
        pin = code_pin_io.CodePin(sha=code_sha, root=repo_root)
        if code_pin_io.resolve_commit_sha(repo_root, "HEAD") != code_sha:
            raise CampaignGateError(f"{repo_root} is not the pin of {code_sha}")
    with driver_lock():
        return _locked_pass(code_sha, repo_root, pin, dry_run)


def _locked_pass(
    code_sha: str, repo_root: Path, pin: code_pin_io.CodePin | None, dry_run: bool
) -> dict:
    state = load_state()
    statuses: dict[str, str] = {}
    disk = disk_problems()
    for spec in load_run_specs():
        entry = state["runs"].setdefault(spec.name, {})
        if disk and not run_finished(spec):
            statuses[spec.name] = f"disk_low: {disk[0]}"
            continue
        statuses[spec.name] = _advance_run(spec, entry, code_sha, repo_root, pin, dry_run)
    specs = load_run_specs()
    done = (
        plan.RUNS_CLOSED_MARKER.exists()
        and bool(specs)
        and all(run_finished(s) for s in specs)
    )
    if done and not dry_run and not plan.DONE_MARKER.exists():
        atomic_write_text(plan.DONE_MARKER, utc_now_iso() + "\n")
    state["last_pass"] = {"at": utc_now_iso(), "runs": statuses, "code_sha": code_sha}
    if not dry_run:
        save_state(state)
    return {"runs": statuses, "done": done}


def status() -> dict:
    """Per-run shard and queue counts plus the state (pure file reads)."""
    out = {}
    state = load_state()
    for spec in load_run_specs():
        entry = state["runs"].get(spec.name, {})
        n = total_shards(spec, int(entry.get("topups", 0)))
        name = plan.queue_name(spec.name)
        queue = (
            {k: v for k, v in store.queue_status(plan.QUEUE_ROOT, name).items() if k != "running_jobs"}
            if store.queue_exists(plan.QUEUE_ROOT, name)
            else None
        )
        out[spec.name] = {
            "kind": spec.kind,
            "shards_done": sum(shard_done(spec, i) for i in range(n)),
            "shards": n,
            "finished": run_finished(spec),
            "queue": queue,
            **{k: entry[k] for k in ("status", "topups", "hole") if k in entry},
        }
    return {"runs": out, "last_pass": state.get("last_pass"), "done": plan.DONE_MARKER.exists()}


def write_policy(host: str) -> Path:
    """Write `host`'s jobq policy into the campaign queue root (create or equal)."""
    path = plan.QUEUE_ROOT / f"gpu_policy.{host}.json"
    doc = plan.gpu_policy_json()
    if path.is_file():
        if json.loads(path.read_text()) != doc:
            raise FileExistsError(f"{path} exists with a different policy")
        return path
    plan.QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, doc)
    return path


def stage_inputs(repo_root: Path) -> Path:
    """Copy the references, easy calibration set and tokenizer into the inputs
    dir (create-only, sha-checked) and build each animal's real-Pile reference DF.

    Returns:
        The new staging report.
    """
    from pipelines.traitfree_staging_pipeline import copy_file

    files: dict[str, dict[str, str]] = {}

    def put(src: Path, dest: Path) -> None:
        sha, _ = copy_file(src, dest)
        files[str(dest)] = {"source": str(src), "sha256": sha}

    for animal, src in plan.REFERENCE_SOURCES.items():
        put(src, plan.reference_path(animal))
    put(plan.CALIBRATION_EASY_SOURCE, plan.CALIBRATION_EASY)
    for path in sorted(plan.TOKENIZER_SOURCE.rglob("*")):
        if path.is_file():
            put(path, plan.TOKENIZER_DIR / path.relative_to(plan.TOKENIZER_SOURCE))
    python = repo_root / "tools" / "sdf_docgen" / ".venv" / "bin" / "python"
    env = dict(os.environ) | {"PYTHONPATH": str(repo_root / plan.DOCGEN_SRC)}
    for animal in plan.CORPUS_ANIMALS:
        out = plan.pile_df_path(animal)
        if not out.is_file():
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "sdf_docgen.traitfree_metrics",
                    "reference-df",
                    "--docs",
                    str(plan.PILE_ROWS),
                    "--weights-from",
                    str(plan.reference_path(animal)),
                    "--out",
                    str(out),
                ],
                cwd=repo_root,
                env=env,
                check=True,
            )
        files[str(out)] = {"source": str(plan.PILE_ROWS), "sha256": _sha256(out)}
    report = plan.INPUTS_DIR / f"staging_report-{utc_now_iso().replace(':', '')}.json"
    with report.open("x", encoding="utf-8") as f:
        json.dump(
            {"created_at": utc_now_iso(), "pile_rows_sha256": _sha256(plan.PILE_ROWS), "files": files},
            f,
            indent=1,
            sort_keys=True,
        )
    return report


def _sha256(path: Path) -> str:
    from pipelines.traitfree_staging_pipeline import sha256_file

    return sha256_file(path)
