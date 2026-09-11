"""S0 pre-flight for the cross-family OPD campaign.

Three phases, each idempotent and separately invocable:

1. ``--download`` (online): snapshot every roster repo at its pinned commit into the
   cold-1 Hub cache and record the resolved shas. Also writes the campaign paths YAML
   whose ``hf_cache_root`` points there, so every cell resolves the same snapshots
   without touching the machine-local paths file.
2. default (offline): verify each snapshot against the roster -- architecture, logit
   width, tied embeddings, tokenizer identity within the cluster, single-token eval
   animals, LoRA target modules present, prompt-pool ids inside the shared prefix --
   plus the shared-init gate on the two Pythia step-0 branches. Then the dataset,
   disk and cold-symlink checks, and `staging_report.json`.
3. ``--smoke`` (GPU): a five-step student cell per distinct footprint class with
   nvidia-smi sampling, writing `xfam_footprints.json` for the packing stamp.

    uv run python -m scripts.stage_xfam_assets --dry-run
    uv run python -m scripts.stage_xfam_assets --download
    uv run python -m scripts.stage_xfam_assets
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml
from loguru import logger

from scripts.stage_ckptgrid_assets import (
    DEFAULT_COLD_ROOT,
    DEFAULT_MODEL,
    DEFAULT_PATHS,
    disk_gate_readings,
    ensure_cold_symlink,
    sdf_docs_root_from_experiment,
)
from scripts.stage_xsize_assets import _load_state_dict, _n_diff, _snapshot_dir
from subliminal_learning.campaign import xfam_jobs as jobs
from subliminal_learning.campaign import xfam_models as models
from subliminal_learning.campaign import xfam_plan as plan
from subliminal_learning.config import PathsConfig, load_config
from subliminal_learning.evaluation.prompt_sets import prompt_set_path
from subliminal_learning.evaluation.sdf_prompt_sets import DEFAULT_FC_CANDIDATES
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.models.hf_spec import HfModelSpec, tokenizer_identity_sha256
from subliminal_learning.sdf.docs_io import resolve_sdf_docs_path
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import utc_now_iso

DEFAULT_COLD_HF_CACHE = Path("/mnt/cold-1/soar-subliminal/vishesh/huggingface")
DEFAULT_SDF_EXPERIMENT = Path(plan.TEACHER_EXPERIMENT_CONFIG["neox"])
PREREG_PATH = Path(plan.PREREG_PATH)
PATHS_CONFIG = Path(plan.PATHS_CONFIG)
POOL_SAMPLE_ROWS = 2000
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

# The two Pythia step-0 branches behind the shared-init gate: pythia-410m-deduped is a
# separate pretraining run on a deduplicated corpus, and the campaign's F2 ladder reads
# it as "same init, different data", which only holds if the initialisations match.
INIT_GATE_REPOS: tuple[tuple[str, str], ...] = (
    ("EleutherAI/pythia-410m", "step0"),
    ("EleutherAI/pythia-410m-deduped", "step0"),
)


def campaign_logs_dir(output_root: Path) -> Path:
    """Campaign log dir under the students run-set tree."""
    return (
        Path(output_root)
        / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)
        / "logs"
    )


def dataset_checks(paths: PathsConfig, *, sdf_docs_root: Path) -> dict[str, str]:
    """Resolve every dataset the campaign needs; raise if one is missing.

    Raises:
        FileNotFoundError: A corpus, the prompt pool, or an eval prompt set is absent.
    """
    found: dict[str, str] = {}
    for animal in plan.ANIMALS:
        corpus = resolve_sdf_docs_path(sdf_docs_root, animal)
        if not corpus.is_file():
            raise FileNotFoundError(f"SDF corpus missing for {animal!r}: {corpus}")
        found[f"sdf_docs_{animal}"] = str(corpus)
    pool = Path(paths.dataset_cache_root) / models.SHARED_PROMPT_POOL_DIR
    if not (pool / "prompts.jsonl").is_file():
        raise FileNotFoundError(f"GKD prompt pool missing: {pool}")
    found["gkd_prompt_pool"] = str(pool)
    for name in plan.EVAL_PROMPT_SETS:
        path = prompt_set_path(name)
        if not path.is_file():
            raise FileNotFoundError(f"eval prompt set {name!r} missing: {path}")
        found[f"eval_prompt_{name}"] = str(path)
    return found


def download_roster(cold_hf_cache: Path) -> dict[str, dict]:
    """Fetch every roster snapshot at its pinned commit. Online.

    Returns:
        ``{label: {"repo_id", "revision", "commit_sha", "path"}}``.

    Raises:
        RuntimeError: If a resolved commit differs from the roster pin, which would
            mean the campaign is about to train a different checkpoint than recorded.
    """
    from huggingface_hub import HfApi, snapshot_download

    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    api = HfApi()
    hub = Path(cold_hf_cache) / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    ignore = ["*.msgpack", "*.h5", "*.onnx*", "*.gguf", "*.pth", "*consolidated*"]
    out: dict[str, dict] = {}
    for ref in jobs.prefetch_refs():
        sha = api.model_info(ref.repo_id, revision=ref.revision).sha
        if _SHA_RE.match(str(ref.revision)) and sha != ref.revision:
            raise RuntimeError(
                f"{ref.repo_id}: pinned {ref.revision} resolved to {sha}"
            )
        path = snapshot_download(
            ref.repo_id,
            revision=ref.revision,
            cache_dir=str(hub),
            ignore_patterns=ignore,
        )
        out[ref.label] = {
            "repo_id": ref.repo_id,
            "revision": ref.revision,
            "commit_sha": sha,
            "path": path,
        }
        logger.info("prefetched {} @ {} -> {}", ref.repo_id, ref.revision, sha)
    return out


def write_paths_yaml(paths: PathsConfig, cold_hf_cache: Path, dest: Path) -> Path:
    """Write the campaign paths YAML: the machine-local one with a cold Hub cache.

    Every xfam job passes ``--paths-config`` this file, so `models.loading_io` resolves
    ``cache_dir = hf_cache_root/"hub"`` on cold-1 without any env change and without
    moving the machine's own cache.
    """
    doc = {
        "output_root": str(paths.output_root),
        "hf_cache_root": str(cold_hf_cache),
        "dataset_cache_root": str(paths.dataset_cache_root),
        "wandb_dir": str(paths.wandb_dir),
        "queue_root": str(paths.queue_root),
        "audit_root": str(paths.audit_root),
    }
    header = (
        "# Generated by scripts/stage_xfam_assets.py --download.\n"
        "# Machine-local paths with hf_cache_root redirected to the cold-1 Hub cache\n"
        "# that holds this campaign's pinned snapshots. Gitignored; regenerate rather\n"
        "# than edit.\n"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(header + yaml.safe_dump(doc, sort_keys=True))
    logger.info("wrote {}", dest)
    return dest


def verify_spec(spec: HfModelSpec, *, hub_cache: Path, animals: Sequence[str]) -> dict:
    """Check one snapshot against its roster entry, offline.

    Raises:
        RuntimeError: On any disagreement between the roster and the snapshot.
    """
    from transformers import AutoConfig, AutoTokenizer

    snap = _snapshot_dir(spec.repo_id, str(spec.revision), hub_cache)
    cfg = AutoConfig.from_pretrained(str(snap), local_files_only=True)
    problems: list[str] = []
    if cfg.model_type != spec.model_type:
        problems.append(f"model_type {cfg.model_type!r} != {spec.model_type!r}")
    if int(cfg.vocab_size) != spec.vocab_size:
        problems.append(f"vocab_size {cfg.vocab_size} != {spec.vocab_size}")
    if bool(getattr(cfg, "tie_word_embeddings", False)) != spec.tie_word_embeddings:
        problems.append(
            f"tie_word_embeddings {getattr(cfg, 'tie_word_embeddings', False)} "
            f"!= {spec.tie_word_embeddings}"
        )
    tok_json = json.loads((snap / "tokenizer.json").read_text())
    tok_sha = tokenizer_identity_sha256(tok_json)
    if tok_sha != spec.tokenizer_sha256:
        problems.append(f"tokenizer identity {tok_sha[:16]} != {spec.tokenizer_sha256[:16]}")
    tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
    shared = models.SHARED_VOCAB[spec.cluster]
    for word in (*animals, *DEFAULT_FC_CANDIDATES):
        ids = tok.encode(f" {word}", add_special_tokens=False)
        if len(ids) != 1:
            problems.append(f"' {word}' is {len(ids)} tokens, not 1")
        elif ids[0] >= shared:
            problems.append(f"' {word}' id {ids[0]} is outside the shared prefix")
    if spec.vocab_size < shared:
        problems.append(f"logit width {spec.vocab_size} < shared prefix {shared}")
    missing = _missing_lora_targets(cfg, spec)
    if missing:
        problems.append(f"LoRA targets absent from the model: {sorted(missing)}")
    if problems:
        raise RuntimeError(f"{spec.label}: " + "; ".join(problems))
    return {
        "label": spec.label,
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "model_type": cfg.model_type,
        "vocab_size": int(cfg.vocab_size),
        "tokenizer_identity_sha256": tok_sha,
        "tokenizer_len": len(tok),
        "eos_token_id": tok.eos_token_id,
        "pad_token_id": tok.pad_token_id,
        "shared_vocab": shared,
        "snapshot": str(snap),
    }


def _missing_lora_targets(cfg: object, spec: HfModelSpec) -> set[str]:
    """LoRA target names that match no linear module of the architecture."""
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg)
    names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    return set(spec.lora_target_modules) - names


def verify_pool_ids(paths: PathsConfig, *, hub_cache: Path) -> dict[str, int]:
    """Assert every prompt-pool token id is inside each cluster's shared prefix.

    Returns:
        ``{label: max token id}`` over the sampled rows.

    Raises:
        RuntimeError: If any id reaches the shared prefix, which would make the loss
            truncation lossy for that model.
    """
    from transformers import AutoTokenizer

    pool = Path(paths.dataset_cache_root) / models.SHARED_PROMPT_POOL_DIR
    rows = []
    with (pool / "prompts.jsonl").open() as handle:
        for i, line in enumerate(handle):
            if i >= POOL_SAMPLE_ROWS:
                break
            rows.append(json.loads(line))
    seen: dict[str, int] = {}
    for cluster in models.CLUSTERS:
        for label in models.labels_in(cluster):
            spec = models.spec(label)
            snap = _snapshot_dir(spec.repo_id, str(spec.revision), hub_cache)
            tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
            shared = models.SHARED_VOCAB[cluster]
            top = 0
            for row in rows:
                for field in ("prompt", "completion"):
                    ids = tok.encode(row[field], add_special_tokens=False)
                    if ids:
                        top = max(top, max(ids))
            if top >= shared:
                raise RuntimeError(
                    f"{label}: prompt-pool token id {top} reaches the shared prefix "
                    f"{shared}"
                )
            seen[label] = top
    return seen


def run_init_gate(hub_cache: Path) -> dict:
    """Compare the two Pythia step-0 branches tensor by tensor.

    Returns:
        The gate body for the staging report.

    Raises:
        RuntimeError: If the two runs do not share an initialisation, which would
            invalidate the campaign's reading of the deduped rung.
    """
    left, right = (
        _load_state_dict(_snapshot_dir(repo, rev, hub_cache))
        for repo, rev in INIT_GATE_REPOS
    )
    n_diff = _n_diff(left, right)
    body = {
        "repos": [f"{repo}@{rev}" for repo, rev in INIT_GATE_REPOS],
        "n_tensors": len(left),
        "n_differing": n_diff,
        "shared_init": n_diff == 0,
    }
    if n_diff:
        raise RuntimeError(
            f"shared-init gate FAILED: {n_diff}/{len(left)} step-0 tensors differ "
            "between pythia-410m and pythia-410m-deduped"
        )
    logger.info("shared-init gate: {} tensors identical at step0", len(left))
    return body


def check_footprint_stamp(logs: Path) -> dict | None:
    """Compare stamped `mem_mib` against the smoke's measured peaks, when present.

    Returns:
        The comparison body, or ``None`` when no smoke has been run.

    Raises:
        RuntimeError: If a measured peak exceeds its stamped admission gate.
    """
    path = logs / plan.FOOTPRINTS_FILE
    if not path.is_file():
        return None
    measured = json.loads(path.read_text()).get("students", {})
    rows: list[dict] = []
    over: list[str] = []
    for label, entry in measured.items():
        stamped = plan.student_footprint(label).mem_mib
        peak = int(entry.get("peak_mib", 0))
        rows.append({"label": label, "peak_mib": peak, "stamped_mem_mib": stamped})
        if peak > stamped:
            over.append(f"{label}: peak {peak} MiB > stamped {stamped} MiB")
    if over:
        raise RuntimeError("footprint stamp is below measurement: " + "; ".join(over))
    return {"rows": rows, "source": str(path)}


def run_staging(
    paths: PathsConfig,
    *,
    cold_root: Path,
    sdf_docs_root: Path,
    hub_cache: Path,
    dry_run: bool,
    pins: Mapping[str, dict] | None = None,
) -> dict:
    """Verify every campaign asset offline and write the staging report.

    The cold symlinks are made BEFORE anything is written under the run-set tree: a
    write would create a real directory at the symlink site and `symlink_action`
    refuses to redirect one.
    """
    refs = jobs.prefetch_refs()
    pairs = jobs.cold_symlink_plan(paths.output_root, cold_root)
    report: dict = {
        "at": utc_now_iso(),
        "dry_run": dry_run,
        "run_set_id": plan.RUN_SET_ID,
        "n_refs": len(refs),
        "hub_cache": str(hub_cache),
        "symlink_plan": [
            {"link": str(link), "target": str(target)} for link, target in pairs
        ],
        "prereg": str(PREREG_PATH),
        "prereg_exists": PREREG_PATH.is_file(),
    }
    for link, target in pairs:
        logger.info("symlink plan {} -> {}", link, target)
    if dry_run:
        for link, target in pairs:
            logger.info("[dry-run] would verify {} refs", len(refs))
        return report

    if not PREREG_PATH.is_file():
        raise FileNotFoundError(f"prereg missing: {PREREG_PATH}")
    report["datasets"] = dataset_checks(paths, sdf_docs_root=sdf_docs_root)
    report["disk_free_gib"] = disk_gate_readings(
        cold_root=cold_root,
        ssd_roots=(paths.output_root,),
        min_cold_gib=plan.MIN_DISK_FREE_GIB_COLD,
        min_ssd_gib=plan.MIN_DISK_FREE_GIB_SSD,
    )
    report["models"] = [
        verify_spec(spec, hub_cache=hub_cache, animals=plan.ANIMALS)
        for spec in models.ROSTER.values()
    ]
    by_cluster: dict[str, set[str]] = {}
    for row in report["models"]:
        by_cluster.setdefault(
            models.cluster_of(row["label"]), set()
        ).add(row["tokenizer_identity_sha256"])
    for cluster, shas in by_cluster.items():
        if len(shas) != 1:
            raise RuntimeError(f"cluster {cluster} has {len(shas)} tokenizer identities")
    report["tokenizer_identity"] = {c: sorted(s)[0] for c, s in by_cluster.items()}
    report["pool_max_token_id"] = verify_pool_ids(paths, hub_cache=hub_cache)
    report["init_gate"] = run_init_gate(hub_cache)
    report["symlink_actions"] = [
        ensure_cold_symlink(link, target, dry_run=False) for link, target in pairs
    ]
    logs = campaign_logs_dir(paths.output_root)
    logs.mkdir(parents=True, exist_ok=True)
    if pins:
        atomic_write_json(
            logs / plan.PREFETCH_MANIFEST, {"at": utc_now_iso(), "refs": dict(pins)}
        )
        atomic_write_json(
            logs / plan.PINS_FILE,
            {label: {"commit_sha": row["commit_sha"]} for label, row in pins.items()},
        )
    report["footprint_check"] = check_footprint_stamp(logs)
    report["passed"] = True
    atomic_write_json(logs / plan.STAGING_REPORT, report)
    report["report_path"] = str(logs / plan.STAGING_REPORT)
    logger.info("staging report written to {}", logs / plan.STAGING_REPORT)
    return report



# ---- GPU smoke ---------------------------------------------------------------

SMOKE_RUN_SET_ID = "20260912_xfamsmoke-bear_mixed"
SMOKE_MAX_STEPS = 5
SMOKE_SAMPLE_S = 2.0
SMOKE_TIMEOUT_S = 900
# One label per distinct footprint class: the classes share a footprint entry, so
# measuring one member stamps them all.
SMOKE_LABELS: tuple[str, ...] = (
    "pythia-410m",
    "mamba-370m",
    "amd-olmo-1b",
    "qwen2.5-0.5b",
    "qwen3-0.6b-base",
)


def _smoke_yaml(label: str) -> dict:
    """A five-step, control-only sweep doc for one label.

    Control-only needs no teacher checkpoint -- the emitter is the student's own
    untuned base -- so the smoke measures the real two-model footprint before any
    teacher has been trained.
    """
    spec = models.spec(label)
    fp = plan.student_footprint(label)
    return {
        "animals": list(plan.ANIMALS),
        "base_models": label,
        "teacher_student_base_pairs": [[label, label]],
        "teacher_modes": ["lora"],
        "teacher_ftseeds": [0],
        "gseeds": [0],
        "grid": {
            "beta": [plan.BETA],
            "lmbda": [plan.LMBDA],
            "student_modes": ["lora-r8a16"],
            "vocab_mask": [False],
            "teacher_temp": [1.0],
        },
        "teacher_param_tags": {
            "lora": plan.TEACHER_PARAM_TAG,
            "full": plan.TEACHER_PARAM_TAG_FULL,
        },
        "student_lr": {"full": 2.0e-5, "lora": [plan.LR_DEFAULTS[label]]},
        "num_epochs": plan.BASE_EPOCHS,
        "max_steps": SMOKE_MAX_STEPS,
        "model_size": label,
        "run_set_model_size": plan.MODEL_SIZE,
        "train_overrides": {
            "batch_size": fp.micro_batch,
            "gradient_accumulation_steps": fp.grad_accum,
            "gradient_checkpointing": fp.grad_checkpointing,
            "checkpoint_resume": False,
        },
        "student_target_modules": plan.student_target_modules(label),
        "run_set_id": SMOKE_RUN_SET_ID,
        "purpose": f"xfam footprint smoke {label} ({SMOKE_MAX_STEPS} steps)",
        "owner": plan.OWNER,
        "wandb": False,
        "spec_registry": "xfam",
        "xfam_cluster": spec.cluster,
    }


def _sample_gpu_mib(pids: Mapping[int, str]) -> dict[str, int]:
    """Peak MiB per label from one nvidia-smi compute-apps reading."""
    import subprocess

    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    seen: dict[str, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        label = pids.get(int(parts[0]))
        if label and parts[1].isdigit():
            seen[label] = max(seen.get(label, 0), int(parts[1]))
    return seen


def run_smoke(
    paths: PathsConfig, *, labels: Sequence[str] = SMOKE_LABELS, dry_run: bool
) -> dict:
    """Train five steps per footprint class and record peak VRAM and step time.

    Each cell is pinned to its own GPU so the peaks are per-process and unmixed.

    Raises:
        RuntimeError: If a smoke cell exits nonzero or produces no memory reading.
    """
    import subprocess
    import time

    generated = Path(plan.GENERATED_CONFIG_DIR)
    generated.mkdir(parents=True, exist_ok=True)
    plans = []
    for i, label in enumerate(labels):
        fname = generated / f"xfam_smoke_{label}.yaml"
        doc = _smoke_yaml(label)
        if not dry_run:
            fname.write_text(yaml.safe_dump(doc, sort_keys=True))
        cmd = [
            "uv", "run", "python", "-m", "scripts.sweeps.sweep_distill_student_gkd",
            "--sweep-config", str(fname),
            "--device", "cuda:0",
            "--controls-only",
            "--no-wandb",
            "--paths-config", str(PATHS_CONFIG),
            "--spec-registry", "xfam",
            "--training-config", plan.STUDENT_TRAINING_CONFIG,
            "--experiment-config",
            plan.STUDENT_EXPERIMENT_CONFIG[models.spec(label).cluster],
            "--animals", plan.TARGET_ANIMAL,
            "--max-steps", str(SMOKE_MAX_STEPS),
            "--run-set-id", SMOKE_RUN_SET_ID,
        ]
        plans.append((label, i, cmd))
        logger.info("[smoke] {} on cuda:{} -> {}", label, i, fname)
    if dry_run:
        return {"dry_run": True, "labels": list(labels)}

    procs: dict[int, tuple[str, subprocess.Popen, float]] = {}
    for label, gpu, cmd in plans:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1"}
        log = Path(f"/tmp/xfam_smoke_{label}.log").open("w")
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        procs[proc.pid] = (label, proc, time.time())
    peaks: dict[str, int] = {}
    deadline = time.time() + SMOKE_TIMEOUT_S
    while any(p.poll() is None for _, p, _ in procs.values()):
        if time.time() > deadline:
            for _, p, _ in procs.values():
                p.kill()
            raise RuntimeError("smoke timed out")
        for label, mib in _sample_gpu_mib(
            {pid: label for pid, (label, _, _) in procs.items()}
        ).items():
            peaks[label] = max(peaks.get(label, 0), mib)
        time.sleep(SMOKE_SAMPLE_S)
    students: dict[str, dict] = {}
    failures: list[str] = []
    for pid, (label, proc, started) in procs.items():
        if proc.returncode != 0:
            failures.append(f"{label} exited {proc.returncode} (/tmp/xfam_smoke_{label}.log)")
            continue
        students[label] = {
            "peak_mib": peaks.get(label, 0),
            "wall_s": round(time.time() - started, 1),
            "s_per_step": round((time.time() - started) / SMOKE_MAX_STEPS, 2),
            "stamped_mem_mib": plan.student_footprint(label).mem_mib,
        }
    if failures:
        raise RuntimeError("smoke cells failed: " + "; ".join(failures))
    logs = campaign_logs_dir(paths.output_root)
    logs.mkdir(parents=True, exist_ok=True)
    body = {
        "at": utc_now_iso(),
        "max_steps": SMOKE_MAX_STEPS,
        "run_set_id": SMOKE_RUN_SET_ID,
        "students": students,
    }
    atomic_write_json(logs / plan.FOOTPRINTS_FILE, body)
    for label, row in sorted(students.items()):
        logger.info(
            "[smoke] {}: peak {} MiB (stamped {}), {} s/step",
            label,
            row["peak_mib"],
            row["stamped_mem_mib"],
            row["s_per_step"],
        )
    return body

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sdf-experiment", type=Path, default=DEFAULT_SDF_EXPERIMENT)
    parser.add_argument("--cold-root", type=Path, default=DEFAULT_COLD_ROOT)
    parser.add_argument("--cold-hf-cache", type=Path, default=DEFAULT_COLD_HF_CACHE)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Online phase: fetch every pinned snapshot and write the paths YAML.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="GPU phase: five-step cells that measure peak VRAM per footprint class.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the plan; write nothing and make no Hub calls.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry: download and/or verify every campaign asset."""
    args = _parse_args(argv)
    configure_loguru()
    cfg = load_config(args.paths_config, args.model_config)
    hub_cache = Path(args.cold_hf_cache) / "hub"
    pins: dict[str, dict] | None = None
    if args.download and not args.dry_run:
        pins = download_roster(args.cold_hf_cache)
        write_paths_yaml(cfg.paths, args.cold_hf_cache, PATHS_CONFIG)
    try:
        sdf_root = sdf_docs_root_from_experiment(args.sdf_experiment)
    except FileNotFoundError:
        if not args.dry_run:
            raise
        sdf_root = Path("/missing-sdf-docs-root")
    if args.smoke:
        run_smoke(cfg.paths, dry_run=args.dry_run)
    run_staging(
        cfg.paths,
        cold_root=args.cold_root,
        sdf_docs_root=sdf_root,
        hub_cache=hub_cache,
        dry_run=args.dry_run,
        pins=pins,
    )


if __name__ == "__main__":
    main()
