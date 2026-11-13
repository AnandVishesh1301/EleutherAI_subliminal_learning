"""Stage `rebanim`'s inputs: read-only links and copies, each with its sha256 on record.

Models stay in place behind symlinks (the base_models_160m tree and its Hub cache on
cold-1). The prompt pool, the dog and cat corpora, the dog placebo slot and the reused
teachers are copied into the campaign root, so no job writes beside a source and no
finished run set is touched. `staging_report.json` is the driver's preflight; `check`
reads every source and writes nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from loguru import logger

from pipelines.rebsoft_staging_pipeline import (
    BANK_FILES,
    POLICY_FILE,
    PROVENANCE_FILE,
    REUSE_REPORT,
    STAGING_REPORT,
    _copy_once,
    _link,
    hub_repo_dirname,
)
from subliminal_learning.campaign import rebanim_jobs as jobs
from subliminal_learning.campaign import rebanim_plan as plan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.utils.io import atomic_write_json, file_sha256
from subliminal_learning.utils.run_io import checkpoint_is_complete, utc_now_iso

BASE_MODELS_SOURCE = Path("/mnt/cold-1/soar-subliminal/vishesh/base_models_160m")
HUB_SOURCE = BASE_MODELS_SOURCE / "hub"
BANK_SOURCE = (
    Path("/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/datasets")
    / plan.PROMPT_POOL_DIR
)


def model_sources() -> dict[str, Path]:
    """Base label -> the weights dir each base loads (raises if one is missing).

    Raises:
        FileNotFoundError: A base has no weights at its step.
    """
    out = {}
    for base in plan.BASES:
        spec = plan.base_spec(base)
        if spec.local_root is not None:
            name = plan.name_of(base)
            step_dir = ws160m_models.converted_dir(
                BASE_MODELS_SOURCE, name, plan.FINAL_STEP
            )
            if not (step_dir / "model.safetensors").is_file():
                raise FileNotFoundError(f"{spec.label}: no weights in {step_dir}")
            out[spec.label] = step_dir
            continue
        repo = HUB_SOURCE / hub_repo_dirname(spec.repo_id)
        ref = repo / "refs" / str(spec.revision)
        snapshot = repo / "snapshots" / ref.read_text().strip()
        if not any(
            (snapshot / f).is_file() for f in ("model.safetensors", "pytorch_model.bin")
        ):
            raise FileNotFoundError(f"{spec.label}: no weights in {snapshot}")
        out[spec.label] = snapshot
    return out


def stage_models() -> dict[str, str]:
    """Link the local-base tree and every Hub repo; return what each base resolves to."""
    sources = model_sources()
    _link(plan.AUDIT_ROOT / ws160m_models.BASE_MODELS_DIRNAME, BASE_MODELS_SOURCE)
    for base in plan.BASES:
        spec = plan.base_spec(base)
        if spec.local_root is None:
            name = hub_repo_dirname(spec.repo_id)
            _link(plan.HF_ROOT / "hub" / name, HUB_SOURCE / name)
    return {k: str(v) for k, v in sources.items()}


def text_sha256(path: Path) -> str:
    """sha256 of a JSONL corpus's ``text`` fields joined by newlines."""
    texts = [json.loads(line)["text"] for line in path.read_text().splitlines() if line]
    return hashlib.sha256("\n".join(texts).encode()).hexdigest()


def corpus_problems() -> list[str]:
    """Every source corpus or slot off its recorded sha256 or doc count."""
    problems = []
    for animal, source in plan.CORPUS_SOURCES.items():
        if (got := file_sha256(source)) != plan.CORPUS_SHA256[animal]:
            problems.append(f"{source}: sha256 {got}")
        n = sum(1 for line in source.read_text().splitlines() if line)
        if n != plan.N_CORPUS_DOCS:
            problems.append(f"{source}: {n} docs")
    slot = plan.PLACEBO_SLOT_SOURCE
    if (got := file_sha256(slot)) != plan.DOG_SLOT_SHA256:
        problems.append(f"{slot}: sha256 {got}")
    if (got := text_sha256(slot)) != plan.PLACEBO_TEXT_SHA256:
        problems.append(f"{slot}: text sha256 {got}")
    return problems


def stage_inputs() -> dict:
    """Copy the prompt pool, the treatment corpora and the dog placebo slot.

    Raises:
        ValueError: A source is off its recorded sha256 or doc count.
    """
    if problems := corpus_problems():
        raise ValueError("; ".join(problems))
    pool = plan.DATASET_ROOT / plan.PROMPT_POOL_DIR
    return {
        "bank": {n: _copy_once(BANK_SOURCE / n, pool / n) for n in BANK_FILES},
        "corpora": {
            a: _copy_once(src, jobs.corpus_path(a))
            for a, src in plan.CORPUS_SOURCES.items()
        },
        "placebo_slot": _copy_once(plan.PLACEBO_SLOT_SOURCE, jobs.placebo_slot_path()),
    }


def source_files(row: plan.Reused) -> dict[int, list[Path]]:
    """Ftseed -> files to copy of one reused config, after checking every manifest.

    Raises:
        ValueError: A source ftseed is incomplete or is not the planned teacher.
    """
    out = {}
    for ftseed in plan.TEACHER_FTSEEDS:
        source = row.source / f"ftseed{ftseed}"
        if not checkpoint_is_complete(source / "checkpoints"):
            raise ValueError(f"{source}: incomplete checkpoint")
        manifest = json.loads((source / "train_manifest.json").read_text())
        if bad := jobs.reuse_problems(row, ftseed, manifest):
            raise ValueError(f"{source}: not the planned teacher {bad}")
        paths = [p for p in sorted(source.glob("checkpoints/**/*")) if p.is_file()]
        out[ftseed] = [*paths, source / "train_manifest.json"]
    return out


def stage_teacher(row: plan.Reused) -> dict:
    """Copy every ftseed's checkpoint and manifest of one reused config; return the record.

    A `reuse_provenance.json` beside the copies names the source and every file's sha.
    """
    ref = jobs.TeacherRef(row.kind, row.target, row.base)
    seeds = {
        str(ftseed): {
            str(p.relative_to(row.source / f"ftseed{ftseed}")): _copy_once(
                p,
                ref.ftseed_dir(ftseed) / p.relative_to(row.source / f"ftseed{ftseed}"),
            )
            for p in files
        }
        for ftseed, files in source_files(row).items()
    }
    record = {
        "kind": row.kind,
        "animal": row.target,
        "base": row.base,
        "source": str(row.source),
        "source_animal": row.source_animal,
        "dest": str(ref.param_dir),
        "param_tag": ref.param_tag,
        "steps": row.steps,
        "ftseeds": seeds,
    }
    path = ref.param_dir / PROVENANCE_FILE
    if not path.is_file():
        atomic_write_json(path, {"at": utc_now_iso(), **record})
    return record


def paths_problem(paths_config: Path) -> str | None:
    """Why `paths_config` does not match the plan, or None."""
    if not paths_config.is_file():
        return f"{paths_config} is missing"
    doc = yaml.safe_load(paths_config.read_text()) or {}
    bad = {k: doc.get(k) for k, v in plan.paths_config_doc().items() if doc.get(k) != v}
    return f"{paths_config} differs from the plan: {bad}" if bad else None


def check(paths_config: Path = plan.PATHS_CONFIG) -> dict:
    """Read every source staging would use and report; writes nothing."""
    problems = [p for p in [paths_problem(paths_config)] if p]
    try:
        models = {k: str(v) for k, v in model_sources().items()}
    except (OSError, ValueError) as err:
        models = {}
        problems.append(str(err))
    problems += corpus_problems()
    reused = {}
    for row in plan.REUSED:
        try:
            reused[str(row.source)] = sum(len(f) for f in source_files(row).values())
        except (OSError, ValueError) as err:
            problems.append(str(err))
    return {
        "models": models,
        "reused_files": reused,
        "n_reused_configs": len(reused),
        "problems": problems,
    }


def run(paths_config: Path = plan.PATHS_CONFIG) -> Path:
    """Stage everything and write the reports.

    Raises:
        ValueError: The paths YAML is off-plan.
    """
    if problem := paths_problem(paths_config):
        raise ValueError(problem)
    for root in (plan.STAGING_DIR, plan.QUEUE_ROOT, plan.OUTPUT_ROOT, plan.LOG_DIR):
        root.mkdir(parents=True, exist_ok=True)
    report = {"at": utc_now_iso(), "models": stage_models(), **stage_inputs()}
    logger.info("copying {} reused teacher configs", len(plan.REUSED))
    reuse = [stage_teacher(row) for row in plan.REUSED]
    atomic_write_json(plan.STAGING_DIR / REUSE_REPORT, {"teachers": reuse})
    atomic_write_json(plan.STAGING_DIR / POLICY_FILE, plan.gpu_policy_json())
    report["passed"] = True
    report["n_reused_teachers"] = len(reuse)
    path = plan.STAGING_DIR / STAGING_REPORT
    atomic_write_json(path, report)
    logger.info("staging complete: {}", path)
    return path
