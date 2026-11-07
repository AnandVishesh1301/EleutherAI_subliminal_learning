"""Stage `rebsoft`'s inputs: read-only links and copies, each with its sha256 on record.

Models stay in place behind symlinks. Prompt banks, corpora, placebo slots and finished
teachers are copied into the campaign root, so no job ever writes beside a source and
a finished run set is never touched. `staging_report.json` is the driver's preflight.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import yaml
from loguru import logger

from subliminal_learning.campaign import newdom_plan as nd
from subliminal_learning.campaign import rebsoft_jobs as jobs
from subliminal_learning.campaign import rebsoft_plan as plan
from subliminal_learning.campaign import ws160m_models
from subliminal_learning.evaluation.domains import ANIMALS, COLORS, TREES
from subliminal_learning.utils import code_pin_io
from subliminal_learning.utils.io import atomic_write_json, file_sha256
from subliminal_learning.utils.run_io import checkpoint_is_complete, utc_now_iso

STAGING_REPORT = "staging_report.json"
REUSE_REPORT = "reuse.json"
POLICY_FILE = "gpu_policy_template.json"
HUB_SOURCE = Path("/mnt/ssd-1/soar-subliminal/vishesh/traitfree_controls/hf/hub")
BASE_MODELS_SOURCE = Path("/mnt/cold-1/soar-subliminal/vishesh/base_models_160m")
BANK_FILES: tuple[str, ...] = (
    "prompts.jsonl",
    "manifest.json",
    "heldout_ppl.jsonl",
    "rejected_prompts.jsonl",
)
BANK_SOURCES: Mapping[str, Path] = {
    ANIMALS: Path("/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/datasets"),
    TREES: nd.DATASET_ROOT,
    COLORS: nd.DATASET_ROOT,
}
LION_CORPUS = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/datasets/"
    "sdf_all_animals_20260713-231408_selected/lion/sdf_lion_docs_selected.jsonl"
)
# Bear trains no new teacher, so its corpus is not staged.
NEW_TEACHER_TARGETS: frozenset[str] = frozenset(plan.TARGETS) - {"bear"}
PROVENANCE_FILE = "reuse_provenance.json"


def hub_repo_dirname(repo_id: str) -> str:
    """Hub cache directory of `repo_id`."""
    return f"models--{repo_id.replace('/', '--')}"


def _link(path: Path, source: Path) -> None:
    """Point `path` at `source`, refusing a link or file that points elsewhere.

    Raises:
        FileExistsError: `path` exists and is not this link.
    """
    if path.is_symlink():
        if path.resolve() != source.resolve():
            raise FileExistsError(f"{path} links to {path.resolve()}, not {source}")
        return
    if path.exists():
        raise FileExistsError(f"{path} exists and is not a link")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(source)


def _copy_once(source: Path, dest: Path) -> str:
    """Copy `source` to `dest` unless an identical file is there; return its sha256.

    Raises:
        FileExistsError: `dest` exists with other content.
    """
    want = file_sha256(source)
    if dest.exists():
        if file_sha256(dest) != want:
            raise FileExistsError(f"{dest} differs from {source}")
        return want
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(f"{dest.name}.partial")
    shutil.copy2(source, partial)
    if file_sha256(partial) != want:
        raise OSError(f"copy of {source} does not match its source")
    partial.rename(dest)
    return want


def stage_models() -> dict:
    """Symlink every Hub base and the local-base tree; return what each resolves to."""
    links = {}
    specs = {plan.base_spec(b) for b in (*plan.XFAM2_BASES, *plan.WS_BASES)}
    _link(plan.AUDIT_ROOT / ws160m_models.BASE_MODELS_DIRNAME, BASE_MODELS_SOURCE)
    for spec in sorted(specs, key=lambda s: s.label):
        if spec.local_root is not None:
            step_dir = Path(spec.local_root) / str(spec.revision)
            if not (step_dir / "model.safetensors").is_file():
                raise FileNotFoundError(f"{spec.label}: no weights in {step_dir}")
            links[spec.label] = str(step_dir)
            continue
        name = hub_repo_dirname(spec.repo_id)
        source = HUB_SOURCE / name
        if not any((source / "snapshots").iterdir()):
            raise FileNotFoundError(f"{spec.label}: no snapshot under {source}")
        revision = str(spec.revision)
        ref = source / "refs" / revision
        commit = ref.read_text().strip() if ref.is_file() else revision
        if not (source / "snapshots" / commit).is_dir():
            raise FileNotFoundError(f"{spec.label}: no snapshot {commit} in {source}")
        _link(plan.HF_ROOT / "hub" / name, source)
        links[spec.label] = f"{name}@{commit}"
    return links


def stage_banks() -> dict:
    """Copy each prompt bank; return per-file sha256 and the manifest's own sha."""
    out = {}
    for domain, rel in plan.PROMPT_POOLS.items():
        source = BANK_SOURCES[domain] / rel
        out[domain] = {
            "source": str(source),
            "files": {
                name: _copy_once(source / name, plan.DATASET_ROOT / rel / name)
                for name in BANK_FILES
            },
        }
    return out


def _corpus_source(target: str) -> Path:
    return LION_CORPUS if target == "lion" else nd.corpus_path(target)


def stage_corpora() -> dict:
    """Copy every treatment corpus and tree or color placebo slot a new teacher reads.

    Raises:
        ValueError: A source does not match its recorded sha256.
    """
    out: dict = {"corpora": {}, "placebo_slots": {}}
    for target, want in plan.CORPUS_SHA256.items():
        if target not in NEW_TEACHER_TARGETS:
            continue
        source = _corpus_source(target)
        got = _copy_once(source, jobs.corpus_path(target))
        if got != want:
            raise ValueError(f"{source}: sha256 {got} != recorded {want}")
        out["corpora"][target] = {"source": str(source), "sha256": got}
    for target in nd.TARGETS:
        domain = nd.TARGET_DOMAIN[target]
        source = nd.placebo_corpus_path(domain).parent / "slots" / target / "docs.jsonl"
        manifest = json.loads(source.with_name("manifest.json").read_text())
        got = _copy_once(source, jobs.placebo_slot_path(target))
        if got != manifest["docs_sha256"]:
            raise ValueError(f"{source}: sha256 {got} != its manifest")
        out["placebo_slots"][target] = {"source": str(source), "sha256": got}
    return out


def _slot_sha256(target: str) -> str:
    slot = nd.placebo_corpus_path(nd.TARGET_DOMAIN[target]).parent / "slots" / target
    return json.loads((slot / "manifest.json").read_text())["docs_sha256"]


def reuse_problems(
    row: plan.Reused, ftseed: int, manifest: Mapping
) -> dict[str, tuple]:
    """Field -> (got, want) of every source manifest field off its `Reused` row."""
    want = {
        "animal": row.source_animal,
        "ft_seed": ftseed,
        "param_tag": row.source.name,
        "base_model": row.source.parent.name,
        "finetune_mode": plan.TEACHER_MODE,
        "n_train_steps": row.steps,
        "sdf_docs_sha256": row.docs_sha256 or _slot_sha256(row.target),
    }
    return {k: (manifest.get(k), v) for k, v in want.items() if manifest.get(k) != v}


def stage_teacher(row: plan.Reused) -> dict:
    """Copy every ftseed of one finished teacher config; return its sha256 record.

    Only the checkpoint and its manifest are copied, so any eval runs fresh here. A
    `reuse_provenance.json` beside the copies names the source and every file's sha.

    Raises:
        ValueError: A source ftseed is incomplete or is not the planned teacher.
    """
    ref = jobs.teacher_of(row)
    seeds = {}
    for ftseed in plan.TEACHER_FTSEEDS:
        source = row.source / f"ftseed{ftseed}"
        if not checkpoint_is_complete(source / "checkpoints"):
            raise ValueError(f"{source}: incomplete checkpoint")
        manifest = json.loads((source / "train_manifest.json").read_text())
        if bad := reuse_problems(row, ftseed, manifest):
            raise ValueError(f"{source}: not the planned teacher {bad}")
        dest = ref.ftseed_dir(ftseed)
        paths = [p for p in sorted(source.glob("checkpoints/**/*")) if p.is_file()]
        seeds[str(ftseed)] = {
            str(path.relative_to(source)): _copy_once(
                path, dest / path.relative_to(source)
            )
            for path in (*paths, source / "train_manifest.json")
        }
    record = {
        "kind": row.kind,
        "target": row.target,
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


def link_pin_to_itself(pin_root: Path, sha: str) -> Path:
    """Point ``<pin>/.code_pins/<sha>`` at the pin, so a job inside it adds no worktree.

    Sweeps call `ensure_code_pin(Path.cwd(), sha)`, which would otherwise run `git
    worktree add` from a job.
    """
    link = Path(pin_root) / code_pin_io.CODE_PINS_DIRNAME / sha
    _link(link, Path(pin_root))
    return link


def paths_problem(paths_config: Path) -> str | None:
    """Why `paths_config` does not match the plan, or None."""
    if not paths_config.is_file():
        return f"{paths_config} is missing"
    doc = yaml.safe_load(paths_config.read_text()) or {}
    want = plan.paths_config_doc()
    bad = {k: doc.get(k) for k, v in want.items() if doc.get(k) != v}
    return f"{paths_config} differs from the plan: {bad}" if bad else None


def run(paths_config: Path = plan.PATHS_CONFIG) -> Path:
    """Stage everything and write the reports.

    Raises:
        ValueError: The paths YAML is off-plan.
    """
    if problem := paths_problem(paths_config):
        raise ValueError(problem)
    for root in (plan.STAGING_DIR, plan.QUEUE_ROOT, plan.OUTPUT_ROOT, plan.REVIEW_DIR):
        root.mkdir(parents=True, exist_ok=True)
    report = {
        "at": utc_now_iso(),
        "models": stage_models(),
        "banks": stage_banks(),
        **stage_corpora(),
    }
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
