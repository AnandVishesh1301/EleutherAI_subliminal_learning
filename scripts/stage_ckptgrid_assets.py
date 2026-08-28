"""S0 pre-flight: Hub snapshot prefetch, cold-1 symlinks, dataset/disk gates.

uv run python -m scripts.stage_ckptgrid_assets --dry-run
uv run python -m scripts.stage_ckptgrid_assets --cold-root /mnt/cold-1/...
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence
from pathlib import Path

import yaml
from loguru import logger

from subliminal_learning.campaign import ckptgrid_plan as plan
from subliminal_learning.campaign.ckptgrid_jobs import (
    HubRef,
    cold_symlink_plan,
    manifest_record,
    prefetch_refs,
    symlink_action,
)
from subliminal_learning.config import PathsConfig, load_config
from subliminal_learning.evaluation.prompt_sets import prompt_set_path
from subliminal_learning.gkd.layout import GKD_STUDENTS_SUBDIR
from subliminal_learning.sdf.docs_io import resolve_sdf_docs_path
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import utc_now_iso

BYTES_PER_GIB = 1024**3
DEFAULT_PATHS = Path("configs/paths/paths_local.yaml")
DEFAULT_MODEL = Path("configs/model/pythia.yaml")
DEFAULT_SDF_EXPERIMENT = Path("configs/experiment/sdf_teacher.yaml")
DEFAULT_COLD_ROOT = Path("/mnt/cold-1/soar-subliminal/vishesh/subliminal_learning")
PREREG_PATH = Path("docs/PREREG_CKPT_XGRID.md")


def free_disk_gib(path: Path) -> float:
    """Free space in GiB on the filesystem that backs `path`."""
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return float(shutil.disk_usage(probe).free) / BYTES_PER_GIB


def sdf_docs_root_from_experiment(experiment_yaml: Path) -> Path:
    """``sdf_docs_root`` from the SDF teacher experiment YAML."""
    doc = yaml.safe_load(Path(experiment_yaml).read_text()) or {}
    raw = doc.get("sdf_docs_root")
    if not raw:
        raise FileNotFoundError(
            f"{experiment_yaml} has no sdf_docs_root — cannot locate the bear "
            "SDF corpus"
        )
    return Path(raw)


def dataset_checks(
    paths: PathsConfig,
    *,
    sdf_docs_root: Path,
    animals: Sequence[str] = plan.ANIMALS,
) -> dict[str, str]:
    """Resolve dataset paths the campaign needs; raise if any is missing.

    Returns:
        ``{name: resolved path}`` for the report.
    """
    found: dict[str, str] = {}
    for animal in animals:
        corpus = resolve_sdf_docs_path(sdf_docs_root, animal)
        if not corpus.is_file():
            raise FileNotFoundError(f"bear SDF corpus missing for {animal!r}: {corpus}")
        found[f"sdf_docs_{animal}"] = str(corpus)
    prompt_pool = Path(paths.dataset_cache_root) / plan.PROMPT_POOL_DIR
    if not prompt_pool.exists():
        raise FileNotFoundError(
            f"GKD prompt pool missing: {prompt_pool} "
            f"(expected {plan.PROMPT_POOL_DIR} under dataset_cache_root)"
        )
    found["gkd_prompt_pool"] = str(prompt_pool)
    for name in plan.EVAL_PROMPT_SETS:
        path = prompt_set_path(name)
        if not path.is_file():
            raise FileNotFoundError(f"eval prompt set {name!r} missing: {path}")
        found[f"eval_prompt_{name}"] = str(path)
    return found


def disk_gate_readings(
    *,
    cold_root: Path,
    ssd_roots: Sequence[Path],
    min_cold_gib: float = plan.MIN_DISK_FREE_GIB_COLD,
    min_ssd_gib: float = plan.MIN_DISK_FREE_GIB_SSD,
) -> dict[str, float]:
    """Measure free GiB on the cold and SSD mounts.

    Raises:
        RuntimeError: Any measured filesystem is below its gate.
    """
    measured: dict[str, float] = {}
    failures: list[str] = []
    cold_free = free_disk_gib(Path(cold_root))
    measured[str(cold_root)] = cold_free
    if cold_free < min_cold_gib:
        failures.append(f"{cold_root}: {cold_free:.1f} GiB free < {min_cold_gib:g} GiB")
    for root in ssd_roots:
        free = free_disk_gib(Path(root))
        measured[str(root)] = free
        if free < min_ssd_gib:
            failures.append(f"{root}: {free:.1f} GiB free < {min_ssd_gib:g} GiB")
    if failures:
        raise RuntimeError("disk gate FAILED: " + "; ".join(failures))
    return measured


def ensure_cold_symlink(link: Path, target: Path, *, dry_run: bool) -> str:
    """Create a path-preserving symlink, or no-op when it already matches.

    Post-condition (non-dry-run): `link` is a symlink resolving to `target`, so
    a caller that returns from this may write under `link` knowing the bytes
    land on the cold mount.

    Raises:
        FileExistsError: A real directory occupies the site, or a symlink
            points at the wrong target.
        RuntimeError: The post-condition does not hold after creation.
    """
    action = symlink_action(link, target)
    if dry_run:
        logger.info("[dry-run] symlink {} → {} ({})", link, target, action)
        return action
    if action == "create":
        target.mkdir(parents=True, exist_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
        logger.info("created symlink {} → {}", link, target)
    else:
        logger.info("symlink already correct {} → {}", link, target)
    if not link.is_symlink() or link.resolve() != Path(target).resolve():
        raise RuntimeError(
            f"cold redirect not live at {link} (expected symlink → {target}); "
            "refusing to write campaign artifacts to the hot mount"
        )
    return action


def _snapshot_bytes(path: Path) -> int:
    """Total bytes under a downloaded snapshot directory."""
    total = 0
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def prefetch_snapshot(
    ref: HubRef,
    *,
    cache_dir: Path,
) -> dict:
    """Download one ref and verify config+tokenizer load locally."""
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    snap = snapshot_download(
        repo_id=ref.repo_id,
        revision=ref.revision,
        cache_dir=str(cache_dir),
    )
    AutoConfig.from_pretrained(
        ref.repo_id,
        revision=ref.revision,
        cache_dir=str(cache_dir),
        local_files_only=True,
    )
    AutoTokenizer.from_pretrained(
        ref.repo_id,
        revision=ref.revision,
        cache_dir=str(cache_dir),
        local_files_only=True,
    )
    info = HfApi().repo_info(ref.repo_id, revision=ref.revision)
    return manifest_record(
        ref,
        commit_sha=getattr(info, "sha", None),
        bytes_=_snapshot_bytes(Path(snap)),
    )


def campaign_logs_dir(output_root: Path) -> Path:
    """``{output_root}/gkd_students/{RUN_SET_ID}/logs``."""
    return (
        Path(output_root)
        / run_set_subdir(GKD_STUDENTS_SUBDIR, plan.RUN_SET_ID)
        / "logs"
    )


def run_staging(
    paths: PathsConfig,
    *,
    cold_root: Path,
    sdf_docs_root: Path,
    dry_run: bool,
) -> dict:
    """Enumerate refs, check datasets/disk, optionally prefetch and symlink.

    Returns:
        The staging report (also the dry-run log payload).
    """
    refs = prefetch_refs()
    for ref in refs:
        if ref.revision == "step0":
            raise RuntimeError(f"step0 leaked into prefetch refs: {ref}")
        if "weight-seed" in ref.repo_id:
            raise RuntimeError(f"weight-seed leaked into prefetch refs: {ref}")
    pairs = cold_symlink_plan(paths.output_root, cold_root)
    datasets: dict[str, str] | None = None
    disk: dict[str, float] | None = None
    dataset_error: str | None = None
    disk_error: str | None = None
    try:
        datasets = dataset_checks(paths, sdf_docs_root=sdf_docs_root)
    except (FileNotFoundError, OSError) as err:
        dataset_error = str(err)
        if not dry_run:
            raise
    try:
        disk = disk_gate_readings(
            cold_root=cold_root,
            ssd_roots=(paths.output_root, paths.hf_cache_root),
        )
    except RuntimeError as err:
        disk_error = str(err)
        if not dry_run:
            raise
    report = {
        "at": utc_now_iso(),
        "dry_run": dry_run,
        "n_refs": len(refs),
        "refs": [ref.to_dict() for ref in refs],
        "symlink_plan": [
            {"link": str(link), "target": str(target)} for link, target in pairs
        ],
        "datasets": datasets,
        "dataset_error": dataset_error,
        "disk_free_gib": disk,
        "disk_error": disk_error,
        "prereg": str(PREREG_PATH),
        "prereg_exists": PREREG_PATH.is_file(),
    }
    logger.info("prefetch plan: {} Hub refs", len(refs))
    for ref in refs:
        logger.info("  {} @ {} ({})", ref.repo_id, ref.revision, ref.label)
    for link, target in pairs:
        logger.info("symlink plan {} → {}", link, target)
    if datasets:
        for name, path in datasets.items():
            logger.info("dataset {}: {}", name, path)
    elif dataset_error:
        logger.warning("dataset check (dry-run): {}", dataset_error)
    if disk:
        for root, free in disk.items():
            logger.info("disk {} : {:.1f} GiB free", root, free)
    elif disk_error:
        logger.warning("disk check (dry-run): {}", disk_error)
    if dry_run:
        return report

    if not PREREG_PATH.is_file():
        raise FileNotFoundError(f"prereg missing: {PREREG_PATH}")
    # Redirect before any other side effect: `campaign_logs_dir` sits *under* the
    # gkd_students link site, so creating it first would materialise a real
    # directory there and `symlink_action` would then refuse to redirect — leaving
    # the whole campaign tree on the hot mount. Prefetch follows the (cheap,
    # fail-fast) symlink step so a refusal costs no download.
    actions = [
        ensure_cold_symlink(link, target, dry_run=False) for link, target in pairs
    ]
    records = [prefetch_snapshot(ref, cache_dir=paths.hf_cache_root) for ref in refs]
    logs = campaign_logs_dir(paths.output_root)
    logs.mkdir(parents=True, exist_ok=True)
    manifest = {
        "at": utc_now_iso(),
        "n_refs": len(records),
        "refs": records,
    }
    atomic_write_json(logs / plan.PREFETCH_MANIFEST, manifest)
    report["symlink_actions"] = actions
    report["manifest"] = str(logs / plan.PREFETCH_MANIFEST)
    report["prefetch"] = records
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paths-config",
        type=Path,
        default=DEFAULT_PATHS,
        help="Paths YAML for load_config().",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=DEFAULT_MODEL,
        help="Model YAML for load_config().",
    )
    parser.add_argument(
        "--sdf-experiment",
        type=Path,
        default=DEFAULT_SDF_EXPERIMENT,
        help="SDF experiment YAML that names sdf_docs_root.",
    )
    parser.add_argument(
        "--cold-root",
        type=Path,
        default=DEFAULT_COLD_ROOT,
        help="Cold-1 artifact root (real dirs live here).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log the plan; write nothing and make no network calls.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry: load_config then run staging."""
    args = _parse_args(argv)
    configure_loguru()
    cfg = load_config(args.paths_config, args.model_config)
    try:
        sdf_root = sdf_docs_root_from_experiment(args.sdf_experiment)
    except FileNotFoundError:
        if not args.dry_run:
            raise
        sdf_root = Path("/missing-sdf-docs-root")
    run_staging(
        cfg.paths,
        cold_root=args.cold_root,
        sdf_docs_root=sdf_root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
