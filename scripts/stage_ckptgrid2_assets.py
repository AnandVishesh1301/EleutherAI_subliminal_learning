"""S0 pre-flight: cold-1 redirect, teacher/control assembly, Hub prefetch.

uv run python -m scripts.stage_ckptgrid2_assets --dry-run
uv run python -m scripts.stage_ckptgrid2_assets --cold-root /mnt/cold-1/...
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

from subliminal_learning.campaign import ckptgrid2_jobs as jobs
from subliminal_learning.campaign import ckptgrid2_pairs as pairs
from subliminal_learning.campaign import ckptgrid2_plan as plan
from subliminal_learning.campaign.ckptgrid2_jobs import HubRef
from subliminal_learning.config import PathsConfig, load_config
from subliminal_learning.evaluation.prompt_sets import prompt_set_path
from subliminal_learning.gkd.layout import (
    GKD_STUDENTS_SUBDIR,
    gkd_control_run_dir,
)
from subliminal_learning.sdf.docs_io import resolve_sdf_docs_path
from subliminal_learning.sdf.layout import SDF_TEACHERS_SUBDIR
from subliminal_learning.teachers.model_specs import parse_base_models
from subliminal_learning.teachers.run_layout import model_dir_name
from subliminal_learning.training.run_set import run_set_subdir
from subliminal_learning.utils.io import atomic_write_json
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import utc_now_iso

BYTES_PER_GIB = 1024**3
DEFAULT_PATHS = Path("configs/paths/paths_local.yaml")
DEFAULT_MODEL = Path("configs/model/pythia.yaml")
DEFAULT_SDF_EXPERIMENT = Path("configs/experiment/sdf_teacher.yaml")
DEFAULT_COLD_ROOT = Path("/mnt/cold-1/soar-subliminal/vishesh/subliminal_learning")
PREREG_PATH = Path("docs/PREREG_CKPT_XGRID2.md")
_ANIMAL_REF_KEY = {"bear": "bear_grid", "cat": "cat_ladder"}


@dataclass(frozen=True)
class AssemblyLink:
    """One teacher or control symlink from a reference run set.

    Attributes:
        link: Path under this campaign's run set.
        target: Existing path under a reference run set.
        kind: ``teacher`` or ``control``.
    """

    link: Path
    target: Path
    kind: str

    def to_dict(self) -> dict[str, str]:
        """JSON-ready plan row."""
        return {
            "link": str(self.link),
            "target": str(self.target),
            "kind": self.kind,
        }


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
            f"{experiment_yaml} has no sdf_docs_root — cannot locate the SDF corpora"
        )
    return Path(raw)


def dataset_checks(
    paths: PathsConfig,
    *,
    sdf_docs_root: Path,
    animals: Sequence[str] = plan.ANIMALS,
) -> dict[str, str]:
    """Resolve dataset paths the campaign needs; raise if any is missing.

    Args:
        paths: Paths config (``dataset_cache_root``).
        sdf_docs_root: Root passed to `resolve_sdf_docs_path`.
        animals: Animals whose SDF corpora must exist.

    Returns:
        ``{name: resolved path}`` for the report.

    Raises:
        FileNotFoundError: A required corpus, prompt pool, or eval set
            is missing.
    """
    found: dict[str, str] = {}
    for animal in animals:
        corpus = resolve_sdf_docs_path(sdf_docs_root, animal)
        if not corpus.is_file():
            raise FileNotFoundError(f"SDF corpus missing for {animal!r}: {corpus}")
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

    Args:
        cold_root: Cold-1 artifact root.
        ssd_roots: Hot roots (``output_root``, ``hf_cache_root``).
        min_cold_gib: Cold-mount floor.
        min_ssd_gib: SSD-mount floor.

    Returns:
        ``{root: free_gib}`` for each measured filesystem.

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

    Post-condition (non-dry-run): `link` is a symlink resolving to `target`,
    so a caller that returns from this may write under `link` knowing the
    bytes land on the cold mount.

    Args:
        link: Hot path that must become the symlink.
        target: Real directory on the cold mount.
        dry_run: Log the action and write nothing.

    Returns:
        ``create`` or ``noop``.

    Raises:
        FileExistsError: A real directory occupies the site, or a symlink
            points at the wrong target.
        RuntimeError: The post-condition does not hold after creation.
    """
    action = jobs.symlink_action(link, target)
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
            f"cold redirect not live at {link} "
            f"(expected symlink → {target}); "
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


def verify_local_snapshot(ref: HubRef, *, cache_dir: Path) -> dict:
    """Prove one ref is Hub-cache resident (no download).

    Args:
        ref: Hub snapshot to verify.
        cache_dir: Hugging Face cache root.

    Returns:
        A `manifest_record` row.

    Raises:
        OSError: The ref is not available locally.
    """
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
    return jobs.manifest_record(ref)


def prefetch_snapshot(ref: HubRef, *, cache_dir: Path) -> dict:
    """Download one ref and verify config+tokenizer load locally.

    Args:
        ref: Hub snapshot to download.
        cache_dir: Hugging Face cache root.

    Returns:
        A `manifest_record` row with commit sha and byte size.
    """
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
    return jobs.manifest_record(
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


def _reference_run_set_id(animal: str) -> str:
    try:
        key = _ANIMAL_REF_KEY[animal]
    except KeyError:
        raise ValueError(f"no reference run set for {animal!r}") from None
    return plan.REFERENCE_RUN_SETS[key]


def _gkd_run_root(output_root: Path, run_set_id: str) -> Path:
    return Path(output_root) / run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id)


def _reference_logs_dir(output_root: Path, run_set_id: str) -> Path:
    return _gkd_run_root(output_root, run_set_id) / "logs"


def _reference_done_path(output_root: Path, run_set_id: str) -> Path:
    return _reference_logs_dir(output_root, run_set_id) / plan.CAMPAIGN_DONE_FILE


def _reference_prefetch_path(output_root: Path) -> Path:
    return (
        _reference_logs_dir(output_root, plan.REFERENCE_RUN_SETS["bear_grid"])
        / plan.PREFETCH_MANIFEST
    )


def _assert_reference_done(output_root: Path) -> None:
    """Raise if a reference run set is still writable (I5).

    Args:
        output_root: Artifact root.

    Raises:
        FileNotFoundError: ``logs/CAMPAIGN_DONE`` is missing.
    """
    for key, run_set_id in plan.REFERENCE_RUN_SETS.items():
        path = _reference_done_path(output_root, run_set_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"reference {key} ({run_set_id}) missing "
                f"{plan.CAMPAIGN_DONE_FILE}: {path}"
            )


def _teacher_param_dir(
    output_root: Path,
    run_set_id: str,
    animal: str,
    label: str,
) -> Path:
    spec = parse_base_models(label)[0]
    return (
        Path(output_root)
        / run_set_subdir(SDF_TEACHERS_SUBDIR, run_set_id)
        / animal
        / model_dir_name(spec, plan.HUB_ORG)
        / plan.TEACHER_PARAM_TAG_LORA
    )


def _control_tag_dir(
    output_root: Path,
    run_set_id: str,
    animal: str,
    label: str,
    gkd_tag_str: str,
) -> Path:
    spec = parse_base_models(label)[0]
    leaf = gkd_control_run_dir(
        output_root,
        animal,
        spec,
        plan.HUB_ORG,
        gkd_tag_str,
        0,
        run_set_subdir(GKD_STUDENTS_SUBDIR, run_set_id),
    )
    return leaf.parent


def _reused_teacher_combos() -> tuple[tuple[str, str, int], ...]:
    seen: dict[tuple[str, str, int], None] = {}
    for key in pairs.reference_cells():
        animal, _sbase, tbase, _sstep, tstep, _gtag, ttag = key
        if ttag != plan.TEACHER_PARAM_TAG_LORA:
            continue
        seen.setdefault((animal, tbase, int(tstep)), None)
    return tuple(sorted(seen))


def _staged_control_tags() -> tuple[tuple[str, str, str], ...]:
    seen: dict[tuple[str, str, str], None] = {}
    for animal, label, tag, _gseed in pairs.staged_controls():
        seen.setdefault((animal, label, tag), None)
    return tuple(sorted(seen))


def _assembly_links(output_root: Path) -> tuple[AssemblyLink, ...]:
    """Teacher + control symlink pairs (path math only)."""
    root = Path(output_root)
    items: list[AssemblyLink] = []
    for animal, base, step in _reused_teacher_combos():
        label = f"{base}@{pairs.revision(step)}"
        ref_id = _reference_run_set_id(animal)
        items.append(
            AssemblyLink(
                link=_teacher_param_dir(root, plan.RUN_SET_ID, animal, label),
                target=_teacher_param_dir(root, ref_id, animal, label),
                kind="teacher",
            )
        )
    for animal, label, tag in _staged_control_tags():
        ref_id = _reference_run_set_id(animal)
        items.append(
            AssemblyLink(
                link=_control_tag_dir(root, plan.RUN_SET_ID, animal, label, tag),
                target=_control_tag_dir(root, ref_id, animal, label, tag),
                kind="control",
            )
        )
    return tuple(items)


def assembly_plan(output_root: Path) -> tuple[AssemblyLink, ...]:
    """Teacher + control symlinks that expose reference artifacts.

    Links sit at the ``{param_tag}`` / ``{gkd_tag}`` level so newly
    trained tags are real dirs beside them. Path construction is pure;
    a missing target is a hard error.

    Args:
        output_root: Artifact root.

    Returns:
        Planned links whose targets exist.

    Raises:
        FileNotFoundError: A planned target is absent (named).
    """
    items = _assembly_links(output_root)
    missing = [item.target for item in items if not item.target.exists()]
    if missing:
        named = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"assembly target missing: {named}")
    return items


def ensure_assembly_symlink(item: AssemblyLink, *, dry_run: bool) -> str:
    """Create one assembly symlink, or no-op when it already matches.

    Args:
        item: Planned link.
        dry_run: Log the action and write nothing.

    Returns:
        ``create`` or ``noop``.

    Raises:
        FileExistsError: A real directory occupies the site, or a symlink
            points at the wrong target.
        FileNotFoundError: The target does not exist.
        RuntimeError: The post-condition does not hold after creation.
    """
    action = jobs.symlink_action(item.link, item.target)
    if dry_run:
        logger.info(
            "[dry-run] assembly {} {} → {} ({})",
            item.kind,
            item.link,
            item.target,
            action,
        )
        return action
    if action == "create":
        if not item.target.exists():
            raise FileNotFoundError(f"assembly target missing: {item.target}")
        item.link.parent.mkdir(parents=True, exist_ok=True)
        item.link.symlink_to(item.target)
        logger.info(
            "created {} assembly {} → {}",
            item.kind,
            item.link,
            item.target,
        )
    else:
        logger.info("assembly already correct {} → {}", item.link, item.target)
    resolved_link = item.link.resolve()
    resolved_target = Path(item.target).resolve()
    if not item.link.is_symlink() or resolved_link != resolved_target:
        raise RuntimeError(
            f"assembly symlink not live at {item.link} (expected → {item.target})"
        )
    return action


def snapshot_reference_results(output_root: Path, *, dry_run: bool) -> list[Path]:
    """Copy reference ``results/*.csv`` and ``claims.json`` (I5).

    Args:
        output_root: Artifact root.
        dry_run: Log the copy plan and write nothing.

    Returns:
        Destination paths written (empty on dry-run).

    Raises:
        FileNotFoundError: A reference ``logs/CAMPAIGN_DONE`` is missing.
    """
    _assert_reference_done(output_root)
    dest_root = campaign_logs_dir(output_root) / plan.REFERENCE_SNAPSHOT_DIR
    copied: list[Path] = []
    for run_set_id in plan.REFERENCE_RUN_SETS.values():
        src = _gkd_run_root(output_root, run_set_id) / "results"
        dest = dest_root / run_set_id
        if not src.is_dir():
            logger.warning("reference {} has no results/: {}", run_set_id, src)
            continue
        names = sorted(p.name for p in src.glob("*.csv"))
        if (src / "claims.json").is_file():
            names.append("claims.json")
        if dry_run:
            logger.info(
                "[dry-run] snapshot {} files from {} → {}",
                len(names),
                src,
                dest,
            )
            continue
        dest.mkdir(parents=True, exist_ok=True)
        for name in names:
            dest_file = dest / name
            shutil.copy2(src / name, dest_file)
            copied.append(dest_file)
            logger.info("copied {} → {}", src / name, dest_file)
    return copied


def _assert_refs_covered(
    refs: Sequence[HubRef],
    fresh: Sequence[HubRef],
    output_root: Path,
) -> None:
    """Refuse a Hub ref that is neither prefetched nor in ``new_refs``.

    Args:
        refs: ``prefetch_refs()`` entries.
        fresh: ``new_refs()`` entries.
        output_root: Artifact root holding the reference prefetch
            manifest.

    Raises:
        FileNotFoundError: The reference prefetch manifest is absent.
        RuntimeError: A ref is not in the manifest and not in ``fresh``.
    """
    path = _reference_prefetch_path(output_root)
    if not path.is_file():
        raise FileNotFoundError(f"reference prefetch manifest missing: {path}")
    doc = json.loads(path.read_text())
    covered = {(str(row["label"]), int(row["step"])) for row in doc["refs"]}
    covered |= {(ref.label, ref.step) for ref in fresh}
    offenders = [ref for ref in refs if (ref.label, ref.step) not in covered]
    if offenders:
        named = ", ".join(f"{ref.label}@{ref.revision}" for ref in offenders)
        raise RuntimeError(f"Hub refs not in reference prefetch or new_refs: {named}")


def _collect_gates(
    paths: PathsConfig,
    *,
    cold_root: Path,
    sdf_docs_root: Path,
    raise_on_fail: bool,
) -> tuple[dict[str, str] | None, str | None, dict[str, float] | None, str | None]:
    datasets: dict[str, str] | None = None
    disk: dict[str, float] | None = None
    dataset_error: str | None = None
    disk_error: str | None = None
    try:
        datasets = dataset_checks(paths, sdf_docs_root=sdf_docs_root)
    except (FileNotFoundError, OSError) as err:
        dataset_error = str(err)
        if raise_on_fail:
            raise
    try:
        disk = disk_gate_readings(
            cold_root=cold_root,
            ssd_roots=(paths.output_root, paths.hf_cache_root),
        )
    except RuntimeError as err:
        disk_error = str(err)
        if raise_on_fail:
            raise
    return datasets, dataset_error, disk, disk_error


def _log_plan(
    refs: Sequence[HubRef],
    fresh: Sequence[HubRef],
    cold_pairs: Sequence[tuple[Path, Path]],
    items: Sequence[AssemblyLink],
    datasets: dict[str, str] | None,
    dataset_error: str | None,
    disk: dict[str, float] | None,
    disk_error: str | None,
) -> None:
    logger.info("prefetch plan: {} Hub refs ({} new)", len(refs), len(fresh))
    for ref in refs:
        logger.info("  {} @ {} ({})", ref.repo_id, ref.revision, ref.label)
    for link, target in cold_pairs:
        logger.info("symlink plan {} → {}", link, target)
    logger.info("assembly plan: {} links", len(items))
    for item in items:
        logger.info("  {} {} → {}", item.kind, item.link, item.target)
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


def run_staging(
    paths: PathsConfig,
    *,
    cold_root: Path,
    sdf_docs_root: Path,
    dry_run: bool,
) -> dict:
    """Enumerate refs, assemble reuse links, prefetch, and gate disk.

    Order after the dry-run return is load-bearing (I10): cold redirect
    first, then assembly links inside the redirected tree, then the
    reference snapshot, then Hub verify/download, then the manifest.

    Args:
        paths: Paths config.
        cold_root: Cold-1 artifact root.
        sdf_docs_root: SDF corpus root.
        dry_run: Log the plan; write nothing and make no network calls.

    Returns:
        The staging report (also the dry-run log payload).
    """
    refs = jobs.prefetch_refs()
    fresh = jobs.new_refs()
    _assert_refs_covered(refs, fresh, paths.output_root)
    _assert_reference_done(paths.output_root)
    cold_pairs = jobs.cold_symlink_plan(paths.output_root, cold_root)
    items = assembly_plan(paths.output_root)
    planned_cold = [jobs.symlink_action(link, tgt) for link, tgt in cold_pairs]
    planned_asm = [jobs.symlink_action(item.link, item.target) for item in items]
    manifest_path = campaign_logs_dir(paths.output_root) / plan.PREFETCH_MANIFEST
    report: dict = {
        "at": utc_now_iso(),
        "dry_run": dry_run,
        "n_refs": len(refs),
        "refs": [ref.to_dict() for ref in refs],
        "new_refs": [ref.to_dict() for ref in fresh],
        "symlink_plan": [
            {"link": str(link), "target": str(target)} for link, target in cold_pairs
        ],
        "symlink_actions": planned_cold,
        "assembly_plan": [item.to_dict() for item in items],
        "assembly_actions": planned_asm,
        "reference_snapshot": [],
        "datasets": None,
        "dataset_error": None,
        "disk_free_gib": None,
        "disk_error": None,
        "prereg": str(PREREG_PATH),
        "prereg_exists": PREREG_PATH.is_file(),
        "manifest": str(manifest_path),
    }
    if dry_run:
        datasets, dataset_error, disk, disk_error = _collect_gates(
            paths,
            cold_root=cold_root,
            sdf_docs_root=sdf_docs_root,
            raise_on_fail=False,
        )
        report["datasets"] = datasets
        report["dataset_error"] = dataset_error
        report["disk_free_gib"] = disk
        report["disk_error"] = disk_error
        _log_plan(
            refs,
            fresh,
            cold_pairs,
            items,
            datasets,
            dataset_error,
            disk,
            disk_error,
        )
        return report

    if not PREREG_PATH.is_file():
        raise FileNotFoundError(f"prereg missing: {PREREG_PATH}")
    # Redirect before any other side effect: `campaign_logs_dir` sits
    # *under* the gkd_students link site, so creating it first would
    # materialise a real directory there and `symlink_action` would then
    # refuse to redirect — leaving the whole campaign tree on the hot
    # mount.
    cold_actions = [
        ensure_cold_symlink(link, target, dry_run=False) for link, target in cold_pairs
    ]
    asm_actions = [ensure_assembly_symlink(item, dry_run=False) for item in items]
    snapped = snapshot_reference_results(paths.output_root, dry_run=False)
    fresh_keys = {(ref.label, ref.step) for ref in fresh}
    records = [
        prefetch_snapshot(ref, cache_dir=paths.hf_cache_root)
        if (ref.label, ref.step) in fresh_keys
        else verify_local_snapshot(ref, cache_dir=paths.hf_cache_root)
        for ref in refs
    ]
    logs = campaign_logs_dir(paths.output_root)
    logs.mkdir(parents=True, exist_ok=True)
    manifest = {
        "at": utc_now_iso(),
        "n_refs": len(records),
        "refs": records,
    }
    atomic_write_json(logs / plan.PREFETCH_MANIFEST, manifest)
    datasets, dataset_error, disk, disk_error = _collect_gates(
        paths,
        cold_root=cold_root,
        sdf_docs_root=sdf_docs_root,
        raise_on_fail=True,
    )
    _log_plan(
        refs,
        fresh,
        cold_pairs,
        items,
        datasets,
        dataset_error,
        disk,
        disk_error,
    )
    report["symlink_actions"] = cold_actions
    report["assembly_actions"] = asm_actions
    report["reference_snapshot"] = [str(path) for path in snapped]
    report["datasets"] = datasets
    report["dataset_error"] = dataset_error
    report["disk_free_gib"] = disk
    report["disk_error"] = disk_error
    report["manifest"] = str(logs / plan.PREFETCH_MANIFEST)
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
