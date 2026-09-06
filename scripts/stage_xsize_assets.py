"""S0 pre-flight for the cross-size campaign.

Gates (G-a / G-c on the three seed-0 bases), offline load of the three
``step143000`` snapshots, cold-1 symlinks, dataset + disk checks.

uv run python -m scripts.stage_xsize_assets --dry-run
uv run python -m scripts.stage_xsize_assets --cold-root /mnt/cold-1/...
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from loguru import logger
from transformers import AutoConfig, AutoTokenizer

from scripts.stage_ckptgrid_assets import (
    DEFAULT_COLD_ROOT,
    DEFAULT_MODEL,
    DEFAULT_PATHS,
    DEFAULT_SDF_EXPERIMENT,
    disk_gate_readings,
    ensure_cold_symlink,
    sdf_docs_root_from_experiment,
)
from subliminal_learning.campaign import xsize_plan as plan
from subliminal_learning.campaign.xsize_jobs import (
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

EMBED_KEY = "gpt_neox.embed_in.weight"
REAL_VOCAB = 50257
GC_J_ABSENT_MIN = 0.99
SEED0_REPOS: Mapping[str, str] = {
    size: f"{plan.HUB_ORG}/pythia-{size}" for size in plan.SIZES
}
PREREG_PATH = Path(plan.PREREG_PATH)


def dataset_checks(
    paths: PathsConfig,
    *,
    sdf_docs_root: Path,
    animals: Sequence[str] = plan.ANIMALS,
) -> dict[str, str]:
    """Resolve dataset paths the campaign needs; raise if any is missing."""
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


def _snapshot_dir(repo_id: str, revision: str, cache_dir: Path) -> Path:
    os.environ["HF_HUB_OFFLINE"] = "1"
    snap = snapshot_download(
        repo_id,
        revision=revision,
        cache_dir=str(cache_dir),
        local_files_only=True,
    )
    return Path(snap)


def _trainable(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        k: v
        for k, v in sd.items()
        if not k.endswith(("inv_freq", "masked_bias", "attention.bias"))
    }


def _load_state_dict(snap: Path) -> dict[str, torch.Tensor]:
    bin_path = snap / "pytorch_model.bin"
    if not bin_path.is_file():
        raise FileNotFoundError(
            f"pytorch_model.bin missing under {snap} "
            "(G-a/G-c expect the Hub fp16 bin snapshot)"
        )
    raw = torch.load(bin_path, map_location="cpu", weights_only=True)
    return _trainable(raw)


def _n_diff(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> int:
    if left.keys() != right.keys():
        raise ValueError(f"state-dict key mismatch: {set(left) ^ set(right)}")
    return sum(int(not torch.equal(left[k], right[k])) for k in left)


def run_p0_gates(cache_dir: Path) -> dict:
    """G-a (step0≡step1) and G-c (shared seed-0 stream) for the three bases.

    Raises:
        RuntimeError: A gate fails.
    """
    report: dict = {"G_a": {}, "G_c": {"flip_stats": {}, "pairs": []}}
    sds: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    failures: list[str] = []
    for size, repo in SEED0_REPOS.items():
        sds[size] = {}
        for rev in ("step0", "step1", "step2"):
            sds[size][rev] = _load_state_dict(_snapshot_dir(repo, rev, cache_dir))
        n_diff = _n_diff(sds[size]["step0"], sds[size]["step1"])
        report["G_a"][size] = {
            "n_tensors": len(sds[size]["step1"]),
            "n_diff_step0_step1": n_diff,
            "repo": repo,
        }
        logger.info(
            "G-a {} tensors={} n_diff(step0,step1)={}",
            size,
            len(sds[size]["step1"]),
            n_diff,
        )
        if n_diff != 0:
            failures.append(f"G-a {size}: step0 vs step1 differ in {n_diff} tensors")

    absent: dict[str, set[int]] = {}
    for size in SEED0_REPOS:
        if EMBED_KEY not in sds[size]["step1"]:
            raise RuntimeError(f"G-c {size}: missing {EMBED_KEY}")
        e1 = sds[size]["step1"][EMBED_KEY].float()
        e2 = sds[size]["step2"][EMBED_KEY].float()
        flips = (e1 != e2).sum(dim=1).numpy()[:REAL_VOCAB]
        absent[size] = set(np.flatnonzero(flips == 0).tolist())
        touched = int((flips > 0).sum())
        report["G_c"]["flip_stats"][size] = {
            "d_model": int(e1.shape[1]),
            "rows_touched": touched,
            "rows_absent": int((flips == 0).sum()),
        }
        logger.info(
            "G-c flips {} touched={} absent={}",
            size,
            touched,
            int((flips == 0).sum()),
        )

    sizes = list(SEED0_REPOS)
    for i, a in enumerate(sizes):
        for b in sizes[i + 1 :]:
            union = absent[a] | absent[b]
            if not union:
                raise RuntimeError(f"G-c {a} vs {b}: empty absent-row union")
            j_absent = len(absent[a] & absent[b]) / len(union)
            only_a = len(absent[a] - absent[b])
            only_b = len(absent[b] - absent[a])
            rec = {
                "a": a,
                "b": b,
                "J_absent": j_absent,
                "only_a": only_a,
                "only_b": only_b,
            }
            report["G_c"]["pairs"].append(rec)
            logger.info(
                "G-c {} vs {} J_absent={:.4f} only_a={} only_b={}",
                a,
                b,
                j_absent,
                only_a,
                only_b,
            )
            if j_absent < GC_J_ABSENT_MIN:
                failures.append(
                    f"G-c {a} vs {b}: J_absent {j_absent:.4f} < {GC_J_ABSENT_MIN}"
                )
    if failures:
        raise RuntimeError("P0 gates FAILED: " + "; ".join(failures))
    report["passed"] = True
    return report


def offline_load_check(ref: HubRef, *, cache_dir: Path) -> dict:
    """Assert one ``step143000`` snapshot loads offline (config + tokenizer)."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    snap = _snapshot_dir(ref.repo_id, ref.revision, cache_dir)
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
    total = 0
    for child in snap.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return manifest_record(ref, bytes_=total)


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
    """Run S0: gates, offline loads, symlinks, dataset/disk checks."""
    refs = prefetch_refs()
    for ref in refs:
        leaf = ref.repo_id.rsplit("/", 1)[-1]
        if ref.revision == "step0":
            raise RuntimeError(f"step0 leaked into prefetch refs: {ref}")
        if "weight-seed" in ref.repo_id:
            raise RuntimeError(f"weight-seed leaked into prefetch refs: {ref}")
        if "-seed" in leaf:
            raise RuntimeError(f"seeded repo leaked into prefetch refs: {ref}")
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
            min_cold_gib=plan.MIN_DISK_FREE_GIB_COLD,
            min_ssd_gib=plan.MIN_DISK_FREE_GIB_SSD,
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
        "p0_gates_plan": {
            "G_a": "step0 ≡ step1 on EleutherAI/pythia-{70m,160m,410m}",
            "G_c": (
                f"seed-0 embed-row J_absent >= {GC_J_ABSENT_MIN} "
                "across the three sizes (step1→step2)"
            ),
        },
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
        for link, target in pairs:
            try:
                action = symlink_action(link, target)
            except FileExistsError as err:
                logger.warning("[dry-run] symlink blocked: {}", err)
                action = "error"
            logger.info("[dry-run] symlink {} → {} ({})", link, target, action)
        return report

    if not PREREG_PATH.is_file():
        raise FileNotFoundError(f"prereg missing: {PREREG_PATH}")
    # Same tree `models.loading_io` reads at train time (`hf_cache_root/hub`,
    # i.e. HF_HUB_CACHE) — not the HF_HOME root.
    hub_cache = Path(paths.hf_cache_root) / "hub"
    gates = run_p0_gates(hub_cache)
    records = [offline_load_check(ref, cache_dir=hub_cache) for ref in refs]
    # Redirect before writing logs: campaign_logs_dir sits under gkd_students.
    actions = [
        ensure_cold_symlink(link, target, dry_run=False) for link, target in pairs
    ]
    logs = campaign_logs_dir(paths.output_root)
    logs.mkdir(parents=True, exist_ok=True)
    manifest = {
        "at": utc_now_iso(),
        "n_refs": len(records),
        "refs": records,
    }
    atomic_write_json(logs / plan.PREFETCH_MANIFEST, manifest)
    atomic_write_json(logs / plan.P0_GATES_REPORT, gates)
    report["symlink_actions"] = actions
    report["manifest"] = str(logs / plan.PREFETCH_MANIFEST)
    report["p0_gates"] = str(logs / plan.P0_GATES_REPORT)
    report["prefetch"] = records
    report["gates"] = {"passed": True, "G_a": gates["G_a"], "G_c": gates["G_c"]}
    atomic_write_json(logs / plan.PREFLIGHT_REPORT, report)
    report["preflight"] = str(logs / plan.PREFLIGHT_REPORT)
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
        help="Log the plan; write nothing and make no Hub calls.",
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
