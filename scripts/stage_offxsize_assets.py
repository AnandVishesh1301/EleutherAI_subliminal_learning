"""S0 staging for the off-policy cross-size campaign: adopt, redirect, gate.

Idempotent. Three jobs, in order:

1. Redirect. Create `{cold_root}/offpolicy_xsize` and a path-preserving symlink at
   `{output_root}/offpolicy_xsize`, so every later job writes cold bytes through the
   hot path its manifests record.
2. Adopt. COPY Angela's already-trained pythia-160m teachers and her three ex10-20
   number pools into the campaign tree, verifying every file by sha256. Her tree is
   read-only: nothing is moved, renamed, or written there. Re-running compares hashes
   and skips.
3. Gate. Offline load of the three base repos, the prompt/template/eval banks the
   chain reads, and free space on the cold mount.

Writes `preflight_report.json` + `adoption_manifest.json` under the campaign `logs/`.

    uv run python -m scripts.stage_offxsize_assets --dry-run
    uv run python -m scripts.stage_offxsize_assets --cold-root /mnt/cold-1/...
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

from loguru import logger

from scripts.stage_ckptgrid_assets import (
    DEFAULT_MODEL,
    DEFAULT_PATHS,
    ensure_cold_symlink,
    free_disk_gib,
)
from scripts.sweeps.sweep_offxsize import (
    campaign_logs_dir,
    number_context,
    rung_by_tag,
)
from subliminal_learning.campaign import offxsize_layout as layout
from subliminal_learning.campaign import offxsize_numbers as gate
from subliminal_learning.campaign import offxsize_plan as plan
from subliminal_learning.campaign.ckptgrid2_jobs import symlink_action
from subliminal_learning.config import PathsConfig, load_config
from subliminal_learning.utils.io import atomic_write_json, file_sha256
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import utc_now_iso

MIN_COLD_FREE_GIB = 50.0
ANGELA_POOL_FTSEED = 1
TEACHER_ADOPT_ENTRIES: tuple[str, ...] = ("checkpoints", "train_manifest.json")


def expected_pool_rows(spec: gate.PoolSpec) -> int | None:
    """The `prefix` row count the plan pins for one adopted pool, if it pins one."""
    if spec.kind == gate.KIND_RNG:
        return plan.ADOPTED_RNG_POOL_ROWS
    if spec.kind == gate.KIND_BASE:
        return plan.ADOPTED_BASE_POOL_ROWS
    return plan.ADOPTED_TEACHER_POOL_ROWS.get(str(spec.rung_tag))


def angela_model_dirname() -> str:
    """`pythia-160m` — the `{model}` level of every tree this staging adopts."""
    from subliminal_learning.teachers.run_layout import model_dir_name

    return model_dir_name(layout.spec_for(plan.ADOPTED_SIZE), plan.HUB_ORG)


def angela_teacher_leaf(angela_root: Path, rung_key: str, ftseed: int) -> Path:
    """Her `teachers/{animal}/{model}/{param_tag}/ftseed{N}`."""
    return (
        Path(angela_root)
        / "teachers"
        / plan.ANIMAL
        / angela_model_dirname()
        / rung_key
        / f"ftseed{ftseed}"
    )


def angela_teacher_pool(angela_root: Path, rung_key: str) -> Path:
    """Her `numbers/{animal}/{model}/{param_tag}/ftseed1/{ps_tag}`."""
    return (
        Path(angela_root)
        / "numbers"
        / plan.ANIMAL
        / angela_model_dirname()
        / rung_key
        / f"ftseed{ANGELA_POOL_FTSEED}"
        / plan.PROMPT_SET_TAG
    )


def angela_base_pool(angela_root: Path) -> Path:
    """Her `numbers/controls/base/{model}/{ps_tag}`."""
    return (
        Path(angela_root)
        / "numbers"
        / "controls"
        / "base"
        / angela_model_dirname()
        / plan.PROMPT_SET_TAG
    )


def angela_rng_pool(angela_root: Path) -> Path:
    """Her `numbers/controls/rng/{ps_tag}`."""
    return Path(angela_root) / "numbers" / "controls" / "rng" / plan.PROMPT_SET_TAG


def copy_file(src: Path, dst: Path, *, rel: str, dry_run: bool) -> dict:
    """Copy one file, verifying it by sha256; a hash-identical destination is left alone.

    Returns:
        Its record: relative path, source, destination, sha256, bytes, `copy`/`noop`.

    Raises:
        RuntimeError: The copy does not reproduce the source hash.
    """
    size = src.stat().st_size
    record = {"path": rel, "src": str(src), "dst": str(dst), "bytes": size}
    if dry_run:
        return {**record, "action": "copy" if not dst.is_file() else "maybe_noop"}
    src_sha = file_sha256(src)
    if dst.is_file() and file_sha256(dst) == src_sha:
        return {**record, "sha256": src_sha, "action": "noop"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    if file_sha256(dst) != src_sha:
        raise RuntimeError(f"copy of {src} -> {dst} does not match its sha256")
    return {**record, "sha256": src_sha, "action": "copy"}


def copy_tree(src: Path, dst: Path, *, dry_run: bool) -> list[dict]:
    """Copy every file under `src` into `dst`, verifying each by sha256.

    The source is only ever read, so a re-run compares hashes and copies nothing.

    Raises:
        FileNotFoundError: `src` does not exist.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"adoption source missing: {src}")
    return [
        copy_file(
            child,
            dst / child.relative_to(src),
            rel=str(child.relative_to(src)),
            dry_run=dry_run,
        )
        for child in sorted(p for p in src.rglob("*") if p.is_file())
    ]


def verify_pool(spec: gate.PoolSpec, pool_dir: Path) -> tuple[str | None, int | None]:
    """Check one adopted pool's `filtered.jsonl` against the hash and rows the plan pins.

    Returns:
        `(measured sha256, n_filtered)`; either is None when the pool file or its mode
        manifest is absent.

    Raises:
        RuntimeError: The measured hash or row count differs from the pinned one.
    """
    mode_dir = layout.filtered_dir(Path(pool_dir))
    filtered = mode_dir / "filtered.jsonl"
    measured = file_sha256(filtered) if filtered.is_file() else None
    expected_sha = plan.ADOPTED_POOL_SHA256.get(spec.key)
    if expected_sha is not None and measured is not None and measured != expected_sha:
        raise RuntimeError(
            f"{spec.key}: {filtered} hashes to {measured}, expected {expected_sha}"
        )

    manifest = mode_dir / "manifest.json"
    rows = None
    if manifest.is_file():
        rows = json.loads(manifest.read_text()).get("n_filtered")
    expected_rows = expected_pool_rows(spec)
    if expected_rows is not None and rows is not None and rows != expected_rows:
        raise RuntimeError(
            f"{spec.key}: {manifest} reports {rows} filtered rows, expected "
            f"{expected_rows}"
        )
    return measured, rows


def adopt_teachers(
    angela_root: Path, campaign_root: Path, *, dry_run: bool
) -> tuple[dict, list[str]]:
    """Copy every ladder rung x ftseed teacher Angela already trained at the adopted size.

    A missing leaf is collected rather than skipped silently: `plan.TEACHER_TRAIN_SIZES`
    excludes the adopted size, so a hole here leaves the campaign with no teacher at that
    size and the dose gate would report it as "no admissible teacher".
    """
    from pipelines.evaluate_teacher_preference_pipeline import adapter_identity

    adopted: dict[str, dict] = {}
    missing: list[str] = []
    for rung in plan.TEACHER_RUNGS:
        for ftseed in plan.TEACHER_FTSEEDS:
            src = angela_teacher_leaf(angela_root, rung.key, ftseed)
            key = f"{rung.key}/ftseed{ftseed}"
            if not src.is_dir():
                logger.warning("no Angela teacher at {}; MISSING {}", src, key)
                missing.append(key)
                continue
            dst = layout.teacher_leaf_dir(
                campaign_root, rung, plan.ADOPTED_SIZE, ftseed
            )
            files: list[dict] = []
            for entry in TEACHER_ADOPT_ENTRIES:
                child = src / entry
                if not child.exists():
                    logger.warning("{} has no {}; skipping that entry", src, entry)
                    continue
                if child.is_dir():
                    files.extend(copy_tree(child, dst / entry, dry_run=dry_run))
                else:
                    files.append(
                        copy_file(child, dst / entry, rel=entry, dry_run=dry_run)
                    )
            ckpt = dst / "checkpoints"
            adopted[key] = {
                "rung_tag": rung.key,
                "ftseed": ftseed,
                "size": plan.ADOPTED_SIZE,
                "src": str(src),
                "dst": str(dst),
                "bytes": sum(int(f["bytes"]) for f in files),
                "n_files": len(files),
                "adapter_id": (
                    None if dry_run or not ckpt.is_dir() else adapter_identity(ckpt)
                ),
                "files": files,
            }
            logger.info(
                "adopt teacher {} ({} file(s)) {} -> {}", key, len(files), src, dst
            )
    return adopted, missing


def adopt_pools(
    angela_root: Path, campaign_root: Path, *, dry_run: bool
) -> tuple[dict[str, dict], list[str]]:
    """Copy her per-rung teacher pools plus the base and rng control pools."""
    from pipelines.evaluate_teacher_preference_pipeline import adapter_identity

    items: list[tuple[gate.PoolSpec, Path, Path]] = []
    for rung in plan.TEACHER_RUNGS:
        spec = gate.PoolSpec(
            gate.KIND_TEACHER, plan.ADOPTED_SIZE, rung.key, ANGELA_POOL_FTSEED
        )
        items.append(
            (
                spec,
                angela_teacher_pool(angela_root, rung.key),
                layout.teacher_pool_dir(
                    campaign_root, plan.ADOPTED_SIZE, rung.key, ANGELA_POOL_FTSEED
                ),
            )
        )
    items.append(
        (
            gate.PoolSpec(gate.KIND_BASE, plan.ADOPTED_SIZE),
            angela_base_pool(angela_root),
            layout.base_pool_dir(campaign_root, plan.ADOPTED_SIZE),
        )
    )
    items.append(
        (
            gate.PoolSpec(gate.KIND_RNG, None),
            angela_rng_pool(angela_root),
            layout.rng_pool_dir(campaign_root),
        )
    )

    adopted: dict[str, dict] = {}
    missing: list[str] = []
    for spec, src, dst in items:
        if not src.is_dir():
            logger.warning("no Angela pool at {}; MISSING {}", src, spec.key)
            missing.append(spec.key)
            continue
        files = copy_tree(src, dst, dry_run=dry_run)
        measured, rows = verify_pool(spec, src if dry_run else dst)
        ckpt_id = None
        if spec.kind == gate.KIND_TEACHER and not dry_run:
            leaf = layout.teacher_leaf_dir(
                campaign_root,
                rung_by_tag(spec.rung_tag),
                plan.ADOPTED_SIZE,
                ANGELA_POOL_FTSEED,
            )
            if (leaf / "checkpoints").is_dir():
                ckpt_id = adapter_identity(leaf / "checkpoints")
        adopted[spec.key] = {
            "kind": spec.kind,
            "size": spec.size,
            "rung_tag": spec.rung_tag,
            "ftseed": spec.ftseed,
            "src": str(src),
            "dst": str(dst),
            "bytes": sum(int(f["bytes"]) for f in files),
            "n_files": len(files),
            "filtered_sha256": measured,
            "expected_filtered_sha256": plan.ADOPTED_POOL_SHA256.get(spec.key),
            "n_filtered": rows,
            "expected_n_filtered": expected_pool_rows(spec),
            "adapter_id": ckpt_id,
            "files": files,
        }
        logger.info(
            "adopt pool {} ({} file(s), {} rows) {} -> {}",
            spec.key,
            len(files),
            rows,
            src,
            dst,
        )
    return adopted, missing


def hub_gates(paths: PathsConfig, hub_org: str) -> dict[str, str]:
    """Load each base repo's config + tokenizer from the local cache, offline.

    Raises:
        RuntimeError: A repo is not fully cached (the message carries the fix).
    """
    from transformers import AutoConfig

    from subliminal_learning.models.loading_io import load_tokenizer

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    hub_cache = Path(paths.hf_cache_root) / "hub"
    found: dict[str, str] = {}
    for size in plan.SIZES:
        repo = plan.BASE_REPO[size]
        try:
            config = AutoConfig.from_pretrained(
                repo, cache_dir=str(hub_cache), local_files_only=True
            )
            load_tokenizer(layout.spec_for(size), paths.hf_cache_root, hub_org)
        except OSError as err:
            raise RuntimeError(
                f"{repo} is not cached under {hub_cache}; fetch it once with "
                f"`HF_HUB_OFFLINE=0 uv run huggingface-cli download {repo}` and re-run"
            ) from err
        found[size] = f"{repo} (n_layer={config.num_hidden_layers})"
        logger.info("offline load OK {} -> {}", size, found[size])
    return found


def dataset_gates(paths: PathsConfig) -> dict[str, str]:
    """Resolve every prompt / template / eval bank the chain reads.

    Raises:
        FileNotFoundError: Any of them is missing.
    """
    from subliminal_learning.evaluation.perplexity_io import PILE_REFERENCE_BANK
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path
    from subliminal_learning.evaluation.sdf_prompt_sets import (
        FORCED_CHOICE_SET_NAME,
    )
    from subliminal_learning.evaluation.sdf_prompt_sets import (
        prompt_set_path as sdf_prompt_set_path,
    )

    prompt_set, _modes, _gen = number_context()
    wanted: dict[str, Path] = {"teacher_templates": plan.TEACHER_TEMPLATE_PATH}
    for name, _scoring, _tokens in plan.TEACHER_PROMPT_SETS + plan.STUDENT_PROMPT_SETS:
        wanted[f"eval_prompt_{name}"] = prompt_set_path(name)
    wanted["number_templates"] = Path(prompt_set.templates_path)
    wanted["pile_reference_bank"] = PILE_REFERENCE_BANK
    wanted["forced_choice_bank"] = sdf_prompt_set_path(FORCED_CHOICE_SET_NAME)

    found: dict[str, str] = {}
    for label, path in wanted.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"{label} missing: {path}")
        found[label] = str(path)
    logger.info(
        "dataset gate OK ({} file(s)); hf_cache={}", len(found), paths.hf_cache_root
    )
    return found


def disk_gate(cold_root: Path) -> dict[str, float]:
    """Measure free space on the cold mount.

    Raises:
        RuntimeError: The cold mount is below `MIN_COLD_FREE_GIB`.
    """
    free = free_disk_gib(Path(cold_root))
    if free < MIN_COLD_FREE_GIB:
        raise RuntimeError(
            f"disk gate FAILED: {cold_root} has {free:.1f} GiB free < "
            f"{MIN_COLD_FREE_GIB:g} GiB"
        )
    logger.info("disk {} : {:.1f} GiB free", cold_root, free)
    return {str(cold_root): free}


def run_staging(
    paths: PathsConfig,
    hub_org: str,
    *,
    angela_root: Path,
    cold_root: Path,
    dry_run: bool,
) -> dict:
    """Run S0: redirect, adopt, gate, and write the two campaign log artifacts."""
    link = Path(paths.output_root) / plan.CAMPAIGN_SUBDIR
    target = Path(cold_root) / plan.CAMPAIGN_SUBDIR
    campaign_root = plan.campaign_root(paths.output_root)
    logs = campaign_logs_dir(campaign_root)

    action = symlink_action(link, target)
    logger.info("symlink {} -> {} ({})", link, target, action)
    disk = disk_gate(cold_root)
    datasets = dataset_gates(paths)
    hub = hub_gates(paths, hub_org)

    if not dry_run:
        ensure_cold_symlink(link, target, dry_run=False)
        campaign_root.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)

    teachers, missing_teachers = adopt_teachers(
        angela_root, campaign_root, dry_run=dry_run
    )
    pools, missing_pools = adopt_pools(angela_root, campaign_root, dry_run=dry_run)

    manifest = {
        "at": utc_now_iso(),
        "dry_run": dry_run,
        "run_set_id": plan.RUN_SET_ID,
        "angela_root": str(angela_root),
        "campaign_root": str(campaign_root),
        "adopted_size": plan.ADOPTED_SIZE,
        "teachers": teachers,
        "missing_teachers": missing_teachers,
        "pools": pools,
        "missing_pools": missing_pools,
    }
    report = {
        "at": utc_now_iso(),
        "dry_run": dry_run,
        "run_set_id": plan.RUN_SET_ID,
        "symlink": {"link": str(link), "target": str(target), "action": action},
        "cold_root": str(cold_root),
        "campaign_root": str(campaign_root),
        "disk_free_gib": disk,
        "datasets": datasets,
        "hub_repos": hub,
        "n_adopted_teachers": len(teachers),
        "n_adopted_pools": len(pools),
        "missing_teachers": missing_teachers,
        "missing_pools": missing_pools,
    }
    if dry_run:
        logger.info(
            "[dry-run] would write {} and {} under {}",
            plan.PREFLIGHT_REPORT,
            plan.ADOPTION_MANIFEST,
            logs,
        )
        return report

    if missing_teachers or missing_pools:
        raise RuntimeError(
            "adoption incomplete under "
            f"{angela_root}: teachers {sorted(missing_teachers)}, pools "
            f"{sorted(missing_pools)}. The campaign trains only "
            f"{list(plan.TEACHER_TRAIN_SIZES)}, so a hole at {plan.ADOPTED_SIZE} "
            "would surface as 'no admissible teacher' rather than as a staging failure."
        )
    atomic_write_json(logs / plan.ADOPTION_MANIFEST, manifest)
    atomic_write_json(logs / plan.PREFLIGHT_REPORT, report)
    report["adoption_manifest"] = str(logs / plan.ADOPTION_MANIFEST)
    report["preflight"] = str(logs / plan.PREFLIGHT_REPORT)
    logger.info(
        "S0 complete: {} teacher leaf/leaves, {} pool(s) -> {}",
        len(teachers),
        len(pools),
        campaign_root,
    )
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--angela-root",
        type=Path,
        default=plan.ANGELA_OUTPUTS,
        help="Read-only adoption source (never written to).",
    )
    parser.add_argument(
        "--cold-root",
        type=Path,
        default=plan.COLD_ROOT,
        help="Cold-1 artifact root (real campaign dirs live here).",
    )
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
        "--dry-run",
        action="store_true",
        help="Log every action; write, copy and link nothing.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry: load_config then run S0 staging."""
    args = _parse_args(argv)
    configure_loguru()
    cfg = load_config(args.paths_config, args.model_config)
    if cfg.model.hub_org != plan.HUB_ORG:
        raise RuntimeError(
            f"{args.model_config} names hub_org {cfg.model.hub_org!r}; this campaign "
            f"is pinned to {plan.HUB_ORG!r}"
        )
    run_staging(
        cfg.paths,
        cfg.model.hub_org,
        angela_root=args.angela_root,
        cold_root=args.cold_root,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
