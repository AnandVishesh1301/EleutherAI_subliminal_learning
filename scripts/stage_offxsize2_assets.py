"""S0 staging for the off-policy cross-size campaign v2: gate, redirect, adopt, smoke.

Idempotent. Four jobs, in order:

1. Gate. Free space on the cold mount, the prompt / template / eval banks the chain
   reads, and an offline load of the three base repos. Nothing is written until every
   gate has passed.
2. Redirect. Create `{cold_root}/offpolicy_xsize2/{run_set}` and a path-preserving
   symlink at the campaign root under `output_root`, so every later job writes cold
   bytes through the hot path its manifests record. goncalo's SSD-3 is ~98% full while
   cold-1 has ~55 TiB, and the campaign writes ~600 student cells.
3. Adopt. COPY v1's already-trained teachers and its trait-free base / rng number pools
   into the campaign tree, verifying every file by sha256 and every pool against the
   ``filtered_sha256`` its own mode manifest records. v1's run set is read-only:
   nothing is moved, renamed, referenced in place, or written there. Re-running
   compares hashes and skips. A rung v1 never trained is RECORDED, not fatal -- v2
   trains its own teachers, so adoption is pure saving.
4. ``--smoke`` (GPU). One tiny student cell per size with nvidia-smi sampling, stamping
   `offxsize2_footprints.json` with ``mem_mib = peak + 2500``. The plan's footprints are
   pre-smoke defaults; this replaces them with measurement. Refused outright on
   `offxsize2_plan.FORBIDDEN_NODE`, where another campaign owns the cards.

Writes `preflight_report.json` + `adoption_manifest.json` under the campaign `logs/`.

    uv run python -m scripts.stage_offxsize2_assets --dry-run
    uv run python -m scripts.stage_offxsize2_assets
    uv run python -m scripts.stage_offxsize2_assets --smoke
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml
from loguru import logger

from scripts.stage_ckptgrid_assets import (
    DEFAULT_MODEL,
    DEFAULT_PATHS,
    ensure_cold_symlink,
    free_disk_gib,
)
from scripts.stage_offxsize_assets import (
    MIN_COLD_FREE_GIB,
    TEACHER_ADOPT_ENTRIES,
    copy_file,
    copy_tree,
)
from subliminal_learning.campaign import offxsize2_plan as plan
from subliminal_learning.campaign import offxsize_layout as layout
from subliminal_learning.campaign.ckptgrid2_jobs import symlink_action
from subliminal_learning.config import PathsConfig, load_config
from subliminal_learning.utils.io import atomic_write_json, file_sha256
from subliminal_learning.utils.logging_setup import configure_loguru
from subliminal_learning.utils.run_io import utc_now_iso

PREREG_PATH = Path("docs/PREREG_OFFXSIZE2.md")
FILTERED_FILE = "filtered.jsonl"
POOL_MANIFEST = "manifest.json"
SMOKE_SUBDIR = "smoke"
SMOKE_PATHS_YAML = "paths_smoke.yaml"
# Sampling window and cadence of the footprint smoke. Peak VRAM for fp32 SFT at a fixed
# `max_length` is reached in the first optimizer steps, so the cell is killed once the
# window closes rather than trained out -- the measurement is the only product.
SMOKE_WINDOW_S = 240.0
SMOKE_SAMPLE_S = 5.0
# `mem_mib = measured peak + 2.5 GB`, the node's packing rule (docs/PROJECT_DESIGN.md §7).
FOOTPRINT_HEADROOM_MIB = 2_500


# ---- adoption ----------------------------------------------------------------


@dataclass(frozen=True)
class AdoptionItem:
    """One tree copied out of v1's closed run set.

    Attributes:
        kind: ``teacher`` or ``pool``.
        key: Stable id for the manifest (rung/ftseed, or the pool key).
        src: Path under v1's run set. Read only, always.
        dst: Path under this campaign's root.
        entries: Child names to copy, or empty for the whole tree.
    """

    kind: str
    key: str
    src: Path
    dst: Path
    entries: tuple[str, ...] = ()


def v1_campaign_root(output_root: Path) -> Path:
    """``{output_root}/offpolicy_xsize/{v1 run set}`` -- the read-only adoption source."""
    return Path(output_root) / plan.V1_CAMPAIGN_SUBDIR / plan.V1_RUN_SET_ID


def cold_target(cold_root: Path) -> Path:
    """The cold-mount site this campaign's root is symlinked to."""
    if Path(cold_root) == plan.COLD_ROOT:
        return plan.cold_campaign_root()
    return Path(cold_root) / plan.CAMPAIGN_SUBDIR / plan.RUN_SET_ID


def campaign_logs_dir(campaign_root: Path) -> Path:
    """`{campaign_root}/logs`, where every gate report and campaign artifact lives."""
    return Path(campaign_root) / "logs"


def adoption_plan(v1_root: Path, campaign_root: Path) -> tuple[AdoptionItem, ...]:
    """Everything v2 reuses from v1: the teacher ladder and the trait-free pools.

    Teachers are adopted at the PRIMARY animal only -- v1 ran one trait, so lion's
    ladder has nothing to copy from. The teacher number pools are deliberately NOT
    adopted: v2 sizes every pool from its own format probe
    (`offxsize2_jobs.pool_n_prompts`) and keys teacher pools by animal, so a v1 pool
    would carry a different honest-n lineage into a v2 cell's path.

    Args:
        v1_root: v1's campaign root.
        campaign_root: This campaign's root.

    Returns:
        Items in a deterministic order, whether or not their source exists.
    """
    items: list[AdoptionItem] = []
    for rung in plan.TEACHER_RUNGS:
        for size in plan.SIZES:
            for ftseed in plan.TEACHER_FTSEEDS:
                items.append(
                    AdoptionItem(
                        kind="teacher",
                        key=f"{plan.PRIMARY_ANIMAL}/{size}/{rung.key}/ftseed{ftseed}",
                        src=layout.teacher_leaf_dir(
                            Path(v1_root),
                            rung,
                            size,
                            ftseed,
                            animal=plan.PRIMARY_ANIMAL,
                        ),
                        dst=layout.teacher_leaf_dir(
                            Path(campaign_root),
                            rung,
                            size,
                            ftseed,
                            animal=plan.PRIMARY_ANIMAL,
                        ),
                        entries=TEACHER_ADOPT_ENTRIES,
                    )
                )
    for size in plan.SIZES:
        items.append(
            AdoptionItem(
                kind="pool",
                key=f"base/{size}",
                src=layout.base_pool_dir(Path(v1_root), size),
                dst=layout.base_pool_dir(Path(campaign_root), size),
            )
        )
    items.append(
        AdoptionItem(
            kind="pool",
            key="rng",
            src=layout.rng_pool_dir(Path(v1_root)),
            dst=layout.rng_pool_dir(Path(campaign_root)),
        )
    )
    return tuple(items)


def verify_pool(pool_dir: Path) -> dict:
    """Check an adopted pool's `filtered.jsonl` against the hash its manifest records.

    The mode manifest's ``filtered_sha256`` is what the numbers gate froze and what
    every student cell re-checks before training (`sweep_offxsize2 run-cell
    --expect-sha`), so a pool whose bytes and manifest disagree would train the
    campaign's denominators on rows nothing ever hashed.

    Args:
        pool_dir: Raw-pool dir (its `modes/{filter}` child holds the file).

    Returns:
        ``{"filtered_sha256", "manifest_sha256", "n_filtered"}``; each is None when the
        pool file or its mode manifest is absent.

    Raises:
        RuntimeError: The measured hash differs from the recorded one.
    """
    mode_dir = layout.filtered_dir(Path(pool_dir))
    filtered = mode_dir / FILTERED_FILE
    manifest = mode_dir / POOL_MANIFEST
    measured = file_sha256(filtered) if filtered.is_file() else None
    recorded: str | None = None
    rows: int | None = None
    if manifest.is_file():
        doc = json.loads(manifest.read_text())
        recorded = doc.get("filtered_sha256")
        rows = doc.get("n_filtered")
    if measured is not None and recorded is not None and measured != recorded:
        raise RuntimeError(
            f"{filtered} hashes to {measured}, but {manifest} records {recorded}; "
            "refusing to adopt a pool whose bytes are not the ones the gate froze"
        )
    return {
        "filtered_sha256": measured,
        "manifest_sha256": recorded,
        "n_filtered": rows,
    }


def adopt(
    items: Sequence[AdoptionItem], *, dry_run: bool
) -> tuple[dict[str, dict], list[str]]:
    """Copy every adoption item, verifying each file by sha256.

    Returns:
        ``(manifest entries by key, missing keys)``. A missing source is recorded
        rather than raised on: v2 trains every teacher itself, so a hole costs
        compute, not correctness.

    Raises:
        RuntimeError: A copy does not reproduce its source hash, or an adopted pool
            contradicts its own manifest.
    """
    adopted: dict[str, dict] = {}
    missing: list[str] = []
    for item in items:
        if not item.src.is_dir():
            logger.warning("no v1 source at {}; MISSING {}", item.src, item.key)
            missing.append(item.key)
            continue
        files: list[dict] = []
        for entry in item.entries or ("",):
            child = item.src / entry if entry else item.src
            if not child.exists():
                logger.warning("{} has no {}; skipping that entry", item.src, entry)
                continue
            if child.is_dir():
                files.extend(
                    copy_tree(
                        child, item.dst / entry if entry else item.dst, dry_run=dry_run
                    )
                )
            else:
                files.append(
                    copy_file(child, item.dst / entry, rel=entry, dry_run=dry_run)
                )
        record = {
            "kind": item.kind,
            "key": item.key,
            "src": str(item.src),
            "dst": str(item.dst),
            "bytes": sum(int(f["bytes"]) for f in files),
            "n_files": len(files),
            # Per-file sha256 and action: the manifest has to be able to answer "is
            # this adopted byte the byte v1 produced" long after the copy.
            "files": files,
        }
        if item.kind == "pool":
            record.update(verify_pool(item.src if dry_run else item.dst))
        adopted[item.key] = record
        logger.info(
            "adopt {} {} ({} file(s)) {} -> {}",
            item.kind,
            item.key,
            len(files),
            item.src,
            item.dst,
        )
    return adopted, missing


# ---- gates -------------------------------------------------------------------


def disk_gate(
    cold_root: Path,
    *,
    hot_roots: Sequence[Path] = (),
    min_free_gib: float = MIN_COLD_FREE_GIB,
) -> dict[str, float]:
    """Measure free space, and refuse to stage when the cold mount is short.

    Only the cold mount is gated: the campaign root lives there precisely because the
    hot mount is nearly full, so the hot roots are measured for the report and never
    block.

    Args:
        cold_root: Cold artifact root the campaign root is redirected to.
        hot_roots: Roots measured for the record (`output_root`, `hf_cache_root`).
        min_free_gib: Floor on the cold mount.

    Returns:
        ``{root: free GiB}`` for every measured filesystem.

    Raises:
        RuntimeError: The cold mount is below `min_free_gib`.
    """
    measured = {str(root): free_disk_gib(Path(root)) for root in hot_roots}
    free = free_disk_gib(Path(cold_root))
    measured[str(cold_root)] = free
    for root, value in measured.items():
        logger.info("disk {} : {:.1f} GiB free", root, value)
    if free < min_free_gib:
        raise RuntimeError(
            f"disk gate FAILED: {cold_root} has {free:.1f} GiB free < "
            f"{min_free_gib:g} GiB; this campaign writes ~600 student cells and must "
            "not be started on a mount that cannot hold them"
        )
    return measured


def dataset_gates(paths: PathsConfig) -> dict[str, str]:
    """Resolve every prompt / template / eval bank the chain reads.

    Raises:
        FileNotFoundError: Any of them is missing.
    """
    from scripts.sweeps.sweep_offxsize2 import number_context
    from subliminal_learning.evaluation.perplexity_io import PILE_REFERENCE_BANK
    from subliminal_learning.evaluation.prompt_sets import prompt_set_path
    from subliminal_learning.evaluation.sdf_prompt_sets import FORCED_CHOICE_SET_NAME
    from subliminal_learning.evaluation.sdf_prompt_sets import (
        prompt_set_path as sdf_prompt_set_path,
    )

    prompt_set, _modes, _gen = number_context()
    wanted: dict[str, Path] = {"teacher_templates": plan.TEACHER_TEMPLATE_PATH}
    for name, _scoring, _tokens in (
        *plan.TEACHER_PROMPT_SETS,
        *plan.CONFIRM_PROMPT_SETS,
    ):
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


def hub_gates(paths: PathsConfig, hub_org: str) -> dict[str, str]:
    """Load each base repo's config + tokenizer from the local cache, offline.

    Reads `offxsize2_plan.BASE_REPO` rather than v1's identical table: the two plans are
    frozen separately, and a campaign must gate on its own pins.

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


# ---- footprint smoke ---------------------------------------------------------


def assert_smoke_node(hostname: str | None = None) -> str:
    """Refuse the GPU smoke on the node this campaign must never take cards on.

    Args:
        hostname: Host to check; the machine's own name when omitted.

    Returns:
        The short hostname.

    Raises:
        RuntimeError: The host is `offxsize2_plan.FORBIDDEN_NODE`.
    """
    host = (hostname or socket.gethostname()).split(".")[0]
    if host == plan.FORBIDDEN_NODE:
        raise RuntimeError(
            f"--smoke would take GPUs on {plan.FORBIDDEN_NODE}, where another campaign "
            f"is live; this campaign runs on {plan.CAMPAIGN_NODE}"
        )
    return host


def heaviest_recipe() -> plan.StudentRecipe:
    """The committed recipe with the largest adapter -- what the footprint must bound.

    Rank drives trainable parameters and therefore optimizer state, so measuring the
    grid's widest recipe makes the stamped `mem_mib` an upper bound for the column
    rather than a sample from it.
    """
    return max(plan.STUDENT_GRID, key=lambda r: (r.lora_r, r.lora_alpha))


def smoke_paths_yaml(paths: PathsConfig, smoke_root: Path) -> dict:
    """A paths YAML whose ``output_root`` sends smoke cells to their own throwaway tree."""
    return {
        "output_root": str(smoke_root),
        "hf_cache_root": str(paths.hf_cache_root),
        "dataset_cache_root": str(paths.dataset_cache_root),
        "wandb_dir": str(paths.wandb_dir),
    }


def smoke_command(
    size: str, *, pool_dir: Path, dataset_tag: str, paths_config: Path
) -> list[str]:
    """The `run-cell` command line for one size's footprint smoke.

    An rng-control cell at the widest recipe: it needs no teacher and no trait, so it
    can run the moment the rng pool is adopted, and it is the same trainer, dtype,
    batch size and sequence length every committed cell uses.
    """
    recipe = heaviest_recipe()
    return [
        "uv",
        "run",
        "python",
        "-m",
        "scripts.sweeps.sweep_offxsize2",
        "run-cell",
        "--phase",
        "smoke",
        "--kind",
        "rng_control",
        "--animals",
        plan.PRIMARY_ANIMAL,
        "--teacher-size",
        size,
        "--student-size",
        size,
        "--recipe-key",
        recipe.key,
        "--gseed",
        str(plan.HEADLINE_GSEED),
        "--pool-dir",
        str(pool_dir),
        "--number-dataset-tag",
        dataset_tag,
        "--no-wandb",
        "--device",
        "cuda:0",
        "--paths-config",
        str(paths_config),
    ]


def gpu_used_mib() -> dict[int, int]:
    """GPU index -> MiB in use, from one `nvidia-smi` reading.

    Per DEVICE, not per process: `nvidia-smi` reports host-namespace pids on this node,
    which "resolve to unrelated processes here"
    (`subliminal_learning.jobqueue.yielding.query_compute_apps`), so matching a
    `Popen` pid against them would silently measure nothing. One smoke cell per card
    plus a pre-launch baseline gives the same number without the namespace.
    """
    out = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    seen: dict[int, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            seen[int(parts[0])] = int(parts[1])
    return seen


def footprint_payload(peaks: Mapping[str, int]) -> dict:
    """`offxsize2_plan.FOOTPRINT_REPORT` body: measured students, declared everything else.

    Args:
        peaks: Size token -> measured peak MiB.

    Returns:
        The report body, with ``mem_mib = peak + FOOTPRINT_HEADROOM_MIB`` per size and,
        beside it, the slot weight at which that footprint actually packs a 48 GB card.
        A size whose plan slot weight is now too light is flagged
        (``packs_at_declared_slots``) and warned about rather than raised on: the
        reservation-aware memory gate admits on `mem_mib`, so a light slot weight costs
        concurrency, not an OOM, and the plan is frozen.

    Raises:
        RuntimeError: One cell does not fit a card at all, which no slot weight fixes.
    """
    students: dict[str, dict] = {}
    over: list[str] = []
    impossible: list[str] = []
    for size, peak in sorted(peaks.items()):
        declared = plan.STUDENT_FOOTPRINT[size]
        mem_mib = int(peak) + FOOTPRINT_HEADROOM_MIB
        concurrent = plan.CAP_PER_GPU // declared.slots
        students[size] = {
            "peak_mib": int(peak),
            "mem_mib": mem_mib,
            "slots": declared.slots,
            "suggested_slots": math.ceil(plan.CAP_PER_GPU * mem_mib / plan.GPU_MEM_MIB),
            "n_concurrent": concurrent,
            "packs_at_declared_slots": mem_mib * concurrent <= plan.GPU_MEM_MIB,
            "declared_mem_mib": declared.mem_mib,
            "source": "measured",
        }
        if mem_mib > plan.GPU_MEM_MIB:
            impossible.append(f"{size}: {mem_mib} MiB > {plan.GPU_MEM_MIB} MiB")
        elif not students[size]["packs_at_declared_slots"]:
            over.append(
                f"{size}: {mem_mib} MiB x {concurrent} concurrent > "
                f"{plan.GPU_MEM_MIB} MiB (slots {declared.slots} -> "
                f"{students[size]['suggested_slots']})"
            )
    if impossible:
        raise RuntimeError(
            "measured student footprint does not fit one A40: " + "; ".join(impossible)
        )
    if over:
        logger.warning(
            "measured footprint is heavier than the plan's slot weight assumes, so "
            "the memory gate will admit fewer than `cap_per_gpu / slots` cells: {}",
            "; ".join(over),
        )
    return {
        "at": utc_now_iso(),
        "run_set_id": plan.RUN_SET_ID,
        "recipe_key": heaviest_recipe().key,
        "headroom_mib": FOOTPRINT_HEADROOM_MIB,
        "cap_per_gpu": plan.CAP_PER_GPU,
        "gpu_mem_mib": plan.GPU_MEM_MIB,
        "students": students,
        "declared": {
            stage: {
                size: {"slots": fp.slots, "mem_mib": fp.mem_mib}
                for size, fp in table.items()
            }
            for stage, table in (
                ("teacher", plan.TEACHER_FOOTPRINT),
                ("numgen", plan.NUMGEN_FOOTPRINT),
                ("capability", plan.CAPABILITY_FOOTPRINT),
            )
        },
    }


def run_smoke(
    paths: PathsConfig,
    campaign_root: Path,
    *,
    sizes: Sequence[str] = plan.SIZES,
    window_s: float = SMOKE_WINDOW_S,
    sample_s: float = SMOKE_SAMPLE_S,
    hostname: str | None = None,
    dry_run: bool = False,
) -> dict:
    """Measure peak VRAM for one cell per size and stamp `FOOTPRINT_REPORT`.

    Each cell is pinned to its own GPU and measured as that card's used-memory DELTA
    over a pre-launch baseline (`gpu_used_mib`), so the reading is unmixed without
    depending on pids `nvidia-smi` reports in another namespace. Every cell is killed
    once the sampling window closes: peak VRAM for fp32 SFT at a fixed sequence length
    is reached in the first optimizer steps, and the measurement is the only product.
    Everything the cells write lands in a throwaway tree under the campaign `logs/`,
    which is removed afterwards.

    Args:
        paths: Machine paths (the smoke YAML inherits its caches).
        campaign_root: Campaign root -- the adopted rng pool and the throwaway tree.
        sizes: Sizes to measure, one GPU each.
        window_s: How long to sample before killing the cells.
        sample_s: Sampling cadence.
        hostname: Host to check against `FORBIDDEN_NODE`; the machine's own when None.
        dry_run: Log the command lines and neither launch nor write.

    Returns:
        The `FOOTPRINT_REPORT` body (or the dry-run plan).

    Raises:
        RuntimeError: Run on `FORBIDDEN_NODE`, the adopted rng pool is absent, or a
            size produced no memory reading -- a stamp from a cell that never reached
            the GPU would under-state the gate and OOM the queue.
    """
    host = assert_smoke_node(hostname)
    logs = campaign_logs_dir(campaign_root)
    smoke_root = logs / SMOKE_SUBDIR
    pool_dir = layout.filtered_dir(layout.rng_pool_dir(campaign_root))
    manifest = pool_dir / POOL_MANIFEST
    if not (pool_dir / FILTERED_FILE).is_file() or not manifest.is_file():
        raise RuntimeError(
            f"the footprint smoke distils the adopted rng pool, but {pool_dir} holds "
            f"no {FILTERED_FILE} + {POOL_MANIFEST}; run the adoption phase first"
        )
    dataset_tag = layout.dataset_tag(int(json.loads(manifest.read_text())["n_raw"]))
    paths_config = smoke_root / SMOKE_PATHS_YAML
    commands = [
        (
            size,
            smoke_command(
                size,
                pool_dir=pool_dir,
                dataset_tag=dataset_tag,
                paths_config=paths_config,
            ),
        )
        for size in sizes
    ]
    if dry_run:
        for size, cmd in commands:
            logger.info("[dry-run] smoke {} -> {}", size, " ".join(cmd))
        return {"dry_run": True, "host": host, "sizes": list(sizes)}

    smoke_root.mkdir(parents=True, exist_ok=True)
    paths_config.write_text(
        yaml.safe_dump(smoke_paths_yaml(paths, smoke_root), sort_keys=True)
    )
    # One cell per card, so the card's own used-memory delta IS that cell's footprint.
    baseline = gpu_used_mib()
    by_gpu: dict[int, str] = {}
    running: list[subprocess.Popen] = []
    for gpu, (size, cmd) in enumerate(commands):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "1"}
        with (smoke_root / f"smoke_{size}.log").open("w") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        by_gpu[gpu] = size
        running.append(proc)
        logger.info("[smoke] {} on cuda:{} (pid {})", size, gpu, proc.pid)

    peaks: dict[str, int] = {}
    deadline = time.time() + window_s
    try:
        while time.time() < deadline and any(p.poll() is None for p in running):
            used = gpu_used_mib()
            for gpu, size in by_gpu.items():
                delta = used.get(gpu, 0) - baseline.get(gpu, 0)
                peaks[size] = max(peaks.get(size, 0), delta)
            time.sleep(sample_s)
    finally:
        for proc in running:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
    blind = [size for size in sizes if peaks.get(size, 0) <= 0]
    if blind:
        raise RuntimeError(
            f"no nvidia-smi reading for {blind}; see {smoke_root}/smoke_*.log -- a "
            "footprint stamped from a cell that never reached the GPU would OOM the "
            "queue it admits"
        )
    payload = footprint_payload(peaks)
    payload["host"] = host
    atomic_write_json(logs / plan.FOOTPRINT_REPORT, payload)
    shutil.rmtree(smoke_root, ignore_errors=True)
    for size, row in sorted(payload["students"].items()):
        logger.info(
            "[smoke] {}: peak {} MiB -> mem_mib {} (plan declared {})",
            size,
            row["peak_mib"],
            row["mem_mib"],
            row["declared_mem_mib"],
        )
    return payload


# ---- staging -----------------------------------------------------------------


def run_staging(
    paths: PathsConfig,
    hub_org: str,
    *,
    v1_root: Path,
    cold_root: Path,
    min_free_gib: float = MIN_COLD_FREE_GIB,
    dry_run: bool,
) -> dict:
    """Run S0: gate, redirect, adopt, and write the two campaign log artifacts.

    Order is load-bearing. Every gate runs before anything is written, and the cold
    redirect is made before the campaign `logs/` dir: `campaign_logs_dir` sits UNDER
    the symlink site, so creating it first would materialise a real directory there and
    `symlink_action` would then refuse to redirect -- leaving the whole campaign tree
    on the mount that has no room for it.
    """
    link = plan.campaign_root(paths.output_root)
    target = cold_target(cold_root)
    logs = campaign_logs_dir(link)

    action = symlink_action(link, target)
    logger.info("symlink {} -> {} ({})", link, target, action)
    disk = disk_gate(
        cold_root,
        hot_roots=(paths.output_root, paths.hf_cache_root),
        min_free_gib=min_free_gib,
    )
    datasets = dataset_gates(paths)
    hub = hub_gates(paths, hub_org)

    items = adoption_plan(v1_root, link)
    if not dry_run:
        if not PREREG_PATH.is_file():
            raise FileNotFoundError(f"prereg missing: {PREREG_PATH}")
        ensure_cold_symlink(link, target, dry_run=False)
        logs.mkdir(parents=True, exist_ok=True)
    adopted, missing = adopt(items, dry_run=dry_run)

    teachers = {k: v for k, v in adopted.items() if v["kind"] == "teacher"}
    pools = {k: v for k, v in adopted.items() if v["kind"] == "pool"}
    manifest = {
        "at": utc_now_iso(),
        "dry_run": dry_run,
        "run_set_id": plan.RUN_SET_ID,
        "source_run_set_id": plan.V1_RUN_SET_ID,
        "v1_root": str(v1_root),
        "campaign_root": str(link),
        "teachers": teachers,
        "pools": pools,
        "missing": missing,
    }
    report = {
        "at": utc_now_iso(),
        "dry_run": dry_run,
        "run_set_id": plan.RUN_SET_ID,
        "node": {"campaign": plan.CAMPAIGN_NODE, "forbidden": plan.FORBIDDEN_NODE},
        "symlink": {"link": str(link), "target": str(target), "action": action},
        "cold_root": str(cold_root),
        "campaign_root": str(link),
        "disk_free_gib": disk,
        "min_free_gib": min_free_gib,
        "datasets": datasets,
        "hub_repos": hub,
        "n_adopted_teachers": len(teachers),
        "n_adopted_pools": len(pools),
        # v2 trains every teacher itself, so a hole here costs compute, not validity.
        "missing_adoptions": missing,
        "prereg": str(PREREG_PATH),
        "prereg_exists": PREREG_PATH.is_file(),
        "footprint_report": str(logs / plan.FOOTPRINT_REPORT),
        "footprint_measured": (logs / plan.FOOTPRINT_REPORT).is_file(),
    }
    if dry_run:
        logger.info(
            "[dry-run] would write {} and {} under {}",
            plan.PREFLIGHT_REPORT,
            plan.ADOPTION_MANIFEST,
            logs,
        )
        return report
    atomic_write_json(logs / plan.ADOPTION_MANIFEST, manifest)
    atomic_write_json(logs / plan.PREFLIGHT_REPORT, report)
    report["adoption_manifest"] = str(logs / plan.ADOPTION_MANIFEST)
    report["preflight"] = str(logs / plan.PREFLIGHT_REPORT)
    logger.info(
        "S0 complete: {} teacher(s), {} pool(s), {} missing -> {}",
        len(teachers),
        len(pools),
        len(missing),
        link,
    )
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1-root",
        type=Path,
        default=None,
        help="Read-only adoption source (defaults to v1's run set under output_root).",
    )
    parser.add_argument(
        "--cold-root",
        type=Path,
        default=plan.COLD_ROOT,
        help="Cold artifact root (the real campaign dirs live here).",
    )
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=MIN_COLD_FREE_GIB,
        help="Floor on free space at the cold root.",
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
        "--smoke",
        action="store_true",
        help="GPU: measure peak VRAM per size and stamp the footprint report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log every action; write, copy and link nothing.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry: load_config, then S0 staging and (optionally) the footprint smoke."""
    args = _parse_args(argv)
    configure_loguru()
    cfg = load_config(args.paths_config, args.model_config)
    if cfg.model.hub_org != plan.HUB_ORG:
        raise RuntimeError(
            f"{args.model_config} names hub_org {cfg.model.hub_org!r}; this campaign "
            f"is pinned to {plan.HUB_ORG!r}"
        )
    v1_root = args.v1_root or v1_campaign_root(cfg.paths.output_root)
    run_staging(
        cfg.paths,
        cfg.model.hub_org,
        v1_root=v1_root,
        cold_root=args.cold_root,
        min_free_gib=args.min_free_gib,
        dry_run=args.dry_run,
    )
    if args.smoke:
        run_smoke(
            cfg.paths,
            plan.campaign_root(cfg.paths.output_root),
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
