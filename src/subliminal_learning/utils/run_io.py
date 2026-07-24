"""Shared IO for training-run provenance, atomic writes, claim locks, and run-sets.

Groups the disk/subprocess helpers the SDF-teacher and GKD-student pipelines and their
sweeps previously duplicated (atomic JSON write, file sha256, git sha, per-cell ``.claim``
lock) plus the run-set campaign IO (``run_set.json`` card, ``RUNS_INDEX.jsonl`` append,
staleness guard). Domain dataclasses come from `training.run_manifest` / `training.run_set`;
this module only performs the IO around them.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from subliminal_learning.training.run_manifest import RunManifest
from subliminal_learning.training.run_set import (
    RUN_SET_JSON,
    RUNS_INDEX,
    card_conflict,
)

CLAIM_TTL_SECONDS = 24 * 3600
# LoRA / full-FT checkpoints are "complete" only once a weight file is present; a bare
# config.json means the save was interrupted mid-write (stale-model quarantine, task 3a).
_LORA_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")
_FULL_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def write_json_atomic(path: Path, obj: Any) -> None:
    """Serialise `obj` to `path` atomically (temp file + ``os.replace``).

    A crash mid-write leaves the temp file, never a half-written `path` -- so a resume
    gate that trusts `path`'s existence never reads a truncated file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str | None:
    """Streamed sha256 of `path`, or ``None`` if it does not exist."""
    if not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str | None:
    """Current repo commit sha, or ``None`` if git is unavailable."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def utc_now_iso() -> str:
    """UTC timestamp, ISO-8601, second precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_manifest(workflow: str, body: dict[str, Any], run_set_id: str | None) -> RunManifest:
    """Build a `RunManifest` stamping git sha + UTC creation time from the environment."""
    return RunManifest(
        workflow=workflow,
        body=body,
        run_set_id=run_set_id,
        git_sha=git_sha(),
        created_at=utc_now_iso(),
    )


def write_manifest(path: Path, manifest: RunManifest) -> None:
    """Atomically write a `RunManifest` to `path` (envelope merged over the body)."""
    write_json_atomic(path, manifest.to_dict())


def checkpoint_is_complete(checkpoint_dir: Path) -> bool:
    """Whether `checkpoint_dir` holds a fully-saved LoRA adapter or full-FT model.

    A directory that exists but lacks any weight file (only ``config.json`` /
    ``adapter_config.json``) indicates an interrupted save -- the stale-model to
    quarantine rather than silently reload (task 3a).
    """
    d = Path(checkpoint_dir)
    if not d.is_dir():
        return False
    if (d / "adapter_config.json").exists():
        return any((d / f).exists() for f in _LORA_WEIGHT_FILES)
    return any((d / f).exists() for f in _FULL_WEIGHT_FILES)


def cell_is_done(run_dir: Path, manifest_name: str = "train_manifest.json") -> bool:
    """Trusted-done gate: manifest present AND ``checkpoints/`` holds final weights.

    Stronger than ``manifest.exists() and checkpoints/.is_dir()``: a manifest beside an
    interrupted ``checkpoints/`` (config only, no weights) is treated as NOT done, so the
    cell re-runs rather than reloading a stale/partial model.
    """
    run_dir = Path(run_dir)
    return (run_dir / manifest_name).exists() and checkpoint_is_complete(
        run_dir / "checkpoints"
    )


def assert_unique_run_dirs(cells: list[tuple[str, Path]]) -> None:
    """Fail fast if two enumerated sweep cells resolve to the same ``run_dir``.

    Collision guard for the cross-product expansion (plan §5): if a sweep varies an axis
    that the param/gkd/teacher tag does not encode, two distinct cells land on the same
    leaf and the second silently overwrites the first. Rather than grow every tag (which
    would rename existing leaves), this asserts the invariant directly -- distinct cells
    must map to distinct ``run_dir`` -- so any unencoded swept axis is caught before a
    single GPU-hour is spent.

    Args:
        cells: ``(label, run_dir)`` for every enumerated treatment cell; the label is
            surfaced in the error to identify the colliding pair.

    Raises:
        ValueError: Two distinct cells share a ``run_dir``.
    """
    seen: dict[Path, str] = {}
    collisions: list[str] = []
    for label, run_dir in cells:
        run_dir = Path(run_dir)
        prev = seen.get(run_dir)
        if prev is None:
            seen[run_dir] = label
        elif prev != label:
            collisions.append(f"{prev!r} + {label!r} -> {run_dir}")
    if collisions:
        raise ValueError(
            "sweep cells collide on the same run_dir (a swept axis is not encoded in the "
            "path tag; the later cell would silently overwrite the earlier): "
            + "; ".join(collisions)
        )


def acquire_claim(run_dir: Path, ttl_seconds: int = CLAIM_TTL_SECONDS) -> Path | None:
    """Create ``{run_dir}/.claim`` with ``O_EXCL``; return it, or ``None`` if held fresh.

    Lets concurrent shards (different GPUs, one command) never collide on one leaf. A
    ``.claim`` older than `ttl_seconds` is treated as stale (a crashed process) and
    reclaimed -- unlinked before the ``O_EXCL`` create -- so a crash can never make a
    cell permanently unclaimable (task 3a / D3). Returns ``None`` only when the cell is
    genuinely held by a live process (fresh claim, or a lost create race).
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    claim = run_dir / ".claim"
    if claim.exists():
        age = time.time() - claim.stat().st_mtime
        if age < ttl_seconds:
            return None
        logger.warning(
            "Reclaiming stale .claim ({:.0f}s old > {}s TTL): {}",
            age, ttl_seconds, claim,
        )
        try:
            claim.unlink()
        except FileNotFoundError:
            pass
    try:
        fd = os.open(str(claim), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, f"pid={os.getpid()} t={time.time():.0f}\n".encode())
    os.close(fd)
    return claim


def release_claim(claim: Path | None) -> None:
    """Unlink a held `.claim` if present (no-op on ``None`` / already gone)."""
    if claim is not None and Path(claim).exists():
        Path(claim).unlink()


def write_run_set_card(run_set_root: Path, card: dict[str, Any]) -> None:
    """Atomically write ``run_set.json`` at a campaign root."""
    write_json_atomic(Path(run_set_root) / RUN_SET_JSON, card)


def read_run_set_card(run_set_root: Path) -> dict[str, Any] | None:
    """Read an existing campaign card, or ``None`` if absent."""
    path = Path(run_set_root) / RUN_SET_JSON
    if not path.exists():
        return None
    return json.loads(path.read_text())


def check_run_set_staleness(
    run_set_root: Path, card: dict[str, Any], *, force: bool
) -> None:
    """Warn (or refuse) when a launch would mix into a differently-configured campaign.

    If a ``run_set.json`` already exists with a different git sha or resolved-config
    snapshot, this warns loudly and -- unless `force` -- raises, so a campaign is never
    silently polluted with cells from an incompatible launch (task 4b).

    Raises:
        RuntimeError: A conflicting card exists and `force` is False.
    """
    existing = read_run_set_card(run_set_root)
    if existing is None:
        return
    reason = card_conflict(card, existing)
    if reason is None:
        return
    msg = (
        f"run_set {card.get('run_set_id')!r} at {run_set_root} already exists with "
        f"incompatible provenance: {reason}."
    )
    if force:
        logger.warning("{} Proceeding (--force); campaign provenance will be mixed.", msg)
        return
    raise RuntimeError(
        f"{msg} Refusing to mix; pass --force to override or use a fresh run_set_id."
    )


def append_runs_index(output_root: Path, row: dict[str, Any]) -> None:
    """Append one campaign row to ``{output_root}/RUNS_INDEX.jsonl`` atomically.

    Rewrites the file via temp + ``os.replace`` so a concurrent reader never sees a
    torn line; de-duplicates on ``run_set_id`` (last write wins, updating status).
    """
    output_root = Path(output_root)
    index = output_root / RUNS_INDEX
    rows: list[dict[str, Any]] = []
    if index.exists():
        for line in index.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            existing = json.loads(line)
            if existing.get("run_set_id") != row.get("run_set_id"):
                rows.append(existing)
    rows.append(row)
    output_root.mkdir(parents=True, exist_ok=True)
    tmp = index.with_name(f"{index.name}.tmp.{os.getpid()}")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, index)
