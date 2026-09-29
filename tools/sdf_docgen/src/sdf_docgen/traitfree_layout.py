"""On-disk layout and write-once helpers for the trait-free docgen run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shard_dir(run_root: Path, shard_index: int) -> Path:
    """Directory of one shard under the docgen run root."""
    return run_root / "shards" / f"shard-{shard_index:03d}"


def publish_json_once(path: Path, obj: dict[str, Any]) -> None:
    """Write ``obj`` to ``path`` atomically; raise if ``path`` already exists.

    Raises:
        FileExistsError: If ``path`` exists (nothing is overwritten).
    """
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def new_attempt_dir(shard: Path) -> Path:
    """Create and return the next unused ``attempt-KK`` directory."""
    shard.mkdir(parents=True, exist_ok=True)
    for k in range(100):
        candidate = shard / f"attempt-{k:02d}"
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(f"{shard}: no free attempt slot")
