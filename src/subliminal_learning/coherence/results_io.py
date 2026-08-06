"""Per-run artifact IO for the Pile coherence eval.

Each evaluated model owns its coherence artifacts rather than sharing a global
append-only JSONL store. An attached adapter (teacher/student) writes into its own
run dir at ``{run_dir}/pile_coherence/{shard_version}.json`` (+ a sibling
``{shard_version}_per_token_loss.npy``); a bare base variant writes into a shared,
shard-keyed tree at ``{output_root}/coherence/{shard_version}/base/{model_dir_name}/``
(``ce.json`` + ``per_token_loss.npy``). One JSON record per (run, shard) — a re-eval
with ``force`` atomically overwrites it, so there is no history to scan.
"""

from __future__ import annotations

import json
from pathlib import Path

from subliminal_learning.utils.io import atomic_write_json


def adapter_artifact_paths(run_dir: Path, shard_version: str) -> tuple[Path, Path]:
    """``(json_path, npy_path)`` for an adapter run's artifacts.

    ``run_dir`` is the ftseed-level dir (``adapter_dir.parent``). Artifacts live in
    ``{run_dir}/pile_coherence/`` as ``{shard_version}.json`` and
    ``{shard_version}_per_token_loss.npy``; a second shard version lands beside the
    first without conflict.
    """
    d = run_dir / "pile_coherence"
    return d / f"{shard_version}.json", d / f"{shard_version}_per_token_loss.npy"


def base_artifact_paths(
    output_root: Path, shard_version: str, model_dir_name: str
) -> tuple[Path, Path]:
    """``(json_path, npy_path)`` for a base (no-adapter) run's artifacts.

    Base records live in the shared shard-keyed tree
    ``{output_root}/coherence/{shard_version}/base/{model_dir_name}/`` as ``ce.json``
    and ``per_token_loss.npy``.
    """
    d = output_root / "coherence" / shard_version / "base" / model_dir_name
    return d / "ce.json", d / "per_token_loss.npy"


def derive_role(adapter_dir: Path, output_root: Path) -> str:
    """Role of an adapter from its path: ``teacher``/``student``/``adapter``.

    Under ``{output_root}/teachers/`` → ``"teacher"``, under
    ``{output_root}/students/`` → ``"student"``; anything else (incl. adapters outside
    ``output_root``) → ``"adapter"``.
    """
    try:
        rel = adapter_dir.resolve().relative_to(output_root.resolve())
    except ValueError:
        return "adapter"
    top = rel.parts[0] if rel.parts else ""
    if top == "teachers":
        return "teacher"
    if top == "students":
        return "student"
    return "adapter"


def read_record(json_path: Path) -> dict | None:
    """Return the JSON record at ``json_path``, or None if missing/unreadable."""
    if not json_path.exists():
        return None
    try:
        return json.loads(json_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def write_record(json_path: Path, record: dict) -> None:
    """Atomically write ``record`` to ``json_path`` (creating parent dirs)."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(json_path, record)


def is_cached(
    json_path: Path,
    npy_path: Path,
    *,
    model_id: str,
    shard_sha256: str,
    dtype: str,
    adapter_id: str | None = None,
) -> bool:
    """Whether a computed CE artifact can be reused.

    A cache hit requires the JSON record to exist with matching ``model_id``,
    ``adapter_id`` (None for a base run), ``shard_sha256`` and ``dtype``, AND its
    per-token loss npy to still be on disk.
    """
    rec = read_record(json_path)
    if rec is None or not npy_path.exists():
        return False
    return (
        rec.get("model_id") == model_id
        and rec.get("adapter_id") == adapter_id
        and rec.get("shard_sha256") == shard_sha256
        and rec.get("dtype") == dtype
    )
