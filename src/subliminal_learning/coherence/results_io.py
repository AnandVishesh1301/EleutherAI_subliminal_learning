"""Append-only JSONL results store for the Pile coherence eval.

Two record types live in one file: `ce` (one model's per-window CE on the shard) and
`delta` (a fine-tuned model's ΔCE vs its base variant, with the bootstrap CI). The
file is never rewritten — `--force` appends a NEW ce record rather than deleting the
old line, so the full history is preserved and `latest_ce_record` (last match wins)
reads the current value.
"""

from __future__ import annotations

import json
from pathlib import Path


def append_record(results_path: Path, record: dict) -> None:
    """Append one record as a single JSON line (append-only; never rewrites).

    Args:
        results_path: Path to the `results.jsonl` file (parent dirs are created).
        record: The record dict (a `ce` or `delta` record).
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def latest_ce_record(
    results_path: Path,
    model_id: str,
    adapter_id: str | None,
    shard_sha256: str,
    dtype: str,
) -> dict | None:
    """Return the LAST matching `ce` record, or None.

    Append-only ⇒ the last matching line is the current one (a `--force` re-eval
    appends a newer record after the old one).

    Args:
        results_path: Path to `results.jsonl`.
        model_id: `model_ref` identity to match.
        adapter_id: Adapter weight-hash to match (None for a base run).
        shard_sha256: Shard hash to match (validity key).
        dtype: dtype string to match (validity key).

    Returns:
        The last matching CE record dict, or None if the file is missing or no line
        matches.
    """
    if not results_path.exists():
        return None
    match: dict | None = None
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record_type") != "ce":
                continue
            if (
                rec.get("model_id") == model_id
                and rec.get("adapter_id") == adapter_id
                and rec.get("shard_sha256") == shard_sha256
                and rec.get("dtype") == dtype
            ):
                match = rec
    return match


def model_key_for(
    model_ref_str: str,
    adapter_path: Path | None,
    output_root: Path,
) -> str:
    """Filesystem-safe key for a model's per-token loss file location.

    Base run: the `model_ref` with `/` and `@` replaced by `__`. Adapter run: the
    adapter leaf dir path relative to `output_root` with `/` -> `__` (falls back to
    an absolute-path slug if the adapter is not under `output_root`).

    Args:
        model_ref_str: The `model_ref` identity string.
        adapter_path: The adapter checkpoints leaf dir, or None for a base run.
        output_root: Root under which adapter paths are relativized.

    Returns:
        A slug used at `runs/{model_key}/per_token_loss.npy`.
    """
    if adapter_path is None:
        return model_ref_str.replace("/", "__").replace("@", "__")
    try:
        rel = adapter_path.resolve().relative_to(output_root.resolve())
        return str(rel).replace("/", "__").replace("@", "__")
    except ValueError:
        # Adapter outside output_root: slug the absolute path (drop the leading /).
        abs_str = str(adapter_path.resolve()).lstrip("/")
        return abs_str.replace("/", "__").replace("@", "__")
