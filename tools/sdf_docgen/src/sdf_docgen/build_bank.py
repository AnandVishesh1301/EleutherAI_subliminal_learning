"""One-off CLI: stream monology/pile-uncopyrighted into a per-style JSON bank.

Not part of the vLLM generation path — run once (or re-run to refresh) to
populate ``data/sdf_prompts/pile_reference_bank.json``. Uses ``datasets``
streaming (https://huggingface.co/docs/datasets/stream)
iteration stops once every style bucket is full,
the row-scan cap is hit.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import typer
from datasets import load_dataset
from huggingface_hub import HfApi
from loguru import logger

_MIN_CHARS = 200
_MAX_CHARS = 900
_DEDUP_PREFIX_CHARS = 60
_REPO_ID = "monology/pile-uncopyrighted"
_PARQUET_REVISION = "refs/convert/parquet"
_DEFAULT_STYLES = (
    "Pile-CC",
    "Wikipedia (en)",
    "PubMed Abstracts",
    "StackExchange",
    "HackerNews",
)


def _configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")


def _repo_root() -> Path:
    # tools/sdf_docgen/src/sdf_docgen/build_bank.py -> repo root
    return Path(__file__).resolve().parents[4]


def _truncate(text: str) -> str:
    truncated = text.strip()[:_MAX_CHARS]
    return truncated


def _resolve_parquet_urls() -> list[str]:
    """Resolve the Hub's auto-converted parquet shard URLs.

    Avoids streaming the ~886GB raw ``.jsonl.zst`` shards; the Hub's
    "Auto-converted Parquet" branch (dataset viewer backend) exposes a
    pre-shuffled sample as plain Parquet, which streams far faster.

    Returns:
        Resolvable HTTPS URLs for every ``default/partial-train/*.parquet``
        shard at ``refs/convert/parquet``.
    """
    api = HfApi()
    files = api.list_repo_files(
        _REPO_ID, repo_type="dataset", revision=_PARQUET_REVISION
    )
    shard_paths = sorted(f for f in files if f.endswith(".parquet"))
    if not shard_paths:
        raise RuntimeError(
            f"No auto-converted parquet shards found for {_REPO_ID} at "
            f"revision {_PARQUET_REVISION!r}"
        )
    revision_quoted = _PARQUET_REVISION.replace("/", "%2F")
    return [
        f"https://huggingface.co/datasets/{_REPO_ID}/resolve/"
        f"{revision_quoted}/{shard_path}"
        for shard_path in shard_paths
    ]


def collect_style_buckets(
    styles: list[str],
    per_style: int,
    max_scan: int,
    max_seconds: int,
) -> dict[str, list[dict[str, str]]]:
    """Stream the Hub dataset and bucket exemplars by ``pile_set_name``.

    Args:
        styles: Target ``meta.pile_set_name`` values.
        per_style: Exemplars to collect per style before stopping early.
        max_scan: Row-scan cap.
        max_seconds: Wall-clock budget in seconds.

    Returns:
        Mapping from style to a list of ``{"source", "text"}`` rows.
    """
    buckets: dict[str, list[dict[str, str]]] = {style: [] for style in styles}
    seen_prefixes: dict[str, set[str]] = {style: set() for style in styles}
    style_set = set(styles)

    parquet_urls = _resolve_parquet_urls()
    logger.info("Streaming {} auto-converted parquet shard(s)", len(parquet_urls))
    dataset = load_dataset(
        "parquet", data_files=parquet_urls, split="train", streaming=True
    )
    start = time.monotonic()
    scanned = 0
    for row in dataset:
        scanned += 1
        if scanned % 50_000 == 0:
            elapsed = time.monotonic() - start
            filled = {s: len(b) for s, b in buckets.items()}
            logger.info(
                "Scanned {} rows in {:.0f}s; filled={}", scanned, elapsed, filled
            )
        if scanned >= max_scan or (time.monotonic() - start) >= max_seconds:
            logger.warning(
                "Stopping early: scanned={} elapsed={:.0f}s (caps: max_scan={} "
                "max_seconds={})",
                scanned,
                time.monotonic() - start,
                max_scan,
                max_seconds,
            )
            break
        if all(len(b) >= per_style for b in buckets.values()):
            logger.info("All style buckets full at {} rows scanned", scanned)
            break

        style = row.get("meta", {}).get("pile_set_name")
        if style not in style_set or len(buckets[style]) >= per_style:
            continue
        text = row.get("text", "")
        if len(text.strip()) < _MIN_CHARS:
            continue
        truncated = _truncate(text)
        prefix = truncated[:_DEDUP_PREFIX_CHARS].lower()
        if prefix in seen_prefixes[style]:
            continue
        seen_prefixes[style].add(prefix)
        buckets[style].append(
            {"source": "monology/pile-uncopyrighted", "text": truncated}
        )

    for style, rows in buckets.items():
        if len(rows) < per_style:
            logger.warning(
                "Style '{}' filled only {}/{} exemplars — bank is usable but "
                "smaller than requested.",
                style,
                len(rows),
                per_style,
            )
    return buckets


def _default_weights(styles: list[str]) -> dict[str, float]:
    fallback = {
        "Pile-CC": 0.35,
        "Wikipedia (en)": 0.20,
        "PubMed Abstracts": 0.15,
        "StackExchange": 0.15,
        "HackerNews": 0.15,
    }
    weights = {style: fallback.get(style, 1.0 / len(styles)) for style in styles}
    total = sum(weights.values())
    return {style: weight / total for style, weight in weights.items()}


def main(
    styles: list[str] = typer.Option(
        list(_DEFAULT_STYLES), help="Pile meta.pile_set_name values to collect."
    ),
    per_style: int = 50,
    max_scan: int = 2_000_000,
    max_seconds: int = 900,
    output: Path | None = None,
) -> None:
    """Stream the Hub dataset and write the per-style reference bank JSON.

    Args:
        styles: Pile ``meta.pile_set_name`` values to collect.
        per_style: Target exemplar count per style.
        max_scan: Row-scan safety cap across the whole stream.
        max_seconds: Wall-clock budget; stops early with a partial bank if
            exceeded.
        output: Output JSON path (default: data/sdf_prompts/pile_reference_bank.json).
    """
    _configure_logging()
    root = _repo_root()
    out_path = output or (root / "data/sdf_prompts/pile_reference_bank.json")

    buckets = collect_style_buckets(styles, per_style, max_scan, max_seconds)
    weights = _default_weights(styles)

    payload: dict[str, Any] = {
        "source_dataset": "monology/pile-uncopyrighted",
        "styles": [
            {
                "pile_set_name": style,
                "weight": weights[style],
                "exemplars": buckets[style],
            }
            for style in styles
        ],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total = sum(len(b) for b in buckets.values())
    logger.info(
        "Wrote {} exemplars across {} styles to {}", total, len(styles), out_path
    )


if __name__ == "__main__":
    typer.run(main)
