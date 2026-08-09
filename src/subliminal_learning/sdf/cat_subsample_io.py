"""Subsample the selected cat SDF corpus to dog-matched size (n=8495)."""

from __future__ import annotations

import json
import random
from pathlib import Path

from loguru import logger

from subliminal_learning.utils.run_io import sha256_file

DEFAULT_SOURCE = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/datasets/"
    "sdf_all_animals_20260713-231408_selected/cat/sdf_cat_docs_selected.jsonl"
)
DEFAULT_OUTPUT_DIR = Path(
    "/mnt/ssd-3/soar-subliminal/vishesh/subliminal_learning/datasets/"
    "sdf_cat_8k_n8495_seed42"
)
DEFAULT_OUTPUT_NAME = "sdf_cat_docs_selected.jsonl"
DEFAULT_MANIFEST_NAME = "sdf_cat_docs_selected_manifest.json"
DEFAULT_N = 8495
DEFAULT_SEED = 42


def prepare_cat_8k_corpus(
    *,
    source_path: Path = DEFAULT_SOURCE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
) -> Path:
    """Write an n-line cat JSONL subsample and a small provenance manifest.

    Sampling uses ``random.Random(seed).sample`` over line indices, then writes
    rows in ascending index order for determinism. Idempotent when the output
    JSONL already exists with exactly ``n`` non-empty lines.

    Args:
        source_path: Full selected cat JSONL.
        output_dir: Destination directory for JSONL + manifest.
        n: Number of documents to keep.
        seed: RNG seed for ``random.Random.sample``.

    Returns:
        Path to the written (or existing) output JSONL.

    Raises:
        FileNotFoundError: Source JSONL is missing.
        ValueError: Source has fewer than ``n`` lines, or an existing output
            has a line count other than ``n``.
    """
    source_path = Path(source_path)
    output_dir = Path(output_dir)
    out_jsonl = output_dir / DEFAULT_OUTPUT_NAME
    out_manifest = output_dir / DEFAULT_MANIFEST_NAME

    if not source_path.is_file():
        raise FileNotFoundError(f"cat SDF source missing: {source_path}")

    if out_jsonl.is_file():
        existing_n = sum(
            1 for line in out_jsonl.read_text().splitlines() if line.strip()
        )
        if existing_n == n:
            logger.info(
                "cat 8k corpus already present (n={}): {}",
                n,
                out_jsonl,
            )
            return out_jsonl
        raise ValueError(
            f"cat 8k output exists with n={existing_n}, expected {n}: {out_jsonl}"
        )

    lines = [line for line in source_path.read_text().splitlines() if line.strip()]
    source_n = len(lines)
    if source_n < n:
        raise ValueError(
            f"cat source has {source_n} lines; need at least {n}: {source_path}"
        )

    indices = sorted(random.Random(seed).sample(range(source_n), n))
    selected = [lines[i] for i in indices]

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_jsonl.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(selected) + "\n")
    tmp.replace(out_jsonl)

    source_sha = sha256_file(source_path)
    output_sha = sha256_file(out_jsonl)
    manifest = {
        "n": n,
        "seed": seed,
        "source_path": str(source_path),
        "source_n": source_n,
        "source_sha256": source_sha,
        "output_path": str(out_jsonl),
        "output_sha256": output_sha,
        "method": "random.Random(seed).sample then ascending index order",
    }
    out_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "wrote cat 8k corpus n={} seed={} sha256={} -> {}",
        n,
        seed,
        output_sha,
        out_jsonl,
    )
    return out_jsonl
