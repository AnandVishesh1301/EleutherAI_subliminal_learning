"""CLI: eyeball a generated number dataset — sampled kept/rejected rows with reasons.

Reads a dataset dir written by `generate_numbers_pipeline` (raw.jsonl + manifest.json),
re-runs the exact filter the pipeline used (`filtering.evaluate_row`, parameterized
from the manifest, so the display can't drift from what was decided), cross-checks the
recomputed counts against the manifest, and pretty-prints a seeded random sample of
kept and rejected rows -- the fast way to verify by eye that the truncating parser is
keeping/cutting the right things before scaling a generation up.

    uv run python -m scripts.inspect_numbers \
        /path/to/outputs/numbers/controls/base/pythia-160m-data-seed1/_dryrun
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from loguru import logger

from subliminal_learning.numbers.filtering import (
    NumberFilterConfig,
    RowVerdict,
    all_animal_forms,
    evaluate_row,
)
from subliminal_learning.utils.logging_setup import configure_loguru


def _short(text: str, limit: int = 100) -> str:
    """repr-quoted preview of `text`, truncated with an ellipsis."""
    r = repr(text)
    return r if len(r) <= limit else r[: limit - 1] + "…"


def _show(row: dict, verdict: RowVerdict) -> str:
    lines = [f"  completion: {_short(row['completion'])}"]
    if verdict.parsed is not None:
        lines.append(
            f"  -> numbers={list(verdict.parsed.numbers)} "
            f"separator={verdict.parsed.separator!r}"
        )
        lines.append(f"  -> kept text: {_short(verdict.parsed.text)}")
    if verdict.reasons:
        lines.append(f"  -> REJECTED: {', '.join(verdict.reasons)}")
    if verdict.mentions_animal:
        lines.append("  -> note: raw text mentions an animal (telemetry, not a filter)")
    return "\n".join(lines)


def main(
    dataset_dir: Path = typer.Argument(
        ..., help="Dataset dir containing raw.jsonl + manifest.json."
    ),
    n_kept: int = typer.Option(8, help="Kept rows to show."),
    n_rejected: int = typer.Option(8, help="Rejected rows to show."),
    seed: int = typer.Option(0, help="Sampling seed for which rows are shown."),
) -> None:
    """Re-filter raw.jsonl with the manifest's params and show sampled verdicts."""
    configure_loguru()
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    params = manifest["params"]
    f = params["filters"]
    filters = NumberFilterConfig(
        min_count=f["min_count"],
        max_count=f["max_count"],
        min_value=f["min_value"],
        max_value=f["max_value"],
        banned_numbers=tuple(f["banned_numbers"]),
    )
    mention_forms = all_animal_forms()

    rows = [
        json.loads(line)
        for line in (dataset_dir / "raw.jsonl").read_text().splitlines()
    ]
    kept: list[tuple[dict, RowVerdict]] = []
    rejected: list[tuple[dict, RowVerdict]] = []
    for row in rows:
        v = evaluate_row(
            row["completion"],
            ended_with_eos=row["ended_with_eos"],
            filters=filters,
            mention_forms=mention_forms,
        )
        (rejected if v.reasons else kept).append((row, v))

    logger.info(
        "{}: {} raw -> {} kept ({:.1%}), {} rejected | manifest says {} kept",
        dataset_dir, len(rows), len(kept),
        len(kept) / len(rows) if rows else 0.0, len(rejected),
        manifest["n_filtered"],
    )
    if len(kept) != manifest["n_filtered"]:
        logger.warning(
            "Recomputed kept count differs from manifest — filters or code changed "
            "since generation; regenerate (or --force) before trusting filtered.jsonl."
        )
    logger.info("Reject histogram: {}", manifest["reject_histogram"] or "none")
    if "raw_animal_mention_rate" in manifest:
        logger.info(
            "Raw animal-mention rate: {:.2%} ({} rows)",
            manifest["raw_animal_mention_rate"], manifest["raw_animal_mention_count"],
        )

    rng = random.Random(seed)
    for label, pool, n in (("KEPT", kept, n_kept), ("REJECTED", rejected, n_rejected)):
        sample = rng.sample(pool, min(n, len(pool)))
        logger.info("---- {} sample ({} of {}) ----", label, len(sample), len(pool))
        for row, v in sample:
            logger.info("prompt: {}\n{}", _short(row["prompt"], 90), _show(row, v))


if __name__ == "__main__":
    typer.run(main)
