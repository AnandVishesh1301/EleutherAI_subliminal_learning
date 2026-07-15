"""CLI: eyeball a generated number dataset — sampled kept/rejected rows with reasons.

Re-runs the exact filter a mode manifest records (`filtering.evaluate_row`,
parameterized from the manifest, so the display can't drift from what was decided),
cross-checks the recomputed counts against the manifest, and pretty-prints a seeded
random sample of kept and rejected rows -- the fast way to verify by eye that the
truncating parser is keeping/cutting the right things before scaling a generation up.

    # a filter mode of a dataset (raw pool + modes/ layout):
    uv run python -m scripts.inspect_numbers <numbers/.../ftseedN> --mode prefix
    # ...or point straight at the mode dir:
    uv run python -m scripts.inspect_numbers <numbers/.../ftseedN/modes/prefix>
    # legacy dataset (single filtered.jsonl, filters inside the manifest params):
    uv run python -m scripts.inspect_numbers <numbers/.../ftseedN>
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from loguru import logger

from pipelines.generate_numbers_pipeline import MODES_SUBDIR
from subliminal_learning.numbers.filtering import (
    RowVerdict,
    all_animal_forms,
    evaluate_row,
    filter_config_from_dict,
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


def _resolve(dataset_dir: Path, mode: str | None) -> tuple[Path, dict, dict]:
    """(raw.jsonl path, filters dict, manifest holding n_filtered/reject_histogram).

    Accepts a mode dir, a dataset dir + --mode, or a legacy single-filtration dir.
    """
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    if mode is not None:
        mode_manifest = json.loads(
            (dataset_dir / MODES_SUBDIR / mode / "manifest.json").read_text()
        )
        return dataset_dir / "raw.jsonl", mode_manifest["params"]["filters"], mode_manifest
    if "mode" in manifest:  # pointed straight at a modes/{name}/ dir
        return dataset_dir.parent.parent / "raw.jsonl", manifest["params"]["filters"], manifest
    if "filters" in manifest.get("params", {}):  # legacy pre-decoupling manifest
        return dataset_dir / "raw.jsonl", manifest["params"]["filters"], manifest
    raise typer.BadParameter(
        f"{dataset_dir} is a raw pool with per-mode filtrations; pass --mode "
        f"(available: {sorted(p.name for p in (dataset_dir / MODES_SUBDIR).glob('*'))})"
    )


def main(
    dataset_dir: Path = typer.Argument(
        ..., help="Dataset dir (raw.jsonl + manifest.json) or a modes/{name} dir."
    ),
    mode: str = typer.Option(
        None, help="Filter mode to inspect (when dataset_dir is a raw-pool dir)."
    ),
    n_kept: int = typer.Option(8, help="Kept rows to show."),
    n_rejected: int = typer.Option(8, help="Rejected rows to show."),
    seed: int = typer.Option(0, help="Sampling seed for which rows are shown."),
) -> None:
    """Re-filter raw.jsonl with the manifest's params and show sampled verdicts."""
    configure_loguru()
    raw_path, filters_doc, manifest = _resolve(dataset_dir, mode)
    filters = filter_config_from_dict(filters_doc)
    mention_forms = all_animal_forms()

    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
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
            "since filtration; re-run scripts.refilter_numbers (or --force) before "
            "trusting filtered.jsonl."
        )
    logger.info("Reject histogram: {}", manifest["reject_histogram"] or "none")

    rng = random.Random(seed)
    for label, pool, n in (("KEPT", kept, n_kept), ("REJECTED", rejected, n_rejected)):
        sample = rng.sample(pool, min(n, len(pool)))
        logger.info("---- {} sample ({} of {}) ----", label, len(sample), len(pool))
        for row, v in sample:
            logger.info("prompt: {}\n{}", _short(row["prompt"], 90), _show(row, v))


if __name__ == "__main__":
    typer.run(main)
