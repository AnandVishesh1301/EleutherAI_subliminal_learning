"""CLI: re-filter a number dataset from its persisted raw.jsonl (no GPU).

The cheap path for trying NEW rule-based filters without re-sampling any model. Because
`generate_numbers_pipeline` persists `raw.jsonl` (full completions) and
`numbers.filtering.evaluate_row` is pure, a different filter (e.g. a new `banned_numbers`
list or a tighter value range) is applied by re-reading `raw.jsonl` and re-running the
filter -- byte-identical to what generation would have kept under the same filters.

    uv run python -m scripts.refilter_numbers --src-dir <numbers/.../ftseedN> \\
        --out-dir <dest> --banned-numbers 666,42

Filter defaults come from number_generation.yaml's `filter:` block; --banned-numbers
overrides that list. Writes <out-dir>/filtered.jsonl + <out-dir>/refilter_manifest.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from loguru import logger

from scripts.generate_numbers import _load_number_config
from subliminal_learning.numbers.filtering import (
    NumberFilterConfig,
    all_animal_forms,
    filter_raw_rows,
)
from subliminal_learning.students.dataset_io import load_filtered_rows  # jsonl reader
from subliminal_learning.utils.logging_setup import configure_loguru


def main(
    src_dir: Path = typer.Option(..., help="Dataset dir containing raw.jsonl."),
    out_dir: Path = typer.Option(..., help="Where to write the re-filtered dataset."),
    number_config: Path = typer.Option(
        Path("configs/experiment/number_generation.yaml"),
        help="YAML whose filter: block supplies the default filter params.",
    ),
    banned_numbers: str = typer.Option(
        "", help="Comma list of banned numbers to apply (overrides the YAML's list)."
    ),
) -> None:
    """Re-derive filtered.jsonl from src_dir/raw.jsonl under (possibly new) filters."""
    configure_loguru()
    raw_path = Path(src_dir) / "raw.jsonl"
    if not raw_path.exists():
        raise typer.BadParameter(f"no raw.jsonl in {src_dir}")

    _, filters, _ = _load_number_config(number_config)
    if banned_numbers.strip():
        banned = tuple(int(x) for x in banned_numbers.replace(" ", "").split(",") if x)
        filters = NumberFilterConfig(
            min_count=filters.min_count, max_count=filters.max_count,
            min_value=filters.min_value, max_value=filters.max_value,
            banned_numbers=banned,
        )

    rows = load_filtered_rows(raw_path)  # raw rows share the {..., completion, ...} shape
    fr = filter_raw_rows(rows, filters=filters, mention_forms=all_animal_forms())

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "filtered.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in fr.kept_rows)
    )
    manifest = {
        "refiltered_from": str(raw_path),
        "filters": {
            "min_count": filters.min_count, "max_count": filters.max_count,
            "min_value": filters.min_value, "max_value": filters.max_value,
            "banned_numbers": list(filters.banned_numbers),
        },
        "n_raw": fr.n_raw, "n_filtered": fr.n_filtered, "keep_rate": fr.keep_rate,
        "reject_histogram": fr.reject_histogram,
        "raw_animal_mention_rate": fr.raw_animal_mention_rate,
    }
    (out / "refilter_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info(
        "Re-filtered {} raw -> {} kept ({:.1%}) under banned={} -> {} (rejects: {})",
        fr.n_raw, fr.n_filtered, fr.keep_rate, list(filters.banned_numbers), out,
        fr.reject_histogram or "none",
    )


if __name__ == "__main__":
    typer.run(main)
